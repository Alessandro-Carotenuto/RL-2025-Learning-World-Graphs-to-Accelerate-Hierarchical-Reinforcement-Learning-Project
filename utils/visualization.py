import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
from PIL import Image, ImageDraw

from utils.graph_manager import GraphManager, GraphVisualizer
from utils.checkpoint import load_phase1_checkpoint, restore_maze_from_grid_state
from wrappers.minigrid_wrapper import MinigridWrapper, EnvModes
from local_networks.hierarchical_system import HierarchicalManager, HierarchicalWorker


def print_grid_image(grid_text, name=''):
    fig, ax = plt.subplots(figsize=(len(grid_text[0]), len(grid_text)))
    ax.set_xlim(0, len(grid_text[0]))
    ax.set_ylim(0, len(grid_text))
    ax.set_aspect('equal')
    ax.axis('off')

    for i, row in enumerate(grid_text):
        for j, char in enumerate(row):
            ax.text(j + 0.5, len(grid_text) - i - 0.5, char,
                    ha='center', va='center', fontsize=20)

    fig.tight_layout()
    fig.savefig(f'grid{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_phase1_run(vae_system, metrics, mu0, title_prefix, save_path):
    history = vae_system.training_history
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].plot([h['total_loss'] for h in history], 'b-')
    axes[0, 0].set_title(f'VAE Loss ({title_prefix})')
    axes[0, 1].plot([h['reconstruction_loss'] for h in history], 'r-')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 2].plot([h['kl_divergence'] for h in history], 'g-')
    axes[0, 2].set_title('KL Divergence')
    axes[1, 0].plot([h['expected_l0'] for h in history], 'purple')
    axes[1, 0].axhline(y=mu0, color='orange', linestyle='--', label='Target')
    axes[1, 0].set_title('Expected L0')
    axes[1, 0].legend()
    axes[1, 1].plot(metrics['num_pivotal_states_per_iteration'], 'cyan', marker='o')
    axes[1, 1].set_title('Pivotal States Discovered')
    axes[1, 2].plot(metrics['policy_success_rates'], 'magenta', marker='s')
    axes[1, 2].set_title('Policy Success Rate')
    axes[1, 2].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_diagnostics(trainer, config, save_path=None):
    """Plot training diagnostics: 6 panels covering task, manager, and worker signals."""

    def moving_average(data, window=20):
        if len(data) < window:
            return np.array([])
        return np.convolve(data, np.ones(window)/window, mode='valid')

    history = trainer.diagnostic_history
    num_episodes = len(history['episode_rewards'])
    episodes = list(range(1, num_episodes + 1))
    ma_start = 20  # moving average window

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Hierarchical RL Training Diagnostics', fontsize=16, fontweight='bold')

    def plot_with_ma(ax, data, color, label, ylabel, title, ylim=None):
        ax.plot(episodes, data, color=color, linewidth=1.5, alpha=0.5, label=label)
        ma = moving_average(data)
        if len(ma) > 0:
            ax.plot(range(ma_start, ma_start + len(ma)), ma, 'k-', linewidth=2, label=f'MA({ma_start})')
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # 1. Episode Rewards
    plot_with_ma(axes[0, 0],
                 history['episode_rewards'],
                 'b', 'Agent', 'Reward',
                 'Episode Rewards')

    # 2. Balls Collected per Episode
    plot_with_ma(axes[0, 1],
                 history['balls_collected_per_episode'],
                 'green', 'Balls', 'Balls Collected',
                 'Balls Collected per Episode',
                 ylim=[0, 6])

    # 3. Manager Goal Diversity  (unique goals / total horizons — drops as manager converges)
    plot_with_ma(axes[0, 2],
                 history['manager_goal_diversity'],
                 'orange', 'Diversity', 'Unique Goals / Horizons',
                 'Manager Goal Diversity\n(↓ = converging on fewer goals)',
                 ylim=[0, 1])

    # 4. Manager Entropy  (absolute nats — drops as policy peaks)
    max_entropy = np.log(len(trainer.manager.pivotal_states))
    entropy_pct = [e / max_entropy * 100 for e in history['manager_entropy']]
    plot_with_ma(axes[1, 0],
                 entropy_pct,
                 'red', 'Entropy %', '% of Max Entropy',
                 'Manager Policy Entropy\n(↓ = more decisive)',
                 ylim=[0, 101])

    # 5. Avg Distance: Manager Goals → Nearest Ball  (drops as manager learns ball locations)
    dist_data = history['goal_distance_to_balls']
    if dist_data:
        dist_episodes = list(range(1, len(dist_data) + 1))
        ax5 = axes[1, 1]
        ax5.plot(dist_episodes, dist_data, 'purple', linewidth=1.5, alpha=0.5, label='Dist')
        ma_d = moving_average(dist_data)
        if len(ma_d) > 0:
            ax5.plot(range(ma_start, ma_start + len(ma_d)), ma_d, 'k-', linewidth=2, label=f'MA({ma_start})')
        ax5.set_title('Avg Distance: Manager Goals → Balls\n(↓ = manager targeting balls)')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Manhattan Distance')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=8)

    # 6. Manager Value Estimate  (rises and stabilises as critic converges)
    plot_with_ma(axes[1, 2],
                 history['manager_value_mean'],
                 'cyan', 'Value', 'Average Value',
                 'Manager Value Estimate\n(stabilises when critic converges)')

    # 7. Worker Goal Achievement  (row 2, left — narrow goal reached rate)
    plot_with_ma(axes[2, 0],
                 history['worker_goal_achievement'],
                 'teal', 'Achievement', 'Success Rate',
                 'Worker Goal Achievement\n(narrow goal reached)',
                 ylim=[0, 1])

    axes[2, 1].set_visible(False)
    axes[2, 2].set_visible(False)

    fig.tight_layout()
    if save_path is None:
        save_path = f"diagnostics_size{config['maze_size'].name}_h{config['manager_horizon']}_n{config['neighborhood_size']}_ep{config['phase2_episodes']}.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nDiagnostic plots saved to {save_path}")
    plt.close(fig)


def save_graph_visualization(world_graph, pivotal_states, mu0, grid_state=None):
    if not pivotal_states or world_graph is None or not world_graph.nodes:
        print("Skipping graph visualization: no pivotal states or empty graph.")
        return
    viz = GraphVisualizer(world_graph, figsize=(12, 12))
    fig, ax = viz.visualize(
        show_weights=True,
        show_labels=True,
        node_size=250,
        edge_width=1.5,
        title=f'World Graph (mu0={mu0}) - Feasible Paths',
        grid_state=grid_state
    )
    filename = f'world_graph_mu{mu0:.1f}.png'
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved graph visualization to '{filename}'")


def create_phase1_gif(all_pivotal_states_history, grid_state, filename='phase1_evolution.gif', fps=2):
    """One frame per Phase 1 iteration — same style as the final graph visualization."""
    if not all_pivotal_states_history:
        print("No Phase 1 history to animate.")
        return

    frames = []
    total = len(all_pivotal_states_history)

    for iteration, pivotal_states in enumerate(all_pivotal_states_history):
        temp_graph = GraphManager()
        for ps in pivotal_states:
            temp_graph.add_node(ps)

        viz = GraphVisualizer(temp_graph, figsize=(10, 10))
        fig, ax = viz.visualize(
            show_weights=False,
            show_labels=True,
            node_size=250,
            edge_width=1.5,
            title=f'Phase 1 — Iteration {iteration + 1}/{total}  |  {len(pivotal_states)} pivotal states',
            grid_state=grid_state,
        )

        frame = np.array(fig.canvas.buffer_rgba())[..., :3]
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(filename, frames, fps=fps)
    print(f"Phase 1 evolution GIF saved to '{filename}' ({len(frames)} frames)")


def render_phase2_episode_gif(checkpoint_path, filename='phase2_final_episode.mp4', fps=15, max_steps=500):
    """
    Standalone: load checkpoint + session file, run one greedy episode, save as MP4.
    Call this after training from anywhere — no training objects needed.
    Requires: checkpoint .pt  +  checkpoint _session.pt (saved automatically at end of Phase 2).
    """
    pivotal_states, world_graph, policy, vae_system, config, grid_state = load_phase1_checkpoint(checkpoint_path)

    session_path = checkpoint_path.replace('.pt', '_session.pt')
    session = torch.load(session_path, map_location='cpu', weights_only=False)
    agent_start = session['agent_start']
    ball_positions = session['ball_positions']

    if 'goal_policy_state_dict' in session:
        policy.load_state_dict(session['goal_policy_state_dict'])

    manager = HierarchicalManager(
        pivotal_states,
        neighborhood_size=config['neighborhood_size'],
        lr=config['manager_lr'],
        horizon=config['manager_horizon'],
        diagnostic_interval=config['diagnostic_interval'],
        diagnostic_checkstart=config['diagnostic_checkstart'],
        device='cpu',
    )
    manager.load_state_dict(session['manager_state_dict'])
    manager.eval()

    worker = HierarchicalWorker(
        world_graph,
        pivotal_states,
        lr=config['worker_lr'],
        goal_policy=policy,
        device='cpu',
    )
    worker.load_state_dict(session['worker_state_dict'])
    worker.eval()

    _run_and_save_episode(manager, worker, config, grid_state, agent_start, ball_positions, filename, fps, max_steps)


def _run_and_save_episode(manager, worker, config, grid_state, agent_start_pos,
                          ball_positions, filename, fps, max_steps,
                          world_graph=None, pivotal_states=None):
    env = MinigridWrapper(
        size=config['maze_size'],
        mode=EnvModes.MULTIGOAL,
        max_steps=config['max_steps_per_episode'],
        render_mode='rgb_array',
    )
    env.reset()
    restore_maze_from_grid_state(env, grid_state)
    env.agent_start_pos = agent_start_pos
    env.agent_pos = agent_start_pos
    env.placeable_grid[agent_start_pos[0]][agent_start_pos[1]] = False
    env.firstgen = False
    env.phase = 2
    if ball_positions is not None:
        env.fixed_ball_positions = ball_positions

    env.reset()
    state = tuple(env.agent_pos)
    manager.reset_manager_state()
    worker.reset_worker_state()

    overlay_enabled = world_graph is not None and pivotal_states is not None
    if overlay_enabled:
        print(f"[VIDEO] Overlay ON: {len(pivotal_states)} nodes, {len(world_graph.edges)} edges")
    else:
        print(f"[VIDEO] Overlay OFF: world_graph={world_graph is not None}, pivotal_states={pivotal_states is not None}")

    def apply_overlay(frame, wg, ng):
        tile_size = frame.shape[1] // env.width

        def px(coord):
            return (coord[0] * tile_size + tile_size // 2,
                    coord[1] * tile_size + tile_size // 2)

        frame_pil = Image.fromarray(frame).convert('RGBA')
        ov = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ov)

        for (start, end) in world_graph.edges:
            draw.line([px(start), px(end)], fill=(255, 220, 0, 64), width=2)

        r = max(2, tile_size // 5)
        for ps in pivotal_states:
            cx, cy = px(ps)
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(255, 220, 0, 64))

        if wg is not None:
            cx, cy = px(wg)
            draw.ellipse([(cx - r * 2, cy - r * 2), (cx + r * 2, cy + r * 2)],
                         fill=(255, 160, 0, 255))
        if ng is not None:
            nx, ny = ng
            draw.rectangle(
                [(nx * tile_size, ny * tile_size),
                 ((nx + 1) * tile_size - 1, (ny + 1) * tile_size - 1)],
                fill=(0, 200, 255, 51)
            )

        return np.array(Image.alpha_composite(frame_pil, ov).convert('RGB'))

    # Goal persistence state — mirrors train_episode exactly
    goal_timeout = config.get('goal_timeout', 3)
    horizon = config['manager_horizon']
    active_wide_goal = None
    active_narrow_goal = None
    horizons_on_goal = 0
    goal_reached_prev = False
    ball_collected_prev = False
    wide_goal = manager.pivotal_states[0]
    narrow_goal = manager.pivotal_states[0]

    first_frame = env.render()
    if overlay_enabled:
        first_frame = apply_overlay(first_frame, None, None)
    frames = [first_frame]

    done = False
    step = 0
    horizon_step = 0
    goal_reached_this_horizon = False
    starting_balls_snapshot = list(env.active_balls)

    with torch.no_grad():
        while not done and step < max_steps:
            # Horizon boundary: goal persistence logic (mirrors train_episode)
            if horizon_step == 0:
                need_new_goal = (
                    active_wide_goal is None
                    or goal_reached_prev
                    or ball_collected_prev
                    or horizons_on_goal >= goal_timeout
                )
                if need_new_goal:
                    # Flush traversal state, keep worker LSTM context
                    worker.current_traversal_path = []
                    worker.traversal_step = 0
                    worker.current_edge_actions = None
                    worker.current_action_idx = 0
                    wide_goal, narrow_goal, _, _, _ = manager.get_manager_action(state, step_count=999999)
                    if manager.hidden_state is not None:
                        manager.hidden_state = tuple(h.detach() for h in manager.hidden_state)
                    active_wide_goal = wide_goal
                    active_narrow_goal = narrow_goal
                    horizons_on_goal = 0
                else:
                    wide_goal = active_wide_goal
                    narrow_goal = active_narrow_goal
                goal_reached_this_horizon = False
                starting_balls_snapshot = list(env.active_balls)

            action, _, _ = worker.get_action(state, wide_goal, narrow_goal, agent_dir=env.agent_dir)
            try:
                obs, _, terminated, truncated, _ = env.step(action)
            except (AssertionError, IndexError):
                terminated, truncated = False, False
            state = tuple(env.agent_pos)

            if state == narrow_goal:
                goal_reached_this_horizon = True

            frame = env.render()
            if overlay_enabled:
                frame = apply_overlay(frame, wide_goal, narrow_goal)
            frames.append(frame)
            done = terminated or truncated
            step += 1
            horizon_step += 1

            # End of horizon: update persistence state
            if horizon_step >= horizon:
                horizon_step = 0
                horizons_on_goal += 1
                goal_reached_prev = goal_reached_this_horizon
                ball_collected_prev = (len(starting_balls_snapshot) - len(env.active_balls)) > 0

    with imageio.get_writer(filename, fps=fps, format='ffmpeg') as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Phase 2 video saved to '{filename}' ({len(frames)} frames, {len(frames)/fps:.1f}s)")
