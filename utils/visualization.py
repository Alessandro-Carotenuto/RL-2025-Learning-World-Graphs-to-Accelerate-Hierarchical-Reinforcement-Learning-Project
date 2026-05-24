import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
from PIL import Image, ImageDraw

from utils.graph_manager import GraphManager, GraphVisualizer
from utils.checkpoint import load_phase1_checkpoint, restore_maze_from_grid_state
from wrappers.minigrid_wrapper import MinigridWrapper, EnvModes
from local_networks.hierarchical_system import HierarchicalManager, HierarchicalWorker, WorkerState


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


def plot_worker_pretrain_diagnostics(achievement_history, reward_history,
                                     length_history=None,
                                     training_metrics=None,
                                     save_path='pretrain_worker_diagnostics.png'):
    """3×3 grid of worker pre-training diagnostics."""
    if not achievement_history:
        return

    def moving_average(data, window=20):
        if len(data) < window:
            return np.array([])
        return np.convolve(data, np.ones(window) / window, mode='valid')

    n = len(achievement_history)
    ep_x = list(range(1, n + 1))
    ma_w = min(20, max(1, n // 10))

    has_metrics = bool(training_metrics)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Intermediate Phase: Worker Pre-training', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    def plot_with_ma(ax, x, data, color, label, ylabel, title, ylim=None):
        ax.plot(x, data, color=color, linewidth=0.8, alpha=0.4, label=label)
        ma = moving_average(data, ma_w)
        if len(ma) > 0:
            ax.plot(range(x[0] + ma_w - 1, x[0] + ma_w - 1 + len(ma)),
                    ma, 'k-', linewidth=2, label=f'MA({ma_w})')
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Row 1: episode-level stats
    plot_with_ma(axes[0], ep_x, [x * 100 for x in achievement_history],
                 'teal', 'Achievement', 'Success Rate (%)', 'Goal Achievement Rate', ylim=[0, 101])
    plot_with_ma(axes[1], ep_x, reward_history,
                 'steelblue', 'Reward', 'Episode Reward', 'Episode Reward')
    if length_history:
        plot_with_ma(axes[2], ep_x, length_history,
                     'darkorange', 'Steps', 'Steps', 'Episode Length (↓ = faster)')
    else:
        axes[2].set_visible(False)

    if has_metrics:
        m_x = list(range(1, len(training_metrics) + 1))

        # Row 2: A2C losses
        plot_with_ma(axes[3], m_x,
                     [m['policy_loss'] for m in training_metrics],
                     'crimson', 'Policy Loss', 'Loss', 'Policy Loss')
        plot_with_ma(axes[4], m_x,
                     [m['value_loss'] for m in training_metrics],
                     'purple', 'Value Loss', 'Loss', 'Value Loss (↓ = critic learning)')

        # Entropy + grad norm twin axes
        ax5 = axes[5]
        ent = [m['entropy'] for m in training_metrics]
        gn  = [m['grad_norm'] for m in training_metrics]
        ax5.plot(m_x, ent, color='green', linewidth=0.8, alpha=0.4, label='Entropy')
        ma_e = moving_average(ent, ma_w)
        if len(ma_e) > 0:
            ax5.plot(range(ma_w, ma_w + len(ma_e)), ma_e, 'g-', linewidth=2)
        ax5.set_ylabel('Entropy', color='green')
        ax5.tick_params(axis='y', labelcolor='green')
        ax5_r = ax5.twinx()
        ax5_r.plot(m_x, gn, color='gray', linewidth=0.8, alpha=0.4, label='Grad Norm')
        ma_g = moving_average(gn, ma_w)
        if len(ma_g) > 0:
            ax5_r.plot(range(ma_w, ma_w + len(ma_g)), ma_g, color='gray', linewidth=2)
        ax5_r.set_ylabel('Grad Norm', color='gray')
        ax5.set_title('Entropy & Grad Norm')
        ax5.set_xlabel('Episode')
        ax5.grid(True, alpha=0.3)
        lines1, lab1 = ax5.get_legend_handles_labels()
        lines2, lab2 = ax5_r.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, lab1 + lab2, fontsize=8)

        # Row 3: critic accuracy, advantage quality, action distribution
        # Panel 6: Value Mean vs Return Mean (how well the critic tracks actual returns)
        ax6 = axes[6]
        vm = [m.get('value_mean', 0) for m in training_metrics]
        rm = [m.get('return_mean', 0) for m in training_metrics]
        ax6.plot(m_x, vm, color='blue', linewidth=0.8, alpha=0.4, label='Value (critic)')
        ax6.plot(m_x, rm, color='orange', linewidth=0.8, alpha=0.4, label='Return (actual)')
        ma_v = moving_average(vm, ma_w)
        ma_r = moving_average(rm, ma_w)
        if len(ma_v) > 0:
            ax6.plot(range(ma_w, ma_w + len(ma_v)), ma_v, 'b-', linewidth=2)
        if len(ma_r) > 0:
            ax6.plot(range(ma_w, ma_w + len(ma_r)), ma_r, color='orange', linewidth=2)
        ax6.set_title('Critic Accuracy (Value vs Return)')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Value / Return')
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=8)

        # Panel 7: Advantage mean ± std (signal quality — should be non-zero and stable)
        ax7 = axes[7]
        adv_m = [m.get('adv_mean', 0) for m in training_metrics]
        adv_s = [m.get('adv_std', 0) for m in training_metrics]
        ax7.plot(m_x, adv_m, color='darkgreen', linewidth=0.8, alpha=0.4, label='Adv Mean')
        ma_am = moving_average(adv_m, ma_w)
        if len(ma_am) > 0:
            xr = list(range(ma_w, ma_w + len(ma_am)))
            ma_as = moving_average(adv_s, ma_w)
            ax7.plot(xr, ma_am, 'g-', linewidth=2)
            if len(ma_as) == len(ma_am):
                ax7.fill_between(xr,
                                 np.array(ma_am) - np.array(ma_as),
                                 np.array(ma_am) + np.array(ma_as),
                                 color='green', alpha=0.15, label='±std')
        ax7.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax7.set_title('Advantage Mean ± Std (signal quality)')
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Advantage')
        ax7.grid(True, alpha=0.3)
        ax7.legend(fontsize=8)

        # Panel 8: Action distribution (which action dominates over time)
        ax8 = axes[8]
        if 'frac_left' in training_metrics[0]:
            fl = [m['frac_left']    for m in training_metrics]
            fr = [m['frac_right']   for m in training_metrics]
            ff = [m['frac_forward'] for m in training_metrics]
            ma_fl = moving_average(fl, ma_w)
            ma_fr = moving_average(fr, ma_w)
            ma_ff = moving_average(ff, ma_w)
            xr = list(range(ma_w, ma_w + len(ma_fl))) if len(ma_fl) > 0 else []
            ax8.plot(m_x, fl, color='royalblue',  linewidth=0.6, alpha=0.3)
            ax8.plot(m_x, fr, color='tomato',     linewidth=0.6, alpha=0.3)
            ax8.plot(m_x, ff, color='seagreen',   linewidth=0.6, alpha=0.3)
            if xr:
                ax8.plot(xr, ma_fl, 'b-',  linewidth=2, label='turn_left')
                ax8.plot(xr, ma_fr, 'r-',  linewidth=2, label='turn_right')
                ax8.plot(xr, ma_ff, 'g-',  linewidth=2, label='move_fwd')
            ax8.axhline(1/3, color='gray', linewidth=1, linestyle='--', label='uniform (1/3)')
            ax8.set_ylim([0, 1])
        ax8.set_title('Action Distribution')
        ax8.set_xlabel('Episode')
        ax8.set_ylabel('Fraction')
        ax8.grid(True, alpha=0.3)
        ax8.legend(fontsize=8)
    else:
        for i in [3, 4, 5, 6, 7, 8]:
            axes[i].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Worker pre-training diagnostics saved to {save_path}")
    plt.close(fig)


def plot_manager_wide_pretrain_diagnostics(coverage_history, reward_history,
                                           save_path='pretrain_manager_wide_diagnostics.png'):
    """1×2: ball coverage (%) and avg reward per horizon over Manager Wide pre-training."""
    if not coverage_history:
        return

    def moving_average(data, window=20):
        if len(data) < window:
            return np.array([])
        return np.convolve(data, np.ones(window) / window, mode='valid')

    n = len(coverage_history)
    episodes = list(range(1, n + 1))
    ma_w = 20

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Intermediate Phase: Manager Wide Goal Pre-training', fontsize=14, fontweight='bold')

    def plot_with_ma(ax, data, color, label, ylabel, title, ylim=None):
        ax.plot(episodes, data, color=color, linewidth=1.0, alpha=0.5, label=label)
        ma = moving_average(data)
        if len(ma) > 0:
            ax.plot(range(ma_w, ma_w + len(ma)), ma, 'k-', linewidth=2, label=f'MA({ma_w})')
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plot_with_ma(axes[0], [x * 100 for x in coverage_history],
                 'green', 'Coverage', 'Balls Covered (%)',
                 'Ball Coverage per Episode\n(↑ = Manager targeting balls)', ylim=[0, 101])

    plot_with_ma(axes[1], reward_history,
                 'orange', 'Reward', 'Avg Reward / Horizon',
                 'Avg Reward per Horizon\n(↑ = wide goals closer to balls)')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Manager Wide pre-training diagnostics saved to {save_path}")
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

        fig.canvas.draw()
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
        maze_size=config['maze_size'].value,
        neighborhood_size=config.get('neighborhood_size', 3),
        device='cpu',
    )
    worker.load_state_dict(session['worker_state_dict'])
    worker.eval()

    _run_and_save_episode(manager, worker, config, grid_state, agent_start, ball_positions, filename, fps, max_steps,
                          world_graph=world_graph, pivotal_states=pivotal_states)


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
    valid_cells = {
        (x, y)
        for x in range(env.width)
        for y in range(env.height)
        if env._is_traversable(env.grid.get(x, y))
    }
    manager.reset_manager_state()
    worker.reset_worker_state()
    worker.valid_cells = valid_cells

    overlay_enabled = world_graph is not None and pivotal_states is not None
    if overlay_enabled:
        print(f"[VIDEO] Overlay ON: {len(pivotal_states)} nodes, {len(world_graph.edges)} edges")
    else:
        print(f"[VIDEO] Overlay OFF: world_graph={world_graph is not None}, pivotal_states={pivotal_states is not None}")

    def apply_overlay(frame, wg, ng, traversal_path=None, active_balls=None, agent_state=None, local_goal=None):
        tile_size = frame.shape[1] // env.width

        def px(coord):
            return (coord[0] * tile_size + tile_size // 2,
                    coord[1] * tile_size + tile_size // 2)

        frame_pil = Image.fromarray(frame).convert('RGBA')
        ov = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ov)

        path_set = set(traversal_path) if traversal_path else set()
        path_edges = set(zip(traversal_path, traversal_path[1:])) if traversal_path else set()

        for (start, end) in world_graph.edges:
            if (start, end) in path_edges:
                draw.line([px(start), px(end)], fill=(0, 220, 80, 160), width=3)
            else:
                draw.line([px(start), px(end)], fill=(255, 220, 0, 64), width=2)

        r = max(2, tile_size // 5)
        for ps in pivotal_states:
            if ps in path_set:
                cx, cy = px(ps)
                draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(0, 220, 80, 160))
            else:
                cx, cy = px(ps)
                draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(255, 220, 0, 64))

        if wg is not None:
            cx, cy = px(wg)
            draw.ellipse([(cx - r * 2, cy - r * 2), (cx + r * 2, cy + r * 2)],
                         fill=(0, 220, 80, 178))  # 70% opacity
        if ng is not None:
            nx, ny = ng
            draw.rectangle(
                [(nx * tile_size, ny * tile_size),
                 ((nx + 1) * tile_size - 1, (ny + 1) * tile_size - 1)],
                fill=(0, 200, 255, 51)
            )
        if local_goal is not None:
            lx, ly = local_goal
            draw.rectangle(
                [(lx * tile_size, ly * tile_size),
                 ((lx + 1) * tile_size - 1, (ly + 1) * tile_size - 1)],
                fill=(255, 200, 50, 120)  # yellow, FINDING local goal
            )

        if agent_state is not None and active_balls:
            nearest_ball = min(active_balls,
                               key=lambda b: abs(b[0] - agent_state[0]) + abs(b[1] - agent_state[1]))
            draw.line([px(agent_state), px(nearest_ball)], fill=(255, 0, 0, 220), width=2)

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
        first_frame = apply_overlay(first_frame, None, None,
                                    active_balls=list(env.active_balls), agent_state=state)
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
                    worker.reset_worker_state()
                    wide_goal, narrow_goal, _, _, _ = manager.get_manager_action(
                        state, step_count=999999, valid_cells=valid_cells,
                        active_balls=list(env.active_balls)
                    )
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
                horizon_step = horizon - 1  # force horizon end at next increment

            # FINDING local goal: nearest pivotal state (excluding current pos)
            if worker._worker_state == WorkerState.FINDING and worker.pivotal_states:
                candidates = [p for p in worker.pivotal_states if p != state]
                local_goal_vis = (min(candidates, key=lambda p: abs(p[0] - state[0]) + abs(p[1] - state[1]))
                                  if candidates else None)
            else:
                local_goal_vis = None

            frame = env.render()
            if overlay_enabled:
                frame = apply_overlay(frame, wide_goal, narrow_goal,
                                      traversal_path=worker.current_traversal_path or None,
                                      active_balls=list(env.active_balls), agent_state=state,
                                      local_goal=local_goal_vis)
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
