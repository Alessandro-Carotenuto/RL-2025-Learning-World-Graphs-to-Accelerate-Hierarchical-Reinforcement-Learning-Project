"""
Quality comparison: NO ER (batch=50) vs ER (buffer=500, sample=50, alpha=0.6)
Same initial weights, same maze, same random seed sequence.
1000 episodes each. Reports AvgDist, AvgRew, Coverage every 100 ep.
"""
import random, time, torch, torch.nn.functional as F, sys, math, copy
sys.path.insert(0, '.')

from wrappers.minigrid_wrapper import MinigridWrapper, EnvModes, EnvSizes
from local_networks.hierarchical_system import HierarchicalManager
from bufferclasses import WidePretrainReplayBuffer
from utils.misc import manhattan_distance
from minigrid.core.world_object import Ball
from minigrid.core.constants import COLOR_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
SEED         = 42
N_EP         = 1000
HORIZONS     = 20
NUM_BALLS    = 5
NUM_PIVOTS   = 58
PRETRAIN_R   = 5
PPO_EPOCHS   = 4
BATCH_SIZE   = 50
DEVICE       = 'cpu'

ER_BUFFER    = 500
ER_FREQ      = 50
ER_SAMPLE    = 50
ER_ALPHA     = 0.6
ER_OFFSET    = 0.5

random.seed(SEED); torch.manual_seed(SEED)

# ── Build maze (shared) ───────────────────────────────────────────────────────
env = MinigridWrapper(size=EnvSizes.MEDIUM, mode=EnvModes.MULTIGOAL, max_steps=2000)
env.randomgen=True; env.firstgen=True; env.reset()
env.phase=2; env.fixed_ball_positions=None

valid_cells   = [(x,y) for x in range(1,env.width-1) for y in range(1,env.height-1)
                 if env._is_traversable(env.grid.get(x,y))]
maze_diagonal = (env.width-2)+(env.height-2)
pivotal_states = random.sample(valid_cells, min(NUM_PIVOTS, len(valid_cells)))

pos = random.sample(valid_cells, NUM_BALLS)
for i,(px,py) in enumerate(pos): env.grid.set(px,py,Ball(COLOR_NAMES[i%len(COLOR_NAMES)]))
env.active_balls=set(pos); env.total_balls=NUM_BALLS; env.balls_collected=0

print("Maze %dx%d | %d traversable | %d pivots | diagonal=%d" % (
    env.width, env.height, len(valid_cells), len(pivotal_states), maze_diagonal))

# ── Shared initial weights ────────────────────────────────────────────────────
base_manager = HierarchicalManager(pivotal_states=pivotal_states, neighborhood_size=3,
                                   lr=5e-4, device=DEVICE)
base_manager.entropy_coef = 0.001

# Pre-generate episode seeds so both runs see identical env states
ep_seeds = [random.randint(0, 2**31) for _ in range(N_EP)]

# ── Run function ──────────────────────────────────────────────────────────────
def run(use_er, manager):
    cov_hist, rew_hist, dist_hist = [], [], []
    rollout_buffer = []
    er_buf = WidePretrainReplayBuffer(ER_BUFFER, alpha=ER_ALPHA, reward_offset=ER_OFFSET) if use_er else None
    er_new = 0

    for ep in range(N_EP):
        random.seed(ep_seeds[ep]); torch.manual_seed(ep_seeds[ep])

        active_balls = env.reset_for_wide_pretrain(valid_cells, NUM_BALLS)
        manager.reset_manager_state()
        state = random.choice(valid_cells)

        m_states,m_wide,m_narrow,m_rewards,m_values = [],[],[],[],[]
        m_log_probs,m_entropies,m_balls,m_idxs,m_dists = [],[],[],[],[]
        initial = len(active_balls)

        for h in range(HORIZONS):
            if not active_balls: break
            m_balls.append(list(active_balls))
            wide_logits,_,value = manager.forward(state, list(active_balls))
            wide_probs = F.softmax(wide_logits, dim=0)
            wide_dist  = torch.distributions.Categorical(wide_probs)
            wide_idx   = wide_dist.sample()
            log_prob   = wide_dist.log_prob(wide_idx)
            entropy    = -(wide_probs*torch.log(wide_probs+1e-8)).sum().detach()
            wide_goal  = manager.pivotal_states[wide_idx.item()]
            manager.prev_wide_goal = wide_goal; value = value.squeeze()
            if manager.hidden_state:
                manager.hidden_state = tuple(hs.detach() for hs in manager.hidden_state)
            m_idxs.append(wide_idx.item())
            covered=[b for b in active_balls if manhattan_distance(wide_goal,b)<=PRETRAIN_R]
            if covered:
                closest=min(covered,key=lambda b:manhattan_distance(wide_goal,b))
                dist=manhattan_distance(wide_goal,closest)
                reward=1.0+2.0*(PRETRAIN_R-dist)/PRETRAIN_R
                active_balls.discard(closest)
            else:
                dist=min(manhattan_distance(wide_goal,b) for b in active_balls)
                reward=-dist/maze_diagonal
            m_states.append(state); m_wide.append(wide_goal); m_narrow.append(wide_goal)
            m_rewards.append(reward); m_values.append(value)
            m_log_probs.append(log_prob); m_entropies.append(entropy)
            m_dists.append(dist); state=wide_goal

        covered_count = initial - len(active_balls)
        cov_hist.append(covered_count / max(1, initial))
        rew_hist.append(sum(m_rewards)/len(m_rewards) if m_rewards else 0.0)
        dist_hist.append(sum(m_dists)/len(m_dists) if m_dists else 0.0)

        if len(m_rewards) > 1:
            ep_avg_rew = sum(m_rewards)/len(m_rewards)
            rollout = (m_states,m_wide,m_narrow,m_rewards,m_values,
                       m_log_probs,m_entropies,m_balls,m_idxs)
            if use_er:
                er_buf.add(rollout, ep_avg_rew)
                er_new += 1
                if er_new >= ER_FREQ and len(er_buf) >= ER_SAMPLE:
                    batch = er_buf.sample(ER_SAMPLE)
                    manager.update_policy_batched(batch, ppo_epochs=PPO_EPOCHS)
                    er_new = 0
            else:
                rollout_buffer.append(rollout)
                if len(rollout_buffer) >= BATCH_SIZE:
                    manager.update_policy_batched(rollout_buffer, ppo_epochs=PPO_EPOCHS)
                    rollout_buffer = []

        if (ep+1) % 100 == 0:
            r = cov_hist[-100:]; d = dist_hist[-100:]; rw = rew_hist[-100:]
            print("  Ep %4d | Cov: %5.1f%% | AvgDist: %5.2f | AvgRew: %+.3f" % (
                ep+1, sum(r)/len(r)*100, sum(d)/len(d), sum(rw)/len(rw)))

    return cov_hist, rew_hist, dist_hist

# ── NO ER ─────────────────────────────────────────────────────────────────────
print("\n%s\nNO ER (batch=50)\n%s" % ("="*50, "="*50))
m_no_er = HierarchicalManager(pivotal_states=pivotal_states, neighborhood_size=3, lr=5e-4, device=DEVICE)
m_no_er.load_state_dict(copy.deepcopy(base_manager.state_dict()))
m_no_er.entropy_coef = 0.001
t0=time.perf_counter()
c1,r1,d1 = run(False, m_no_er)
t_no_er = time.perf_counter()-t0

# ── WITH ER ───────────────────────────────────────────────────────────────────
print("\n%s\nWITH ER (buffer=500, sample=50, alpha=0.6)\n%s" % ("="*50, "="*50))
m_er = HierarchicalManager(pivotal_states=pivotal_states, neighborhood_size=3, lr=5e-4, device=DEVICE)
m_er.load_state_dict(copy.deepcopy(base_manager.state_dict()))
m_er.entropy_coef = 0.001
t0=time.perf_counter()
c2,r2,d2 = run(True, m_er)
t_er = time.perf_counter()-t0

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n%s\nSUMMARY — last 100 ep\n%s" % ("="*50,"="*50))
print("%-30s  %10s  %10s" % ("Metric","NO ER","WITH ER"))
print("-"*52)
for lbl, h1, h2 in [("Coverage (%)",    [x*100 for x in c1[-100:]], [x*100 for x in c2[-100:]]),
                     ("AvgDist",          d1[-100:], d2[-100:]),
                     ("AvgRew",           r1[-100:], r2[-100:])]:
    v1=sum(h1)/len(h1); v2=sum(h2)/len(h2)
    print("  %-28s  %10.3f  %10.3f  %s" % (lbl, v1, v2,
          "ER better" if (lbl=="AvgDist" and v2<v1) or (lbl!="AvgDist" and v2>v1) else "NO ER better"))
print("-"*52)
print("  Wall time:                        %8.1fs  %10.1fs" % (t_no_er, t_er))
