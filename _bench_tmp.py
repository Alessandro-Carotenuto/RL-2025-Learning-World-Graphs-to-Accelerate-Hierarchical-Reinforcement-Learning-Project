"""
Benchmark: OLD (update every ep) vs NEW (batch=50 + sequence LSTM)
Simulates the actual training loop — no pre-collected rollouts.
250 episodes, per-component timing.
"""
import random, time, torch, torch.nn.functional as F, sys, math
sys.path.insert(0, '.')

from wrappers.minigrid_wrapper import MinigridWrapper, EnvModes, EnvSizes
from local_networks.hierarchical_system import HierarchicalManager
from utils.misc import manhattan_distance
from minigrid.core.world_object import Ball
from minigrid.core.constants import COLOR_NAMES

N_EP       = 250
HORIZONS   = 20
NUM_BALLS  = 5
NUM_PIVOTS = 58
PRETRAIN_R = 5
PPO_EPOCHS = 4
BATCH_SIZE = 50
DEVICE     = 'cpu'

# ── Build maze ────────────────────────────────────────────────────────────────
env = MinigridWrapper(size=EnvSizes.MEDIUM, mode=EnvModes.MULTIGOAL, max_steps=2000)
env.randomgen=True; env.firstgen=True; env.reset()
env.phase=2; env.fixed_ball_positions=None

valid_cells   = [(x,y) for x in range(1,env.width-1) for y in range(1,env.height-1)
                 if env._is_traversable(env.grid.get(x,y))]
maze_diagonal = (env.width-2)+(env.height-2)
pivotal_states = random.sample(valid_cells, min(NUM_PIVOTS, len(valid_cells)))

# seed initial balls
pos = random.sample(valid_cells, NUM_BALLS)
for i,(px,py) in enumerate(pos): env.grid.set(px,py,Ball(COLOR_NAMES[i%len(COLOR_NAMES)]))
env.active_balls=set(pos); env.total_balls=NUM_BALLS; env.balls_collected=0

print("Maze %dx%d | %d traversable | %d pivots | diagonal=%d" % (
    env.width, env.height, len(valid_cells), len(pivotal_states), maze_diagonal))

def make_manager():
    m = HierarchicalManager(pivotal_states=pivotal_states, neighborhood_size=3, lr=5e-4, device=DEVICE)
    m.entropy_coef = 0.001
    return m

class T:
    def __init__(self): self.s=0.0; self.n=0
    def add(self,dt): self.s+=dt; self.n+=1
    def ms(self): return self.s/self.n*1000 if self.n else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Shared rollout collection — identical in both versions
# Returns rollout tuple + per-component times
# ─────────────────────────────────────────────────────────────────────────────
def collect_episode(manager, t_reset, t_mgr, t_fwd, t_dist):
    t=time.perf_counter()
    active_balls = env.reset_for_wide_pretrain(valid_cells, NUM_BALLS)
    t_reset.add(time.perf_counter()-t)

    t=time.perf_counter()
    manager.reset_manager_state()
    t_mgr.add(time.perf_counter()-t)

    state = random.choice(valid_cells)
    m_states,m_wide,m_narrow,m_rewards,m_values,m_log_probs,m_entropies = [],[],[],[],[],[],[]
    m_dists,m_balls,m_idxs = [],[],[]

    for h in range(HORIZONS):
        if not active_balls: break
        m_balls.append(list(active_balls))

        t=time.perf_counter()
        wide_logits,_,value = manager.forward(state, list(active_balls))
        wide_probs = F.softmax(wide_logits, dim=0)
        wide_dist  = torch.distributions.Categorical(wide_probs)
        wide_idx   = wide_dist.sample()
        log_prob   = wide_dist.log_prob(wide_idx)
        entropy    = -(wide_probs*torch.log(wide_probs+1e-8)).sum().detach()
        wide_goal  = manager.pivotal_states[wide_idx.item()]
        manager.prev_wide_goal = wide_goal; value = value.squeeze()
        if manager.hidden_state is not None:
            manager.hidden_state = tuple(hs.detach() for hs in manager.hidden_state)
        t_fwd.add(time.perf_counter()-t)
        m_idxs.append(wide_idx.item())

        t=time.perf_counter()
        covered=[b for b in active_balls if manhattan_distance(wide_goal,b)<=PRETRAIN_R]
        if covered:
            closest=min(covered,key=lambda b:manhattan_distance(wide_goal,b))
            dist=manhattan_distance(wide_goal,closest)
            reward=1.0+2.0*(PRETRAIN_R-dist)/PRETRAIN_R
            active_balls.discard(closest)
        else:
            dist=min(manhattan_distance(wide_goal,b) for b in active_balls)
            reward=-dist/maze_diagonal
        t_dist.add(time.perf_counter()-t)

        m_states.append(state); m_wide.append(wide_goal); m_narrow.append(wide_goal)
        m_rewards.append(reward); m_values.append(value)
        m_log_probs.append(log_prob); m_entropies.append(entropy); m_dists.append(dist)
        state=wide_goal

    return (m_states,m_wide,m_narrow,m_rewards,m_values,m_log_probs,m_entropies,m_balls,m_idxs)

# ─────────────────────────────────────────────────────────────────────────────
# OLD: update every episode (slow Python loop in update_policy)
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning OLD (update/ep) for %d ep..." % N_EP)
m_old = make_manager()
tr_o=T(); tm_o=T(); tf_o=T(); td_o=T(); tp_o=T(); te_o=T()

for ep in range(N_EP):
    t0=time.perf_counter()
    rollout = collect_episode(m_old, tr_o, tm_o, tf_o, td_o)
    rs,ws,ns,rews,vals,lps,ents,bls,idxs = rollout
    if len(rews) > 1:
        t=time.perf_counter()
        m_old.update_policy(rs,ws,ns,rews,vals,lps,ents, step_count=ep,
            balls_snapshots=bls, wide_idxs=idxs, wide_only=False, ppo_epochs=PPO_EPOCHS)
        tp_o.add(time.perf_counter()-t)
    te_o.add(time.perf_counter()-t0)

# ─────────────────────────────────────────────────────────────────────────────
# NEW: batch=50 + sequence LSTM
# ─────────────────────────────────────────────────────────────────────────────
print("Running NEW (batch=%d + seqLSTM) for %d ep..." % (BATCH_SIZE, N_EP))
m_new = make_manager()
m_new.load_state_dict(m_old.state_dict())
tr_n=T(); tm_n=T(); tf_n=T(); td_n=T(); tp_n=T(); te_n=T()

buffer=[]
for ep in range(N_EP):
    t0=time.perf_counter()
    rollout = collect_episode(m_new, tr_n, tm_n, tf_n, td_n)
    rs,ws,ns,rews,vals,lps,ents,bls,idxs = rollout
    if len(rews) > 1:
        buffer.append(rollout)
    if len(buffer) >= BATCH_SIZE:
        t=time.perf_counter()
        m_new.update_policy_batched(buffer, ppo_epochs=PPO_EPOCHS)
        # spread PPO time across the 50 episodes in the batch
        dt = time.perf_counter()-t
        for _ in range(len(buffer)): tp_n.add(dt/len(buffer))
        buffer=[]
    te_n.add(time.perf_counter()-t0)

# flush
if buffer:
    t=time.perf_counter()
    m_new.update_policy_batched(buffer, ppo_epochs=PPO_EPOCHS)
    dt=time.perf_counter()-t
    for _ in range(len(buffer)): tp_n.add(dt/len(buffer))

# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def run_batched(batch_size, base_manager, label):
    m = make_manager(); m.load_state_dict(base_manager.state_dict())
    tr=T(); tm=T(); tf=T(); td=T(); tp=T(); te=T()
    buffer=[]
    for ep in range(N_EP):
        t0=time.perf_counter()
        rollout = collect_episode(m, tr, tm, tf, td)
        rs,ws,ns,rews,vals,lps,ents,bls,idxs = rollout
        if len(rews) > 1:
            buffer.append(rollout)
        if len(buffer) >= batch_size:
            t=time.perf_counter()
            m.update_policy_batched(buffer, ppo_epochs=PPO_EPOCHS)
            dt=time.perf_counter()-t
            for _ in range(len(buffer)): tp.add(dt/len(buffer))
            buffer=[]
        te.add(time.perf_counter()-t0)
    if buffer:
        t=time.perf_counter()
        m.update_policy_batched(buffer, ppo_epochs=PPO_EPOCHS)
        dt=time.perf_counter()-t
        for _ in range(len(buffer)): tp.add(dt/len(buffer))
    return tr,tm,tf,td,tp,te

print("\nRunning batch variants...")
results = {}
results['OLD(1/ep)'] = (tr_o,tm_o,tf_o,td_o,tp_o,te_o)
for bs in [50, 100, 250]:
    label = "NEW(b=%d)" % bs
    print("  batch=%d..." % bs)
    results[label] = run_batched(bs, m_old, label)

# torch.compile variant (batch=50)
try:
    m_compiled = make_manager(); m_compiled.load_state_dict(m_old.state_dict())
    m_compiled.lstm       = torch.compile(m_compiled.lstm)
    m_compiled.wide_head  = torch.compile(m_compiled.wide_head)
    m_compiled.critic     = torch.compile(m_compiled.critic)
    print("  torch.compile + batch=50 (warmup)...")
    # warmup
    _buf=[]
    for ep in range(10):
        r=collect_episode(m_compiled,T(),T(),T(),T())
        if len(r[3])>1: _buf.append(r)
    if _buf: m_compiled.update_policy_batched(_buf[:5],ppo_epochs=1)
    print("  torch.compile + batch=50 (timed)...")
    results['compiled(b=50)'] = run_batched(50, m_compiled, 'compiled')
    has_compile=True
except Exception as e:
    print("  torch.compile not available: %s" % e)
    has_compile=False

# ── Report ────────────────────────────────────────────────────────────────────
cols = list(results.keys())
header = "%-18s" % "Component"
for c in cols: header += "  %13s" % c
print("\n" + header)
print("-"*(18+15*len(cols)))

labels = [("env reset",0),("mgr reset",1),("LSTM fwd/step",2),
          ("dist+rew/step",3),("PPO update/ep",4),("TOTAL/episode",5)]
for lbl,idx in labels:
    row = "  %-16s" % lbl
    base = results['OLD(1/ep)'][idx].ms()
    for c in cols:
        v = results[c][idx].ms()
        sp = base/v if v>0 else 0
        row += "  %10.3fms" % v
        if c != 'OLD(1/ep)': row += "(%.1fx)" % sp
        else: row += "       "
    print(row)

print("-"*(18+15*len(cols)))
row = "  %-16s" % "wall time(s)"
base_t = results['OLD(1/ep)'][5].s
for c in cols:
    t_s = results[c][5].s
    sp = base_t/t_s if t_s>0 else 0
    row += "  %10.1fs  " % t_s
    if c != 'OLD(1/ep)': row += "(%.1fx)" % sp
    else: row += "     "
print(row)
