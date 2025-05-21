# ---------------------------------------------------------------------------
#  cmu_planner.py  –  exact planning for the 5-job single-server queue
# ---------------------------------------------------------------------------
import itertools
import matplotlib.pyplot as plt
import numpy as np
from pyRDDLGym import RDDLEnv

DOMAIN   = "cmu_domain.rddl"
INSTANCE = "cmu_n5.rddl"

# ---------------------------------------------------------------------------
# 1.  Load model and pull μ and c arrays
# ---------------------------------------------------------------------------
env    = RDDLEnv(domain=DOMAIN, instance=INSTANCE)
model  = env.model
jobs   = tuple(sorted(model.object_to_type))          # ('j1', 'j2', …)
N      = len(jobs)

MU     = np.array(model.non_fluents['MU'])
COST   = np.array(model.non_fluents['COST'])

# ---------------------------------------------------------------------------
# 2.  Enumerate all states  (tuple of booleans, length N)
# ---------------------------------------------------------------------------
all_states = [tuple(bits) for bits in itertools.product([False, True], repeat=N)]
state_index = {s: k for k, s in enumerate(all_states)}      # for quick lookup

def unfinished_jobs(state):
    """Return list of *indices* of jobs not yet finished in this state."""
    return [i for i, done in enumerate(state) if not done]

def next_state(state, job_idx):
    """Return new state where job_idx has become finished."""
    as_list = list(state)
    as_list[job_idx] = True
    return tuple(as_list)

def step_cost(state):
    """Instantaneous holding cost  Σ c_i  over unfinished jobs."""
    return COST[ [not done for done in state] ].sum()

# ---------------------------------------------------------------------------
# 3.  Policy generators
# ---------------------------------------------------------------------------
def make_pi_c():
    """π_c : highest remaining COST."""
    return [ (None if all(s) else
              max(unfinished_jobs(s), key=lambda i: COST[i]) )
             for s in all_states ]

def make_pi_cmu():
    """π_cμ : highest remaining c_i * μ_i."""
    return [ (None if all(s) else
              max(unfinished_jobs(s), key=lambda i: COST[i]*MU[i]) )
             for s in all_states ]

# ---------------------------------------------------------------------------
# 4.  Exact Bellman evaluation for a deterministic policy
# ---------------------------------------------------------------------------
def evaluate_policy(pi):
    V = np.zeros(len(all_states))
    # iterate states in reverse order of #finished so successors already done
    for state in sorted(all_states, key=sum, reverse=True):
        if all(state):            # terminal
            continue
        j   = pi[state_index[state]]
        ns  = next_state(state, j)
        V[state_index[state]] = step_cost(state)/MU[j] + V[state_index[ns]]
    return V

# ---------------------------------------------------------------------------
# 5.  Policy-iteration (starting from π_c)
# ---------------------------------------------------------------------------
def policy_iteration(pi_init):
    pi   = pi_init.copy()
    trace= []

    while True:
        V = evaluate_policy(pi)
        trace.append(V[state_index[tuple([False]*N)]])     # V(s0)
        stable = True
        for s in all_states:
            if all(s):
                continue
            best_j, best_val = pi[state_index[s]], V[state_index[s]]
            for j in unfinished_jobs(s):
                cand = step_cost(s)/MU[j] + V[state_index[next_state(s,j)]]
                if cand < best_val - 1e-12:
                    best_val, best_j, stable = cand, j, False
            pi[state_index[s]] = best_j
        if stable:
            break
    return pi, trace

# ---------------------------------------------------------------------------
# 6.  Run everything ---------------------------------------------------------
pi_c     = make_pi_c()
pi_cmu   = make_pi_cmu()
V_c      = evaluate_policy(pi_c)
V_cmu    = evaluate_policy(pi_cmu)

pi_star, V_trace = policy_iteration(pi_c)
def run_q2_simulation_episode(env, policy_func, job_names_list,**kwargs):
    """Runs a single episode with a given policy function."""
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    # print(f"Initial RDDL State: {state}")
    # print(f"Initial Finished Mask: {get_finished_mask(state):0{N_JOBS_N5}b}")

    for t in range(env.horizon):  # Max steps per episode
        action = policy_func(state, job_names_list,**kwargs)

        if not action:  # No action means all jobs should be finished (or error in policy)
            # print(f"Policy returned no action at step {t+1}. Assuming all jobs done.")
            break

        # print(f"Step {t+1}: State Mask {get_finished_mask(state):0{N_JOBS_N5}b}, Action: {action}")
        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1
        state = next_state

        if terminated or truncated:
            # print(f"Episode finished at step {t+1}. Reason: {'Terminated' if terminated else 'Truncated'}")
            break

    return total_reward, steps
V_star   = V_trace[-1]

def policy_simulate(num_of_episodes:int)
    for i in range(num_of_episodes):

        ep_total_reward, ep_steps = run_q2_simulation_episode(env_q2, random_policy_q2, JOB_NAMES_N5)

# ---------------------------------------------------------------------------
# 7.  Print summary
# ---------------------------------------------------------------------------
print("\nValue at initial state s₀ (no jobs finished)")
print(f" π_c   (max cost) : {V_c[state_index[tuple([False]*N)]]:.2f}")
print(f" cμ-law           : {V_cmu[state_index[tuple([False]*N)]]:.2f}")
print(f" π*    (optimal)  : {V_star:.2f}")
print(f"\nPolicy-iteration converged in {len(V_trace)-1} improvement step(s).")

# ---------------------------------------------------------------------------
# 8.  Plot convergence
# ---------------------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(V_trace, marker='o')
# plt.axhline(V_c[state_index[tuple([False]*N)]],   ls='--', c='C1', label='π_c')
# plt.axhline(V_cmu[state_index[tuple([False]*N)]], ls='--', c='C2', label='cμ-law')
plt.xlabel('Policy-iteration step')
plt.ylabel('V(s₀)')
plt.title('Convergence starting from π_c')
plt.legend()
plt.tight_layout()
plt.savefig("convergence.png")
print("Saved plot: convergence.png")


# Gather all non-terminal states
non_term_states = [s for s in all_states if not all(s)]
N_nt = len(non_term_states)

# 5.1 Count agreements
agree = 0
diffs = []
for s in non_term_states:
    idx = state_index[s]
    a_star = pi_star[idx]
    a_mu   = pi_cmu[idx]
    if a_star == a_mu:
        agree += 1
    else:
        diffs.append((s, a_mu, a_star))

print(f"\nπ* vs cμ-law: {agree}/{N_nt} states agree ({agree/N_nt:.1%})")
if diffs:
    print("\nStates where π* ≠ π_cμ:")
    for s, a_mu, a_star in diffs:
        print(f"  {s} : π_cμ→j{a_mu+1}, π*→j{a_star+1}")

# 5.2 Scatter plot of actions
plt.figure(figsize=(5,5))
x = [pi_cmu[state_index[s]] for s in non_term_states]
y = [pi_star[state_index[s]] for s in non_term_states]
colors = ['green' if xi==yi else 'red' for xi,yi in zip(x,y)]
plt.scatter(x, y, c=colors, s=80, edgecolors='k')
plt.plot([0,N-1],[0,N-1],'k--', lw=1)
plt.xlabel('π_cμ(s) (job index)')
plt.ylabel('π*(s) (job index)')
plt.title('Action comparison: cμ-law vs optimal')
plt.xlim(-0.5, N-0.5)
plt.ylim(-0.5, N-0.5)
plt.xticks(range(N), [f"j{i+1}" for i in range(N)])
plt.yticks(range(N), [f"j{i+1}" for i in range(N)])
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("comaprsion.png")


