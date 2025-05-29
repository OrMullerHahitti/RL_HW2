# ---------------------------------------------------------------------------
#  cmu_planner.py  –  exact planning for the 5-job single-server queue
# ---------------------------------------------------------------------------
import itertools
import matplotlib.pyplot as plt
import numpy as np
from pyRDDLGym import RDDLEnv





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

