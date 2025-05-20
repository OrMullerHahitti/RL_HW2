import numpy as np
import matplotlib.pyplot as plt
from pyRDDLGym import RDDLEnv
import random  # For simple policies or exploration

# --- RDDL Setup for Question 2 ---
RDDL_DOMAIN_Q2_FILE = "cmu_domain.rddl"
RDDL_INSTANCE_Q2_FILE = "cmu_n5.rddl"

# --- Constants for the N=5 instance ---
# These are from the problem description, useful for implementing policies/value functions
# (job_idx: (mu, cost)) - Using 0-indexed for Python lists/arrays
JOB_PARAMS_N5 = {
    0: {'mu': 0.6, 'cost': 1, 'name': 'j1'},  # Job 1
    1: {'mu': 0.5, 'cost': 4, 'name': 'j2'},  # Job 2
    2: {'mu': 0.3, 'cost': 6, 'name': 'j3'},  # Job 3
    3: {'mu': 0.7, 'cost': 2, 'name': 'j4'},  # Job 4
    4: {'mu': 0.1, 'cost': 9, 'name': 'j5'}  # Job 5
}
N_JOBS_N5 = 5
JOB_NAMES_N5 = [JOB_PARAMS_N5[i]['name'] for i in range(N_JOBS_N5)]


# --- Helper: State Representation ---
# A common way to represent the set of finished jobs is a bitmask.
# State 0b00000 means no jobs finished. State 0b11111 means all 5 jobs finished.

def get_finished_mask(rddl_state):
    """Converts RDDL state dict to a bitmask of finished jobs."""
    mask = 0
    for i in range(N_JOBS_N5):
        job_name = JOB_NAMES_N5[i]
        if rddl_state.get(f"finished___{job_name}", False):
            mask |= (1 << i)
    return mask


def get_unfinished_job_indices(mask):
    """Returns a list of 0-indexed job indices that are NOT finished in the mask."""
    unfinished = []
    for i in range(N_JOBS_N5):
        if not (mask & (1 << i)):
            unfinished.append(i)
    return unfinished


# --- Example: Simple Random Policy for pyRDDLGym ---
def random_policy_q2(rddl_state, job_names):
    """
    A simple policy that picks a random unfinished job to process.
    Returns an action dictionary for pyRDDLGym.
    """
    mask = get_finished_mask(rddl_state)
    unfinished_indices = get_unfinished_job_indices(mask)

    if not unfinished_indices:  # All jobs finished
        return {}  # No action to take

    chosen_job_idx = random.choice(unfinished_indices)
    chosen_job_name = job_names[chosen_job_idx]

    # Action format for pyRDDLGym (using ___ separator, adjust if your version differs)
    action = {f"process_job___{chosen_job_name}": True}
    return action
def max_cost_policy_q2(rddl_state, job_names, costs_arr):
    """Pick the unfinished job with the largest cost c_i."""
    mask = get_finished_mask(rddl_state)
    unfinished = get_unfinished_job_indices(mask)

    if not unfinished:
        return {}                                  # all jobs done

    # choose arg-max cost among unfinished
    chosen_idx   = max(unfinished, key=lambda i: costs_arr[i])
    chosen_name  = job_names[chosen_idx]
    return {f"process_job___{chosen_name}": True}

# --- Example: Basic Simulation Run ---
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

    # print(f"Final RDDL State: {state}")
    # print(f"Final Finished Mask: {get_finished_mask(state):0{N_JOBS_N5}b}")
    # print(f"Total Reward (Negative Cost): {total_reward}, Steps: {steps}\n")
    return total_reward, steps


# --- Planning Part (Q2.Planning.1) ---
def print_mdp_details():
    print("--- Q2.Planning.1: MDP Details ---")
    num_states = 2 ** N_JOBS_N5
    print(f"Number of jobs (N): {N_JOBS_N5}")
    print(f"Number of states: 2^N = {num_states}")
    print(f"Actions per state: Varies. In a state with 'k' unfinished jobs, 'k' actions are available.")
    print(f"Max actions in any non-terminal state: {N_JOBS_N5}")
    print("Bellman's Equation (for minimizing cost V(s)):")
    print("  V(s_terminal) = 0")
    print("  V(s) = (sum_{k in s} c_k) + min_{j in s_unfinished} { mu_j * V(s_union_{j}) + (1-mu_j) * V(s) }")
    print("  Rearranged: V(s) = min_{j in s_unfinished} { ( (sum_{k in s} c_k) / mu_j ) + V(s_union_{j}) }")
    print("  where s_union_{j} is the state s with job j additionally marked as finished.")
    print("  Note: sum_{k in s} c_k is the instantaneous cost for being in state s (all unfinished jobs).")
    print("-" * 30)


# --- MAIN SCRIPT EXECUTION ---
if __name__ == "__main__":
    print("--- Starting Question 2: The c-mu Rule ---")

    # Print MDP details as per Q2.Planning.1
    print_mdp_details()

    # Initialize pyRDDLGym environment for Question 2
    try:
        env_q2 = RDDLEnv(domain=RDDL_DOMAIN_Q2_FILE, instance=RDDL_INSTANCE_Q2_FILE)
    except Exception as e:
        print(f"Error initializing RDDLEnv for Q2: {e}")
        print(f"Ensure RDDL files '{RDDL_DOMAIN_Q2_FILE}' and '{RDDL_INSTANCE_Q2_FILE}' are correct.")
        exit()

    model = env_q2.model
    costs = model.non_fluents['COST']
    print("\n--- Running a few episodes with a Random Policy ---")
    num_random_episodes = 3
    for i in range(num_random_episodes):
        print(f"Random Policy Episode {i + 1}:")
        ep_total_reward, ep_steps = run_q2_simulation_episode(env_q2, random_policy_q2, JOB_NAMES_N5)
        print(f"  Total Reward (Negative Cost): {ep_total_reward:.2f}, Steps Taken: {ep_steps}")
    li =[]
    for i in range(20):
        print(f"Max Cost Policy Episode {i + 1}:")
        ep_total_reward, ep_steps = run_q2_simulation_episode(env_q2, max_cost_policy_q2, JOB_NAMES_N5,costs_arr=costs)
        li.append(ep_total_reward)

    job_objs = sorted(model.object_to_type)
    MU = np.array(model.non_fluents['MU'])
    COSTS = np.array(model.non_fluents['COST'])

    N_STATE = 1 << N_JOBS_N5
    ALL_DONE = N_STATE - 1

    # pre-compute Σ c_k for every mask (instantaneous cost)
    SUMC = np.zeros(N_STATE)
    for m in range(N_STATE):
        for i in range(N_JOBS_N5):
            if not (m & (1 << i)):  # job i unfinished
                SUMC[m] += COSTS[i]


    # --------------------------------------------------------------------------
    def eval_policy_bellman(pi):
        """Exact V[mask] for deterministic pi[mask] (job index)."""
        V = np.zeros(N_STATE)
        for m in range(ALL_DONE - 1, -1, -1):  # reverse topological order
            j = pi[m]
            next_m = m | (1 << j)
            V[m] = SUMC[m] / MU[j] + V[next_m]  # Bellman equation for this MDP
        return V


    # --------------------------------------------------------------------------
    #  π_c  – choose unfinished job with largest *cost* c_i --------------------
    pi_c = np.full(N_STATE, -1, dtype=int)
    for m in range(ALL_DONE):
        jobs_left = [i for i in range(N_JOBS_N5) if not (m & (1 << i))]
        pi_c[m] = max(jobs_left, key=lambda i: COSTS[i])

    V_pi_c = eval_policy_bellman(pi_c)
    print(f"\nTheoretical value under π_c   : Vπc(s0) = {V_pi_c[0]:.2f}")

    # ---------- run one *simulation* episode with π_c -------------------------


    # --------------------------------------------------------------------------
    #  Policy-iteration starting from π_c --------------------------------------
    V_trace = [V_pi_c[0]]
    pi = pi_c.copy()
    improve_it = 0

    while True:
        V = eval_policy_bellman(pi)  # policy evaluation
        # ------- improvement step ---------------------------------------------
        stable = True
        for m in range(ALL_DONE):
            best_j, best_val = pi[m], V[m]
            for j in range(N_JOBS_N5):
                if m & (1 << j):  # already finished
                    continue
                cand_val = SUMC[m] / MU[j] + V[m | (1 << j)]
                if cand_val < best_val - 1e-12:
                    best_val, best_j = cand_val, j
            if best_j != pi[m]:
                pi[m] = best_j
                stable = False
        V_new = eval_policy_bellman(pi)
        V_trace.append(V_new[0])
        improve_it += 1
        if stable:
            break

    print(f"Optimal value V*(s0)            : {V_trace[-1]:.2f}")
    print(f"Policy-iteration improvement steps needed: {improve_it}")

    # --------------------------------------------------------------------------
    #  PLOT:  V(s0) during policy iteration ------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(V_trace)), V_pi_c[0], marker='o')
    plt.xticks(range(len(V_trace)))
    plt.xlabel("Policy-iteration step")
    plt.ylabel("V(s₀)")
    plt.title("Policy-evaulation for π_c)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots_q2_pi_c.png")