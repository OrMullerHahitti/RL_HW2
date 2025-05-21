import math

import numpy as np
import matplotlib.pyplot as plt
from pyRDDLGym import RDDLEnv
import random  # For simple policies or exploration
import time    # For timing policy evaluation

np.random.seed(42)
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
    env_q2 = RDDLEnv(domain=RDDL_DOMAIN_Q2_FILE, instance=RDDL_INSTANCE_Q2_FILE)

    model = env_q2.model

    costs = model.non_fluents['COST']

    print("\n---Random Policy ---")
    num_random_episodes = 3
    for i in range(num_random_episodes):
        print(f"Random Policy Episode {i + 1}:")
        ep_total_reward, ep_steps = run_q2_simulation_episode(env_q2, random_policy_q2, JOB_NAMES_N5)
        print(f"  Total Reward (Negative Cost): {ep_total_reward:.2f}, Steps Taken: {ep_steps}")

    #maximum cost policy
    # Separate rewards and steps


    import matplotlib.pyplot as plt




    li =[]

    for i in range(100):
        print(f"Max Cost Policy Episode {i + 1}:")
        ep_total_reward, ep_steps = run_q2_simulation_episode(env_q2, max_cost_policy_q2, JOB_NAMES_N5,costs_arr=costs)
        li.append((ep_total_reward, ep_steps))

    rewards = [x[0] for x in li]
    steps = [x[1] for x in li]


    flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.6}


    colors = ['#6baed6', '#fd8d3c']

    plt.figure(figsize=(10, 6))
    box = plt.boxplot(
        rewards,
        widths=0.5,
    )


    plt.title('Policy evaluation for pi_c over 100 episodes', fontsize=16)
    plt.xlabel(f'V(S_0) mean : {np.mean(li[0])}')
    plt.ylabel('V(S_0)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("boxplot.png")
    plt.show()

    #bellman equations
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
    #  π_c  – choose unfini`shed job with largest *cost* c_i --------------------
    def run_belman_starting_pi_c():
        pi_c = np.full(N_STATE, -1, dtype=int)
        for m in range(ALL_DONE):
            jobs_left = [i for i in range(N_JOBS_N5) if not (m & (1 << i))]
            pi_c[m] = max(jobs_left, key=lambda i: COSTS[i])

        V_pi_c = eval_policy_bellman(pi_c)
        print(f"\nTheoretical value under π_c   : Vπc(s0) = {V_pi_c[0]:.2f}")

        # ---------- run one *simulation* episode with π_c -------------------------
        print("\nRunning simulation with π_c (max cost policy):")
        ep_total_reward, ep_steps = run_q2_simulation_episode(env_q2, max_cost_policy_q2, JOB_NAMES_N5, costs_arr=COSTS)
        V_trace = [V_pi_c[0]]  # Track value of initial state across iterations
        pi = pi_c.copy()       # Start with the max cost policy
        improve_it = 0         # Count improvement iterations


        # Policy iteration loop
        while True:
            # Policy evaluation step
            V = eval_policy_bellman(pi)

            # Policy improvement step
            stable = True
            for m in range(ALL_DONE):
                best_j, best_val = pi[m], V[m]

                # Try all possible actions (jobs) in this state
                for j in range(N_JOBS_N5):
                    if m & (1 << j):  # Skip if job j is already finished
                        continue


                    cand_val = SUMC[m] / MU[j] + V[m | (1 << j)]

                    # If better than current best, update
                    if cand_val < best_val - 1e-12:  # Small epsilon for numerical stability
                        best_val, best_j = cand_val, j

                # If policy changed for this state, mark as unstable
                if best_j != pi[m]:
                    pi[m] = best_j
                    stable = False

            # Evaluate new policy and track value
            V_new = eval_policy_bellman(pi)
            V_trace.append(V_new[0])
            improve_it += 1

            # Print progress
            print(f"Iteration {improve_it}: V(s0) = {V_new[0]:.2f}")

            # Check for convergence
            if stable:
                break


    # Report results


    # --------------------------------------------------------------------------
    #  Implement cµ law policy ------------------------------------------------
    print("\n--- Implementing cµ law policy ---")

    # cµ law: choose job with highest c_i * µ_i value
    pi_cmu = np.full(N_STATE, -1, dtype=int)
    for m in range(ALL_DONE):
        jobs_left = [i for i in range(N_JOBS_N5) if not (m & (1 << i))]
        if jobs_left:
            pi_cmu[m] = max(jobs_left, key=lambda i: COSTS[i] * MU[i])

    # Evaluate cµ law policy
    V_pi_cmu = eval_policy_bellman(pi_cmu)
    print(f"Value under cµ law policy: V_cmu(s0) = {V_pi_cmu[0]:.2f}")

    # Define optimal policy from policy iteration


    # Compare policies
    print("\n--- Comparing Policies ---")
    print(f"Value under π_c (max cost)   : V_c(s0) = {V_pi_c[0]:.2f}")
    print(f"Value under cµ law           : V_cmu(s0) = {V_pi_cmu[0]:.2f}")
    print(f"Value under optimal policy π*: V*(s0) = {V_pi_star:.2f}")

    # Run simulation with cµ law policy
    def cmu_policy_q2(rddl_state, job_names, costs_arr, mu_arr):
        """Pick the unfinished job with the largest c_i * µ_i value."""
        mask = get_finished_mask(rddl_state)
        unfinished = get_unfinished_job_indices(mask)

        if not unfinished:
            return {}  # all jobs done

        # choose arg-max c_i * µ_i among unfinished
        chosen_idx = max(unfinished, key=lambda i: costs_arr[i] * mu_arr[i])
        chosen_name = job_names[chosen_idx]
        return {f"process_job___{chosen_name}": True}

    print("\nRunning simulation with cµ law policy:")
    ep_total_reward, ep_steps = run_q2_simulation_episode(env_q2, cmu_policy_q2, JOB_NAMES_N5, 
                                                         costs_arr=COSTS, mu_arr=MU)
    print(f"  Total Reward (Negative Cost): {ep_total_reward:.2f}, Steps Taken: {ep_steps}")

    # --------------------------------------------------------------------------
    #  PLOTS: Visualize results -----------------------------------------------
    print("\n--- Creating Plots ---")

    # PLOT 1: Value of initial state during policy iteration
    print("Plot 1: Value of initial state during policy iteration")
    plt.figure(figsize=(10, 6))

    # Plot the value trace during policy iteration
    plt.plot(range(len(V_trace)), V_trace, marker='o', linewidth=2, 
             label='V(s₀) during policy iteration')

    # Add reference lines for different policies
    plt.axhline(y=V_pi_c[0], color='r', linestyle='--', linewidth=2, 
                label='V(s₀) for π_c (max cost policy)')
    plt.axhline(y=V_pi_cmu[0], color='g', linestyle='--', linewidth=2, 
                label='V(s₀) for cµ law policy')

    # Customize plot
    plt.xticks(range(len(V_trace)))
    plt.xlabel("Policy-iteration step", fontsize=12)
    plt.ylabel("Value V(s₀)", fontsize=12)
    plt.title("Value of Initial State During Policy Iteration", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plt.savefig("policy_iteration_convergence.png")
    print("Saved plot to: policy_iteration_convergence.png")

    # PLOT 2: Comparison of policy values (V π* vs. V πc vs. V cµ)
    print("Plot 2: Comparison of policy values")
    plt.figure(figsize=(10, 6))

    # Data for bar chart
    policies = ['π_c (max cost)', 'cµ law', 'π* (optimal)']
    values = [V_pi_c[0], V_pi_cmu[0], V_pi_star]
    colors = ['#3498db', '#2ecc71', '#e74c3c']  # Blue, Green, Red

    # Create bar chart
    bars = plt.bar(policies, values, color=colors, width=0.6)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', fontsize=12)

    # Customize plot
    plt.ylabel('Value V(s₀)', fontsize=12)
    plt.title('Comparison of Policy Values', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save plot
    plt.savefig("policy_comparison.png")
    print("Saved plot to: policy_comparison.png")

    # Print policy decisions for each state (for small number of states)
    if N_JOBS_N5 <= 5:  # Only print for small instances
        print("\n--- Policy Decisions for Each State ---")
        print("State (binary) | π_c | cµ law | π*")
        print("-" * 40)
        for m in range(N_STATE):
            if m == ALL_DONE:
                continue  # Skip terminal state
            state_bin = format(m, f'0{N_JOBS_N5}b')
            print(f"{state_bin} | j{pi_c[m]+1} | j{pi_cmu[m]+1} | j{pi_star[m]+1}")

    # --------------------------------------------------------------------------
    #  Summary and Conclusions ------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS".center(70))
    print("="*70)

    # Policy comparison
    print("\n1. Policy Values Comparison:")
    print(f"   - π_c (max cost policy)  : {V_pi_c[0]:.2f}")
    print(f"   - cµ law policy          : {V_pi_cmu[0]:.2f}")
    print(f"   - π* (optimal policy)    : {V_pi_star:.2f}")

    # Performance improvement
    pi_c_improvement = ((V_pi_c[0] - V_pi_star) / V_pi_c[0]) * 100
    cmu_improvement = ((V_pi_cmu[0] - V_pi_star) / V_pi_cmu[0]) * 100
    print("\n2. Performance Improvement:")
    print(f"   - π* improves over π_c by {pi_c_improvement:.2f}%")
    print(f"   - π* improves over cµ law by {cmu_improvement:.2f}%")

    # Policy iteration convergence
    print("\n3. Policy Iteration:")
    print(f"   - Convergence steps required: {improve_it}")
    print(f"   - Initial value (π_c): {V_trace[0]:.2f}")
    print(f"   - Final value (π*): {V_trace[-1]:.2f}")

    # Comparison to cµ law
    if abs(V_pi_cmu[0] - V_pi_star) < 1e-6:
        cmu_optimality = "The cµ law policy is optimal for this problem!"
    elif V_pi_cmu[0] - V_pi_star < 1:
        cmu_optimality = "The cµ law policy is very close to optimal."
    else:
        cmu_optimality = "The cµ law policy is not optimal for this problem."

    print("\n4. Comparison to cµ Law:")
    print(f"   - {cmu_optimality}")

    # Generated plots
    print("\n5. Generated Plots:")
    print("   - policy_iteration_convergence.png: Shows how V(s₀) changes during policy iteration")
    print("   - policy_comparison.png: Compares the values of different policies")

    print("\nAnalysis complete. Check the generated plots for visualization.")
    print("="*70)
