import math
import itertools
import matplotlib.pyplot as plt
import numpy as np
from pyRDDLGym import RDDLEnv

import os
import random


# --- Constants for the N=5 instance ---
# These are from the problem description, useful for implementing policies/value functions
# (job_idx: (mu, cost)) - Using 0-indexed for Python lists/arrays
np.random.seed(42)
# --- RDDL Setup for Question 2 ---
DOMAIN = "cmu_domain.rddl"
INSTANCE = "cmu_n5.rddl"
env = RDDLEnv(domain=DOMAIN, instance=INSTANCE)
model = env.model
jobs = tuple(sorted(model.object_to_type))  # ('j1', 'j2', …)
N = len(jobs)
N_JOBS_N5 = 5
JOB_PARAMS_N5 = {
    0: {'mu': 0.6, 'cost': 1, 'name': 'j1'},  # Job 1
    1: {'mu': 0.5, 'cost': 4, 'name': 'j2'},  # Job 2
    2: {'mu': 0.3, 'cost': 6, 'name': 'j3'},  # Job 3
    3: {'mu': 0.7, 'cost': 2, 'name': 'j4'},  # Job 4
    4: {'mu': 0.1, 'cost': 9, 'name': 'j5'}  # Job 5
}
JOB_NAMES_N5 = [JOB_PARAMS_N5[i]['name'] for i in range(N_JOBS_N5)]
MU = np.array(model.non_fluents['MU'])
COST = np.array(model.non_fluents['COST'])
all_states = [tuple(bits) for bits in itertools.product([False, True], repeat=N)]
state_index = {s: k for k, s in enumerate(all_states)}  # for quick lookup
SUMC = np.zeros(all_states.__len__())  # sum of costs for each state
for m in range(all_states.__len__()):
    for i in range(N_JOBS_N5):
        if not (m & (1 << i)):  # job i unfinished
            SUMC[m] += COST[i]





##------ those 2 functions are only for the simulation runs "episodes" ----------##
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
##------ those 2 functions are only for the simulation runs "episodes" ----------##


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

def cu_poilcy(rddl_state, job_names, costs_arr, mu_arr):
    """Pick the unfinished job with the largest c_i * mu_i value."""
    mask = get_finished_mask(rddl_state)
    unfinished = get_unfinished_job_indices(mask)

    if not unfinished:
        return {}  # all jobs done

    # choose arg-max c_i * mu_i among unfinished
    chosen_idx = max(unfinished, key=lambda i: costs_arr[i] * mu_arr[i])
    chosen_name = job_names[chosen_idx]
    return {f"process_job___{chosen_name}": True}

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
    return COST[[not done for done in state]].sum()


# ---------------------------------------------------------------------------
# 3.  Policy generators
# ---------------------------------------------------------------------------
def make_pi_c():
    """π_c : highest remaining COST."""
    return [(None if all(s) else
             max(unfinished_jobs(s), key=lambda i: COST[i]))
            for s in all_states]


def make_pi_cmu():
    """π_cμ : highest remaining c_i * μ_i."""
    return [(None if all(s) else
             max(unfinished_jobs(s), key=lambda i: COST[i] * MU[i]))
            for s in all_states]


# ---------------------------------------------------------------------------
# 4.  Exact Bellman evaluation for a deterministic policy
# ---------------------------------------------------------------------------
def evaluate_policy(pi):
    V = np.zeros(len(all_states))
    # iterate states in reverse order of #finished so successors already done
    for state in sorted(all_states, key=sum, reverse=True):
        if all(state):  # terminal
            continue
        j = pi[state_index[state]]
        ns = next_state(state, j)
        V[state_index[state]] = step_cost(state) / MU[j] + V[state_index[ns]]
    return V


# ---------------------------------------------------------------------------
# 5.  Policy-iteration (starting from π_c)
# ---------------------------------------------------------------------------
def policy_iteration(pi_init):
    pi = pi_init.copy()
    trace = []

    while True:
        V = evaluate_policy(pi)
        trace.append(V[state_index[tuple([False] * N)]])  # V(s0)
        stable = True
        for s in all_states:
            if all(s):
                continue
            best_j, best_val = pi[state_index[s]], V[state_index[s]]
            for j in unfinished_jobs(s):
                cand = step_cost(s) / MU[j] + V[state_index[next_state(s, j)]]
                if cand < best_val - 1e-12:
                    best_val, best_j, stable = cand, j, False
            pi[state_index[s]] = best_j
        if stable:
            break
    return pi, trace

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


import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')  # Use non-interactive backend when no display is available


def rddl_state_to_tuple(rddl_state, job_names):
    """Convert RDDL state dict to tuple state representation."""
    state = []
    for job_name in job_names:
        finished = rddl_state.get(f"finished___{job_name}", False)
        state.append(finished)
    return tuple(state)


def get_valid_actions_indices(rddl_state):
    """Get list of valid actions (unfinished job indices) from RDDL state."""
    mask = get_finished_mask(rddl_state)
    return get_unfinished_job_indices(mask)


def action_idx_to_rddl(action_idx, job_names):
    """Convert action index to RDDL action dict."""
    if action_idx is None:
        return {}
    job_name = job_names[action_idx]
    return {f"process_job___{job_name}": True}


def td_0_policy_evaluation(env, policy_func, job_names, costs_arr,
                                    step_size_schedule='constant', alpha_constant=0.01,
                                    num_episodes=1000, V_true=None, gamma=1.0):
    """
    TD(0) algorithm for policy evaluation with different step-size schedules.
    """
    # Initialize value function and visit counts
    V = np.zeros(len(all_states))
    visit_counts = np.zeros(len(all_states))

    # Error tracking
    errors_inf = []
    errors_s0 = []

    for episode in range(num_episodes):
        # Reset environment
        rddl_state, _ = env.reset()

        # Convert to tuple state for indexing
        tuple_state = rddl_state_to_tuple(rddl_state, job_names)
        state_idx = state_index[tuple_state]

        done = False
        while not done:
            # Update visit count
            visit_counts[state_idx] += 1

            action_dict = policy_func(rddl_state, job_names, costs_arr)

            # Check if all jobs are done
            if not action_dict:
                break

            # Step environment
            next_rddl_state, reward, terminated, truncated, info = env.step(action_dict)

            # Convert next state
            next_tuple_state = rddl_state_to_tuple(next_rddl_state, job_names)
            next_state_idx = state_index[next_tuple_state]

            # Determine step size based on schedule
            if step_size_schedule == 'visits':
                alpha = 1.0 / visit_counts[state_idx] if visit_counts[state_idx] > 0 else 1.0
            elif step_size_schedule == 'constant':
                alpha = alpha_constant
            elif step_size_schedule == 'decreasing':
                alpha = 10.0 / (100 + visit_counts[state_idx])
            else:
                raise ValueError(f"Unknown step size schedule: {step_size_schedule}")

            # TD(0) update
            td_target = np.abs(reward) + gamma * V[next_state_idx]
            td_error = td_target - V[state_idx]
            V[state_idx] += alpha * td_error

            # Move to next state
            rddl_state = next_rddl_state
            tuple_state = next_tuple_state
            state_idx = next_state_idx
            done = terminated or truncated

        # Calculate errors if true value function is provided
        if V_true is not None:
            error_inf = np.max(np.abs(V_true - V))
            initial_state = tuple([False] * N)
            error_s0 = abs(V_true[state_index[initial_state]] - V[state_index[initial_state]])
            errors_inf.append(error_inf)
            errors_s0.append(error_s0)

    return V, errors_inf, errors_s0, visit_counts



def q_learning_integrated(env, job_names, step_size_schedule='constant', alpha_constant=0.01,
                          epsilon=0.1, num_episodes=5000, V_star=None, gamma=1.0):
    """
    Q-learning algorithm that integrates with  existing state representation.
    """
    Q = np.zeros((len(all_states), N))  # |States| x |Actions|
    visit_counts = np.zeros((len(all_states), N))

    # Error tracking
    errors_inf = []
    errors_s0 = []

    # Track episodes for periodic error calculation
    error_check_interval = 100

    for episode in range(num_episodes):
        # Reset environment
        rddl_state, _ = env.reset()
        tuple_state = rddl_state_to_tuple(rddl_state, job_names)
        state_idx = state_index[tuple_state]

        done = False
        while not done:
            valid_actions = get_valid_actions_indices(rddl_state)

            if not valid_actions:  # All jobs finished
                break

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Explore: choose random valid action
                action_idx = random.choice(valid_actions)
            else:
                # Exploit: choose best valid action according to Q
                q_values_valid = [(a, Q[state_idx, a]) for a in valid_actions]
                action_idx = min(q_values_valid, key=lambda x: x[1])[0]

            # Convert action to RDDL format
            action_dict = action_idx_to_rddl(action_idx, job_names)

            # Step environment
            next_rddl_state, reward, terminated, truncated, info = env.step(action_dict)
            next_tuple_state = rddl_state_to_tuple(next_rddl_state, job_names)
            next_state_idx = state_index[next_tuple_state]

            # Update visit count
            visit_counts[state_idx, action_idx] += 1

            # Determine step size
            if step_size_schedule == 'visits':
                alpha = 1.0 / visit_counts[state_idx, action_idx] if visit_counts[state_idx, action_idx] > 0 else 1.0
            elif step_size_schedule == 'constant':
                alpha = alpha_constant
            elif step_size_schedule == 'decreasing':
                alpha = 10.0 / (100 + visit_counts[state_idx, action_idx])
            else:
                raise ValueError(f"Unknown step size schedule: {step_size_schedule}")

            # Q-learning update
            next_valid_actions = get_valid_actions_indices(next_rddl_state)
            if next_valid_actions:  # Not terminal
                min_next_q = min(Q[next_state_idx, a] for a in next_valid_actions)
            else:  # Terminal state
                min_next_q = 0.0

            td_target = np.abs(reward) + gamma * min_next_q
            td_error = td_target - Q[state_idx, action_idx]
            Q[state_idx, action_idx] += alpha * td_error

            # Move to next state
            rddl_state = next_rddl_state
            tuple_state = next_tuple_state
            state_idx = next_state_idx
            done = terminated or truncated

        # Calculate errors periodically
        if V_star is not None and (episode + 1) % error_check_interval == 0:
            pi_from_q = []
            for s in all_states:
                if all(s):  # Terminal state
                    pi_from_q.append(None)
                else:
                    # Get unfinished jobs for this state
                    valid_acts = [i for i, finished in enumerate(s) if not finished]
                    s_idx = state_index[s]
                    # Choose action with highest Q-value
                    best_action = min(valid_acts, key=lambda a: Q[s_idx, a])
                    pi_from_q.append(best_action)

            # Evaluate the greedy policy using  existing function
            V_pi_from_q = evaluate_policy(pi_from_q)

            error_inf = np.max(np.abs(V_star - V_pi_from_q))
            initial_state = tuple([False] * N)
            error_s0 = abs(V_star[state_index[initial_state]] - V_pi_from_q[state_index[initial_state]])
            errors_inf.append(error_inf)
            errors_s0.append(error_s0)

    return Q, errors_inf, errors_s0


# Plotting functions (reusing  existing color scheme and style)
def plot_td_errors_integrated(errors_dict, title_suffix=""):
    """Plot TD(0) errors """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = ['#6baed6', '#fd8d3c', '#31a354']

    for i, (schedule, (errors_inf, errors_s0)) in enumerate(errors_dict.items()):
        episodes = range(1, len(errors_inf) + 1)
        ax1.plot(episodes, errors_inf, label=f'{schedule}', color=colors[i])
        ax2.plot(episodes, errors_s0, label=f'{schedule}', color=colors[i])

    ax1.set_xlabel('episodes')
    ax1.set_ylabel('||V_true - V_TD||_∞')
    ax1.set_title(f'Max Error {title_suffix}', fontsize=16)
    ax1.legend(loc='upper right')
    ax1.set_ylim(bottom=260)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.set_xlabel('episodes')
    ax2.set_ylabel('|V_true(s0) - V_TD(s0)|')
    ax2.set_title(f'Initial State Error {title_suffix}', fontsize=16)
    ax2.legend(loc='upper right')
    ax2.set_ylim(bottom=0)

    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig


def plot_q_learning_errors_integrated(errors_dict, title_suffix=""):
    """Plot Q-learning errors using  existing plotting style."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = ['#6baed6', '#fd8d3c', '#31a354']

    for i, (schedule, (errors_inf, errors_s0)) in enumerate(errors_dict.items()):
        episodes = [100 * (i + 1) for i in range(len(errors_inf))]
        ax1.plot(episodes, errors_inf, label=f'{schedule}', color=colors[i])
        ax2.plot(episodes, errors_s0, label=f'{schedule}', color=colors[i])

    ax1.set_xlabel('episodes')
    ax1.set_ylabel('||V* - V_π_Q||_∞')
    ax1.set_title(f'NORM-infinity Error {title_suffix}', fontsize=16)
    ax1.legend(loc='upper right')
    ax1.set_ylim(bottom=0)

    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.set_xlabel('episodes')
    ax2.set_ylabel('|V*(s0) - Q(s0,a*)|')
    ax2.set_title(f'Initial State Error {title_suffix}', fontsize=16)
    ax2.legend(loc='upper right')
    ax2.set_ylim(bottom=1e-4)

    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig


def run_learning_experiments_integrated():
    """Run all learning experiments """

    print("Computing reference value functions using existing functions...")
    pi_c = make_pi_c()
    V_c_true = evaluate_policy(pi_c)
    #0,1,5,13,15
    # zero_indices = [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14,16,17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,30,31]
    # for i in zero_indices:
    #     V_c_true[i] = 0  # Set these indices to zero as per the problem statement
    pi_star, V_trace = policy_iteration(pi_c)
    V_star = evaluate_policy(pi_star)

    print(f"True V_c(s0) = {V_c_true[state_index[tuple([False] * N)]]:.4f}")
    print(f"Optimal V*(s0) = {V_star[state_index[tuple([False] * N)]]:.4f}")

    print("\n--- TD(0) Policy Evaluation Experiments ---")

    # TD(0) experiments with different step-size schedules
    step_schedules = ['visits', 'constant', 'decreasing']
    schedule_names = ['1/visits', 'α=0.01', '10/(100+visits)']
    td_errors = {}

    for i, schedule in enumerate(step_schedules):
        print(f"Running TD(0) with {schedule_names[i]} step-size schedule...")
        V_td, errors_inf, errors_s0, _ = td_0_policy_evaluation(
            env, max_cost_policy_q2, JOB_NAMES_N5, COST,
            step_size_schedule=schedule, V_true=V_c_true, num_episodes=2000
        )
        td_errors[schedule_names[i]] = (errors_inf, errors_s0)
        print(f"  Final V_TD(s0) = {V_td[state_index[tuple([False] * N)]]:.4f}")

    # Plot TD(0) results
    print("Plotting TD(0) results...")
    fig_td = plot_td_errors_integrated(td_errors, "(TD(0) Policy Evaluation)")
    plt.savefig("td0_errors_integrated.png", dpi=150, bbox_inches='tight')
    print("Saved plot: td0_errors_integrated.png")

    print("\n--- Q-Learning Experiments ---")

    # Q-learning experiments with ε = 0.1
    print("Running Q-learning experiments with ε = 0.1...")
    q_errors_01 = {}
    for i, schedule in enumerate(step_schedules):
        print(f"Running Q-learning (ε=0.1) with {schedule_names[i]} step-size schedule...")
        Q, errors_inf, errors_s0 = q_learning_integrated(
            env, JOB_NAMES_N5, step_size_schedule=schedule,
            epsilon=0.1, V_star=V_star, num_episodes=5000
        )
        q_errors_01[schedule_names[i]] = (errors_inf, errors_s0)

    # Plot Q-learning results (ε = 0.1)
    print("Plotting Q-learning (ε=0.1) results...")
    fig_q1 = plot_q_learning_errors_integrated(q_errors_01, "(Q-learning, ε=0.1)")
    plt.savefig("qlearning_errors_eps01_integrated.png", dpi=150, bbox_inches='tight')
    print("Saved plot: qlearning_errors_eps01_integrated.png")

    # Q-learning with ε = 0.01 (using decreasing step-size schedule)
    print("\nRunning Q-learning with ε=0.01...")
    favorite_schedule = 'decreasing'
    favorite_name = '10/(100+visits)'
    Q_001, errors_inf_001, errors_s0_001 = q_learning_integrated(
        env, JOB_NAMES_N5, step_size_schedule=favorite_schedule,
        epsilon=0.01, V_star=V_star, num_episodes=5000
    )

    # Compare ε = 0.1 vs ε = 0.01
    print("Plotting epsilon comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    episodes = [100 * (i + 1) for i in range(len(errors_inf_001))]
    colors = ['#6baed6', '#fd8d3c']

    ax1.plot(episodes, q_errors_01[favorite_name][0], label='ε=0.1', color=colors[0])
    ax1.plot(episodes, errors_inf_001, label='ε=0.01', color=colors[1])
    ax1.set_xlabel('episodes')
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel('||V* - V_π_Q||_∞')
    ax1.set_title(f'Norm-INF Error Comparison ({favorite_name} schedule)', fontsize=16)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.plot(episodes, q_errors_01[favorite_name][1], label='ε=0.1', color=colors[0])
    ax2.plot(episodes, errors_s0_001, label='ε=0.01', color=colors[1])
    ax2.set_xlabel('episodes')
    ax2.set_ylim(bottom=0)

    ax2.set_ylabel('|V*(s0) - Q(s0,a*)|')
    ax2.set_title(f'Initial State Error Comparison ({favorite_name} schedule)', fontsize=16)
    ax2.legend(loc='lower right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("qlearning_epsilon_comparison_integrated.png", dpi=150, bbox_inches='tight')
    print("Saved plot: qlearning_epsilon_comparison_integrated.png")

    print("\n--- Learning Experiments Completed ---")

    # Print summary
    print("\nSummary:")
    print(f"True V_πc(s0) = {V_c_true[state_index[tuple([False] * N)]]:.4f}")
    print(f"Optimal V*(s0) = {V_star[state_index[tuple([False] * N)]]:.4f}")

    return td_errors, q_errors_01, (errors_inf_001, errors_s0_001)


def main_with_learning():
    """Extended main function """

    print("--- Starting Question 2: The c-mu Rule ---")

    # πc policy
    print("Running πc policy evaluation...")
    li = []
    for i in range(1000):
        if (i + 1) % 100 == 0:
            print(f"Max Cost Policy Episode {i + 1}")
        ep_total_reward, ep_steps = run_q2_simulation_episode(env, max_cost_policy_q2, JOB_NAMES_N5, costs_arr=COST)
        li.append((ep_total_reward, ep_steps))

    rewards_only = [x[0] for x in li]
    rewards_c = [np.mean(rewards_only[:i + 1]) for i in range(len(rewards_only))]

    print("Running πcμ policy evaluation...")
    li2 = []
    for i in range(1000):
        if (i + 1) % 100 == 0:
            print(f"cμ Policy Episode {i + 1}")
        ep_total_reward, ep_steps = run_q2_simulation_episode(env, cu_poilcy, JOB_NAMES_N5, costs_arr=COST, mu_arr=MU)
        li2.append((ep_total_reward, ep_steps))

    rewards_only2 = [x[0] for x in li2]
    rewards_cu = [np.cumsum(rewards_only2[:i + 1])[i] / (i + 1) for i in range(len(li2))]

    print("Computing exact value functions...")
    pi_c = make_pi_c()
    pi_cmu = make_pi_cmu()
    V_c = evaluate_policy(pi_c)
    V_cmu = evaluate_policy(pi_cmu)

    pi_star, V_trace = policy_iteration(pi_c)
    V_star_final = V_trace[-1]
    V_star_vector = evaluate_policy(pi_star)

    colors = ['#6baed6', '#fd8d3c', '#31a354']
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards_c) + 1), rewards_c, label='πc', color=colors[2])
    plt.legend(loc='upper right')
    plt.title('Policy evaluation for V(S0) for πc over 1000 episodes', fontsize=16)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("policy_eval_cost_integrated.png")
    print("Saved plot: policy_eval_cmu_cost_integrated.png")

    # Plot simulation results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards_cu) + 1), rewards_cu, label='πcμ', color=colors[0])
    plt.plot(range(1, len(rewards_c) + 1), rewards_c, label='πc', color=colors[1])
    plt.legend(loc='upper right')
    plt.title('Policy evaluation for cμ and cost over 1000 episodes', fontsize=16)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("policy_eval_cmu_cost_integrated.png")
    print("Saved plot: policy_eval_cmu_cost_integrated.png")



    # Create a table comparing the value functions
    import pandas as pd

    # Create a DataFrame to store the state and value data
    data = []
    for i, state in enumerate(all_states):
        state_str = ''.join(['1' if x else '0' for x in state])
        data.append({
            'State': state_str,
            'V(πcμ)': V_cmu[i],
            'V(π*)': V_star_vector[i],
            'V(πc)': V_c[i],
        })

    # Create DataFrame and format it
    df = pd.DataFrame(data)
    pd.set_option('display.precision', 4)  # Show 4 decimal places

    # Print the table to console
    print("\nValue Function Comparison Table:")
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv("value_functions_comparison.csv", index=False)
    print("Saved table to: value_functions_comparison.csv")

    # Plot policy iteration convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(V_trace)), V_trace, label='V(π*)', color=colors[0])
    plt.legend(loc='upper right')
    plt.title('Convergence of V(s0) Across Policy‐Iteration Steps', fontsize=16)
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("policy_eval_pi_star_integrated.png")
    print("Saved plot: policy_eval_pi_star_integrated.png")

    print("--- Finished Question 2 Planning Part ---")

    # Now run the learning experiments
    print("\n" + "=" * 50)
    print("STARTING LEARNING EXPERIMENTS")
    print("=" * 50)

    run_learning_experiments_integrated()

    print("\n--- All Experiments Completed ---")


if __name__ == "__main__":
    main_with_learning()

#### part 3 running max cost evaluation
#     li =[]
#     for i in range(1000):
#         print(f"Max Cost Policy Episode {i + 1}:")
#         ep_total_reward, ep_steps = run_q2_simulation_episode(env, max_cost_policy_q2, JOB_NAMES_N5,costs_arr=COST)
#         li.append((ep_total_reward, ep_steps))
#     rewards_only = [x[0] for x in li]
#     rewards_c = [np.mean(rewards_only[:i + 1]) for i in range(len(rewards_only))]
#
#     steps = [x[1] for x in li]
#
#
#     flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.6}
#
#
#     colors = ['#6baed6', '#fd8d3c']
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, len(rewards_c)+1), rewards_c, label='πc', color=colors[0])
#     plt.legend(loc='upper right')
#     plt.title('Policy evaluation for πc over 1000 episodes', fontsize=16)
#     plt.xlabel('episodes')
#     plt.ylabel('reward')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig("ploicy_eval_pi_c.png")
#
#
# #### part 5 running cu evaluation
#     li = []
#     for i in range(1000):
#         print(f"cμ Policy Episode {i + 1}:")
#         ep_total_reward, ep_steps = run_q2_simulation_episode(env, cu_poilcy, JOB_NAMES_N5, costs_arr=COST,
#                                                              mu_arr=MU)
#         li.append((ep_total_reward, ep_steps))
#     rewards_only = [x[0] for x in li]
#     rewards_cu = [np.cumsum(rewards_only[:i+1])[i] / (i + 1) for i in range(len(li))]
#     steps = [x[1] for x in li]
#     flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.6}
#     colors = ['#6baed6', '#fd8d3c','#31a354']
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, len(rewards_cu)+1), rewards_cu, label='πcu', color=colors[0])
#     plt.legend(loc='upper right')
#     plt.title('Policy evaluation for πcμ over 1000 episodes', fontsize=16)
#     plt.xlabel('episodes')
#     plt.ylabel('reward')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.savefig("policy_eval_cmu.png")
#
#     pi_c = make_pi_c()
#     pi_cmu = make_pi_cmu()
#     V_c = evaluate_policy(pi_c)
#     V_cmu = evaluate_policy(pi_cmu)
#
#     pi_star, V_trace = policy_iteration(pi_c)
#
#
#     V_star = V_trace[-1]
#
#
#     #plot reward_c and reward_cu together
#
# #plot reward_c and reward_cu together
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, len(rewards_cu)+1), rewards_cu, label='πcμ', color=colors[0])
#     plt.plot(range(1, len(rewards_c)+1), rewards_c, label='πc', color=colors[1])
#     plt.legend(loc='upper right')
#     plt.title('Policy evaluation for cμ and cost over 1000 episodes', fontsize=16)
#     plt.xlabel('episodes')
#     plt.ylabel('reward')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.savefig("policy_eval_cmu_cost.png")
#
# #show a plot of the V function for c, cμ and π*
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(V_cmu)), V_cmu, label='V(πcμ)', color=colors[1])
#     plt.plot(range(len(V_trace)), V_trace, label='V(π*)', color=colors[2])
#     plt.legend(loc='upper right')
#     plt.title('Value function for  πcμ and π*', fontsize=16)
#     plt.xlabel('state index')
#     plt.ylabel('value')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.savefig("policy_eval_V.png")
#
# #show policy iteration steps
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(V_trace)), V_trace, label='V(π*)', color=colors[0])
#     plt.legend(loc='upper right')
#     plt.title('Value function for π* over iterations', fontsize=16)
#     plt.xlabel('iteration')
#     plt.ylabel('value')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.savefig("policy_eval_pi_star.png")
#
#
#
#     print("--- Finished Question 2 ---")
