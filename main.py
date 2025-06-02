import numpy as np
import matplotlib.pyplot as plt
from pyRDDLGym import RDDLEnv
import random
import math

# --- RDDL Setup ---
RDDL_DOMAIN_FILE = "bandit_domain.rddl"
RDDL_INSTANCE_FILE = "bandit_n100.rddl"

# --- Simulation Parameters ---
N_ARMS = 100
SIMULATION_LENGTH = 20000
N_EXPERIMENTS = 20


# --- Agent Base Class ---
class BanditAgent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # Arm names in RDDL are "a1", "a2", ..., "a100"
        self.arm_names = [f"a{i + 1}" for i in range(n_arms)]
        self.reset()

    def choose_arm(self, timestep):
        """Returns the string name of the arm to pull (e.g., 'a5')."""
        raise NotImplementedError("choose_arm method must be implemented by subclass")

    def update(self, chosen_arm_name, reward, timestep):
        """Updates agent's internal knowledge based on the reward."""
        pass

    def reset(self):
        """Resets agent's internal state for a new experiment."""
        pass


# --- Random Agent ---
class RandomAgent(BanditAgent):
    def choose_arm(self, timestep):
        chosen_idx = random.randint(0, self.n_arms - 1)
        return self.arm_names[chosen_idx]


# --- Greedy (Explore-then-Exploit) Agent ---
class GreedyExploreExploitAgent(BanditAgent):
    def __init__(self, n_arms, explore_pulls_per_arm=100):
        super().__init__(n_arms)
        self.explore_pulls_per_arm = explore_pulls_per_arm
        self.total_explore_steps = n_arms * explore_pulls_per_arm
        self.best_arm_name_after_exploration = None

    def reset(self):
        self.pull_counts = np.zeros(self.n_arms, dtype=float)
        self.sum_rewards = np.zeros(self.n_arms, dtype=float)
        self.empirical_means = np.zeros(self.n_arms, dtype=float)
        self.current_explore_arm_idx = 0
        self.pulls_for_current_explore_arm = 0
        self.best_arm_name_after_exploration = None

    def choose_arm(self, timestep):
        if timestep <= self.total_explore_steps:
            chosen_arm_name = self.arm_names[self.current_explore_arm_idx]
            return chosen_arm_name
        else:
            if self.best_arm_name_after_exploration is None:
                if np.sum(self.pull_counts) == 0:
                    return self.arm_names[random.randint(0, self.n_arms - 1)]
                best_idx = np.argmax(self.empirical_means)
                self.best_arm_name_after_exploration = self.arm_names[best_idx]
            return self.best_arm_name_after_exploration

    def update(self, chosen_arm_name, reward, timestep):
        arm_idx = self.arm_names.index(chosen_arm_name)

        self.pull_counts[arm_idx] += 1
        self.sum_rewards[arm_idx] += reward
        self.empirical_means[arm_idx] = self.sum_rewards[arm_idx] / self.pull_counts[arm_idx]

        if timestep <= self.total_explore_steps:
            self.pulls_for_current_explore_arm += 1
            if self.pulls_for_current_explore_arm >= self.explore_pulls_per_arm:
                self.current_explore_arm_idx += 1
                self.pulls_for_current_explore_arm = 0
                if self.current_explore_arm_idx >= self.n_arms and timestep < self.total_explore_steps:
                    self.current_explore_arm_idx = self.n_arms - 1


# --- UCB1 Agent ---
class UCB1Agent(BanditAgent):
    def __init__(self, n_arms, c_exploration_factor=np.sqrt(2)):
        super().__init__(n_arms)
        self.c = c_exploration_factor

    def reset(self):
        self.pull_counts = np.zeros(self.n_arms, dtype=float)
        self.sum_rewards = np.zeros(self.n_arms, dtype=float)
        self.empirical_means = np.zeros(self.n_arms, dtype=float)

    def choose_arm(self, timestep):
        # Initial phase: pull each arm once
        for i in range(self.n_arms):
            if self.pull_counts[i] == 0:
                return self.arm_names[i]

        # UCB selection phase
        ucb_scores = np.zeros(self.n_arms)
        log_total_pulls = math.log(max(1, timestep))

        for i in range(self.n_arms):
            exploration_bonus = self.c * math.sqrt(log_total_pulls / self.pull_counts[i])
            ucb_scores[i] = self.empirical_means[i] + exploration_bonus

        chosen_idx = np.argmax(ucb_scores)
        return self.arm_names[chosen_idx]

    def update(self, chosen_arm_name, reward, timestep):
        arm_idx = self.arm_names.index(chosen_arm_name)

        self.pull_counts[arm_idx] += 1
        self.sum_rewards[arm_idx] += reward
        self.empirical_means[arm_idx] = self.sum_rewards[arm_idx] / self.pull_counts[arm_idx]


# --- Helper function to create action dictionary ---
def create_action_dict(arm_name, env):
    """Create action dictionary with proper format for pyRDDLGym."""
    # Try different possible action formats
    possible_formats = [
        {f"choose_arm___{arm_name}": True},
        {f"choose_arm__{arm_name}": True},
        {f"choose-arm__{arm_name}": True},
        {('choose_arm', arm_name): True}
    ]

    # For now, let's use the triple underscore format as in your original code
    return {f"choose_arm___{arm_name}": True}


# --- Simulation Function for a single experiment ---
def run_single_experiment(agent, env, sim_length):
    """Runs one experiment for a given agent and returns rewards at each step."""
    agent.reset()
    env.reset()

    rewards_at_each_step = np.zeros(sim_length)

    for t in range(sim_length):
        arm_to_pull_name = agent.choose_arm(timestep=t + 1)

        # Create action dictionary
        action_for_rddl = create_action_dict(arm_to_pull_name, env)

        try:
            _next_state, reward_from_env, _terminated, _truncated, _info = env.step(action_for_rddl)
            rewards_at_each_step[t] = reward_from_env
            agent.update(arm_to_pull_name, reward_from_env, timestep=t + 1)
        except Exception as e:
            print(f"Error at timestep {t + 1}: {e}")
            print(f"Action attempted: {action_for_rddl}")
            print(f"Arm name: {arm_to_pull_name}")
            # Try to inspect the expected action format
            if hasattr(env, 'action_space'):
                print(f"Action space: {env.action_space}")
            break

    return rewards_at_each_step


# --- Main Execution Logic ---
if __name__ == "__main__":
    # Initialize the pyRDDLGym environment
    try:
        env = RDDLEnv(domain=RDDL_DOMAIN_FILE, instance=RDDL_INSTANCE_FILE)
        print("Environment initialized successfully")

        # Print some debug info about the environment
        if hasattr(env, 'action_space'):
            print(f"Action space: {env.action_space}")
        if hasattr(env, 'model'):
            print(f"Action fluents: {list(env.model.action_fluents.keys())}")

    except Exception as e:
        print(f"Error initializing RDDLEnv: {e}")
        print(f"Please ensure '{RDDL_DOMAIN_FILE}' and '{RDDL_INSTANCE_FILE}' are correct and in the path.")
        exit()

    # Optimal arm probability (arm a100 has prob 100/101)
    p_star = N_ARMS / (N_ARMS + 1.0)

    # Store average cumulative regrets for plotting
    all_agents_avg_cumulative_regrets = {}

    # Define the agents to run
    agents_to_test = {
        "Random": RandomAgent(N_ARMS),
        "Greedy (100 pulls/arm)": GreedyExploreExploitAgent(N_ARMS, explore_pulls_per_arm=100),
        "UCB1 (c=sqrt(2))": UCB1Agent(N_ARMS, c_exploration_factor=math.sqrt(2))
    }

    for agent_name, current_agent in agents_to_test.items():
        print(f"Running simulations for: {agent_name}")

        # Store cumulative regrets for each experiment
        all_runs_cumulative_regrets_for_this_agent = np.zeros((N_EXPERIMENTS, SIMULATION_LENGTH))

        for exp_num in range(N_EXPERIMENTS):
            if (exp_num + 1) % 5 == 0 or exp_num == 0 or exp_num == N_EXPERIMENTS - 1:
                print(f"  Experiment {exp_num + 1}/{N_EXPERIMENTS} for {agent_name}...")

            # Run one full simulation
            rewards_this_run = run_single_experiment(current_agent, env, SIMULATION_LENGTH)

            # Calculate cumulative rewards for this run
            cumulative_rewards_this_run = np.cumsum(rewards_this_run)

            # Calculate optimal cumulative rewards
            timesteps_for_regret = np.arange(1, SIMULATION_LENGTH + 1)
            optimal_cumulative_rewards = timesteps_for_regret * p_star

            # Calculate cumulative regret for this run
            cumulative_regret_this_run = optimal_cumulative_rewards - cumulative_rewards_this_run
            all_runs_cumulative_regrets_for_this_agent[exp_num, :] = cumulative_regret_this_run

        # Average the cumulative regret over all experiments
        avg_cumulative_regret_for_this_agent = np.mean(all_runs_cumulative_regrets_for_this_agent, axis=0)
        all_agents_avg_cumulative_regrets[agent_name] = avg_cumulative_regret_for_this_agent

    env.close()

    # --- Plotting Results ---
    plt.figure(figsize=(14, 9))
    plot_timesteps = np.arange(1, SIMULATION_LENGTH + 1)

    for agent_name, avg_regret_data in all_agents_avg_cumulative_regrets.items():
        plt.plot(plot_timesteps, avg_regret_data, label=agent_name, linewidth=2)

    # Calculate and plot UCB1 Theoretical Upper Bound
    if "UCB1 (c=sqrt(2))" in all_agents_avg_cumulative_regrets:
        ucb1_theoretical_bound = np.zeros(SIMULATION_LENGTH)
        arm_probabilities = np.array([(i + 1) / (N_ARMS + 1.0) for i in range(N_ARMS)])

        for t_idx, t_val_sim_step in enumerate(plot_timesteps):
            current_sum_for_bound = 0
            log_t_for_bound = math.log(max(1, t_val_sim_step))

            for arm_idx in range(N_ARMS):
                p_i = arm_probabilities[arm_idx]
                if p_i < p_star:
                    delta_i = p_star - p_i
                    if delta_i > 1e-9:
                        term1_bound = (8 * log_t_for_bound) / delta_i
                        term2_bound = (1 + (math.pi ** 2) / 3) * delta_i
                        current_sum_for_bound += term1_bound + term2_bound
            ucb1_theoretical_bound[t_idx] = current_sum_for_bound

        plt.plot(plot_timesteps, ucb1_theoretical_bound, label="UCB1 Theoretical Upper Bound",
                 linestyle='--', color='black', linewidth=2)

    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Average Cumulative Regret", fontsize=12)
    plt.title(f"Bandit Algorithms Performance (Arms={N_ARMS}, Averaged over {N_EXPERIMENTS} Experiments)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.7)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("bandit_regret_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("Script finished. Plot saved as 'bandit_regret_plot.png'")

    # Print final regret values
    print("\nFinal cumulative regret values:")
    for agent_name, regret_data in all_agents_avg_cumulative_regrets.items():
        print(f"{agent_name}: {regret_data[-1]:.2f}")