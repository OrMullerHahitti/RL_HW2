# Planning and Reinforcement Learning - Homework 2

This project implements solutions for two reinforcement learning problems using RDDL (Relational Dynamic influence Diagram Language) domains and pyRDDLGym.

## Project Structure

```
├── bandit_domain.rddl          # RDDL domain for multi-armed bandit
├── bandit_n100.rddl           # Instance with 100 arms
├── cmu_domain.rddl            # RDDL domain for job scheduling (cμ rule)
├── cmu_n5.rddl               # Instance with 5 jobs
├── main.py                   # Question 1: Bandit algorithms
├── main2.py                  # Question 2: Job scheduling with RL
└── README.md
```

## Requirements

Install the required dependencies:
```bash
pip install pyRDDLGym numpy matplotlib pandas
```

## How to Run

### Question 1: Multi-Armed Bandit Simulation
```bash
python main.py
```
This runs:
- Random policy
- Greedy explore-then-exploit (100 pulls per arm)
- UCB1 algorithm with theoretical upper bound
- Generates `bandit_regret_plot.png` comparing all algorithms

### Question 2: Job Scheduling (cμ Rule)
```bash
python main2.py
```
This runs the complete pipeline:
- **Planning**: Policy iteration, value function evaluation
- **Learning**: TD(0) policy evaluation and Q-learning with different step-size schedules
- Generates multiple plots and a comparison table

## Output Files

**Question 1:**
- `bandit_regret_plot.png` - Regret comparison across 20,000 timesteps

**Question 2:**
- `policy_eval_cost_integrated.png` - πc policy evaluation
- `estimated_value_of_V(S0)-pi_star_vs_pi_c.png` - Policy comparison
- `policy_eval_pi_star_integrated.png` - Policy iteration convergence
- `td0_errors_integrated.png` - TD(0) learning errors
- `qlearning_errors_eps01_integrated.png` - Q-learning errors (ε=0.1)
- `qlearning_epsilon_comparison_integrated.png` - ε comparison
- `value_functions_comparison.csv` - State-value table

## Problem Descriptions

**Question 1**: 100-armed bandit where arm `i` has Bernoulli reward with probability `i/(n+1)`. Compares random, greedy, and UCB1 strategies.

**Question 2**: Single-server job scheduling where jobs have completion probabilities μᵢ and waiting costs cᵢ. Implements the optimal cμ rule and tests various RL algorithms.

## Key Results

- UCB1 significantly outperforms random and greedy strategies
- The cμ rule (πcμ) is proven optimal for job scheduling
- TD(0) and Q-learning successfully learn near-optimal policies with appropriate step-size schedules