	Fahim Tajwar ...
	26 May 2025
	ICML 2025
# Abstract

This paper introduces PAPRIKA, a fine-tuning approach designed to enhance the multi-turn decision-making capabilities of large language models (LLMs). The method focuses on enabling LLMs to strategically gather information required to solve tasks efficiently, rather than exploring all knowable information. PAPRIKA demonstrates improvements over baseline models in various environments such as guessing games and cellular automata, showing both quantitative and qualitative enhancements in performance.

---

# Conclusion

The study highlights that PAPRIKA effectively improves the exploration strategies of LLMs across multiple unseen tasks, generalizing zero-shot without explicit training on those tasks. It addresses the exploration-exploitation trade-off by focusing only on task-relevant information, leading to efficient problem-solving. These findings suggest that PAPRIKA provides a scalable and effective framework for training generally curious agents capable of strategic decision-making in partially observable, multi-turn environments.

---

# Definitions and Literature Review

### Exploration-Exploitation Trade-off
A fundamental concept in reinforcement learning where an agent must balance between exploring new actions to discover potentially better rewards and exploiting known actions that yield immediate rewards.

### Intrinsic Motivation
A paradigm in reinforcement learning where agents are driven by internal incentives, often curiosity-based, to explore their environment without external rewards.

### Multi-Turn Interaction
Refers to sequential decision-making scenarios where an agent interacts with an environment over multiple steps, incorporating historical context into its decisions.

### Curriculum Learning
A training strategy where the complexity of tasks presented to the model increases gradually, mimicking how humans learn complex concepts progressively.

### Reinforcement Learning (RL)
A machine learning paradigm where an agent learns to make decisions by performing actions in an environment to maximize some notion of cumulative reward.

### Chain-of-Thought Prompting (COT)
A prompting technique used in LLMs to encourage reasoning by generating intermediate logical steps before producing a final answer.

---

### Literature Review

Efficient exploration remains a core challenge in intelligent systems, particularly when interacting with partially observable environments. Traditional approaches like intrinsic motivation emphasize curiosity-driven exploration but often lead to inefficient or unfocused behavior [Chen et al., 2017; Osband et al., 2016]. The exploration-exploitation dilemma has been extensively studied in reinforcement learning, with algorithms like Thompson Sampling [Thompson, 1933] and Upper Confidence Bound [Auer et al., 2002] offering principled solutions in tabular settings.

Recent works have explored the use of chain-of-thought prompting to improve reasoning capabilities in LLMs [Wei et al., 2022; Kojima et al., 2022], while others focus on multi-turn interaction benchmarks to evaluate sequential decision-making [Abdulhai et al., 2023]. Curriculum learning has also shown promise in improving generalization by structuring training from simpler to more complex tasks [Bengio et al., 2009].

PAPRIKA builds upon these foundations by combining strategic exploration with curriculum-based fine-tuning, enabling LLMs to generalize across diverse environments while maintaining efficiency.

# Mathematics Needed to Understand the Concepts

To fully grasp the mathematical foundations underlying the concepts in the paper *Training a Generally Curious Agent (PAPRIKA)*, one must be familiar with several key areas of probability, statistics, and reinforcement learning theory. These are essential for understanding how PAPRIKA balances exploration and exploitation, evaluates task difficulty, and optimizes decision-making over multiple turns.

---

## Markov Decision Processes (MDPs)

The paper operates under the assumption that tasks follow a **Partially Observable Markov Decision Process (POMDP)**, which is a generalization of MDPs. An MDP is defined by the tuple $(S, A, T, R, \gamma)$:

- $S$: State space
- $A$: Action space
- $T(s' | s, a)$: Transition dynamics
- $R(s, a)$: Reward function
- $\gamma \in [0,1]$: Discount factor

The goal is to find a policy $\pi(a|s)$ that maximizes the expected cumulative reward:
$$
J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^\infty \gamma^t r_t \right]
$$

Since the environment is partially observable, the agent only receives observations $o \in O$ rather than full states $s$, making belief-state tracking or memory-based strategies necessary.

---

## Exploration Strategies and Bandit Algorithms

PAPRIKA draws inspiration from classical exploration strategies used in multi-armed bandits and RL:

### Upper Confidence Bound (UCB)
UCB selects actions based on an upper confidence bound of their estimated rewards:
$$
a_t = \arg\max_a \left( \hat{\mu}_a + c \sqrt{\frac{\log t}{n_a}} \right)
$$
where:
- $\hat{\mu}_a$: empirical mean reward of action $a$
- $n_a$: number of times action $a$ has been taken
- $c$: exploration constant

This encourages the agent to explore less-known actions while still exploiting high-reward ones.

### Thompson Sampling
Thompson Sampling balances exploration and exploitation probabilistically by sampling from the posterior distribution of rewards and selecting the best sampled action.

These methods formalize the trade-off between gathering information and maximizing utility—central to PAPRIKA's design.

---

## Coefficient of Variation for Task Difficulty Estimation

One of the core metrics used in PAPRIKA is the **coefficient of variation**, denoted as:
$$
\nu_\pi(\tau) = \frac{\sqrt{\sigma_\pi^2(\tau)}}{R^\pi(\tau)}
$$
where:
- $\sigma_\pi^2(\tau)$: variance of the reward distribution under policy $\pi$ on task $\tau$
- $R^\pi(\tau)$: mean reward of the policy on task $\tau$

This dimensionless quantity measures the relative variability of performance, helping assess task difficulty and guide curriculum learning by prioritizing tasks that offer the most learning signal per unit of effort.

For small sample sizes, the paper uses an unbiased estimator of the coefficient of variation:
$$
\nu = \left(1 + \frac{1}{4n} \right) \frac{s}{\bar{x}}
$$
where $s$ is the sample standard deviation and $\bar{x}$ is the sample mean.

---

## Preference Optimization and DPO-like Losses

PAPRIKA employs preference data generated through interaction to fine-tune models using a loss similar to **Direct Preference Optimization (DPO)**:
$$
\mathcal{L}_{\text{DPO}} = -\log \left( \frac{\exp(\beta \cdot (Q_\theta(s, a^+) - Q_\theta(s, a^-)))}{1 + \exp(\beta \cdot (Q_\theta(s, a^+) - Q_\theta(s, a^-)))} \right)
$$
where:
- $a^+$: preferred action (e.g., from successful trajectory)
- $a^-$: dispreferred action (e.g., from failed trajectory)
- $\beta$: temperature parameter controlling the strength of preference
- $Q_\theta(s,a)$: model-estimated Q-value

This formulation allows the model to learn from comparisons without explicit reward modeling, aligning with PAPRIKA’s focus on efficient, targeted exploration.

---

## Summary of Required Mathematical Background

To understand the mathematical framework behind PAPRIKA, one should be comfortable with:

- **Probability Theory**: Understanding distributions, expectations, and Bayesian inference.
- **Statistics**: Familiarity with variance, standard deviation, and estimation techniques like the coefficient of variation.
- **Reinforcement Learning Basics**: Knowledge of MDPs, policies, value functions, and exploration-exploitation strategies.
- **Optimization**: Understanding gradient-based optimization and how it applies to preference-based learning objectives like DPO.
- **Curriculum Learning Techniques**: Awareness of how task difficulty can be modeled and used to structure training.

These mathematical tools enable the construction and evaluation of PAPRIKA, allowing large language models to efficiently learn complex, multi-turn decision-making tasks.

# Algorithms

The paper introduces **PAPRIKA**, a novel fine-tuning framework for large language models (LLMs) that enables strategic exploration and efficient multi-turn decision-making. It leverages curriculum learning, preference optimization, and task difficulty estimation to guide the model toward better performance across diverse tasks.

### 1. Task Sampling Algorithm with UCB-inspired Strategy

PAPRIKA employs an Upper Confidence Bound (UCB)-inspired algorithm to select training tasks dynamically during curriculum learning. This balances exploration of new or difficult tasks with exploitation of already learned ones.

**Algorithm: Task Selection with UCB**
```
Input: Number of arms K (task groups), number of samples C, number of rounds T, policy π
Initialize: s_k = 0, n_k = 0 for all k, Buffer
for each round t = 1 to T do
    Compute θ_k = (s_k / n_k) + sqrt(2 * log(sum(n)) / n_k) for each k
    Select k* = argmax_k θ_k
    Sample τ from group k*
    Sample C trajectories from τ using π and add to Buffer
    Estimate ν̂_π(τ) using coefficient of variation
    Update: s_{k*} += ν̂_π(τ), n_{k*} += 1
end for
Construct dataset D from Buffer and train π
```

This approach ensures that the model is exposed to increasingly informative tasks by prioritizing those with high uncertainty or high potential learning signal.

---

### 2. Preference Data Generation

PAPRIKA generates preference data by encouraging diversity in sampling strategies. It uses rejection sampling over self-generated trajectories to collect both preferred (successful) and dispreferred (unsuccessful) sequences.

- **Diversity-Encouraging Sampling**: Trajectories are generated with different temperature settings to encourage varied reasoning paths.
- **Rejection Sampling**: Only trajectories that demonstrate meaningful differences in decision quality are retained.

---

### 3. Preference Optimization (RPO)

The paper uses **Reward Preference Optimization (RPO)**—a variant of Direct Preference Optimization (DPO)—to fine-tune the LLM based on pairwise comparisons between successful and unsuccessful trajectories.

The RPO loss is defined as:
$$
\mathcal{L}_{\text{RPO}} = -\log \left( \frac{\exp(\beta \cdot (Q_\theta(s, a^+) - Q_\theta(s, a^-)))}{1 + \exp(\beta \cdot (Q_\theta(s, a^+) - Q_\theta(s, a^-)))} \right)
$$
where:
- $a^+$: action from a successful trajectory
- $a^-$: action from a failed trajectory
- $\beta$: temperature parameter
- $Q_\theta$: model’s Q-value estimator

This allows PAPRIKA to learn from relative preferences without requiring explicit reward signals.

---

### 4. Task Difficulty Estimation Using Coefficient of Variation

PAPRIKA estimates task difficulty using the **coefficient of variation (CV)**, which measures the relative variability of task performance:

$$
\nu_\pi(\tau) = \frac{\sqrt{\sigma_\pi^2(\tau)}}{R^\pi(\tau)}
$$

Where:
- $\sigma_\pi^2(\tau)$: variance of reward under policy $\pi$ on task $\tau$
- $R^\pi(\tau)$: mean reward

This metric helps determine how much exploration is needed per task and guides the curriculum learning process.

---

### Summary of Key Components

| Component                      | Purpose                                                  |
| ------------------------------ | -------------------------------------------------------- |
| UCB-inspired Task Sampling     | Balances exploration and exploitation across task groups |
| Diversity-Encouraging Sampling | Generates varied trajectories for preference learning    |
| Rejection Sampling             | Filters out low-quality or redundant trajectories        |
| RPO Loss                       | Learns from pairwise trajectory comparisons              |
| Coefficient of Variation       | Estimates task difficulty and guides curriculum          |


---

# Comparison with Other Methods

Below is a comparison of **PAPRIKA** against other prominent methods used for training LLM agents in sequential decision-making tasks.

| Method                       | Approach                                              | Exploration Strategy   | Preference Learning   | Curriculum Learning           | Scalability | Generalization                     |
| ---------------------------- | ----------------------------------------------------- | ---------------------- | --------------------- | ----------------------------- | ----------- | ---------------------------------- |
| **PAPRIKA (Ours)**           | Preference-based fine-tuning with curriculum learning | UCB-inspired sampling  | Yes (RPO)             | Yes (adaptive task selection) | High        | Strong (zero-shot to unseen tasks) |
| **DPO**                      | Offline preference optimization                       | None (uses fixed data) | Yes                   | No                            | Medium      | Moderate                           |
| **RLHF**                     | Reinforcement Learning from Human Feedback            | Policy gradient        | Requires human labels | Manual                        | Low         | Moderate                           |
| **In-context RL**            | Learns from demonstrations within context             | None                   | No                    | No                            | High        | Weak                               |
| **Online RL (e.g., PPO)**    | On-policy reinforcement learning                      | Entropy regularization | No                    | No                            | Low         | Strong (with enough data)          |
| **Curriculum Learning (CL)** | Progressive task difficulty                           | Uniform sampling       | No                    | Yes (static)                  | Medium      | Moderate                           |
| **Self-Play + Search**       | Tree search with self-play                            | MCTS                   | No                    | No                            | Low         | Strong (specific domains)          |

### Key Observations

- **PAPRIKA** uniquely combines **preference learning**, **curriculum learning**, and **strategic exploration** into a unified pipeline, making it more adaptive and scalable than alternatives.
- Unlike **DPO** or **RLHF**, PAPRIKA does not rely solely on static datasets and actively curates its own training signal through dynamic task selection.
- Compared to **online RL**, PAPRIKA is more resource-efficient and avoids the instability often associated with direct policy updates.
- While **self-play + search** methods excel in structured environments like games, they are less applicable to open-ended language tasks where environment feedback is ambiguous or incomplete.

---
