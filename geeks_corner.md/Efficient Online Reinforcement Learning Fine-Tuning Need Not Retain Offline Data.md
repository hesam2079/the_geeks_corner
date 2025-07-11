	Zhiyuan Zhou
	02 JUL 2025 arXiv

### 1. Abstract
The modern paradigm in machine learning involves pre-training on diverse data, followed by task-specific fine-tuning. In reinforcement learning (RL), this translates to learning via offline RL on a diverse historical dataset, followed by rapid online RL fine-tuning using interaction data. Most RL fine-tuning methods require continued training on offline data for stability and performance. However, this is undesirable because training on large offline datasets is slow and expensive, and may limit the potential for performance improvement due to constraints or pessimism inherent in offline RL. This paper demonstrates that retaining offline data during online fine-tuning is unnecessary if a properly designed online RL approach is used. The proposed method, called Warm-start RL (WSRL), introduces a warmup phase where a small number of rollouts from the pre-trained policy are collected to recalibrate the Q-function and bridge the distribution mismatch between offline data and online interactions. WSRL eliminates the need for retaining offline data while achieving faster learning and higher performance compared to existing algorithms.

---

### 2. Conclusion
This paper explores the possibility of fine-tuning RL agents online without retaining or co-training on any offline datasets. Such a setting is crucial for scalable RL, where offline RL is used to pre-train on diverse data, followed by online RL fine-tuning where retaining offline data is expensive or impractical. It was found that previous offline-to-online RL algorithms fail completely in this setting due to Q-value divergence caused by distribution shift. However, by using an online RL algorithm for fine-tuning and allowing the Q-values to stabilize through a warmup phase, Q-divergence can be prevented. The proposed WSRL method demonstrates that retaining offline data is not necessary for effective fine-tuning, opening new directions for research into no-retention RL paradigms.

---

### 3. Definitions and Literature Review

#### Reinforcement Learning (RL)  
A type of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative reward.

#### Offline Reinforcement Learning (Offline RL)  
A variant of RL where the agent learns exclusively from a fixed dataset of previously collected experiences without further interaction with the environment.

#### Online Reinforcement Learning (Online RL)  
An RL paradigm where the agent continuously interacts with the environment to learn and improve its policy in real-time.

#### Fine-Tuning  
A process where a pre-trained model (e.g., a policy or value function) is adapted to a specific downstream task through additional training on task-specific data.

#### Distribution Shift  
A phenomenon where the data distribution encountered during deployment differs from the one used during training, often leading to performance degradation.

#### Catastrophic Forgetting  
The tendency of neural networks to lose previously learned knowledge when trained on new tasks or data distributions.

#### Q-Function (Q-value)  
In RL, the Q-function $Q(s, a)$ estimates the expected cumulative reward starting from state $s$, taking action $a$, and following a policy thereafter.

#### Policy Gradient  
A class of RL algorithms that directly optimize the policy by computing gradients of the expected return with respect to the policy parameters.

#### Temporal Difference (TD) Learning  
A core RL algorithm that updates value estimates based on the difference between predicted and observed rewards (TD error).

#### Conservative Q-Learning (CQL)  
An offline RL algorithm that regularizes Q-values to prevent overestimation by penalizing out-of-distribution actions.

#### Calibrated Q-Learning (CalQL)  
An extension of CQL that calibrates Q-values for better generalization during online fine-tuning.

#### Warm Start / Warmup Phase  
A technique where the agent starts with a small amount of initial experience to stabilize learning before full-scale training begins.

#### Update-to-Data (UTD) Ratio  
The ratio of network updates per environment step, commonly used in off-policy RL algorithms to accelerate learning.

---

### 4. Mathematics Needed to Understand the Concepts

#### Markov Decision Process (MDP)  
An MDP is defined as $\mathcal{M} = \{S, A, P, r, \gamma, \rho\}$, where:
- $S$: State space.
- $A$: Action space.
- $P(s' | s, a)$: Transition probability function.
- $r(s, a)$: Reward function.
- $\gamma \in [0, 1)$: Discount factor.
- $\rho(s_0)$: Initial state distribution.

#### Objective Function in RL  
The goal in RL is to maximize the expected discounted return:
$$
\eta(\pi) = \mathbb{E}_{s_0 \sim \rho, a_t \sim \pi(\cdot|s_t), s_{t+1} \sim P(\cdot|s_t, a_t)} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right]
$$

#### Bellman Equation  
The optimal Q-function satisfies the Bellman equation:
$$
Q^*(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s, a)} \left[ \max_{a'} Q^*(s', a') \right]
$$

#### Temporal Difference (TD) Error  
For a given transition $(s, a, r, s')$, the TD error is computed as:
$$
\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)
$$

#### Conservative Q-Learning (CQL) Regularizer  
To penalize out-of-distribution actions, CQL adds a regularizer to the standard Bellman loss:
$$
\mathcal{L}_{\text{CQL}} = \alpha \log \left( \frac{1}{|\mathcal{A}|} \sum_{a} \exp(Q(s, a)) \right) - \mathbb{E}_{a \sim \mu}[Q(s, a)]
$$
where $\alpha$ is a regularization coefficient and $\mu$ is the behavior policy.

#### Policy Gradient Theorem  
The gradient of the expected return with respect to the policy parameters $\psi$ is:
$$
\nabla_\psi \eta(\pi_\psi) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\psi(\cdot|s)} \left[ \nabla_\psi \log \pi_\psi(a|s) \cdot G_t \right]
$$
where $G_t$ is the discounted return from time $t$.

#### Soft Actor-Critic (SAC) Objective  
SAC maximizes entropy-regularized expected returns:
$$
\eta(\pi) = \mathbb{E}_{s_t, a_t \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) + \beta \mathcal{H}(\pi(\cdot|s_t)) \right) \right]
$$
where $\mathcal{H}(\pi(\cdot|s_t))$ is the entropy of the policy at state $s_t$, and $\beta$ is the temperature coefficient.

#### Kullback-Leibler (KL) Divergence  
Used to measure how much the fine-tuned policy or Q-function deviates from the pre-trained one:
$$
D_{\text{KL}}(P || Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
$$

These mathematical formulations underpin the concepts discussed in the paper, particularly in analyzing Q-value recalibration, distribution shifts, and catastrophic forgetting during online fine-tuning.

### 5. Main Idea and Novelty

#### **Main Idea**
This paper proposes a novel approach to **fine-tune reinforcement learning (RL) agents online without retaining any offline dataset**, challenging the conventional paradigm where continued training on offline data is considered essential for stability and performance. The authors show that **offline data retention is unnecessary** if the online RL fine-tuning is carefully designed.

The key insight is that **distribution mismatch** between the pre-trained policy's behavior and the early online interactions causes **Q-value divergence** and catastrophic forgetting of the pre-trained knowledge. This issue can be mitigated by introducing a **warmup phase**, where a small number of rollouts from the pre-trained policy are collected to recalibrate the Q-function before full online fine-tuning begins. This warmup helps bridge the distribution gap and stabilizes the transition from offline initialization to online adaptation.

Once this warmup phase is complete, the agent can continue fine-tuning using standard **high update-to-data (UTD) online RL algorithms** (e.g., SAC), without needing to retain or retrain on the original offline dataset.

---

#### **Novelty**
- **No Data Retention Fine-Tuning Paradigm**: This work introduces and analyzes a new RL fine-tuning setting—**no-retention fine-tuning**—where the agent must adapt online without access to the offline dataset after pre-training.
- **Warm Start Reinforcement Learning (WSRL)**: The proposed method uses a **warmup phase** with only a few rollouts from the pre-trained policy to stabilize the Q-function and prevent catastrophic forgetting. This simple but effective idea enables efficient online fine-tuning without offline data.
- **Empirical Insight into Catastrophic Forgetting**: The paper provides an in-depth analysis of how current offline-to-online RL methods suffer from Q-value underestimation and unlearning when offline data is not retained, and shows how WSRL overcomes these issues.
- **State-of-the-Art Performance Without Offline Data**: Despite not using the offline dataset during fine-tuning, WSRL achieves **superior or comparable performance** to existing methods that do retain offline data, while being faster and more scalable.

---

### 6. Algorithms and Methods in This Work

#### **Algorithm Name**:  
**Warm Start Reinforcement Learning (WSRL)**

---

#### **High-Level Overview of WSRL**
1. **Offline Pre-Training**: Use any offline RL algorithm (e.g., CalQL, CQL, IQL) to train a policy $\pi_{\text{pre}}$ and Q-function $Q_{\text{pre}}$ on a static dataset $D_{\text{off}}$.
2. **Warmup Phase**:
   - Collect a small number of transitions (e.g., 5000 steps) using the frozen pre-trained policy $\pi_{\text{pre}}$.
   - These transitions help “recalibrate” the Q-values and bridge the distribution shift between the offline data and the online environment.
3. **Online Fine-Tuning**:
   - After warmup, discard the offline dataset.
   - Continue fine-tuning using a high UTD off-policy RL algorithm (e.g., SAC) with:
     - Ensemble of Q-functions
     - Layer normalization
     - Policy gradient updates
   - No further use of offline data.

---

#### **Key Components of WSRL**
1. **Warmup Phase**:
   - Purpose: Stabilize the Q-function at the start of online training.
   - Mechanism: Run the pre-trained policy for a few episodes to collect in-distribution experience.
   - Outcome: Prevents the "downward spiral" of Q-value underestimation due to out-of-distribution backups.

2. **Online RL Algorithm**:
   - Uses **Soft Actor-Critic (SAC)** with the following enhancements:
     - **Ensemble of 10 Q-networks**: Helps with robustness and uncertainty estimation.
     - **Layer Normalization**: Improves training stability.
     - **High Update-to-Data (UTD) Ratio (4:1)**: Accelerates learning speed.
     - **Actor Delay**: Updates the actor less frequently than the critic (every 4 steps).

3. **Policy and Q-Function Initialization**:
   - Policy $\pi_\psi$ and Q-function $Q_\theta$ are initialized from the pre-trained models.
   - Optimizer states are also inherited from the pre-trained models to ensure smooth continuation of learning.

---

#### **Algorithm Pseudocode (Simplified)**

```python
def WSRL(D_off, A_off, A_on, K=5000):
    # Step 1: Offline Pre-Training
    Q_pre, π_pre = TrainOffline(A_off, D_off)

    # Step 2: Initialize Online RL
    Q = Q_pre
    π = π_pre
    replay_buffer = []

    # Step 3: Warmup Phase
    for step in range(K):
        s, a, s', r = interact(π_pre, env)  # Use frozen pre-trained policy
        replay_buffer.append((s, a, s', r))

    # Step 4: Online Fine-Tuning
    while step < max_steps:
        s, a, s', r = interact(π, env)
        replay_buffer.append((s, a, s', r))

        if step > K:
            # Sample batches from the online buffer
            for _ in range(UTD):
                batch = sample(replay_buffer)
                Q = update_critic(Q, batch)  # Temporal Difference Update
            π = update_actor(π, batch)      # Policy Gradient Update

    return π
```

---

#### **Comparison with Other Methods**
| Method | Uses Offline Data During Fine-Tuning? | Warmup Phase? | Q-Function Stability | Asymptotic Performance |
|-------|----------------------------------------|----------------|----------------------|-------------------------|
| **CQL / IQL / CalQL** | ✅ Yes | ❌ No | ❌ Unstable (diverges without offline data) | ❌ Slower, limited |
| **RLPD / SAC(fast)** | ❌ No | ❌ No | ❌ Poor initial performance | ❌ Worse on complex tasks |
| **JSRL** | ❌ No | ⚠️ Similar idea (roll-in) | ⚠️ Somewhat stable | ⚠️ Moderate performance |
| **WSRL (Ours)** | ❌ No | ✅ Yes (warmup) | ✅ Stable Q-values | ✅ Fast and high performance |

---

#### **Why WSRL Works**
- **Bridges Distribution Shift**: The warmup phase collects data close to the pre-trained policy’s distribution, preventing abrupt divergence in Q-values.
- **Avoids Over-Pessimism**: Unlike conservative offline RL methods, WSRL switches to a non-pessimistic online RL objective after warmup, enabling faster learning.
- **Efficient and Scalable**: Does not require storing or computing on large offline datasets during fine-tuning, making it suitable for real-world deployment.

---

#### **Summary of Key Contributions**
- Introduces **no-retention fine-tuning** as a new paradigm in RL.
- Proposes **WSRL**, a simple yet effective method that eliminates the need for offline data retention.
- Provides theoretical and empirical insights into **catastrophic forgetting**, **Q-value recalibration**, and **distribution shift**.
- Demonstrates superior performance across multiple domains (Antmaze, Kitchen, Adroit, Mujoco, and real-world robotics), outperforming both offline-to-online and purely online baselines.