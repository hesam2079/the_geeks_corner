	Swetha Ganesh 
	Washim Uddin mondal
	Vaneet Aggarwall
	42nd international conference on machine learning, Vancouver, Canada, 2025
### Abstract

This paper presents a Multi-level Monte Carlo-based Natural Actor-Critic (MLMC-NAC) algorithm for addressing average-reward reinforcement learning challenges. The proposed method achieves an order-optimal global convergence rate of $\tilde{O}(1/\sqrt{T})$, where $T$ is the horizon length, significantly surpassing state-of-the-art results in this domain, particularly for actor-critic approaches with general policy parametrization. The convergence rate does not scale with the size of the state space, making it applicable to infinite state spaces. The key contributions include a refined analysis that leads to sharper results and the elimination of reliance on knowledge of mixing and hitting times, which are impractical in many settings.

### Conclusion

The work introduces the MLMC-NAC algorithm, achieving a global convergence rate of $\tilde{O}(1/\sqrt{T})$ for average-reward Markov Decision Processes (MDPs) using an Actor-Critic approach. This result is significant as it addresses limitations of existing methods, such as suboptimal convergence rates and dependencies on impractical assumptions about mixing and hitting times. The algorithm's practicality is enhanced by its ability to function without precise knowledge of these times, thereby broadening its applicability to large or infinite state spaces.

### Main Algorithm

**Algorithm 1: Multi-level Monte Carlo-based Natural Actor-Critic (MLMC-NAC)**

1. **Input**: Initial parameters $\theta_0$, $\{\omega_{H_k}\}$, and $\{\xi_0^k\}$, policy update stepsize $\alpha$, parameters for NPG update $\gamma$, parameters for critic update $\beta, c_\beta$, initial state $s_0 \sim \rho$, outer loop size $K$, inner loop size $H$, $T_{\text{max}}$
2. **Initialization**: $T \leftarrow 0$
3. **For** $k = 0, 1, \ldots, K - 1$:
   - **For** $h = 0, 1, \ldots, H - 1$:
     - Obtain estimates of average reward and critic
     - Update parameters based on the Multi-Level Monte Carlo estimator
   - **Update Policy**: $\theta_{k+1} = \theta_k + \alpha \omega_k$

### Easy Math Summary

This paper analyzes the convergence properties of the Multi-level Monte Carlo-based Natural Actor-Critic (MLMC-NAC) algorithm in the context of average reward reinforcement learning. The main mathematical contributions are:

- **Convergence Rate**: The algorithm achieves a global convergence rate of $\tilde{O}(1/\sqrt{T})$, which is optimal in terms of the horizon length $T$. This is formalized in Theorem 1, where the expected difference between the optimal reward $J^*$ and the average reward obtained by the algorithm over $K$ iterations is bounded by:
  
  $$
  J^* - \frac{1}{K} \sum_{k=0}^{K-1} \mathbb{E}[J(\theta_k)] \leq O\left(\sqrt{\epsilon_{\text{app}}} + \sqrt{\epsilon_{\text{bias}}}\right) + \tilde{O}\left(t_{\text{mix}}^3 T^{-1/2}\right)
  $$

  Here, $\epsilon_{\text{app}}$ represents approximation error, $\epsilon_{\text{bias}}$ represents bias in gradient estimation, and $t_{\text{mix}}$ is the mixing time of the MDP.

- **Key Assumptions**: 
  - The average reward objective $J$ is $L$-smooth.
  - The Fisher information matrix has eigenvalues bounded from below.
  - The algorithm uses unbiased gradient estimates via the MLMC technique.

- **Policy Update Analysis**: The policy update rule $\theta_{k+1} = \theta_k + \alpha \omega_k$ ensures that the algorithm converges globally, with the bound on the expected gradient norm given by:

  $$
  \frac{1}{K} \sum_{k=0}^{K-1} \mathbb{E} \|\nabla_\theta J(\theta_k)\|^2 \leq \frac{32 L G_1^4}{\mu^4 K} + \frac{(2 G_1^4 / \mu^2 + 1)}{K} \sum_{k=0}^{K-1} \mathbb{E} \|\omega_k - \omega_k^*\|^2
  $$

  Where $\omega_k^*$ is the natural policy gradient direction, $G_1$ and $\mu$ are constants related to the problem's geometry.

This summary encapsulates the theoretical guarantees and practical advantages of the MLMC-NAC algorithm in handling complex reinforcement learning tasks under average reward criteria.