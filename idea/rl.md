## Efficient Reinforcement Learning with Dual-Model PPO and KL-guided Sampling Allocation

Let \(\pi_{\theta}(a|s)\) denote a policy parameterized by \(\theta\), and let \(\phi\) denote an additional set of **low-rank adaptation parameters** (e.g., similar to LoRA). The combined policy with LoRA applied is denoted as \(\pi_{\theta,\phi}(a|s)\).

We assume access to two simulators:

- A **low-fidelity environment** \(\mathcal{E}_{\text{low}}\), which is fast but approximates the dynamics coarsely.
- A **high-fidelity environment** \(\mathcal{E}_{\text{high}}\), which is accurate but computationally expensive.

### Sampling Budget and Data Allocation

At each training iteration \(t\), we are constrained by a fixed sampling budget \(N\), distributed between the two environments:

\[
N = N_{\text{low}}^{(t)} + N_{\text{high}}^{(t)}, \quad \text{with} \quad N_{\text{low}}^{(t)}, N_{\text{high}}^{(t)} \in \mathbb{N}
\]

We control the allocation based on a divergence signal that reflects the mismatch between the simplified and high-fidelity environments.

### KL-based Divergence Metric

We estimate the KL divergence between the two models' action distributions over trajectories drawn from the simplified environment:

\[
\mathcal{D}_{\text{KL}}^{(t)} = \mathbb{E}_{s \sim \mathcal{E}_{\text{low}}} \left[ D_{\text{KL}}\left( \pi_{\theta}( \cdot | s) \,\|\, \pi_{\theta, \phi}( \cdot | s ) \right) \right]
\]

This KL signal informs whether the simplified environment remains an informative proxy. A **low** \(\mathcal{D}_{\text{KL}}^{(t)}\) implies high agreement between models and justifies more sampling from \(\mathcal{E}_{\text{low}}\), while a **high** divergence suggests that further learning from \(\mathcal{E}_{\text{low}}\) may be misleading.

### Adaptive Sampling Strategy

We define a sampling ratio function \(\rho^{(t)} \in [0, 1]\) that allocates sampling proportion to \(\mathcal{E}_{\text{low}}\):

\[
\rho^{(t)} = f\left( \mathcal{D}_{\text{KL}}^{(t)} \right), \quad \text{where} \quad N_{\text{low}}^{(t)} = \rho^{(t)} N, \quad N_{\text{high}}^{(t)} = (1 - \rho^{(t)}) N
\]

A possible functional form for \(f(\cdot)\) is exponential decay or inverse-sigmoid:

\[
\rho^{(t)} = \exp\left( -\alpha \cdot \mathcal{D}_{\text{KL}}^{(t)} \right), \quad \alpha > 0
\]

---

### Policy Update

The simplified policy \(\pi_{\theta}\) is trained with PPO loss \(\mathcal{L}_{\text{PPO}}^{\text{low}}\) using data from \(\mathcal{E}_{\text{low}}\):

\[
\mathcal{L}_{\text{PPO}}^{\text{low}}(\theta) = \mathbb{E}_{(s, a) \sim \mathcal{E}_{\text{low}}} \left[ \mathcal{L}_{\text{PPO}}\left( \pi_{\theta}(a|s) \right) \right]
\]

The high-fidelity policy \(\pi_{\theta,\phi}\) is trained jointly on \(\theta\) and \(\phi\) with loss \(\mathcal{L}_{\text{PPO}}^{\text{high}}\):

\[
\mathcal{L}_{\text{PPO}}^{\text{high}}(\theta, \phi) = \mathbb{E}_{(s, a) \sim \mathcal{E}_{\text{high}}} \left[ \mathcal{L}_{\text{PPO}}\left( \pi_{\theta, \phi}(a|s) \right) \right]
\]

The total update is then:

- Update \(\theta\) using both \(\mathcal{L}_{\text{PPO}}^{\text{low}}\) and \(\mathcal{L}_{\text{PPO}}^{\text{high}}\)
- Update \(\phi\) using only \(\mathcal{L}_{\text{PPO}}^{\text{high}}\)

---

### Optimization Scheme

At each iteration \(t\), we perform the following steps:

1. Sample \(N_{\text{low}}^{(t)}\) trajectories from \(\mathcal{E}_{\text{low}}\) using \(\pi_{\theta}\)
2. Sample \(N_{\text{high}}^{(t)}\) trajectories from \(\mathcal{E}_{\text{high}}\) using \(\pi_{\theta,\phi}\)
3. Estimate \(\mathcal{D}_{\text{KL}}^{(t)}\)
4. Adjust sampling ratio \(\rho^{(t+1)}\)
5. Update \(\theta\) and \(\phi\) via gradient descent:

\[
\theta \leftarrow \theta - \eta_\theta \nabla_\theta \left( \mathcal{L}_{\text{PPO}}^{\text{low}} + \mathcal{L}_{\text{PPO}}^{\text{high}} \right)
\]
\[
\phi \leftarrow \phi - \eta_\phi \nabla_\phi \mathcal{L}_{\text{PPO}}^{\text{high}}
\]
