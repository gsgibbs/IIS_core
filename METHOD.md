# Method

The IIS pipeline turns a dataset of $N$ variables into an **interaction spectrum**: how much statistical dependence lives at each order $k = 2, \dots, N$ (pairs, triplets, …, all $N$ together).

## Step 1 — Copula transform

* **Continuous:** each column is replaced by its normal scores, $z = \Phi^{-1}\!\big(\mathrm{rank}/(n+1)\big)$. This removes the marginal distributions and keeps only the dependence (the copula).
* **Discrete:** values are left unchanged.

## Step 2 — Subset entropies

For every non-empty group of variables $S$, estimate its entropy in bits.

* **Continuous** (Gaussian-copula, closed form), with $C$ the correlation matrix of the normal scores:

$$H(S) = \tfrac12 \log_2\!\big[(2\pi e)^{|S|}\det C_S\big]$$

* **Discrete** (plug-in), with $\hat p(x)$ the fraction of samples showing the joint pattern $x$:

$$H(S) = -\sum_x \hat p(x)\log_2 \hat p(x)$$

Any number of categories is supported: each variable is relabelled $0..K-1$ and each joint pattern is encoded as one integer.

The closed-form continuous estimator replaces the k-nearest-neighbour estimator used in earlier versions and removes its estimator error. Both estimators have a small downward bias that grows with $|S|$ (see `results/table_step2_entropies.csv`).

## Step 3 — Möbius inversion

$$\mu(S) = \sum_{\emptyset \ne T \subseteq S} (-1)^{|S|-|T|}\, H(T)$$

$\mu(S)$ is the interaction information unique to group $S$. Its sign depends on the group size and on whether redundancy or synergy dominates, so only $|\mu(S)|$ is used below.

## Step 4 — Architecture mass

$$A_k = \sum_{|S| = k} |\mu(S)|, \qquad k = 2, \dots, N$$

## Step 5 — Null calibration

Generate $B$ independent reference datasets of the same size:

* continuous: independent standard normals;
* discrete: independent columns that keep each variable's category frequencies.

Run steps 1–4 on each to get the noise floor $\bar A^{\text{null}}_k$, then

$$A^*_k = \max\!\big(0,\; A_k - \bar A^{\text{null}}_k\big)$$

$$Z_{total} = \frac{\sum_k A_k - \operatorname{mean}_b \sum_k A^{(b)}_k}{\operatorname{sd}_b \sum_k A^{(b)}_k}$$

$Z_{total}$ measures how far the data sit above sampling noise. The structure is only interpreted when $Z_{total} > 3$.

## Step 6 — Normalize

$$a_k = A^*_k \Big/ \sum_j A^*_j$$

## Validation benchmarks

| Benchmark | Type | Generating distribution | Expected structure |
|---|---|---|---|
| Gaussian IID | continuous | $N$ independent standard normals | none (below noise) |
| Correlated Gaussian | continuous | AR(1), $\rho = 0.6$ | mostly pairwise, decaying with $k$ |
| Shared Ancestry | discrete | latent coin picks $p \in \{0.8, 0.2\}$; all variables Bernoulli($p$) | spread across orders |
| XOR Epistasis | discrete | $X_N = X_1 \oplus \dots \oplus X_{N-1}$ | all mass at $k = N$ |

Exact entropies are computed from each generating distribution (closed form for the Gaussians, enumeration of all $2^N$ states for the binary models), so every step can be checked against ground truth.
