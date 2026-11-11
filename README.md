# IIS — Information Interaction Structure (core pipeline)

IIS measures **at which order variables depend on one another**. Given a dataset of $N$ variables, it returns an interaction spectrum $a_2, \dots, a_N$: the share of total dependence that lives in pairs, triplets, …, up to all $N$ variables at once, plus a single statistic $Z_{total}$ that says whether there is any structure above sampling noise.

The pipeline handles continuous data (via a Gaussian copula) and discrete data with any number of categories. Every step is validated against exact values computed from known generating distributions.

## The six steps

| Step | What it does |
|---|---|
| 1. Copula transform | Continuous: ranks → normal scores. Discrete: unchanged. |
| 2. Subset entropies | Entropy (bits) of every group of variables: closed-form Gaussian-copula entropy, or plug-in counts. |
| 3. Möbius inversion | Interaction information $\mu(S)$ unique to each group $S$. |
| 4. Architecture mass | $A_k = \sum_{\lvert S\rvert=k} \lvert\mu(S)\rvert$. |
| 5. Null calibration | Subtract the mean mass of $B$ independent reference datasets; report $Z_{total}$. |
| 6. Normalize | $a_k = A^*_k / \sum A^*$. |

Full formulas: [`docs/METHOD.md`](docs/METHOD.md).

## Installation

```bash
git clone https://github.com/<your-username>/iis-core.git
cd iis-core
pip install -r requirements.txt
```

Requires Python ≥ 3.9, NumPy, SciPy and pandas.

## Usage

### Run the validation

```bash
python iis_core.py
```

This runs 100 datasets per benchmark ($N = 5$, $n = 1000$, $B = 100$; about 40 s on a laptop), prints the tables and writes them to `results/`. Options:

```bash
python iis_core.py --reps 20 --n 500 --B 50 --seed 1 --out my_results
python iis_core.py --help
```

### Use on your own data

```python
import numpy as np
from iis_core import iis

X = np.loadtxt("my_data.csv", delimiter=",")       # shape (n samples, N variables)
out = iis(X, kind="continuous", B=100, rng=np.random.default_rng(0))

out["Z"]       # Z_total: read the structure only if > 3
out["a"]       # normalized spectrum a_2 .. a_N
out["A_star"]  # calibrated mass per order (bits)
out["A"]       # raw mass per order (bits)
out["noise"]   # noise floor per order (bits)
```

Use `kind="discrete"` for categorical data. The cost grows as $2^N$ (every subset of variables is evaluated), so the core pipeline is intended for small $N$.

### Notebook

[`notebooks/IIS_core.ipynb`](notebooks/IIS_core.ipynb) contains the same pipeline step by step, with the tables inline.

## Validation results

Benchmarks: Gaussian IID (no structure), Correlated Gaussian (AR(1), $\rho = 0.6$), Shared Ancestry (latent common cause), XOR Epistasis (pure $N$-way synergy). Defaults as above, seed 2026.

**Structure recovery** (mean [95% range] over datasets with $Z_{total} > 3$):

| Benchmark | median $Z_{total}$ | $a_2$ | $a_3$ | $a_4$ | $a_5$ |
|---|---|---|---|---|---|
| Gaussian IID | −0.35 | below noise | below noise | below noise | below noise |
| Correlated Gaussian | 669 | 0.738 [0.695, 0.806] | 0.210 [0.170, 0.233] | 0.047 [0.023, 0.065] | 0.006 [0.002, 0.009] |
| *exact* | | *0.738* | *0.210* | *0.047* | *0.005* |
| Shared Ancestry | 299 | 0.597 [0.576, 0.618] | 0.279 [0.264, 0.291] | 0.105 [0.090, 0.122] | 0.020 [0.013, 0.029] |
| *exact* | | *0.589* | *0.286* | *0.104* | *0.020* |
| XOR Epistasis | 185 | 0.002 [0.000, 0.009] | 0.002 [0.000, 0.010] | 0.001 [0.000, 0.004] | 0.996 [0.981, 1.000] |
| *exact* | | *0* | *0* | *0* | *1* |

**Step 2 accuracy.** Subset entropies are within about 0.02 bits of exact on average at every group size. There is a small downward bias that grows with $\lvert S\rvert$, which is expected of both estimators at finite $n$.

All tables: [`results/`](results/)

| File | Contents |
|---|---|
| `table_step2_entropies.csv` | Estimated vs exact $H(S)$, by group size |
| `table_architecture_mass.csv` | Raw mass, noise floor, calibrated mass and exact mass, by order |
| `table_iis_structure_core.csv` | $Z_{total}$ and normalized spectrum vs exact |

## Repository layout

```
iis-core/
├── iis_core.py            # pipeline + validation script
├── notebooks/
│   └── IIS_core.ipynb     # same pipeline, step by step
├── docs/
│   └── METHOD.md          # formulas and benchmark definitions
├── results/               # tables produced by iis_core.py
├── requirements.txt
└── README.md
```

## Known limitations

* Cost is $O(2^N)$ subsets; practical for roughly $N \le 15$.
* Plug-in discrete entropy needs enough samples per joint pattern; sparse high-cardinality data will be biased downward.
* The Gaussian copula captures dependence through rank correlations; non-monotone continuous dependence is not seen.
* Calibrated mass $A^*_k$ subtracts a noise floor estimated under independence; at orders with real structure this slightly over-corrects (see `table_architecture_mass.csv`).

## Citation

If you use this code, please cite:
GS.Gibbs et. al *Information Interaction Structure (IIS).* GitHub repository, 2026.

## Author

Gregory Gibbs — PhD Candidate, Biomedical Data Science, Meharry Medical College.
