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


### Notebook

[`notebooks/IIS_core.ipynb`](notebooks/IIS_core.ipynb) contains the same pipeline step by step, with the tables inline.



## Known limitations

* Cost is $O(2^N)$ subsets; practical for roughly $N \le 15$.
* Plug-in discrete entropy needs enough samples per joint pattern; sparse high-cardinality data will be biased downward.
* The Gaussian copula captures dependence through rank correlations; non-monotone continuous dependence is not seen.
* Calibrated mass $A^*_k$ subtracts a noise floor estimated under independence; at orders with real structure this slightly over-corrects (see `table_architecture_mass.csv`).

## Citation

If you use this code, please cite:
G.S.Gibbs et. al *Information Interaction Structure (IIS).* GitHub repository, 2026.

## Author

Gregory Gibbs — PhD Candidate, Biomedical Data Science, Meharry Medical College.
