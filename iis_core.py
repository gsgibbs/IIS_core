"""
IIS core pipeline — Information Interaction Structure.

Six steps:
    1. Copula transform        (continuous: ranks -> normal scores; discrete: unchanged)
    2. Subset entropies        (closed-form Gaussian-copula entropy, or plug-in counts)
    3. Moebius inversion       (interaction information mu(S) for every group S)
    4. Architecture mass       (A_k = sum of |mu(S)| over groups of size k)
    5. Null calibration        (subtract mean mass of B independent reference datasets; Z_total)
    6. Normalize               (a_k = A*_k / sum A*)

Running this file validates every step against exact values computed from four
generating distributions and writes the tables to ./results/.

Usage:
    python iis_core.py                 # defaults: N=5, n=1000, B=100, 100 datasets per benchmark
    python iis_core.py --reps 20       # quicker run
    python iis_core.py --help

As a library:
    from iis_core import iis
    out = iis(X, kind="continuous")    # X: (n samples, N variables) array
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def subsets(N: int) -> list[tuple[int, ...]]:
    """All non-empty groups of the N variables, ordered by size."""
    return [S for k in range(1, N + 1) for S in itertools.combinations(range(N), k)]


def gaussian_entropy(C: np.ndarray, S: tuple[int, ...]) -> float:
    """H(S) = 1/2 log2[(2 pi e)^|S| det C_S]  (bits)."""
    return 0.5 * (len(S) * np.log2(2 * np.pi * np.e)
                  + np.linalg.slogdet(C[np.ix_(S, S)])[1] / np.log(2))


# --------------------------------------------------------------------------------------
# The six steps
# --------------------------------------------------------------------------------------

def step1_copula(X: np.ndarray, kind: str) -> np.ndarray:
    """Continuous: ranks -> normal scores. Discrete: unchanged."""
    if kind == "continuous":
        return norm.ppf(rankdata(X, axis=0) / (len(X) + 1))
    return X


def step2_entropies(Z: np.ndarray, kind: str) -> dict[tuple[int, ...], float]:
    """Entropy (bits) of every non-empty group of variables.

    Continuous: closed-form Gaussian-copula entropy from the correlation matrix C of the
                normal scores.
    Discrete:   plug-in entropy -sum p_hat(x) log2 p_hat(x), where p_hat(x) is how often
                each joint pattern occurs. Works for any number of categories.
    """
    H = {}
    if kind == "continuous":
        C = np.corrcoef(Z, rowvar=False)
        for S in subsets(Z.shape[1]):
            H[S] = gaussian_entropy(C, S)
    else:
        # relabel each variable 0..K-1, then give each joint pattern one integer code
        L = np.column_stack([np.unique(c, return_inverse=True)[1] for c in Z.T])
        K = L.max() + 1
        for S in subsets(Z.shape[1]):
            p = np.bincount(L[:, S] @ (K ** np.arange(len(S)))) / len(Z)
            p = p[p > 0]
            H[S] = -(p * np.log2(p)).sum()
    return H


def step3_mobius(H: dict) -> dict:
    """mu(S) = sum over non-empty subsets T of S of (-1)^(|S|-|T|) H(T)."""
    return {S: sum((-1) ** (len(S) - len(T)) * H[T]
                   for j in range(1, len(S) + 1) for T in itertools.combinations(S, j))
            for S in H}


def step4_mass(mu: dict) -> np.ndarray:
    """Architecture mass A_k = sum of |mu(S)| over groups of size k, for k = 2..N."""
    N = max(len(S) for S in mu)
    return np.array([sum(abs(mu[S]) for S in mu if len(S) == k) for k in range(2, N + 1)])


def mass(X: np.ndarray, kind: str) -> np.ndarray:
    """Steps 1-4."""
    return step4_mass(step3_mobius(step2_entropies(step1_copula(X, kind), kind)))


def step5_null(X: np.ndarray, kind: str, A: np.ndarray, B: int = 100,
               rng: np.random.Generator | None = None):
    """Subtract the average mass of B independent reference datasets; report Z_total.

    Continuous reference: independent standard normals of the same size.
    Discrete reference:   independent columns with the same category frequencies.
    """
    rng = np.random.default_rng() if rng is None else rng

    def reference():
        if kind == "continuous":
            return rng.standard_normal(X.shape)
        cols = []
        for c in X.T:
            values, counts = np.unique(c, return_counts=True)
            cols.append(rng.choice(values, size=len(c), p=counts / len(c)))
        return np.column_stack(cols)

    null = np.array([mass(reference(), kind) for _ in range(B)])
    A_star = np.maximum(0, A - null.mean(0))
    Z_total = (A.sum() - null.sum(1).mean()) / null.sum(1).std(ddof=1)
    return A_star, Z_total, null.mean(0)


def step6_normalize(A_star: np.ndarray) -> np.ndarray:
    """a_k = A*_k / sum A*  (NaN if nothing survives calibration)."""
    return A_star / A_star.sum() if A_star.sum() > 0 else np.full(len(A_star), np.nan)


def iis(X: np.ndarray, kind: str, B: int = 100, rng: np.random.Generator | None = None) -> dict:
    """Full pipeline.

    Parameters
    ----------
    X    : array (n samples, N variables)
    kind : "continuous" or "discrete"
    B    : number of independent reference datasets in step 5

    Returns
    -------
    dict with A (raw mass), noise (null mean), A_star (calibrated), a (normalized), Z (Z_total)
    """
    A = mass(X, kind)
    A_star, Z_total, noise = step5_null(X, kind, A, B, rng)
    return {"A": A, "noise": noise, "A_star": A_star, "a": step6_normalize(A_star), "Z": Z_total}


# --------------------------------------------------------------------------------------
# Benchmarks with exact values
# --------------------------------------------------------------------------------------

KIND = {"Gaussian IID": "continuous", "Correlated Gaussian": "continuous",
        "Shared Ancestry": "discrete", "XOR Epistasis": "discrete"}


def ar1(N: int, rho: float) -> np.ndarray:
    return rho ** np.abs(np.subtract.outer(range(N), range(N)))


def sample(m: str, n: int, N: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Draw one dataset of size n from benchmark m."""
    if m == "Gaussian IID":
        return rng.standard_normal((n, N))
    if m == "Correlated Gaussian":
        return rng.multivariate_normal(np.zeros(N), ar1(N, rho), size=n)
    if m == "Shared Ancestry":
        p = np.where(rng.random(n) < 0.5, 0.8, 0.2)[:, None]
        return (rng.random((n, N)) < p).astype(int)
    if m == "XOR Epistasis":
        X = rng.integers(0, 2, (n, N))
        X[:, -1] = X[:, :-1].sum(1) % 2
        return X
    raise ValueError(f"unknown benchmark: {m}")


def exact_H(m: str, N: int, rho: float) -> dict:
    """Exact subset entropies (bits) of the generating distribution."""
    if KIND[m] == "continuous":
        C = np.eye(N) if m == "Gaussian IID" else ar1(N, rho)
        return {S: gaussian_entropy(C, S) for S in subsets(N)}
    states = np.array(list(itertools.product([0, 1], repeat=N)))
    if m == "Shared Ancestry":
        p = sum(0.5 * np.prod(np.where(states == 1, q, 1 - q), axis=1) for q in (0.8, 0.2))
    else:
        ok = states[:, :-1].sum(1) % 2 == states[:, -1]
        p = ok / ok.sum()
    H = {}
    for S in subsets(N):
        q = np.bincount(states[:, S] @ (2 ** np.arange(len(S))), weights=p)
        q = q[q > 0]
        H[S] = -(q * np.log2(q)).sum()
    return H


# --------------------------------------------------------------------------------------
# Validation tables
# --------------------------------------------------------------------------------------

def table_step2_entropies(N, n, rho, reps, rng) -> pd.DataFrame:
    """Error of estimated H(S) vs exact H(S), pooled by group size |S|."""
    rows = []
    for m in KIND:
        Hx = exact_H(m, N, rho)
        err = {k: [] for k in range(1, N + 1)}
        for _ in range(reps):
            Hh = step2_entropies(step1_copula(sample(m, n, N, rho, rng), KIND[m]), KIND[m])
            for S in subsets(N):
                err[len(S)].append(Hh[S] - Hx[S])
        for k, e in err.items():
            e = np.array(e)
            rows.append({"benchmark": m, "|S|": k, "mean error": e.mean(),
                         "sd": e.std(ddof=1), "max |error|": np.abs(e).max()})
    return pd.DataFrame(rows).set_index(["benchmark", "|S|"])


def table_architecture_mass(runs, exact_A, orders) -> pd.DataFrame:
    """Raw mass, noise floor, calibrated mass and exact mass at each order."""
    rows = []
    for m, R in runs.items():
        for i, k in enumerate(orders):
            rows.append({"benchmark": m, "order k": k,
                         "A (raw)": np.mean([r["A"][i] for r in R]),
                         "noise": np.mean([r["noise"][i] for r in R]),
                         "A* (calibrated)": np.mean([r["A_star"][i] for r in R]),
                         "exact A": exact_A[m][i]})
    t = pd.DataFrame(rows).set_index(["benchmark", "order k"])
    t["error"] = t["A* (calibrated)"] - t["exact A"]
    return t


def table_iis_structure(runs, exact_A, orders, z_cut=3.0, min_keep=50) -> pd.DataFrame:
    """Normalized structure a_k (mean [2.5%, 97.5%]) over datasets with Z_total > z_cut."""
    rows = []
    for m, R in runs.items():
        Z = np.array([r["Z"] for r in R])
        keep = [r for r in R if r["Z"] > z_cut]
        ex = step6_normalize(exact_A[m]) if exact_A[m].sum() > 1e-9 else np.full(len(orders), np.nan)
        row = {"benchmark": m, "median Z_total": np.median(Z), f"share Z_total > {z_cut:g}": np.mean(Z > z_cut)}
        for i, k in enumerate(orders):
            if len(keep) >= min_keep:
                v = np.array([r["a"][i] for r in keep])
                row[f"a{k}"] = f"{v.mean():.3f} [{np.quantile(v, .025):.3f}, {np.quantile(v, .975):.3f}]"
            else:
                row[f"a{k}"] = "below noise"
            row[f"exact a{k}"] = ex[i]
        rows.append(row)
    return pd.DataFrame(rows).set_index("benchmark")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Validate the IIS core pipeline on benchmarks with exact values.")
    ap.add_argument("--N", type=int, default=5, help="number of variables (default 5)")
    ap.add_argument("--n", type=int, default=1000, help="samples per dataset (default 1000)")
    ap.add_argument("--rho", type=float, default=0.6, help="AR(1) correlation (default 0.6)")
    ap.add_argument("--B", type=int, default=100, help="reference datasets in step 5 (default 100)")
    ap.add_argument("--reps", type=int, default=100, help="datasets per benchmark (default 100)")
    ap.add_argument("--seed", type=int, default=2026, help="random seed (default 2026)")
    ap.add_argument("--out", type=Path, default=Path("results"), help="output folder (default ./results)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    orders = list(range(2, args.N + 1))
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.precision", 3)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)

    exact_A = {m: step4_mass(step3_mobius(exact_H(m, args.N, args.rho))) for m in KIND}

    print(f"Running {args.reps} datasets per benchmark (N={args.N}, n={args.n}, B={args.B}) ...")
    runs = {m: [iis(sample(m, args.n, args.N, args.rho, rng), KIND[m], args.B, rng)
                for _ in range(args.reps)] for m in KIND}

    t_mass = table_architecture_mass(runs, exact_A, orders)
    t_struct = table_iis_structure(runs, exact_A, orders)
    t_H = table_step2_entropies(args.N, args.n, args.rho, args.reps, rng)

    t_mass.to_csv(args.out / "table_architecture_mass.csv")
    t_struct.to_csv(args.out / "table_iis_structure_core.csv")
    t_H.to_csv(args.out / "table_step2_entropies.csv")

    print("\n== Step 2: estimated vs exact subset entropies (bits) ==\n", t_H)
    print("\n== Architecture mass: estimate vs exact (bits) ==\n", t_mass)
    print("\n== IIS structure and Z_total ==\n", t_struct)
    print(f"\nTables written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
