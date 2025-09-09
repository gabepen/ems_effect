import numpy as np
import pandas as pd
import argparse
import os
import glob
from typing import Tuple, List, Dict, Any
from scipy.optimize import minimize
from scipy.stats import chi2

# --- LOAD YOUR DATA -------------------------------------------------
# Replace with your data load. DataFrame must have columns 'x' and 'n'
# Example: df = pd.read_csv("ems_counts.tsv", sep="\t")
parser = argparse.ArgumentParser()
parser.add_argument('inputs', nargs='+', help='One or more input TSV files or directories (each file with columns: site, x, n or x, n)')
parser.add_argument('--G', type=float, default=3.0, help='Number of generations (default: 3)')
parser.add_argument('--summary-out', help='Path to write a per-input summary TSV (optional).')
parser.add_argument('--plot-out', help='Path to write a read-vs-trial scatter plot comparing mu_hat (optional).')
parser.add_argument('--label-points', action='store_true', help='Annotate points in the plot with sample names (optional).')
args = parser.parse_args()

G = float(args.G)


def load_counts(path: str) -> Tuple[np.ndarray, np.ndarray]:
    # First try tab-delimited; if it collapses to one column, retry with generic whitespace
    _df = pd.read_csv(path, sep="\t")
    if _df.shape[1] == 1 and _df.columns.size == 1:
        df = pd.read_csv(path, sep=r"\s+", engine='python')
    else:
        df = _df

    # Normalize/verify columns for x and n
    cols_lower = {c.lower().strip(): c for c in df.columns}
    if 'x' in cols_lower and 'n' in cols_lower:
        x_col = cols_lower['x']
        n_col = cols_lower['n']
    else:
        # If exactly two columns, assume they are x and n
        if df.shape[1] == 2:
            df.columns = ['x', 'n']
            x_col, n_col = 'x', 'n'
        # If three columns and one looks like 'site', try to map remaining two to x,n
        elif df.shape[1] >= 3 and ('site' in cols_lower or 'locus' in cols_lower):
            if 'x' in cols_lower and 'n' in cols_lower:
                x_col = cols_lower['x']
                n_col = cols_lower['n']
            else:
                raise ValueError(f"Input file must contain 'x' and 'n' columns. Found columns: {list(df.columns)}")
        else:
            raise ValueError(f"Input file must contain 'x' and 'n' columns (optionally with 'site'). Found columns: {list(df.columns)}")

    x = pd.to_numeric(df[x_col], errors='coerce').to_numpy(dtype=float)
    n = pd.to_numeric(df[n_col], errors='coerce').to_numpy(dtype=float)

    # Drop any rows where x or n is NaN after coercion
    mask = ~np.isnan(x) & ~np.isnan(n)
    x = x[mask]
    n = n[mask]
    return x, n


# --- LIKELIHOOD FUNCTIONS -------------------------------------------

def pi_of_mu(mu: float) -> float:
    return 1.0 - np.exp(-mu * G)


def neg_loglik(mu_arr: np.ndarray, x: np.ndarray, n: np.ndarray) -> float:
    mu = mu_arr[0] if isinstance(mu_arr, (list, np.ndarray)) else float(mu_arr)
    if mu <= 0:
        return 1e30
    pi = pi_of_mu(mu)
    pi = np.clip(pi, 1e-12, 1 - 1e-12)
    ll = np.sum(x * np.log(pi) + (n - x) * np.log(1 - pi))
    return -ll


def observed_info(mu: float, x: np.ndarray, n: np.ndarray) -> float:
    pi = pi_of_mu(mu)
    q = np.exp(-mu * G)
    dpi = G * q
    ddpi = -G * dpi  # -G^2 q
    part1 = -np.sum(x / (pi**2) * (dpi**2))
    part2 = -np.sum((n - x) / ((1 - pi)**2) * (dpi**2))
    part3 = np.sum((x / pi - (n - x) / (1 - pi)) * ddpi)
    d2ll = part1 + part2 + part3
    return -d2ll


def fit_mu(x: np.ndarray, n: np.ndarray) -> Tuple[float, float, float, float]:
    init_mu = max(1e-8, np.sum(x) / (np.sum(n) * G))
    res = minimize(lambda m: neg_loglik(m, x, n), x0=np.array([init_mu]), bounds=[(1e-15, None)], method='L-BFGS-B')
    mu_hat = float(res.x[0])

    I = observed_info(mu_hat, x, n)
    se_mu = np.sqrt(1.0 / I) if I > 0 else np.nan

    ll_hat = -neg_loglik(np.array([mu_hat]), x, n)
    alpha = 0.05
    crit = chi2.ppf(1 - alpha, df=1)
    mus = np.linspace(max(1e-12, mu_hat * 0.01), mu_hat * 10 + 1e-9, 2000)
    lls = np.array([-neg_loglik(np.array([m]), x, n) for m in mus])
    delta = 2 * (ll_hat - lls)
    accept = (delta <= crit)
    if accept.any():
        mu_lo = mus[accept].min()
        mu_hi = mus[accept].max()
    else:
        mu_lo, mu_hi = np.nan, np.nan

    return mu_hat, se_mu, mu_lo, mu_hi


# --- PROCESS INPUTS --------------------------------------------------

def collect_input_files(items: list) -> list:
    files = []
    for item in items:
        if os.path.isdir(item):
            files.extend(sorted(glob.glob(os.path.join(item, '*.tsv'))))
        else:
            files.append(item)
    # deduplicate while preserving order
    seen = set()
    dedup = []
    for f in files:
        if f not in seen:
            seen.add(f)
            dedup.append(f)
    return dedup


def parse_sample_and_approach(path: str) -> Tuple[str, str]:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    approach = 'read' if '.read' in name else ('trial' if '.trial' in name else 'unknown')
    name = name.replace('.read', '').replace('.trial', '')
    return name, approach


input_files = collect_input_files(args.inputs)

results: List[Dict[str, Any]] = []

for path in input_files:
    try:
        x, n = load_counts(path)
        mu_hat, se_mu, mu_lo, mu_hi = fit_mu(x, n)
        sample, approach = parse_sample_and_approach(path)
        sum_x = float(np.sum(x))
        sum_n = float(np.sum(n))
        print(f"{os.path.basename(path)}\tmu_hat={mu_hat:.3e}\tSE={se_mu:.3e}\t95%CI=[{mu_lo:.3e},{mu_hi:.3e}]\tG={G}")
        results.append({
            'file': path,
            'sample': sample,
            'approach': approach,
            'mu_hat': mu_hat,
            'se': se_mu,
            'mu_lo': mu_lo,
            'mu_hi': mu_hi,
            'G': G,
            'sum_x': sum_x,
            'sum_n': sum_n,
        })
    except Exception as e:
        print(f"{os.path.basename(path)}\tERROR: {e}")

# Write summary if requested
if args.summary_out:
    os.makedirs(os.path.dirname(args.summary_out) or '.', exist_ok=True)
    cols = ['sample', 'approach', 'mu_hat', 'se', 'mu_lo', 'mu_hi', 'G', 'sum_x', 'sum_n', 'file']
    pd.DataFrame(results)[cols].to_csv(args.summary_out, sep='\t', index=False)

# Generate plot if requested
if args.plot_out:
    # Only plot samples that have both trial and read
    df = pd.DataFrame(results)
    if not df.empty:
        pivot = df.pivot_table(index='sample', columns='approach', values='mu_hat', aggfunc='first')
        pivot = pivot.dropna(subset=['trial', 'read'], how='any') if 'trial' in pivot.columns and 'read' in pivot.columns else pd.DataFrame()
        if not pivot.empty:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(pivot['trial'], pivot['read'], s=40, alpha=0.8)
            # Diagonal
            lims = [min(pivot.min().min(), pivot.min().min()), max(pivot.max().max(), pivot.max().max())]
            ax.plot(lims, lims, 'k--', linewidth=1)
            ax.set_xlabel('mu_hat (trial)')
            ax.set_ylabel('mu_hat (read)')
            ax.set_title('Read vs Trial mutation rate estimates')
            if args.label_points:
                for sample, row in pivot.iterrows():
                    ax.annotate(sample, (row['trial'], row['read']), fontsize=8, alpha=0.8)
            os.makedirs(os.path.dirname(args.plot_out) or '.', exist_ok=True)
            fig.tight_layout()
            fig.savefig(args.plot_out, dpi=200)
            plt.close(fig)
        else:
            # Create an empty placeholder plot if no pairs found
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'No samples with both trial and read inputs.', ha='center', va='center')
            ax.axis('off')
            os.makedirs(os.path.dirname(args.plot_out) or '.', exist_ok=True)
            fig.tight_layout()
            fig.savefig(args.plot_out, dpi=200)
            plt.close(fig)