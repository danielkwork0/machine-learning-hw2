import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from scripts.common import N, SEED, TEST_SIZE, corrupt_labels, poly_svm, rbf_svm, save

DIMS = [2, 5, 10, 15]
HD_NOISE = [0.0, 0.15, 0.25, 0.5]
HD_REPEATS = 3
LIN_CENTER_DIST = 6 * np.sqrt(2)
DIST_DIMS = [2, 3, 5, 10, 15, 30, 50]
N_DIST = 1_000
NOISE_COLORS = dict(zip(HD_NOISE, ["k", "tab:green", "tab:orange", "tab:red"]))


def simulate_linear_nd(n, d, rng):
    mu = np.full(d, LIN_CENTER_DIST / 2 / np.sqrt(d))
    X = np.vstack([rng.normal(0, 1, (n // 2, d)) - mu, rng.normal(0, 1, (n - n // 2, d)) + mu])
    return X, np.r_[-np.ones(n // 2, int), np.ones(n - n // 2, int)]


def simulate_circular_nd(n, d, rng, inner=(0.0, 1.0), outer=(2.0, 3.0), noise=0.25):
    n0, n1 = n // 2, n - n // 2
    r = np.r_[rng.uniform(inner[0] ** d, inner[1] ** d, n0) ** (1 / d),
              rng.uniform(outer[0] ** d, outer[1] ** d, n1) ** (1 / d)] + rng.normal(0, noise, n)
    u = rng.normal(size=(n, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u * r[:, None], np.r_[-np.ones(n0, int), np.ones(n1, int)]


def simulate_circular_noise_features(n, d, rng):
    X, y = simulate_circular_nd(n, 2, rng)
    return np.c_[X, rng.normal(0, 1, (n, d - 2))], y


HD_MODELS = {
    "Linear: linear SVM (C=1)": (simulate_linear_nd, lambda: SVC(kernel="linear", C=1)),
    "Circular: poly d=2 (C=1)": (simulate_circular_nd, poly_svm),
    "Circular: RBF γ=1 (C=10)": (simulate_circular_nd, rbf_svm),
    "Circular: RBF γ=scale (C=10)": (simulate_circular_nd, lambda: rbf_svm(gamma="scale")),
}


def run_dimensions(models, dims, levels, repeats, seed):
    rng = np.random.default_rng(seed)
    out = []
    for name, (simulate, make) in models.items():
        for d in dims:
            X, y = simulate(N, d, rng)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
            for p in levels:
                for rep in range(repeats if p > 0 else 1):
                    m = make().fit(Xtr, corrupt_labels(ytr, p, rng))
                    out.append({"Model": name, "d": d, "p": p, "Test acc (clean)": m.score(Xte, yte), "# SV": len(m.support_)})
    return pd.DataFrame(out)


def plot_performance_vs_dimension(hd_summary):
    fig, axes = plt.subplots(1, len(HD_MODELS), figsize=(22, 5), sharey=True)
    for ax, name in zip(axes, HD_MODELS):
        for p in HD_NOISE:
            s = hd_summary.loc[(name, p), "Test acc (clean)"]
            ax.errorbar(DIMS, s["mean"], yerr=s["std"].fillna(0), color=NOISE_COLORS[p], marker="o", capsize=3, label=f"{p:.0%} flipped")
        ax.axhline(0.5, color="0.6", ls=":")
        ax.set(xlabel="Dimension d", title=name, xticks=DIMS)
        ax.legend(loc="center right", fontsize=9)
    axes[0].set_ylabel("Clean test accuracy")
    plt.tight_layout()
    save(fig, "q6_performance_vs_dimension")
    plt.show()


def plot_zoom(hd_summary):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 5))
    for name, col in zip(HD_MODELS, ["tab:blue", "tab:green", "tab:purple", "tab:pink"]):
        for ax, p in [(ax0, 0.15), (ax1, 0.25)]:
            s = hd_summary.loc[(name, p), "Test acc (clean)"]
            ax.errorbar(DIMS, s["mean"], yerr=s["std"].fillna(0), color=col, marker="o", capsize=3, label=name)
        s = hd_summary.loc[(name, 0.25), "# SV"]
        ax2.errorbar(DIMS, s["mean"], yerr=s["std"].fillna(0), color=col, marker="o", capsize=3, label=name)
    for ax, title in [(ax0, "15% flipped"), (ax1, "25% flipped")]:
        ax.set(xlabel="Dimension d", ylabel="Clean test accuracy", title=f"Clean test accuracy, {title}", xticks=DIMS)
        ax.legend(fontsize=9)
    ax2.set(xlabel="Dimension d", ylabel="Support vectors (of 7,000)", title="Support vectors, 25% flipped", xticks=DIMS)
    ax2.legend(fontsize=9)
    plt.tight_layout()
    save(fig, "q6_zoom_and_support_vectors")
    plt.show()


def plot_distance_concentration():
    rng = np.random.default_rng(SEED + 7)
    ratio = []
    for d in DIST_DIMS:
        D = squareform(pdist(simulate_circular_nd(N_DIST, d, rng)[0]))
        np.fill_diagonal(D, np.nan)
        ratio.append(np.nanmean(np.nanmax(D, 1) / np.nanmin(D, 1)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(DIST_DIMS, ratio, marker="o")
    for d in DIMS:
        ax.axvline(d, color="0.85", lw=0.8, zorder=0)
    ax.set(xlabel="Dimension d", ylabel="Farthest / nearest neighbour distance", yscale="log",
           title=f"Distance concentration ({N_DIST:,} points, circular dataset)")
    save(fig, "q6_distance_concentration")
    plt.show()
    print(dict(zip(DIST_DIMS, np.round(ratio, 1))))


def plot_noise_features():
    nf_models = {k: (simulate_circular_noise_features, HD_MODELS[k][1])
                 for k in ["Circular: poly d=2 (C=1)", "Circular: RBF γ=scale (C=10)"]}
    nf_runs = run_dimensions(nf_models, DIMS, [0.0, 0.25], HD_REPEATS, SEED + 8)
    nf_summary = nf_runs.groupby(["Model", "p", "d"])["Test acc (clean)"].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, col in zip(nf_models, ["tab:green", "tab:pink"]):
        for p, ls in [(0.0, "-"), (0.25, "--")]:
            s = nf_summary.loc[(name, p)]
            ax.errorbar(DIMS, s["mean"], yerr=s["std"].fillna(0), color=col, ls=ls, marker="o", capsize=3, label=f"{name}, {p:.0%} flipped")
    ax.set(xlabel="Dimension d (2 informative + d−2 noise features)", ylabel="Clean test accuracy",
           title="Circle + noise features", xticks=DIMS)
    ax.legend(fontsize=9)
    save(fig, "q6_noise_features")
    plt.show()
    print(nf_summary["mean"].unstack("d").round(3))


def main():
    hd_runs = run_dimensions(HD_MODELS, DIMS, HD_NOISE, HD_REPEATS, SEED + 6)
    hd_summary = hd_runs.groupby(["Model", "p", "d"])[["Test acc (clean)", "# SV"]].agg(["mean", "std"])
    print(hd_summary["Test acc (clean)"]["mean"].unstack("d").round(3))
    plot_performance_vs_dimension(hd_summary)
    plot_zoom(hd_summary)
    plot_distance_concentration()
    plot_noise_features()


if __name__ == "__main__":
    main()
