from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
SEED = 42
N = 10_000
TEST_SIZE = 0.3
COLORS = {0: "white", 1: "royalblue"}
MAX_SV_DRAWN = 200
POLY_DEGREE = 2
# picked by the Q3 grid search (q3/kernel_comparison.py); Q4–Q6 reuse them without re-tuning
BEST_C_POLY = 1
BEST_C_RBF = 10
BEST_GAMMA_RBF = 1


def simulate_linear(n, rng, center=3.0, rho=0.4):
    cov = [[1, rho], [rho, 1]]
    x0 = rng.multivariate_normal([-center, -center], cov, n // 2)
    x1 = rng.multivariate_normal([center, center], cov, n - n // 2)
    X = np.vstack([x0, x1])
    y = np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)]
    return X[:, 0], X[:, 1], y


def simulate_circular(n, rng, inner=(0.0, 1.0), outer=(2.0, 3.0), noise=0.25):
    n0, n1 = n // 2, n - n // 2
    # sqrt of uniform r² gives uniform density over the area, not clumped at the center
    r0 = np.sqrt(rng.uniform(inner[0] ** 2, inner[1] ** 2, n0))
    r1 = np.sqrt(rng.uniform(outer[0] ** 2, outer[1] ** 2, n1))
    r = np.r_[r0, r1] + rng.normal(0, noise, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    y = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    return r * np.cos(theta), r * np.sin(theta), y


def simulate_datasets():
    rng = np.random.default_rng(SEED)
    return {"Linear": simulate_linear(N, rng), "Circular": simulate_circular(N, rng)}


def split_datasets():
    datasets = {}
    for name, (x1, x2, y01) in simulate_datasets().items():
        X, y = np.c_[x1, x2], 2 * y01 - 1
        datasets[name] = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    return datasets


def poly_svm(C=BEST_C_POLY, degree=POLY_DEGREE):
    return SVC(kernel="poly", degree=degree, gamma=1, coef0=1, C=C)


def rbf_svm(gamma=BEST_GAMMA_RBF, C=BEST_C_RBF):
    return SVC(kernel="rbf", gamma=gamma, C=C)


def corrupt_labels(y, p, rng, mode="flip"):
    y = y.copy()
    idx = rng.choice(len(y), int(round(p * len(y))), replace=False)
    y[idx] = -y[idx] if mode == "flip" else rng.permutation(y[idx])
    return y


def scatter(ax, x1, x2, y, title):
    for c in (0, 1):
        m = y == c
        ax.scatter(x1[m], x2[m], s=8, c=COLORS[c], edgecolors="k", linewidths=0.3,
                   alpha=0.6, label=f"Y = {c}")
    ax.set(title=title, xlabel="X1", ylabel="X2", aspect="equal")
    ax.legend(loc="upper left")


def plot_boundary(ax, X, y, predict, title, svm=None):
    pad = 0.5
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300),
                         np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    ax.contourf(xx, yy, predict(grid).reshape(xx.shape), levels=[-2, 0, 2], colors=["0.92", "#c9d6f5"])
    scatter(ax, X[:, 0], X[:, 1], (y > 0).astype(int), title)
    if svm is not None:
        ax.contour(xx, yy, svm.decision_function(grid).reshape(xx.shape), levels=[-1, 0, 1],
                   colors="k", linestyles=["--", "-", "--"], linewidths=[1, 1.8, 1])
        sv = svm.support_vectors_
        # on non-separable data most training points become SVs; circling them all hides the data
        if len(sv) <= MAX_SV_DRAWN:
            ax.scatter(sv[:, 0], sv[:, 1], s=60, facecolors="none", edgecolors="red", linewidths=1.2, label="Support vectors")
            ax.legend(loc="upper left")
    else:
        ax.contour(xx, yy, predict(grid).reshape(xx.shape), levels=[0], colors="k", linewidths=1.8)


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
