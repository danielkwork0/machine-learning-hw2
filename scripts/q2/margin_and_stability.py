import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

from scripts.common import SEED, plot_boundary, save, scatter, split_datasets
from scripts.q2.perceptron import min_distance, perc_predict, perceptron
from scripts.q2.perceptron_vs_svm import C_HARD

N_RUNS = 20
NOISE_SCALES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
N_STRESS = 20_000
C_VALUES = [0.01, 1, C_HARD]


def plot_runs_vs_svm(Xtr, ytr, runs, svm_lin):
    perc_dist = [min_distance(w[1:], w[0], Xtr, ytr) for w in runs]
    svm_dist = min_distance(svm_lin.coef_[0], svm_lin.intercept_[0], Xtr, ytr)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    scatter(ax0, Xtr[:, 0], Xtr[:, 1], (ytr > 0).astype(int), f"{N_RUNS} perceptron runs vs. SVM (train set)")
    xs = np.linspace(Xtr[:, 0].min(), Xtr[:, 0].max(), 2)
    for i, w in enumerate(runs):
        ax0.plot(xs, -(w[0] + w[1] * xs) / w[2], color="darkorange", alpha=0.6, lw=1, label="Perceptron" if i == 0 else None)
    (w1, w2), b = svm_lin.coef_[0], svm_lin.intercept_[0]
    for off, ls in [(0, "-"), (1, "--"), (-1, "--")]:
        ax0.plot(xs, -(b - off + w1 * xs) / w2, "k", ls=ls, lw=2 if off == 0 else 1, label="SVM" if off == 0 else None)
    ax0.set_ylim(Xtr[:, 1].min(), Xtr[:, 1].max())
    ax0.legend(loc="upper left")

    ax1.hist(perc_dist, bins=10, color="darkorange", alpha=0.8, label="Perceptron runs")
    ax1.axvline(svm_dist, color="k", lw=2, label=f"SVM = {svm_dist:.3f}")
    ax1.set(xlabel="Distance from boundary to closest training point", ylabel="Runs",
            title="Margin: perceptron runs vs. SVM")
    ax1.legend()
    plt.tight_layout()
    save(fig, "q2_margin_runs")
    plt.show()
    print(f"Perceptron min distance: mean {np.mean(perc_dist):.3f}, range [{min(perc_dist):.3f}, {max(perc_dist):.3f}]; SVM {svm_dist:.3f}")


def plot_generalisation(runs, svm_lin):
    rng_stress = np.random.default_rng(SEED + 1)
    perc_acc, svm_acc = [], []
    for s in NOISE_SCALES:
        cov = np.array([[1, 0.4], [0.4, 1]]) * s ** 2
        Xs = np.vstack([rng_stress.multivariate_normal([-3, -3], cov, N_STRESS // 2),
                        rng_stress.multivariate_normal([3, 3], cov, N_STRESS // 2)])
        ys = np.r_[-np.ones(N_STRESS // 2), np.ones(N_STRESS // 2)]
        perc_acc.append([np.mean(perc_predict(w, Xs) == ys) for w in runs])
        svm_acc.append(svm_lin.score(Xs, ys))

    perc_acc = np.array(perc_acc)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(NOISE_SCALES, perc_acc.min(1), perc_acc.max(1), color="darkorange", alpha=0.25, label="Perceptron (min–max of runs)")
    ax.plot(NOISE_SCALES, perc_acc.mean(1), color="darkorange", marker="o", label="Perceptron (mean)")
    ax.plot(NOISE_SCALES, svm_acc, color="k", marker="s", label="Linear SVM")
    ax.set(xlabel="Noise scale s (cluster std multiplier)", ylabel="Accuracy on fresh samples",
           title="Generalisation of the learned boundaries under increasing noise")
    ax.legend()
    save(fig, "q2_generalisation")
    plt.show()


def plot_effect_of_c(Xtr, ytr):
    fig, axes = plt.subplots(1, len(C_VALUES), figsize=(18, 6))
    for ax, C in zip(axes, C_VALUES):
        m = SVC(kernel="linear", C=C).fit(Xtr, ytr)
        plot_boundary(ax, Xtr, ytr, m.predict,
                      f"C = {C}: margin 2/||w|| = {2 / np.linalg.norm(m.coef_):.2f}, {len(m.support_)} SVs", svm=m)
    plt.tight_layout()
    save(fig, "q2_effect_of_c")
    plt.show()


def main():
    Xtr, Xte, ytr, yte = split_datasets()["Linear"]
    svm_lin = SVC(kernel="linear", C=C_HARD).fit(Xtr, ytr)
    runs = [perceptron(Xtr, ytr, seed=s)[0] for s in range(N_RUNS)]
    plot_runs_vs_svm(Xtr, ytr, runs, svm_lin)
    plot_generalisation(runs, svm_lin)
    plot_effect_of_c(Xtr, ytr)


if __name__ == "__main__":
    main()
