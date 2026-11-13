import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

from scripts.common import save, scatter, split_datasets
from scripts.q2.margin_and_stability import N_RUNS
from scripts.q2.perceptron import perc_predict, perceptron
from scripts.q2.perceptron_vs_svm import C_SOFT


def main():
    Xtr_c, Xte_c, ytr_c, yte_c = split_datasets()["Circular"]
    svm_circ = SVC(kernel="linear", C=C_SOFT).fit(Xtr_c, ytr_c)
    runs_c = [perceptron(Xtr_c, ytr_c, seed=s)[0] for s in range(N_RUNS)]
    runs_c_acc = [np.mean(perc_predict(w, Xte_c) == yte_c) for w in runs_c]

    history = []
    perceptron(Xtr_c, ytr_c, history=history)
    epoch_acc = [np.mean(perc_predict(w, Xtr_c) == ytr_c) for w in history]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.2]})
    scatter(ax0, Xtr_c[:, 0], Xtr_c[:, 1], (ytr_c > 0).astype(int), f"{N_RUNS} perceptron runs vs. SVM (train set)")
    xs = np.linspace(-4, 4, 2)
    xx, yy = np.meshgrid(xs, xs)
    for w in runs_c:
        # some runs end with a near-vertical line, so draw w·x = 0 as a contour instead of solving for x2
        ax0.contour(xx, yy, w[0] + w[1] * xx + w[2] * yy, levels=[0], colors="darkorange", alpha=0.6, linewidths=1)
    ax0.plot([], [], color="darkorange", lw=1, label="Perceptron")
    (w1, w2), b = svm_circ.coef_[0], svm_circ.intercept_[0]
    ax0.contour(xx, yy, b + w1 * xx + w2 * yy, levels=[0], colors="k", linewidths=2)
    ax0.plot([], [], "k", lw=2, label="SVM")
    ax0.set(xlim=(-4, 4), ylim=(-4, 4))
    ax0.legend(loc="upper left")

    ax1.plot(range(1, len(epoch_acc) + 1), epoch_acc, color="darkorange", marker="o", ms=3, label="Perceptron (one run)")
    ax1.axhline(svm_circ.score(Xtr_c, ytr_c), color="k", lw=2, label="Linear SVM")
    ax1.axhline(0.5, color="0.5", ls=":", label="Chance")
    ax1.set(xlabel="Epoch", ylabel="Train accuracy", title="Perceptron accuracy per epoch (circular)", ylim=(0.3, 0.8))
    ax1.legend(loc="lower right")
    plt.tight_layout()
    save(fig, "q2_circular_perceptron")
    plt.show()
    print(f"Perceptron test acc over {N_RUNS} runs: mean {np.mean(runs_c_acc):.3f}, range [{min(runs_c_acc):.3f}, {max(runs_c_acc):.3f}]; "
          f"SVM {svm_circ.score(Xte_c, yte_c):.3f}")
    print(f"Per-epoch train acc (one run): range [{min(epoch_acc):.3f}, {max(epoch_acc):.3f}]")


if __name__ == "__main__":
    main()
