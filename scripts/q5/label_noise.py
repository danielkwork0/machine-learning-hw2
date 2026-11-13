import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from scripts.common import SEED, corrupt_labels, plot_boundary, poly_svm, rbf_svm, save, split_datasets

NOISE_LEVELS = [0.0, 0.15, 0.25, 0.35, 0.45, 0.5]
REQUIRED_LEVELS = [0.15, 0.25, 0.5]
N_REPEATS = 5
P_SHOW = 0.25
OVERFIT_GAMMA = 100

NOISE_MODELS = {
    "Linear: linear SVM (C=1)": ("Linear", lambda: SVC(kernel="linear", C=1)),
    "Circular: poly d=2 (C=1)": ("Circular", poly_svm),
    "Circular: RBF γ=1 (C=10)": ("Circular", rbf_svm),
    "Circular: RBF γ=100 (C=10)": ("Circular", lambda: rbf_svm(gamma=OVERFIT_GAMMA)),
}
MODEL_COLORS = dict(zip(NOISE_MODELS, ["tab:blue", "tab:green", "tab:purple", "tab:red"]))


def run_label_noise(datasets, mode, models, levels, repeats, seed):
    rng = np.random.default_rng(seed)
    out = []
    for name, (ds, make) in models.items():
        Xtr, Xte, ytr, yte = datasets[ds]
        for p in levels:
            for rep in range(repeats if p > 0 else 1):
                y_noisy = corrupt_labels(ytr, p, rng, mode)
                m = make().fit(Xtr, y_noisy)
                pred = m.predict(Xte)
                out.append({"Model": name, "p": p, "rep": rep, "Wrong labels": np.mean(y_noisy != ytr),
                            "Test acc (clean)": accuracy_score(yte, pred), "Test F1 (clean)": f1_score(yte, pred),
                            "Train acc (noisy labels)": m.score(Xtr, y_noisy), "# SV": len(m.support_)})
    return pd.DataFrame(out)


def plot_performance(flip_summary):
    pct = np.array(NOISE_LEVELS) * 100
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for name in NOISE_MODELS:
        s = flip_summary.loc[name]
        for ax, col in zip(axes, ["Test acc (clean)", "Train acc (noisy labels)", "# SV"]):
            ax.errorbar(pct, s[(col, "mean")], yerr=s[(col, "std")].fillna(0), color=MODEL_COLORS[name], marker="o", capsize=3, label=name)
    axes[1].plot(pct, 1 - np.array(NOISE_LEVELS), "k--", lw=1.2, label="1 − p (ignores the noise)")
    axes[0].axhline(0.5, color="0.5", ls=":", label="Chance")
    for ax, (ylabel, title) in zip(axes, [("Accuracy", "Clean test accuracy"), ("Accuracy", "Train accuracy on noisy labels"),
                                          ("Support vectors (of 7,000)", "Number of support vectors")]):
        for p in REQUIRED_LEVELS:
            ax.axvline(p * 100, color="0.8", lw=0.8, zorder=0)
        ax.set(xlabel="Flipped training labels (%)", ylabel=ylabel, title=title, xticks=pct)
        ax.legend(fontsize=8.5, loc="lower left" if ax is not axes[2] else "upper left")
    plt.tight_layout()
    save(fig, "q5_performance_vs_flipped")
    plt.show()


def plot_learned_boundaries(datasets):
    Xtr, Xte, ytr, yte = datasets["Circular"]
    y_show = corrupt_labels(ytr, P_SHOW, np.random.default_rng(SEED + 4))
    flipped = y_show != ytr
    circ_models = [n for n in NOISE_MODELS if n.startswith("Circular")]

    fig, axes = plt.subplots(1, len(circ_models), figsize=(18, 6))
    for ax, name in zip(axes, circ_models):
        m = NOISE_MODELS[name][1]().fit(Xtr, y_show)
        plot_boundary(ax, Xtr, y_show, m.predict,
                      f"{name}\ntrain acc (noisy) {m.score(Xtr, y_show):.3f}, clean test acc {m.score(Xte, yte):.3f}")
        ax.scatter(Xtr[flipped, 0], Xtr[flipped, 1], marker="x", c="red", s=10, linewidths=0.6, alpha=0.6, label="Flipped label")
        ax.legend(loc="upper left")
    plt.tight_layout()
    save(fig, "q5_learned_boundaries")
    plt.show()


def plot_shuffle_vs_flip(datasets, flip_runs):
    shuffle_models = {k: NOISE_MODELS[k] for k in ["Circular: poly d=2 (C=1)"]}
    shuffle_runs = run_label_noise(datasets, "shuffle", shuffle_models, NOISE_LEVELS, N_REPEATS, SEED + 5)
    both = pd.concat([flip_runs[flip_runs["Model"].isin(shuffle_models)].assign(Mode="flip"), shuffle_runs.assign(Mode="shuffle")])
    summ = both.groupby(["Mode", "p"])[["Wrong labels", "Test acc (clean)"]].agg(["mean", "std"])
    pct = np.array(NOISE_LEVELS) * 100

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for mode, col in [("flip", "tab:green"), ("shuffle", "tab:orange")]:
        s = summ.loc[mode]
        ax0.errorbar(pct, s[("Test acc (clean)", "mean")], yerr=s[("Test acc (clean)", "std")].fillna(0), color=col, marker="o", capsize=3, label=mode)
        ax1.errorbar(s[("Wrong labels", "mean")] * 100, s[("Test acc (clean)", "mean")], yerr=s[("Test acc (clean)", "std")].fillna(0),
                     color=col, marker="o", capsize=3, label=mode)
    ax0.set(xlabel="Selected training labels (%)", ylabel="Clean test accuracy", title="Poly d=2: vs. selected percentage", xticks=pct)
    ax1.set(xlabel="Training labels actually wrong (%)", title="Poly d=2: vs. actual noise rate")
    for ax in (ax0, ax1):
        ax.legend()
    plt.tight_layout()
    save(fig, "q5_shuffle_vs_flip")
    plt.show()
    print(summ.round(3))


def main():
    datasets = split_datasets()
    flip_runs = run_label_noise(datasets, "flip", NOISE_MODELS, NOISE_LEVELS, N_REPEATS, SEED + 3)
    flip_summary = flip_runs.groupby(["Model", "p"]).agg(["mean", "std"]).drop(columns="rep")
    print(flip_summary.loc[(slice(None), [0.0] + REQUIRED_LEVELS), ["Test acc (clean)", "Train acc (noisy labels)", "# SV"]].round(3))
    plot_performance(flip_summary)
    plot_learned_boundaries(datasets)
    plot_shuffle_vs_flip(datasets, flip_runs)


if __name__ == "__main__":
    main()
