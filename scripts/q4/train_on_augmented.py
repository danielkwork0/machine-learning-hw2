import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score
from sklearn.svm import SVC

from scripts.common import BEST_C_RBF, SEED, plot_boundary, poly_svm, save, split_datasets
from scripts.q4.data_augmentation import (GAMMA_BONUS, K_AUG, K_BONUS, K_SWEEP, N_AUG, SD_LEVELS, SD_SWEEP, THRESHOLD,
                                          augment, reference_rule, threshold_crossing)


def fit_augmentations(Xtr, ytr, Xte, yte, rng):
    aug, aug_metrics = {}, {}
    for name, sd in SD_LEVELS.items():
        Xa, ya = augment(Xtr, ytr, sd, K_AUG, N_AUG, rng)
        train_acc, scores = np.empty(K_AUG), np.empty((K_AUG, len(Xte)))
        for i, (X, y) in enumerate(zip(Xa, ya)):
            m = poly_svm().fit(X, y)
            train_acc[i] = m.score(X, y)
            scores[i] = m.decision_function(Xte)
        pred = np.where(scores > 0, 1, -1)
        tp = ((pred == 1) & (yte == 1)).sum(1)
        fp = ((pred == 1) & (yte == -1)).sum(1)
        fn = ((pred == -1) & (yte == 1)).sum(1)
        tn = ((pred == -1) & (yte == -1)).sum(1)
        aug[name] = (Xa[0], ya[0], poly_svm().fit(Xa[0], ya[0]))
        aug_metrics[name] = {
            "train_acc": train_acc,
            "acc": (tp + tn) / len(yte),
            "f1": 2 * tp / (2 * tp + fp + fn),
            "auc": np.array([roc_auc_score(yte, s) for s in scores]),
            "cm": np.array([[tn.mean(), fp.mean()], [fn.mean(), tp.mean()]]),
        }
    return aug, aug_metrics


def results_table(full, Xtr, ytr, Xte, yte, aug_metrics):
    rows = [{"Training set": f"Full train set ({len(Xtr):,} points)", "Train acc": f"{full.score(Xtr, ytr):.4f}",
             "Test acc": f"{full.score(Xte, yte):.4f}"}]
    for name, m in aug_metrics.items():
        lo, hi = np.percentile(m["acc"], [2.5, 97.5])
        rows.append({"Training set": f"{name}, k = {K_AUG:,}",
                     "Train acc": f"{m['train_acc'].mean():.4f} ± {m['train_acc'].std():.4f}",
                     "Test acc": f"{m['acc'].mean():.4f} ± {m['acc'].std():.4f}", "Test acc 95% range": f"[{lo:.3f}, {hi:.3f}]",
                     "Train − test gap": f"{(m['train_acc'] - m['acc']).mean():+.4f}",
                     "F1": f"{m['f1'].mean():.4f} ± {m['f1'].std():.4f}", "AUC": f"{m['auc'].mean():.4f} ± {m['auc'].std():.4f}"})
    return pd.DataFrame(rows).set_index("Training set").fillna("–")


def plot_augmentations(aug, Xte, yte):
    fig, axes = plt.subplots(2, len(SD_LEVELS), figsize=(18, 12))
    for (ax_tr, ax_te), (name, (X0, y0, m)) in zip(axes.T, aug.items()):
        plot_boundary(ax_tr, X0, y0, m.predict, f"{name}: one augmented train set\ntrain acc {m.score(X0, y0):.3f}, {len(m.support_)} SVs", svm=m)
        plot_boundary(ax_te, Xte, yte, m.predict, f"Same model on the clean test set, acc {m.score(Xte, yte):.3f}")
        wrong = m.predict(Xte) != yte
        ax_te.scatter(Xte[wrong, 0], Xte[wrong, 1], marker="x", c="red", s=40, linewidths=1.5, label=f"Misclassified ({wrong.sum()})")
        ax_te.legend(loc="upper left")
    plt.tight_layout()
    save(fig, "q4b_augmentations")
    plt.show()


def plot_distributions(full, Xte, yte, aug_metrics):
    fig, axes = plt.subplots(1, 4, figsize=(24, 4.5))
    colors = ["0.4", "tab:green", "tab:red"]
    for ax, (key, label) in zip(axes, [("train_acc", "Train accuracy (own augmented set)"), ("acc", "Test accuracy"),
                                       ("f1", "Test F1-score"), ("auc", "Test AUC")]):
        for (name, m), col in zip(aug_metrics.items(), colors):
            ax.hist(m[key], bins=40, color=col, alpha=0.6, label=f"{name}: {m[key].mean():.3f}")
        ax.set(xlabel=label, ylabel="Trained models", title=f"{label}\nover {K_AUG:,} models")
        ax.legend(fontsize=9)
    axes[1].axvline(full.score(Xte, yte), color="k", ls="--", lw=1.5)
    axes[1].text(full.score(Xte, yte), axes[1].get_ylim()[1] * 0.95, " full-train-set model", ha="right", va="top", rotation=90)
    plt.tight_layout()
    save(fig, "q4b_metric_distributions")
    plt.show()


def plot_confusion_matrices(aug_metrics, n_test):
    fig, axes = plt.subplots(1, len(SD_LEVELS), figsize=(16, 4.5))
    for ax, (name, m) in zip(axes, aug_metrics.items()):
        ConfusionMatrixDisplay(m["cm"], display_labels=["Y = 0", "Y = 1"]).plot(ax=ax, cmap="Blues", colorbar=False, values_format=".1f")
        ax.set_title(f"Trained on {name}\nmean test confusion matrix (of {n_test:,} points)")
    plt.tight_layout()
    save(fig, "q4b_confusion_matrices")
    plt.show()


def plot_noise_sweep(Xtr, ytr, Xte, yte, rng):
    sweep = {"train": [], "test": []}
    for sd in SD_SWEEP:
        Xa, ya = augment(Xtr, ytr, sd, K_SWEEP, N_AUG, rng)
        models = [poly_svm().fit(X, y) for X, y in zip(Xa, ya)]
        sweep["train"].append([m.score(X, y) for m, X, y in zip(models, Xa, ya)])
        sweep["test"].append([m.score(Xte, yte) for m in models])
    sweep = {k: np.array(v) for k, v in sweep.items()}

    fig, ax = plt.subplots(figsize=(9, 5))
    for key, col, label in [("train", "tab:blue", "Train accuracy (own augmented set)"), ("test", "tab:green", "Clean test accuracy")]:
        m, s = sweep[key].mean(1), sweep[key].std(1)
        ax.plot(SD_SWEEP, m, color=col, marker="o", label=label)
        ax.fill_between(SD_SWEEP, m - s, m + s, color=col, alpha=0.15)
    ax.axhline((reference_rule(Xte) == yte).mean(), color="k", ls="--", label="Reference rule on test set")
    for sd, col in [(0.1, "tab:green"), (0.5, "tab:red")]:
        ax.axvline(sd, color=col, ls=":", alpha=0.7)
    ax.set(xlabel="Noise SD σ added to the training points", ylabel=f"Accuracy (mean ± SD over {K_SWEEP:,} models)",
           title="Poly SVM trained on augmented sets of increasing noise")
    ax.legend()
    save(fig, "q4b_noise_sweep")
    plt.show()
    print("σ:      ", SD_SWEEP.round(1))
    print("train:  ", sweep["train"].mean(1).round(4))
    print("test:   ", sweep["test"].mean(1).round(4))
    print("gap:    ", (sweep["train"].mean(1) - sweep["test"].mean(1)).round(4))


def bonus_threshold(Xte, yte, Xtr, ytr, rng):
    sets = {name: augment(Xtr, ytr, sd, K_BONUS, N_AUG, rng) for name, sd in SD_LEVELS.items()}
    train_acc, test_acc = ({name: [] for name in sets} for _ in range(2))
    for g in GAMMA_BONUS:
        for name, (Xa, ya) in sets.items():
            models = [SVC(kernel="rbf", gamma=g, C=BEST_C_RBF).fit(X, y) for X, y in zip(Xa, ya)]
            train_acc[name].append(np.mean([m.score(X, y) for m, X, y in zip(models, Xa, ya)]))
            test_acc[name].append(np.mean([m.score(Xte, yte) for m in models]))
    train_acc, test_acc = ({k: np.array(v) for k, v in d.items()} for d in (train_acc, test_acc))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (name, a), col in zip(test_acc.items(), ["0.4", "tab:green", "tab:red"]):
        ax.plot(GAMMA_BONUS, train_acc[name], color=col, ls=":", alpha=0.8)
        ax.plot(GAMMA_BONUS, a, color=col, marker="o", ms=4, label=name)
        g = threshold_crossing(a)
        ax.axvline(g, color=col, ls=":", alpha=0.8)
        ax.plot(g, THRESHOLD, "o", color=col, ms=9, mec="k")
    ax.plot([], [], color="0.4", ls=":", label="Train accuracy (dotted)")
    ax.axhline(THRESHOLD, color="0.3", lw=1)
    ax.set(xscale="log", xlabel=f"RBF γ (C = {BEST_C_RBF}), larger = more overfit",
           ylabel=f"Clean test accuracy (mean over {K_BONUS} models)",
           title=f"Model complexity at which test accuracy drops below {THRESHOLD:.0%}, per training augmentation")
    ax.legend()
    save(fig, "q4b_bonus_threshold")
    plt.show()
    print(pd.DataFrame({"γ": GAMMA_BONUS.round(1), **{f"{k} train": v for k, v in train_acc.items()},
                        **{f"{k} test": v for k, v in test_acc.items()}}).round(3).to_string(index=False))
    return pd.DataFrame({"Augmentation": list(test_acc), f"γ at {THRESHOLD:.0%}": [threshold_crossing(a) for a in test_acc.values()]}).set_index("Augmentation").round(0)


def main():
    Xtr, Xte, ytr, yte = split_datasets()["Circular"]
    full = poly_svm().fit(Xtr, ytr)
    rng = np.random.default_rng(SEED + 9)
    aug, aug_metrics = fit_augmentations(Xtr, ytr, Xte, yte, rng)
    print(results_table(full, Xtr, ytr, Xte, yte, aug_metrics))
    plot_augmentations(aug, Xte, yte)
    plot_distributions(full, Xte, yte, aug_metrics)
    plot_confusion_matrices(aug_metrics, len(yte))
    plot_noise_sweep(Xtr, ytr, Xte, yte, rng)
    print(bonus_threshold(Xte, yte, Xtr, ytr, rng))


if __name__ == "__main__":
    main()
