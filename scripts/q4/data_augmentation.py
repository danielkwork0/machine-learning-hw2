import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score

from scripts.common import SEED, plot_boundary, poly_svm, save, split_datasets

K_AUG = 5_000
N_AUG = 1_000
SD_LEVELS = {"No noise (σ = 0)": 0.0, "Low SD (σ = 0.1)": 0.1, "High SD (σ = 0.5)": 0.5}
R_SPLIT = 1.5
SD_SWEEP = np.linspace(0, 1, 11)
K_SWEEP = 1_000


def augment(X, y, sd, k, n, rng):
    idx = np.argsort(rng.random((k, len(X))), axis=1)[:, :n]
    return X[idx] + rng.normal(0, sd, (k, n, X.shape[1])), y[idx]


def reference_rule(X):
    return np.where(np.hypot(X[..., 0], X[..., 1]) < R_SPLIT, -1, 1)


def evaluate_augmentations(best, Xte, yte, rng):
    aug, aug_metrics = {}, {}
    for name, sd in SD_LEVELS.items():
        Xa, ya = augment(Xte, yte, sd, K_AUG, N_AUG, rng)
        score = best.decision_function(Xa.reshape(-1, 2)).reshape(K_AUG, N_AUG)
        pred = np.where(score > 0, 1, -1)
        tp = ((pred == 1) & (ya == 1)).sum(1)
        fp = ((pred == 1) & (ya == -1)).sum(1)
        fn = ((pred == -1) & (ya == 1)).sum(1)
        tn = ((pred == -1) & (ya == -1)).sum(1)
        aug[name] = (Xa, ya, pred)
        aug_metrics[name] = {
            "acc": (tp + tn) / N_AUG,
            "f1": 2 * tp / (2 * tp + fp + fn),
            "auc": np.array([roc_auc_score(y, s) for y, s in zip(ya, score)]),
            "ref_acc": (reference_rule(Xa) == ya).mean(1),
            "cm": np.array([[tn.mean(), fp.mean()], [fn.mean(), tp.mean()]]),
        }
    return aug, aug_metrics


def results_table(best, Xtr, ytr, Xte, yte, aug_metrics):
    rows = [{"Augmentation": "Train set (no augmentation)", "Accuracy": f"{best.score(Xtr, ytr):.4f}"},
            {"Augmentation": "Full test set (no augmentation)", "Accuracy": f"{best.score(Xte, yte):.4f}"}]
    for name, m in aug_metrics.items():
        lo, hi = np.percentile(m["acc"], [2.5, 97.5])
        rows.append({"Augmentation": f"{name}, k = {K_AUG:,}",
                     "Accuracy": f"{m['acc'].mean():.4f} ± {m['acc'].std():.4f}", "Accuracy 95% range": f"[{lo:.3f}, {hi:.3f}]",
                     "F1": f"{m['f1'].mean():.4f} ± {m['f1'].std():.4f}", "AUC": f"{m['auc'].mean():.4f} ± {m['auc'].std():.4f}",
                     f"Reference rule r < {R_SPLIT} acc": f"{m['ref_acc'].mean():.4f}"})
    return pd.DataFrame(rows).set_index("Augmentation").fillna("–")


def plot_augmentations(best, aug):
    fig, axes = plt.subplots(1, len(SD_LEVELS), figsize=(18, 6))
    for ax, (name, (Xa, ya, pred)) in zip(axes, aug.items()):
        X0, y0, p0 = Xa[0], ya[0], pred[0]
        plot_boundary(ax, X0, y0, best.predict, f"{name}: one augmented set, acc {np.mean(p0 == y0):.3f}")
        wrong = p0 != y0
        ax.scatter(X0[wrong, 0], X0[wrong, 1], marker="x", c="red", s=40, linewidths=1.5, label=f"Misclassified ({wrong.sum()})")
        ax.legend(loc="upper left")
    plt.tight_layout()
    save(fig, "q4_augmentations")
    plt.show()


def plot_distributions(best, Xtr, ytr, aug_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    colors = ["0.4", "tab:green", "tab:red"]
    for ax, (key, label) in zip(axes, [("acc", "Accuracy"), ("f1", "F1-score"), ("auc", "AUC")]):
        for (name, m), col in zip(aug_metrics.items(), colors):
            ax.hist(m[key], bins=40, color=col, alpha=0.6, label=f"{name}: {m[key].mean():.3f}")
        ax.set(xlabel=label, ylabel="Augmented datasets", title=f"{label} over {K_AUG:,} augmented datasets")
        ax.legend(fontsize=9)
    axes[0].axvline(best.score(Xtr, ytr), color="k", ls="--", lw=1.5)
    axes[0].text(best.score(Xtr, ytr), axes[0].get_ylim()[1] * 0.95, " train acc", ha="right", va="top", rotation=90)
    plt.tight_layout()
    save(fig, "q4_metric_distributions")
    plt.show()


def plot_confusion_matrices(aug_metrics):
    fig, axes = plt.subplots(1, len(SD_LEVELS), figsize=(16, 4.5))
    for ax, (name, m) in zip(axes, aug_metrics.items()):
        ConfusionMatrixDisplay(m["cm"], display_labels=["Y = 0", "Y = 1"]).plot(ax=ax, cmap="Blues", colorbar=False, values_format=".1f")
        ax.set_title(f"{name}\nmean confusion matrix (of {N_AUG:,} points)")
    plt.tight_layout()
    save(fig, "q4_confusion_matrices")
    plt.show()


def plot_noise_sweep(best, Xtr, ytr, Xte, yte, rng):
    sweep = {"model": [], "ref": []}
    for sd in SD_SWEEP:
        Xa, ya = augment(Xte, yte, sd, K_SWEEP, N_AUG, rng)
        sweep["model"].append((best.predict(Xa.reshape(-1, 2)).reshape(K_SWEEP, N_AUG) == ya).mean(1))
        sweep["ref"].append((reference_rule(Xa) == ya).mean(1))
    sweep = {k: np.array(v) for k, v in sweep.items()}

    fig, ax = plt.subplots(figsize=(9, 5))
    for key, col, label in [("model", "tab:green", "Polynomial SVM (d=2)"), ("ref", "k", f"Reference rule r < {R_SPLIT}")]:
        m, s = sweep[key].mean(1), sweep[key].std(1)
        ax.plot(SD_SWEEP, m, color=col, marker="o", ls="-" if key == "model" else "--", label=label)
        ax.fill_between(SD_SWEEP, m - s, m + s, color=col, alpha=0.15)
    ax.axhline(best.score(Xtr, ytr), color="tab:blue", ls=":", label="Train accuracy")
    for sd, col in [(0.1, "tab:green"), (0.5, "tab:red")]:
        ax.axvline(sd, color=col, ls=":", alpha=0.7)
    ax.set(xlabel="Noise SD σ", ylabel=f"Accuracy (mean ± SD over {K_SWEEP:,} sets)", title="Accuracy under increasing augmentation noise")
    ax.legend()
    save(fig, "q4_noise_sweep")
    plt.show()
    print("σ:      ", SD_SWEEP.round(1))
    print("model:  ", sweep["model"].mean(1).round(4))
    print("ref:    ", sweep["ref"].mean(1).round(4))
    print("gap:    ", (sweep["model"].mean(1) - sweep["ref"].mean(1)).round(4))


def main():
    Xtr, Xte, ytr, yte = split_datasets()["Circular"]
    best = poly_svm().fit(Xtr, ytr)
    rng = np.random.default_rng(SEED + 2)
    aug, aug_metrics = evaluate_augmentations(best, Xte, yte, rng)
    print(results_table(best, Xtr, ytr, Xte, yte, aug_metrics))
    plot_augmentations(best, aug)
    plot_distributions(best, Xtr, ytr, aug_metrics)
    plot_confusion_matrices(aug_metrics)
    plot_noise_sweep(best, Xtr, ytr, Xte, yte, rng)


if __name__ == "__main__":
    main()
