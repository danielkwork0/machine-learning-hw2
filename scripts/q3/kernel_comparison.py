import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from scripts.common import POLY_DEGREE, SEED, plot_boundary, poly_svm, save, split_datasets
from scripts.q2.perceptron_vs_svm import C_SOFT

CV_FOLDS = 5
C_GRID = [0.1, 1, 10, 100]
GAMMA_GRID = [0.01, 0.1, 1, 10, 100]
DEGREES = [1, 2, 3, 4, 5, 6]
GAMMAS = np.logspace(-3, 3, 13)
R2_SPLIT = 2.25  # (1.5)², halfway between the disc (r ≤ 1) and the ring (r ≥ 2)


def tune_kernels(Xtr, ytr, Xte, yte, cv):
    searches = {
        "Polynomial (d=2)": GridSearchCV(SVC(kernel="poly", degree=POLY_DEGREE, gamma=1, coef0=1), {"C": C_GRID}, cv=cv, n_jobs=-1),
        "RBF": GridSearchCV(SVC(kernel="rbf"), {"C": C_GRID, "gamma": GAMMA_GRID}, cv=cv, n_jobs=-1),
    }
    kernels, rows = {}, []
    for name, gs in searches.items():
        gs.fit(Xtr, ytr)
        model = gs.best_estimator_
        t = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_time = time.perf_counter() - t
        pred, score = model.predict(Xte), model.decision_function(Xte)
        kernels[name] = model
        rows.append({"Kernel": name, "Best params": gs.best_params_, "CV acc": gs.best_score_,
                     "Test acc": accuracy_score(yte, pred), "Test F1": f1_score(yte, pred),
                     "Test AUC": roc_auc_score(yte, score), "# SV": len(model.support_), "Fit time (s)": fit_time})
    return kernels, pd.DataFrame(rows).set_index("Kernel")


def plot_boundaries(Xte, yte, svm_lin, kernels):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_boundary(axes[0], Xte, yte, svm_lin.predict, f"Linear SVM (Q2): test acc {svm_lin.score(Xte, yte):.3f}", svm=svm_lin)
    for ax, (name, model) in zip(axes[1:], kernels.items()):
        plot_boundary(ax, Xte, yte, model.predict,
                      f"{name}: test acc {model.score(Xte, yte):.3f}, {len(model.support_)} SVs", svm=model)
    plt.tight_layout()
    save(fig, "q3_decision_boundaries")
    plt.show()


def plot_lifted(Xtr, ytr):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    r2 = Xtr[:, 0] ** 2 + Xtr[:, 1] ** 2
    for c, col in [(-1, "white"), (1, "royalblue")]:
        m = ytr == c
        ax.scatter(Xtr[m, 0], Xtr[m, 1], r2[m], s=4, c=col, edgecolors="k", linewidths=0.2, alpha=0.6, label=f"Y = {int(c > 0)}")
    gx, gy = np.meshgrid(np.linspace(-3.5, 3.5, 2), np.linspace(-3.5, 3.5, 2))
    ax.plot_surface(gx, gy, np.full_like(gx, R2_SPLIT), color="green", alpha=0.3)
    ax.set(xlabel="X1", ylabel="X2", zlabel="X1² + X2²", title="Lifting to z = X1² + X2²: a plane separates the classes")
    ax.view_init(elev=15, azim=35)
    ax.legend()
    save(fig, "q3_lifted")
    plt.show()


def plot_confusion_matrices(Xte, yte, kernels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (name, model) in zip(axes, kernels.items()):
        ConfusionMatrixDisplay(confusion_matrix(yte, model.predict(Xte)), display_labels=["Y = 0", "Y = 1"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{name}: confusion matrix (test)")
    plt.tight_layout()
    save(fig, "q3_confusion_matrices")
    plt.show()

    errors = {name: set(np.flatnonzero(m.predict(Xte) != yte)) for name, m in kernels.items()}
    (n1, e1), (n2, e2) = errors.items()
    r_err = np.hypot(*Xte[sorted(e1 | e2)].T)
    print(f"Misclassified: {n1} {len(e1)}, {n2} {len(e2)}, shared {len(e1 & e2)}; "
          f"their radii lie in [{r_err.min():.2f}, {r_err.max():.2f}] (the gap between disc and ring is 1–2)")


def plot_roc(Xte, yte, svm_lin, kernels):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (name, model), col in zip(kernels.items(), ["tab:green", "tab:purple"]):
        fpr, tpr, _ = roc_curve(yte, model.decision_function(Xte))
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC = {roc_auc_score(yte, model.decision_function(Xte)):.5f})")
    fpr, tpr, _ = roc_curve(yte, svm_lin.decision_function(Xte))
    ax.plot(fpr, tpr, color="0.5", ls="--", label=f"Linear SVM (AUC = {roc_auc_score(yte, svm_lin.decision_function(Xte)):.3f})")
    ax.set(xscale="log", xlim=(1e-4, 1), ylim=(0, 1.01), xlabel="False positive rate (log scale)", ylabel="True positive rate",
           title="ROC curves (test)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    save(fig, "q3_roc")
    plt.show()


def plot_sensitivity(Xtr, ytr, cv, kernels):
    C_poly, C_rbf = kernels["Polynomial (d=2)"].C, kernels["RBF"].C

    def cv_curve(make_model, values):
        scores = np.array([cross_val_score(make_model(v), Xtr, ytr, cv=cv, n_jobs=-1) for v in values])
        return scores.mean(1), scores.std(1)

    poly_m, poly_s = cv_curve(lambda d: poly_svm(C=C_poly, degree=d), DEGREES)
    rbf_m, rbf_s = cv_curve(lambda g: SVC(kernel="rbf", gamma=g, C=C_rbf), GAMMAS)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    ax0.errorbar(DEGREES, poly_m, yerr=poly_s, color="tab:green", marker="o", capsize=3)
    ax0.axvline(POLY_DEGREE, color="k", ls=":", label=f"chosen d = {POLY_DEGREE}")
    ax0.set(xlabel="Polynomial degree d", ylabel="5-fold CV accuracy", title=f"Polynomial kernel (C = {C_poly})")
    ax0.legend()
    ax1.errorbar(GAMMAS, rbf_m, yerr=rbf_s, color="tab:purple", marker="o", capsize=3)
    ax1.axvline(kernels["RBF"].gamma, color="k", ls=":", label=f"chosen γ = {kernels['RBF'].gamma}")
    ax1.set(xscale="log", xlabel="γ", title=f"RBF kernel (C = {C_rbf})")
    ax1.legend()
    plt.tight_layout()
    save(fig, "q3_sensitivity")
    plt.show()
    print("Poly CV acc by degree:", dict(zip(DEGREES, poly_m.round(4))))
    print("RBF  CV acc by gamma: ", dict(zip(GAMMAS.round(3), rbf_m.round(4))))


def main():
    Xtr, Xte, ytr, yte = split_datasets()["Circular"]
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    svm_lin = SVC(kernel="linear", C=C_SOFT).fit(Xtr, ytr)
    kernels, results = tune_kernels(Xtr, ytr, Xte, yte, cv)
    print(results)
    plot_boundaries(Xte, yte, svm_lin, kernels)
    plot_lifted(Xtr, ytr)
    plot_confusion_matrices(Xte, yte, kernels)
    plot_roc(Xte, yte, svm_lin, kernels)
    plot_sensitivity(Xtr, ytr, cv, kernels)
    return kernels


if __name__ == "__main__":
    main()
