import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from scripts.common import plot_boundary, save, split_datasets
from scripts.q2.perceptron import min_distance, perc_predict, perceptron

C_HARD = 1000
C_SOFT = 1


def fit_models(datasets):
    models, rows = {}, []
    for name, (Xtr, Xte, ytr, yte) in datasets.items():
        t = time.perf_counter()
        w, mistakes = perceptron(Xtr, ytr)
        t_perc = time.perf_counter() - t

        C = C_HARD if name == "Linear" else C_SOFT
        t = time.perf_counter()
        svm = SVC(kernel="linear", C=C).fit(Xtr, ytr)
        t_svm = time.perf_counter() - t

        models[name] = {"perc": (w, mistakes), "svm": svm}
        rows += [
            {"Dataset": name, "Model": "Perceptron",
             "Train acc": np.mean(perc_predict(w, Xtr) == ytr), "Test acc": np.mean(perc_predict(w, Xte) == yte),
             "Converged": mistakes[-1] == 0, "Epochs": len(mistakes),
             "Min dist. to boundary": min_distance(w[1:], w[0], Xtr, ytr), "# SV": "-", "Time (s)": t_perc},
            {"Dataset": name, "Model": f"Linear SVM (C={C})",
             "Train acc": svm.score(Xtr, ytr), "Test acc": svm.score(Xte, yte),
             "Converged": True, "Epochs": "-",
             "Min dist. to boundary": min_distance(svm.coef_[0], svm.intercept_[0], Xtr, ytr),
             "# SV": len(svm.support_), "Time (s)": t_svm},
        ]
    return models, pd.DataFrame(rows).set_index(["Dataset", "Model"])


def plot_decision_boundaries(datasets, models, results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for row, (name, (Xtr, Xte, ytr, yte)) in enumerate(datasets.items()):
        w, _ = models[name]["perc"]
        svm = models[name]["svm"]
        plot_boundary(axes[row, 0], Xte, yte, lambda g: perc_predict(w, g),
                      f"{name} — Perceptron (test acc {results.loc[(name, 'Perceptron'), 'Test acc']:.3f})")
        plot_boundary(axes[row, 1], Xte, yte, svm.predict,
                      f"{name} — Linear SVM (test acc {svm.score(Xte, yte):.3f}, {len(svm.support_)} SVs)", svm=svm)
    plt.tight_layout()
    save(fig, "q2_decision_boundaries")
    plt.show()


def plot_convergence(models):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, m in models.items():
        _, mistakes = m["perc"]
        ax.plot(range(1, len(mistakes) + 1), mistakes, marker="o", ms=3, label=name)
    ax.set(xlabel="Epoch", ylabel="Mistakes (weight updates)", title="Perceptron mistakes per epoch", yscale="symlog")
    ax.legend()
    save(fig, "q2_convergence")
    plt.show()


def main():
    datasets = split_datasets()
    models, results = fit_models(datasets)
    print(results.round(3))
    plot_decision_boundaries(datasets, models, results)
    plot_convergence(models)
    return datasets, models


if __name__ == "__main__":
    main()
