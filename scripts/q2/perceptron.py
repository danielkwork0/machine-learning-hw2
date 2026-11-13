import numpy as np

ETA = 0.1
MAX_EPOCHS = 100


def perceptron(X, y, eta=ETA, max_epochs=MAX_EPOCHS, seed=0, history=None):
    r = np.random.default_rng(seed)
    Xa = np.c_[np.ones(len(X)), X]
    w = r.normal(0, 0.01, Xa.shape[1])
    mistakes = []
    for _ in range(max_epochs):
        m = 0
        for i in r.permutation(len(Xa)):
            if y[i] * (Xa[i] @ w) <= 0:
                w += eta * y[i] * Xa[i]
                m += 1
        mistakes.append(m)
        if history is not None:
            history.append(w.copy())
        if m == 0:
            break
    return w, mistakes


def perc_predict(w, X):
    return np.sign(np.c_[np.ones(len(X)), X] @ w)


def min_distance(w, b, X, y):
    d = (y * (X @ w + b)).min() / np.linalg.norm(w)
    # negative means some point is misclassified, so there is no margin to report
    return d if d > 0 else np.nan
