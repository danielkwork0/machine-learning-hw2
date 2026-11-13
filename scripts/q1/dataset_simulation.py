import matplotlib.pyplot as plt

from scripts.common import save, scatter, simulate_datasets


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (name, (x1, x2, y)), label in zip(axes, simulate_datasets().items(), "ab"):
        scatter(ax, x1, x2, y, f"({label}) {name} dataset")
    plt.tight_layout()
    save(fig, "q1_datasets")
    plt.show()


if __name__ == "__main__":
    main()
