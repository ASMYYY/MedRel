from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


def plot_training_curve(values: Sequence[float], title: str, ylabel: str, save_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(values, label=ylabel)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_multi_curves(curves: Dict[str, Sequence[float]], title: str, ylabel: str, save_path: str) -> None:
    plt.figure(figsize=(6, 4))
    for label, vals in curves.items():
        plt.plot(vals, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
