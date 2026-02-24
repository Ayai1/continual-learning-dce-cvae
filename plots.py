import matplotlib.pyplot as plt
import numpy as np

def plot_forgetting_curve(acc_matrix, title, filename):
    """
    Plots a forgetting/accuracy curve from a list of lists.
    Each row in acc_matrix is accuracy after each task.
    """
    plt.figure(figsize=(10, 6))

    # Number of tasks
    T = len(acc_matrix)

    # Plot each task's accuracy evolution
    for i in range(T):
        # pad missing points with NaN
        row = acc_matrix[i]
        xs = np.arange(1, len(row) + 1)
        plt.plot(xs, row, marker='o', label=f"Task {i+1}")

    plt.title(title)
    plt.xlabel("Evaluation Task")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()