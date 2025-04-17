# visualize_embeddings.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_embeddings_3d(embeddings, labels, title="3D Embedding Visualization"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        embeddings[:, 2],
        c=labels,
        cmap='tab10',
        s=25,
        alpha=0.8
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()
