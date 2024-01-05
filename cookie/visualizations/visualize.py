import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne_visualization(data, labels, class_labels):
    """Visualize the data in a 2D space using t-SNE.

    Arguments:
        data: numpy array, data to visualize
        labels: numpy array, labels of data
        class_labels: list, list of class labels
    """

    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(data)

    # Visualize features in a 2D space with different colors for each class
    plt.figure(figsize=(10, 8))
    for i in range(len(class_labels)):
        indices = labels == i
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=class_labels[i], alpha=0.5)

    plt.legend()
    plt.title("t-SNE Visualization of Fashion MNIST Features")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Save the visualization to a file
    # os.makedirs('reports/figures', exist_ok=True)
    plt.savefig("reports/figures/tsne_visualization.png")
    plt.show()
