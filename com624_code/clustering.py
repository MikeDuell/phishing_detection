from collections import defaultdict

import seaborn as sns
from matplotlib import pyplot as plt


def plot_clusters(x_pca, df, algo, eda="before EDA preprocessing"):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=df, palette='Set2')
    plt.title(f"Phishing Email Clusters {algo}(PCA-reduced) on {eda}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title=f'{algo} Cluster')
    plt.tight_layout()
    plt.show()







