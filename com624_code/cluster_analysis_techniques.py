import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from ds_utils.unsupervised import plot_cluster_cardinality, plot_cluster_magnitude, plot_magnitude_vs_cardinality
from scipy.spatial.distance import euclidean

from supervised_learning import evaluate_model, plot_metrics


def elbow_visualization(x_axis, k_range=(2, 10)):
    fig, ax = plt.subplots()
    visualizer = KElbowVisualizer(KMeans(random_state=42), k=k_range, ax=ax)
    visualizer.fit(x_axis)
    elbow_k = visualizer.elbow_value_

    if elbow_k is not None:
        ax.axvline(elbow_k, color='red', linestyle='--', label=f"Elbow at k={elbow_k}")
        ax.text(elbow_k, ax.get_ylim()[1] * 0.9, f"k={elbow_k}", color='red', ha='center')
        ax.legend()
    else:
        print("No clear elbow detected — consider expanding k_range or inspecting the curve manually.")

    plt.tight_layout()
    plt.show()
    return elbow_k


def plot_cluster_diagnostics(x_std, km_fit, model):
    """
    Generate cluster diagnostic plots: cardinality, magnitude, and magnitude vs cardinality.
    Parameters
    ----------
    x_std : pandas.DataFrame
        Scaled dataset used for clustering.
    km_fit : sklearn.cluster.KMeans
        Fitted KMeans model.
   """

    if model == 'kmeans':
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        plot_cluster_cardinality(km_fit.labels_, ax=ax1, title="Cardinality")
        plot_cluster_magnitude(x_std, km_fit.labels_, km_fit.cluster_centers_, euclidean, ax=ax2, title="Magnitude")
        plot_magnitude_vs_cardinality(x_std, km_fit.labels_, km_fit.cluster_centers_, euclidean, ax=ax3,
                                      title="Magnitude vs. Cardinality")
        plt.tight_layout()
        plt.show()

    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(12, 4))
        plot_cluster_cardinality(km_fit.labels_, ax=ax1, title="Cardinality")
        plt.tight_layout()
        plt.show()


def plot_feature_distributions_by_cluster(x, cluster_labels, cluster_colors, title="Feature distributions per cluster"):
    """
    Plots boxplots of each feature grouped by cluster, with median, IQR, and outlier counts
    annotated at the top of each subplot.
    """
    # Add cluster column for plotting
    x_plot = x.copy()
    x_plot['cluster'] = cluster_labels

    features = x.columns
    ncols = 4
    nrows = len(features) // ncols + (len(features) % ncols > 0)

    fig = plt.figure(figsize=(30, 20))

    for n, feature in enumerate(features):
        ax = plt.subplot(nrows, ncols, n + 1)
        box = x_plot[[feature, 'cluster']].boxplot(
            by='cluster', ax=ax, return_type='both', patch_artist=True
        )

        for row_key, (ax, row) in box.items():
            ax.set_xlabel('Cluster')
            ax.set_title(feature, fontweight="bold")
            for i, b in enumerate(row['boxes']):
                if i < len(cluster_colors):
                    b.set_facecolor(cluster_colors[i])

        # Compute stats per cluster
        grouped = x_plot.groupby('cluster')[feature]
        stats_text = []
        for cluster, values in grouped:
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            median = values.median()
            # Outliers: beyond 1.5*IQR
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((values < lower_bound) | (values > upper_bound)).sum()
            stats_text.append(f"C{cluster}: Med={median:.2f}, IQR={iqr:.2f}, Outliers={outliers}")

        # Place one combined annotation at the top of the subplot
        ax.text(0.5, 1.05, "\n".join(stats_text),
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=8, color='blue')

    fig.suptitle(title, fontsize=28, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.tight_layout()
    plt.show()


def compute_risk_score(df):
    """Add a 'risk_score' column based on how many diagnostic flags are triggered per sample."""
    flags = [
        "is_long_url",
        "https_with_long_url",
        "rare_domain_with_subdomains",
        "suspicious_tld",
        "has_at_symbol",
        "digit_count_domain",
    ]
    df["risk_score"] = df[flags].sum(axis=1)
    return df


def display_confusion_matrix(df, algo="kmeans_cluster"):
    comp = df.groupby(algo)["label"].mean().round(3)

    correction_map = {}
    for cluster, proportion in comp.items():
        if proportion >= 0.5:  # majority phishing
            correction_map[cluster] = 1
        else:  # majority benign
            correction_map[cluster] = 0

    # --- 5. Apply correction map ---
    pred_col = f"{algo}_predicted"
    df[pred_col] = df[algo].map(correction_map)

    df["algo_predicted"] = df[algo].map(correction_map)

    # --- 6. Confusion matrix ---
    y_true = df["label"]  # ground truth (1=phishing, 0=benign)
    y_pred = df["algo_predicted"]

    cm = confusion_matrix(y_true, y_pred)
    metrics = evaluate_model(y_true, y_pred)
    plot_metrics(algo, metrics)


    sns.heatmap(cm, annot=True, fmt='g', xticklabels=['benign', 'Phishing'], yticklabels=['benign', 'Phishing'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix for {algo}")
    plt.show()
