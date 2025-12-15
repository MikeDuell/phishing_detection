import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler

from cluster_analysis_techniques import plot_cluster_diagnostics, plot_feature_distributions_by_cluster, \
    display_confusion_matrix
from cluster_metrics import calc_metrics
from clustering import plot_clusters

# -----------------------------
# Utility functions
# -----------------------------

def plot_feature_histograms(df, transformed_df=None, scaled_df=None, features=None):
    """Compare raw, log-transformed, and scaled distributions side-by-side."""
    for col in features:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f"Raw {col}")

        if transformed_df is not None:
            sns.histplot(transformed_df[col], kde=True, ax=axes[1])
            axes[1].set_title(f"Log {col}")

        if scaled_df is not None:
            sns.histplot(scaled_df[col], kde=True, ax=axes[2])
            axes[2].set_title(f"Scaled {col}")

        plt.tight_layout()
        plt.show()


def tune_hdbscan(X, sizes=[100, 500, 1000], samples=[5, 10, 20]):
    """Sweep HDBSCAN parameters and return cluster/noise diagnostics."""
    results = []
    for size in sizes:
        for sample in samples:
            clusterer = HDBSCAN(min_cluster_size=size, min_samples=sample).fit(X)
            labels = clusterer.labels_
            noise_frac = (labels == -1).sum() / len(labels)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            results.append((size, sample, n_clusters, noise_frac))

    return pd.DataFrame(results, columns=["min_cluster_size","min_samples","clusters","noise_fraction"])


def profile_clusters(df, cluster_col, numeric_features, binary_flags, label_col="label"):
    """Summarize cluster composition, numeric means, and binary flag proportions."""
    summary = {}
    for c in sorted(df[cluster_col].unique()):
        subset = df[df[cluster_col] == c]
        summary[c] = {
            "size": len(subset),
            "label_composition": subset[label_col].value_counts(normalize=True).round(2).to_dict(),
            "numeric_means": subset[numeric_features].mean().round(2).to_dict(),
            "binary_flags": subset[binary_flags].mean().round(2).to_dict()
        }
    return summary


def noise_diagnostics(labels):
    """Print noise fraction for cluster labels."""
    noise_frac = (labels == -1).sum() / len(labels)
    print(f"Noise fraction: {noise_frac:.2%}")


# -----------------------------
# Before EDA
# -----------------------------

features_before = [
    "sender_domain_freq", "path_length", "url_length",
    "subdomain_density", "path_to_domain_ratio", "time_zone_diff"
]

def HDBscan_before_eda(df, min_cluster_size=1000, min_samples=5, n_components=2):
    X = df[features_before].copy()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)



    # PCA for visualization only
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(X_scaled)

    # HDBSCAN clustering
    cluster_model = HDBSCAN(
        min_samples=min_samples,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
        min_cluster_size=min_cluster_size,
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.1,
    )
    db_fit = cluster_model.fit(X_scaled)
    df['DBSCAN_cluster'] = db_fit.labels_

    # Diagnostics
    plot_cluster_diagnostics(X_scaled, db_fit, "DBSCAN")
    calc_metrics(df, df['DBSCAN_cluster'], X.columns, 0, x_pca, algo="DBSCAN")
    plot_clusters(x_pca, df['DBSCAN_cluster'], algo="DBSCAN")
    cluster_colours = sns.color_palette("Set2", )
    plot_feature_distributions_by_cluster(X_scaled_df, db_fit.labels_, cluster_colours, "HDBScan Before EDA")

    # Cluster composition
    cluster_composition_dbscan = (
        df.groupby(["DBSCAN_cluster", "label"])
        .size()
        .unstack(fill_value=0)
    )
    print("\nCluster composition HDBscan", cluster_composition_dbscan)

    # Binary flags
    binary_flags = ['is_long_url', 'https_with_long_url', 'rare_domain_with_subdomains']
    flag_summary = df.groupby('DBSCAN_cluster')[binary_flags].mean().round(2)
    print("\nBinary flag proportions per cluster:\n", flag_summary)

    # Noise diagnostics
    noise_diagnostics(db_fit.labels_)
    display_confusion_matrix(df, "DBSCAN_cluster")


    return db_fit, cluster_composition_dbscan, flag_summary


# -----------------------------
# After EDA
# -----------------------------

features_after = [
    "sender_domain_freq", "path_length", "url_length",
    "subdomain_density", "path_to_domain_ratio", "time_zone_diff"
]

def HDBscan_after_eda(df, min_cluster_size=500, min_samples=5, n_components=4):
    X = df[features_after].copy()

    # Log-transform skewed features
    skewed = ["sender_domain_freq", "path_length", "subdomain_density", "path_to_domain_ratio"]
    for col in skewed:
        X[col] = np.log1p(np.clip(X[col], a_min=0, a_max=None))

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    tune_stats =tune_hdbscan(X_scaled_df)
    print("\n\n\n\n\n Stats: ", tune_stats.to_string())


    # Histograms
    plot_feature_histograms(df, transformed_df=X, scaled_df=X_scaled_df, features=skewed)

    # PCA for visualization only
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(X_scaled)

    # HDBSCAN clustering
    cluster_model = HDBSCAN(
        min_samples=min_samples,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
        min_cluster_size=min_cluster_size,
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.3,
    )
    db_fit = cluster_model.fit(X_scaled)
    df['DBSCAN_cluster_after'] = db_fit.labels_

    # Diagnostics
    plot_cluster_diagnostics(X_scaled, db_fit, "DBSCAN_after")
    calc_metrics(df, df['DBSCAN_cluster_after'], X.columns, 0, x_pca, algo="DBSCAN")
    plot_clusters(x_pca, df['DBSCAN_cluster_after'], algo="DBSCAN")
    cluster_colours = sns.color_palette("Set2", )
    plot_feature_distributions_by_cluster(X_scaled_df, db_fit.labels_, cluster_colours, "HDBSCAN After EDA")


    # Cluster composition
    cluster_composition = (
        df.groupby(["DBSCAN_cluster_after", "label"])
        .size()
        .unstack(fill_value=0)
    )
    print("\nCluster composition after EDA:\n", cluster_composition)

    # Binary flags
    binary_flags = ['is_long_url', 'https_with_long_url', 'rare_domain_with_subdomains']
    flag_summary = df.groupby('DBSCAN_cluster_after')[binary_flags].mean().round(2)
    print("\nBinary flag proportions per cluster:\n", flag_summary)

    # Noise diagnostics
    noise_diagnostics(db_fit.labels_)

    # Cluster profiling summary
    summary = profile_clusters(df, 'DBSCAN_cluster_after', features_after, binary_flags)
    print("\nCluster profiles:\n", summary)
    display_confusion_matrix(df, "DBSCAN_cluster_after")


    return db_fit, cluster_composition, flag_summary, summary