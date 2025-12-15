import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cluster_analysis_techniques import plot_cluster_diagnostics, plot_feature_distributions_by_cluster, \
    display_confusion_matrix
from cluster_metrics import calc_metrics
from clustering import plot_clusters

# Define your features here
features_clustering_b4_eda = [
    "sender_domain_freq", "path_length", "url_length",
    "subdomain_density", "path_to_domain_ratio", "time_zone_diff"
]

def agglomerative_b4_eda(df, n_components=2):
    # Select features
    X = df[features_clustering_b4_eda].copy()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # PCA for visualization only
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(X_scaled)

    # Agglomerative clustering
    clustering = AgglomerativeClustering(n_clusters=7, linkage='ward')
    agg_fit = clustering.fit(x_pca)

    # Add labels to df
    df['agglomerative_cluster'] = agg_fit.labels_

    # Diagnostics
    plot_cluster_diagnostics(x_pca, agg_fit, "Agglomerative")
    calc_metrics(df, df['agglomerative_cluster'], features_clustering_b4_eda, 0, x_pca, algo="Agglomerative")
    plot_clusters(x_pca, df['agglomerative_cluster'], algo="Agglomerative")
    cluster_colours = sns.color_palette("Set2", )
    plot_feature_distributions_by_cluster(X_scaled_df, agg_fit.labels_, cluster_colours, "Agglomerative before EDA")
    # Cluster composition
    cluster_composition_agglomerate = (
        df.groupby(["agglomerative_cluster", "label"])
        .size()
        .unstack(fill_value=0)
    )
    print("\nCluster composition (Agglomerative):\n", cluster_composition_agglomerate)
    display_confusion_matrix(df, "agglomerative_cluster")

    return agg_fit, cluster_composition_agglomerate

def agglomerative_after_eda(df, n_components=2):
    # Select features
    X = df[features_clustering_b4_eda].copy()
    skewed = ["sender_domain_freq", "path_length", "subdomain_density", "path_to_domain_ratio"]
    for col in skewed:
        X[col] = np.log1p(np.clip(X[col], a_min=0, a_max=None))

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # PCA for visualization only
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(X_scaled)

    # Agglomerative clustering
    clustering = AgglomerativeClustering(n_clusters=7, linkage='ward')
    agg_fit = clustering.fit(x_pca)

    # Add labels to df
    df['agglomerative_cluster'] = agg_fit.labels_

    # Diagnostics
    plot_cluster_diagnostics(x_pca, agg_fit, "Agglomerative")
    calc_metrics(df, df['agglomerative_cluster'], features_clustering_b4_eda, 0, x_pca, algo="Agglomerative")
    plot_clusters(x_pca, df['agglomerative_cluster'], algo="Agglomerative")
    cluster_colours = sns.color_palette("Set2", )
    plot_feature_distributions_by_cluster(X_scaled_df, agg_fit.labels_, cluster_colours, "Agglomerative after EDA")
    # Cluster composition
    cluster_composition_agglomerate = (
        df.groupby(["agglomerative_cluster", "label"])
        .size()
        .unstack(fill_value=0)
    )
    print("\n\n\n\n\n Clusters: count", df.groupby("agglomerative_cluster")["label"].value_counts(normalize=True))
    print("\nCluster composition (Agglomerative):\n", cluster_composition_agglomerate)
    summary = (
        df.groupby("agglomerative_cluster")[features_clustering_b4_eda]
        .median()
        .round(2)
    )

    summary["phishing_proportion"] = (
        df.groupby("agglomerative_cluster")["label"].mean().round(2)
    )

    print("\n\n\n\n\n Summary :",summary)
    display_confusion_matrix(df, "agglomerative_cluster")

    return agg_fit, cluster_composition_agglomerate