import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from agglomerative_model import features_clustering_b4_eda
from cluster_analysis_techniques import elbow_visualization, plot_cluster_diagnostics, \
    plot_feature_distributions_by_cluster, compute_risk_score, display_confusion_matrix
from cluster_metrics import calc_metrics
from clustering import plot_clusters

features_before = ["subdomain_count", 'sender_domain_freq', 'path_length', 'time_zone_diff', 'url_length', 'uses_https',
                   "subdomain_density", "path_to_domain_ratio", "is_long_url", "https_with_long_url",
                   "rare_domain_with_subdomains"]


def kmeans_clustering_before_eda(df):
    X = df[features_before].copy()
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    # Reduce dimensionality for visualization
    pca = PCA(n_components=5)
    x_pca = pca.fit_transform(X_scaled)
    elbow_visualization(x_pca)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    loadings = pd.DataFrame(pca.components_, columns=X.columns, index=[f'PC{i + 1}' for i in range(pca.n_components)])
    print("\n loadings: ", loadings)
    k = 7  # start with 3 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    km_fit = kmeans.fit(x_pca)
    df['kmeans_cluster'] = km_fit.labels_
    plot_cluster_diagnostics(x_pca, km_fit, "kmeans")
    cluster_colours = sns.color_palette("Set2", n_colors=km_fit.n_clusters)
    plot_feature_distributions_by_cluster(X_scaled_df, km_fit.labels_, cluster_colours,
                                          "Cluster Distribution - kmeans before eda")
    calc_metrics(df, df['kmeans_cluster'], features_before, k, x_pca, algo="KMeans_clean")
    plot_clusters(x_pca, df['kmeans_cluster'], algo="KMeans", eda="before eda")
    cluster_composition_kmeans = (
        df.groupby(["kmeans_cluster", "label"])
        .size()
        .unstack(fill_value=0)
    )
    print("Cluster composition kmeans : ", cluster_composition_kmeans)
    compute_risk_score(df)
    display_confusion_matrix(df)


def kmeans_clustering_after_eda(df):
    X = df[features_before].copy()
    scaler = RobustScaler()
    drop_binary = ['uses_https', 'is_long_url', 'https_with_long_url', 'rare_domain_with_subdomains']
    X.drop(columns=drop_binary, inplace=True)
    skewed = ['sender_domain_freq', 'path_length', 'subdomain_density', 'path_to_domain_ratio']
    for col in skewed:
        X[col] = np.log1p(X[col])
    X_scaled = scaler.fit_transform(X)

    # Reduce dimensionality for visualization
    pca = PCA(n_components=5)
    x_pca = pca.fit_transform(X_scaled)
    elbow_visualization(x_pca)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    loadings = pd.DataFrame(pca.components_, columns=X.columns, index=[f'PC{i + 1}' for i in range(pca.n_components)])
    print("\n loadings: ", loadings)
    k = 7  # start with 3 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    km_fit = kmeans.fit(x_pca)
    df['kmeans_cluster'] = km_fit.labels_
    plot_cluster_diagnostics(x_pca, km_fit, "kmeans_engineered")
    cluster_colours = sns.color_palette("Set2", n_colors=km_fit.n_clusters)
    plot_feature_distributions_by_cluster(X_scaled_df, km_fit.labels_, cluster_colours,
                                          "Cluster distribution - kmeans after eda")
    calc_metrics(df, df['kmeans_cluster'], features_before, k, x_pca, algo="KMeans")
    plot_clusters(x_pca, df['kmeans_cluster'], algo="KMeans", eda="after eda corrections")
    cluster_composition_kmeans = (
        df.groupby(["kmeans_cluster", "label"])
        .size()
        .unstack(fill_value=0)
    )
    print("Cluster composition kmeans : ", cluster_composition_kmeans)
    compute_risk_score(df)
    display_confusion_matrix(df)





