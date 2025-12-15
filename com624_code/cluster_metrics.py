from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, \
    mutual_info_score

import seaborn as sns


def calc_metrics(df, cluster_label, features_clustering, k, x_pca, algo):
    print(f"\n{algo} clustering")
    silhouette = silhouette_score(x_pca, cluster_label)
    if k > 0:
        print(f"\nSilhouette Score for k={k}: {silhouette:.3f}")
        db_index = davies_bouldin_score(x_pca, cluster_label)
        print(f"Davies-Bouldin Score for k={k}: {db_index:.3f}")
        ch_index = calinski_harabasz_score(x_pca, cluster_label)
        print(f"Calinski Harabasz Score for k={k}: {ch_index:.3f}")
    else:
        print(f"\nSilhouette Score : {silhouette:.3f}")
        db_index = davies_bouldin_score(x_pca, cluster_label)
        print(f"Davies-Bouldin Score {db_index:.3f}")
        ch_index = calinski_harabasz_score(x_pca, cluster_label)
        print(f"Calinski Harabasz Score: {ch_index:.3f}")

    ari = adjusted_rand_score(df['label'], cluster_label)
    print(f"Adjusted Rand Score: {ari:.3f}")
    mi = mutual_info_score(df['label'], cluster_label)
    print(f"Adjusted mutual info score: {mi:.3f}")
    # cluster information
    print(f"-------\n\ncluster label: {algo}\n---------")
    print(df.groupby(cluster_label)[features_clustering].mean().round(2))



