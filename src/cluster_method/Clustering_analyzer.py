"""Main clustering analyzer that combines all clustering methods."""

import logging
from typing import Dict, Any, Optional, Literal
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from .kmeans import KMeansClustering
from .dbscan import DBSCANClustering
from .leiden import LeidenClustering

logger = logging.getLogger(__name__)


class ClusteringAnalyzer:
    """Main analyzer for performing clustering analysis using multiple methods."""

    def __init__(self):
        """Initialize the ClusteringAnalyzer."""
        self.kmeans_clusterer = KMeansClustering()
        self.dbscan_clusterer = DBSCANClustering()
        self.leiden_clusterer = LeidenClustering()
        logger.info("ClusteringAnalyzer initialized with all methods")

    def apply_kmeans_clustering(
        self,
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None,
        auto_k: bool = True
    ) -> Dict[str, Any]:
        """
        Apply KMeans clustering.

        Args:
            embeddings: Input embedding vectors
            n_clusters: Number of clusters (if None and auto_k=True, will be determined automatically)
            auto_k: Whether to determine k automatically

        Returns:
            Dictionary with model, labels, and metadata
        """
        model, labels, n_clusters_used, kmeans_metrics = self.kmeans_clusterer.cluster(embeddings, n_clusters, auto_k)
        return {
            'method': 'kmeans',
            'model': model,
            'labels': labels,
            'n_clusters': n_clusters_used,
            'embeddings_shape': embeddings.shape
            
        }

    def apply_dbscan_clustering(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 30,
        min_samples: int = 10,
        metric: str = 'euclidean'
    ) -> Dict[str, Any]:
        """
        Apply DBSCAN clustering.

        Args:
            embeddings: Input embedding vectors
            min_cluster_size: Minimum samples in a cluster
            min_samples: Minimum samples threshold
            metric: Distance metric

        Returns:
            Dictionary with model, labels, and metadata
        """
        model, labels, dbscan_metrics = self.dbscan_clusterer.cluster(embeddings, min_cluster_size, min_samples, metric)

        # Calculate noise percentage
        noise_count = np.sum(labels == -1)
        noise_percentage = (noise_count / len(labels)) * 100 if len(labels) > 0 else 0

        return {
            'method': 'dbscan',
            'model': model,
            'labels': labels,
            'noise_percentage': noise_percentage,
            'n_clusters': len(np.unique(labels[labels != -1])),
            'embeddings_shape': embeddings.shape,
            **dbscan_metrics
        }

    def apply_leiden_clustering(
        self,
        embeddings: np.ndarray,
        k: int = 30,
        use_snn: bool = True,
        resolution_parameter: float = 0.7,
        metric: Literal["cosine", "euclidean"] = "cosine",
        return_graph: bool = False,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Apply Leiden clustering.

        Args:
            embeddings: Input embedding vectors
            k: Number of nearest neighbors
            resolution_parameter: Resolution parameter for Leiden
            metric: Distance metric
            random_state: Random seed

        Returns:
            Dictionary with labels and comprehensive results
        """
        labels, results = self.leiden_clusterer.cluster(
            embeddings, k,use_snn,  resolution_parameter, metric, return_graph, random_state
        )

        return {
            'method': 'leiden',
            'labels':labels,
            **results
        }

    def select_best_clustering_method(
        self,
        embeddings: np.ndarray,
        kmeans_result: Dict[str, Any],
        dbscan_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare KMeans and DBSCAN clustering results and return the most explainable method.

        Args:
            embeddings: Input embeddings
            kmeans_result: Result from KMeans clustering
            dbscan_result: Result from DBSCAN clustering

        Returns:
            Dictionary with best method and metrics
        """
        kmeans_labels = kmeans_result['labels']
        dbscan_labels = dbscan_result['labels']
        noise_percentage = dbscan_result['noise_percentage']

        # Remove noise points from DBSCAN for fair comparison
        valid_dbscan_mask = dbscan_labels != -1
        dbscan_labels_clean = dbscan_labels[valid_dbscan_mask]
        embeddings_filtered = embeddings[valid_dbscan_mask]

        # Get cluster counts
        kmeans_n_clusters = kmeans_result['n_clusters']
        dbscan_n_clusters = dbscan_result['n_clusters']

        # Calculate metrics
        has_valid_dbscan_silhouette = len(embeddings_filtered) > 1 and dbscan_n_clusters > 1

        metrics = {}

        # KMeans Metrics
        kmeans_metrics = {}
        kmeans_silhouette = silhouette_score(embeddings, kmeans_labels)
        kmeans_silhouette_normalized = ((kmeans_silhouette + 1) / 2) * 100
        kmeans_metrics['silhouette_score'] = kmeans_silhouette_normalized

        kmeans_db_index = davies_bouldin_score(embeddings, kmeans_labels)
        kmeans_metrics['davies_bouldin_normalized'] = 100 / (1 + kmeans_db_index)

        kmeans_ch_index = calinski_harabasz_score(embeddings, kmeans_labels)
        kmeans_metrics['calinski_harabasz_normalized'] = min(kmeans_ch_index / 10, 100)

        kmeans_counts = np.bincount(kmeans_labels)
        kmeans_balance = 100 - (np.std(kmeans_counts) / np.mean(kmeans_counts) * 100)
        kmeans_balance = max(0, min(100, kmeans_balance))
        kmeans_metrics['cluster_balance_score'] = kmeans_balance

        kmeans_cluster_count_score = 100 - abs((kmeans_n_clusters - 5) * 10)
        kmeans_cluster_count_score = max(0, min(100, kmeans_cluster_count_score))
        kmeans_metrics['cluster_count_score'] = kmeans_cluster_count_score
        kmeans_metrics['cluster_count'] = kmeans_n_clusters

        kmeans_composite = (
            kmeans_silhouette_normalized * 40 +
            kmeans_metrics['davies_bouldin_normalized'] * 30 +
            kmeans_metrics['calinski_harabasz_normalized'] * 30
        ) / 100
        kmeans_metrics['composite_score'] = kmeans_composite

        # DBSCAN Metrics
        dbscan_metrics = {}

        try:
            if has_valid_dbscan_silhouette:
                dbscan_silhouette = silhouette_score(embeddings_filtered, dbscan_labels_clean)
                dbscan_silhouette_normalized = ((dbscan_silhouette + 1) / 2) * 100
                dbscan_metrics['silhouette_score'] = dbscan_silhouette_normalized
            else:
                dbscan_silhouette_normalized = 0
                dbscan_metrics['silhouette_score'] = 0
        except:
            dbscan_silhouette_normalized = 0
            dbscan_metrics['silhouette_score'] = 0

        if has_valid_dbscan_silhouette:
            try:
                dbscan_db_index = davies_bouldin_score(embeddings_filtered, dbscan_labels_clean)
                dbscan_metrics['davies_bouldin_normalized'] = 100 / (1 + dbscan_db_index)
            except:
                dbscan_metrics['davies_bouldin_normalized'] = 0
        else:
            dbscan_metrics['davies_bouldin_normalized'] = 0

        if has_valid_dbscan_silhouette:
            try:
                dbscan_ch_index = calinski_harabasz_score(embeddings_filtered, dbscan_labels_clean)
                dbscan_metrics['calinski_harabasz_normalized'] = min(dbscan_ch_index / 10, 100)
            except:
                dbscan_metrics['calinski_harabasz_normalized'] = 0
        else:
            dbscan_metrics['calinski_harabasz_normalized'] = 0

        if len(embeddings_filtered) > 0:
            dbscan_counts = np.bincount(dbscan_labels_clean)
            if len(dbscan_counts) > 1:
                dbscan_balance = 100 - (np.std(dbscan_counts) / np.mean(dbscan_counts) * 100)
                dbscan_balance = max(0, min(100, dbscan_balance))
            else:
                dbscan_balance = 0
        else:
            dbscan_balance = 0
        dbscan_metrics['cluster_balance_score'] = dbscan_balance

        dbscan_cluster_count_score = 100 - abs((dbscan_n_clusters - 5) * 10)
        dbscan_cluster_count_score = max(0, min(100, dbscan_cluster_count_score))
        dbscan_metrics['cluster_count_score'] = dbscan_cluster_count_score
        dbscan_metrics['cluster_count'] = dbscan_n_clusters

        noise_score = 100 - min(noise_percentage, 100)
        dbscan_metrics['noise_percentage'] = noise_percentage
        dbscan_metrics['noise_score'] = noise_score

        silhouette_weight = 40 if has_valid_dbscan_silhouette else 10
        dbscan_composite = (
            (dbscan_metrics.get('silhouette_score', 0) * silhouette_weight) +
            dbscan_metrics['davies_bouldin_normalized'] * 30 +
            dbscan_metrics.get('calinski_harabasz_normalized', 50) * 30
        ) / 100
        dbscan_metrics['composite_score'] = min(dbscan_composite, 100)

        metrics['kmeans'] = kmeans_metrics
        metrics['dbscan'] = dbscan_metrics

        # Decision rules
        if noise_percentage >= 30:
            best_method = 'kmeans'
            explainability_score = 100
            decision_rule = 'NOISE_THRESHOLD'
        elif kmeans_n_clusters < 3:
            best_method = 'dbscan'
            explainability_score = 100
            decision_rule = 'KMEANS_INSUFFICIENT_CLUSTERS'
        elif dbscan_n_clusters < 3:
            best_method = 'kmeans'
            explainability_score = 100
            decision_rule = 'DBSCAN_INSUFFICIENT_CLUSTERS'
        elif not has_valid_dbscan_silhouette:
            best_method = 'kmeans'
            explainability_score = 100
            decision_rule = 'DBSCAN_INSUFFICIENT_DATA'
        else:
            best_method = 'kmeans' if kmeans_metrics['composite_score'] >= dbscan_metrics['composite_score'] else 'dbscan'
            explainability_score = max(kmeans_metrics['composite_score'], dbscan_metrics['composite_score'])
            decision_rule = 'COMPOSITE_SCORE_COMPARISON'

        return {
            'best_method': best_method,
            'explainability_score': round(explainability_score, 2),
            'metrics': metrics,
            'decision_rule': decision_rule,
            'noise_percentage': noise_percentage
        }