"""Main theme analyzer that orchestrates the complete theme analysis pipeline."""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from .data_processing import DataProcessor
from .embeddings import EmbeddingProcessor
from ..Visualization.visualization import ClusterVisualizer
from .evaluation import ClusterEvaluator
from .topic_analysis import TopicAnalyzer
from ..cluster_method import ClusteringAnalyzer

logger = logging.getLogger(__name__)


class ThemeAnalyzer:
    """Complete theme analysis pipeline for complaint data."""

    def __init__(self, client=None):
        """Initialize ThemeAnalyzer.

        Args:
            client: OpenAI client for embeddings and topic analysis
        """
        self.client = client
        self.data_processor = DataProcessor()
        self.embedding_processor = EmbeddingProcessor(client)
        self.cluster_analyzer = ClusteringAnalyzer()
        self.visualizer = ClusterVisualizer()
        self.evaluator = ClusterEvaluator()
        self.topic_analyzer = TopicAnalyzer(client)

        logger.info("ThemeAnalyzer initialized with all components")

    def process_complaint_data(self, folder_path: str, pattern: str = "*.json") -> pd.DataFrame:
        """
        Process complaint journey analysis files.

        Args:
            folder_path: Path to folder containing JSON files
            pattern: File pattern to match

        Returns:
            Processed DataFrame
        """
        return self.data_processor.process_case_journey_folder(folder_path, pattern)

    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for text data.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            Embedding array
        """
        return self.embedding_processor.get_embeddings_in_batches(texts, batch_size=batch_size)

    def reduce_dimensions(self, embeddings: np.ndarray, method: str = "auto") -> tuple:
        """
        Apply dimension reduction to embeddings.

        Args:
            embeddings: Input embeddings
            method: Reduction method ('pca', 'umap', 'auto')

        Returns:
            Tuple of (reduced_embeddings, reduction_info)
        """
        return self.embedding_processor.apply_dimension_reduction(embeddings, method=method)

    def perform_clustering(self, embeddings: np.ndarray, method: str = "auto") -> Dict[str, Any]:
        """
        Perform clustering analysis.

        Args:
            embeddings: Input embeddings
            method: Clustering method ('kmeans', 'dbscan', 'leiden', 'auto')

        Returns:
            Clustering results
        """
        if method == "auto":
            # Try different methods and select best
            kmeans_result = self.cluster_analyzer.apply_kmeans_clustering(embeddings)
            dbscan_result = self.cluster_analyzer.apply_dbscan_clustering(embeddings)

            selection = self.cluster_analyzer.select_best_clustering_method(
                embeddings, kmeans_result, dbscan_result
            )

            if selection['best_method'] == 'kmeans':
                return kmeans_result
            else:
                return dbscan_result
        elif method == "kmeans":
            return self.cluster_analyzer.apply_kmeans_clustering(embeddings)
        elif method == "dbscan":
            return self.cluster_analyzer.apply_dbscan_clustering(embeddings)
        elif method == "leiden":
            return self.cluster_analyzer.apply_leiden_clustering(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def visualize_clusters(self, embeddings: np.ndarray, cluster_labels: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Visualize clustering results.

        Args:
            embeddings: Input embeddings
            cluster_labels: Cluster labels
            **kwargs: Additional visualization parameters

        Returns:
            Visualization data
        """
        return self.visualizer.cluster_visual(embeddings, cluster_labels=cluster_labels, **kwargs)

    def evaluate_clusters(self, embeddings: np.ndarray, labels: np.ndarray, model_name: str = "clusters") -> Dict[str, Any]:
        """
        Evaluate clustering results.

        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            model_name: Name for the model

        Returns:
            Evaluation metrics
        """
        return self.evaluator.evaluate_clustering_result(embeddings, labels, model_name)

    def extract_topics(self, embeddings: np.ndarray, cluster_labels: np.ndarray, texts: List[str]) -> Dict[str, Any]:
        """
        Extract topics from clusters.

        Args:
            embeddings: Input embeddings
            cluster_labels: Cluster labels
            texts: Original texts

        Returns:
            Topic analysis results
        """
        # Get representative points for each cluster
        cluster_payloads = self.topic_analyzer.get_top_n_closest_points_per_cluster(
            embeddings, cluster_labels, texts, top_n=15
        )

        # Extract topics using LLM
        raw_response = self.topic_analyzer.find_topics_all_clusters(cluster_payloads)

        return {
            'cluster_payloads': cluster_payloads,
            'raw_response': raw_response,
            'topics': self._parse_topic_response(raw_response)
        }

    def _parse_topic_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response for topics.

        Args:
            response: Raw LLM response

        Returns:
            Parsed topic data
        """
        try:
            # Try to parse as JSON
            import json
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse topic response as JSON: {response}")
            return []

    def run_complete_analysis(
        self,
        data_folder: str,
        text_column: str = "primary_complaint_issue_clean",
        clustering_method: str = "auto",
        reduce_dimensions: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete theme analysis pipeline.

        Args:
            data_folder: Path to folder containing complaint data
            text_column: Column name containing text to analyze
            clustering_method: Clustering method to use
            reduce_dimensions: Whether to apply dimension reduction

        Returns:
            Complete analysis results
        """
        logger.info("Starting complete theme analysis pipeline")

        # 1. Process data
        logger.info("Processing complaint data...")
        df = self.process_complaint_data(data_folder)
        texts = df[text_column].dropna().tolist()

        # 2. Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.generate_embeddings(texts)

        # 3. Optional dimension reduction
        if reduce_dimensions:
            logger.info("Reducing dimensions...")
            embeddings, reduction_info = self.reduce_dimensions(embeddings)
        else:
            reduction_info = None

        # 4. Perform clustering
        logger.info("Performing clustering...")
        clustering_result = self.perform_clustering(embeddings, method=clustering_method)

        # 5. Extract topics
        logger.info("Extracting topics...")
        topic_result = self.extract_topics(embeddings, clustering_result['labels'], texts)

        # 6. Evaluate results
        logger.info("Evaluating results...")
        evaluation = self.evaluate_clusters(embeddings, clustering_result['labels'])

        return {
            'data': df,
            'texts': texts,
            'embeddings': embeddings,
            'reduction_info': reduction_info,
            'clustering': clustering_result,
            'topics': topic_result,
            'evaluation': evaluation
        }