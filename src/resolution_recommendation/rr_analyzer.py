import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI

# Import standardized modules from the project structure
from data_processing.data_processing import DataProcessor
from data_processing.embeddings import EmbeddingProcessor
from cluster_method import ClusteringAnalyzer

# Import local specialized components
from .analyzer import ResolutionRecommendationAnalyzer as RRTopicAnalyzer
from .prompts import build_topic_extraction_prompt
from data_processing.utils import safe_json_loads

logger = logging.getLogger(__name__)

class ResolutionRecommendationAnalyzer:
    """
    End-to-end analyzer for Strategic Resolution Recommendations.
    
    Optimized Pipeline using standardized project modules:
    1. Load JSONs -> Extract Recommendations (via DataProcessor + Custom Extraction)
    2. Generate Embeddings (via EmbeddingProcessor)
    3. Reduce Dimensions & Cluster (via ClusteringAnalyzer)
    4. Extract Topics via LLM (via RRTopicAnalyzer)
    """

    def __init__(self, azure_client: AzureOpenAI, llm_model: str = "gpt-4o"):
        self.llm_client = azure_client
        self.llm_model = llm_model
        
        # Initialize standardized processors
        self.embedding_processor = EmbeddingProcessor(azure_client)
        self.clustering_analyzer = ClusteringAnalyzer()
        self.data_processor = DataProcessor()
        
        # Initialize specialized topic analyzer for Resolution Recommendations
        self.topic_analyzer = RRTopicAnalyzer(client=azure_client, model=llm_model)

    def _load_resolution_data(self, data_folder: str) -> pd.DataFrame:
        """
        Load and process resolution recommendation data from JSON files.
        
        Uses DataProcessor to load raw JSONs, then extracts and explodes 
        'strategic_recommendations' into a flat DataFrame suitable for clustering.
        """
        logger.info(f"Loading raw data from {data_folder}")
        # Load raw case journey data
        df_raw = self.data_processor.process_case_journey_folder(data_folder)
        
        if df_raw.empty:
            raise ValueError("No valid data found in data folder.")
        
        # Check if strategic_recommendations exists
        if 'strategic_recommendations' not in df_raw.columns:
            raise KeyError("Column 'strategic_recommendations' not found in loaded data.")
            
        # Explode the list column to get one recommendation per row
        df_exploded = df_raw.explode('strategic_recommendations')
        df_exploded = df_exploded.dropna(subset=['strategic_recommendations'])
        
        # Ensure text is string
        df_exploded['recommendation_text'] = df_exploded['strategic_recommendations'].astype(str)
        
        logger.info(f"Loaded and exploded {len(df_exploded)} recommendations.")
        return df_exploded

    def run_pipeline(
        self, 
        data_folder: str, 
        clustering_method: str = "auto",
        dim_reduction_method: str = "umap",
        top_n_representatives: int = 15,
        normalize_embeddings: bool = True,
        **clustering_params
    ) -> Dict[str, Any]:
        """
        Execute the full analysis pipeline using standardized components.
        """
        logger.info("Starting Resolution Recommendation Analysis Pipeline")
        
        # 1. Load Data
        df = self._load_resolution_data(data_folder)
        texts = df["recommendation_text"].tolist()
        
        if not texts:
            raise ValueError("No valid recommendation texts found after processing.")

        # 2. Embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_processor.get_embeddings_in_batches(texts)
        
        # 3. Normalization (Optional but recommended for cosine similarity)
        if normalize_embeddings:
            logger.info("Normalizing embeddings...")
            embeddings = self.embedding_processor.normalize_embeddings(embeddings)

        # 4. Dimension Reduction
        logger.info(f"Applying dimension reduction ({dim_reduction_method})...")
        reduced_embeddings, reduction_info = self.embedding_processor.apply_dimension_reduction(
            embeddings, method=dim_reduction_method
        )

        # 5. Clustering
        logger.info(f"Running clustering algorithm: {clustering_method}...")
        clustering_result = self.clustering_analyzer.apply_clustering(
            reduced_embeddings, 
            method=clustering_method, 
            **clustering_params
        )
        
        final_labels = clustering_result['labels']
        logger.info(f"Clustering complete. Method: {clustering_result['method']}, Clusters: {clustering_result['n_clusters']}")

        # 6. Topic Extraction using Specialized RR Analyzer
        logger.info("Extracting topics via LLM...")
        # We use the specialized analyzer which has the correct prompts for RR
        topic_results = self.topic_analyzer.analyze_recommendations(
            embedding_vectors=reduced_embeddings,
            cluster_labels=final_labels,
            original_texts=texts,
            top_n=top_n_representatives
        )
        topics = topic_results['topics']

        # 7. Assemble Results
        results_df = df.copy()
        results_df["cluster_label"] = final_labels
        
        # Merge topics
        topics_df = pd.DataFrame(topics)
        if not topics_df.empty:
            topics_df["cluster_label"] = topics_df["label"].astype(int)
            results_df = results_df.merge(
                topics_df[["cluster_label", "topic", "description", "short_example"]], 
                on="cluster_label", 
                how="left"
            )

        return {
            "dataframe": results_df,
            "topics": topics,
            "clustering_method": clustering_result['method'],
            "clustering_metrics": clustering_result.get('metrics', {}),
            "embeddings": reduced_embeddings,
            "labels": final_labels,
            "reduction_info": reduction_info
        }