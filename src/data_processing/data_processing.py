"""Data processing utilities for theme analysis."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Union
import json
import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data extraction and processing for complaint journey analysis."""

    @staticmethod
    def safe_get(dct: Dict, *keys: str, default=None) -> Any:
        """
        Safely navigate nested dictionaries.

        Args:
            dct: Dictionary to navigate
            *keys: Keys to traverse
            default: Default value if key not found

        Returns:
            Value at the nested key path or default
        """
        current = dct
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current

    @staticmethod
    def extract_case_journey_analysis(data_or_file: Union[Dict, str, Path]) -> Dict[str, Any]:
        """
        Extract all relevant fields from a case-level complaint journey analysis JSON.

        Parameters
        ----------
        data_or_file : dict or str or Path
            Either:
            - a loaded Python dictionary containing the JSON data
            - or a path to a JSON file

        Returns
        -------
        dict
            Flattened case-level record
        """
        try:
            # Load JSON if a file path is provided
            if isinstance(data_or_file, (str, Path)):
                file_path = Path(data_or_file)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                file_name = file_path.name
            else:
                data = data_or_file
                file_name = None

            ccts_journey = DataProcessor.safe_get(data, "ccts_complaint_journey_analysis", default={}) or {}
            customer_complaint_genesis = DataProcessor.safe_get(ccts_journey, "customer_complaint_genesis", default={}) or {}
            response_assessment = DataProcessor.safe_get(ccts_journey, "response_assessment", default={}) or {}
            journey_failure_points = DataProcessor.safe_get(ccts_journey, "journey_failure_points", default={}) or {}

            value_gap_analysis = DataProcessor.safe_get(data, "value_gap_analysis", default={}) or {}
            rationality_assessment = DataProcessor.safe_get(value_gap_analysis, "rationality_assessment", default={}) or {}

            prevention_opportunity_analysis = DataProcessor.safe_get(data, "prevention_opportunity_analysis", default={}) or {}

            resolution_recommendations = DataProcessor.safe_get(data, "resolution_recommendations", default={}) or {}
            root_cause_identification = DataProcessor.safe_get(resolution_recommendations, "root_cause_identification", default={}) or {}

            record = {
                # File metadata
                "file_name": file_name,

                # -----------------------------
                # ccts_complaint_journey_analysis
                # -----------------------------
                'case_number': ccts_journey.get("case_number"),

                # customer_complaint_genesis
                "primary_complaint_issue": customer_complaint_genesis.get("primary_complaint_issue"),
                "issue_evolution": customer_complaint_genesis.get("issue_evolution"),
                "unresolved_issues": customer_complaint_genesis.get("unresolved_issues", []),

                # response_assessment
                "solutions_offered": response_assessment.get("solutions_offered", []),
                "implementation_gaps": response_assessment.get("implementation_gaps", []),
                "consistency_of_handling": response_assessment.get("consistency_of_handling"),

                # journey_failure_points
                "critical_breakdown_moments": journey_failure_points.get("critical_breakdown_moments", []),
                "repeat_contact_pattern": journey_failure_points.get("repeat_contact_pattern"),
                "final_straw_incident": journey_failure_points.get("final_straw_incident"),

                # -----------------------------
                # value_gap_analysis
                # -----------------------------
                "offer_vs_expectation_matrix": value_gap_analysis.get("offer_vs_expectation_matrix"),

                # rationality_assessment
                "customer_demand_rationality": rationality_assessment.get("customer_demand_rationality"),
                "customer_demand_rationality_justification": rationality_assessment.get("customer_demand_rationality_justification"),
                "company_offer_adequacy": rationality_assessment.get("company_offer_adequacy"),
                "company_offer_adequacy_justification": rationality_assessment.get("company_offer_adequacy_justification"),

                # -----------------------------
                # prevention_opportunity_analysis
                # -----------------------------
                "proactive_outreach": prevention_opportunity_analysis.get("proactive_outreach", []),
                "compensation_timing": prevention_opportunity_analysis.get("compensation_timing", []),
                "escalation_management": prevention_opportunity_analysis.get("escalation_management", []),

                # -----------------------------
                # resolution_recommendations
                # -----------------------------
                # root_cause_identification
                "primary_root_cause": root_cause_identification.get("primary_root_cause"),
                "contributing_factors": root_cause_identification.get("contributing_factors", []),
                "systemic_vs_individual": root_cause_identification.get("systemic_vs_individual"),
                "cause_explanation": root_cause_identification.get("cause_explanation"),
                "evidence_base": root_cause_identification.get("evidence_base", []),

                # strategic recommendations
                "strategic_recommendations": resolution_recommendations.get("strategic_recommendations", []),

                # -----------------------------
                # Optional raw blocks for traceability
                # -----------------------------
                "ccts_complaint_journey_analysis_raw": ccts_journey,
                "value_gap_analysis_raw": value_gap_analysis,
                "prevention_opportunity_analysis_raw": prevention_opportunity_analysis,
                "resolution_recommendations_raw": resolution_recommendations,
            }

            return record

        except Exception as e:
            logging.error(f"Error extracting case journey analysis: {e}")
            return {}

    @staticmethod
    def process_case_journey_folder(folder_path: Union[str, Path], pattern: str = "*.json") -> pd.DataFrame:
        """
        Process all case journey analysis JSON files in a folder and return a DataFrame.

        Args:
            folder_path: Path to folder containing JSON files
            pattern: Glob pattern for files to process

        Returns:
            DataFrame with extracted records
        """
        records = []

        for file_path in Path(folder_path).glob(pattern):
            record = DataProcessor.extract_case_journey_analysis(file_path)
            if record:
                records.append(record)

        return pd.DataFrame(records)