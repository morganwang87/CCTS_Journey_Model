# If there is a specific error or issue
import pandas as pd
from openai import AzureOpenAI

import logging
from pathlib import Path
import pandas as pd
from openai import AzureOpenAI
from report_generation import InteractionAnalyzer, save_analysis_results

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Integration Test Entry Point
# -----------------------------------------------------------------------------
def test_full_report_generation_pipeline():
    logger.info("=" * 80)
    logger.info("STARTING FULL REPORT-GENERATION INTEGRATION TEST")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Azure OpenAI Client (REAL)
    # -------------------------------------------------------------------------
    client = AzureOpenAI(
        api_key="28f2878eb3244d7083072353c59d2bb7",
        api_version="2024-02-01",
        azure_endpoint="https://mazcaeprdmkencicognitiveopenai01.openai.azure.com/",
    )

    # -------------------------------------------------------------------------
    # Load Real Data
    # -------------------------------------------------------------------------
    pickle_path = (
        "/Workspace/Users/morgan.wang@rci.rogers.ca/"
        "share_workspeace/Interactions/(Clone) Jan2026_all_interactions.pkl"
    )

    logger.info(f"Loading pickle file: {pickle_path}")
    df = pd.read_pickle(pickle_path)

    df_relevancy = df[df["relevancy"] == "yes"].copy()
    logger.info(f"Total relevant interactions: {len(df_relevancy)}")

    if df_relevancy.empty:
        raise RuntimeError("No relevant interactions found — test cannot proceed")

    # -------------------------------------------------------------------------
    # Output Directories (REAL FILE OUTPUT)
    # -------------------------------------------------------------------------
    conversation_dir = Path(
        "/Workspace/Users/morgan.wang@rci.rogers.ca/"
        "share_workspeace/Levels_report_generation/Conversation_test"
    )
    journey_dir = Path(
        "/Workspace/Users/morgan.wang@rci.rogers.ca/"
        "share_workspeace/Levels_report_generation/Journey_test"
    )

    conversation_dir.mkdir(parents=True, exist_ok=True)
    journey_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Analyzer
    # -------------------------------------------------------------------------
    processing_config = {
        # add real config values if required
    }

    analyzer = InteractionAnalyzer(client, processing_config)

    # -------------------------------------------------------------------------
    # Process Cases (LIMIT FIRST RUN!)
    # -------------------------------------------------------------------------
    cases = df_relevancy["Case Number"].unique()
    logger.info(f"Found {len(cases)} unique cases")

    # 🔴 SAFETY: limit during testing
    cases_to_test = cases[:2]

    successful = 0
    failed = 0

    for idx, case_number in enumerate(cases_to_test, 1):
        logger.info(f"[{idx}/{len(cases_to_test)}] Processing case {case_number}")
        try:
            df_case = df_relevancy[df_relevancy["Case Number"] == case_number]

            # Step 1: Analyze interactions
            results = analyzer.analyze_all_interactions(df_case)

            if not results:
                raise RuntimeError("No analysis results produced")

            # Step 2: Generate journey summary
            summary = analyzer.generate_summary_report(results, case_number)

            # Step 3: Save outputs
            interaction_path, summary_path = save_analysis_results(
                results,
                summary,
                case_number,
                conversation_dir,
                journey_dir,
            )

            logger.info("✅ Case completed successfully")
            logger.info(f"   Interaction output: {interaction_path}")
            logger.info(f"   Journey output:     {summary_path}")

            successful += 1

        except Exception as e:
            logger.exception(f"❌ Case {case_number} FAILED: {e}")
            failed += 1

    # -------------------------------------------------------------------------
    # Final Assertion
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info(
        f"INTEGRATION TEST COMPLETE | "
        f"Successful: {successful}, Failed: {failed}"
    )
    logger.info("=" * 80)

    if failed > 0:
        raise AssertionError("One or more cases failed during integration test")
