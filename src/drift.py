"""Data drift detection module."""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data drift between reference and current datasets."""
    
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold
    
    def load_data(self) -> tuple:
        """Load reference and current datasets."""
        reference_path = Path("artifacts/data/feature_data.csv")
        current_path = Path("artifacts/data/current_data.csv")
        
        if not reference_path.exists():
            logger.error("Reference data not found")
            return None, None
        
        if not current_path.exists():
            logger.warning("Current data not found, creating from reference")
            current_data = pd.read_csv(reference_path).sample(frac=0.8, random_state=42)
            current_data.to_csv(current_path, index=False)
        
        reference_df = pd.read_csv(reference_path)
        current_df = pd.read_csv(current_path)
        
        logger.info(f"Reference data: {len(reference_df)} rows")
        logger.info(f"Current data: {len(current_df)} rows")
        
        return reference_df, current_df
    
    def detect_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
        """Detect drift for all features."""
        logger.info("Detecting data drift...")
        
        drift_results = {}
        alerts = []
        
        # Numerical features
        numerical_features = ['age', 'tenure_months', 'monthly_spend', 
                              'support_tickets', 'last_login_days', 'satisfaction_score']
        
        for feature in numerical_features:
            if feature in reference_df.columns and feature in current_df.columns:
                ks_stat, p_value = stats.ks_2samp(
                    reference_df[feature].dropna(), 
                    current_df[feature].dropna()
                )
                drift_detected = p_value < 0.05
                drift_results[feature] = {
                    "type": "numerical",
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_value),
                    "drift_detected": drift_detected
                }
                if drift_detected:
                    alerts.append(f"Drift detected in {feature}")
        
        # Categorical features
        categorical_features = ['gender', 'contract_type']
        
        for feature in categorical_features:
            if feature in reference_df.columns and feature in current_df.columns:
                contingency = pd.crosstab(reference_df[feature], current_df[feature])
                chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
                drift_detected = p_value < 0.05
                drift_results[feature] = {
                    "type": "categorical",
                    "chi2_statistic": float(chi2_stat),
                    "p_value": float(p_value),
                    "drift_detected": drift_detected
                }
                if drift_detected:
                    alerts.append(f"Drift detected in {feature}")
        
        # Calculate overall drift score
        drift_scores = [v['ks_statistic'] for k, v in drift_results.items() if v['type'] == 'numerical']
        overall_score = np.mean(drift_scores) if drift_scores else 0
        
        return {
            "features": drift_results,
            "alerts": alerts,
            "drift_score": float(overall_score),
            "drift_threshold": self.threshold,
            "retraining_needed": overall_score > self.threshold,
            "timestamp": str(pd.Timestamp.now())
        }
    
    def run_analysis(self) -> dict:
        """Run complete drift analysis."""
        reference_df, current_df = self.load_data()
        
        if reference_df is None or current_df is None:
            return {"error": "Could not load data", "drift_score": 0, "retraining_needed": False}
        
        drift_report = self.detect_drift(reference_df, current_df)
        
        # Save report
        report_path = Path("artifacts/reports/drift_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(drift_report, f, indent=2)
        
        logger.info(f"Drift analysis complete. Score: {drift_report['drift_score']:.3f}")
        logger.info(f"Retraining needed: {drift_report['retraining_needed']}")
        
        return drift_report


def main():
    """Main drift detection entry point."""
    parser = argparse.ArgumentParser(description="Detect data drift")
    parser.add_argument("--full_analysis", action="store_true", help="Run full analysis")
    
    args = parser.parse_args()
    
    detector = DriftDetector()
    drift_report = detector.run_analysis()
    
    # Log summary
    logger.info("\n" + "="*50)
    logger.info("DRIFT DETECTION SUMMARY")
    logger.info("="*50)
    logger.info(f"Overall Drift Score: {drift_report.get('drift_score', 0):.3f}")
    logger.info(f"Retraining Needed: {drift_report.get('retraining_needed', False)}")
    
    for alert in drift_report.get('alerts', []):
        logger.warning(f"  - {alert}")


if __name__ == "__main__":
    main()
