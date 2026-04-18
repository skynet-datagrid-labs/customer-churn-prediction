"""Data drift detection module."""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import logger

class DriftDetector:
    """Detect data drift between reference and current datasets."""
    
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold
        self.drift_report = {}
        
    def load_data(self) -> tuple:
        """Load reference and current datasets."""
        reference_path = Path("artifacts/data/reference_data.csv")
        current_path = Path("artifacts/data/current_data.csv")
        
        if not reference_path.exists():
            logger.warning("Reference data not found, creating from raw data")
            raw_data = pd.read_csv("artifacts/data/raw_data.csv")
            reference_data = raw_data.sample(frac=0.7, random_state=42)
            reference_data.to_csv(reference_path, index=False)
        
        if not current_path.exists():
            logger.warning("Current data not found, creating from raw data")
            raw_data = pd.read_csv("artifacts/data/raw_data.csv")
            current_data = raw_data.drop(reference_data.index) if 'reference_data' in locals() else raw_data
            current_data.to_csv(current_path, index=False)
        
        reference_df = pd.read_csv(reference_path)
        current_df = pd.read_csv(current_path)
        
        logger.info(f"Reference data: {len(reference_df)} rows")
        logger.info(f"Current data: {len(current_df)} rows")
        
        return reference_df, current_df
    
    def detect_numerical_drift(self, ref_col: pd.Series, curr_col: pd.Series, col_name: str) -> Dict:
        """Detect drift in numerical features using KS test."""
        # Remove missing values
        ref_clean = ref_col.dropna()
        curr_clean = curr_col.dropna()
        
        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return {"drift_detected": False, "error": "Empty column"}
        
        # Perform Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(ref_clean, curr_clean)
        
        drift_detected = p_value < 0.05
        
        # Calculate distribution statistics
        ref_stats = {
            "mean": float(ref_clean.mean()),
            "std": float(ref_clean.std()),
            "min": float(ref_clean.min()),
            "max": float(ref_clean.max()),
            "q25": float(ref_clean.quantile(0.25)),
            "median": float(ref_clean.median()),
            "q75": float(ref_clean.quantile(0.75))
        }
        
        curr_stats = {
            "mean": float(curr_clean.mean()),
            "std": float(curr_clean.std()),
            "min": float(curr_clean.min()),
            "max": float(curr_clean.max()),
            "q25": float(curr_clean.quantile(0.25)),
            "median": float(curr_clean.median()),
            "q75": float(curr_clean.quantile(0.75))
        }
        
        return {
            "feature_type": "numerical",
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "drift_detected": drift_detected,
            "drift_severity": "high" if ks_stat > 0.3 else "medium" if ks_stat > 0.2 else "low",
            "reference_stats": ref_stats,
            "current_stats": curr_stats,
            "relative_change_mean": (curr_stats["mean"] - ref_stats["mean"]) / (abs(ref_stats["mean"]) + 1e-6)
        }
    
    def detect_categorical_drift(self, ref_col: pd.Series, curr_col: pd.Series, col_name: str) -> Dict:
        """Detect drift in categorical features using Chi-square test."""
        # Fill missing with 'Unknown'
        ref_clean = ref_col.fillna('Unknown')
        curr_clean = curr_col.fillna('Unknown')
        
        # Get value distributions
        ref_dist = ref_clean.value_counts(normalize=True)
        curr_dist = curr_clean.value_counts(normalize=True)
        
        # Calculate JS divergence for categorical
        all_categories = set(ref_dist.index) | set(curr_dist.index)
        ref_probs = np.array([ref_dist.get(cat, 0) for cat in all_categories])
        curr_probs = np.array([curr_dist.get(cat, 0) for cat in all_categories])
        
        # Jensen-Shannon divergence
        m = (ref_probs + curr_probs) / 2
        js_divergence = 0.5 * stats.entropy(ref_probs, m) + 0.5 * stats.entropy(curr_probs, m)
        
        # Chi-square test
        contingency = pd.crosstab(ref_clean, curr_clean)
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        drift_detected = p_value < 0.05 or js_divergence > 0.1
        
        return {
            "feature_type": "categorical",
            "chi2_statistic": float(chi2_stat),
            "p_value": float(p_value),
            "js_divergence": float(js_divergence),
            "drift_detected": drift_detected,
            "drift_severity": "high" if js_divergence > 0.2 else "medium" if js_divergence > 0.1 else "low",
            "reference_distribution": ref_dist.to_dict(),
            "current_distribution": curr_dist.to_dict(),
            "top_changes": self._get_top_category_changes(ref_dist, curr_dist)
        }
    
    def _get_top_category_changes(self, ref_dist, curr_dist, n=3):
        """Get categories with largest distribution changes."""
        all_cats = set(ref_dist.index) | set(curr_dist.index)
        changes = []
        for cat in all_cats:
            ref_val = ref_dist.get(cat, 0)
            curr_val = curr_dist.get(cat, 0)
            changes.append({
                "category": str(cat),
                "reference_proportion": float(ref_val),
                "current_proportion": float(curr_val),
                "change": float(curr_val - ref_val)
            })
        changes.sort(key=lambda x: abs(x['change']), reverse=True)
        return changes[:n]
    
    def detect_all_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict:
        """Detect drift for all features."""
        logger.info("Detecting data drift for all features...")
        
        drift_results = {}
        alerts = []
        
        # Define feature types
        numerical_features = ['age', 'tenure_months', 'monthly_spend', 
                              'support_tickets', 'last_login_days', 'satisfaction_score']
        categorical_features = ['gender', 'contract_type']
        
        # Check numerical features
        for feature in numerical_features:
            if feature in reference_df.columns and feature in current_df.columns:
                result = self.detect_numerical_drift(
                    reference_df[feature], current_df[feature], feature
                )
                drift_results[feature] = result
                if result['drift_detected']:
                    alerts.append(f"Drift detected in {feature} ({result['drift_severity']})")
        
        # Check categorical features
        for feature in categorical_features:
            if feature in reference_df.columns and feature in current_df.columns:
                result = self.detect_categorical_drift(
                    reference_df[feature], current_df[feature], feature
                )
                drift_results[feature] = result
                if result['drift_detected']:
                    alerts.append(f"Drift detected in {feature} ({result['drift_severity']})")
        
        # Calculate overall drift score
        drift_scores = []
        for feature, result in drift_results.items():
            if result['feature_type'] == 'numerical':
                drift_scores.append(result['ks_statistic'])
            else:
                drift_scores.append(result['js_divergence'])
        
        overall_score = np.mean(drift_scores) if drift_scores else 0
        
        return {
            "features": drift_results,
            "alerts": alerts,
            "drift_score": float(overall_score),
            "drift_threshold": self.threshold,
            "retraining_needed": overall_score > self.threshold or len(alerts) > 2,
            "timestamp": str(pd.Timestamp.now())
        }
    
    def run_full_analysis(self) -> Dict:
        """Run complete drift analysis."""
        reference_df, current_df = self.load_data()
        drift_report = self.detect_all_drift(reference_df, current_df)
        
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
    parser.add_argument("--threshold", type=float, default=0.2, help="Drift threshold")
    
    args = parser.parse_args()
    
    detector = DriftDetector(threshold=args.threshold)
    drift_report = detector.run_full_analysis()
    
    # Log summary
    logger.info("\n" + "="*50)
    logger.info("DRIFT DETECTION SUMMARY")
    logger.info("="*50)
    logger.info(f"Overall Drift Score: {drift_report['drift_score']:.3f}")
    logger.info(f"Threshold: {drift_report['drift_threshold']}")
    logger.info(f"Retraining Needed: {drift_report['retraining_needed']}")
    logger.info(f"Alerts: {len(drift_report['alerts'])}")
    
    for alert in drift_report['alerts']:
        logger.warning(f"  - {alert}")
    
    # Exit with appropriate code
    if drift_report['retraining_needed']:
        logger.info("Drift detected - retraining recommended")
        sys.exit(0)  # Success but indicate drift
    else:
        logger.info("No significant drift detected")
        sys.exit(0)

if __name__ == "__main__":
    main()
