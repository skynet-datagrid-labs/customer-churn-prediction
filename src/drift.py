"""
Data Drift Detection Module
Detects distribution changes between reference and current data
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DriftDetector:
    """Detect data drift in features"""
    
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.drift_results = {}
        
    def load_reference_data(self, path="reference_data/features.pkl"):
        """Load reference data (training data)"""
        try:
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.reference_data = data['X']
            logger.info(f"Loaded reference data: {self.reference_data.shape}")
            return self.reference_data
        except:
            # Generate synthetic reference if not available
            logger.warning("Reference data not found, generating synthetic reference")
            self.reference_data = pd.DataFrame({
                'age': np.random.normal(45, 15, 1000),
                'tenure_months': np.random.exponential(30, 1000),
                'monthly_spend': np.random.gamma(5, 50, 1000),
                'support_tickets': np.random.poisson(3, 1000),
                'last_login_days': np.random.uniform(0, 60, 1000),
                'satisfaction_score': np.random.choice(range(1, 11), 1000)
            })
            return self.reference_data
    
    def load_current_data(self, path="current_data/current_data.csv"):
        """Load current data for drift detection"""
        try:
            self.current_data = pd.read_csv(path)
            logger.info(f"Loaded current data: {self.current_data.shape}")
            return self.current_data
        except:
            # Generate synthetic current data
            logger.warning("Current data not found, generating synthetic current data")
            drift_amount = 0.15
            self.current_data = pd.DataFrame({
                'age': np.random.normal(45 + drift_amount*10, 15, 1000),
                'tenure_months': np.random.exponential(30 * (1 + drift_amount), 1000),
                'monthly_spend': np.random.gamma(5, 50 * (1 + drift_amount*0.5), 1000),
                'support_tickets': np.random.poisson(3 + drift_amount*2, 1000),
                'last_login_days': np.random.uniform(0, 60 * (1 - drift_amount*0.3), 1000),
                'satisfaction_score': np.random.choice(range(1, 11), 1000)
            })
            return self.current_data
    
    def calculate_psi(self, expected, actual, buckets=10):
        """Calculate Population Stability Index"""
        # Discretize into buckets
        expected_percents = np.histogram(expected, bins=buckets, density=True)[0]
        actual_percents = np.histogram(actual, bins=buckets, density=True)[0]
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        return psi
    
    def calculate_ks(self, reference, current):
        """Calculate Kolmogorov-Smirnov statistic"""
        statistic, p_value = stats.ks_2samp(reference, current)
        return statistic, p_value
    
    def detect_drift_for_feature(self, feature_name, reference_values, current_values):
        """Detect drift for a single feature"""
        # PSI calculation
        psi = self.calculate_psi(reference_values, current_values)
        
        # KS test
        ks_statistic, ks_pvalue = self.calculate_ks(reference_values, current_values)
        
        # Determine drift status
        drift_detected = psi > self.threshold or ks_pvalue < 0.05
        
        return {
            'feature': feature_name,
            'psi_score': float(psi),
            'ks_statistic': float(ks_statistic),
            'ks_pvalue': float(ks_pvalue),
            'drift_detected': drift_detected,
            'severity': 'HIGH' if psi > 0.25 else 'MEDIUM' if psi > 0.1 else 'LOW',
            'reference_mean': float(np.mean(reference_values)),
            'current_mean': float(np.mean(current_values)),
            'reference_std': float(np.std(reference_values)),
            'current_std': float(np.std(current_values))
        }
    
    def detect_all_drifts(self):
        """Detect drift for all features"""
        common_features = list(set(self.reference_data.columns) & set(self.current_data.columns))
        
        for feature in common_features:
            if feature in self.reference_data.columns and feature in self.current_data.columns:
                result = self.detect_drift_for_feature(
                    feature,
                    self.reference_data[feature].dropna(),
                    self.current_data[feature].dropna()
                )
                self.drift_results[feature] = result
        
        # Overall drift assessment
        drift_features = [f for f, r in self.drift_results.items() if r['drift_detected']]
        
        overall_assessment = {
            'drift_detected': len(drift_features) > 0,
            'total_features': len(self.drift_results),
            'features_with_drift': len(drift_features),
            'drift_percentage': (len(drift_features) / len(self.drift_results)) * 100 if self.drift_results else 0,
            'severity': 'HIGH' if len(drift_features) > len(self.drift_results) * 0.3 else 'MEDIUM' if len(drift_features) > 0 else 'LOW',
            'affected_features': drift_features
        }
        
        return overall_assessment
    
    def save_drift_report(self, output_path="reports/drift_report.json"):
        """Save drift detection report"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        overall = self.detect_all_drifts()
        
        report = {
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'threshold': self.threshold,
            'overall_assessment': overall,
            'feature_drifts': self.drift_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Drift report saved to {output_path}")
        return report

def main():
    """Main execution"""
    try:
        logger.info("Starting drift detection")
        
        detector = DriftDetector(threshold=0.1)
        detector.load_reference_data()
        detector.load_current_data()
        report = detector.save_drift_report()
        
        print("\n" + "="*60)
        print("DATA DRIFT DETECTION REPORT")
        print("="*60)
        
        overall = report['overall_assessment']
        print(f"Drift Detected: {overall['drift_detected']}")
        print(f"Severity: {overall['severity']}")
        print(f"Features with Drift: {overall['features_with_drift']}/{overall['total_features']}")
        
        if overall['drift_detected']:
            print(f"\nAffected Features: {', '.join(overall['affected_features'])}")
            print("\nDetailed Drift Metrics:")
            for feature, metrics in report['feature_drifts'].items():
                if metrics['drift_detected']:
                    print(f"  {feature}: PSI={metrics['psi_score']:.4f}, "
                          f"KS p-value={metrics['ks_pvalue']:.4f}")
        
        print("\n✅ Drift detection completed!")
        
    except Exception as e:
        logger.error(f"Drift detection failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
