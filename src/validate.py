"""
Data Validation Module
Validates data quality, missing values, outliers, and business rules
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation"""
    
    def __init__(self):
        self.validation_rules = {
            'age': {'min': 18, 'max': 100, 'dtype': 'int64'},
            'tenure_months': {'min': 0, 'max': 120, 'dtype': 'int64'},
            'monthly_spend': {'min': 0, 'max': 10000, 'dtype': 'float64'},
            'support_tickets': {'min': 0, 'max': 50, 'dtype': 'int64'},
            'last_login_days': {'min': 0, 'max': 365, 'dtype': 'int64'},
            'satisfaction_score': {'min': 1, 'max': 10, 'dtype': 'int64'},
            'gender': {'allowed_values': ['Male', 'Female']},
            'contract_type': {'allowed_values': ['Monthly', 'Yearly']},
            'churn': {'allowed_values': [0, 1], 'dtype': 'int64'}
        }
        self.validation_results = {}
        
    def load_data(self, data_path: str = "artifacts/data/ingested_data.parquet") -> pd.DataFrame:
        """Load ingested data"""
        try:
            logger.info(f"Loading data from {data_path}")
            self.data = pd.read_parquet(data_path)
            logger.info(f"Loaded {len(self.data)} rows")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def check_missing_values(self) -> dict:
        """Check for missing values in all columns"""
        missing_counts = self.data.isnull().sum()
        missing_percentages = (missing_counts / len(self.data)) * 100
        
        # Convert numpy types to Python native types for JSON serialization
        result = {
            'total_missing': int(missing_counts.sum()),
            'columns_with_missing': {k: int(v) for k, v in missing_counts[missing_counts > 0].to_dict().items()},
            'missing_percentages': {k: float(v) for k, v in missing_percentages[missing_percentages > 0].to_dict().items()},
            'has_missing': bool(missing_counts.sum() > 0)  # Convert to Python bool
        }
        
        logger.info(f"Missing values check: {result['total_missing']} total missing values")
        return result
    
    def check_duplicates(self) -> dict:
        """Check for duplicate rows"""
        duplicate_count = self.data.duplicated().sum()
        
        # Check duplicate customer_ids
        customer_id_duplicates = self.data['customer_id'].duplicated().sum()
        
        result = {
            'duplicate_rows': int(duplicate_count),
            'duplicate_customer_ids': int(customer_id_duplicates),
            'has_duplicates': bool(duplicate_count > 0 or customer_id_duplicates > 0)  # Convert to Python bool
        }
        
        logger.info(f"Duplicates check: {duplicate_count} duplicate rows, {customer_id_duplicates} duplicate customer IDs")
        return result
    
    def check_outliers(self) -> dict:
        """Detect outliers using IQR method"""
        outliers = {}
        numerical_cols = ['age', 'tenure_months', 'monthly_spend', 'support_tickets', 
                         'last_login_days', 'satisfaction_score']
        
        for col in numerical_cols:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
                outliers_percentage = (outliers_count / len(self.data)) * 100
                
                outliers[col] = {
                    'count': int(outliers_count),
                    'percentage': float(outliers_percentage),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'Q1': float(Q1),
                    'Q3': float(Q3),
                    'IQR': float(IQR)
                }
        
        logger.info(f"Outlier detection completed for {len(outliers)} columns")
        return outliers
    
    def check_data_types(self) -> dict:
        """Validate data types"""
        type_validation = {}
        
        for col, rules in self.validation_rules.items():
            if col in self.data.columns:
                expected_type = rules.get('dtype')
                actual_type = str(self.data[col].dtype)
                
                type_validation[col] = {
                    'expected': expected_type,
                    'actual': actual_type,
                    'is_correct': bool(expected_type in actual_type if expected_type else True)  # Convert to Python bool
                }
        
        return type_validation
    
    def check_value_ranges(self) -> dict:
        """Validate value ranges for numerical columns"""
        range_validation = {}
        
        for col, rules in self.validation_rules.items():
            if col in self.data.columns and 'min' in rules and 'max' in rules:
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                
                out_of_range = ((self.data[col] < rules['min']) | (self.data[col] > rules['max'])).sum()
                
                range_validation[col] = {
                    'min_expected': rules['min'],
                    'min_actual': float(min_val),
                    'max_expected': rules['max'],
                    'max_actual': float(max_val),
                    'out_of_range_count': int(out_of_range),
                    'is_valid': bool(out_of_range == 0)  # Convert to Python bool
                }
        
        return range_validation
    
    def check_categorical_values(self) -> dict:
        """Validate categorical values"""
        categorical_validation = {}
        categorical_cols = ['gender', 'contract_type']
        
        for col in categorical_cols:
            if col in self.data.columns and col in self.validation_rules:
                allowed = self.validation_rules[col]['allowed_values']
                unique_values = self.data[col].unique().tolist()
                invalid_values = [val for val in unique_values if val not in allowed]
                
                categorical_validation[col] = {
                    'allowed_values': allowed,
                    'unique_values': unique_values,
                    'invalid_values': invalid_values,
                    'is_valid': bool(len(invalid_values) == 0)  # Convert to Python bool
                }
        
        # Check churn values
        if 'churn' in self.data.columns:
            allowed = [0, 1]
            unique_values = self.data['churn'].unique().tolist()
            invalid_values = [val for val in unique_values if val not in allowed]
            
            categorical_validation['churn'] = {
                'allowed_values': allowed,
                'unique_values': unique_values,
                'invalid_values': invalid_values,
                'is_valid': bool(len(invalid_values) == 0),  # Convert to Python bool
                'class_distribution': {int(k): int(v) for k, v in self.data['churn'].value_counts().to_dict().items()}  # Convert numpy types
            }
        
        return categorical_validation
    
    def check_statistical_balance(self) -> dict:
        """Check statistical balance of target variable"""
        if 'churn' not in self.data.columns:
            return {}
        
        churn_counts = self.data['churn'].value_counts()
        churn_percentages = (churn_counts / len(self.data)) * 100
        
        result = {
            'total_samples': int(len(self.data)),
            'churn_0_count': int(churn_counts.get(0, 0)),
            'churn_1_count': int(churn_counts.get(1, 0)),
            'churn_0_percentage': float(churn_percentages.get(0, 0)),
            'churn_1_percentage': float(churn_percentages.get(1, 0)),
            'is_balanced': bool(20 <= churn_percentages.get(1, 0) <= 80)  # Convert to Python bool
        }
        
        return result
    
    def run_full_validation(self) -> dict:
        """Run all validation checks"""
        logger.info("Starting comprehensive data validation")
        
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': int(len(self.data)),
            'total_columns': int(len(self.data.columns)),
            'missing_values': self.check_missing_values(),
            'duplicates': self.check_duplicates(),
            'outliers': self.check_outliers(),
            'data_types': self.check_data_types(),
            'value_ranges': self.check_value_ranges(),
            'categorical_values': self.check_categorical_values(),
            'statistical_balance': self.check_statistical_balance()
        }
        
        # Determine overall validation status
        issues = []
        
        if self.validation_results['missing_values']['has_missing']:
            issues.append("Missing values detected")
        
        if self.validation_results['duplicates']['has_duplicates']:
            issues.append("Duplicate records detected")
        
        for col, validation in self.validation_results['value_ranges'].items():
            if not validation['is_valid']:
                issues.append(f"Out of range values in {col}")
        
        for col, validation in self.validation_results['categorical_values'].items():
            if not validation.get('is_valid', True):
                issues.append(f"Invalid categorical values in {col}")
        
        self.validation_results['overall_status'] = {
            'is_valid': bool(len(issues) == 0),  # Convert to Python bool
            'issues_count': int(len(issues)),
            'issues_list': issues,
            'severity': 'HIGH' if len(issues) > 3 else 'MEDIUM' if len(issues) > 0 else 'LOW'
        }
        
        logger.info(f"Validation complete. Status: {'PASSED' if self.validation_results['overall_status']['is_valid'] else 'FAILED'}")
        if issues:
            logger.warning(f"Issues found: {issues}")
        
        return self.validation_results
    
    def save_validation_report(self, output_path: str = "artifacts/reports/validation_report.json"):
        """Save validation report"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Custom JSON encoder to handle any remaining numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.bool_):
                        return bool(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            
            with open(output_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"Validation report saved to {output_path}")
            
            # Also save a human-readable summary
            summary_path = output_path.replace('.json', '_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("DATA VALIDATION REPORT\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Validation Time: {self.validation_results['timestamp']}\n")
                f.write(f"Total Rows: {self.validation_results['total_rows']}\n")
                f.write(f"Total Columns: {self.validation_results['total_columns']}\n\n")
                
                f.write("MISSING VALUES:\n")
                f.write(f"  Total Missing: {self.validation_results['missing_values']['total_missing']}\n")
                f.write(f"  Has Missing: {self.validation_results['missing_values']['has_missing']}\n\n")
                
                f.write("DUPLICATES:\n")
                f.write(f"  Duplicate Rows: {self.validation_results['duplicates']['duplicate_rows']}\n")
                f.write(f"  Duplicate Customer IDs: {self.validation_results['duplicates']['duplicate_customer_ids']}\n\n")
                
                f.write("OVERALL STATUS:\n")
                f.write(f"  Valid: {self.validation_results['overall_status']['is_valid']}\n")
                f.write(f"  Issues Found: {self.validation_results['overall_status']['issues_count']}\n")
                if self.validation_results['overall_status']['issues_list']:
                    f.write("  Issues:\n")
                    for issue in self.validation_results['overall_status']['issues_list']:
                        f.write(f"    - {issue}\n")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving validation report: {str(e)}")
            raise

def main():
    """Main execution"""
    try:
        logger.info("Starting data validation")
        
        validator = DataValidator()
        validator.load_data()
        
        # Run validation
        results = validator.run_full_validation()
        
        # Print summary
        print("\n" + "="*60)
        print("DATA VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Rows: {results['total_rows']}")
        print(f"Total Columns: {results['total_columns']}")
        print(f"\nMissing Values: {results['missing_values']['total_missing']}")
        print(f"Duplicate Rows: {results['duplicates']['duplicate_rows']}")
        print(f"\nOverall Status: {'✅ PASSED' if results['overall_status']['is_valid'] else '❌ FAILED'}")
        print(f"Severity: {results['overall_status']['severity']}")
        
        if results['overall_status']['issues_list']:
            print("\nIssues Found:")
            for issue in results['overall_status']['issues_list']:
                print(f"  ⚠️ {issue}")
        
        # Print class distribution
        if 'statistical_balance' in results and results['statistical_balance']:
            sb = results['statistical_balance']
            print(f"\nTarget Variable Distribution:")
            print(f"  Churn=0: {sb['churn_0_count']} ({sb['churn_0_percentage']:.2f}%)")
            print(f"  Churn=1: {sb['churn_1_count']} ({sb['churn_1_percentage']:.2f}%)")
        
        # Save report
        validator.save_validation_report()
        
        # Exit with error if validation fails
        if not results['overall_status']['is_valid']:
            logger.error("Data validation failed")
            sys.exit(1)
        
        print("\n✅ Data validation completed successfully!")
        
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
