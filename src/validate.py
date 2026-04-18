"""Data validation module."""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validate dataset for quality and consistency."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.validation_rules = self.config.get('data', {}).get('validation_rules', {})
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}, using defaults")
            return {}
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate dataframe against rules."""
        logger.info("Starting data validation")
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
            "missing_values": {},
            "duplicates": 0
        }
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        validation_results["missing_values"] = missing_counts.to_dict()
        
        if len(missing_cols) > 0:
            validation_results["warnings"].append(
                f"Missing values found in columns: {missing_cols.index.tolist()}"
            )
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        validation_results["duplicates"] = int(duplicates)
        if duplicates > 0:
            validation_results["warnings"].append(f"Found {duplicates} duplicate rows")
        
        # Validate each column
        for column, rules in self.validation_rules.items():
            if column not in df.columns:
                validation_results["errors"].append(f"Required column '{column}' missing")
                validation_results["valid"] = False
                continue
            
            # Type validation
            if rules.get('required', False) and df[column].isnull().all():
                validation_results["errors"].append(f"Column '{column}' has all null values")
                validation_results["valid"] = False
            
            # Numeric range validation
            if 'min' in rules:
                invalid_min = df[column] < rules['min']
                if invalid_min.any():
                    validation_results["warnings"].append(
                        f"Column '{column}' has {invalid_min.sum()} values below {rules['min']}"
                    )
            
            if 'max' in rules:
                invalid_max = df[column] > rules['max']
                if invalid_max.any():
                    validation_results["warnings"].append(
                        f"Column '{column}' has {invalid_max.sum()} values above {rules['max']}"
                    )
            
            # Categorical validation
            if 'allowed_values' in rules:
                invalid_cats = ~df[column].isin(rules['allowed_values'])
                if invalid_cats.any() and not df[column].isnull().any():
                    invalid_values = df[column][invalid_cats].unique().tolist()
                    validation_results["warnings"].append(
                        f"Column '{column}' has invalid values: {invalid_values}"
                    )
            
            # Calculate statistics
            if pd.api.types.is_numeric_dtype(df[column]):
                validation_results["stats"][column] = {
                    "mean": float(df[column].mean()),
                    "std": float(df[column].std()),
                    "min": float(df[column].min()),
                    "max": float(df[column].max()),
                    "null_count": int(df[column].isnull().sum()),
                    "unique_count": int(df[column].nunique())
                }
            else:
                validation_results["stats"][column] = {
                    "null_count": int(df[column].isnull().sum()),
                    "unique_count": int(df[column].nunique()),
                    "top_values": df[column].value_counts().head(3).to_dict()
                }
        
        # Additional data quality checks
        if len(df) == 0:
            validation_results["errors"].append("Dataset is empty")
            validation_results["valid"] = False
        
        # Check target column
        target = 'churn'
        if target in df.columns:
            target_dist = df[target].value_counts(normalize=True).to_dict()
            validation_results["stats"]["target_distribution"] = target_dist
            logger.info(f"Target distribution: {target_dist}")
            
            if 0 not in target_dist or 1 not in target_dist:
                validation_results["errors"].append("Target column missing one of the classes")
                validation_results["valid"] = False
        
        # Log results
        if validation_results["valid"]:
            logger.info("Data validation passed")
        else:
            logger.error(f"Data validation failed: {validation_results['errors']}")
        
        for warning in validation_results["warnings"]:
            logger.warning(warning)
        
        return validation_results["valid"], validation_results
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column type."""
        logger.info("Handling missing values")
        
        for column in df.columns:
            if df[column].isnull().any():
                if pd.api.types.is_numeric_dtype(df[column]):
                    # Fill numeric with median
                    median_val = df[column].median()
                    df[column].fillna(median_val, inplace=True)
                    logger.info(f"Filled missing in '{column}' with median: {median_val}")
                else:
                    # Fill categorical with mode
                    mode_val = df[column].mode()
                    if len(mode_val) > 0:
                        df[column].fillna(mode_val[0], inplace=True)
                        logger.info(f"Filled missing in '{column}' with mode: {mode_val[0]}")
                    else:
                        df[column].fillna("Unknown", inplace=True)
        
        return df

def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="Validate data")
    parser.add_argument("--retraining", action="store_true", help="Running in retraining mode")
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path("artifacts/data/raw_data.csv")
    if not data_path.exists():
        logger.error(f"Raw data not found at {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows for validation")
    
    # Validate
    validator = DataValidator()
    is_valid, results = validator.validate(df)
    
    # Handle missing values
    df_cleaned = validator.handle_missing_values(df)
    
    # Save validation report
    report_path = Path("artifacts/reports/validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save validated data
    output_path = Path("artifacts/data/validated_data.csv")
    df_cleaned.to_csv(output_path, index=False)
    logger.info(f"Validated data saved to {output_path}")
    
    if not is_valid:
        logger.error("Data validation failed with critical errors")
        sys.exit(1)
    
    logger.info("Data validation completed successfully")

if __name__ == "__main__":
    main()
