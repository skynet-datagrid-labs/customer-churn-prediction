"""Data ingestion module."""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import logger

class DataIngestor:
    """Handle data ingestion from various sources."""
    
    def __init__(self, data_path: str = "data/fictitious_company_data.xlsx"):
        self.data_path = Path(data_path)
        self.raw_data_dir = Path("artifacts/data")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
    def ingest(self, retraining: bool = False) -> pd.DataFrame:
        """Load and return the dataset."""
        try:
            logger.info(f"Loading data from {self.data_path}")
            
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            # Load Excel file
            df = pd.read_excel(self.data_path, sheet_name=0)
            
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Basic info logging
            logger.info(f"Data types:\n{df.dtypes}")
            logger.info(f"Missing values:\n{df.isnull().sum()}")
            
            # Save raw data
            output_path = self.raw_data_dir / "raw_data.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Raw data saved to {output_path}")
            
            # Save data info
            info = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "retraining": retraining
            }
            
            import json
            info_path = self.raw_data_dir / "data_info.json"
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting data: {str(e)}")
            raise

def main():
    """Main entry point for data ingestion."""
    parser = argparse.ArgumentParser(description="Ingest company data")
    parser.add_argument("--retraining", action="store_true", help="Running in retraining mode")
    parser.add_argument("--save_reference", action="store_true", help="Save as reference dataset")
    parser.add_argument("--production", action="store_true", help="Load production data")
    
    args = parser.parse_args()
    
    ingestor = DataIngestor()
    df = ingestor.ingest(retraining=args.retraining)
    
    if args.save_reference:
        df.to_csv("artifacts/data/reference_data.csv", index=False)
        logger.info("Saved reference dataset")
    
    if args.production:
        df.to_csv("artifacts/data/current_data.csv", index=False)
        logger.info("Saved current production dataset")
    
    logger.info("Data ingestion completed successfully")

if __name__ == "__main__":
    main()
