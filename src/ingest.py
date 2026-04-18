"""
Data Ingestion Module
Loads company data from Excel file with validation
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIngestor:
    """Handles data ingestion from Excel files"""
    
    def __init__(self, file_path: str = "data/ficticious_company_data.xlsx"):
        self.file_path = file_path
        self.data = None
        self.metadata = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load data from Excel file"""
        try:
            logger.info(f"Loading data from {self.file_path}")
            
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"Data file not found: {self.file_path}")
            
            # Load Excel file
            self.data = pd.read_excel(self.file_path, engine='openpyxl')
            
            # Store metadata
            self.metadata = {
                'load_time': datetime.now().isoformat(),
                'file_path': self.file_path,
                'num_rows': len(self.data),
                'num_columns': len(self.data.columns),
                'columns': list(self.data.columns),
                'dtypes': self.data.dtypes.astype(str).to_dict(),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            logger.info(f"Successfully loaded {len(self.data)} rows with {len(self.data.columns)} columns")
            logger.info(f"Columns: {list(self.data.columns)}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_columns(self) -> dict:
        """Validate required columns exist"""
        expected_columns = [
            'customer_id', 'age', 'gender', 'tenure_months', 'monthly_spend',
            'contract_type', 'support_tickets', 'last_login_days', 
            'satisfaction_score', 'churn'
        ]
        
        missing_columns = [col for col in expected_columns if col not in self.data.columns]
        
        validation_result = {
            'is_valid': len(missing_columns) == 0,
            'missing_columns': missing_columns,
            'present_columns': [col for col in expected_columns if col in self.data.columns],
            'extra_columns': [col for col in self.data.columns if col not in expected_columns]
        }
        
        if validation_result['is_valid']:
            logger.info("All required columns present")
        else:
            logger.warning(f"Missing columns: {missing_columns}")
            
        return validation_result
    
    def save_ingested_data(self, output_path: str = "artifacts/data/"):
        """Save ingested data for next stages"""
        try:
            # Create directory if it doesn't exist
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            # Save data as parquet for efficiency
            output_file = os.path.join(output_path, "ingested_data.parquet")
            self.data.to_parquet(output_file, index=False)
            logger.info(f"Saved ingested data to {output_file}")
            
            # Save metadata
            metadata_file = os.path.join(output_path, "ingestion_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_file}")
            
            # Save column validation
            validation = self.validate_columns()
            validation_file = os.path.join(output_path, "column_validation.json")
            with open(validation_file, 'w') as f:
                json.dump(validation, f, indent=2)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving ingested data: {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        logger.info("Starting data ingestion process")
        
        # Initialize ingestor
        ingestor = DataIngestor()
        
        # Load data
        data = ingestor.load_data()
        
        # Display basic info
        print("\n" + "="*50)
        print("DATA INGESTION SUMMARY")
        print("="*50)
        print(f"Total rows: {len(data)}")
        print(f"Total columns: {len(data.columns)}")
        print(f"Memory usage: {ingestor.metadata['memory_usage_mb']:.2f} MB")
        print(f"\nFirst 5 rows:")
        print(data.head())
        print(f"\nData types:")
        print(data.dtypes)
        print(f"\nMissing values:")
        print(data.isnull().sum())
        
        # Validate columns
        validation = ingestor.validate_columns()
        if not validation['is_valid']:
            logger.error("Column validation failed")
            sys.exit(1)
        
        # Save data
        ingestor.save_ingested_data()
        
        print("\n✅ Data ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
