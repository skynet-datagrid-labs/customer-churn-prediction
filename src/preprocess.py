"""
Data Preprocessing Module
Handles missing values, encoding, scaling, and data cleaning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import json
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles all preprocessing steps"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
    def load_data(self, data_path: str = "artifacts/data/ingested_data.parquet") -> pd.DataFrame:
        """Load ingested data"""
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(self.data)} rows")
        return self.data
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        logger.info("Handling missing values")
        
        missing_before = self.data.isnull().sum().sum()
        
        # Numerical columns - fill with median
        numerical_cols = ['age', 'tenure_months', 'monthly_spend', 'support_tickets', 
                         'last_login_days', 'satisfaction_score']
        
        for col in numerical_cols:
            if col in self.data.columns and self.data[col].isnull().any():
                median_val = self.data[col].median()
                self.data[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        # Categorical columns - fill with mode
        categorical_cols = ['gender', 'contract_type']
        
        for col in categorical_cols:
            if col in self.data.columns and self.data[col].isnull().any():
                mode_val = self.data[col].mode()[0]
                self.data[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        missing_after = self.data.isnull().sum().sum()
        logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
        
        return missing_before - missing_after
    
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        """Remove outliers using IQR method"""
        if columns is None:
            columns = ['age', 'tenure_months', 'monthly_spend', 'support_tickets', 'last_login_days']
        
        rows_before = len(self.data)
        
        for col in columns:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap instead of remove for better data retention
                self.data[col] = self.data[col].clip(lower_bound, upper_bound)
                logger.info(f"Capped outliers in {col} to [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        rows_after = len(self.data)
        logger.info(f"Outlier capping completed. Rows: {rows_before} -> {rows_after}")
        
        return rows_before - rows_after
    
    def encode_categorical(self):
        """Encode categorical variables"""
        logger.info("Encoding categorical variables")
        
        categorical_cols = ['gender', 'contract_type']
        
        for col in categorical_cols:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])
                self.label_encoders[col] = le
                logger.info(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        return self.label_encoders
    
    def scale_features(self, feature_columns=None):
        """Scale numerical features"""
        if feature_columns is None:
            feature_columns = ['age', 'tenure_months', 'monthly_spend', 'support_tickets', 
                              'last_login_days', 'satisfaction_score']
        
        # Filter to only existing columns
        feature_columns = [col for col in feature_columns if col in self.data.columns]
        
        logger.info(f"Scaling features: {feature_columns}")
        
        # Fit and transform
        self.data[feature_columns] = self.scaler.fit_transform(self.data[feature_columns])
        self.is_fitted = True
        
        logger.info(f"Features scaled successfully")
        
        return feature_columns
    
    def create_preprocessing_pipeline(self):
        """Run complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline")
        
        # 1. Handle missing values
        missing_filled = self.handle_missing_values()
        
        # 2. Remove/cap outliers
        outliers_capped = self.remove_outliers_iqr()
        
        # 3. Encode categorical variables
        encoders = self.encode_categorical()
        
        # 4. Scale numerical features (will be done in feature engineering)
        
        logger.info("Preprocessing pipeline completed")
        
        return {
            'missing_values_filled': missing_filled,
            'outliers_capped': outliers_capped,
            'encoders': {k: v.classes_.tolist() for k, v in encoders.items()}
        }
    
    def save_preprocessed_data(self, output_path: str = "artifacts/data/preprocessed_data.pkl"):
        """Save preprocessed data and preprocessor objects"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save preprocessed data
            preprocessed_data = {
                'data': self.data,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'preprocessing_metadata': {
                    'is_fitted': self.is_fitted,
                    'num_rows': len(self.data),
                    'num_columns': len(self.data.columns),
                    'column_names': list(self.data.columns),
                    'data_types': self.data.dtypes.astype(str).to_dict()
                }
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(preprocessed_data, f)
            
            logger.info(f"Preprocessed data saved to {output_path}")
            
            # Also save a CSV version for inspection
            csv_path = output_path.replace('.pkl', '.csv')
            self.data.to_csv(csv_path, index=False)
            logger.info(f"CSV version saved to {csv_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {str(e)}")
            raise
    
    def generate_preprocessing_report(self):
        """Generate preprocessing report"""
        report = {
            'original_shape': (len(self.data), len(self.data.columns)),
            'missing_values_handled': True,
            'outliers_capped': True,
            'categorical_encoding': {col: le.classes_.tolist() for col, le in self.label_encoders.items()},
            'scaling_applied': self.is_fitted,
            'scaling_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaling_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'data_statistics': {
                col: {
                    'mean': float(self.data[col].mean()),
                    'std': float(self.data[col].std()),
                    'min': float(self.data[col].min()),
                    'max': float(self.data[col].max())
                } for col in self.data.select_dtypes(include=[np.number]).columns
            }
        }
        
        return report

def main():
    """Main execution"""
    try:
        logger.info("Starting data preprocessing")
        
        preprocessor = DataPreprocessor()
        preprocessor.load_data()
        
        # Run preprocessing
        results = preprocessor.create_preprocessing_pipeline()
        
        # Generate report
        report = preprocessor.generate_preprocessing_report()
        
        # Print summary
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Data shape: {report['original_shape']}")
        print(f"Missing values filled: {results['missing_values_filled']}")
        print(f"Outliers capped: {results['outliers_capped']}")
        print(f"Categorical encoders created: {len(results['encoders'])}")
        
        print("\nEncoded Categories:")
        for col, classes in results['encoders'].items():
            print(f"  {col}: {classes}")
        
        print("\nNumerical Features Statistics:")
        for col, stats in report['data_statistics'].items():
            if col not in ['customer_id', 'churn']:
                print(f"  {col}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                      f"min={stats['min']:.3f}, max={stats['max']:.3f}")
        
        # Save preprocessed data
        preprocessor.save_preprocessed_data()
        
        # Save preprocessing report
        report_path = "artifacts/reports/preprocessing_report.json"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n✅ Data preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
