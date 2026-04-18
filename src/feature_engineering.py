"""
Feature Engineering Module
Creates new features, transformations, and prepares final feature set
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering and transformation"""
    
    def __init__(self):
        self.feature_columns = None
        self.created_features = []
        
    def load_preprocessed_data(self, data_path: str = "artifacts/data/preprocessed_data.pkl"):
        """Load preprocessed data"""
        logger.info(f"Loading preprocessed data from {data_path}")
        
        with open(data_path, 'rb') as f:
            preprocessed = pickle.load(f)
        
        self.data = preprocessed['data']
        self.scaler = preprocessed.get('scaler')
        self.label_encoders = preprocessed.get('label_encoders', {})
        
        logger.info(f"Loaded {len(self.data)} rows with {len(self.data.columns)} columns")
        return self.data
    
    def create_engagement_features(self):
        """Create features related to customer engagement"""
        logger.info("Creating engagement features")
        
        # Engagement score (combination of tenure and login frequency)
        if 'tenure_months' in self.data.columns and 'last_login_days' in self.data.columns:
            # Normalize tenure and login recency
            max_tenure = self.data['tenure_months'].max()
            max_login = self.data['last_login_days'].max()
            
            self.data['engagement_score'] = (
                (self.data['tenure_months'] / max_tenure) * 0.6 + 
                (1 - self.data['last_login_days'] / max_login) * 0.4
            )
            self.created_features.append('engagement_score')
            logger.info(f"Created engagement_score feature")
        
        # Activity ratio (support tickets per tenure month)
        if 'support_tickets' in self.data.columns and 'tenure_months' in self.data.columns:
            self.data['tickets_per_month'] = self.data['support_tickets'] / (self.data['tenure_months'] + 1)
            self.created_features.append('tickets_per_month')
            logger.info(f"Created tickets_per_month feature")
        
        return self.data
    
    def create_spending_features(self):
        """Create features related to customer spending"""
        logger.info("Creating spending features")
        
        # Spending efficiency (spend per tenure month)
        if 'monthly_spend' in self.data.columns and 'tenure_months' in self.data.columns:
            self.data['spend_per_tenure'] = self.data['monthly_spend'] / (self.data['tenure_months'] + 1)
            self.created_features.append('spend_per_tenure')
            logger.info(f"Created spend_per_tenure feature")
        
        # Contract value indicator
        if 'contract_type' in self.data.columns and 'monthly_spend' in self.data.columns:
            # Yearly contracts might have different spending patterns
            self.data['is_yearly_contract'] = (self.data['contract_type'] == 1).astype(int)
            self.created_features.append('is_yearly_contract')
            logger.info(f"Created is_yearly_contract feature")
        
        return self.data
    
    def create_risk_features(self):
        """Create risk indicator features"""
        logger.info("Creating risk features")
        
        # High risk indicator (multiple risk factors)
        risk_factors = []
        
        if 'support_tickets' in self.data.columns:
            high_tickets = (self.data['support_tickets'] > 5).astype(int)
            risk_factors.append(high_tickets)
            logger.info("Added high support tickets as risk factor")
        
        if 'satisfaction_score' in self.data.columns:
            low_satisfaction = (self.data['satisfaction_score'] < 4).astype(int)
            risk_factors.append(low_satisfaction)
            logger.info("Added low satisfaction as risk factor")
        
        if 'last_login_days' in self.data.columns:
            inactive = (self.data['last_login_days'] > 30).astype(int)
            risk_factors.append(inactive)
            logger.info("Added inactivity as risk factor")
        
        if risk_factors:
            self.data['risk_score'] = np.mean(risk_factors, axis=0)
            self.created_features.append('risk_score')
            logger.info(f"Created risk_score feature from {len(risk_factors)} factors")
        
        return self.data
    
    def create_interaction_features(self):
        """Create interaction features between important variables"""
        logger.info("Creating interaction features")
        
        # Age group and contract type interaction
        if 'age' in self.data.columns and 'contract_type' in self.data.columns:
            # Create age groups
            self.data['age_group'] = pd.cut(self.data['age'], 
                                           bins=[0, 30, 45, 60, 100], 
                                           labels=[0, 1, 2, 3]).astype(int)
            self.created_features.append('age_group')
            
            # Interaction between age group and contract type
            self.data['age_contract_interaction'] = self.data['age_group'] * self.data['contract_type']
            self.created_features.append('age_contract_interaction')
            logger.info(f"Created age group and interaction features")
        
        # Tenure and satisfaction interaction
        if 'tenure_months' in self.data.columns and 'satisfaction_score' in self.data.columns:
            self.data['tenure_satisfaction'] = self.data['tenure_months'] * self.data['satisfaction_score']
            self.created_features.append('tenure_satisfaction')
            logger.info(f"Created tenure_satisfaction interaction feature")
        
        return self.data
    
    def create_polynomial_features(self):
        """Create polynomial features for important numerical columns"""
        logger.info("Creating polynomial features")
        
        polynomial_cols = ['monthly_spend', 'support_tickets', 'satisfaction_score']
        
        for col in polynomial_cols:
            if col in self.data.columns:
                # Square term
                squared_col = f"{col}_squared"
                self.data[squared_col] = self.data[col] ** 2
                self.created_features.append(squared_col)
                
                # Log transform for skewed features
                if col == 'monthly_spend':
                    log_col = f"{col}_log"
                    self.data[log_col] = np.log1p(self.data[col])
                    self.created_features.append(log_col)
                
                logger.info(f"Created polynomial features for {col}")
        
        return self.data
    
    def scale_features(self):
        """Scale all numerical features"""
        # Get all numerical features (excluding target and ID)
        exclude_cols = ['customer_id', 'churn']
        feature_cols = [col for col in self.data.columns 
                       if col not in exclude_cols 
                       and self.data[col].dtype in ['int64', 'float64']]
        
        logger.info(f"Scaling {len(feature_cols)} features")
        
        # Initialize scaler if not exists
        if not hasattr(self, 'scaler') or self.scaler is None:
            self.scaler = StandardScaler()
        
        # Scale features
        self.data[feature_cols] = self.scaler.fit_transform(self.data[feature_cols])
        
        self.feature_columns = feature_cols
        logger.info(f"Features scaled successfully")
        
        return feature_cols
    
    def prepare_features_for_training(self):
        """Complete feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline")
        
        # Create all feature groups
        self.create_engagement_features()
        self.create_spending_features()
        self.create_risk_features()
        self.create_interaction_features()
        self.create_polynomial_features()
        
        # Scale features
        feature_cols = self.scale_features()
        
        # Separate features and target
        if 'churn' in self.data.columns:
            self.X = self.data[feature_cols]
            self.y = self.data['churn']
        else:
            self.X = self.data[feature_cols]
            self.y = None
        
        logger.info(f"Feature engineering completed. Created {len(self.created_features)} new features")
        logger.info(f"Final feature set: {len(feature_cols)} features")
        
        return self.X, self.y
    
    def save_features(self, output_path: str = "artifacts/data/features.pkl"):
        """Save engineered features"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            features_data = {
                'X': self.X,
                'y': self.y,
                'feature_columns': self.feature_columns,
                'created_features': self.created_features,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'metadata': {
                    'num_samples': len(self.X),
                    'num_features': len(self.feature_columns),
                    'feature_names': self.feature_columns,
                    'target_distribution': self.y.value_counts().to_dict() if self.y is not None else None
                }
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(features_data, f)
            
            logger.info(f"Features saved to {output_path}")
            
            # Save feature importance metadata
            metadata_path = output_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'feature_columns': self.feature_columns,
                    'num_features': len(self.feature_columns),
                    'created_features': self.created_features,
                    'feature_types': {col: str(self.X[col].dtype) for col in self.feature_columns}
                }, f, indent=2)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise

def main():
    """Main execution"""
    try:
        logger.info("Starting feature engineering")
        
        engineer = FeatureEngineer()
        engineer.load_preprocessed_data()
        
        # Run feature engineering
        X, y = engineer.prepare_features_for_training()
        
        # Print summary
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Original data shape: {engineer.data.shape}")
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape if y is not None else 'N/A'}")
        print(f"\nNew features created: {len(engineer.created_features)}")
        print(f"Created features: {', '.join(engineer.created_features)}")
        print(f"\nFinal feature columns: {len(engineer.feature_columns)}")
        
        print("\nFeature Statistics:")
        for col in engineer.feature_columns[:5]:  # Show first 5 features
            print(f"  {col}: mean={X[col].mean():.3f}, std={X[col].std():.3f}, "
                  f"min={X[col].min():.3f}, max={X[col].max():.3f}")
        
        if len(engineer.feature_columns) > 5:
            print(f"  ... and {len(engineer.feature_columns) - 5} more features")
        
        # Save features
        engineer.save_features()
        
        print("\n✅ Feature engineering completed successfully!")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
