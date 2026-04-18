"""Feature engineering module."""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import logger

class FeatureEngineer:
    """Create and transform features."""
    
    def __init__(self):
        self.feature_columns = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        logger.info("Starting feature engineering")
        
        df_feat = df.copy()
        
        # Create tenure to spend ratio
        if 'tenure_months' in df_feat.columns and 'monthly_spend' in df_feat.columns:
            df_feat['tenure_spend_ratio'] = df_feat['tenure_months'] / (df_feat['monthly_spend'] + 1)
            logger.info("Created 'tenure_spend_ratio' feature")
        
        # Create interaction between satisfaction and support tickets
        if 'satisfaction_score' in df_feat.columns and 'support_tickets' in df_feat.columns:
            df_feat['satisfaction_support_interaction'] = df_feat['satisfaction_score'] * df_feat['support_tickets']
            logger.info("Created 'satisfaction_support_interaction' feature")
        
        # Create age groups
        if 'age' in df_feat.columns:
            df_feat['age_group'] = pd.cut(df_feat['age'], 
                                          bins=[0, 25, 35, 45, 55, 100],
                                          labels=['Young', 'Early Career', 'Mid Career', 'Senior', 'Retirement'])
            logger.info("Created 'age_group' feature")
        
        # Create tenure groups
        if 'tenure_months' in df_feat.columns:
            df_feat['tenure_group'] = pd.cut(df_feat['tenure_months'],
                                             bins=[0, 12, 24, 36, 60, 120],
                                             labels=['New', 'Short-term', 'Medium-term', 'Long-term', 'Loyal'])
            logger.info("Created 'tenure_group' feature")
        
        # Create churn risk score based on multiple factors
        if all(col in df_feat.columns for col in ['support_tickets', 'last_login_days', 'satisfaction_score']):
            df_feat['churn_risk_score'] = (
                df_feat['support_tickets'] * 0.4 +
                df_feat['last_login_days'] * 0.3 +
                (10 - df_feat['satisfaction_score']) * 0.3
            )
            logger.info("Created 'churn_risk_score' feature")
        
        # Create monthly spend per tenure
        if 'monthly_spend' in df_feat.columns and 'tenure_months' in df_feat.columns:
            df_feat['lifetime_value'] = df_feat['monthly_spend'] * df_feat['tenure_months']
            logger.info("Created 'lifetime_value' feature")
        
        # Create inactivity score
        if 'last_login_days' in df_feat.columns:
            df_feat['inactivity_score'] = np.where(df_feat['last_login_days'] > 30, 1, 0)
            logger.info("Created 'inactivity_score' feature")
        
        # Create high spender flag
        if 'monthly_spend' in df_feat.columns:
            spend_threshold = df_feat['monthly_spend'].quantile(0.75)
            df_feat['high_spender'] = (df_feat['monthly_spend'] > spend_threshold).astype(int)
            logger.info(f"Created 'high_spender' feature (threshold: {spend_threshold:.2f})")
        
        # Create support intensive flag
        if 'support_tickets' in df_feat.columns:
            support_threshold = df_feat['support_tickets'].quantile(0.75)
            df_feat['support_intensive'] = (df_feat['support_tickets'] > support_threshold).astype(int)
            logger.info(f"Created 'support_intensive' feature (threshold: {support_threshold})")
        
        # Encode categorical features created
        categorical_created = ['age_group', 'tenure_group']
        for col in categorical_created:
            if col in df_feat.columns:
                # Convert to category codes
                df_feat[col] = df_feat[col].astype('category').cat.codes
                logger.info(f"Encoded '{col}' as categorical code")
        
        # Log feature engineering results
        self.feature_columns = [col for col in df_feat.columns if col != 'churn']
        logger.info(f"Created {len(self.feature_columns)} total features")
        logger.info(f"Feature list: {self.feature_columns}")
        
        return df_feat
    
    def get_feature_importance_info(self, model=None) -> dict:
        """Get feature importance information."""
        return {
            "feature_names": self.feature_columns,
            "n_features": len(self.feature_columns),
            "created_timestamp": str(pd.Timestamp.now())
        }

def main():
    """Main feature engineering entry point."""
    parser = argparse.ArgumentParser(description="Create features")
    parser.add_argument("--retraining", action="store_true", help="Running in retraining mode")
    
    args = parser.parse_args()
    
    # Load preprocessed data
    data_path = Path("artifacts/data/preprocessed_data.csv")
    if not data_path.exists():
        logger.error(f"Preprocessed data not found at {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows for feature engineering")
    
    # Create features
    engineer = FeatureEngineer()
    df_feat = engineer.create_features(df)
    
    # Save feature-engineered data
    output_path = Path("artifacts/data/feature_data.csv")
    df_feat.to_csv(output_path, index=False)
    logger.info(f"Feature data saved to {output_path}")
    
    # Save feature info
    import json
    feature_info = engineer.get_feature_importance_info()
    
    info_path = Path("artifacts/reports/feature_info_enhanced.json")
    with open(info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    logger.info("Feature engineering completed successfully")
    
    # Display feature statistics
    logger.info("Feature statistics:")
    for col in engineer.feature_columns[:5]:  # Show first 5
        logger.info(f"  {col}: mean={df_feat[col].mean():.3f}, std={df_feat[col].std():.3f}")

if __name__ == "__main__":
    main()
