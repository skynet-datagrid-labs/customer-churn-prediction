# ML Company Workflow - Production ML Pipeline

## 🚀 Overview

A production-grade, fully automated ML pipeline for customer churn prediction. This repository demonstrates a complete MLOps workflow using GitHub Actions for automation, including data validation, feature engineering, model training, evaluation, deployment, monitoring, and retraining.

## 📊 Dataset

The pipeline uses customer data with the following features:
- **customer_id**: Unique identifier
- **age**: Customer age (18-100)
- **gender**: Male/Female
- **tenure_months**: Months as customer
- **monthly_spend**: Average monthly spending
- **contract_type**: Monthly or Yearly
- **support_tickets**: Number of support tickets
- **last_login_days**: Days since last login
- **satisfaction_score**: 1-10 satisfaction rating
- **churn**: Target variable (0=No churn, 1=Churn)

## 🏗️ Architecture

### Pipeline Stages
1. **Data Ingestion** - Load Excel data
2. **Data Validation** - Quality checks, missing values, outliers
3. **Preprocessing** - Handle missing values, encode categories
4. **Feature Engineering** - Create 10+ advanced features
5. **Model Training** - Parallel training of 3 models
6. **Evaluation** - Comprehensive metrics comparison
7. **Model Selection** - Choose best performing model
8. **Artifact Saving** - Save model, preprocessor, metadata
9. **API Deployment** - FastAPI for predictions
10. **Monitoring** - Drift detection, performance tracking
11. **Automated Retraining** - Scheduled retraining on drift

## 🛠️ Technologies

- **CI/CD**: GitHub Actions (15+ jobs)
- **ML Framework**: scikit-learn, XGBoost
- **API**: FastAPI, Uvicorn
- **Containerization**: Docker
- **Testing**: Pytest
- **Monitoring**: Drift detection (PSI, KS tests)

## 📦 Models Trained

1. **Logistic Regression** - Baseline linear model
2. **Random Forest** - Ensemble method
3. **XGBoost** - Gradient boosting

## 🔄 GitHub Actions Workflows

### Main Pipeline (`pipeline.yml`)
- 17 interconnected jobs
- Parallel model training
- Artifact passing between stages
- Automatic deployment simulation

### Scheduled Retraining (`retrain.yml`)
- Weekly automatic retraining
- Triggered by drift detection
- Model versioning

### API Testing (`api-test.yml`)
- Automated API endpoint testing
- Integration tests
- Performance validation

### Drift Detection (`drift-detection.yml`)
- Every 12 hours monitoring
- PSI and KS statistical tests
- Automatic retraining trigger

## 🚦 Getting Started

### Prerequisites
- GitHub account
- GitHub Actions enabled

### Running the Pipeline

1. **Push to trigger pipeline:**
```bash
git push origin main
