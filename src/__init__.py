"""ML Pipeline for Company Churn Prediction.

This package provides a complete machine learning pipeline for customer churn prediction,
including data ingestion, validation, preprocessing, feature engineering, model training,
evaluation, selection, and deployment.
"""

__version__ = "1.0.0"
__author__ = "ML Engineering Team"
__license__ = "Proprietary"

import logging
from typing import Dict, Any, List, Optional, Tuple

# Configure logging for the entire package
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create package-level logger
logger = logging.getLogger(__name__)

# Define public API
__all__ = [
    'logger',
    '__version__',
    '__author__',
    '__license__'
]

# Package metadata
PACKAGE_INFO = {
    "name": "ml-company-workflow",
    "version": __version__,
    "author": __author__,
    "description": "Production ML pipeline for customer churn prediction",
    "python_requires": ">=3.9"
}

def get_package_info() -> Dict[str, str]:
    """Return package information."""
    return PACKAGE_INFO.copy()

def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration for the package.
    
    Args:
        level: Logging level (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.setLevel(level)
    logger.info(f"Logging configured with level: {logging.getLevelName(level)}")
