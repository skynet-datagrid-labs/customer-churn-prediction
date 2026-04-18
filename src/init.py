"""ML Pipeline for Company Churn Prediction."""

__version__ = "1.0.0"
__author__ = "ML Engineering Team"

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

__all__ = ['logger', '__version__']
