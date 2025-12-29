"""
JOOLA Customer Analytics - Source Code Module
"""

from .data_processing import (
    standardize_column_names,
    clean_numeric_column,
    parse_dates,
    create_customer_features,
    create_churn_label,
    calculate_rfm_segments,
    get_top_products,
)

__version__ = "1.0.0"
__author__ = "Next Gen Consulting"

__all__ = [
    "standardize_column_names",
    "clean_numeric_column",
    "parse_dates",
    "create_customer_features",
    "create_churn_label",
    "calculate_rfm_segments",
    "get_top_products",
]
