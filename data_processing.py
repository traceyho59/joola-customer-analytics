"""
JOOLA Customer Analytics - Data Processing Utilities
=====================================================

This module contains helper functions for data cleaning, feature engineering,
and preprocessing used throughout the analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with any column naming convention
        
    Returns
    -------
    pd.DataFrame
        Dataframe with standardized column names
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace('.', '_', regex=False)
        .str.replace(' ', '_', regex=False)
        .str.replace('-', '_', regex=False)
    )
    return df


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Clean numeric columns by removing currency symbols and formatting.
    
    Parameters
    ----------
    series : pd.Series
        Series that may contain formatted numbers
        
    Returns
    -------
    pd.Series
        Cleaned numeric series
    """
    if series.dtype == 'object':
        # Remove currency symbols, commas, and parentheses (for negatives)
        cleaned = (
            series.astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace('%', '', regex=False)
            .str.replace('(', '-', regex=False)
            .str.replace(')', '', regex=False)
            .str.strip()
        )
        return pd.to_numeric(cleaned, errors='coerce')
    return series


def parse_dates(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """
    Parse date columns to datetime.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_columns : List[str]
        List of column names to parse as dates
        
    Returns
    -------
    pd.DataFrame
        Dataframe with parsed date columns
    """
    df = df.copy()
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def create_customer_features(df: pd.DataFrame, 
                              email_col: str = 'email',
                              date_col: str = 'order_date',
                              total_col: str = 'total',
                              items_col: str = 'lineitem_quantity',
                              discount_col: str = 'discount_amount',
                              marketing_col: str = 'accepts_marketing',
                              order_id_col: str = 'id') -> pd.DataFrame:
    """
    Aggregate order-level data to customer-level features for churn modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Order-level dataframe
    email_col : str
        Column name for customer email
    date_col : str
        Column name for order date
    total_col : str
        Column name for order total
    items_col : str
        Column name for line item quantity
    discount_col : str
        Column name for discount amount
    marketing_col : str
        Column name for marketing opt-in
    order_id_col : str
        Column name for order ID
        
    Returns
    -------
    pd.DataFrame
        Customer-level feature dataframe
    """
    # Aggregate to customer level
    customer_agg = df.groupby(email_col).agg(
        n_orders=(order_id_col, 'nunique'),
        first_purchase=(date_col, 'min'),
        last_purchase=(date_col, 'max'),
        avg_spend=(total_col, 'mean'),
        total_spend=(total_col, 'sum'),
        avg_items=(items_col, 'mean'),
        marketing_optin=(marketing_col, 'max'),
        n_discounts=(discount_col, lambda x: (x > 0).sum()),
        avg_discount=(discount_col, 'mean'),
    ).reset_index()
    
    # Calculate additional features
    customer_agg['frequency'] = customer_agg['n_orders']
    customer_agg['monetary'] = customer_agg['total_spend']
    
    # Calculate average gap between orders
    order_dates = df.groupby(email_col)[date_col].apply(
        lambda x: x.sort_values().diff().dt.days.mean()
    ).reset_index()
    order_dates.columns = [email_col, 'avg_gap_days']
    
    customer_agg = customer_agg.merge(order_dates, on=email_col, how='left')
    customer_agg['avg_gap_days'] = customer_agg['avg_gap_days'].fillna(0)
    
    return customer_agg


def create_churn_label(df: pd.DataFrame,
                       last_purchase_col: str = 'last_purchase',
                       observation_date: Optional[datetime] = None,
                       churn_days: int = 180) -> pd.DataFrame:
    """
    Create binary churn label based on recency.
    
    Parameters
    ----------
    df : pd.DataFrame
        Customer-level dataframe with last_purchase column
    last_purchase_col : str
        Column name for last purchase date
    observation_date : datetime, optional
        Date to calculate recency from (defaults to max date in data)
    churn_days : int
        Number of days without purchase to consider as churned
        
    Returns
    -------
    pd.DataFrame
        Dataframe with churn_label column added
    """
    df = df.copy()
    
    if observation_date is None:
        observation_date = df[last_purchase_col].max()
    
    df['recency_days'] = (observation_date - df[last_purchase_col]).dt.days
    df['churn_label'] = (df['recency_days'] >= churn_days).astype(int)
    
    return df


def calculate_rfm_segments(df: pd.DataFrame,
                           recency_col: str = 'recency_days',
                           frequency_col: str = 'frequency',
                           monetary_col: str = 'monetary',
                           n_segments: int = 4) -> pd.DataFrame:
    """
    Calculate RFM segments using quantile-based bucketing.
    
    Parameters
    ----------
    df : pd.DataFrame
        Customer-level dataframe
    recency_col : str
        Column name for recency
    frequency_col : str
        Column name for frequency
    monetary_col : str
        Column name for monetary value
    n_segments : int
        Number of segments for each dimension
        
    Returns
    -------
    pd.DataFrame
        Dataframe with RFM segment labels
    """
    df = df.copy()
    
    # Create quantile-based scores (lower recency is better, higher F/M is better)
    df['R_score'] = pd.qcut(df[recency_col], q=n_segments, labels=range(n_segments, 0, -1), duplicates='drop')
    df['F_score'] = pd.qcut(df[frequency_col], q=n_segments, labels=range(1, n_segments + 1), duplicates='drop')
    df['M_score'] = pd.qcut(df[monetary_col], q=n_segments, labels=range(1, n_segments + 1), duplicates='drop')
    
    # Combine into RFM segment
    df['RFM_score'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)
    
    return df


def get_top_products(df: pd.DataFrame,
                     product_col: str = 'lineitem_name',
                     metric_col: str = 'quantity',
                     top_n: int = 10,
                     aggregation: str = 'sum') -> pd.DataFrame:
    """
    Get top products by specified metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        Order-level dataframe
    product_col : str
        Column name for product identifier
    metric_col : str
        Column name for metric to aggregate
    top_n : int
        Number of top products to return
    aggregation : str
        Aggregation method ('sum', 'mean', 'count')
        
    Returns
    -------
    pd.DataFrame
        Top products with aggregated metric
    """
    agg_func = {'sum': 'sum', 'mean': 'mean', 'count': 'count'}
    
    top_products = (
        df.groupby(product_col)[metric_col]
        .agg(agg_func[aggregation])
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    
    top_products.columns = [product_col, f'{metric_col}_{aggregation}']
    return top_products


if __name__ == "__main__":
    # Example usage
    print("JOOLA Data Processing Utilities")
    print("=" * 40)
    print("Available functions:")
    print("  - standardize_column_names()")
    print("  - clean_numeric_column()")
    print("  - parse_dates()")
    print("  - create_customer_features()")
    print("  - create_churn_label()")
    print("  - calculate_rfm_segments()")
    print("  - get_top_products()")
