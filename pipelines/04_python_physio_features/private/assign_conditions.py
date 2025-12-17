"""
Condition Assignment Utility

Assigns condition labels to physiological data based on conditions.yaml filters.
Replicates logic from xdf_to_set pipeline.

Author: VR-TSST Project
Date: December 2025
"""

import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, Any


def load_conditions_config(config_path: str = "config/conditions.yaml") -> Dict:
    """Load conditions configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['conditions']


def match_condition(row: pd.Series, filters: Dict[str, Any]) -> bool:
    """
    Check if a row matches all filters for a condition.
    
    Parameters
    ----------
    row : Series
        Data row to check
    filters : dict
        Filter specifications from conditions.yaml
        
    Returns
    -------
    bool
        True if row matches all filters
    """
    for col, expected_val in filters.items():
        if col == 'duration':  # Skip duration metadata
            continue
            
        if col not in row.index:
            return False
        
        actual_val = row[col]
        
        # Handle list of acceptable values
        if isinstance(expected_val, list):
            if actual_val not in expected_val:
                return False
        else:
            if actual_val != expected_val:
                return False
    
    return True


def assign_conditions_to_dataframe(
    df: pd.DataFrame,
    conditions_config: Dict[str, Dict],
    condition_col: str = 'Condition'
) -> pd.DataFrame:
    """
    Assign condition labels to dataframe based on conditions.yaml filters.
    
    Parameters
    ----------
    df : DataFrame
        Physiological data with columns like Unity Scene, Study_Phase, etc.
    conditions_config : dict
        Conditions configuration from conditions.yaml
    condition_col : str
        Name of column to create with condition labels
        
    Returns
    -------
    DataFrame
        DataFrame with added Condition column
    """
    df = df.copy()
    df[condition_col] = None
    
    # Apply each condition's filters
    for condition_name, filters in conditions_config.items():
        # Create boolean mask for this condition
        mask = df.apply(lambda row: match_condition(row, filters), axis=1)
        
        # Assign condition label where mask is True
        df.loc[mask, condition_col] = condition_name
        
        matched_count = mask.sum()
        if matched_count > 0:
            logging.debug(f"  Matched {matched_count} rows to condition: {condition_name}")
    
    # Log unmatched rows
    unmatched = df[condition_col].isna().sum()
    if unmatched > 0:
        logging.warning(f"  {unmatched} rows did not match any condition")
    
    return df
