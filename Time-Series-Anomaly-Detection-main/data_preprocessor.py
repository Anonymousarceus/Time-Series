"""
Data Preprocessor Module

This module handles data loading, cleaning, validation, and preprocessing
for multivariate time series anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import datetime, timedelta
import logging


class DataPreprocessor:
    """
    Data preprocessing class for multivariate time series data.
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.logger = logging.getLogger(__name__)
        self.training_start = datetime(2004, 1, 1, 0, 0)
        self.training_end = datetime(2004, 1, 5, 23, 59)
        self.analysis_start = datetime(2004, 1, 1, 0, 0)
        self.analysis_end = datetime(2004, 1, 19, 7, 59)
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data_structure(self, df: pd.DataFrame) -> None:
        """
        Validate the structure of the input data.
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        # Check if dataframe is not empty
        if df.empty:
            raise ValueError("Input dataframe is empty")
        
        # Check minimum number of rows
        if len(df) < 72:  # Minimum 72 hours of data
            raise ValueError("Insufficient data: Require minimum 72 hours of training data")
        
        # Check for at least one numeric column
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found in the data")
        
        self.logger.info(f"Data validation passed. Numeric columns: {len(numeric_columns)}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Get numeric columns only
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        # Handle missing values
        missing_counts = df_clean[numeric_columns].isnull().sum()
        if missing_counts.sum() > 0:
            self.logger.warning(f"Found {missing_counts.sum()} missing values")
            # Forward fill then backward fill
            df_clean[numeric_columns] = df_clean[numeric_columns].fillna(method='ffill').fillna(method='bfill')
        
        # Replace infinite values with NaN then handle
        df_clean[numeric_columns] = df_clean[numeric_columns].replace([np.inf, -np.inf], np.nan)
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(method='ffill').fillna(method='bfill')
        
        # Handle constant features (zero variance)
        constant_features = []
        for col in numeric_columns:
            if df_clean[col].std() == 0 or df_clean[col].std() < 1e-10:
                constant_features.append(col)
        
        if constant_features:
            self.logger.warning(f"Found constant features: {constant_features}")
            # Add small noise to constant features
            for col in constant_features:
                df_clean[col] += np.random.normal(0, 1e-6, len(df_clean))
        
        self.logger.info("Data cleaning completed")
        return df_clean
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and analysis periods based on timestamps or indices.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training data and analysis data
        """
        # Since we don't have explicit timestamps, we'll use row indices
        # Assuming data is hourly and starts from 1/1/2004 0:00
        
        # Training period: 1/1/2004 0:00 to 1/5/2004 23:59 (120 hours)
        training_hours = 120  # 5 days * 24 hours
        
        # Analysis period: 1/1/2004 0:00 to 1/19/2004 7:59 (439 hours)
        analysis_hours = 439
        
        # Ensure we have enough data
        if len(df) < analysis_hours:
            self.logger.warning(f"Dataset has {len(df)} rows, but analysis requires {analysis_hours} hours")
            analysis_hours = len(df)
        
        # Get numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Split the data
        train_data = df[numeric_columns].iloc[:training_hours].copy()
        analysis_data = df[numeric_columns].iloc[:analysis_hours].copy()
        
        self.logger.info(f"Training data: {len(train_data)} rows")
        self.logger.info(f"Analysis data: {len(analysis_data)} rows")
        
        return train_data, analysis_data
    
    def normalize_features(self, train_data: pd.DataFrame, 
                          analysis_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize features using training data statistics.
        
        Args:
            train_data (pd.DataFrame): Training data
            analysis_data (pd.DataFrame): Analysis data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Normalized training and analysis data
        """
        # Calculate statistics from training data only
        train_mean = train_data.mean()
        train_std = train_data.std()
        
        # Avoid division by zero
        train_std = train_std.replace(0, 1)
        
        # Normalize both datasets using training statistics
        train_normalized = (train_data - train_mean) / train_std
        analysis_normalized = (analysis_data - train_mean) / train_std
        
        self.logger.info("Feature normalization completed")
        return train_normalized, analysis_normalized
