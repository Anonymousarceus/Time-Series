#!/usr/bin/env python3
"""
Multivariate Time Series Anomaly Detection System

This module provides a comprehensive solution for detecting anomalies in multivariate
time series data and identifying the primary contributing features for each anomaly.

Author: Anomaly Detection Team
Date: August 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from anomaly_detector import AnomalyDetector
from data_preprocessor import DataPreprocessor
from feature_attribution import FeatureAttributor
from utils import validate_inputs, setup_logging

import logging

def main(input_csv_path: str, output_csv_path: str) -> None:
    """
    Main function to detect anomalies in multivariate time series data.
    
    Args:
        input_csv_path (str): Path to the input CSV file
        output_csv_path (str): Path to save the output CSV file with anomaly scores
    
    Returns:
        None
    """
    # Setup logging
    logger = setup_logging()
    logger.info("Starting multivariate time series anomaly detection")
    
    try:
        # Validate inputs
        validate_inputs(input_csv_path, output_csv_path)
        
        # Initialize components
        preprocessor = DataPreprocessor()
        detector = AnomalyDetector()
        attributor = FeatureAttributor()
        
        # Load and preprocess data
        logger.info(f"Loading data from {input_csv_path}")
        df = preprocessor.load_data(input_csv_path)
        
        # Validate data structure
        preprocessor.validate_data_structure(df)
        
        # Handle missing values and outliers
        df_clean = preprocessor.clean_data(df)
        
        # Split data into training and analysis periods
        train_data, analysis_data = preprocessor.split_data(df_clean)
        logger.info(f"Training data shape: {train_data.shape}")
        logger.info(f"Analysis data shape: {analysis_data.shape}")
        
        # Train anomaly detection model
        logger.info("Training anomaly detection model")
        detector.fit(train_data)
        
        # Detect anomalies in the full analysis period
        logger.info("Detecting anomalies")
        anomaly_scores = detector.predict(analysis_data)
        
        # Calculate feature attributions
        logger.info("Calculating feature attributions")
        feature_attributions = attributor.calculate_attributions(
            detector, analysis_data, anomaly_scores
        )
        
        # Convert scores to 0-100 scale
        normalized_scores = detector.normalize_scores(anomaly_scores)
        
        # Create output dataframe
        output_df = analysis_data.copy()
        output_df['Abnormality_score'] = normalized_scores
        
        # Add top contributing features
        for i in range(7):
            col_name = f'top_feature_{i+1}'
            output_df[col_name] = [
                attr[i] if i < len(attr) else '' 
                for attr in feature_attributions
            ]
        
        # Save results
        logger.info(f"Saving results to {output_csv_path}")
        output_df.to_csv(output_csv_path, index=False)
        
        # Validation and summary
        training_period_scores = normalized_scores[:len(train_data)]
        mean_training_score = np.mean(training_period_scores)
        max_training_score = np.max(training_period_scores)
        
        logger.info(f"Training period - Mean score: {mean_training_score:.2f}, Max score: {max_training_score:.2f}")
        logger.info(f"Analysis complete. Results saved to {output_csv_path}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("ANOMALY DETECTION SUMMARY")
        print("="*60)
        print(f"Total data points analyzed: {len(output_df)}")
        print(f"Training period mean score: {mean_training_score:.2f}")
        print(f"Training period max score: {max_training_score:.2f}")
        print(f"Overall mean anomaly score: {np.mean(normalized_scores):.2f}")
        print(f"Number of high anomalies (>60): {np.sum(normalized_scores > 60)}")
        print(f"Number of severe anomalies (>90): {np.sum(normalized_scores > 90)}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_csv_path> <output_csv_path>")
        print("Example: python main.py TEP_Train_Test.csv output_anomalies.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    main(input_path, output_path)
