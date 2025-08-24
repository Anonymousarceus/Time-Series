"""
Utility functions for the anomaly detection system.
"""

import os
import logging
import pandas as pd
from typing import Optional


def validate_inputs(input_path: str, output_path: str) -> None:
    """
    Validate input and output file paths.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to output CSV file
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Check if input file is CSV
    if not input_path.lower().endswith('.csv'):
        raise ValueError("Input file must be a CSV file")
    
    # Check if output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if output file is CSV
    if not output_path.lower().endswith('.csv'):
        raise ValueError("Output file must be a CSV file")


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level (str): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/anomaly_detection.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_sample_data() -> pd.DataFrame:
    """
    Generate sample multivariate time series data for testing.
    
    Returns:
        pd.DataFrame: Sample data
    """
    import numpy as np
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate 500 hours of data (more than required 439 hours)
    n_hours = 500
    n_features = 15
    
    # Create time-based features
    time_index = np.arange(n_hours)
    
    # Generate base patterns
    data = {}
    
    # Temperature-like features (with daily cycles)
    data['Temperature_1'] = 20 + 5 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.5, n_hours)
    data['Temperature_2'] = 22 + 4 * np.sin(2 * np.pi * time_index / 24 + np.pi/4) + np.random.normal(0, 0.4, n_hours)
    
    # Pressure features (correlated with temperature)
    data['Pressure_1'] = 1013 + 2 * data['Temperature_1'] + np.random.normal(0, 1, n_hours)
    data['Pressure_2'] = 1015 + 1.5 * data['Temperature_2'] + np.random.normal(0, 0.8, n_hours)
    
    # Flow rate features
    data['Flow_Rate_1'] = 50 + 3 * np.sin(2 * np.pi * time_index / 12) + np.random.normal(0, 2, n_hours)
    data['Flow_Rate_2'] = 45 + 2.5 * np.cos(2 * np.pi * time_index / 8) + np.random.normal(0, 1.5, n_hours)
    
    # Vibration features
    data['Vibration_1'] = 0.1 + 0.02 * np.random.normal(0, 1, n_hours)
    data['Vibration_2'] = 0.12 + 0.015 * np.random.normal(0, 1, n_hours)
    
    # Chemical composition features
    data['Chemical_A'] = 75 + 2 * np.sin(2 * np.pi * time_index / 48) + np.random.normal(0, 1, n_hours)
    data['Chemical_B'] = 25 - 0.5 * data['Chemical_A'] + np.random.normal(0, 0.5, n_hours)
    
    # Motor-related features
    data['Motor_Current'] = 10 + 0.1 * data['Flow_Rate_1'] + np.random.normal(0, 0.2, n_hours)
    data['Motor_Speed'] = 1800 + 5 * np.sin(2 * np.pi * time_index / 6) + np.random.normal(0, 10, n_hours)
    
    # Additional features
    data['Humidity'] = 60 + 10 * np.sin(2 * np.pi * time_index / 24 + np.pi/2) + np.random.normal(0, 2, n_hours)
    data['Power_Consumption'] = 100 + 0.5 * data['Motor_Current'] + 0.02 * data['Motor_Speed'] + np.random.normal(0, 2, n_hours)
    data['Efficiency'] = 85 + 2 * np.sin(2 * np.pi * time_index / 72) + np.random.normal(0, 1, n_hours)
    
    # Inject some anomalies in the later period (after training)
    anomaly_periods = [
        (150, 155),  # 5-hour anomaly
        (200, 203),  # 3-hour anomaly
        (300, 305),  # 5-hour anomaly
        (400, 402),  # 2-hour anomaly
    ]
    
    for start, end in anomaly_periods:
        if end < n_hours:
            # Inject different types of anomalies
            # Type 1: Threshold violation
            data['Temperature_1'][start:end] += 15  # Sudden temperature spike
            
            # Type 2: Relationship change
            data['Pressure_1'][start:end] -= 10  # Pressure drop while temp is high
            
            # Type 3: Pattern deviation
            data['Flow_Rate_1'][start:end] *= 0.5  # Flow reduction
            data['Vibration_1'][start:end] *= 3    # Vibration increase
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df


def save_sample_data(filename: str = 'TEP_Train_Test.csv') -> None:
    """
    Save sample data to CSV file.
    
    Args:
        filename (str): Output filename
    """
    df = load_sample_data()
    df.to_csv(filename, index=False)
    print(f"Sample data saved to {filename}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


def analyze_results(output_csv_path: str) -> None:
    """
    Analyze the results from anomaly detection.
    
    Args:
        output_csv_path (str): Path to the output CSV file
    """
    try:
        df = pd.read_csv(output_csv_path)
        
        print("\n" + "="*60)
        print("RESULTS ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print(f"Total data points: {len(df)}")
        
        if 'Abnormality_score' in df.columns:
            scores = df['Abnormality_score']
            print(f"Anomaly score statistics:")
            print(f"  Mean: {scores.mean():.2f}")
            print(f"  Std: {scores.std():.2f}")
            print(f"  Min: {scores.min():.2f}")
            print(f"  Max: {scores.max():.2f}")
            print(f"  Median: {scores.median():.2f}")
            
            # Count anomalies by severity
            normal = np.sum(scores <= 10)
            slight = np.sum((scores > 10) & (scores <= 30))
            moderate = np.sum((scores > 30) & (scores <= 60))
            significant = np.sum((scores > 60) & (scores <= 90))
            severe = np.sum(scores > 90)
            
            print(f"\nAnomaly distribution:")
            print(f"  Normal (0-10): {normal} ({normal/len(df)*100:.1f}%)")
            print(f"  Slight (11-30): {slight} ({slight/len(df)*100:.1f}%)")
            print(f"  Moderate (31-60): {moderate} ({moderate/len(df)*100:.1f}%)")
            print(f"  Significant (61-90): {significant} ({significant/len(df)*100:.1f}%)")
            print(f"  Severe (91-100): {severe} ({severe/len(df)*100:.1f}%)")
        
        # Analyze top contributing features
        feature_cols = [col for col in df.columns if col.startswith('top_feature_')]
        if feature_cols:
            print(f"\nTop contributing features analysis:")
            all_features = []
            for col in feature_cols:
                all_features.extend(df[col].dropna().tolist())
            
            # Remove empty strings
            all_features = [f for f in all_features if f]
            
            if all_features:
                from collections import Counter
                feature_counts = Counter(all_features)
                print("Most frequently contributing features:")
                for feature, count in feature_counts.most_common(10):
                    print(f"  {feature}: {count} times ({count/len(df)*100:.1f}%)")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error analyzing results: {str(e)}")


if __name__ == "__main__":
    # Generate and save sample data for testing
    save_sample_data()
