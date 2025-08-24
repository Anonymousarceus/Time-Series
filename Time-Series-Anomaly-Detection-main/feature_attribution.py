"""
Feature Attribution Module

This module calculates which features contribute most to each anomaly
detection result.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging


class FeatureAttributor:
    """
    Feature attribution calculator for anomaly detection results.
    """
    
    def __init__(self):
        """Initialize the FeatureAttributor."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_attributions(self, detector, analysis_data: pd.DataFrame, 
                             anomaly_scores: np.ndarray) -> List[List[str]]:
        """
        Calculate feature attributions for each anomaly.
        
        Args:
            detector: Trained anomaly detector
            analysis_data (pd.DataFrame): Analysis data
            anomaly_scores (np.ndarray): Anomaly scores
            
        Returns:
            List[List[str]]: List of top contributing features for each sample
        """
        self.logger.info("Calculating feature attributions")
        
        # Get feature importance from the detector
        feature_importance = detector.get_feature_importance(analysis_data)
        feature_names = detector.feature_names
        
        attributions = []
        
        for i in range(len(analysis_data)):
            # Get importance scores for this sample
            sample_importance = feature_importance[i]
            
            # Create feature-importance pairs
            feature_importance_pairs = list(zip(feature_names, sample_importance))
            
            # Sort by importance (descending)
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Filter features that contribute more than 1%
            max_importance = max(sample_importance) if max(sample_importance) > 0 else 1
            threshold = max_importance * 0.01  # 1% threshold
            
            significant_features = [
                feature for feature, importance in feature_importance_pairs
                if importance >= threshold
            ]
            
            # Take top 7 features
            top_features = significant_features[:7]
            
            # Handle ties using alphabetical order
            if len(significant_features) > 7:
                # Check if there are ties at the boundary
                boundary_importance = feature_importance_pairs[6][1]
                tied_features = [
                    feature for feature, importance in feature_importance_pairs
                    if abs(importance - boundary_importance) < 1e-10
                ]
                
                if len(tied_features) > 1:
                    tied_features.sort()  # Alphabetical order
                    # Replace the last feature if it's part of a tie
                    if top_features[6] in tied_features:
                        top_features[6] = tied_features[0]
            
            # Pad with empty strings if fewer than 7 features
            while len(top_features) < 7:
                top_features.append('')
            
            attributions.append(top_features)
        
        self.logger.info("Feature attribution calculation completed")
        return attributions
    
    def analyze_feature_patterns(self, attributions: List[List[str]], 
                                feature_names: List[str]) -> Dict[str, int]:
        """
        Analyze patterns in feature attributions.
        
        Args:
            attributions (List[List[str]]): Feature attributions for each sample
            feature_names (List[str]): All feature names
            
        Returns:
            Dict[str, int]: Count of how often each feature appears in top contributors
        """
        feature_counts = {feature: 0 for feature in feature_names}
        
        for attribution in attributions:
            for feature in attribution:
                if feature and feature in feature_counts:
                    feature_counts[feature] += 1
        
        return feature_counts
    
    def get_anomaly_explanation(self, sample_index: int, attributions: List[List[str]], 
                               analysis_data: pd.DataFrame, 
                               anomaly_score: float) -> str:
        """
        Generate a human-readable explanation for a specific anomaly.
        
        Args:
            sample_index (int): Index of the sample to explain
            attributions (List[List[str]]): Feature attributions
            analysis_data (pd.DataFrame): Analysis data
            anomaly_score (float): Anomaly score for this sample
            
        Returns:
            str: Human-readable explanation
        """
        if sample_index >= len(attributions):
            return "Invalid sample index"
        
        top_features = [f for f in attributions[sample_index] if f]
        
        if not top_features:
            return f"Anomaly score: {anomaly_score:.2f} - No significant contributing features identified"
        
        severity = self._get_severity_level(anomaly_score)
        
        explanation = f"Anomaly score: {anomaly_score:.2f} ({severity})\n"
        explanation += f"Primary contributing features: {', '.join(top_features[:3])}\n"
        
        if len(top_features) > 3:
            explanation += f"Additional contributors: {', '.join(top_features[3:])}\n"
        
        # Add feature values for context
        sample_values = analysis_data.iloc[sample_index]
        feature_values = [f"{feature}: {sample_values[feature]:.3f}" 
                         for feature in top_features[:3] if feature in sample_values.index]
        
        if feature_values:
            explanation += f"Feature values: {', '.join(feature_values)}"
        
        return explanation
    
    def _get_severity_level(self, score: float) -> str:
        """Get severity level description for anomaly score."""
        if score <= 10:
            return "Normal"
        elif score <= 30:
            return "Slightly unusual"
        elif score <= 60:
            return "Moderate anomaly"
        elif score <= 90:
            return "Significant anomaly"
        else:
            return "Severe anomaly"
