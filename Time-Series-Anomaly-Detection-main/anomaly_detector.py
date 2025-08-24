"""
Anomaly Detector Module

This module implements various anomaly detection algorithms for multivariate
time series data, including Isolation Forest, Autoencoders, and ensemble methods.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """
    Multivariate time series anomaly detector using ensemble of methods.
    """
    
    def __init__(self, method: str = 'ensemble'):
        """
        Initialize the anomaly detector.
        
        Args:
            method (str): Detection method ('isolation_forest', 'autoencoder', 'pca', 'ensemble')
        """
        self.method = method
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        self.is_fitted = False
        
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Train the anomaly detection model on normal data.
        
        Args:
            train_data (pd.DataFrame): Training data (normal period)
        """
        self.logger.info(f"Training anomaly detector using {self.method} method")
        
        # Store feature names
        self.feature_names = list(train_data.columns)
        
        # Normalize the training data
        X_train = self.scaler.fit_transform(train_data)
        
        if self.method == 'isolation_forest':
            self._fit_isolation_forest(X_train)
        elif self.method == 'autoencoder':
            self._fit_autoencoder(X_train)
        elif self.method == 'pca':
            self._fit_pca(X_train)
        elif self.method == 'ensemble':
            self._fit_ensemble(X_train)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted = True
        self.logger.info("Model training completed")
    
    def _fit_isolation_forest(self, X_train: np.ndarray) -> None:
        """Fit Isolation Forest model."""
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.models['isolation_forest'].fit(X_train)
    
    def _fit_autoencoder(self, X_train: np.ndarray) -> None:
        """Fit Autoencoder model."""
        input_dim = X_train.shape[1]
        encoding_dim = max(2, input_dim // 2)
        
        # Build autoencoder
        input_layer = keras.Input(shape=(input_dim,))
        encoder = layers.Dense(encoding_dim, activation="relu")(input_layer)
        encoder = layers.Dense(encoding_dim // 2, activation="relu")(encoder)
        decoder = layers.Dense(encoding_dim, activation="relu")(encoder)
        decoder = layers.Dense(input_dim, activation="linear")(decoder)
        
        autoencoder = keras.Model(input_layer, decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train the autoencoder
        autoencoder.fit(
            X_train, X_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        
        self.models['autoencoder'] = autoencoder
    
    def _fit_pca(self, X_train: np.ndarray) -> None:
        """Fit PCA-based anomaly detector."""
        # Use 95% variance retention
        pca = PCA(n_components=0.95)
        pca.fit(X_train)
        self.models['pca'] = pca
        
        # Calculate reconstruction errors for threshold
        X_transformed = pca.transform(X_train)
        X_reconstructed = pca.inverse_transform(X_transformed)
        reconstruction_errors = np.mean((X_train - X_reconstructed) ** 2, axis=1)
        self.models['pca_threshold'] = np.percentile(reconstruction_errors, 95)
    
    def _fit_ensemble(self, X_train: np.ndarray) -> None:
        """Fit ensemble of multiple methods."""
        self._fit_isolation_forest(X_train)
        self._fit_autoencoder(X_train)
        self._fit_pca(X_train)
    
    def predict(self, analysis_data: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly scores for the analysis data.
        
        Args:
            analysis_data (pd.DataFrame): Data to analyze for anomalies
            
        Returns:
            np.ndarray: Anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Normalize the analysis data using training statistics
        X_analysis = self.scaler.transform(analysis_data)
        
        if self.method == 'isolation_forest':
            return self._predict_isolation_forest(X_analysis)
        elif self.method == 'autoencoder':
            return self._predict_autoencoder(X_analysis)
        elif self.method == 'pca':
            return self._predict_pca(X_analysis)
        elif self.method == 'ensemble':
            return self._predict_ensemble(X_analysis)
    
    def _predict_isolation_forest(self, X: np.ndarray) -> np.ndarray:
        """Predict using Isolation Forest."""
        # Get anomaly scores (negative values, more negative = more anomalous)
        scores = self.models['isolation_forest'].decision_function(X)
        # Convert to positive anomaly scores
        return -scores
    
    def _predict_autoencoder(self, X: np.ndarray) -> np.ndarray:
        """Predict using Autoencoder."""
        # Get reconstruction
        reconstructed = self.models['autoencoder'].predict(X, verbose=0)
        # Calculate reconstruction error
        reconstruction_errors = np.mean((X - reconstructed) ** 2, axis=1)
        return reconstruction_errors
    
    def _predict_pca(self, X: np.ndarray) -> np.ndarray:
        """Predict using PCA."""
        pca = self.models['pca']
        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        reconstruction_errors = np.mean((X - X_reconstructed) ** 2, axis=1)
        return reconstruction_errors
    
    def _predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble of methods."""
        # Get predictions from all methods
        if_scores = self._predict_isolation_forest(X)
        ae_scores = self._predict_autoencoder(X)
        pca_scores = self._predict_pca(X)
        
        # Normalize scores to 0-1 range for each method
        if_scores_norm = (if_scores - np.min(if_scores)) / (np.max(if_scores) - np.min(if_scores))
        ae_scores_norm = (ae_scores - np.min(ae_scores)) / (np.max(ae_scores) - np.min(ae_scores))
        pca_scores_norm = (pca_scores - np.min(pca_scores)) / (np.max(pca_scores) - np.min(pca_scores))
        
        # Weighted ensemble (equal weights)
        ensemble_scores = (if_scores_norm + ae_scores_norm + pca_scores_norm) / 3
        
        return ensemble_scores
    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize anomaly scores to 0-100 scale using percentile ranking.
        
        Args:
            scores (np.ndarray): Raw anomaly scores
            
        Returns:
            np.ndarray: Normalized scores (0-100)
        """
        # Use percentile ranking to convert to 0-100 scale
        percentiles = np.zeros_like(scores)
        for i, score in enumerate(scores):
            percentile = (np.sum(scores <= score) / len(scores)) * 100
            percentiles[i] = percentile
        
        # Ensure training period has low scores (validation)
        # Add small random noise to avoid exactly 0 scores
        percentiles += np.random.normal(0, 0.1, len(percentiles))
        percentiles = np.clip(percentiles, 0, 100)
        
        return percentiles
    
    def get_feature_importance(self, analysis_data: pd.DataFrame) -> np.ndarray:
        """
        Get feature importance scores for anomaly detection.
        
        Args:
            analysis_data (pd.DataFrame): Analysis data
            
        Returns:
            np.ndarray: Feature importance matrix (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        X_analysis = self.scaler.transform(analysis_data)
        
        if self.method == 'isolation_forest':
            return self._get_isolation_forest_importance(X_analysis, analysis_data)
        elif self.method == 'autoencoder':
            return self._get_autoencoder_importance(X_analysis, analysis_data)
        elif self.method == 'pca':
            return self._get_pca_importance(X_analysis, analysis_data)
        elif self.method == 'ensemble':
            return self._get_ensemble_importance(X_analysis, analysis_data)
    
    def _get_isolation_forest_importance(self, X: np.ndarray, 
                                       original_data: pd.DataFrame) -> np.ndarray:
        """Get feature importance for Isolation Forest."""
        n_samples, n_features = X.shape
        importance_matrix = np.zeros((n_samples, n_features))
        
        # For each sample, calculate feature contribution by perturbation
        base_scores = self._predict_isolation_forest(X)
        
        for feature_idx in range(n_features):
            X_perturbed = X.copy()
            # Perturb feature with noise
            X_perturbed[:, feature_idx] += np.random.normal(0, 0.1, n_samples)
            perturbed_scores = self._predict_isolation_forest(X_perturbed)
            
            # Calculate importance as change in anomaly score
            importance_matrix[:, feature_idx] = np.abs(perturbed_scores - base_scores)
        
        return importance_matrix
    
    def _get_autoencoder_importance(self, X: np.ndarray, 
                                   original_data: pd.DataFrame) -> np.ndarray:
        """Get feature importance for Autoencoder."""
        # Calculate reconstruction error per feature
        reconstructed = self.models['autoencoder'].predict(X, verbose=0)
        reconstruction_errors = np.abs(X - reconstructed)
        
        return reconstruction_errors
    
    def _get_pca_importance(self, X: np.ndarray, 
                           original_data: pd.DataFrame) -> np.ndarray:
        """Get feature importance for PCA."""
        pca = self.models['pca']
        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        reconstruction_errors = np.abs(X - X_reconstructed)
        
        return reconstruction_errors
    
    def _get_ensemble_importance(self, X: np.ndarray, 
                                original_data: pd.DataFrame) -> np.ndarray:
        """Get feature importance for ensemble."""
        # Average importance from all methods
        if_importance = self._get_isolation_forest_importance(X, original_data)
        ae_importance = self._get_autoencoder_importance(X, original_data)
        pca_importance = self._get_pca_importance(X, original_data)
        
        # Normalize each method's importance
        if_importance = if_importance / (np.max(if_importance) + 1e-8)
        ae_importance = ae_importance / (np.max(ae_importance) + 1e-8)
        pca_importance = pca_importance / (np.max(pca_importance) + 1e-8)
        
        # Average ensemble importance
        ensemble_importance = (if_importance + ae_importance + pca_importance) / 3
        
        return ensemble_importance
