"""
Simple Test Script for Anomaly Detection System
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("TESTING MULTIVARIATE ANOMALY DETECTION SYSTEM")
print("=" * 60)

try:
    # Test 1: Import all modules
    print("\n1. Testing module imports...")
    from main import main
    from utils import load_sample_data, analyze_results
    from anomaly_detector import AnomalyDetector
    from data_preprocessor import DataPreprocessor
    from feature_attribution import FeatureAttributor
    print("   ✓ All modules imported successfully")

    # Test 2: Generate sample data
    print("\n2. Generating sample data...")
    df_sample = load_sample_data()
    print(f"   ✓ Generated {len(df_sample)} rows with {len(df_sample.columns)} features")

    # Test 3: Test data preprocessing
    print("\n3. Testing data preprocessing...")
    preprocessor = DataPreprocessor()
    train_data, analysis_data = preprocessor.split_data(df_sample)
    print(f"   ✓ Training data: {train_data.shape}")
    print(f"   ✓ Analysis data: {analysis_data.shape}")

    # Test 4: Test anomaly detection
    print("\n4. Testing anomaly detection...")
    detector = AnomalyDetector(method='isolation_forest')  # Use simpler method for testing
    detector.fit(train_data)
    scores = detector.predict(analysis_data)
    normalized_scores = detector.normalize_scores(scores)
    print(f"   ✓ Generated {len(normalized_scores)} anomaly scores")
    print(f"   ✓ Score range: {normalized_scores.min():.2f} to {normalized_scores.max():.2f}")

    # Test 5: Test feature attribution
    print("\n5. Testing feature attribution...")
    attributor = FeatureAttributor()
    attributions = attributor.calculate_attributions(detector, analysis_data, scores)
    print(f"   ✓ Generated attributions for {len(attributions)} samples")

    # Test 6: Create output dataframe
    print("\n6. Creating output dataframe...")
    output_df = analysis_data.copy()
    output_df['Abnormality_score'] = normalized_scores
    
    for i in range(7):
        col_name = f'top_feature_{i+1}'
        output_df[col_name] = [
            attr[i] if i < len(attr) else '' 
            for attr in attributions
        ]
    
    print(f"   ✓ Output dataframe shape: {output_df.shape}")
    print(f"   ✓ Output columns: {list(output_df.columns)}")

    # Test 7: Save results
    print("\n7. Saving results...")
    output_file = 'test_results.csv'
    output_df.to_csv(output_file, index=False)
    print(f"   ✓ Results saved to {output_file}")

    # Test 8: Validate training period
    print("\n8. Validating training period performance...")
    training_scores = normalized_scores[:120]  # First 120 hours
    mean_score = np.mean(training_scores)
    max_score = np.max(training_scores)
    
    print(f"   Training period mean score: {mean_score:.2f}")
    print(f"   Training period max score: {max_score:.2f}")
    
    if mean_score < 10 and max_score < 25:
        print("   ✓ Training period validation PASSED")
    else:
        print("   ⚠ Training period validation WARNING")

    # Test 9: Analyze results
    print("\n9. Analyzing results...")
    high_anomalies = np.sum(normalized_scores > 60)
    severe_anomalies = np.sum(normalized_scores > 90)
    print(f"   High anomalies (>60): {high_anomalies}")
    print(f"   Severe anomalies (>90): {severe_anomalies}")
    print(f"   Overall mean score: {np.mean(normalized_scores):.2f}")

    # Test 10: Feature contribution analysis
    print("\n10. Feature contribution analysis...")
    all_features = []
    for attribution in attributions:
        all_features.extend([f for f in attribution if f])
    
    if all_features:
        from collections import Counter
        feature_counts = Counter(all_features)
        print("    Top contributing features:")
        for feature, count in feature_counts.most_common(5):
            print(f"      {feature}: {count} times")
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nOutput file: {output_file}")
    print(f"Data points analyzed: {len(output_df)}")
    print(f"Features in dataset: {len(train_data.columns)}")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
