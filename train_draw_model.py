#!/usr/bin/env python3
"""
Train a separate binary classifier specifically for predicting draws.
This will be used in conjunction with the main 3-class model.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TRAINING BINARY DRAW CLASSIFIER")
print("="*80)

# Load predictor
print("\n📊 Loading data...")
from train_features import ComprehensiveMatchPredictor

predictor = ComprehensiveMatchPredictor()
predictor.load_data()

print(f"\n🔄 Generating features for all {len(predictor.matches)} matches...")
X_data = []
y_data = []

for idx, (_, match) in enumerate(predictor.matches.iterrows()):
    if idx % 2000 == 0:
        print(f"   {idx}/{len(predictor.matches)}...")
    
    try:
        features = predictor.create_comprehensive_features(match)
        if features is None:
            continue
        
        # Binary outcome: Is it a draw?
        is_draw = 1 if match['home_club_goals'] == match['away_club_goals'] else 0
        
        X_data.append(features)
        y_data.append(is_draw)
        
    except Exception as e:
        continue

print(f"\n✅ Successfully processed {len(X_data)} matches")

# Convert to DataFrame
X = pd.DataFrame(X_data)
y = np.array(y_data)

print(f"\n📊 Dataset statistics:")
print(f"   Total matches: {len(y)}")
print(f"   Draws: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"   Not draws: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Split:")
print(f"   Train: {len(X_train)} ({y_train.sum()} draws, {y_train.mean()*100:.1f}%)")
print(f"   Test:  {len(X_test)} ({y_test.sum()} draws, {y_test.mean()*100:.1f}%)")

# Scale features
print("\n🔄 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# Train XGBoost
print("\n🚀 Training XGBoost binary draw classifier...")

# Use scale_pos_weight to handle class imbalance
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f"   Using scale_pos_weight: {scale_pos_weight:.2f}")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,  # Handle imbalance
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = xgb.predict(X_train_scaled)
y_test_pred = xgb.predict(X_test_scaled)
y_train_proba = xgb.predict_proba(X_train_scaled)[:, 1]
y_test_proba = xgb.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*80)
print("XGBOOST RESULTS")
print("="*80)

print(f"\n📊 Train Accuracy: {(y_train_pred == y_train).mean()*100:.1f}%")
print(f"📊 Test Accuracy:  {(y_test_pred == y_test).mean()*100:.1f}%")
print(f"📊 Test ROC-AUC:   {roc_auc_score(y_test, y_test_proba):.3f}")

print(f"\n📊 Test Predictions:")
print(f"   Predicted draws: {y_test_pred.sum()} ({y_test_pred.mean()*100:.1f}%)")
print(f"   Actual draws:    {y_test.sum()} ({y_test.mean()*100:.1f}%)")

print(f"\n📊 Test Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Not Draw', 'Draw']))

print(f"\n📊 Test Confusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(f"                Predicted")
print(f"                Not Draw  Draw")
print(f"   Actual")
print(f"   Not Draw     {cm[0,0]:6d}  {cm[0,1]:5d}")
print(f"   Draw         {cm[1,0]:6d}  {cm[1,1]:5d}")

# Train Random Forest for comparison
print("\n🚀 Training Random Forest binary draw classifier...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',  # Handle imbalance
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

y_test_pred_rf = rf.predict(X_test_scaled)
y_test_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*80)
print("RANDOM FOREST RESULTS")
print("="*80)

print(f"\n📊 Test Accuracy: {(y_test_pred_rf == y_test).mean()*100:.1f}%")
print(f"📊 Test ROC-AUC:  {roc_auc_score(y_test, y_test_proba_rf):.3f}")

print(f"\n📊 Test Predictions:")
print(f"   Predicted draws: {y_test_pred_rf.sum()} ({y_test_pred_rf.mean()*100:.1f}%)")
print(f"   Actual draws:    {y_test.sum()} ({y_test.mean()*100:.1f}%)")

print(f"\n📊 Test Classification Report:")
print(classification_report(y_test, y_test_pred_rf, target_names=['Not Draw', 'Draw']))

# Cross-validation
print("\n🔄 Cross-validation (XGBoost)...")
cv_scores = cross_val_score(xgb, X_train_scaled, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"   ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Feature importance
print("\n📊 Top 20 Features for Draw Prediction:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(20).iterrows():
    print(f"   {i+1:2d}. {row['feature']:30s} {row['importance']*100:5.2f}%")

# Save model
print("\n💾 Saving model...")
model_data = {
    'xgb_model': xgb,
    'rf_model': rf,
    'scaler': scaler,
    'feature_names': list(X.columns),
    'train_draw_rate': y_train.mean(),
    'test_draw_rate': y_test.mean(),
    'test_accuracy': (y_test_pred == y_test).mean(),
    'test_roc_auc': roc_auc_score(y_test, y_test_proba)
}

with open('models/binary_draw_classifier.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("   ✅ Saved to models/binary_draw_classifier.pkl")

# Analyze probability distributions
print("\n" + "="*80)
print("PROBABILITY ANALYSIS")
print("="*80)

print(f"\n📊 Draw probability distribution:")
print(f"   Mean: {y_test_proba.mean()*100:.1f}%")
print(f"   Median: {np.median(y_test_proba)*100:.1f}%")
print(f"   For actual draws: {y_test_proba[y_test == 1].mean()*100:.1f}%")
print(f"   For non-draws: {y_test_proba[y_test == 0].mean()*100:.1f}%")

# Threshold analysis
print(f"\n📊 Threshold analysis:")
for threshold in [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
    y_pred_thresh = (y_test_proba >= threshold).astype(int)
    accuracy = (y_pred_thresh == y_test).mean()
    predicted_draws = y_pred_thresh.sum()
    actual_draws = y_test.sum()
    
    # Precision and recall
    tp = ((y_pred_thresh == 1) & (y_test == 1)).sum()
    precision = tp / predicted_draws if predicted_draws > 0 else 0
    recall = tp / actual_draws if actual_draws > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"   {threshold:.0%}: Acc={accuracy*100:4.1f}% Draws={predicted_draws:4d} ({predicted_draws/len(y_test)*100:4.1f}%) "
          f"P={precision*100:4.1f}% R={recall*100:4.1f}% F1={f1*100:4.1f}%")

print("\n✅ Binary draw classifier training complete!")
print("\nNext steps:")
print("1. Use this model to predict P(draw) for each match")
print("2. Combine with main 3-class model for final predictions")
print("3. Test on 2025-26 season data")
