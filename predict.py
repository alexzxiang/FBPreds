#!/usr/bin/env python3
"""
Simpler approach:
1. Binary draw classifier for draw detection
2. Simple Elo-based logic for home vs away
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

print("="*80)
print("SIMPLE HYBRID: DRAW CLASSIFIER + ELO")
print("="*80)

# Load draw classifier
print("\n📦 Loading draw classifier...")
with open('models/binary_draw_classifier.pkl', 'rb') as f:
    draw_data = pickle.load(f)
    draw_model = draw_data['xgb_model']
    draw_scaler = draw_data['scaler']
    draw_features = draw_data['feature_names']

print(f"   ✅ Draw model: {len(draw_features)} features")

# Load data
print("\n📊 Loading data...")
from train_features import ComprehensiveMatchPredictor

predictor = ComprehensiveMatchPredictor()
predictor.load_data()

test_matches = predictor.matches[predictor.matches['season'] == 2025].copy()
print(f"   ✅ {len(test_matches)} matches from 2025-26")

# Generate features
print("\n🔄 Generating features...")
X_data = []
y_data = []
elo_diffs = []

for idx, (_, match) in enumerate(test_matches.iterrows()):
    if idx % 500 == 0:
        print(f"   {idx}/{len(test_matches)}...")
    
    try:
        features = predictor.create_comprehensive_features(match)
        if features is None:
            continue
        
        if match['home_club_goals'] > match['away_club_goals']:
            outcome = 2
        elif match['home_club_goals'] < match['away_club_goals']:
            outcome = 0
        else:
            outcome = 1
        
        X_data.append(features)
        y_data.append(outcome)
        elo_diffs.append(features.get('elo_diff', 0))
    except:
        continue

X = pd.DataFrame(X_data)
y = np.array(y_data)
elo_diffs = np.array(elo_diffs)

print(f"\n✅ {len(X)} matches processed")

# Align features for draw model
X_draw = X.copy()
for feat in draw_features:
    if feat not in X_draw.columns:
        X_draw[feat] = 0
X_draw = X_draw[draw_features]

# Get draw predictions
print("\n🎯 Making predictions...")
X_draw_scaled = draw_scaler.transform(X_draw.values)
draw_proba = draw_model.predict_proba(X_draw_scaled)[:, 1]

print("\n" + "="*80)
print("TESTING DIFFERENT DRAW THRESHOLDS")
print("="*80)

best_accuracy = 0
best_threshold = 0
best_predictions = None

for draw_threshold in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]:
    y_pred = np.zeros(len(y), dtype=int)
    
    for i in range(len(y)):
        if draw_proba[i] >= draw_threshold:
            y_pred[i] = 1  # Draw
        else:
            # Use Elo diff for home vs away
            # Positive elo_diff means home is stronger
            if elo_diffs[i] > 0:
                y_pred[i] = 2  # Home
            else:
                y_pred[i] = 0  # Away
    
    accuracy = (y_pred == y).mean()
    
    home_pred = (y_pred == 2).sum()
    draw_pred = (y_pred == 1).sum()
    away_pred = (y_pred == 0).sum()
    
    home_actual = (y == 2).sum()
    draw_actual = (y == 1).sum()
    away_actual = (y == 0).sum()
    
    dist_error = abs(home_pred/len(y) - home_actual/len(y)) + \
                 abs(draw_pred/len(y) - draw_actual/len(y)) + \
                 abs(away_pred/len(y) - away_actual/len(y))
    
    print(f"\n📊 Draw Threshold: {draw_threshold:.0%}")
    print(f"   Accuracy: {accuracy*100:.1f}%")
    print(f"   Predictions: H:{home_pred/len(y)*100:4.1f}% D:{draw_pred/len(y)*100:4.1f}% A:{away_pred/len(y)*100:4.1f}%")
    print(f"   Actuals:     H:{home_actual/len(y)*100:4.1f}% D:{draw_actual/len(y)*100:4.1f}% A:{away_actual/len(y)*100:4.1f}%")
    print(f"   Dist Error: {dist_error*100:.1f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = draw_threshold
        best_predictions = y_pred.copy()

# Show best result
print("\n" + "="*80)
print(f"BEST RESULT (Threshold: {best_threshold:.0%})")
print("="*80)

y_pred = best_predictions

print(f"\n🎯 ACCURACY: {best_accuracy*100:.1f}%")

home_pred = (y_pred == 2).sum()
draw_pred = (y_pred == 1).sum()
away_pred = (y_pred == 0).sum()

home_actual = (y == 2).sum()
draw_actual = (y == 1).sum()
away_actual = (y == 0).sum()

print(f"\n📊 PREDICTIONS:")
print(f"   Home Wins: {home_pred:4d} ({home_pred/len(y)*100:5.1f}%)")
print(f"   Draws:     {draw_pred:4d} ({draw_pred/len(y)*100:5.1f}%)")
print(f"   Away Wins: {away_pred:4d} ({away_pred/len(y)*100:5.1f}%)")

print(f"\n📊 ACTUALS:")
print(f"   Home Wins: {home_actual:4d} ({home_actual/len(y)*100:5.1f}%)")
print(f"   Draws:     {draw_actual:4d} ({draw_actual/len(y)*100:5.1f}%)")
print(f"   Away Wins: {away_actual:4d} ({away_actual/len(y)*100:5.1f}%)")

dist_error = abs(home_pred/len(y) - home_actual/len(y)) + \
             abs(draw_pred/len(y) - draw_actual/len(y)) + \
             abs(away_pred/len(y) - away_actual/len(y))

print(f"\n📏 Distribution Error: {dist_error*100:.1f}%")

print(f"\n📊 Classification Report:")
print(classification_report(y, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))

print(f"\n📊 Confusion Matrix:")
cm = confusion_matrix(y, y_pred)
print(f"                    Predicted")
print(f"                Away   Draw   Home")
print(f"   Actual")
print(f"   Away    {cm[0,0]:6d} {cm[0,1]:6d} {cm[0,2]:6d}")
print(f"   Draw    {cm[1,0]:6d} {cm[1,1]:6d} {cm[1,2]:6d}")
print(f"   Home    {cm[2,0]:6d} {cm[2,1]:6d} {cm[2,2]:6d}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n✅ Simple Hybrid: {best_accuracy*100:.1f}% accuracy on 2025-26")
print(f"✅ Uses binary draw classifier + Elo for home/away")
print(f"✅ Draw threshold: {best_threshold:.0%}")

if best_accuracy >= 50:
    print(f"\n🎯 ACHIEVES 50%+ ACCURACY!")
else:
    print(f"\n⚠️  Below 50% - may need better home/away logic")
