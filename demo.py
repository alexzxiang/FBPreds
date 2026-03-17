#!/usr/bin/env python3
"""
Show example predictions from the final hybrid model.
"""

import pickle
import pandas as pd
import numpy as np

print("="*80)
print("EXAMPLE PREDICTIONS - HYBRID MODEL")
print("Draw Classifier (60% threshold) + Elo for Home/Away")
print("="*80)

# Load draw classifier
with open('models/binary_draw_classifier.pkl', 'rb') as f:
    draw_data = pickle.load(f)
    draw_model = draw_data['xgb_model']
    draw_scaler = draw_data['scaler']
    draw_features = draw_data['feature_names']

# Load data
from train_features import ComprehensiveMatchPredictor

predictor = ComprehensiveMatchPredictor()
predictor.load_data()

test_matches = predictor.matches[predictor.matches['season'] == 2025].copy()

# Generate features
X_data, y_data, elo_diffs, match_info = [], [], [], []

for _, match in test_matches.iterrows():
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
        match_info.append({
            'home': match['home_club_name'],
            'away': match['away_club_name'],
            'score': f"{int(match['home_club_goals'])}-{int(match['away_club_goals'])}",
            'date': match['date'],
            'outcome': outcome
        })
    except:
        continue

X = pd.DataFrame(X_data)
y = np.array(y_data)
elo_diffs = np.array(elo_diffs)

# Align and predict
X_draw = X.copy()
for feat in draw_features:
    if feat not in X_draw.columns:
        X_draw[feat] = 0
X_draw = X_draw[draw_features]

X_draw_scaled = draw_scaler.transform(X_draw.values)
draw_proba = draw_model.predict_proba(X_draw_scaled)[:, 1]

DRAW_THRESHOLD = 0.60
y_pred = np.array([1 if draw_proba[i] >= DRAW_THRESHOLD else (2 if elo_diffs[i] > 0 else 0) for i in range(len(y))])

# Helper
def show_pred(i):
    m = match_info[i]
    names = ['Away Win', 'Draw', 'Home Win']
    
    print(f"\n{'─'*80}")
    print(f"🏟️  {m['home']} vs {m['away']}")
    print(f"⚽ Score: {m['score']} | 📅 {m['date']}")
    print(f"\n🤖 Analysis:")
    print(f"   Draw Prob: {draw_proba[i]*100:.1f}% | Elo Diff: {elo_diffs[i]:+.0f}")
    
    if draw_proba[i] >= DRAW_THRESHOLD:
        print(f"   → DRAW prediction ({draw_proba[i]*100:.1f}% ≥ 60%)")
    else:
        winner = "HOME" if elo_diffs[i] > 0 else "AWAY"
        print(f"   → {winner} WIN (Elo {elo_diffs[i]:+.0f})")
    
    result = "✅ CORRECT" if y_pred[i] == m['outcome'] else "❌ WRONG"
    print(f"\n{result}: Predicted {names[y_pred[i]]} | Actual {names[m['outcome']]}")

# Examples
print("\n🎯 HIGH-CONFIDENCE DRAW PREDICTIONS (Correct)")
print("="*80)
correct_draws = [(i, draw_proba[i]) for i in range(len(y)) if y_pred[i] == 1 and y[i] == 1 and draw_proba[i] > 0.75]
correct_draws.sort(key=lambda x: x[1], reverse=True)
for i, _ in correct_draws[:5]:
    show_pred(i)

print("\n\n🏠 STRONG HOME TEAM WINS (Correct)")
print("="*80)
home_wins = [(i, elo_diffs[i]) for i in range(len(y)) if y_pred[i] == 2 and y[i] == 2 and elo_diffs[i] > 100]
home_wins.sort(key=lambda x: x[1], reverse=True)
for i, _ in home_wins[:5]:
    show_pred(i)

print("\n\n✈️  STRONG AWAY TEAM WINS (Correct)")
print("="*80)
away_wins = [(i, elo_diffs[i]) for i in range(len(y)) if y_pred[i] == 0 and y[i] == 0 and elo_diffs[i] < -100]
away_wins.sort(key=lambda x: x[1])
for i, _ in away_wins[:5]:
    show_pred(i)

print("\n\n⚠️  CLOSE CALLS - Missed Draws")
print("="*80)
close = [(i, draw_proba[i]) for i in range(len(y)) if y_pred[i] != 1 and y[i] == 1 and 0.50 < draw_proba[i] < 0.60]
close.sort(key=lambda x: x[1], reverse=True)
for i, _ in close[:3]:
    show_pred(i)

# Summary
acc = (y_pred == y).mean()
print(f"\n\n{'='*80}")
print(f"SUMMARY: {acc*100:.1f}% Accuracy on 2025-26 Season")
print(f"{'='*80}")
print(f"Predictions: Home {(y_pred==2).sum()} ({(y_pred==2).mean()*100:.1f}%) | Draw {(y_pred==1).sum()} ({(y_pred==1).mean()*100:.1f}%) | Away {(y_pred==0).sum()} ({(y_pred==0).mean()*100:.1f}%)")
print(f"Actuals:     Home {(y==2).sum()} ({(y==2).mean()*100:.1f}%) | Draw {(y==1).sum()} ({(y==1).mean()*100:.1f}%) | Away {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
