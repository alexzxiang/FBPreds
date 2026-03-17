#!/usr/bin/env python3
"""
Predict Milan vs Parma using the full 120-feature model
"""
import pickle
import pandas as pd
import numpy as np
from train_features import ComprehensiveMatchPredictor

print("=" * 80)
print("MILAN vs PARMA - FULL MODEL")
print("=" * 80)

# Initialize predictor
predictor = ComprehensiveMatchPredictor()
predictor.load_data()

# Load binary draw classifier
with open('models/binary_draw_classifier.pkl', 'rb') as f:
    d = pickle.load(f)
    draw_model = d['xgb_model']
    draw_scaler = d['scaler']
    draw_feats = d['feature_names']

# Find Milan and Parma in matches
milan_names = ['AC Milan', 'Milan', 'AC Milan']
parma_names = ['Parma', 'Parma Calcio 1913']

# Find recent H2H match or use a template match
h2h = predictor.matches[
    (
        (predictor.matches['home_club_name'].isin(milan_names) & 
         predictor.matches['away_club_name'].isin(parma_names)) |
        (predictor.matches['away_club_name'].isin(milan_names) & 
         predictor.matches['home_club_name'].isin(parma_names))
    )
]

if len(h2h) > 0:
    print(f"Found {len(h2h)} H2H matches")
    template = h2h.iloc[-1].copy()
    
    # Swap if needed to make Milan home
    if template['away_club_name'] in milan_names:
        print("Swapping template to make Milan home team")
        template['home_club_name'], template['away_club_name'] = template['away_club_name'], template['home_club_name']
        template['home_club_id'], template['away_club_id'] = template['away_club_id'], template['home_club_id']
else:
    # Use a recent Milan home match as template
    print("No H2H found, using recent Milan home match as template")
    milan_home = predictor.matches[
        predictor.matches['home_club_name'].isin(milan_names) &
        (predictor.matches['date'] > '2024-01-01')
    ]
    if len(milan_home) > 0:
        template = milan_home.iloc[-1].copy()
        # Replace away team with Parma
        parma_match = predictor.matches[
            (predictor.matches['home_club_name'].isin(parma_names)) |
            (predictor.matches['away_club_name'].isin(parma_names))
        ]
        if len(parma_match) > 0:
            template['away_club_name'] = parma_match.iloc[-1]['home_club_name'] if parma_match.iloc[-1]['home_club_name'] in parma_names else parma_match.iloc[-1]['away_club_name']
            template['away_club_id'] = parma_match.iloc[-1]['home_club_id'] if parma_match.iloc[-1]['home_club_name'] in parma_names else parma_match.iloc[-1]['away_club_id']
    else:
        # Fallback: use any recent match
        print("No Milan home match found, using fallback template")
        template = predictor.matches[predictor.matches['date'] > '2024-01-01'].iloc[-1].copy()
        template['home_club_name'] = 'AC Milan'
        template['away_club_name'] = 'Parma'

print(f"\nUsing template: {template['home_club_name']} vs {template['away_club_name']}")
print(f"Template date: {template['date']}")

# Generate comprehensive features
features = predictor.create_comprehensive_features(template)

# Get Elo ratings for actual teams
elo_data = pd.read_csv('elite_leagues_elo_ratings.csv')
milan_elo = elo_data[elo_data['team_name'].str.contains('Milan', case=False, na=False)]
parma_elo = elo_data[elo_data['team_name'].str.contains('Parma', case=False, na=False)]

if len(milan_elo) > 0:
    home_elo = milan_elo.iloc[0]['elo_rating']
    features['home_elo'] = home_elo
else:
    home_elo = features.get('home_elo', 1500)

if len(parma_elo) > 0:
    away_elo = parma_elo.iloc[0]['elo_rating']
    features['away_elo'] = away_elo
else:
    away_elo = features.get('away_elo', 1500)

features['elo_diff'] = home_elo - away_elo

print(f"\nELO: Milan {home_elo:.0f} vs Parma {away_elo:.0f} (diff: {features['elo_diff']:+.0f})")

# Prepare features for draw model
X = pd.DataFrame([features])

# Fill missing features with 0
for feat in draw_feats:
    if feat not in X.columns:
        X[feat] = 0

# Predict draw probability
X_scaled = draw_scaler.transform(X[draw_feats])
draw_prob = draw_model.predict_proba(X_scaled)[0][1]

# Apply hybrid model logic (60% threshold for draw)
if draw_prob >= 0.60:
    prediction = "DRAW"
    home_win_prob = (1 - draw_prob) / 2
    away_win_prob = (1 - draw_prob) / 2
else:
    # Use Elo to determine winner
    elo_diff = features['elo_diff']
    if elo_diff > 0:
        prediction = "MILAN WIN"
        # Stronger team gets more of the non-draw probability
        home_win_prob = (1 - draw_prob) * 0.7
        away_win_prob = (1 - draw_prob) * 0.3
    else:
        prediction = "PARMA WIN"
        home_win_prob = (1 - draw_prob) * 0.3
        away_win_prob = (1 - draw_prob) * 0.7

print(f"\nPROBABILITIES:")
print(f"  Milan:  {home_win_prob*100:.1f}%")
print(f"  Draw:   {draw_prob*100:.1f}%")
print(f"  Parma:  {away_win_prob*100:.1f}%")
print(f"\n✅ PREDICTED: {prediction}")

# Show some key features
if 'home_form_points' in features and 'away_form_points' in features:
    print(f"\nForm: Home {features['home_form_points']:.1f}pts | Away {features['away_form_points']:.1f}pts")

print("=" * 80)
