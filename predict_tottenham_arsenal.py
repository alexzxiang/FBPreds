#!/usr/bin/env python3
import pickle, pandas as pd, numpy as np
from train_features import ComprehensiveMatchPredictor

print("="*80); print("TOTTENHAM vs ARSENAL - FULL MODEL"); print("="*80)

predictor = ComprehensiveMatchPredictor()
predictor.load_data()

with open('models/binary_draw_classifier.pkl', 'rb') as f:
    d = pickle.load(f)
    draw_model, draw_scaler, draw_feats = d['xgb_model'], d['scaler'], d['feature_names']

h2h = predictor.matches[
    ((predictor.matches['home_club_name'].str.contains('Tottenham', case=False, na=False)) &
     (predictor.matches['away_club_name'].str.contains('Arsenal', case=False, na=False))) |
    ((predictor.matches['home_club_name'].str.contains('Arsenal', case=False, na=False)) &
     (predictor.matches['away_club_name'].str.contains('Tottenham', case=False, na=False)))
]

t = h2h.iloc[-1].copy() if len(h2h) > 0 else predictor.matches.iloc[-1].copy()
t['home_club_name'], t['away_club_name'] = 'Tottenham Hotspur', 'Arsenal FC'

f = predictor.create_comprehensive_features(t)
if f:
    X = pd.DataFrame([f])
    for feat in draw_feats:
        if feat not in X.columns: X[feat] = 0
    
    draw_prob = draw_model.predict_proba(draw_scaler.transform(X[draw_feats]))[0][1]
    elo_diff = f.get('elo_diff', 0)
    
    if draw_prob >= 0.60:
        pred, hp, ap = "DRAW", (1-draw_prob)/2, (1-draw_prob)/2
    elif elo_diff > 0:
        pred = "TOTTENHAM"
        hp = 0.50 + min(elo_diff/500, 0.35)
        ap = max(1 - hp - draw_prob, 0.01)
    else:
        pred = "ARSENAL"
        ap = 0.50 + min(abs(elo_diff)/500, 0.35)
        hp = max(1 - ap - draw_prob, 0.01)
    
    tot = hp + draw_prob + ap
    hp, draw_prob, ap = hp/tot, draw_prob/tot, ap/tot
    
    print(f"\nELO: Tottenham {f.get('home_elo',1500):.0f} vs Arsenal {f.get('away_elo',1500):.0f} (diff: {elo_diff:+.0f})")
    print(f"\nPROBABILITIES:")
    print(f"  Tottenham: {hp:6.1%}")
    print(f"  Draw:      {draw_prob:6.1%}")
    print(f"  Arsenal:   {ap:6.1%}")
    print(f"\n✅ PREDICTED: {pred} WIN" if pred != "DRAW" else "\n✅ PREDICTED: DRAW")
    print(f"\nForm: Home {f.get('home_form_points',0):.1f}pts | Away {f.get('away_form_points',0):.1f}pts")
print("="*80)
