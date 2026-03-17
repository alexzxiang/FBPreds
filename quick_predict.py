#!/usr/bin/env python3
"""
Quick match prediction using Elo ratings
Usage: python quick_predict.py "Team A" "Team B" [--home-team-first]
Example: python quick_predict.py "Tottenham" "Arsenal" --home-team-first
"""

import sys
import pandas as pd

def find_team(df, team_name):
    """Find team in Elo ratings"""
    # Try exact match first
    exact = df[df['name'] == team_name]
    if len(exact) > 0:
        return exact.iloc[0]
    
    # Try partial match
    partial = df[df['name'].str.contains(team_name, case=False, na=False)]
    if len(partial) > 0:
        if len(partial) > 1:
            print(f"\n⚠️  Multiple teams found for '{team_name}':")
            for _, team in partial.iterrows():
                print(f"   • {team['name']}")
            return partial.iloc[0]
        return partial.iloc[0]
    
    return None

def predict_match(home_team, away_team, home_elo, away_elo):
    """Predict match outcome based on Elo"""
    elo_diff = home_elo - away_elo
    
    # Draw probability heuristic
    if abs(elo_diff) < 50:
        draw_prob = 0.35
    elif abs(elo_diff) < 100:
        draw_prob = 0.28
    elif abs(elo_diff) < 150:
        draw_prob = 0.22
    else:
        draw_prob = 0.18
    
    # Win probabilities (simplified)
    if elo_diff > 0:
        home_prob = 0.50 + min(elo_diff / 500, 0.35)
        away_prob = 1.0 - home_prob - draw_prob
    else:
        away_prob = 0.50 + min(abs(elo_diff) / 500, 0.35)
        home_prob = 1.0 - away_prob - draw_prob
    
    return home_prob, draw_prob, away_prob

if __name__ == "__main__":
    # Load Elo ratings
    try:
        elo_df = pd.read_csv('elite_leagues_elo_ratings.csv')
    except FileNotFoundError:
        print("❌ Error: elite_leagues_elo_ratings.csv not found")
        sys.exit(1)
    
    # Get team names from command line or use default
    if len(sys.argv) >= 3:
        team1_name = sys.argv[1]
        team2_name = sys.argv[2]
        home_first = '--home-team-first' in sys.argv or '--home' in sys.argv
        
        if home_first:
            home_name, away_name = team1_name, team2_name
        else:
            # Ask user
            print(f"\nWhich team is playing at home?")
            print(f"1. {team1_name}")
            print(f"2. {team2_name}")
            choice = input("Enter 1 or 2: ").strip()
            if choice == "1":
                home_name, away_name = team1_name, team2_name
            else:
                home_name, away_name = team2_name, team1_name
    else:
        # Interactive mode
        print("\n⚽ FOOTBALL MATCH PREDICTOR")
        print("=" * 60)
        home_name = input("Home team: ").strip()
        away_name = input("Away team: ").strip()
    
    # Find teams
    home = find_team(elo_df, home_name)
    away = find_team(elo_df, away_name)
    
    if home is None:
        print(f"\n❌ Could not find '{home_name}' in database")
        print(f"\nAvailable teams with '{home_name[:4]}':")
        matches = elo_df[elo_df['name'].str.contains(home_name[:4], case=False, na=False)]
        for _, team in matches.head(10).iterrows():
            print(f"   • {team['name']} (Elo: {team['elo']:.0f})")
        sys.exit(1)
    
    if away is None:
        print(f"\n❌ Could not find '{away_name}' in database")
        print(f"\nAvailable teams with '{away_name[:4]}':")
        matches = elo_df[elo_df['name'].str.contains(away_name[:4], case=False, na=False)]
        for _, team in matches.head(10).iterrows():
            print(f"   • {team['name']} (Elo: {team['elo']:.0f})")
        sys.exit(1)
    
    # Make prediction
    home_elo = home['elo']
    away_elo = away['elo']
    home_prob, draw_prob, away_prob = predict_match(home['name'], away['name'], home_elo, away_elo)
    
    # Display results
    print("\n" + "=" * 80)
    print(f"🏟️  {home['name']} vs {away['name']}")
    print("=" * 80)
    
    print(f"\n📊 ELO RATINGS:")
    print(f"   🏠 {home['name']}: {home_elo:.0f}")
    print(f"   ✈️  {away['name']}: {away_elo:.0f}")
    print(f"   📈 Difference: {home_elo - away_elo:+.0f}")
    
    print(f"\n🎯 PREDICTION:")
    print(f"   Home Win: {home_prob:.1%}")
    print(f"   Draw:     {draw_prob:.1%}")
    print(f"   Away Win: {away_prob:.1%}")
    
    # Predicted outcome
    if home_prob > draw_prob and home_prob > away_prob:
        prediction = f"🏆 {home['name']} WIN"
        confidence = home_prob
    elif away_prob > draw_prob and away_prob > home_prob:
        prediction = f"🏆 {away['name']} WIN"
        confidence = away_prob
    else:
        prediction = "🤝 DRAW"
        confidence = draw_prob
    
    print(f"\n✅ MOST LIKELY: {prediction}")
    print(f"   Confidence: {confidence:.1%}")
    
    # Analysis
    elo_diff = home_elo - away_elo
    print(f"\n💡 ANALYSIS:")
    if abs(elo_diff) < 30:
        print("   • Extremely evenly matched teams")
        print("   • Any result possible")
        print("   • Home advantage may be decisive")
    elif abs(elo_diff) < 80:
        print("   • Close match expected")
        print("   • Small details could make the difference")
    elif abs(elo_diff) < 150:
        print("   • Moderate favorite exists")
        print("   • Upset still possible")
    else:
        print("   • Clear favorite")
        if elo_diff > 0:
            print(f"   • Home team strongly favored")
        else:
            print(f"   • Away team strongly favored despite no home advantage")
    
    print("\n" + "=" * 80)
