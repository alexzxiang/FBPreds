#!/usr/bin/env python3
"""
Dual-Track Elo System: Separate Domestic and European Performance
Domestic league wins in weak leagues contribute LESS to overall Elo
European performance contributes MORE
"""

import pandas as pd
import numpy as np
from collections import defaultdict

class DualTrackEloCalculator:
    def __init__(self, initial_rating=1500):
        self.initial_rating = initial_rating
        self.ratings = {}
        
        # Load league strength data
        learned_k = pd.read_csv('learned_league_k_factors.csv')
        
        # Create league quality tiers based on European win rate
        self.league_tiers = {}
        
        for _, row in learned_k.iterrows():
            league = row['league_code']
            win_rate = row['win_rate']
            
            # Tier 1: Elite leagues (>40% win rate)
            # Tier 2: Good leagues (30-40% win rate) 
            # Tier 3: Weak leagues (<30% win rate)
            if win_rate >= 0.40:
                tier = 1
                domestic_k = 40
                european_boost = 1.5
            elif win_rate >= 0.30:
                tier = 2
                domestic_k = 28
                european_boost = 1.3
            else:
                tier = 3
                domestic_k = 18  # HARSH penalty for weak leagues
                european_boost = 1.8  # But big boost if they do well in Europe
            
            self.league_tiers[league] = {
                'tier': tier,
                'domestic_k': domestic_k,
                'european_k': 50,  # All European matches matter
                'european_boost': european_boost,
                'win_rate': win_rate
            }
        
        # European competitions - HUGE disparity
        self.european_comps = {
            'CL': {'k': 65, 'boost': 2.2},  # Champions League - MAXIMUM prestige
            'EL': {'k': 30, 'boost': 1.0},  # Europa League - MODERATE
            'ECL': {'k': 18, 'boost': 0.7},  # Conference League - BARELY VALUABLE
            'ECLQ': {'k': 15, 'boost': 0.6},  # Conference qualifiers - very weak
            'ELQ': {'k': 25, 'boost': 0.9},  # Europa qualifiers
            'CLQ': {'k': 50, 'boost': 1.7}  # Champions League qualifiers still valuable
        }
        
        # Cup competitions - minimal impact
        self.cup_comps = {'FAC', 'CDR', 'RUP', 'DFB', 'CDF', 'COPA', 'TDC'}
        
        print("🎯 DUAL-TRACK ELO SYSTEM")
        print("\n📊 League Tiers:")
        print(f"{'League':<8} {'Tier':<6} {'Dom K':<8} {'Eur K':<8} {'Boost':<8} {'Win %':<8}")
        print("-" * 60)
        
        for league, data in sorted(self.league_tiers.items(), key=lambda x: x[1]['win_rate'], reverse=True):
            print(f"{league:<8} {data['tier']:<6} {data['domestic_k']:<8} {data['european_k']:<8} {data['european_boost']:<8.1f} {data['win_rate']*100:.1f}%")
    
    def get_match_params(self, competition_id):
        """Get K-factor and boost for a match"""
        # European competition
        if competition_id in self.european_comps:
            params = self.european_comps[competition_id]
            return params['k'], params['boost'], 'european'
        
        # Cup competition
        if competition_id in self.cup_comps:
            return 15, 0.5, 'cup'
        
        # Domestic league
        if competition_id in self.league_tiers:
            league_data = self.league_tiers[competition_id]
            return league_data['domestic_k'], 1.0, 'domestic'
        
        # Unknown - assume weak domestic
        return 25, 0.9, 'unknown'
    
    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_rating(self, rating, actual_score, expected_score, k_factor, boost):
        base_change = k_factor * (actual_score - expected_score)
        boosted_change = base_change * boost
        return rating + boosted_change
    
    def process_match(self, home_id, away_id, home_goals, away_goals, competition_id):
        home_rating = self.ratings.get(home_id, self.initial_rating)
        away_rating = self.ratings.get(away_id, self.initial_rating)
        
        k_factor, boost, match_type = self.get_match_params(competition_id)
        
        home_expected = self.expected_score(home_rating, away_rating)
        away_expected = 1 - home_expected
        
        if home_goals > away_goals:
            home_actual, away_actual = 1.0, 0.0
        elif home_goals < away_goals:
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual, away_actual = 0.5, 0.5
        
        new_home = self.update_rating(home_rating, home_actual, home_expected, k_factor, boost)
        new_away = self.update_rating(away_rating, away_actual, away_expected, k_factor, boost)
        
        self.ratings[home_id] = new_home
        self.ratings[away_id] = new_away
        
        return new_home, new_away, k_factor, boost, match_type


def main():
    print("="*80)
    print("DUAL-TRACK ELO: DOMESTIC vs EUROPEAN WEIGHTING")
    print("Weak league domestic wins = low Elo gain")
    print("European wins = high Elo gain (especially for weak league teams)")
    print("="*80)
    
    # Load matches
    print("\n📊 Loading matches...")
    matches = pd.read_csv('games/games.csv', low_memory=False)
    matches['season'] = pd.to_numeric(matches['season'], errors='coerce')
    matches = matches[matches['season'] >= 2020].copy()
    matches = matches[matches['home_club_goals'].notna()].copy()
    matches = matches.sort_values('date')
    
    print(f"   2020+ matches: {len(matches):,}\n")
    
    # Calculate Elo
    elo_calc = DualTrackEloCalculator(initial_rating=1500)
    
    match_type_stats = defaultdict(lambda: {'count': 0, 'avg_k': 0, 'avg_boost': 0})
    
    print("\n🔢 Processing matches...")
    for idx, row in matches.iterrows():
        new_home, new_away, k, boost, match_type = elo_calc.process_match(
            row['home_club_id'],
            row['away_club_id'],
            row['home_club_goals'],
            row['away_club_goals'],
            row['competition_id']
        )
        
        match_type_stats[match_type]['count'] += 1
        match_type_stats[match_type]['avg_k'] += k
        match_type_stats[match_type]['avg_boost'] += boost
        
        if idx % 5000 == 0:
            print(f"   {idx:,}/{len(matches):,} matches...", flush=True)
    
    print(f"\n   ✅ Processed all {len(matches):,} matches")
    print(f"   Teams rated: {len(elo_calc.ratings)}")
    
    # Show match type distribution
    print("\n📊 MATCH TYPE DISTRIBUTION:")
    print(f"{'Type':<15} {'Matches':<10} {'Avg K':<10} {'Avg Boost':<12} {'Effective K':<12}")
    print("-" * 65)
    
    for match_type, stats in sorted(match_type_stats.items(), key=lambda x: x[1]['count'], reverse=True):
        count = stats['count']
        avg_k = stats['avg_k'] / count
        avg_boost = stats['avg_boost'] / count
        effective_k = avg_k * avg_boost
        pct = count / len(matches) * 100
        print(f"{match_type:<15} {count:<10} {avg_k:<10.1f} {avg_boost:<12.2f} {effective_k:<12.1f} ({pct:.1f}%)")
    
    # Create DataFrame
    elo_data = []
    for team_id, rating in elo_calc.ratings.items():
        team_name = f"Team_{team_id}"
        home_match = matches[matches['home_club_id'] == team_id]
        if len(home_match) > 0:
            team_name = home_match.iloc[0]['home_club_name']
        else:
            away_match = matches[matches['away_club_id'] == team_id]
            if len(away_match) > 0:
                team_name = away_match.iloc[0]['away_club_name']
        
        elo_data.append({
            'team_id': team_id,
            'team_name': team_name,
            'elo_rating': rating
        })
    
    elo_df = pd.DataFrame(elo_data).sort_values('elo_rating', ascending=False)
    elo_df.to_csv('elite_leagues_elo_ratings.csv', index=False)
    
    print("\n📊 DUAL-TRACK ELO STATISTICS:")
    print(f"   Mean: {elo_df['elo_rating'].mean():.0f}")
    print(f"   Std: {elo_df['elo_rating'].std():.0f}")
    print(f"   Range: {elo_df['elo_rating'].min():.0f} to {elo_df['elo_rating'].max():.0f}")
    
    # Top 30
    print("\n🏆 TOP 30 TEAMS:")
    for rank, (_, row) in enumerate(elo_df.head(30).iterrows(), 1):
        print(f"   {rank:2d}. {row['team_name']:50s} {row['elo_rating']:.0f}")
    
    # Load old ratings for comparison
    old_elo = pd.read_csv('elite_leagues_elo_ratings_2020plus.csv')
    
    # Compare teams dominating weak leagues
    print("\n🔍 WEAK LEAGUE DOMINATORS (Should DROP):")
    print(f"{'Rank':<6} {'Team':<45} {'Old Elo':<10} {'New Elo':<10} {'Change':<10}")
    print("-" * 85)
    
    weak_league_teams = [
        'Olympiakos', 'Midtjylland', 'Celtic', 'Rangers',
        'Galatasaray', 'Fenerbahce', 'AEK', 'PAOK'
    ]
    
    for team_name in weak_league_teams:
        new_team = elo_df[elo_df['team_name'].str.contains(team_name, case=False, na=False)]
        old_team = old_elo[old_elo['team_name'].str.contains(team_name, case=False, na=False)]
        
        if len(new_team) > 0 and len(old_team) > 0:
            new_row = new_team.iloc[0]
            old_row = old_team.iloc[0]
            
            new_rank = (elo_df['elo_rating'] > new_row['elo_rating']).sum() + 1
            change = new_row['elo_rating'] - old_row['elo_rating']
            change_str = f"+{change:.0f}" if change > 0 else f"{change:.0f}"
            
            print(f"{new_rank:<6} {new_row['team_name'][:43]:<45} {old_row['elo_rating']:.0f}        {new_row['elo_rating']:.0f}        {change_str}")
    
    # Premier League teams should RISE
    print("\n⚽ PREMIER LEAGUE TEAMS (Should RISE):")
    print(f"{'Rank':<6} {'Team':<45} {'Old Elo':<10} {'New Elo':<10} {'Change':<10}")
    print("-" * 85)
    
    pl_teams = ['Arsenal', 'Man City', 'Liverpool', 'Chelsea', 'Aston Villa', 'Newcastle']
    
    for team_name in pl_teams:
        new_team = elo_df[elo_df['team_name'].str.contains(team_name, case=False, na=False)]
        old_team = old_elo[old_elo['team_name'].str.contains(team_name, case=False, na=False)]
        
        if len(new_team) > 0 and len(old_team) > 0:
            new_row = new_team.iloc[0]
            old_row = old_team.iloc[0]
            
            new_rank = (elo_df['elo_rating'] > new_row['elo_rating']).sum() + 1
            change = new_row['elo_rating'] - old_row['elo_rating']
            change_str = f"+{change:.0f}" if change > 0 else f"{change:.0f}"
            
            print(f"{new_rank:<6} {new_row['team_name'][:43]:<45} {old_row['elo_rating']:.0f}        {new_row['elo_rating']:.0f}        {change_str}")
    
    print("\n" + "="*80)
    print("✅ DUAL-TRACK ELO COMPLETE")
    print("="*80)
    print("\nSystem design:")
    print("  • Elite leagues (GB1, L1, ES1): Domestic K=40")
    print("  • Weak leagues (GR1, DK1, SC1): Domestic K=18 (56% LESS impact!)")
    print("  • Champions League: K=65, Boost=2.2 → Effective K=143 (MAXIMUM PRESTIGE)")
    print("  • Europa League: K=30, Boost=1.0 → Effective K=30 (21% of CL)")
    print("  • Conference League: K=18, Boost=0.7 → Effective K=12.6 (9% of CL!)")
    print("\nExpected Elo gain per win (50-50 matchup):")
    print("  Champions League:     +71.5 Elo")
    print("  Europa League:        +15.0 Elo")
    print("  Conference League:    +6.3 Elo (barely more than Danish domestic at +9)")
    print("  Premier League (dom): +20.0 Elo")
    print("  Greek League (dom):   +14.0 Elo")
    print("  Danish League (dom):  +9.0 Elo")


if __name__ == '__main__':
    main()
