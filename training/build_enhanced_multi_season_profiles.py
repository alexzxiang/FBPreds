"""
Build multi-season player profiles with:
1. All 3 seasons (23-24, 24-25, 25-26)
2. Temporal weighting (more recent = higher weight)
3. League quality weighting (Premier League > others)
4. Normalize weights for players with missing seasons
"""

import pandas as pd
import numpy as np
from pathlib import Path

# League quality weights based on UEFA coefficient and competition level
LEAGUE_WEIGHTS = {
    'Premier League': 1.10,     # Highest quality
    'La Liga': 1.05,            # Slightly below PL
    'Bundesliga': 1.00,         # Baseline
    'Serie A': 1.00,            # Equal to Bundesliga
    'Ligue 1': 0.95,            # Slightly below
    'Other': 0.90               # Catch-all for non-Big 5
}

# Temporal weights: More recent seasons are more important
# 25-26 (current, partial): 0.50
# 24-25 (last full): 0.35
# 23-24 (2 years ago): 0.15
TEMPORAL_WEIGHTS = {
    '2023-24': 0.15,
    '2024-25': 0.35,
    '2025-26': 0.50
}

def get_league_name(comp):
    """Map competition to standardized league name"""
    comp_lower = str(comp).lower()
    
    if 'premier league' in comp_lower or 'gb1' in comp_lower:
        return 'Premier League'
    elif 'la liga' in comp_lower or 'es1' in comp_lower:
        return 'La Liga'
    elif 'bundesliga' in comp_lower or 'l1' in comp_lower or 'de1' in comp_lower:
        return 'Bundesliga'
    elif 'serie a' in comp_lower or 'it1' in comp_lower:
        return 'Serie A'
    elif 'ligue 1' in comp_lower or 'fr1' in comp_lower:
        return 'Ligue 1'
    else:
        return 'Other'

def load_season_data(season_file, season_key):
    """Load and process data for one season"""
    print(f"\n📂 Loading {season_key}...")
    df = pd.read_csv(season_file)
    
    # Filter to players with meaningful playing time (at least 1 full match worth)
    df = df[df['90s'] >= 1.0].copy()
    
    # Add league classification
    df['league'] = df['Comp'].apply(get_league_name)
    
    # Add weights
    df['temporal_weight'] = TEMPORAL_WEIGHTS[season_key]
    df['league_weight'] = df['league'].map(LEAGUE_WEIGHTS)
    df['combined_weight'] = df['temporal_weight'] * df['league_weight']
    
    print(f"   ✅ {len(df)} players (filtered to 90s >= 1.0)")
    print(f"   League distribution:")
    for league in df['league'].value_counts().head(10).items():
        print(f"      {league[0]:20s}: {league[1]:4d} players")
    
    return df

def calculate_per90_stats(df):
    """Calculate per-90 statistics"""
    stats_df = df.copy()
    
    # Already have these from FBRef
    stats_df['goals_per90'] = stats_df['Gls'] / stats_df['90s']
    stats_df['assists_per90'] = stats_df['Ast'] / stats_df['90s']
    
    # Progressive stats (if available)
    if 'PrgC' in stats_df.columns:
        stats_df['progressive_carries_per90'] = stats_df['PrgC'] / stats_df['90s']
    else:
        stats_df['progressive_carries_per90'] = 0
    
    if 'PrgP' in stats_df.columns:
        stats_df['progressive_passes_per90'] = stats_df['PrgP'] / stats_df['90s']
    else:
        stats_df['progressive_passes_per90'] = 0
    
    # Expected stats (if available)
    if 'xG' in stats_df.columns:
        stats_df['xg_per90'] = stats_df['xG'] / stats_df['90s']
    else:
        stats_df['xg_per90'] = 0
    
    if 'xAG' in stats_df.columns:
        stats_df['xag_per90'] = stats_df['xAG'] / stats_df['90s']
    else:
        stats_df['xag_per90'] = 0
    
    return stats_df

def aggregate_player_seasons(player_seasons):
    """
    Aggregate multiple seasons for a player with normalized weights
    
    If player has all 3 seasons: use standard weights
    If player has 2 seasons: normalize weights to sum to 1.0
    If player has 1 season: weight = 1.0
    """
    # Calculate weighted averages
    total_weight = player_seasons['combined_weight'].sum()
    
    # Normalize weights
    player_seasons['normalized_weight'] = player_seasons['combined_weight'] / total_weight
    
    # Key stats to aggregate
    stat_cols = [
        'goals_per90', 'assists_per90', 
        'progressive_carries_per90', 'progressive_passes_per90',
        'xg_per90', 'xag_per90'
    ]
    
    aggregated = {}
    
    for stat in stat_cols:
        if stat in player_seasons.columns:
            weighted_stat = (player_seasons[stat] * player_seasons['normalized_weight']).sum()
            aggregated[f'weighted_{stat}'] = weighted_stat
        else:
            aggregated[f'weighted_{stat}'] = 0
    
    # Metadata
    aggregated['player_name'] = player_seasons.iloc[0]['Player']
    aggregated['team'] = player_seasons.iloc[-1]['Squad']  # Most recent team
    aggregated['position'] = player_seasons.iloc[-1]['Pos']  # Most recent position
    aggregated['age'] = player_seasons.iloc[-1]['Age']  # Current age
    aggregated['seasons_played'] = len(player_seasons)
    aggregated['total_90s'] = player_seasons['90s'].sum()
    aggregated['primary_league'] = player_seasons.iloc[-1]['league']  # Most recent league
    
    # Season breakdown
    aggregated['played_2023_24'] = '2023-24' in player_seasons['season'].values
    aggregated['played_2024_25'] = '2024-25' in player_seasons['season'].values
    aggregated['played_2025_26'] = '2025-26' in player_seasons['season'].values
    
    return aggregated

def main():
    print("="*80)
    print("BUILDING ENHANCED MULTI-SEASON PLAYER PROFILES")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ 3 seasons (23-24, 24-25, 25-26)")
    print("  ✓ Temporal weighting (more recent = higher weight)")
    print("  ✓ League quality weighting")
    print("  ✓ Normalized weights for players with missing seasons")
    print("  ✓ Per-90 statistics")
    
    # Load all seasons
    seasons = {
        '2023-24': '23-24stats/top5-players.csv',
        '2024-25': '24-25stats/players_data_light-2024_2025.csv',
        '2025-26': '25-26stats/players_data_light-2025_2026.csv'
    }
    
    all_data = []
    
    for season_key, file_path in seasons.items():
        if Path(file_path).exists():
            df = load_season_data(file_path, season_key)
            df['season'] = season_key
            
            # Calculate per-90 stats
            df = calculate_per90_stats(df)
            
            all_data.append(df)
        else:
            print(f"   ⚠️  {file_path} not found, skipping...")
    
    # Combine all seasons
    print(f"\n🔄 Combining {len(all_data)} seasons...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"   ✅ Total records: {len(combined_df)}")
    
    # Group by player and team (since players can transfer)
    print(f"\n🔄 Aggregating by player...")
    
    # Create unique player identifier (player name + current team)
    # We'll aggregate all records per player and use most recent team
    player_groups = combined_df.groupby('Player')
    
    profiles = []
    for player_name, group in player_groups:
        # Sort by season to get most recent data last
        group = group.sort_values('season')
        
        profile = aggregate_player_seasons(group)
        profiles.append(profile)
    
    profiles_df = pd.DataFrame(profiles)
    
    print(f"   ✅ {len(profiles_df)} unique players")
    
    # Show statistics
    print(f"\n📊 Profile Statistics:")
    print(f"   Players with all 3 seasons: {profiles_df['seasons_played'].value_counts().get(3, 0):4d} ({profiles_df['seasons_played'].value_counts().get(3, 0)/len(profiles_df)*100:.1f}%)")
    print(f"   Players with 2 seasons:     {profiles_df['seasons_played'].value_counts().get(2, 0):4d} ({profiles_df['seasons_played'].value_counts().get(2, 0)/len(profiles_df)*100:.1f}%)")
    print(f"   Players with 1 season:      {profiles_df['seasons_played'].value_counts().get(1, 0):4d} ({profiles_df['seasons_played'].value_counts().get(1, 0)/len(profiles_df)*100:.1f}%)")
    
    print(f"\n   League distribution:")
    for league, count in profiles_df['primary_league'].value_counts().items():
        print(f"      {league:20s}: {count:4d} players")
    
    # Save
    output_file = 'multi_season_player_profiles.csv'
    profiles_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to {output_file}")
    
    # Show sample profiles
    print(f"\n📋 Sample profiles:")
    sample = profiles_df[profiles_df['seasons_played'] >= 2].head(5)
    display_cols = ['player_name', 'team', 'position', 'seasons_played', 
                    'weighted_goals_per90', 'weighted_assists_per90', 
                    'weighted_progressive_carries_per90']
    print(sample[display_cols].to_string(index=False))
    
    # Check coverage for previously missing teams
    print(f"\n🔍 Checking coverage for previously missing teams:")
    missing_teams = ['Luton', 'Sheffield', 'Almería', 'Cádiz', 'Granada', 
                     'Salernitana', 'Darmstadt', 'Clermont']
    
    for team in missing_teams:
        count = len(profiles_df[profiles_df['team'].str.contains(team, case=False, na=False)])
        status = "✅" if count >= 10 else ("⚠️" if count >= 5 else "❌")
        print(f"   {status} {team:20s}: {count:3d} players")
    
    print("\n" + "="*80)
    print("✅ ENHANCED MULTI-SEASON PROFILES COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
