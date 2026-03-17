#!/usr/bin/env python3
"""
Build comprehensive multi-season player profiles using optimal temporal weighting.

Combines data from all 4 seasons (22-23, 23-24, 24-25, 25-26) with exponential
decay weighting to create present-day player ratings.

Weighting scheme (systematically optimized):
- 25-26 (current): 43.9%
- 24-25 (1 yr ago): 27.7%
- 23-24 (2 yr ago): 17.4%
- 22-23 (3 yr ago): 11.0%
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class MultiSeasonProfileBuilder:
    """Build player profiles from multiple seasons with temporal weighting."""
    
    def __init__(self):
        self.temporal_weights = self.load_temporal_weights()
        self.ml_models = self.load_ml_models()
        self.seasons_data = {}
        self.player_profiles = {}
        
    def load_temporal_weights(self):
        """Load the systematically optimized temporal weights."""
        weights_path = Path('models/temporal_weights.pkl')
        
        with open(weights_path, 'rb') as f:
            weights_data = pickle.load(f)
        
        print("✅ Loaded temporal weights:")
        print(f"   Scheme: {weights_data['scheme_name']}")
        print(f"   Function: {weights_data['function']}")
        for season, weight in weights_data['weights_by_season'].items():
            print(f"   {season}: {weight:.3f} ({weight*100:.1f}%)")
        
        return weights_data
    
    def load_ml_models(self):
        """Load ML feature importance models."""
        models_path = Path('models/feature_importance_models.pkl')
        
        if not models_path.exists():
            print("⚠️  ML models not found")
            return None
        
        with open(models_path, 'rb') as f:
            models = pickle.load(f)
        
        print(f"\n✅ Loaded ML models for: {list(models.keys())}")
        return models
    
    def normalize_position(self, pos_str):
        """Normalize position strings."""
        if pd.isna(pos_str):
            return 'UNKNOWN'
        
        pos_str = str(pos_str).upper().strip()
        
        if 'GK' in pos_str:
            return 'GK'
        elif any(x in pos_str for x in ['DF', 'DEF', 'CB', 'LB', 'RB', 'WB']):
            return 'DEF'
        elif any(x in pos_str for x in ['FW', 'FWD', 'ST', 'CF', 'LW', 'RW']):
            return 'FWD'
        elif any(x in pos_str for x in ['MF', 'MID', 'CM', 'DM', 'AM']):
            return 'MID'
        elif ',' in pos_str:
            parts = pos_str.split(',')
            if 'FW' in parts or 'FWD' in parts:
                return 'FWD'
            return 'MID'
        
        return 'MID'
    
    def load_season_22_23(self):
        """Load 2022-23 data (semicolon-separated, Latin-1)."""
        filepath = Path('2022-2023 Football Player Stats.csv')
        
        if not filepath.exists():
            print("⚠️  22-23 data not found")
            return None
        
        df = pd.read_csv(filepath, sep=';', encoding='latin-1', low_memory=False)
        
        # Normalize
        df['position'] = df['Pos'].apply(self.normalize_position)
        df['player_name'] = df['Player'].str.strip()
        df['team'] = df.get('Squad', 'Unknown')
        df['competition'] = df.get('Comp', 'Unknown')
        
        # Key stats - use column directly, not .get()
        df['matches'] = pd.to_numeric(df['MP'] if 'MP' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['minutes'] = pd.to_numeric(df['Min'] if 'Min' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['goals'] = pd.to_numeric(df['Gls'] if 'Gls' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['assists'] = pd.to_numeric(df['Ast'] if 'Ast' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['progressive_carries'] = pd.to_numeric(df['PrgC'] if 'PrgC' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['progressive_passes'] = pd.to_numeric(df['PrgP'] if 'PrgP' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        
        print(f"\n✅ Loaded 22-23: {len(df)} players")
        return df
    
    def load_season_23_24(self):
        """Load 2023-24 data (comma-separated, UTF-8, simple format)."""
        filepath = Path('23-24stats/top5-players.csv')
        
        if not filepath.exists():
            print("⚠️  23-24 data not found")
            return None
        
        df = pd.read_csv(filepath, encoding='utf-8')
        
        # Normalize
        df['position'] = df['Pos'].apply(self.normalize_position)
        df['player_name'] = df['Player'].str.strip()
        df['team'] = df.get('Squad', 'Unknown')
        df['competition'] = df.get('Comp', 'Unknown')
        
        # Key stats
        df['matches'] = pd.to_numeric(df['MP'] if 'MP' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['minutes'] = pd.to_numeric(df['Min'] if 'Min' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['goals'] = pd.to_numeric(df['Gls'] if 'Gls' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['assists'] = pd.to_numeric(df['Ast'] if 'Ast' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['progressive_carries'] = pd.to_numeric(df['PrgC'] if 'PrgC' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['progressive_passes'] = pd.to_numeric(df['PrgP'] if 'PrgP' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['xG'] = pd.to_numeric(df['xG'] if 'xG' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['xAG'] = pd.to_numeric(df['xAG'] if 'xAG' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        
        print(f"✅ Loaded 23-24: {len(df)} players")
        return df
    
    def load_season_24_25(self):
        """Load 2024-25 data (comprehensive format)."""
        filepath = Path('24-25stats/players_data-2024_2025.csv')
        
        if not filepath.exists():
            print("⚠️  24-25 data not found")
            return None
        
        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
        
        # Normalize
        df['position'] = df['Pos'].apply(self.normalize_position)
        df['player_name'] = df['Player'].str.strip()
        df['team'] = df.get('Squad', 'Unknown')
        df['competition'] = df.get('Comp', 'Unknown')
        
        # Key stats
        df['matches'] = pd.to_numeric(df['MP'] if 'MP' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['minutes'] = pd.to_numeric(df['Min'] if 'Min' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['goals'] = pd.to_numeric(df['Gls'] if 'Gls' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['assists'] = pd.to_numeric(df['Ast'] if 'Ast' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['progressive_carries'] = pd.to_numeric(df['PrgC'] if 'PrgC' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['progressive_passes'] = pd.to_numeric(df['PrgP'] if 'PrgP' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['xG'] = pd.to_numeric(df['xG'] if 'xG' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['xAG'] = pd.to_numeric(df['xAG'] if 'xAG' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        
        print(f"✅ Loaded 24-25: {len(df)} players")
        return df
    
    def load_season_25_26(self):
        """Load 2025-26 data (comprehensive format, current season)."""
        filepath = Path('25-26stats/players_data-2025_2026.csv')
        
        if not filepath.exists():
            print("⚠️  25-26 data not found")
            return None
        
        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
        
        # Normalize
        df['position'] = df['Pos'].apply(self.normalize_position)
        df['player_name'] = df['Player'].str.strip()
        df['team'] = df.get('Squad', 'Unknown')
        df['competition'] = df.get('Comp', 'Unknown')
        
        # Key stats
        df['matches'] = pd.to_numeric(df['MP'] if 'MP' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['minutes'] = pd.to_numeric(df['Min'] if 'Min' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['goals'] = pd.to_numeric(df['Gls'] if 'Gls' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['assists'] = pd.to_numeric(df['Ast'] if 'Ast' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['progressive_carries'] = pd.to_numeric(df['PrgC'] if 'PrgC' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['progressive_passes'] = pd.to_numeric(df['PrgP'] if 'PrgP' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['xG'] = pd.to_numeric(df['xG'] if 'xG' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        df['xAG'] = pd.to_numeric(df['xAG'] if 'xAG' in df.columns else pd.Series([0]*len(df)), errors='coerce').fillna(0)
        
        print(f"✅ Loaded 25-26: {len(df)} players (current season)")
        return df
    
    def fuzzy_match_player(self, name, team, possible_matches):
        """Fuzzy match player across seasons (name + team)."""
        # Clean name
        name_clean = name.lower().strip()
        team_clean = team.lower().strip()
        
        # Exact match first
        for match in possible_matches:
            if match['name'].lower().strip() == name_clean:
                if match['team'].lower().strip() == team_clean:
                    return match['key'], 1.0  # Perfect match
        
        # Name match (different teams - transfers)
        for match in possible_matches:
            if match['name'].lower().strip() == name_clean:
                return match['key'], 0.8  # Name match
        
        return None, 0.0
    
    def build_profiles(self):
        """Build multi-season profiles with temporal weighting."""
        
        print("\n" + "="*80)
        print("BUILDING MULTI-SEASON PLAYER PROFILES")
        print("="*80)
        
        # Load all seasons
        self.seasons_data['22-23'] = self.load_season_22_23()
        self.seasons_data['23-24'] = self.load_season_23_24()
        self.seasons_data['24-25'] = self.load_season_24_25()
        self.seasons_data['25-26'] = self.load_season_25_26()
        
        # Remove None values
        self.seasons_data = {k: v for k, v in self.seasons_data.items() if v is not None}
        
        print(f"\n✅ Loaded {len(self.seasons_data)} seasons")
        
        # Build player index from most recent season
        player_index = {}
        
        if '25-26' in self.seasons_data:
            current_df = self.seasons_data['25-26']
            print(f"\n📊 Using 25-26 as base ({len(current_df)} players)")
            
            for _, row in current_df.iterrows():
                if row['matches'] >= 1:  # At least 1 match
                    key = f"{row['player_name']}_{row['team']}"
                    player_index[key] = {
                        'name': row['player_name'],
                        'team': row['team'],
                        'position': row['position'],
                        'competition': row['competition'],
                        'seasons': {}
                    }
        
        print(f"✅ Created index for {len(player_index)} active players")
        
        # Gather stats from each season
        weights = self.temporal_weights['weights_by_season']
        
        for season_key, df in self.seasons_data.items():
            print(f"\n📊 Processing {season_key} (weight: {weights.get(season_key, 0):.1%})...")
            
            matched = 0
            for _, row in df.iterrows():
                if row['matches'] < 1:
                    continue
                
                key = f"{row['player_name']}_{row['team']}"
                
                if key in player_index:
                    # Exact match
                    player_index[key]['seasons'][season_key] = row.to_dict()
                    matched += 1
            
            print(f"   ✅ Matched {matched} players from {season_key}")
        
        # Calculate weighted stats
        print(f"\n🔄 Calculating weighted multi-season stats...")
        
        profiles = []
        
        for key, player in player_index.items():
            if len(player['seasons']) == 0:
                continue
            
            # Calculate weighted averages
            total_weight = 0
            weighted_stats = defaultdict(float)
            
            for season_key, season_data in player['seasons'].items():
                weight = weights.get(season_key, 0)
                total_weight += weight
                
                # Weight each stat
                for stat in ['matches', 'minutes', 'goals', 'assists', 'progressive_carries', 'progressive_passes', 'xG', 'xAG']:
                    value = season_data.get(stat, 0)
                    if pd.notna(value):
                        weighted_stats[stat] += float(value) * weight
            
            # Normalize by total weight
            if total_weight > 0:
                for stat in weighted_stats:
                    weighted_stats[stat] /= total_weight
            
            # Create profile
            profile = {
                'player_name': player['name'],
                'team': player['team'],
                'position': player['position'],
                'competition': player['competition'],
                'seasons_count': len(player['seasons']),
                'seasons_list': ','.join(sorted(player['seasons'].keys())),
                'total_weight': total_weight,
                **{f'weighted_{k}': v for k, v in weighted_stats.items()}
            }
            
            # Add recency score (more recent data = higher score)
            if '25-26' in player['seasons']:
                profile['recency_score'] = 1.0
            elif '24-25' in player['seasons']:
                profile['recency_score'] = 0.7
            elif '23-24' in player['seasons']:
                profile['recency_score'] = 0.4
            else:
                profile['recency_score'] = 0.2
            
            profiles.append(profile)
        
        print(f"✅ Created {len(profiles)} multi-season profiles")
        
        # Convert to DataFrame
        profiles_df = pd.DataFrame(profiles)
        
        # Calculate per-90 stats
        profiles_df['weighted_90s'] = profiles_df['weighted_minutes'] / 90
        
        for stat in ['goals', 'assists', 'progressive_carries', 'progressive_passes']:
            profiles_df[f'weighted_{stat}_per90'] = (
                profiles_df[f'weighted_{stat}'] / profiles_df['weighted_90s'].replace(0, 1)
            )
        
        # Export
        output_path = Path('multi_season_player_profiles.csv')
        profiles_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Exported to: {output_path}")
        
        # Show summary
        print("\n" + "="*80)
        print("MULTI-SEASON PROFILES SUMMARY")
        print("="*80)
        
        print(f"\nTotal profiles: {len(profiles_df)}")
        print(f"\nBy position:")
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            count = len(profiles_df[profiles_df['position'] == pos])
            print(f"   {pos}: {count} players")
        
        print(f"\nBy seasons covered:")
        for i in range(1, 5):
            count = len(profiles_df[profiles_df['seasons_count'] == i])
            print(f"   {i} season(s): {count} players")
        
        print(f"\nBy recency:")
        print(f"   Active in 25-26: {len(profiles_df[profiles_df['recency_score'] == 1.0])}")
        print(f"   Last seen 24-25: {len(profiles_df[profiles_df['recency_score'] == 0.7])}")
        print(f"   Last seen 23-24: {len(profiles_df[profiles_df['recency_score'] == 0.4])}")
        print(f"   Last seen 22-23: {len(profiles_df[profiles_df['recency_score'] == 0.2])}")
        
        # Top performers
        print("\n" + "="*80)
        print("TOP PERFORMERS (weighted multi-season stats)")
        print("="*80)
        
        qualified = profiles_df[profiles_df['weighted_matches'] >= 10].copy()
        
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            pos_df = qualified[qualified['position'] == pos]
            
            if pos == 'GK':
                # Sort GK by matches played
                top = pos_df.nlargest(10, 'weighted_matches')
            else:
                # Sort by goal contributions per 90
                pos_df['weighted_contributions_per90'] = (
                    pos_df['weighted_goals_per90'] + pos_df['weighted_assists_per90']
                )
                top = pos_df.nlargest(10, 'weighted_contributions_per90')
            
            print(f"\n{pos} Top 10:")
            for i, (_, row) in enumerate(top.iterrows(), 1):
                if pos == 'GK':
                    print(f"   {i}. {row['player_name']} ({row['team']}) - "
                          f"{row['weighted_matches']:.1f} matches, "
                          f"{row['seasons_count']} seasons")
                else:
                    print(f"   {i}. {row['player_name']} ({row['team']}) - "
                          f"{row['weighted_goals_per90']:.2f} G/90, "
                          f"{row['weighted_assists_per90']:.2f} A/90, "
                          f"{row['weighted_contributions_per90']:.2f} G+A/90")
        
        return profiles_df


def main():
    builder = MultiSeasonProfileBuilder()
    profiles_df = builder.build_profiles()
    
    print("\n" + "="*80)
    print("✅ MULTI-SEASON PROFILE BUILDING COMPLETE")
    print("="*80)
    print(f"\n📊 {len(profiles_df)} player profiles created")
    print(f"📁 Saved to: multi_season_player_profiles.csv")
    print(f"\n💡 Next step: Use these profiles for match predictions")


if __name__ == '__main__':
    main()
