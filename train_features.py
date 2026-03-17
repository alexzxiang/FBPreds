#!/usr/bin/env python3
"""
Comprehensive Match Prediction Model
Integrates: Player        # Elo ratings
        print("\n📊 Loading Elo ratings...")
        self.elo_ratings = pd.read_csv('elite_leagues_elo_ratings.csv')
        self.elo_dict = dict(zip(self.elo_ratings['team_name'], self.elo_ratings['elo_rating']))
        self.elo_mean = self.elo_ratings['elo_rating'].mean()
        self.elo_std = self.elo_ratings['elo_rating'].std()
        print(f"   ✅ {len(self.elo_dict)} teams with Elo ratings")
        print(f"   Mean: {self.elo_mean:.0f}, Std: {self.elo_std:.0f}")ics, Elo ratings, team form, head-to-head, tactical matchups

This model uses ALL available data to determine feature importance for predicting wins.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveMatchPredictor:
    
    def __init__(self):
        self.player_profiles = None
        self.player_mapping = None
        self.tm_id_to_profile = {}
        self.matches = None
        self.appearances = None
        self.players_db = None
        self.elo_dict = {}
        self.elo_mean = 1500
        self.elo_std = 76
        self.team_form = {}  # Recent form by team
        self.head_to_head = {}  # Historical H2H records
        
    def load_data(self):
        print("="*80)
        print("COMPREHENSIVE MATCH PREDICTION - LOADING ALL DATA")
        print("="*80)
        
        # Player profiles
        print("\n📊 Loading multi-season player profiles...")
        self.player_profiles = pd.read_csv('multi_season_player_profiles.csv')
        print(f"   ✅ {len(self.player_profiles)} FBRef players")
        
        # Player mapping
        print("\n📊 Loading player mapping...")
        with open('player_mapping.pkl', 'rb') as f:
            mapping_data = pickle.load(f)
        self.player_mapping = mapping_data['mapping']
        print(f"   ✅ {len(self.player_mapping)} mapped players")
        
        # Build reverse mapping
        print("\n🔄 Building TM ID to profile cache...")
        for key, tm_id in self.player_mapping.items():
            player_name = key.rsplit('_', 1)[0]
            profile = self.player_profiles[
                self.player_profiles['player_name'] == player_name
            ]
            if len(profile) > 0:
                self.tm_id_to_profile[tm_id] = profile.iloc[0].to_dict()
        print(f"   ✅ {len(self.tm_id_to_profile)} profiles cached")
        
        # Matches
        print("\n📊 Loading matches...")
        self.matches = pd.read_csv('games/games.csv', low_memory=False)
        self.matches['season'] = pd.to_numeric(self.matches['season'], errors='coerce')
        self.matches = self.matches[self.matches['season'] >= 2020].copy()
        self.matches = self.matches[self.matches['home_club_goals'].notna()].copy()
        print(f"   ✅ {len(self.matches)} matches (2020+)")
        
        # Appearances
        print("\n📊 Loading appearances...")
        self.appearances = pd.read_csv('games/appearances.csv', low_memory=False)
        print(f"   ✅ {len(self.appearances)} appearances")
        
        # Players DB
        print("\n📊 Loading Transfermarkt players...")
        self.players_db = pd.read_csv('games/players.csv', low_memory=False)
        print(f"   ✅ {len(self.players_db)} players")
        
        # Elo ratings
        print("\n📊 Loading Elo ratings...")
        self.elo_ratings = pd.read_csv('elite_leagues_elo_ratings.csv')
        self.elo_dict = dict(zip(self.elo_ratings['team_name'], self.elo_ratings['elo_rating']))
        self.elo_mean = self.elo_ratings['elo_rating'].mean()
        self.elo_std = self.elo_ratings['elo_rating'].std()
        print(f"   ✅ {len(self.elo_dict)} teams with Elo ratings")
        print(f"   Mean: {self.elo_mean:.0f}, Std: {self.elo_std:.0f}")
        print(f"   ✅ {len(self.elo_dict)} teams with Elo ratings")
        print(f"   Mean: {self.elo_mean:.0f}, Std: {self.elo_std:.0f}")
        
    def calculate_team_form(self, team_id, current_match_date, n_matches=5):
        """Calculate recent form for a team (last N matches before current date)."""
        team_matches = self.matches[
            ((self.matches['home_club_id'] == team_id) | 
             (self.matches['away_club_id'] == team_id)) &
            (self.matches['date'] < current_match_date)
        ].sort_values('date', ascending=False).head(n_matches)
        
        if len(team_matches) == 0:
            return {
                'form_wins': 0, 'form_draws': 0, 'form_losses': 0,
                'form_goals_scored': 0, 'form_goals_conceded': 0,
                'form_points': 0, 'form_goal_diff': 0
            }
        
        wins = draws = losses = 0
        goals_scored = goals_conceded = 0
        
        for _, match in team_matches.iterrows():
            is_home = match['home_club_id'] == team_id
            team_goals = match['home_club_goals'] if is_home else match['away_club_goals']
            opp_goals = match['away_club_goals'] if is_home else match['home_club_goals']
            
            goals_scored += team_goals
            goals_conceded += opp_goals
            
            if team_goals > opp_goals:
                wins += 1
            elif team_goals == opp_goals:
                draws += 1
            else:
                losses += 1
        
        points = wins * 3 + draws
        
        return {
            'form_wins': wins,
            'form_draws': draws,
            'form_losses': losses,
            'form_goals_scored': goals_scored,
            'form_goals_conceded': goals_conceded,
            'form_points': points,
            'form_goal_diff': goals_scored - goals_conceded
        }
    
    def calculate_away_specific_form(self, team_id, current_match_date, n_matches=10):
        """Calculate form ONLY from away matches (for away team quality)."""
        team_matches = self.matches[
            (self.matches['away_club_id'] == team_id) &  # ONLY away matches
            (self.matches['date'] < current_match_date)
        ].sort_values('date', ascending=False).head(n_matches)
        
        if len(team_matches) == 0:
            return {
                'away_specific_wins': 0,
                'away_specific_draws': 0,
                'away_specific_losses': 0,
                'away_specific_goals_per_game': 0,
                'away_specific_win_rate': 0,
                'away_specific_points_per_game': 0
            }
        
        wins = draws = losses = 0
        goals_scored = goals_conceded = 0
        
        for _, match in team_matches.iterrows():
            away_goals = match['away_club_goals']
            home_goals = match['home_club_goals']
            
            goals_scored += away_goals
            goals_conceded += home_goals
            
            if away_goals > home_goals:
                wins += 1
            elif away_goals == home_goals:
                draws += 1
            else:
                losses += 1
        
        n = len(team_matches)
        points = wins * 3 + draws
        
        return {
            'away_specific_wins': wins,
            'away_specific_draws': draws,
            'away_specific_losses': losses,
            'away_specific_goals_per_game': goals_scored / n if n > 0 else 0,
            'away_specific_win_rate': wins / n if n > 0 else 0,
            'away_specific_points_per_game': points / n if n > 0 else 0
        }
    
    def calculate_home_specific_form(self, team_id, current_match_date, n_matches=10):
        """Calculate form ONLY from home matches (for home team vulnerability)."""
        team_matches = self.matches[
            (self.matches['home_club_id'] == team_id) &  # ONLY home matches
            (self.matches['date'] < current_match_date)
        ].sort_values('date', ascending=False).head(n_matches)
        
        if len(team_matches) == 0:
            return {
                'home_specific_wins': 0,
                'home_specific_draws': 0,
                'home_specific_losses': 0,
                'home_specific_loss_rate': 0,
                'home_specific_win_rate': 0
            }
        
        wins = draws = losses = 0
        
        for _, match in team_matches.iterrows():
            home_goals = match['home_club_goals']
            away_goals = match['away_club_goals']
            
            if home_goals > away_goals:
                wins += 1
            elif home_goals == away_goals:
                draws += 1
            else:
                losses += 1
        
        n = len(team_matches)
        
        return {
            'home_specific_wins': wins,
            'home_specific_draws': draws,
            'home_specific_losses': losses,
            'home_specific_loss_rate': losses / n if n > 0 else 0,
            'home_specific_win_rate': wins / n if n > 0 else 0
        }
    
    def calculate_head_to_head(self, home_id, away_id, current_match_date, n_matches=10):
        """Calculate historical head-to-head record."""
        h2h = self.matches[
            ((self.matches['home_club_id'] == home_id) & (self.matches['away_club_id'] == away_id)) |
            ((self.matches['home_club_id'] == away_id) & (self.matches['away_club_id'] == home_id))
        ]
        h2h = h2h[h2h['date'] < current_match_date].sort_values('date', ascending=False).head(n_matches)
        
        if len(h2h) == 0:
            return {
                'h2h_home_wins': 0, 'h2h_draws': 0, 'h2h_away_wins': 0,
                'h2h_matches': 0, 'h2h_home_goals': 0, 'h2h_away_goals': 0
            }
        
        home_wins = draws = away_wins = 0
        home_goals = away_goals = 0
        away_wins_when_away = 0  # Track how many times "away team" won when playing away
        
        for _, match in h2h.iterrows():
            match_home_id = match['home_club_id']
            
            # Adjust for perspective (our home team vs away team)
            if match_home_id == home_id:
                # Same orientation
                home_goals += match['home_club_goals']
                away_goals += match['away_club_goals']
                if match['home_club_goals'] > match['away_club_goals']:
                    home_wins += 1
                elif match['home_club_goals'] < match['away_club_goals']:
                    away_wins += 1
                    away_wins_when_away += 1  # Away team won while away
                else:
                    draws += 1
            else:
                # Reversed orientation
                home_goals += match['away_club_goals']
                away_goals += match['home_club_goals']
                if match['away_club_goals'] > match['home_club_goals']:
                    home_wins += 1
                elif match['away_club_goals'] < match['home_club_goals']:
                    away_wins += 1
                else:
                    draws += 1
        
        return {
            'h2h_home_wins': home_wins,
            'h2h_draws': draws,
            'h2h_away_wins': away_wins,
            'h2h_matches': len(h2h),
            'h2h_home_goals': home_goals,
            'h2h_away_goals': away_goals,
            'h2h_goal_diff': home_goals - away_goals,
            'h2h_away_wins_when_away': away_wins_when_away  # New: away team's H2H away wins
        }
    
    def normalize_position(self, position):
        """Normalize position to GK/DEF/MID/FWD."""
        if pd.isna(position):
            return 'UNKNOWN'
        pos = str(position).upper()
        if 'GOALKEEPER' in pos or pos == 'GK':
            return 'GK'
        elif 'DEFENDER' in pos or 'DEFENCE' in pos or pos in ['DF', 'CB', 'LB', 'RB']:
            return 'DEF'
        elif 'MIDFIELD' in pos or pos in ['MF', 'CM', 'CDM', 'CAM', 'LM', 'RM']:
            return 'MID'
        elif 'FORWARD' in pos or 'ATTACK' in pos or 'WINGER' in pos or pos in ['FW', 'ST', 'CF', 'LW', 'RW']:
            return 'FWD'
        return 'UNKNOWN'
    
    def create_team_tactical_features(self, game_id, team_id):
        """Create tactical features from player stats (if available)."""
        team_apps = self.appearances[
            (self.appearances['game_id'] == game_id) & 
            (self.appearances['player_club_id'] == team_id)
        ]
        
        if len(team_apps) == 0:
            return None
        
        position_stats = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
        matched = 0
        
        for _, app in team_apps.iterrows():
            player_info = self.players_db[self.players_db['player_id'] == app['player_id']]
            if len(player_info) == 0:
                continue
            
            position = self.normalize_position(player_info.iloc[0]['position'])
            if position == 'UNKNOWN' or position == 'GK':
                continue
            
            profile = self.tm_id_to_profile.get(app['player_id'])
            if profile is None:
                continue
            
            matched += 1
            stats = {
                'goals_per90': profile.get('weighted_goals_per90', 0),
                'assists_per90': profile.get('weighted_assists_per90', 0),
                'prog_carries_per90': profile.get('weighted_progressive_carries_per90', 0),
                'prog_passes_per90': profile.get('weighted_progressive_passes_per90', 0),
                'contributions_per90': profile.get('weighted_goals_per90', 0) + profile.get('weighted_assists_per90', 0)
            }
            position_stats[position].append(stats)
        
        if matched < 3:  # Need at least 3 matched players
            return None
        
        # Aggregate by position
        features = {}
        for pos in ['DEF', 'MID', 'FWD']:
            stats_list = position_stats[pos]
            features[f'{pos.lower()}_count'] = len(stats_list)
            
            if len(stats_list) > 0:
                features[f'{pos.lower()}_avg_goals_per90'] = np.mean([s['goals_per90'] for s in stats_list])
                features[f'{pos.lower()}_avg_assists_per90'] = np.mean([s['assists_per90'] for s in stats_list])
                features[f'{pos.lower()}_avg_prog_carries_per90'] = np.mean([s['prog_carries_per90'] for s in stats_list])
                features[f'{pos.lower()}_avg_prog_passes_per90'] = np.mean([s['prog_passes_per90'] for s in stats_list])
                features[f'{pos.lower()}_avg_contributions_per90'] = np.mean([s['contributions_per90'] for s in stats_list])
                features[f'{pos.lower()}_max_contributions_per90'] = max([s['contributions_per90'] for s in stats_list])
            else:
                for stat in ['goals_per90', 'assists_per90', 'prog_carries_per90', 'prog_passes_per90', 'contributions_per90', 'max_contributions_per90']:
                    features[f'{pos.lower()}_avg_{stat}'] = 0
        
        # Team-level features
        all_contribs = []
        all_prog = []
        for pos in ['DEF', 'MID', 'FWD']:
            for s in position_stats[pos]:
                all_contribs.append(s['contributions_per90'])
                all_prog.append(s['prog_carries_per90'] + s['prog_passes_per90'])
        
        features['team_avg_contributions'] = np.mean(all_contribs) if all_contribs else 0
        features['team_max_contributions'] = max(all_contribs) if all_contribs else 0
        features['team_avg_progressive'] = np.mean(all_prog) if all_prog else 0
        features['matched_players'] = matched
        
        return features
    
    def get_team_elo(self, team_name):
        """Get Elo rating for a team."""
        return self.elo_dict.get(team_name, self.elo_mean)
    
    def create_comprehensive_features(self, match):
        """Create ALL features: tactical, Elo, form, H2H, etc."""
        features = {}
        
        # 1. TACTICAL FEATURES (from player stats)
        home_tactical = self.create_team_tactical_features(match['game_id'], match['home_club_id'])
        away_tactical = self.create_team_tactical_features(match['game_id'], match['away_club_id'])
        
        has_tactical = (home_tactical is not None and away_tactical is not None)
        
        if has_tactical:
            # Add tactical features with prefixes
            for k, v in home_tactical.items():
                features[f'home_{k}'] = v
            for k, v in away_tactical.items():
                features[f'away_{k}'] = v
            
            # Tactical differentials
            features['quality_diff'] = home_tactical['team_avg_contributions'] - away_tactical['team_avg_contributions']
            features['star_diff'] = home_tactical['team_max_contributions'] - away_tactical['team_max_contributions']
            features['progressive_diff'] = home_tactical['team_avg_progressive'] - away_tactical['team_avg_progressive']
            features['mid_creativity_diff'] = home_tactical['mid_avg_assists_per90'] - away_tactical['mid_avg_assists_per90']
            features['home_fwd_vs_away_def'] = home_tactical['fwd_avg_contributions_per90'] - away_tactical['def_avg_prog_carries_per90']
            features['away_fwd_vs_home_def'] = away_tactical['fwd_avg_contributions_per90'] - home_tactical['def_avg_prog_carries_per90']
        else:
            # Fill with zeros if no tactical data
            for pos in ['def', 'mid', 'fwd']:
                for prefix in ['home', 'away']:
                    features[f'{prefix}_{pos}_count'] = 0
                    for stat in ['goals_per90', 'assists_per90', 'prog_carries_per90', 'prog_passes_per90', 'contributions_per90', 'max_contributions_per90']:
                        features[f'{prefix}_{pos}_avg_{stat}'] = 0
                features[f'home_team_avg_contributions'] = 0
                features[f'away_team_avg_contributions'] = 0
            features['quality_diff'] = 0
            features['star_diff'] = 0
            features['progressive_diff'] = 0
        
        # 2. ELO FEATURES
        home_elo = self.get_team_elo(match['home_club_name'])
        away_elo = self.get_team_elo(match['away_club_name'])
        
        features['home_elo'] = home_elo
        features['away_elo'] = away_elo
        features['elo_diff'] = home_elo - away_elo
        features['elo_ratio'] = home_elo / away_elo if away_elo > 0 else 1.0
        features['home_elo_normalized'] = (home_elo - self.elo_mean) / self.elo_std
        features['away_elo_normalized'] = (away_elo - self.elo_mean) / self.elo_std
        
        # Calibrated home advantage (analysis shows home advantage = ~45 Elo)
        HOME_ADVANTAGE_ELO = 45
        features['adjusted_elo_diff'] = (home_elo + HOME_ADVANTAGE_ELO) - away_elo
        
        # ASYMMETRIC AWAY-SPECIFIC FEATURES (based on analysis)
        # Analysis showed: 29.6% of away wins happen when home has higher Elo
        # This means absolute team quality matters, not just difference
        
        # Away team absolute quality (percentile among all teams)
        features['away_elo_percentile'] = (away_elo - self.elo_ratings['elo_rating'].min()) / (self.elo_ratings['elo_rating'].max() - self.elo_ratings['elo_rating'].min())
        features['home_elo_percentile'] = (home_elo - self.elo_ratings['elo_rating'].min()) / (self.elo_ratings['elo_rating'].max() - self.elo_ratings['elo_rating'].min())
        
        # Elite team flags (analysis shows elite teams overcome home advantage easier)
        features['away_is_elite'] = 1 if away_elo > 1900 else 0
        features['home_is_elite'] = 1 if home_elo > 1900 else 0
        features['away_is_weak'] = 1 if away_elo < 1450 else 0
        features['home_is_weak'] = 1 if home_elo < 1450 else 0
        
        # Elo advantage asymmetry (captures absolute quality, not just difference)
        # Example: Away 2000 vs Home 1900 = (+100, -100) asymmetry shows away is elite
        #          Away 1600 vs Home 1500 = (+100, -100) same diff but away is weak
        features['away_elo_from_mean'] = away_elo - self.elo_mean
        features['home_elo_from_mean'] = home_elo - self.elo_mean
        features['elo_asymmetry'] = (away_elo - self.elo_mean) - (home_elo - self.elo_mean)
        
        # 3. FORM FEATURES
        home_form = self.calculate_team_form(match['home_club_id'], match['date'])
        away_form = self.calculate_team_form(match['away_club_id'], match['date'])
        
        for k, v in home_form.items():
            features[f'home_{k}'] = v
        for k, v in away_form.items():
            features[f'away_{k}'] = v
        
        # Form differentials
        features['form_points_diff'] = home_form['form_points'] - away_form['form_points']
        features['form_goal_diff_diff'] = home_form['form_goal_diff'] - away_form['form_goal_diff']
        features['form_wins_diff'] = home_form['form_wins'] - away_form['form_wins']
        
        # 3b. AWAY/HOME SPECIFIC FORM (based on analysis showing location-specific patterns)
        away_specific = self.calculate_away_specific_form(match['away_club_id'], match['date'])
        home_specific = self.calculate_home_specific_form(match['home_club_id'], match['date'])
        
        for k, v in away_specific.items():
            features[k] = v
        for k, v in home_specific.items():
            features[k] = v
        
        # Away team quality metrics (analysis: away teams need to be MUCH stronger)
        # These capture: "How good is this away team at winning away matches?"
        features['away_quality_advantage'] = (
            away_specific['away_specific_win_rate'] * 100 -  # Away team's away win rate
            home_specific['home_specific_loss_rate'] * 100   # Home team's home loss rate
        )
        
        # 4. HEAD-TO-HEAD FEATURES
        h2h = self.calculate_head_to_head(match['home_club_id'], match['away_club_id'], match['date'])
        
        for k, v in h2h.items():
            features[k] = v
        
        # H2H win rate
        if h2h['h2h_matches'] > 0:
            features['h2h_home_win_rate'] = h2h['h2h_home_wins'] / h2h['h2h_matches']
            features['h2h_away_win_rate'] = h2h['h2h_away_wins'] / h2h['h2h_matches']
            features['h2h_away_dominance'] = h2h['h2h_away_wins_when_away'] / h2h['h2h_matches']
        else:
            features['h2h_home_win_rate'] = 0.33
            features['h2h_away_win_rate'] = 0.33
            features['h2h_away_dominance'] = 0.0
        
        # 5. METADATA FEATURES
        features['is_home'] = 1  # Always 1 for this perspective
        features['has_tactical_data'] = 1 if has_tactical else 0
        
        return features
    
    def prepare_training_data(self, max_matches=20000):
        """Prepare comprehensive training data."""
        print("\n" + "="*80)
        print("PREPARING COMPREHENSIVE TRAINING DATA (2020-2026)")
        print("="*80)
        print(f"Using up to {max_matches:,} matches")
        print("Features: Tactical + Elo + Form + H2H + Metadata")
        print("Data includes 2025-26 season for improved future predictions")
        
        X_data = []
        y_data = []
        
        # Sort by date to ensure temporal ordering
        sample = self.matches.sort_values('date').tail(max_matches)
        print(f"\n🔄 Processing {len(sample):,} matches...", flush=True)
        
        processed = 0
        skipped = 0
        
        for idx, (_, match) in enumerate(sample.iterrows()):
            if idx % 2000 == 0:
                print(f"   {idx:,}/{len(sample):,} ({processed:,} valid, {skipped:,} skipped)", flush=True)
            
            try:
                features = self.create_comprehensive_features(match)
                
                if features is None:
                    skipped += 1
                    continue
                
                # Determine outcome
                if match['home_club_goals'] > match['away_club_goals']:
                    outcome = 2  # Home win
                elif match['home_club_goals'] < match['away_club_goals']:
                    outcome = 0  # Away win
                else:
                    outcome = 1  # Draw
                
                X_data.append(features)
                y_data.append(outcome)
                processed += 1
                
            except Exception as e:
                if idx < 10:  # Print first few errors for debugging
                    print(f"   ⚠️  Error processing match {idx}: {e}")
                skipped += 1
                continue
        
        print(f"\n✅ Valid: {processed:,}, Skipped: {skipped:,}, Success rate: {processed/(processed+skipped)*100:.1f}%")
        
        if processed == 0:
            print("\n❌ No valid matches found!")
            return None, None
        
        X = pd.DataFrame(X_data)
        y = pd.Series(y_data)
        
        print(f"\n📊 Outcome distribution:")
        print(f"   Home wins: {sum(y==2):,} ({sum(y==2)/len(y)*100:.1f}%)")
        print(f"   Draws: {sum(y==1):,} ({sum(y==1)/len(y)*100:.1f}%)")
        print(f"   Away wins: {sum(y==0):,} ({sum(y==0)/len(y)*100:.1f}%)")
        
        print(f"\n📊 Feature summary:")
        print(f"   Total features: {len(X.columns)}")
        print(f"   Tactical features: ~50 (if available)")
        print(f"   Elo features: 7")
        print(f"   Form features: ~15")
        print(f"   H2H features: ~9")
        print(f"   Metadata: 2")
        
        # Check tactical data coverage
        tactical_coverage = X['has_tactical_data'].mean()
        print(f"\n   Matches with tactical data: {tactical_coverage*100:.1f}%")
        
        return X, y
    
    def train_models(self, X, y):
        """Train comprehensive RF and XGBoost models."""
        print("\n" + "="*80)
        print("TRAINING COMPREHENSIVE MODELS")
        print("="*80)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        print(f"\nTrain: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Random Forest
        print(f"\n🌲 Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_sc, y_train)
        rf_test = accuracy_score(y_test, rf.predict(X_test_sc))
        rf_cv = cross_val_score(rf, X_train_sc, y_train, cv=5, n_jobs=-1)
        print(f"   Test accuracy: {rf_test:.1%}")
        print(f"   CV accuracy: {rf_cv.mean():.1%} (+/- {rf_cv.std()*2:.1%})")
        
        # XGBoost with class weights
        print(f"\n🚀 Training XGBoost...")
        
        # Calculate class weights for XGBoost
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight('balanced', y_train)
        
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            scale_pos_weight=None  # We'll use sample_weight instead
        )
        xgb.fit(X_train_sc, y_train, sample_weight=sample_weights)
        xgb_test = accuracy_score(y_test, xgb.predict(X_test_sc))
        xgb_cv = cross_val_score(xgb, X_train_sc, y_train, cv=5, n_jobs=-1)
        print(f"   Test accuracy: {xgb_test:.1%}")
        print(f"   CV accuracy: {xgb_cv.mean():.1%} (+/- {xgb_cv.std()*2:.1%})")
        
        # Detailed performance
        print("\n📊 XGBoost Detailed Performance:")
        print(classification_report(
            y_test, xgb.predict(X_test_sc),
            target_names=['Away Win', 'Draw', 'Home Win']
        ))
        
        return rf, xgb, scaler, X
    
    def extract_insights(self, rf, xgb, X):
        """Extract comprehensive feature importance."""
        print("\n" + "="*80)
        print("COMPREHENSIVE FEATURE IMPORTANCE")
        print("="*80)
        
        rf_importance = rf.feature_importances_
        xgb_importance = xgb.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'rf_importance': rf_importance,
            'xgb_importance': xgb_importance,
            'combined_importance': (rf_importance + xgb_importance) / 2
        })
        importance_df = importance_df.sort_values('combined_importance', ascending=False)
        
        print("\n🔍 TOP 40 MOST IMPORTANT FEATURES:")
        for idx, row in importance_df.head(40).iterrows():
            print(f"   {row['feature']:50s} RF:{row['rf_importance']:.4f} XGB:{row['xgb_importance']:.4f} Avg:{row['combined_importance']:.4f}")
        
        # Category analysis
        print("\n📊 FEATURE IMPORTANCE BY CATEGORY:")
        
        categories = {
            'Elo': importance_df[importance_df['feature'].str.contains('elo', case=False)],
            'Form': importance_df[importance_df['feature'].str.contains('form', case=False)],
            'H2H': importance_df[importance_df['feature'].str.contains('h2h', case=False)],
            'Tactical': importance_df[
                (importance_df['feature'].str.contains('def|mid|fwd|quality|star|progressive', case=False)) &
                (~importance_df['feature'].str.contains('elo|form|h2h', case=False))
            ],
        }
        
        for cat_name, cat_df in categories.items():
            if len(cat_df) > 0:
                avg_importance = cat_df['combined_importance'].mean()
                top_feature = cat_df.iloc[0]
                print(f"\n{cat_name}:")
                print(f"   Features: {len(cat_df)}")
                print(f"   Avg importance: {avg_importance:.4f}")
                print(f"   Top feature: {top_feature['feature']} ({top_feature['combined_importance']:.4f})")
        
        return importance_df
    
    def save_models(self, rf, xgb, scaler, importance_df):
        """Save comprehensive models."""
        print("\n" + "="*80)
        print("SAVING MODELS")
        print("="*80)
        
        importance_df.to_csv('comprehensive_feature_importance.csv', index=False)
        print("✅ Saved comprehensive_feature_importance.csv")
        
        models = {
            'rf_model': rf,
            'xgb_model': xgb,
            'scaler': scaler,
            'feature_names': list(importance_df['feature']),
            'elo_dict': self.elo_dict,
            'elo_mean': self.elo_mean,
            'elo_std': self.elo_std,
            'HOME_ADVANTAGE_ELO': 30
        }
        
        Path('models').mkdir(exist_ok=True)
        with open('models/comprehensive_match_predictor.pkl', 'wb') as f:
            pickle.dump(models, f)
        
        print("✅ Saved models/comprehensive_match_predictor.pkl")


def main():
    predictor = ComprehensiveMatchPredictor()
    predictor.load_data()
    
    # Use ALL available data (2020-2026, ~32k matches)
    X, y = predictor.prepare_training_data(max_matches=35000)
    
    if X is None:
        print("\n❌ No training data available")
        return
    
    rf, xgb, scaler, X_orig = predictor.train_models(X, y)
    importance_df = predictor.extract_insights(rf, xgb, X_orig)
    predictor.save_models(rf, xgb, scaler, importance_df)
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE MODEL TRAINING COMPLETE")
    print("="*80)
    print("\nThis model integrates:")
    print("  • Player statistics (tactical features)")
    print("  • Elo ratings (team quality)")
    print("  • Recent form (last 5 matches)")
    print("  • Head-to-head history")
    print("  • Home advantage")
    print("\nUse models/comprehensive_match_predictor.pkl for predictions!")


if __name__ == '__main__':
    main()
