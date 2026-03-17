"""
Enhanced Player Profiler using ML-derived feature importance weights.
Uses XGBoost and Random Forest to determine actual win contribution.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Optional
from collections import defaultdict

class MLEnhancedPlayerProfiler:
    """
    Player profiler that uses machine learning to determine:
    1. Which stats matter most for each position
    2. How much each stat contributes to winning
    3. Player's actual win contribution score
    """
    
    def __init__(self):
        self.feature_models = None
        self.player_profiles = defaultdict(lambda: {
            'player_name': '',
            'position': 'MID',
            'stats': {},
            'win_contribution_score': 0.0,
            'predicted_win_probability': 0.5,
            'matches': 0,
            'actual_wins': 0,
            'actual_draws': 0,
            'actual_losses': 0,
            'actual_win_rate': 0.0,
            'actual_ppg': 0.0,
            'minutes': 0.0,
            'match_history': []
        })
        
    def load_feature_importance_models(self, model_path='models/feature_importance_models.pkl'):
        """Load pre-trained XGBoost and Random Forest models"""
        try:
            with open(model_path, 'rb') as f:
                self.feature_models = pickle.load(f)
            print(f"✅ Loaded feature importance models for {len(self.feature_models)} positions")
            return True
        except Exception as e:
            print(f"⚠️  Could not load feature importance models: {e}")
            return False
    
    def calculate_win_contribution_score(self, player_stats: Dict, position: str) -> float:
        """
        Calculate a player's win contribution score using ML feature importance.
        
        Args:
            player_stats: Dictionary of player statistics
            position: Player position (GK, DEF, MID, FWD)
        
        Returns:
            Win contribution score (0-10)
        """
        if not self.feature_models or position not in self.feature_models:
            # Fallback to basic calculation
            return self._calculate_basic_win_contribution(player_stats, position)
        
        model_data = self.feature_models[position]
        
        # Get feature importances (average of RF and XGBoost)
        rf_importance = model_data['rf_importance']
        xgb_importance = model_data['xgb_importance']
        
        # Create combined importance scores
        importance_dict = {}
        for _, row in rf_importance.iterrows():
            importance_dict[row['feature']] = row['importance']
        
        for _, row in xgb_importance.iterrows():
            if row['feature'] in importance_dict:
                importance_dict[row['feature']] = (importance_dict[row['feature']] + row['importance']) / 2
            else:
                importance_dict[row['feature']] = row['importance']
        
        # Calculate weighted score based on player's stats and feature importance
        total_score = 0.0
        total_weight = 0.0
        
        for feature, importance in importance_dict.items():
            if feature in player_stats:
                stat_value = player_stats[feature]
                
                # Normalize stat value (simple min-max based on expected ranges)
                normalized_value = self._normalize_stat(feature, stat_value, position)
                
                # Weight by importance
                weighted_value = normalized_value * importance
                total_score += weighted_value
                total_weight += importance
        
        # Scale to 0-10
        if total_weight > 0:
            score = (total_score / total_weight) * 10
        else:
            score = 5.0
        
        return min(10.0, max(0.0, score))
    
    def _normalize_stat(self, stat_name: str, value: float, position: str) -> float:
        """Normalize a stat value to 0-1 range based on expected ranges"""
        # Define expected ranges for common stats
        ranges = {
            # Playing time
            'MP': (0, 38), '90s': (0, 38), 'Min': (0, 3420),
            
            # Goals & Assists
            'Gls': (0, 30), 'Ast': (0, 20), 'G+A': (0, 40),
            
            # Passing
            'PasTotCmp': (0, 2500), 'PasTotAtt': (0, 3000), 'PasTotCmp%': (0, 100),
            'PasProg': (0, 300), 'PPA': (0, 100),
            
            # Shooting
            'Sh': (0, 150), 'SoT': (0, 60), 'SoT%': (0, 100),
            'G/Sh': (0, 0.5), 'G/SoT': (0, 0.8),
            
            # Defending
            'Tkl': (0, 100), 'TklWon': (0, 80), 'Int': (0, 100),
            'Blocks': (0, 50), 'Clr': (0, 200),
            
            # Aerial
            'AerWon': (0, 150), 'AerLost': (0, 150), 'AerWon%': (0, 100),
            
            # Creating
            'SCA': (0, 150), 'GCA': (0, 30),
            
            # Progression
            'Carries': (0, 2000), 'PrgC': (0, 300), 'PrgP': (0, 300),
            
            # Discipline
            'CrdY': (0, 15), 'CrdR': (0, 3),
        }
        
        if stat_name in ranges:
            min_val, max_val = ranges[stat_name]
            normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            return min(1.0, max(0.0, normalized))
        
        # Default normalization for unknown stats
        return min(1.0, max(0.0, value / 100))
    
    def _calculate_basic_win_contribution(self, player_stats: Dict, position: str) -> float:
        """Fallback basic win contribution calculation"""
        # Position-specific weights (from our earlier analysis)
        if position == 'FWD':
            score = (
                player_stats.get('Gls', 0) * 0.4 +
                player_stats.get('Ast', 0) * 0.2 +
                player_stats.get('SoT', 0) * 0.02 +
                player_stats.get('GCA', 0) * 0.15
            )
        elif position == 'MID':
            score = (
                player_stats.get('PasProg', 0) * 0.02 +
                player_stats.get('SCA', 0) * 0.1 +
                player_stats.get('Ast', 0) * 0.2 +
                player_stats.get('Gls', 0) * 0.15
            )
        elif position == 'DEF':
            score = (
                player_stats.get('Tkl', 0) * 0.05 +
                player_stats.get('Int', 0) * 0.05 +
                player_stats.get('Clr', 0) * 0.02 +
                player_stats.get('Blocks', 0) * 0.05
            )
        else:  # GK
            score = (
                player_stats.get('Save%', 0) * 0.05 +
                player_stats.get('CS', 0) * 0.2
            )
        
        return min(10.0, max(0.0, score))
    
    def predict_win_probability(self, player_stats: Dict, position: str) -> float:
        """
        Use ML models to predict win probability when this player plays.
        
        Returns:
            Probability of winning (0-1) when this player is in the lineup
        """
        if not self.feature_models or position not in self.feature_models:
            # Fallback: use win contribution score
            win_score = self.calculate_win_contribution_score(player_stats, position)
            return 0.3 + (win_score / 10) * 0.4  # Range: 0.3 to 0.7
        
        model_data = self.feature_models[position]
        
        # Prepare features for prediction
        features = model_data['features']
        X = []
        for feature in features:
            X.append(player_stats.get(feature, 0))
        
        X = np.array([X])
        
        # Use both models and average predictions
        try:
            # Random Forest prediction
            rf_proba = model_data['rf_model'].predict_proba(X)[0]
            rf_win_prob = rf_proba[1] if len(rf_proba) > 1 else 0.5
            
            # XGBoost prediction (needs scaled data)
            X_scaled = model_data['scaler'].transform(X)
            xgb_proba = model_data['xgb_model'].predict_proba(X_scaled)[0]
            xgb_win_prob = xgb_proba[1] if len(xgb_proba) > 1 else 0.5
            
            # Average the predictions
            win_prob = (rf_win_prob + xgb_win_prob) / 2
            
            return win_prob
        except Exception as e:
            print(f"⚠️  Prediction error: {e}")
            return 0.5
    
    def update_player_profile(self, player_id: str, player_name: str, position: str,
                            stats: Dict, match_result: str, minutes_played: float = 90.0):
        """
        Update a player's profile with new match data.
        
        Args:
            player_id: Unique player identifier
            player_name: Player name
            position: Position (GK, DEF, MID, FWD)
            stats: Dictionary of player statistics
            match_result: 'W', 'D', or 'L'
            minutes_played: Minutes played in the match
        """
        profile = self.player_profiles[player_id]
        
        # Update basic info
        profile['player_name'] = player_name
        profile['position'] = position
        profile['stats'] = stats
        profile['matches'] += 1
        profile['minutes'] += minutes_played
        
        # Update win/draw/loss record
        if match_result == 'W':
            profile['actual_wins'] += 1
        elif match_result == 'D':
            profile['actual_draws'] += 1
        elif match_result == 'L':
            profile['actual_losses'] += 1
        
        # Calculate actual performance metrics
        total_matches = profile['matches']
        if total_matches > 0:
            profile['actual_win_rate'] = profile['actual_wins'] / total_matches
            points = (profile['actual_wins'] * 3) + profile['actual_draws']
            profile['actual_ppg'] = points / total_matches
        
        # Calculate ML-based win contribution
        profile['win_contribution_score'] = self.calculate_win_contribution_score(stats, position)
        profile['predicted_win_probability'] = self.predict_win_probability(stats, position)
        
        # Add to match history
        profile['match_history'].append({
            'result': match_result,
            'minutes': minutes_played,
            'win_contribution': profile['win_contribution_score'],
            'predicted_win_prob': profile['predicted_win_probability']
        })
    
    def get_player_rating(self, player_id: str) -> Dict:
        """Get comprehensive rating for a player"""
        if player_id not in self.player_profiles:
            return None
        
        profile = self.player_profiles[player_id]
        
        # Combine ML prediction with actual performance
        if profile['matches'] >= 5:
            # Weight actual performance more heavily with more matches
            match_weight = min(0.7, profile['matches'] / 20)
            rating = (
                profile['actual_win_rate'] * match_weight +
                profile['predicted_win_probability'] * (1 - match_weight)
            ) * 10
        else:
            # Rely more on ML prediction for new players
            rating = profile['predicted_win_probability'] * 10
        
        return {
            'player_id': player_id,
            'player_name': profile['player_name'],
            'position': profile['position'],
            'overall_rating': rating,
            'win_contribution_score': profile['win_contribution_score'],
            'predicted_win_probability': profile['predicted_win_probability'],
            'actual_win_rate': profile['actual_win_rate'],
            'actual_ppg': profile['actual_ppg'],
            'matches': profile['matches'],
            'record': f"{profile['actual_wins']}-{profile['actual_draws']}-{profile['actual_losses']}",
            'minutes': profile['minutes']
        }
    
    def export_profiles(self, output_file='ml_enhanced_player_profiles.csv'):
        """Export all player profiles to CSV"""
        data = []
        for player_id, profile in self.player_profiles.items():
            rating = self.get_player_rating(player_id)
            if rating:
                data.append(rating)
        
        df = pd.DataFrame(data)
        df = df.sort_values('overall_rating', ascending=False)
        df.to_csv(output_file, index=False)
        print(f"✅ Exported {len(df)} ML-enhanced player profiles to {output_file}")
        return df
    
    def get_top_players_by_position(self, position: str, limit: int = 10) -> List[Dict]:
        """Get top players for a specific position"""
        position_players = []
        
        for player_id, profile in self.player_profiles.items():
            if profile['position'] == position and profile['matches'] >= 5:
                rating = self.get_player_rating(player_id)
                if rating:
                    position_players.append(rating)
        
        # Sort by overall rating
        position_players.sort(key=lambda x: x['overall_rating'], reverse=True)
        
        return position_players[:limit]


if __name__ == "__main__":
    # Test the profiler
    print("="*80)
    print("ML-ENHANCED PLAYER PROFILER - TEST")
    print("="*80)
    
    profiler = MLEnhancedPlayerProfiler()
    
    # Load feature importance models
    if profiler.load_feature_importance_models():
        print("\n✅ Models loaded successfully!")
        print("\nAvailable positions:", list(profiler.feature_models.keys()))
        
        # Test with sample player stats
        print("\n" + "="*80)
        print("TESTING WIN CONTRIBUTION CALCULATION")
        print("="*80)
        
        # Example: Elite striker
        striker_stats = {
            'Gls': 25, 'Ast': 8, 'Sh': 120, 'SoT': 55, 'SoT%': 45.8,
            'G/Sh': 0.21, 'G/SoT': 0.45, 'GCA': 15, 'SCA': 80,
            'MP': 35, '90s': 33, 'Min': 2970
        }
        
        win_contribution = profiler.calculate_win_contribution_score(striker_stats, 'FWD')
        win_prob = profiler.predict_win_probability(striker_stats, 'FWD')
        
        print(f"\nElite Striker Stats:")
        print(f"  Win Contribution Score: {win_contribution:.2f}/10")
        print(f"  Predicted Win Probability: {win_prob*100:.1f}%")
        
        # Example: Elite midfielder
        mid_stats = {
            'Gls': 8, 'Ast': 12, 'PasTotCmp': 2200, 'PasTotAtt': 2500,
            'PasTotCmp%': 88, 'PasProg': 250, 'SCA': 120, 'GCA': 12,
            'Tkl': 45, 'Int': 35, 'MP': 36, '90s': 35
        }
        
        win_contribution = profiler.calculate_win_contribution_score(mid_stats, 'MID')
        win_prob = profiler.predict_win_probability(mid_stats, 'MID')
        
        print(f"\nElite Midfielder Stats:")
        print(f"  Win Contribution Score: {win_contribution:.2f}/10")
        print(f"  Predicted Win Probability: {win_prob*100:.1f}%")
        
        print("\n✅ ML-Enhanced Player Profiler is ready!")
    else:
        print("\n⚠️  Run analyze_feature_importance.py first to generate models")
