"""
XGBoost-based match outcome predictor
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple
import joblib


class MatchPredictor:
    """XGBoost model for predicting match outcomes"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.label_mapping = {
            'home_win': 0,
            'draw': 1,
            'away_win': 2
        }
        self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
    
    def prepare_features(self, matches_df: pd.DataFrame, 
                        team_stats: pd.DataFrame,
                        team_elo: pd.DataFrame,
                        player_profiler = None,
                        manager_profiler = None,
                        form_tracker = None) -> pd.DataFrame:
        """
        Prepare feature matrix from match data
        
        Args:
            matches_df: DataFrame with match information
            team_stats: DataFrame with team statistics
            team_elo: DataFrame with team ELO ratings
            player_profiler: PlayerProfiler instance (optional)
            manager_profiler: ManagerProfiler instance (optional)
            form_tracker: FormTracker instance (optional)
            
        Returns:
            DataFrame with features for each match
        """
        features = []
        
        for idx, match in matches_df.iterrows():
            home_team_id = match['home_team_id']
            away_team_id = match['away_team_id']
            
            # Get ELO ratings
            home_elo = team_elo[team_elo['team_id'] == home_team_id]['elo_rating'].values
            away_elo = team_elo[team_elo['team_id'] == away_team_id]['elo_rating'].values
            
            if len(home_elo) == 0 or len(away_elo) == 0:
                continue
            
            feature_dict = {
                'match_id': match.get('match_id'),
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_elo': home_elo[0],
                'away_elo': away_elo[0],
                'elo_diff': home_elo[0] - away_elo[0],
            }
            
            # Add team statistics if available
            if not team_stats.empty and 'team_id' in team_stats.columns:
                home_stats = team_stats[team_stats['team_id'] == home_team_id]
                away_stats = team_stats[team_stats['team_id'] == away_team_id]
                
                if not home_stats.empty:
                    for col in ['possession', 'passes', 'shots', 'shots_on_target', 'goals', 'fouls']:
                        if col in home_stats.columns:
                            feature_dict[f'home_{col}'] = home_stats[col].values[0]
                
                if not away_stats.empty:
                    for col in ['possession', 'passes', 'shots', 'shots_on_target', 'goals', 'fouls']:
                        if col in away_stats.columns:
                            feature_dict[f'away_{col}'] = away_stats[col].values[0]
            
            # Add player-based features if profiler available
            if player_profiler is not None:
                home_player_ratings = self._get_team_player_ratings(
                    home_team_id, player_profiler
                )
                away_player_ratings = self._get_team_player_ratings(
                    away_team_id, player_profiler
                )
                
                if home_player_ratings:
                    feature_dict['home_attack_rating'] = home_player_ratings.get('attack', 50.0)
                    feature_dict['home_defense_rating'] = home_player_ratings.get('defense', 50.0)
                    feature_dict['home_midfield_rating'] = home_player_ratings.get('midfield', 50.0)
                
                if away_player_ratings:
                    feature_dict['away_attack_rating'] = away_player_ratings.get('attack', 50.0)
                    feature_dict['away_defense_rating'] = away_player_ratings.get('defense', 50.0)
                    feature_dict['away_midfield_rating'] = away_player_ratings.get('midfield', 50.0)
                
                if home_player_ratings and away_player_ratings:
                    feature_dict['player_quality_diff'] = (
                        home_player_ratings.get('overall', 50.0) - 
                        away_player_ratings.get('overall', 50.0)
                    )
            
            # Add manager features if profiler available
            if manager_profiler is not None:
                # Get manager IDs from match data (stored directly in DataFrame)
                home_manager_id = match.get('home_manager_id')
                away_manager_id = match.get('away_manager_id')
                
                if home_manager_id and home_manager_id in manager_profiler.manager_profiles:
                    home_mgr = manager_profiler.manager_profiles[home_manager_id]
                    feature_dict['home_manager_rating'] = home_mgr['ratings']['overall']
                    feature_dict['home_manager_offensive'] = home_mgr['ratings']['offensive']
                    feature_dict['home_manager_defensive'] = home_mgr['ratings']['defensive']
                
                if away_manager_id and away_manager_id in manager_profiler.manager_profiles:
                    away_mgr = manager_profiler.manager_profiles[away_manager_id]
                    feature_dict['away_manager_rating'] = away_mgr['ratings']['overall']
                    feature_dict['away_manager_offensive'] = away_mgr['ratings']['offensive']
                    feature_dict['away_manager_defensive'] = away_mgr['ratings']['defensive']
                
                if 'home_manager_rating' in feature_dict and 'away_manager_rating' in feature_dict:
                    feature_dict['manager_rating_diff'] = (
                        feature_dict['home_manager_rating'] - feature_dict['away_manager_rating']
                    )
            
            # Add form-based features if tracker available
            if form_tracker is not None:
                # Overall form
                home_form = form_tracker.get_team_form(home_team_id)
                away_form = form_tracker.get_team_form(away_team_id)
                
                feature_dict['home_form_score'] = home_form['form_score']
                feature_dict['away_form_score'] = away_form['form_score']
                feature_dict['form_diff'] = home_form['form_score'] - away_form['form_score']
                
                # Home/Away specific form
                home_home_form = form_tracker.get_team_form(home_team_id, is_home=True)
                away_away_form = form_tracker.get_team_form(away_team_id, is_home=False)
                
                feature_dict['home_home_form'] = home_home_form['form_score']
                feature_dict['away_away_form'] = away_away_form['form_score']
                
                # Momentum
                feature_dict['home_momentum'] = form_tracker.get_momentum_score(home_team_id)
                feature_dict['away_momentum'] = form_tracker.get_momentum_score(away_team_id)
                feature_dict['momentum_diff'] = (
                    feature_dict['home_momentum'] - feature_dict['away_momentum']
                )
                
                # Scoring/Defensive form
                home_scoring = form_tracker.get_scoring_form(home_team_id)
                away_scoring = form_tracker.get_scoring_form(away_team_id)
                
                feature_dict['home_scoring_form'] = home_scoring['scoring_form']
                feature_dict['home_defensive_form'] = home_scoring['defensive_form']
                feature_dict['away_scoring_form'] = away_scoring['scoring_form']
                feature_dict['away_defensive_form'] = away_scoring['defensive_form']
                
                # Head-to-head
                h2h_home = form_tracker.get_h2h_stats(home_team_id, away_team_id, perspective_team=home_team_id)
                feature_dict['h2h_home_wins'] = h2h_home.get('wins', 0)
                feature_dict['h2h_home_win_rate'] = h2h_home.get('win_rate', 0.0)
            
            # Add match outcome as label
            if 'home_score' in match and 'away_score' in match:
                home_score = match['home_score']
                away_score = match['away_score']
                
                if home_score > away_score:

                    feature_dict['outcome'] = self.label_mapping['home_win']
                elif home_score < away_score:
                    feature_dict['outcome'] = self.label_mapping['away_win']
                else:
                    feature_dict['outcome'] = self.label_mapping['draw']
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _get_team_player_ratings(self, team_id: int, player_profiler) -> Dict:
        """
        Get aggregated player ratings for a team
        
        Returns dict with attack, defense, midfield, and overall ratings
        """
        # Get all players who have played for this team
        team_players = {}
        for player_id, profile in player_profiler.player_profiles.items():
            # Check if player has played for this team
            if team_id in profile.get('teams', set()) or profile.get('current_team') == team_id:
                team_players[player_id] = profile
        
        if not team_players:
            # No player data for this team, return defaults
            return None
        
        # Separate by position
        forwards = []
        midfielders = []
        defenders = []
        goalkeepers = []
        
        for player_id, profile in team_players.items():
            pos_group = profile.get('position_group', 'MID')
            current_rating = player_profiler.get_player_current_rating(player_id)
            overall = current_rating.get('overall', 5.0)
            
            if pos_group == 'FWD':
                forwards.append(overall)
            elif pos_group == 'MID':
                midfielders.append(overall)
            elif pos_group == 'DEF':
                defenders.append(overall)
            elif pos_group == 'GK':
                goalkeepers.append(overall)
        
        # Calculate ratings (average of top players, scaled to 0-100)
        attack_rating = np.mean(sorted(forwards, reverse=True)[:3]) * 10 if forwards else 50.0
        midfield_rating = np.mean(sorted(midfielders, reverse=True)[:3]) * 10 if midfielders else 50.0
        defense_rating = np.mean(sorted(defenders, reverse=True)[:4]) * 10 if defenders else 50.0
        gk_rating = np.mean(sorted(goalkeepers, reverse=True)[:1]) * 10 if goalkeepers else 50.0
        
        # Overall rating weighted by importance
        overall_rating = (attack_rating * 0.3 + midfield_rating * 0.3 + 
                         defense_rating * 0.3 + gk_rating * 0.1)
        
        return {
            'attack': attack_rating,
            'midfield': midfield_rating,
            'defense': defense_rating,
            'goalkeeper': gk_rating,
            'overall': overall_rating
        }
    
    def train(self, features_df: pd.DataFrame, 
              test_size: float = 0.2,
              params: Dict = None) -> Dict:
        """
        Train XGBoost model
        
        Args:
            features_df: DataFrame with features and outcomes
            test_size: Proportion of data for testing
            params: XGBoost parameters (optional)
            
        Returns:
            Dictionary with training metrics
        """
        # Separate features and labels
        X = features_df.drop(['outcome', 'match_id', 'home_team_id', 'away_team_id'], 
                            axis=1, errors='ignore')
        y = features_df['outcome']
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Default XGBoost parameters
        if params is None:
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=['Home Win', 'Draw', 'Away Win']
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict match outcomes
        
        Args:
            features: DataFrame with match features
            
        Returns:
            DataFrame with predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure correct feature columns
        X = features[self.feature_columns]
        
        # Get predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = pd.DataFrame({
            'prediction': [self.inverse_label_mapping[p] for p in predictions],
            'home_win_prob': probabilities[:, 0],
            'draw_prob': probabilities[:, 1],
            'away_win_prob': probabilities[:, 2],
        })
        
        # Add original match info if available
        if 'match_id' in features.columns:
            results['match_id'] = features['match_id'].values
        if 'home_team_id' in features.columns:
            results['home_team_id'] = features['home_team_id'].values
        if 'away_team_id' in features.columns:
            results['away_team_id'] = features['away_team_id'].values
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance = self.model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'label_mapping': self.label_mapping
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.label_mapping = data['label_mapping']
        self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
