"""
Complete pipeline for processing data and training predictor
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import pickle


from .data_loader import DataLoader
from .feature_extractor import FeatureExtractor
from .elo_system import ELOSystem
from .match_predictor import MatchPredictor
from .enhanced_player_profiler import EnhancedPlayerProfiler
from .manager_profiler import ManagerProfiler
from .form_tracker import FormTracker


class FootballPredictionPipeline:
    """End-to-end pipeline for football match prediction"""
    
    def __init__(self, data_path: str = "open-data-master/data", 
                 use_player_profiling: bool = True,
                 use_manager_profiling: bool = True,
                 use_form_tracking: bool = True):
        self.data_loader = DataLoader(data_path)
        self.feature_extractor = FeatureExtractor()
        self.elo_system = ELOSystem(k_factor=32, initial_rating=1500)
        self.predictor = MatchPredictor()
        
        # Advanced profiling systems
        self.use_player_profiling = use_player_profiling
        self.use_manager_profiling = use_manager_profiling
        self.use_form_tracking = use_form_tracking
        
        if use_player_profiling:
            self.player_profiler = EnhancedPlayerProfiler()
        if use_manager_profiling:
            self.manager_profiler = ManagerProfiler()
        if use_form_tracking:
            self.form_tracker = FormTracker(form_window=5)
        
        self.player_stats_history = []
        self.team_stats_history = []
        self.match_results = []
    
    def process_match(self, match: Dict) -> Dict:
        """
        Process a single match and update all tracking systems.
        Compatible with both StatsBomb and Transfermarkt formats.
        
        Args:
            match: Match dictionary with lineup and result info
            
        Returns:
            Match result dictionary
        """
        match_id = match.get('match_id')
        match_date = match.get('match_date')
        
        home_team_info = match.get('home_team', {})
        away_team_info = match.get('away_team', {})
        
        home_team_id = home_team_info.get('home_team_id')
        away_team_id = away_team_info.get('away_team_id')
        
        home_score = match.get('home_score')
        away_score = match.get('away_score')
        
        if not all([home_team_id, away_team_id]):
            return None
        
        # Process lineup to update player profiles
        if self.use_player_profiling:
            home_lineup = home_team_info.get('lineup', [])
            away_lineup = away_team_info.get('lineup', [])
            
            # Determine match result for win contribution tracking
            home_result = None
            away_result = None
            if home_score is not None and away_score is not None:
                if home_score > away_score:
                    home_result, away_result = 'W', 'L'
                elif home_score < away_score:
                    home_result, away_result = 'L', 'W'
                else:
                    home_result, away_result = 'D', 'D'
            
            for player in home_lineup:
                player_id = player.get('player_id')
                player_name = player.get('player_name', 'Unknown')
                position = player.get('position', 'Unknown')
                
                if player_id and position != 'Unknown':
                    # Build stats from player data
                    stats = {
                        'minutes_played': player.get('minutes_played', 0),
                        'goals': player.get('goals', 0),
                        'assists': player.get('assists', 0),
                        'shots': player.get('shots', 0),
                        'passes': player.get('passes', 0),
                        'tackles': player.get('tackles', 0),
                        'interceptions': player.get('interceptions', 0),
                    }
                    
                    # Update player profile WITH match result
                    self.player_profiler.update_profile_from_stats(
                        player_id, player_name, position, match_date,
                        stats, home_team_id, match_result=home_result
                    )
            
            for player in away_lineup:
                player_id = player.get('player_id')
                player_name = player.get('player_name', 'Unknown')
                position = player.get('position', 'Unknown')
                
                if player_id and position != 'Unknown':
                    stats = {
                        'minutes_played': player.get('minutes_played', 0),
                        'goals': player.get('goals', 0),
                        'assists': player.get('assists', 0),
                        'shots': player.get('shots', 0),
                        'passes': player.get('passes', 0),
                        'tackles': player.get('tackles', 0),
                        'interceptions': player.get('interceptions', 0),
                    }
                    
                    # Update player profile WITH match result
                    self.player_profiler.update_profile_from_stats(
                        player_id, player_name, position, match_date,
                        stats, away_team_id, match_result=away_result
                    )
        
        # Update ELO ratings if scores available
        if home_score is not None and away_score is not None:
            self.elo_system.process_match_team(
                home_team_id, away_team_id,
                home_score, away_score,
                match_date
            )
            
            # Update form tracking
            if self.use_form_tracking:
                self.form_tracker.update_team_form(
                    home_team_id, match_date, home_score, away_score, is_home=True
                )
                self.form_tracker.update_team_form(
                    away_team_id, match_date, away_score, home_score, is_home=False
                )
                self.form_tracker.update_h2h(
                    home_team_id, away_team_id, match_date, home_score, away_score
                )
        
        # Update manager profiles
        if self.use_manager_profiling:
            home_managers = home_team_info.get('managers', [])
            if home_managers and len(home_managers) > 0:
                manager_id = home_managers[0].get('id', home_managers[0].get('name'))
                manager_name = home_managers[0].get('name')
                if manager_id and home_score is not None:
                    self.manager_profiler.update_match_result(
                        manager_id, manager_name, home_team_id, match_date,
                        home_score, away_score, is_home=True
                    )
            
            away_managers = away_team_info.get('managers', [])
            if away_managers and len(away_managers) > 0:
                manager_id = away_managers[0].get('id', away_managers[0].get('name'))
                manager_name = away_managers[0].get('name')
                if manager_id and away_score is not None:
                    self.manager_profiler.update_match_result(
                        manager_id, manager_name, away_team_id, match_date,
                        away_score, home_score, is_home=False
                    )
        
        # Store match result
        match_result = {
            'match_id': match_id,
            'match_date': match_date,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_score': home_score,
            'away_score': away_score,
            'home_team_name': home_team_info.get('home_team_name'),
            'away_team_name': away_team_info.get('away_team_name'),
            'home_manager_name': home_team_info.get('managers', [{}])[0].get('name') if home_team_info.get('managers') else None,
            'away_manager_name': away_team_info.get('managers', [{}])[0].get('name') if away_team_info.get('managers') else None,
        }
        
        if home_score is not None and away_score is not None:
            self.match_results.append(match_result)
        
        return match_result
    
    def process_all_matches(self, limit: int = None) -> pd.DataFrame:
        """
        Process all available matches and extract features
        
        Args:
            limit: Maximum number of matches to process (None for all)
            
        Returns:
            DataFrame with all processed match data
        """
        print("Loading matches...")
        matches_df = self.data_loader.load_matches()
        
        if limit:
            matches_df = matches_df.head(limit)
        
        print(f"Processing {len(matches_df)} matches...")
        
        for idx, match in tqdm(matches_df.iterrows(), total=len(matches_df)):
            match_id = match['match_id']
            
            # Load events for this match
            events = self.data_loader.load_match_events(match_id)
            
            if not events:
                continue
            
            # Extract player stats
            player_stats = self.feature_extractor.extract_player_stats_from_events(events)
            if not player_stats.empty:
                player_stats['match_id'] = match_id
                player_stats['match_date'] = match.get('match_date')
                self.player_stats_history.append(player_stats)
                
                # Update player profiles if enabled
                if self.use_player_profiling:
                    match_date = match.get('match_date')
                    home_team_id = match.get('home_team', {}).get('home_team_id')
                    away_team_id = match.get('away_team', {}).get('away_team_id')
                    
                    for idx_player, player_row in player_stats.iterrows():
                        player_id = player_row.get('player_id')
                        player_name = player_row.get('player_name', 'Unknown')
                        position = player_row.get('position', 'Unknown')
                        team_id = player_row.get('team_id')  # Get team from player stats
                        
                        if player_id and position != 'Unknown':
                            # Calculate stats and ratings for this match
                            player_match_stats = self.player_profiler.calculate_player_stats(
                                events, player_id, position
                            )
                            player_ratings = self.player_profiler.calculate_player_ratings(
                                player_match_stats
                            )
                            
                            # Update profile with team_id
                            self.player_profiler.update_profile(
                                player_id, player_name, match_date,
                                player_match_stats, player_ratings, team_id
                            )
            
            # Extract team stats
            team_stats = self.feature_extractor.extract_team_stats_from_events(events)
            
            # Get match result
            home_team_id = match.get('home_team', {}).get('home_team_id')
            away_team_id = match.get('away_team', {}).get('away_team_id')
            
            if home_team_id and away_team_id and team_stats:
                home_stats = team_stats.get(home_team_id, {})
                away_stats = team_stats.get(away_team_id, {})
                
                home_goals = home_stats.get('goals', 0)
                away_goals = away_stats.get('goals', 0)
                
                # Update ELO ratings
                match_date = match.get('match_date')
                self.elo_system.process_match_team(
                    home_team_id, away_team_id, 
                    home_goals, away_goals,
                    match_date
                )
                
                # Update form tracking
                if self.use_form_tracking:
                    self.form_tracker.update_team_form(
                        home_team_id, match_date, home_goals, away_goals, is_home=True
                    )
                    self.form_tracker.update_team_form(
                        away_team_id, match_date, away_goals, home_goals, is_home=False
                    )
                    self.form_tracker.update_h2h(
                        home_team_id, away_team_id, match_date, home_goals, away_goals
                    )
                
                # Store team stats with match info
                for team_id, stats in team_stats.items():
                    stats['match_id'] = match_id
                    stats['match_date'] = match_date
                    stats['is_home'] = team_id == home_team_id
                    self.team_stats_history.append(stats)
                
                # Store match result
                home_team_info = match.get('home_team', {})
                away_team_info = match.get('away_team', {})
                
                match_result = {
                    'match_id': match_id,
                    'match_date': match_date,
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'home_score': home_goals,
                    'away_score': away_goals,
                    'home_team_name': home_team_info.get('home_team_name'),
                    'away_team_name': away_team_info.get('away_team_name'),
                }
                
                # Add manager information if available
                home_managers = home_team_info.get('managers', [])
                if home_managers and len(home_managers) > 0:
                    match_result['home_manager_id'] = home_managers[0].get('id')
                    match_result['home_manager_name'] = home_managers[0].get('name')
                    if self.use_manager_profiling:
                        manager_id = home_managers[0].get('id')
                        manager_name = home_managers[0].get('name')
                        # Update manager profile with match outcome
                        if manager_id:
                            self.manager_profiler.update_match_result(
                                manager_id, manager_name, home_team_id, match_date,
                                home_goals, away_goals, is_home=True
                            )
                
                away_managers = away_team_info.get('managers', [])
                if away_managers and len(away_managers) > 0:
                    match_result['away_manager_id'] = away_managers[0].get('id')
                    match_result['away_manager_name'] = away_managers[0].get('name')
                    if self.use_manager_profiling:
                        manager_id = away_managers[0].get('id')
                        manager_name = away_managers[0].get('name')
                        # Update manager profile with match outcome
                        if manager_id:
                            self.manager_profiler.update_match_result(
                                manager_id, manager_name, away_team_id, match_date,
                                away_goals, home_goals, is_home=False
                            )
                
                self.match_results.append(match_result)
        
        # Manager profiling is now done match-by-match above
        if self.use_manager_profiling:
            print(f"Manager profiles created: {len(self.manager_profiler.manager_profiles)}")
        
        return pd.DataFrame(self.match_results)
    
    def build_training_dataset(self, processed_matches: List[Dict] = None) -> Tuple[pd.DataFrame, pd.Series, List]:
        """
        Build training dataset with all features
        
        Args:
            processed_matches: Optional list of pre-processed match results.
                             If None, uses self.match_results
        
        Returns:
            Tuple of (X features, y labels, match_ids)
        """
        print("Building training dataset...")
        
        # Use provided matches or stored match results
        if processed_matches:
            matches_df = pd.DataFrame(processed_matches)
        else:
            matches_df = pd.DataFrame(self.match_results)
        
        if matches_df.empty:
            return pd.DataFrame(), pd.Series(), []
        
        # Get aggregated team stats (average per team)
        team_stats_df = pd.DataFrame(self.team_stats_history)
        if not team_stats_df.empty:
            team_stats_agg = team_stats_df.groupby('team_id').agg({
                'possession': 'mean',
                'passes': 'mean',
                'passes_completed': 'mean',
                'shots': 'mean',
                'shots_on_target': 'mean',
                'goals': 'mean',
                'fouls': 'mean',
            }).reset_index()
        else:
            team_stats_agg = pd.DataFrame()
        
        # Get ELO ratings
        team_elo_df = self.elo_system.get_team_ratings_df()
        
        # Prepare features
        features_df = self.predictor.prepare_features(
            matches_df, 
            team_stats_agg, 
            team_elo_df,
            player_profiler=self.player_profiler if self.use_player_profiling else None,
            manager_profiler=self.manager_profiler if self.use_manager_profiling else None,
            form_tracker=self.form_tracker if self.use_form_tracking else None
        )
        
        if features_df.empty:
            return pd.DataFrame(), pd.Series(), []
        
        # Separate features and labels
        # The predictor uses 'outcome' as the label column
        label_col = 'outcome' if 'outcome' in features_df.columns else 'result'
        X = features_df.drop([label_col, 'match_id'], axis=1, errors='ignore')
        y = features_df[label_col] if label_col in features_df.columns else pd.Series()
        match_ids = features_df['match_id'].tolist() if 'match_id' in features_df.columns else []
        
        return X, y, match_ids
    
    
    def train_model(self, test_size: float = 0.2, processed_matches: List[Dict] = None) -> Dict:
        """
        Train the prediction model
        
        Args:
            test_size: Proportion of data for testing
            processed_matches: Optional list of pre-processed matches
            
        Returns:
            Dictionary with training metrics
        """
        X, y, match_ids = self.build_training_dataset(processed_matches)
        
        if X.empty:
            raise ValueError("No training data available!")
        
        print(f"Training on {len(X)} matches...")
        
        # The predictor expects a DataFrame with 'outcome' column (not 'result')
        # Reconstruct it temporarily
        features_df = X.copy()
        features_df['outcome'] = y
        features_df['match_id'] = match_ids
        
        metrics = self.predictor.train(features_df, test_size=test_size)
        
        print(f"\nTraining Results:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Cross-validation: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']:.3f})")
        print(f"\nClassification Report:")
        print(metrics['classification_report'])
        
        return metrics
    
    def predict_match(self, home_team_id: int, away_team_id: int) -> Dict:
        """
        Predict outcome of a single match
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            
        Returns:
            Dictionary with prediction and probabilities
        """
        # Get team stats
        team_stats_df = pd.DataFrame(self.team_stats_history)
        team_stats_agg = team_stats_df.groupby('team_id').agg({
            'possession': 'mean',
            'passes': 'mean',
            'shots': 'mean',
            'shots_on_target': 'mean',
            'goals': 'mean',
            'fouls': 'mean',
        }).reset_index()
        
        team_elo_df = self.elo_system.get_team_ratings_df()
        
        # Create feature row
        match_data = pd.DataFrame([{
            'match_id': 'prediction',
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_score': 0,  # Placeholder
            'away_score': 0,  # Placeholder
        }])
        
        features = self.predictor.prepare_features(
            match_data, team_stats_agg, team_elo_df,
            player_profiler=self.player_profiler if self.use_player_profiling else None,
            manager_profiler=self.manager_profiler if self.use_manager_profiling else None,
            form_tracker=self.form_tracker if self.use_form_tracking else None
        )
        
        if features.empty:
            return {'error': 'Insufficient data for prediction'}
        
        # Get prediction
        features = features.drop('outcome', axis=1, errors='ignore')
        prediction = self.predictor.predict(features)
        
        return {
            'prediction': prediction.iloc[0]['prediction'],
            'home_win_prob': prediction.iloc[0]['home_win_prob'],
            'draw_prob': prediction.iloc[0]['draw_prob'],
            'away_win_prob': prediction.iloc[0]['away_win_prob'],
            'home_elo': features.iloc[0]['home_elo'],
            'away_elo': features.iloc[0]['away_elo'],
        }
    
    def save_pipeline(self, filepath: str):
        """Save entire pipeline state"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'elo_system': self.elo_system,
                'player_stats_history': self.player_stats_history,
                'team_stats_history': self.team_stats_history,
                'match_results': self.match_results,
            }, f)
        
        # Save model separately
        model_path = filepath.replace('.pkl', '_model.pkl')
        self.predictor.save_model(model_path)
        
        print(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """Load pipeline state from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.elo_system = data['elo_system']
        self.player_stats_history = data['player_stats_history']
        self.team_stats_history = data['team_stats_history']
        self.match_results = data['match_results']
        
        # Load model
        model_path = filepath.replace('.pkl', '_model.pkl')
        self.predictor.load_model(model_path)
        
        print(f"Pipeline loaded from {filepath}")
    
    def extract_features_for_prediction(self, match_data: Dict) -> np.ndarray:
        """
        Extract features for a single match prediction.
        Used when predicting future matches.
        
        Args:
            match_data: Dictionary with match info (from process_match)
            
        Returns:
            Feature vector for prediction
        """
        # Prepare minimal DataFrame for feature extraction
        matches_df = pd.DataFrame([match_data])
        
        # Get aggregated team stats
        team_stats_df = pd.DataFrame(self.team_stats_history)
        if not team_stats_df.empty:
            team_stats_agg = team_stats_df.groupby('team_id').agg({
                'possession': 'mean',
                'passes': 'mean',
                'passes_completed': 'mean',
                'shots': 'mean',
                'shots_on_target': 'mean',
                'goals': 'mean',
                'fouls': 'mean',
            }).reset_index()
        else:
            team_stats_agg = pd.DataFrame()
        
        # Get ELO ratings
        team_elo_df = self.elo_system.get_team_ratings_df()
        
        # Extract features
        features_df = self.predictor.prepare_features(
            matches_df,
            team_stats_agg,
            team_elo_df,
            player_profiler=self.player_profiler if self.use_player_profiling else None,
            manager_profiler=self.manager_profiler if self.use_manager_profiling else None,
            form_tracker=self.form_tracker if self.use_form_tracking else None
        )
        
        if features_df.empty:
            return None
        
        # Return feature vector (without target column if it exists)
        feature_cols = [col for col in features_df.columns if col not in ['result', 'match_id']]
        return features_df[feature_cols].iloc[0].values
