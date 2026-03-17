"""
ELO rating system for teams and players
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from collections import defaultdict


class ELOSystem:
    """ELO rating system for football teams and players"""
    
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        """
        Initialize ELO system
        
        Args:
            k_factor: Maximum rating change per match
            initial_rating: Starting rating for new entities
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.team_ratings = {}
        self.player_ratings = {}
        self.rating_history = {
            'team': {},
            'player': {}
        }
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for team/player A vs B
        
        Args:
            rating_a: ELO rating of entity A
            rating_b: ELO rating of entity B
            
        Returns:
            Expected score (0-1) for entity A
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_rating(self, rating: float, actual_score: float, expected_score: float) -> float:
        """
        Update rating based on match result
        
        Args:
            rating: Current rating
            actual_score: Actual result (1=win, 0.5=draw, 0=loss)
            expected_score: Expected result from expected_score()
            
        Returns:
            Updated rating
        """
        return rating + self.k_factor * (actual_score - expected_score)
    
    def process_match_team(self, home_team_id: int, away_team_id: int, 
                          home_goals: int, away_goals: int,
                          match_date: str = None) -> Tuple[float, float]:
        """
        Process a match and update team ELO ratings
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
            match_date: Date of match (for history tracking)
            
        Returns:
            Tuple of (new_home_rating, new_away_rating)
        """
        # Get current ratings
        home_rating = self.team_ratings.get(home_team_id, self.initial_rating)
        away_rating = self.team_ratings.get(away_team_id, self.initial_rating)
        
        # Calculate expected scores
        home_expected = self.expected_score(home_rating, away_rating)
        away_expected = 1 - home_expected
        
        # Determine actual scores
        if home_goals > away_goals:
            home_actual, away_actual = 1.0, 0.0
        elif home_goals < away_goals:
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual, away_actual = 0.5, 0.5
        
        # Goal difference multiplier (optional enhancement)
        goal_diff = abs(home_goals - away_goals)
        multiplier = np.log(max(goal_diff, 1) + 1)
        
        # Update ratings
        home_new = self.update_rating(home_rating, home_actual * multiplier, home_expected * multiplier)
        away_new = self.update_rating(away_rating, away_actual * multiplier, away_expected * multiplier)
        
        self.team_ratings[home_team_id] = home_new
        self.team_ratings[away_team_id] = away_new
        
        # Track history
        if match_date:
            if home_team_id not in self.rating_history['team']:
                self.rating_history['team'][home_team_id] = []
            if away_team_id not in self.rating_history['team']:
                self.rating_history['team'][away_team_id] = []
            
            self.rating_history['team'][home_team_id].append({
                'date': match_date,
                'rating': home_new,
                'opponent': away_team_id,
                'result': home_actual
            })
            self.rating_history['team'][away_team_id].append({
                'date': match_date,
                'rating': away_new,
                'opponent': home_team_id,
                'result': away_actual
            })
        
        return home_new, away_new
    
    def process_match_player(self, player_id: int, player_rating_change: float, 
                            match_date: str = None):
        """
        Update player ELO based on performance
        
        Args:
            player_id: Player ID
            player_rating_change: Rating change based on performance
            match_date: Date of match (for history tracking)
        """
        current_rating = self.player_ratings.get(player_id, self.initial_rating)
        new_rating = current_rating + player_rating_change
        
        self.player_ratings[player_id] = new_rating
        
        if match_date:
            if player_id not in self.rating_history['player']:
                self.rating_history['player'][player_id] = []
            self.rating_history['player'][player_id].append({
                'date': match_date,
                'rating': new_rating
            })
    
    def get_team_rating(self, team_id: int) -> float:
        """Get current team ELO rating"""
        return self.team_ratings.get(team_id, self.initial_rating)
    
    def get_player_rating(self, player_id: int) -> float:
        """Get current player ELO rating"""
        return self.player_ratings.get(player_id, self.initial_rating)
    
    def get_team_ratings_df(self) -> pd.DataFrame:
        """Get all team ratings as DataFrame"""
        return pd.DataFrame([
            {'team_id': team_id, 'elo_rating': rating}
            for team_id, rating in self.team_ratings.items()
        ]).sort_values('elo_rating', ascending=False)
    
    def get_player_ratings_df(self) -> pd.DataFrame:
        """Get all player ratings as DataFrame"""
        return pd.DataFrame([
            {'player_id': player_id, 'elo_rating': rating}
            for player_id, rating in self.player_ratings.items()
        ]).sort_values('elo_rating', ascending=False)
    
    def predict_match_outcome(self, home_team_id: int, away_team_id: int) -> Dict:
        """
        Predict match outcome based on ELO ratings
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            
        Returns:
            Dictionary with win/draw/loss probabilities
        """
        home_rating = self.team_ratings.get(home_team_id, self.initial_rating)
        away_rating = self.team_ratings.get(away_team_id, self.initial_rating)
        
        # Expected score (simplified model)
        home_win_prob = self.expected_score(home_rating, away_rating)
        away_win_prob = 1 - home_win_prob
        
        # Adjust for draw (simple heuristic: closer ratings = higher draw probability)
        rating_diff = abs(home_rating - away_rating)
        draw_prob = max(0, 0.3 - (rating_diff / 1000))
        
        # Normalize probabilities
        total = home_win_prob + away_win_prob + draw_prob
        
        return {
            'home_win': home_win_prob / total,
            'draw': draw_prob / total,
            'away_win': away_win_prob / total,
            'home_rating': home_rating,
            'away_rating': away_rating
        }
