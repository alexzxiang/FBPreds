"""
Form tracker for time-based features
Tracks recent performance (last N matches) for teams, players, and managers
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from collections import defaultdict, deque
from datetime import datetime, timedelta


class FormTracker:
    """Track recent form and time-based features"""
    
    def __init__(self, form_window: int = 5):
        """
        Args:
            form_window: Number of recent matches to consider for form
        """
        self.form_window = form_window
        
        # Team form tracking
        self.team_recent_matches = defaultdict(lambda: deque(maxlen=form_window))
        self.team_form_scores = {}
        
        # Player form tracking  
        self.player_recent_matches = defaultdict(lambda: deque(maxlen=form_window))
        self.player_form_scores = {}
        
        # Manager form tracking
        self.manager_recent_matches = defaultdict(lambda: deque(maxlen=form_window))
        self.manager_form_scores = {}
        
        # Head-to-head tracking
        self.h2h_history = defaultdict(list)
        
        # Home/Away specific form
        self.team_home_form = defaultdict(lambda: deque(maxlen=form_window))
        self.team_away_form = defaultdict(lambda: deque(maxlen=form_window))
    
    def update_team_form(self, team_id: int, match_date: str, 
                        goals_for: int, goals_against: int, 
                        is_home: bool = True):
        """Update team's recent form"""
        # Calculate result score (3 for win, 1 for draw, 0 for loss)
        if goals_for > goals_against:
            result_score = 3
            result = 'W'
        elif goals_for < goals_against:
            result_score = 0
            result = 'L'
        else:
            result_score = 1
            result = 'D'
        
        match_data = {
            'date': match_date,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'result_score': result_score,
            'result': result,
            'is_home': is_home
        }
        
        # Add to general form
        self.team_recent_matches[team_id].append(match_data)
        
        # Add to home/away specific form
        if is_home:
            self.team_home_form[team_id].append(match_data)
        else:
            self.team_away_form[team_id].append(match_data)
    
    def get_team_form(self, team_id: int, is_home: bool = None) -> Dict:
        """
        Get team's current form metrics
        
        Args:
            team_id: Team identifier
            is_home: If True, get home form; If False, get away form; If None, get overall
            
        Returns:
            Dict with form metrics
        """
        # Select appropriate match history
        if is_home is True:
            recent_matches = list(self.team_home_form.get(team_id, []))
        elif is_home is False:
            recent_matches = list(self.team_away_form.get(team_id, []))
        else:
            recent_matches = list(self.team_recent_matches.get(team_id, []))
        
        if not recent_matches:
            return {
                'form_score': 1.5,  # Neutral (between 0-3)
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'goals_scored_avg': 0.0,
                'goals_conceded_avg': 0.0,
                'goal_diff_avg': 0.0,
                'points_per_match': 1.0,
                'matches_played': 0
            }
        
        # Calculate metrics
        wins = sum(1 for m in recent_matches if m['result'] == 'W')
        draws = sum(1 for m in recent_matches if m['result'] == 'D')
        losses = sum(1 for m in recent_matches if m['result'] == 'L')
        
        total_goals_scored = sum(m['goals_for'] for m in recent_matches)
        total_goals_conceded = sum(m['goals_against'] for m in recent_matches)
        
        num_matches = len(recent_matches)
        
        # Weight recent matches more heavily
        weights = [0.5 ** i for i in range(num_matches)][::-1]  # More recent = higher weight
        weights = [w / sum(weights) for w in weights]  # Normalize
        
        form_score = sum(m['result_score'] * w for m, w in zip(recent_matches, weights))
        
        return {
            'form_score': form_score,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_scored_avg': total_goals_scored / num_matches,
            'goals_conceded_avg': total_goals_conceded / num_matches,
            'goal_diff_avg': (total_goals_scored - total_goals_conceded) / num_matches,
            'points_per_match': (wins * 3 + draws) / num_matches,
            'matches_played': num_matches
        }
    
    def update_h2h(self, home_team_id: int, away_team_id: int, 
                   match_date: str, home_goals: int, away_goals: int):
        """Update head-to-head history between two teams"""
        matchup_key = tuple(sorted([home_team_id, away_team_id]))
        
        h2h_match = {
            'date': match_date,
            'home_team': home_team_id,
            'away_team': away_team_id,
            'home_goals': home_goals,
            'away_goals': away_goals
        }
        
        self.h2h_history[matchup_key].append(h2h_match)
    
    def get_h2h_stats(self, team1_id: int, team2_id: int, perspective_team: int = None) -> Dict:
        """
        Get head-to-head statistics between two teams
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            perspective_team: Calculate stats from this team's perspective (optional)
            
        Returns:
            Dict with H2H metrics
        """
        matchup_key = tuple(sorted([team1_id, team2_id]))
        h2h_matches = self.h2h_history.get(matchup_key, [])
        
        if not h2h_matches:
            return {
                'h2h_matches': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'goals_for_avg': 0.0,
                'goals_against_avg': 0.0
            }
        
        # If no perspective specified, just count total matches
        if perspective_team is None:
            return {'h2h_matches': len(h2h_matches)}
        
        # Calculate from perspective team's view
        wins = 0
        draws = 0
        losses = 0
        total_goals_for = 0
        total_goals_against = 0
        
        for match in h2h_matches:
            if match['home_team'] == perspective_team:
                goals_for = match['home_goals']
                goals_against = match['away_goals']
            else:
                goals_for = match['away_goals']
                goals_against = match['home_goals']
            
            total_goals_for += goals_for
            total_goals_against += goals_against
            
            if goals_for > goals_against:
                wins += 1
            elif goals_for < goals_against:
                losses += 1
            else:
                draws += 1
        
        num_matches = len(h2h_matches)
        
        return {
            'h2h_matches': num_matches,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_for_avg': total_goals_for / num_matches,
            'goals_against_avg': total_goals_against / num_matches,
            'win_rate': wins / num_matches if num_matches > 0 else 0.0
        }
    
    def get_momentum_score(self, team_id: int) -> float:
        """
        Calculate team momentum (weighted recent results with exponential decay)
        
        Returns:
            Momentum score (0-100, higher = better momentum)
        """
        recent_matches = list(self.team_recent_matches.get(team_id, []))
        
        if not recent_matches:
            return 50.0  # Neutral
        
        # Exponential weighting (most recent match has most weight)
        momentum = 0.0
        total_weight = 0.0
        
        for i, match in enumerate(reversed(recent_matches)):
            weight = (0.7 ** i)  # Exponential decay
            result_value = match['result_score'] / 3.0  # Normalize to 0-1
            
            # Also consider goal difference
            goal_diff = match['goals_for'] - match['goals_against']
            goal_diff_factor = min(3, max(-3, goal_diff)) / 3.0  # Normalize to -1 to 1
            
            combined_value = (result_value * 0.7 + (goal_diff_factor + 1) / 2 * 0.3)
            
            momentum += combined_value * weight
            total_weight += weight
        
        return (momentum / total_weight) * 100
    
    def get_scoring_form(self, team_id: int) -> Dict:
        """Get team's recent scoring and defensive form"""
        recent_matches = list(self.team_recent_matches.get(team_id, []))
        
        if not recent_matches:
            return {
                'scoring_form': 1.0,
                'defensive_form': 1.0,
                'clean_sheets': 0,
                'failed_to_score': 0
            }
        
        goals_scored = [m['goals_for'] for m in recent_matches]
        goals_conceded = [m['goals_against'] for m in recent_matches]
        
        clean_sheets = sum(1 for g in goals_conceded if g == 0)
        failed_to_score = sum(1 for g in goals_scored if g == 0)
        
        return {
            'scoring_form': np.mean(goals_scored),
            'defensive_form': np.mean(goals_conceded),
            'clean_sheets': clean_sheets,
            'failed_to_score': failed_to_score,
            'scoring_consistency': np.std(goals_scored) if len(goals_scored) > 1 else 0.0
        }
    
    def export_form_data(self) -> pd.DataFrame:
        """Export all team form data to DataFrame"""
        form_data = []
        
        for team_id in self.team_recent_matches.keys():
            overall_form = self.get_team_form(team_id)
            home_form = self.get_team_form(team_id, is_home=True)
            away_form = self.get_team_form(team_id, is_home=False)
            momentum = self.get_momentum_score(team_id)
            scoring_form = self.get_scoring_form(team_id)
            
            form_data.append({
                'team_id': team_id,
                'overall_form_score': overall_form['form_score'],
                'home_form_score': home_form['form_score'],
                'away_form_score': away_form['form_score'],
                'momentum': momentum,
                'scoring_form': scoring_form['scoring_form'],
                'defensive_form': scoring_form['defensive_form'],
                'clean_sheets': scoring_form['clean_sheets'],
                'failed_to_score': scoring_form['failed_to_score'],
                'matches_played': overall_form['matches_played']
            })
        
        return pd.DataFrame(form_data)
