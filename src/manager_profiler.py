"""
Manager Profiling System - Track manager performance and tactical tendencies
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime


class ManagerProfiler:
    """Build and maintain manager profiles and tactical analysis"""
    
    def __init__(self):
        self.manager_profiles = {}  # manager_id -> profile dict
        self.manager_match_history = defaultdict(list)  # manager_id -> matches
        
    def extract_manager_stats(self, matches_df: pd.DataFrame) -> Dict:
        """
        Extract manager statistics from match data
        
        Args:
            matches_df: DataFrame with matches including manager information
        
        Returns:
            Dictionary of manager profiles
        """
        manager_stats = defaultdict(lambda: {
            'manager_id': None,
            'manager_name': None,
            'teams_managed': set(),
            'matches': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'home_wins': 0,
            'home_draws': 0,
            'home_losses': 0,
            'away_wins': 0,
            'away_draws': 0,
            'away_losses': 0,
            'goals_for': 0,
            'goals_against': 0,
            'formations_used': defaultdict(int),
            'avg_possession': 0.0,
            'match_history': []
        })
        
        # Process each match
        for idx, match in matches_df.iterrows():
            # Home manager
            home_managers = match.get('home_team', {}).get('managers', [])
            if home_managers and len(home_managers) > 0:
                manager = home_managers[0]
                manager_id = manager.get('id')
                manager_name = manager.get('name')
                
                if manager_id:
                    stats = manager_stats[manager_id]
                    stats['manager_id'] = manager_id
                    stats['manager_name'] = manager_name
                    stats['teams_managed'].add(match.get('home_team', {}).get('home_team_id'))
                    stats['matches'] += 1
                    
                    # Get score (would need from team stats or separate data)
                    # For now, placeholder
                    home_score = match.get('home_score', 0)
                    away_score = match.get('away_score', 0)
                    
                    if home_score > away_score:
                        stats['wins'] += 1
                        stats['home_wins'] += 1
                    elif home_score < away_score:
                        stats['losses'] += 1
                        stats['home_losses'] += 1
                    else:
                        stats['draws'] += 1
                        stats['home_draws'] += 1
                    
                    stats['goals_for'] += home_score
                    stats['goals_against'] += away_score
                    
                    stats['match_history'].append({
                        'date': match.get('match_date'),
                        'team_id': match.get('home_team', {}).get('home_team_id'),
                        'is_home': True,
                        'result': 'W' if home_score > away_score else ('D' if home_score == away_score else 'L'),
                        'goals_for': home_score,
                        'goals_against': away_score
                    })
            
            # Away manager
            away_managers = match.get('away_team', {}).get('managers', [])
            if away_managers and len(away_managers) > 0:
                manager = away_managers[0]
                manager_id = manager.get('id')
                manager_name = manager.get('name')
                
                if manager_id:
                    stats = manager_stats[manager_id]
                    stats['manager_id'] = manager_id
                    stats['manager_name'] = manager_name
                    stats['teams_managed'].add(match.get('away_team', {}).get('away_team_id'))
                    stats['matches'] += 1
                    
                    home_score = match.get('home_score', 0)
                    away_score = match.get('away_score', 0)
                    
                    if away_score > home_score:
                        stats['wins'] += 1
                        stats['away_wins'] += 1
                    elif away_score < home_score:
                        stats['losses'] += 1
                        stats['away_losses'] += 1
                    else:
                        stats['draws'] += 1
                        stats['away_draws'] += 1
                    
                    stats['goals_for'] += away_score
                    stats['goals_against'] += home_score
                    
                    stats['match_history'].append({
                        'date': match.get('match_date'),
                        'team_id': match.get('away_team', {}).get('away_team_id'),
                        'is_home': False,
                        'result': 'W' if away_score > home_score else ('D' if away_score == home_score else 'L'),
                        'goals_for': away_score,
                        'goals_against': home_score
                    })
        
        return dict(manager_stats)
    
    def calculate_manager_ratings(self, manager_stats: Dict) -> Dict:
        """
        Calculate comprehensive manager ratings
        
        Returns ratings on various dimensions
        """
        matches = manager_stats['matches']
        if matches == 0:
            return {
                'overall': 50.0,
                'win_rate': 0.0,
                'home_performance': 50.0,
                'away_performance': 50.0,
                'offensive': 50.0,
                'defensive': 50.0,
                'consistency': 50.0
            }
        
        # Win rate based rating (0-100 scale)
        win_rate = manager_stats['wins'] / matches
        draw_rate = manager_stats['draws'] / matches
        points_per_match = (manager_stats['wins'] * 3 + manager_stats['draws']) / matches
        
        # Home vs Away performance
        home_matches = (manager_stats['home_wins'] + manager_stats['home_draws'] + 
                       manager_stats['home_losses'])
        away_matches = (manager_stats['away_wins'] + manager_stats['away_draws'] + 
                       manager_stats['away_losses'])
        
        home_points = (manager_stats['home_wins'] * 3 + manager_stats['home_draws'])
        away_points = (manager_stats['away_wins'] * 3 + manager_stats['away_draws'])
        
        home_performance = (home_points / (home_matches * 3) * 100) if home_matches > 0 else 50
        away_performance = (away_points / (away_matches * 3) * 100) if away_matches > 0 else 50
        
        # Offensive rating (goals per match)
        goals_per_match = manager_stats['goals_for'] / matches
        offensive_rating = min(100, goals_per_match * 40)  # 2.5 goals/match = 100
        
        # Defensive rating (goals conceded per match, inverted)
        goals_conceded_per_match = manager_stats['goals_against'] / matches
        defensive_rating = max(0, 100 - (goals_conceded_per_match * 50))  # 2 goals/match = 0
        
        # Consistency (standard deviation of results)
        recent_results = manager_stats['match_history'][-20:]  # Last 20 matches
        if len(recent_results) > 5:
            result_scores = [3 if m['result'] == 'W' else (1 if m['result'] == 'D' else 0) 
                           for m in recent_results]
            consistency = max(0, 100 - (np.std(result_scores) * 30))
        else:
            consistency = 50
        
        # Overall rating (weighted combination)
        overall = (
            points_per_match * 25 +  # Win rate (most important)
            offensive_rating * 0.2 +
            defensive_rating * 0.2 +
            (home_performance + away_performance) / 2 * 0.2 +
            consistency * 0.15
        )
        
        return {
            'overall': overall,
            'win_rate': win_rate * 100,
            'points_per_match': points_per_match,
            'home_performance': home_performance,
            'away_performance': away_performance,
            'offensive': offensive_rating,
            'defensive': defensive_rating,
            'consistency': consistency,
            'goals_per_match': goals_per_match,
            'goals_conceded_per_match': goals_conceded_per_match
        }
    
    def classify_manager_style(self, manager_stats: Dict, team_stats: List[Dict] = None) -> Dict:
        """
        Classify manager tactical style based on team performance patterns
        
        Returns style classification and confidence
        """
        ratings = self.calculate_manager_ratings(manager_stats)
        
        # Determine style based on offensive/defensive balance
        offensive = ratings['offensive']
        defensive = ratings['defensive']
        
        styles = []
        
        # Attacking style (high goals, may concede)
        if offensive > 70 and defensive < 60:
            styles.append(('Attacking', 0.8))
        elif offensive > 60:
            styles.append(('Attacking', 0.5))
        
        # Defensive style (low goals conceded, may not score much)
        if defensive > 70 and offensive < 60:
            styles.append(('Defensive', 0.8))
        elif defensive > 60:
            styles.append(('Defensive', 0.5))
        
        # Balanced style (good at both)
        if abs(offensive - defensive) < 15 and offensive > 55:
            styles.append(('Balanced', 0.7))
        
        # Pragmatic (results over style)
        if ratings['consistency'] > 70:
            styles.append(('Pragmatic', 0.6))
        
        # Determine primary style
        if styles:
            primary_style = max(styles, key=lambda x: x[1])
        else:
            primary_style = ('Balanced', 0.5)
        
        return {
            'primary_style': primary_style[0],
            'style_confidence': primary_style[1],
            'all_styles': styles,
            'tactical_flexibility': len(styles) * 20  # More styles = more flexible
        }
    
    def calculate_manager_team_fit(self, manager_id: int, team_id: int, 
                                   team_player_profiles: List[Dict]) -> float:
        """
        Calculate how well manager's style fits team's player profiles
        
        Returns fit score (0-100)
        """
        if manager_id not in self.manager_profiles:
            return 50.0  # Unknown manager
        
        manager_profile = self.manager_profiles[manager_id]
        style = manager_profile.get('style', {})
        
        if not team_player_profiles:
            return 50.0
        
        # Analyze team composition
        defenders = [p for p in team_player_profiles if p.get('position_group') == 'DEF']
        midfielders = [p for p in team_player_profiles if p.get('position_group') == 'MID']
        forwards = [p for p in team_player_profiles if p.get('position_group') == 'FWD']
        
        # Average player ratings by position
        def avg_rating(players, rating_type='overall'):
            if not players:
                return 5.0
            return np.mean([p.get(rating_type, 5.0) for p in players])
        
        team_defensive_quality = avg_rating(defenders, 'overall')
        team_offensive_quality = avg_rating(forwards, 'overall')
        team_midfield_quality = avg_rating(midfielders, 'overall')
        
        # Match manager style to team strengths
        primary_style = style.get('primary_style', 'Balanced')
        
        if primary_style == 'Attacking':
            # Attacking managers need good forwards and creative midfielders
            fit = (team_offensive_quality * 0.5 + team_midfield_quality * 0.3 + 
                  team_defensive_quality * 0.2) * 10
        elif primary_style == 'Defensive':
            # Defensive managers need solid defense and midfield
            fit = (team_defensive_quality * 0.5 + team_midfield_quality * 0.3 + 
                  team_offensive_quality * 0.2) * 10
        else:  # Balanced or Pragmatic
            # Need balance across all areas
            fit = ((team_defensive_quality + team_midfield_quality + team_offensive_quality) / 3) * 10
        
        return min(100, max(0, fit))
    
    def get_manager_h2h(self, manager1_id: int, manager2_id: int) -> Dict:
        """
        Get head-to-head record between two managers
        
        Returns win/draw/loss record
        """
        if (manager1_id not in self.manager_profiles or 
            manager2_id not in self.manager_profiles):
            return {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0}
        
        # This would require tracking opponents in match history
        # Placeholder implementation
        return {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0}
    
    def update_match_result(self, manager_id: int, manager_name: str, team_id: int,
                           match_date: str, goals_for: int, goals_against: int, is_home: bool = True):
        """Update manager profile with a single match result"""
        # Initialize profile if doesn't exist
        if manager_id not in self.manager_profiles:
            self.manager_profiles[manager_id] = {
                'manager_id': manager_id,
                'manager_name': manager_name,
                'stats': {
                    'matches': 0,
                    'wins': 0,
                    'draws': 0,
                    'losses': 0,
                    'goals_for': 0,
                    'goals_against': 0,
                    'home_wins': 0,
                    'home_draws': 0,
                    'home_losses': 0,
                    'away_wins': 0,
                    'away_draws': 0,
                    'away_losses': 0,
                    'teams_managed': set(),
                    'match_history': [],
                },
                'ratings': {
                    'overall': 50.0,
                    'offensive': 50.0,
                    'defensive': 50.0,
                    'home_performance': 50.0,
                    'away_performance': 50.0,
                    'consistency': 50.0,
                    'win_rate': 0.0,
                    'points_per_match': 0.0,
                },
                'style': {
                    'primary_style': 'Balanced',
                    'style_confidence': 0.5,
                },
                'teams_managed': set(),
                'total_matches': 0
            }
        
        profile = self.manager_profiles[manager_id]
        stats = profile['stats']
        
        # Update stats
        stats['matches'] += 1
        stats['goals_for'] += goals_for
        stats['goals_against'] += goals_against
        stats['teams_managed'].add(team_id)
        
        # Determine result
        if goals_for > goals_against:
            result = 'W'
            stats['wins'] += 1
            if is_home:
                stats['home_wins'] += 1
            else:
                stats['away_wins'] += 1
        elif goals_for < goals_against:
            result = 'L'
            stats['losses'] += 1
            if is_home:
                stats['home_losses'] += 1
            else:
                stats['away_losses'] += 1
        else:
            result = 'D'
            stats['draws'] += 1
            if is_home:
                stats['home_draws'] += 1
            else:
                stats['away_draws'] += 1
        
        # Add to match history
        stats['match_history'].append({
            'date': match_date,
            'team_id': team_id,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'result': result,
            'is_home': is_home
        })
        
        # Recalculate ratings if we have enough matches
        if stats['matches'] >= 5:
            # Create a temporary stats dict for rating calculation
            temp_stats = {
                'manager_name': manager_name,
                'matches': stats['matches'],
                'wins': stats['wins'],
                'draws': stats['draws'],
                'losses': stats['losses'],
                'goals_for': stats['goals_for'],
                'goals_against': stats['goals_against'],
                'home_wins': stats['home_wins'],
                'home_draws': stats['home_draws'],
                'home_losses': stats['home_losses'],
                'away_wins': stats['away_wins'],
                'away_draws': stats['away_draws'],
                'away_losses': stats['away_losses'],
                'teams_managed': stats['teams_managed'],
                'match_history': stats['match_history'],
            }
            profile['ratings'] = self.calculate_manager_ratings(temp_stats)
            profile['style'] = self.classify_manager_style(temp_stats)
        
        profile['total_matches'] = stats['matches']
        profile['teams_managed'] = stats['teams_managed']
    
    def update_profile(self, manager_id: int, manager_stats: Dict):
        """Update or create manager profile"""
        ratings = self.calculate_manager_ratings(manager_stats)
        style = self.classify_manager_style(manager_stats)
        
        self.manager_profiles[manager_id] = {
            'manager_id': manager_id,
            'manager_name': manager_stats['manager_name'],
            'stats': manager_stats,
            'ratings': ratings,
            'style': style,
            'teams_managed': list(manager_stats['teams_managed']),
            'total_matches': manager_stats['matches']
        }
    
    def export_profiles(self) -> pd.DataFrame:
        """Export all manager profiles to DataFrame"""
        profiles_list = []
        
        for manager_id, profile in self.manager_profiles.items():
            profiles_list.append({
                'manager_id': manager_id,
                'manager_name': profile['manager_name'],
                'total_matches': profile['total_matches'],
                'teams_managed_count': len(profile['teams_managed']),
                'primary_style': profile['style']['primary_style'],
                'overall_rating': profile['ratings']['overall'],
                'win_rate': profile['ratings']['win_rate'],
                'offensive_rating': profile['ratings']['offensive'],
                'defensive_rating': profile['ratings']['defensive'],
                'home_performance': profile['ratings']['home_performance'],
                'away_performance': profile['ratings']['away_performance'],
                'consistency': profile['ratings']['consistency']
            })
        
        return pd.DataFrame(profiles_list)
