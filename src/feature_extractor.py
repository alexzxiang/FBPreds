"""
Feature extraction from match events
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class FeatureExtractor:
    """Extract features from match events for players and teams"""
    
    def __init__(self):
        self.event_weights = {
            'Pass': 1.0,
            'Shot': 3.0,
            'Goal': 10.0,
            'Dribble': 2.0,
            'Interception': 2.5,
            'Tackle': 2.5,
            'Clearance': 1.5,
            'Block': 2.0,
            'Foul Committed': -1.5,
            'Foul Won': 1.5,
            'Dispossessed': -1.0,
            'Error': -2.0,
        }
    
    def extract_player_stats_from_events(self, events: List[Dict]) -> pd.DataFrame:
        """Extract player statistics from event data"""
        player_stats = defaultdict(lambda: {
            'player_id': None,
            'player_name': None,
            'team_id': None,
            'team_name': None,
            'position': None,
            'passes': 0,
            'passes_completed': 0,
            'key_passes': 0,
            'shots': 0,
            'shots_on_target': 0,
            'goals': 0,
            'assists': 0,
            'dribbles': 0,
            'dribbles_successful': 0,
            'tackles': 0,
            'interceptions': 0,
            'clearances': 0,
            'blocks': 0,
            'fouls_committed': 0,
            'fouls_won': 0,
            'dispossessed': 0,
            'errors': 0,
            'minutes_played': 0,
            'yellow_cards': 0,
            'red_cards': 0,
        })
        
        for event in events:
            event_type = event.get('type', {}).get('name', '')
            player_id = event.get('player', {}).get('id')
            
            if not player_id:
                continue
            
            player_name = event.get('player', {}).get('name', '')
            team_id = event.get('team', {}).get('id')
            team_name = event.get('team', {}).get('name', '')
            position = event.get('position', {}).get('name', '')
            
            stats = player_stats[player_id]
            stats['player_id'] = player_id
            stats['player_name'] = player_name
            stats['team_id'] = team_id
            stats['team_name'] = team_name
            if not stats['position']:
                stats['position'] = position
            
            # Extract statistics based on event type
            if event_type == 'Pass':
                stats['passes'] += 1
                if event.get('pass', {}).get('outcome') is None:
                    stats['passes_completed'] += 1
                if event.get('pass', {}).get('goal_assist'):
                    stats['assists'] += 1
                if event.get('pass', {}).get('shot_assist'):
                    stats['key_passes'] += 1
                    
            elif event_type == 'Shot':
                stats['shots'] += 1
                outcome = event.get('shot', {}).get('outcome', {}).get('name', '')
                if outcome in ['Goal', 'Saved']:
                    stats['shots_on_target'] += 1
                if outcome == 'Goal':
                    stats['goals'] += 1
                    
            elif event_type == 'Dribble':
                stats['dribbles'] += 1
                if event.get('dribble', {}).get('outcome', {}).get('name') == 'Complete':
                    stats['dribbles_successful'] += 1
                    
            elif event_type == 'Duel':
                duel_type = event.get('duel', {}).get('type', {}).get('name', '')
                if duel_type == 'Tackle':
                    stats['tackles'] += 1
                    
            elif event_type == 'Interception':
                stats['interceptions'] += 1
                
            elif event_type == 'Clearance':
                stats['clearances'] += 1
                
            elif event_type == 'Block':
                stats['blocks'] += 1
                
            elif event_type == 'Foul Committed':
                stats['fouls_committed'] += 1
                if event.get('foul_committed', {}).get('card'):
                    card = event.get('foul_committed', {}).get('card', {}).get('name', '')
                    if card == 'Yellow Card':
                        stats['yellow_cards'] += 1
                    elif card in ['Red Card', 'Second Yellow']:
                        stats['red_cards'] += 1
                        
            elif event_type == 'Foul Won':
                stats['fouls_won'] += 1
                
            elif event_type == 'Dispossessed':
                stats['dispossessed'] += 1
                
            elif event_type == 'Error':
                stats['errors'] += 1
        
        # Calculate minutes played from Starting XI and Substitution events
        for event in events:
            if event.get('type', {}).get('name') == 'Starting XI':
                lineup = event.get('tactics', {}).get('lineup', [])
                for player in lineup:
                    player_id = player.get('player', {}).get('id')
                    if player_id in player_stats:
                        player_stats[player_id]['minutes_played'] = 90  # Default
        
        return pd.DataFrame(list(player_stats.values()))
    
    def extract_team_stats_from_events(self, events: List[Dict]) -> Dict[int, Dict]:
        """Extract team-level statistics from events"""
        team_stats = defaultdict(lambda: {
            'team_id': None,
            'team_name': None,
            'possession': 0.0,
            'passes': 0,
            'passes_completed': 0,
            'shots': 0,
            'shots_on_target': 0,
            'goals': 0,
            'corners': 0,
            'fouls': 0,
            'yellow_cards': 0,
            'red_cards': 0,
            'offsides': 0,
        })
        
        total_possessions = defaultdict(int)
        total_possession_events = 0
        
        for event in events:
            team_id = event.get('team', {}).get('id')
            team_name = event.get('team', {}).get('name', '')
            event_type = event.get('type', {}).get('name', '')
            
            if not team_id:
                continue
            
            stats = team_stats[team_id]
            stats['team_id'] = team_id
            stats['team_name'] = team_name
            
            # Track possession
            if event.get('possession_team', {}).get('id') == team_id:
                total_possessions[team_id] += 1
            total_possession_events += 1
            
            # Event-based stats
            if event_type == 'Pass':
                stats['passes'] += 1
                if event.get('pass', {}).get('outcome') is None:
                    stats['passes_completed'] += 1
                    
            elif event_type == 'Shot':
                stats['shots'] += 1
                outcome = event.get('shot', {}).get('outcome', {}).get('name', '')
                if outcome in ['Goal', 'Saved']:
                    stats['shots_on_target'] += 1
                if outcome == 'Goal':
                    stats['goals'] += 1
                    
            elif event_type == 'Foul Committed':
                stats['fouls'] += 1
                if event.get('foul_committed', {}).get('card'):
                    card = event.get('foul_committed', {}).get('card', {}).get('name', '')
                    if card == 'Yellow Card':
                        stats['yellow_cards'] += 1
                    elif card in ['Red Card', 'Second Yellow']:
                        stats['red_cards'] += 1
                        
            elif event_type == 'Offside':
                stats['offsides'] += 1
        
        # Calculate possession percentages
        for team_id in team_stats:
            if total_possession_events > 0:
                team_stats[team_id]['possession'] = (total_possessions[team_id] / total_possession_events) * 100
        
        return dict(team_stats)
    
    def calculate_player_rating(self, player_stats: Dict) -> float:
        """Calculate a simple rating for a player based on their stats"""
        rating = 5.0  # Base rating
        
        # Offensive contributions
        rating += player_stats.get('goals', 0) * 1.5
        rating += player_stats.get('assists', 0) * 1.0
        rating += player_stats.get('key_passes', 0) * 0.3
        rating += player_stats.get('shots_on_target', 0) * 0.2
        
        # Passing
        passes = player_stats.get('passes', 0)
        passes_completed = player_stats.get('passes_completed', 0)
        if passes > 0:
            pass_accuracy = passes_completed / passes
            rating += pass_accuracy * 2.0
        
        # Dribbling
        dribbles = player_stats.get('dribbles', 0)
        dribbles_successful = player_stats.get('dribbles_successful', 0)
        if dribbles > 0:
            dribble_success = dribbles_successful / dribbles
            rating += dribble_success * 1.0
        
        # Defensive contributions
        rating += player_stats.get('tackles', 0) * 0.3
        rating += player_stats.get('interceptions', 0) * 0.3
        rating += player_stats.get('clearances', 0) * 0.2
        rating += player_stats.get('blocks', 0) * 0.2
        
        # Negative actions
        rating -= player_stats.get('fouls_committed', 0) * 0.3
        rating -= player_stats.get('dispossessed', 0) * 0.2
        rating -= player_stats.get('errors', 0) * 0.5
        rating -= player_stats.get('yellow_cards', 0) * 0.5
        rating -= player_stats.get('red_cards', 0) * 2.0
        
        return max(0, min(10, rating))  # Clamp between 0-10
