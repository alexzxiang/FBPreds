"""
Player Profiling System - Track individual player performance over time
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime


class PlayerProfiler:
    """Build and maintain detailed player profiles"""
    
    # Position categories for specialized stats
    POSITION_GROUPS = {
        'GK': ['Goalkeeper'],
        'DEF': ['Left Back', 'Right Back', 'Left Center Back', 'Right Center Back', 
                'Center Back', 'Left Wing Back', 'Right Wing Back'],
        'MID': ['Left Defensive Midfield', 'Right Defensive Midfield', 'Center Defensive Midfield',
                'Left Midfield', 'Right Midfield', 'Center Midfield', 'Left Center Midfield', 
                'Right Center Midfield', 'Left Attacking Midfield', 'Right Attacking Midfield',
                'Center Attacking Midfield'],
        'FWD': ['Left Wing', 'Right Wing', 'Left Center Forward', 'Right Center Forward',
                'Center Forward', 'Secondary Striker']
    }
    
    def __init__(self):
        self.player_profiles = {}  # player_id -> profile dict
        self.match_history = defaultdict(list)  # player_id -> list of match performances
        
    def get_position_group(self, position: str) -> str:
        """Classify position into GK, DEF, MID, or FWD"""
        for group, positions in self.POSITION_GROUPS.items():
            if position in positions:
                return group
        return 'MID'  # Default
    
    def calculate_player_stats(self, events: List[Dict], player_id: int, 
                              position: str) -> Dict:
        """
        Calculate comprehensive player statistics from match events
        
        Returns position-specific stats
        """
        pos_group = self.get_position_group(position)
        
        # Base stats for all positions
        stats = {
            'player_id': player_id,
            'position': position,
            'position_group': pos_group,
            'minutes_played': 90,  # Default, can be refined
            
            # Passing
            'passes_attempted': 0,
            'passes_completed': 0,
            'progressive_passes': 0,
            'key_passes': 0,
            'through_balls': 0,
            'long_balls': 0,
            'crosses': 0,
            
            # Shooting
            'shots': 0,
            'shots_on_target': 0,
            'goals': 0,
            'assists': 0,
            'xg': 0.0,  # To be calculated
            
            # Dribbling & Ball Carrying
            'dribbles_attempted': 0,
            'dribbles_successful': 0,
            'carries': 0,
            'progressive_carries': 0,
            
            # Defensive
            'tackles': 0,
            'tackles_won': 0,
            'interceptions': 0,
            'clearances': 0,
            'blocks': 0,
            'aerial_duels': 0,
            'aerial_duels_won': 0,
            
            # Goalkeeper specific
            'saves': 0,
            'saves_from_shots': 0,
            'goals_conceded': 0,
            'clean_sheet': False,
            
            # Physical & Discipline
            'fouls_committed': 0,
            'fouls_won': 0,
            'yellow_cards': 0,
            'red_cards': 0,
            'errors_leading_to_shot': 0,
            
            # Advanced metrics
            'pressures': 0,
            'pressure_success': 0,
            'dispossessed': 0,
            'miscontrols': 0,
        }
        
        # Extract stats from events
        for event in events:
            event_player_id = event.get('player', {}).get('id')
            if event_player_id != player_id:
                continue
            
            event_type = event.get('type', {}).get('name', '')
            
            # Passing events
            if event_type == 'Pass':
                stats['passes_attempted'] += 1
                pass_data = event.get('pass', {})
                
                if pass_data.get('outcome') is None:
                    stats['passes_completed'] += 1
                
                if pass_data.get('goal_assist'):
                    stats['assists'] += 1
                
                if pass_data.get('shot_assist'):
                    stats['key_passes'] += 1
                
                if pass_data.get('through_ball'):
                    stats['through_balls'] += 1
                
                if pass_data.get('cross'):
                    stats['crosses'] += 1
                
                # Progressive pass: moves ball significantly forward
                if pass_data.get('end_location') and pass_data.get('length', 0) > 30:
                    stats['progressive_passes'] += 1
                
                if pass_data.get('height', {}).get('name') == 'High Pass':
                    stats['long_balls'] += 1
            
            # Shooting events
            elif event_type == 'Shot':
                stats['shots'] += 1
                shot_data = event.get('shot', {})
                outcome = shot_data.get('outcome', {}).get('name', '')
                
                if outcome in ['Goal', 'Saved', 'Saved to Post']:
                    stats['shots_on_target'] += 1
                
                if outcome == 'Goal':
                    stats['goals'] += 1
                
                # Calculate basic xG (simplified)
                location = event.get('location', [0, 0])
                goal_location = [120, 40]  # Approximate goal center
                distance = np.sqrt((location[0] - goal_location[0])**2 + 
                                 (location[1] - goal_location[1])**2)
                shot_xg = max(0, 0.5 - (distance / 100))  # Simplified xG
                stats['xg'] += shot_xg
            
            # Dribbling events
            elif event_type == 'Dribble':
                stats['dribbles_attempted'] += 1
                if event.get('dribble', {}).get('outcome', {}).get('name') == 'Complete':
                    stats['dribbles_successful'] += 1
            
            # Carry events
            elif event_type == 'Carry':
                stats['carries'] += 1
                carry_data = event.get('carry', {})
                if carry_data.get('end_location') and carry_data.get('end_location')[0] > 60:
                    stats['progressive_carries'] += 1
            
            # Defensive events
            elif event_type == 'Duel':
                duel_type = event.get('duel', {}).get('type', {}).get('name', '')
                if duel_type == 'Tackle':
                    stats['tackles'] += 1
                    if event.get('duel', {}).get('outcome', {}).get('name') == 'Won':
                        stats['tackles_won'] += 1
                elif 'Aerial' in duel_type:
                    stats['aerial_duels'] += 1
                    if event.get('duel', {}).get('outcome', {}).get('name') == 'Won':
                        stats['aerial_duels_won'] += 1
            
            elif event_type == 'Interception':
                stats['interceptions'] += 1
            
            elif event_type == 'Clearance':
                stats['clearances'] += 1
            
            elif event_type == 'Block':
                stats['blocks'] += 1
            
            # Goalkeeper events
            elif event_type == 'Goal Keeper':
                gk_type = event.get('goalkeeper', {}).get('type', {}).get('name', '')
                if 'Save' in gk_type:
                    stats['saves'] += 1
                    stats['saves_from_shots'] += 1
            
            # Pressure events
            elif event_type == 'Pressure':
                stats['pressures'] += 1
                # Check if next event by same team is successful
                # This is simplified - need full context
            
            # Negative events
            elif event_type == 'Foul Committed':
                stats['fouls_committed'] += 1
                card = event.get('foul_committed', {}).get('card', {}).get('name', '')
                if card == 'Yellow Card':
                    stats['yellow_cards'] += 1
                elif card in ['Red Card', 'Second Yellow']:
                    stats['red_cards'] += 1
            
            elif event_type == 'Foul Won':
                stats['fouls_won'] += 1
            
            elif event_type == 'Dispossessed':
                stats['dispossessed'] += 1
            
            elif event_type == 'Miscontrol':
                stats['miscontrols'] += 1
            
            elif event_type == 'Error':
                stats['errors_leading_to_shot'] += 1
        
        return stats
    
    def calculate_player_ratings(self, stats: Dict) -> Dict:
        """
        Calculate position-specific ratings from stats
        
        Returns ratings on 0-10 scale for different attributes
        """
        pos_group = stats['position_group']
        minutes = max(stats['minutes_played'], 1)
        
        ratings = {}
        
        # Normalize per 90 minutes
        p90 = 90 / minutes
        
        if pos_group == 'GK':
            # Goalkeeper ratings
            saves = stats['saves'] * p90
            goals_conceded = stats['goals_conceded'] * p90
            
            ratings['shot_stopping'] = min(10, saves * 2)
            ratings['distribution'] = min(10, (stats['passes_completed'] / max(stats['passes_attempted'], 1)) * 10)
            ratings['command'] = min(10, stats['clearances'] * p90 * 0.5)
            ratings['overall'] = np.mean([ratings['shot_stopping'], ratings['distribution'], ratings['command']])
        
        elif pos_group == 'DEF':
            # Defender ratings
            tackles = stats['tackles'] * p90
            interceptions = stats['interceptions'] * p90
            clearances = stats['clearances'] * p90
            aerial_success = stats['aerial_duels_won'] / max(stats['aerial_duels'], 1)
            
            ratings['defending'] = min(10, (tackles + interceptions) * 0.3)
            ratings['aerial'] = aerial_success * 10
            ratings['positioning'] = min(10, (clearances + stats['blocks'] * p90) * 0.3)
            ratings['passing'] = min(10, (stats['passes_completed'] / max(stats['passes_attempted'], 1)) * 10)
            ratings['overall'] = np.mean([ratings['defending'], ratings['aerial'], 
                                         ratings['positioning'], ratings['passing']])
        
        elif pos_group == 'MID':
            # Midfielder ratings
            pass_accuracy = stats['passes_completed'] / max(stats['passes_attempted'], 1)
            key_passes = stats['key_passes'] * p90
            tackles = stats['tackles'] * p90
            
            ratings['passing'] = pass_accuracy * 10
            ratings['vision'] = min(10, (key_passes + stats['through_balls'] * p90) * 2)
            ratings['work_rate'] = min(10, (tackles + stats['pressures'] * p90 * 0.1))
            ratings['dribbling'] = min(10, (stats['dribbles_successful'] / max(stats['dribbles_attempted'], 1)) * 10)
            ratings['overall'] = np.mean([ratings['passing'], ratings['vision'], 
                                         ratings['work_rate'], ratings['dribbling']])
        
        elif pos_group == 'FWD':
            # Forward ratings
            goals = stats['goals'] * p90
            shots_accuracy = stats['shots_on_target'] / max(stats['shots'], 1)
            dribble_success = stats['dribbles_successful'] / max(stats['dribbles_attempted'], 1)
            
            ratings['finishing'] = min(10, goals * 5 + shots_accuracy * 5)
            ratings['shooting'] = min(10, stats['shots'] * p90 * 0.5)
            ratings['dribbling'] = dribble_success * 10
            ratings['positioning'] = min(10, (stats['xg'] * p90) * 5)
            ratings['overall'] = np.mean([ratings['finishing'], ratings['shooting'], 
                                         ratings['dribbling'], ratings['positioning']])
        
        # Add speed/strength indicators (inferred from stats)
        ratings['pace'] = min(10, (stats['progressive_carries'] * p90) * 0.5)  # Simplified
        ratings['physicality'] = min(10, (stats['aerial_duels_won'] + stats['fouls_won'] * p90))
        
        return ratings
    
    def update_profile(self, player_id: int, player_name: str, match_date: str,
                      match_stats: Dict, match_ratings: Dict, team_id: int = None):
        """Update player profile with new match data"""
        if player_id not in self.player_profiles:
            self.player_profiles[player_id] = {
                'player_id': player_id,
                'player_name': player_name,
                'position_group': match_stats['position_group'],
                'matches_played': 0,
                'total_minutes': 0,
                'career_stats': defaultdict(float),
                'recent_form': [],  # Last 5 matches
                'career_ratings': defaultdict(list),
                'teams': set(),  # Track which teams player has played for
                'current_team': team_id,
            }
        
        profile = self.player_profiles[player_id]
        
        # Update team information
        if team_id:
            profile['teams'].add(team_id)
            profile['current_team'] = team_id
        
        # Update career stats
        profile['matches_played'] += 1
        profile['total_minutes'] += match_stats['minutes_played']
        
        for key, value in match_stats.items():
            if isinstance(value, (int, float)):
                profile['career_stats'][key] += value
        
        # Track ratings over time
        for rating_type, rating_value in match_ratings.items():
            profile['career_ratings'][rating_type].append(rating_value)
        
        # Update recent form (last 5 matches)
        self.match_history[player_id].append({
            'date': match_date,
            'stats': match_stats,
            'ratings': match_ratings
        })
        
        # Keep only recent matches for form
        profile['recent_form'] = self.match_history[player_id][-5:]
    
    def get_player_current_rating(self, player_id: int) -> Dict:
        """Get player's current overall rating (weighted recent form)"""
        if player_id not in self.player_profiles:
            return {'overall': 5.0}  # Default
        
        profile = self.player_profiles[player_id]
        recent_matches = profile['recent_form']
        
        if not recent_matches:
            return {'overall': 5.0}
        
        # Weight recent matches more heavily
        weights = [0.3, 0.25, 0.20, 0.15, 0.10][:len(recent_matches)]
        weights = weights[::-1]  # Most recent has highest weight
        
        # Get all unique rating types across all recent matches
        all_rating_types = set()
        for match in recent_matches:
            all_rating_types.update(match['ratings'].keys())
        
        weighted_ratings = {}
        for rating_type in all_rating_types:
            # Only include matches that have this rating type
            values_and_weights = [
                (match['ratings'].get(rating_type, 5.0), weight)
                for match, weight in zip(recent_matches[::-1], weights)
                if rating_type in match['ratings']
            ]
            
            if values_and_weights:
                total_weight = sum(w for _, w in values_and_weights)
                weighted_rating = sum(v * w for v, w in values_and_weights) / total_weight
                weighted_ratings[rating_type] = weighted_rating
        
        # Ensure 'overall' is present
        if 'overall' not in weighted_ratings:
            # Calculate overall from available ratings
            if weighted_ratings:
                weighted_ratings['overall'] = sum(weighted_ratings.values()) / len(weighted_ratings)
            else:
                weighted_ratings['overall'] = 5.0
        
        return weighted_ratings
    
    def get_players_by_position(self, team_id: int, position_group: str) -> List[int]:
        """Get all players for a team in a specific position group"""
        # This would need team roster data - placeholder
        return []
    
    def export_profiles(self) -> pd.DataFrame:
        """Export all player profiles to DataFrame"""
        profiles_list = []
        
        for player_id, profile in self.player_profiles.items():
            current_ratings = self.get_player_current_rating(player_id)
            
            avg_stats = {}
            if profile['matches_played'] > 0:
                for stat, total in profile['career_stats'].items():
                    avg_stats[f'avg_{stat}'] = total / profile['matches_played']
            
            profiles_list.append({
                'player_id': player_id,
                'player_name': profile['player_name'],
                'position_group': profile['position_group'],
                'matches_played': profile['matches_played'],
                'total_minutes': profile['total_minutes'],
                'current_overall_rating': current_ratings.get('overall', 5.0),
                **current_ratings,
                **avg_stats
            })
        
        return pd.DataFrame(profiles_list)
