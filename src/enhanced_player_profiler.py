"""
Enhanced Player Profiling with Position-Specific Metrics
Properly weighted statistics based on position and impact on winning
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from collections import defaultdict


class EnhancedPlayerProfiler:
    """
    Enhanced player profiling with proper position-specific weighting
    
    Position Groups:
    - GK (Goalkeeper): Shot stopping, distribution, command of area
    - DEF (Defender): Defending, aerial ability, positioning, ball progression
    - MID (Midfielder): Passing, ball progression, defensive contribution, creativity
    - FWD (Forward/Winger): Goal scoring, shot creation, dribbling, pressing
    """
    
    def __init__(self):
        self.player_profiles = {}
        self.match_history = defaultdict(list)
        
        # Position-specific importance weights (based on impact on winning)
        # These weights are derived from football analytics research
        self.position_weights = {
            'GK': {
                # Goalkeepers: preventing goals is paramount
                'shot_stopping': 0.40,      # Most important - saves win games
                'distribution': 0.25,        # Build-up from back
                'command_of_area': 0.20,     # Crosses, set pieces
                'sweeping': 0.15,            # Modern keeper role
            },
            'DEF': {
                # Defenders: preventing dangerous situations
                'defensive_actions': 0.35,   # Tackles, interceptions, blocks
                'aerial_duels': 0.20,        # Set pieces, long balls
                'positioning': 0.20,         # Reducing chances
                'ball_progression': 0.15,    # Modern defender role
                'passing_accuracy': 0.10,    # Not losing possession
            },
            'MID': {
                # Midfielders: most balanced, controlling the game
                'passing_quality': 0.25,     # Controlling possession
                'chance_creation': 0.25,     # Assists, key passes
                'ball_progression': 0.20,    # Moving team forward
                'defensive_contribution': 0.15,  # Helping defense
                'dribbling': 0.15,           # Beating press
            },
            'FWD': {
                # Forwards: creating and scoring goals
                'goal_scoring': 0.40,        # Primary job
                'shot_creation': 0.25,       # Quality of chances
                'dribbling': 0.20,           # Creating space
                'link_up_play': 0.10,        # Assists, combinations
                'pressing': 0.05,            # Defensive work
            }
        }
        
        # Statistical benchmarks (per 90 minutes) for 7/10 rating
        # Based on top-level professional football
        self.benchmarks = {
            'GK': {
                'saves_per90': 3.0,
                'pass_completion': 0.75,
                'clearances_per90': 1.0,
                'high_claims_per90': 0.5,
            },
            'DEF': {
                'tackles_interceptions_per90': 4.0,
                'aerial_win_rate': 0.65,
                'clearances_per90': 4.0,
                'progressive_passes_per90': 3.0,
                'pass_completion': 0.85,
            },
            'MID': {
                'pass_completion': 0.85,
                'key_passes_per90': 1.5,
                'progressive_passes_per90': 5.0,
                'tackles_per90': 2.0,
                'dribble_success_rate': 0.60,
            },
            'FWD': {
                'goals_per90': 0.5,
                'xg_per90': 0.4,
                'shots_per90': 3.0,
                'shot_accuracy': 0.40,
                'dribble_success_rate': 0.55,
                'key_passes_per90': 1.0,
            }
        }
    
    def calculate_advanced_metrics(self, stats: Dict, events: List[Dict], player_id: int) -> Dict:
        """
        Calculate advanced metrics from events
        
        Returns comprehensive per-90 statistics
        """
        minutes = max(stats['minutes_played'], 1)
        p90 = 90 / minutes
        
        metrics = {
            'minutes_played': minutes,
            'position_group': stats['position_group'],
        }
        
        # Basic conversions to per-90
        metrics['goals_per90'] = stats['goals'] * p90
        metrics['assists_per90'] = stats['assists'] * p90
        metrics['shots_per90'] = stats['shots'] * p90
        metrics['shots_on_target_per90'] = stats['shots_on_target'] * p90
        
        # Passing metrics
        pass_attempts = max(stats['passes_attempted'], 1)
        metrics['pass_completion_rate'] = stats['passes_completed'] / pass_attempts
        metrics['passes_per90'] = stats['passes_attempted'] * p90
        metrics['progressive_passes_per90'] = stats.get('progressive_passes', 0) * p90
        metrics['key_passes_per90'] = stats['key_passes'] * p90
        
        # Defensive metrics
        metrics['tackles_per90'] = stats['tackles'] * p90
        metrics['interceptions_per90'] = stats['interceptions'] * p90
        metrics['clearances_per90'] = stats['clearances'] * p90
        metrics['tackles_interceptions_per90'] = (stats['tackles'] + stats['interceptions']) * p90
        
        # Aerial metrics
        aerial_attempts = max(stats['aerial_duels'], 1)
        metrics['aerial_win_rate'] = stats['aerial_duels_won'] / aerial_attempts
        metrics['aerials_won_per90'] = stats['aerial_duels_won'] * p90
        
        # Dribbling metrics
        dribble_attempts = max(stats['dribbles_attempted'], 1)
        metrics['dribble_success_rate'] = stats['dribbles_successful'] / dribble_attempts
        metrics['dribbles_per90'] = stats['dribbles_successful'] * p90
        
        # Shot quality
        shot_attempts = max(stats['shots'], 1)
        metrics['shot_accuracy'] = stats['shots_on_target'] / shot_attempts
        metrics['conversion_rate'] = stats['goals'] / shot_attempts if shot_attempts > 0 else 0
        
        # Goalkeeper specific
        if stats['position_group'] == 'GK':
            metrics['saves_per90'] = stats['saves'] * p90
            metrics['goals_conceded_per90'] = stats['goals_conceded'] * p90
            metrics['save_percentage'] = stats['saves'] / max(stats['saves'] + stats['goals_conceded'], 1)
        
        # Expected Goals (simplified - would need shot location data for real xG)
        # Estimate based on shot quality
        metrics['xg_per90'] = (stats['shots_on_target'] * 0.15 + stats['shots'] * 0.05) * p90
        
        # Ball carries (progressive movement with ball)
        metrics['carries_per90'] = stats.get('carries', 0) * p90
        metrics['progressive_carries_per90'] = stats.get('progressive_carries', 0) * p90
        
        # Pressures
        metrics['pressures_per90'] = stats['pressures'] * p90
        
        # Fouls and discipline
        metrics['fouls_per90'] = stats['fouls_committed'] * p90
        metrics['fouls_won_per90'] = stats.get('fouls_won', 0) * p90
        
        return metrics
    
    def calculate_position_specific_rating(self, metrics: Dict, position: str) -> Dict:
        """
        Calculate position-specific ratings using proper weighting
        
        Returns detailed ratings dict with overall score
        """
        if position not in self.position_weights:
            return {'overall': 5.0}
        
        benchmarks = self.benchmarks[position]
        weights = self.position_weights[position]
        ratings = {}
        
        if position == 'GK':
            # Goalkeeper ratings
            # Shot stopping: saves per 90 and save percentage
            saves = metrics.get('saves_per90', 0)
            save_pct = metrics.get('save_percentage', 0.5)
            ratings['shot_stopping'] = self._normalize_rating(
                (saves / benchmarks['saves_per90']) * 0.6 + save_pct * 0.4 * 10
            )
            
            # Distribution: pass completion
            pass_comp = metrics.get('pass_completion_rate', 0.5)
            ratings['distribution'] = self._normalize_rating(
                (pass_comp / benchmarks['pass_completion']) * 10
            )
            
            # Command of area: clearances and high claims
            clearances = metrics.get('clearances_per90', 0)
            ratings['command_of_area'] = self._normalize_rating(
                (clearances / benchmarks['clearances_per90']) * 10
            )
            
            # Sweeping: successful actions outside box (simplified)
            ratings['sweeping'] = 5.0  # Would need position data
            
        elif position == 'DEF':
            # Defender ratings
            # Defensive actions: tackles + interceptions + blocks
            def_actions = metrics.get('tackles_interceptions_per90', 0)
            ratings['defensive_actions'] = self._normalize_rating(
                (def_actions / benchmarks['tackles_interceptions_per90']) * 10
            )
            
            # Aerial duels
            aerial_rate = metrics.get('aerial_win_rate', 0.5)
            ratings['aerial_duels'] = self._normalize_rating(
                (aerial_rate / benchmarks['aerial_win_rate']) * 10
            )
            
            # Positioning: clearances and blocks
            clearances = metrics.get('clearances_per90', 0)
            ratings['positioning'] = self._normalize_rating(
                (clearances / benchmarks['clearances_per90']) * 10
            )
            
            # Ball progression: progressive passes and carries
            prog_passes = metrics.get('progressive_passes_per90', 0)
            ratings['ball_progression'] = self._normalize_rating(
                (prog_passes / benchmarks['progressive_passes_per90']) * 10
            )
            
            # Passing accuracy
            pass_comp = metrics.get('pass_completion_rate', 0.5)
            ratings['passing_accuracy'] = self._normalize_rating(
                (pass_comp / benchmarks['pass_completion']) * 10
            )
            
        elif position == 'MID':
            # Midfielder ratings
            # Passing quality: completion rate weighted by volume
            pass_comp = metrics.get('pass_completion_rate', 0.5)
            passes_per90 = metrics.get('passes_per90', 0)
            ratings['passing_quality'] = self._normalize_rating(
                (pass_comp / benchmarks['pass_completion']) * 8 +
                min(2, (passes_per90 / 50))  # Bonus for high volume
            )
            
            # Chance creation: key passes and assists
            key_passes = metrics.get('key_passes_per90', 0)
            assists = metrics.get('assists_per90', 0)
            ratings['chance_creation'] = self._normalize_rating(
                (key_passes / benchmarks['key_passes_per90']) * 7 +
                assists * 15  # Assists are highly valuable
            )
            
            # Ball progression: progressive passes and carries
            prog_passes = metrics.get('progressive_passes_per90', 0)
            prog_carries = metrics.get('progressive_carries_per90', 0)
            ratings['ball_progression'] = self._normalize_rating(
                (prog_passes / benchmarks['progressive_passes_per90']) * 6 +
                prog_carries * 0.5
            )
            
            # Defensive contribution: tackles and pressures
            tackles = metrics.get('tackles_per90', 0)
            pressures = metrics.get('pressures_per90', 0)
            ratings['defensive_contribution'] = self._normalize_rating(
                (tackles / benchmarks['tackles_per90']) * 7 +
                min(3, pressures * 0.1)
            )
            
            # Dribbling: success rate and volume
            dribble_rate = metrics.get('dribble_success_rate', 0.5)
            dribbles = metrics.get('dribbles_per90', 0)
            ratings['dribbling'] = self._normalize_rating(
                (dribble_rate / benchmarks['dribble_success_rate']) * 7 +
                min(3, dribbles * 0.5)
            )
            
        elif position == 'FWD':
            # Forward ratings
            # Goal scoring: goals and conversion rate
            goals = metrics.get('goals_per90', 0)
            conversion = metrics.get('conversion_rate', 0)
            xg = metrics.get('xg_per90', 0)
            ratings['goal_scoring'] = self._normalize_rating(
                (goals / benchmarks['goals_per90']) * 6 +
                conversion * 20 +  # Efficiency matters
                (xg / benchmarks['xg_per90']) * 4  # Getting into positions
            )
            
            # Shot creation: shots and xG
            shots = metrics.get('shots_per90', 0)
            shot_acc = metrics.get('shot_accuracy', 0.3)
            ratings['shot_creation'] = self._normalize_rating(
                (shots / benchmarks['shots_per90']) * 5 +
                (shot_acc / benchmarks['shot_accuracy']) * 5
            )
            
            # Dribbling: success rate for forwards
            dribble_rate = metrics.get('dribble_success_rate', 0.5)
            dribbles = metrics.get('dribbles_per90', 0)
            ratings['dribbling'] = self._normalize_rating(
                (dribble_rate / benchmarks['dribble_success_rate']) * 7 +
                min(3, dribbles * 0.4)
            )
            
            # Link-up play: key passes and assists
            key_passes = metrics.get('key_passes_per90', 0)
            assists = metrics.get('assists_per90', 0)
            ratings['link_up_play'] = self._normalize_rating(
                (key_passes / benchmarks['key_passes_per90']) * 5 +
                assists * 20
            )
            
            # Pressing: pressures and defensive actions
            pressures = metrics.get('pressures_per90', 0)
            ratings['pressing'] = self._normalize_rating(
                min(10, pressures * 0.15)
            )
        
        # Calculate weighted overall rating
        overall = sum(ratings[attr] * weights[attr] for attr in weights.keys())
        ratings['overall'] = overall
        
        # Add raw overall (simple average) for comparison
        ratings['overall_unweighted'] = np.mean(list(ratings.values()))
        
        return ratings
    
    def _normalize_rating(self, raw_score: float) -> float:
        """
        Normalize rating to 0-10 scale with proper distribution
        
        - 5.0 is average professional level
        - 7.0 is good/above average
        - 8.5+ is elite
        - 10.0 is world class (very rare)
        """
        # Clamp to 0-10 range
        return max(0.0, min(10.0, raw_score))
    
    def get_position_importance_ranking(self, metrics: Dict, position: str) -> List[tuple]:
        """
        Return ranked list of (metric, importance) for a position
        Helps understand what matters most for each position
        """
        if position not in self.position_weights:
            return []
        
        weights = self.position_weights[position]
        return sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    def compare_players(self, player1_id: int, player2_id: int) -> Dict:
        """
        Compare two players with position-aware metrics
        Only meaningful if same position
        """
        if player1_id not in self.player_profiles or player2_id not in self.player_profiles:
            return {'error': 'Player not found'}
        
        p1 = self.player_profiles[player1_id]
        p2 = self.player_profiles[player2_id]
        
        if p1.get('position_group') != p2.get('position_group'):
            return {'warning': 'Players play different positions - comparison may not be meaningful'}
        
        # Get latest ratings
        p1_rating = p1.get('current_rating', {})
        p2_rating = p2.get('current_rating', {})
        
        comparison = {
            'player1': {
                'name': p1.get('player_name'),
                'position': p1.get('position_group'),
                'overall': p1_rating.get('overall', 0),
                'matches': p1.get('matches_played', 0)
            },
            'player2': {
                'name': p2.get('player_name'),
                'position': p2.get('position_group'),
                'overall': p2_rating.get('overall', 0),
                'matches': p2.get('matches_played', 0)
            },
            'attribute_comparison': {}
        }
        
        # Compare each attribute
        for attr in p1_rating.keys():
            if attr in p2_rating and attr != 'overall_unweighted':
                comparison['attribute_comparison'][attr] = {
                    'player1': p1_rating[attr],
                    'player2': p2_rating[attr],
                    'difference': p1_rating[attr] - p2_rating[attr]
                }
        
        return comparison
    
    # =========================================================================
    # Pipeline Integration Methods
    # =========================================================================
    
    # Reuse position mapping from original profiler
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
        Wrapper for backward compatibility with pipeline
        """
        from .player_profiler import PlayerProfiler
        
        # Use original stat calculation (comprehensive event parsing)
        temp_profiler = PlayerProfiler()
        basic_stats = temp_profiler.calculate_player_stats(events, player_id, position)
        
        # Calculate advanced metrics
        advanced_metrics = self.calculate_advanced_metrics(basic_stats, events, player_id)
        
        return advanced_metrics
    
    def calculate_player_ratings(self, stats: Dict) -> Dict:
        """
        Calculate position-specific ratings from stats
        Wrapper for backward compatibility
        """
        position = stats.get('position_group', 'MID')
        return self.calculate_position_specific_rating(stats, position)
    
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
                'teams': set(),
                'current_team': team_id,
                'current_rating': match_ratings,
            }
        
        profile = self.player_profiles[player_id]
        
        # Update team information
        if team_id:
            profile['teams'].add(team_id)
            profile['current_team'] = team_id
        
        # Update career stats
        profile['matches_played'] += 1
        profile['total_minutes'] += match_stats.get('minutes_played', 90)
        
        for key, value in match_stats.items():
            if isinstance(value, (int, float)):
                profile['career_stats'][key] += value
        
        # Track ratings over time
        for rating_type, rating_value in match_ratings.items():
            # Ensure we only store numeric values
            if isinstance(rating_value, (int, float)):
                profile['career_ratings'][rating_type].append(rating_value)
        
        # Update recent form (last 5 matches)
        # Ensure ratings are flat (no nested dicts)
        flat_ratings = {
            k: float(v) if isinstance(v, (int, float)) else 5.0
            for k, v in match_ratings.items()
        }
        
        self.match_history[player_id].append({
            'date': match_date,
            'stats': match_stats,
            'ratings': flat_ratings
        })
        
        # Keep only recent matches for form
        profile['recent_form'] = self.match_history[player_id][-5:]
        
        # Update current rating (weighted recent form)
        profile['current_rating'] = self.get_player_current_rating(player_id)
    
    
    def get_player_current_rating(self, player_id: int) -> Dict:
        """Get player's current overall rating (weighted recent form)"""
        if player_id not in self.player_profiles:
            return {'overall': 5.0}
        
        profile = self.player_profiles[player_id]
        recent_matches = profile.get('recent_form', [])
        
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
            if weighted_ratings:
                weighted_ratings['overall'] = sum(weighted_ratings.values()) / len(weighted_ratings)
            else:
                weighted_ratings['overall'] = 5.0
        
        return weighted_ratings
    
    def export_profiles(self) -> pd.DataFrame:
        """Export all player profiles to DataFrame"""
        profiles_list = []
        
        for player_id, profile in self.player_profiles.items():
            current_ratings = self.get_player_current_rating(player_id)
            
            profile_data = {
                'player_id': player_id,
                'player_name': profile['player_name'],
                'position': profile['position_group'],
                'matches_played': profile['matches_played'],
                'total_minutes': profile['total_minutes'],
                'current_team': profile.get('current_team'),
                'overall_rating': current_ratings.get('overall', 0),
            }
            
            # Add position-specific ratings
            for rating_type, value in current_ratings.items():
                if rating_type != 'overall_unweighted':
                    profile_data[f'{rating_type}_rating'] = value
            
            profiles_list.append(profile_data)
        
        return pd.DataFrame(profiles_list)
    
    def update_profile_from_stats(self, player_id: int, player_name: str,
                                   position: str, match_date: str,
                                   stats: Dict, team_id: int = None,
                                   match_result: str = None):
        """
        Update player profile from simple stats dictionary.
        Used for Transfermarkt data integration.
        
        Args:
            player_id: Player ID
            player_name: Player name
            position: Player position
            match_date: Match date
            stats: Dictionary with stats (minutes_played, goals, assists, etc.)
            team_id: Team ID
            match_result: 'W', 'D', or 'L' for tracking win contribution
        """
        # Map position to position group
        position_map = {
            'Goalkeeper': 'GK',
            'Goalie': 'GK',
            'Defender': 'DEF',
            'Defence': 'DEF',
            'Centre-Back': 'DEF',
            'Left-Back': 'DEF',
            'Right-Back': 'DEF',
            'Defensive Midfield': 'MID',
            'Central Midfield': 'MID',
            'Attacking Midfield': 'MID',
            'Midfielder': 'MID',
            'Left Midfield': 'MID',
            'Right Midfield': 'MID',
            'Left Winger': 'FWD',
            'Right Winger': 'FWD',
            'Forward': 'FWD',
            'Striker': 'FWD',
            'Centre-Forward': 'FWD',
            'Attack': 'FWD',
        }
        
        position_group = position_map.get(position, 'MID')
        
        # Build match stats
        match_stats = {
            'position_group': position_group,
            'minutes_played': stats.get('minutes_played', 0),
            'goals': stats.get('goals', 0),
            'assists': stats.get('assists', 0),
            'shots': stats.get('shots', 0),
            'shots_on_target': stats.get('shots_on_target', 0),
            'passes': stats.get('passes', 0),
            'passes_completed': stats.get('passes_completed', 0),
            'tackles': stats.get('tackles', 0),
            'interceptions': stats.get('interceptions', 0),
            'clearances': stats.get('clearances', 0),
            'saves': stats.get('saves', 0),
            'yellow_cards': stats.get('yellow_cards', 0),
            'red_cards': stats.get('red_cards', 0),
        }
        
        # Calculate basic ratings from stats
        match_ratings = self.calculate_player_ratings_from_basic_stats(
            match_stats, position_group
        )
        
        # Update profile using existing method
        self.update_profile(
            player_id, player_name, match_date,
            match_stats, match_ratings, team_id
        )
        
        # Track win contribution if result provided
        if match_result and player_id in self.player_profiles:
            profile = self.player_profiles[player_id]
            if 'match_results' not in profile:
                profile['match_results'] = {'wins': 0, 'draws': 0, 'losses': 0}
            
            if match_result == 'W':
                profile['match_results']['wins'] += 1
            elif match_result == 'D':
                profile['match_results']['draws'] += 1
            elif match_result == 'L':
                profile['match_results']['losses'] += 1
            
            # Calculate win rate (important for player value)
            total_matches = sum(profile['match_results'].values())
            if total_matches > 0:
                profile['win_rate'] = profile['match_results']['wins'] / total_matches
                # Points per game (3 for win, 1 for draw)
                profile['points_per_game'] = (
                    (profile['match_results']['wins'] * 3 + profile['match_results']['draws']) / total_matches
                )
            else:
                profile['win_rate'] = 0.0
                profile['points_per_game'] = 0.0
    
    def calculate_player_ratings_from_basic_stats(self, stats: Dict, 
                                                   position_group: str) -> Dict:
        """
        Calculate player ratings from comprehensive stats with position-specific weighting.
        Uses detailed statistics from PlayerStats folder when available.
        """
        minutes = stats.get('minutes_played', 0)
        if minutes == 0:
            return {'overall': 5.0}
        
        # Calculate per-90 multiplier
        per_90 = 90.0 / minutes if minutes > 0 else 1.0
        
        # Initialize ratings dict
        ratings = {}
        
        # === ATTACKING RATINGS ===
        
        # Goal scoring (weighted by position)
        goals_per_90 = stats.get('goals', 0) * per_90
        shots_per_90 = stats.get('shots', 0) * per_90
        sot_per_90 = stats.get('shots_on_target', 0) * per_90
        
        if position_group == 'FWD':
            # Strikers: goals per 90, shot accuracy, volume
            shot_accuracy = stats.get('shots_on_target', 0) / max(stats.get('shots', 1), 1)
            ratings['goal_scoring'] = min(10, (goals_per_90 * 4) + (shot_accuracy * 2) + (shots_per_90 * 0.3))
        elif position_group == 'MID':
            # Midfielders: moderate goal threat
            ratings['goal_scoring'] = min(10, goals_per_90 * 6 + sot_per_90 * 0.5)
        else:
            # Defenders/GK: goals are bonus
            ratings['goal_scoring'] = min(10, goals_per_90 * 10)
        
        # Creativity (assists, key passes)
        assists_per_90 = stats.get('assists', 0) * per_90
        
        if position_group in ['MID', 'FWD']:
            # Midfielders and forwards: creativity crucial
            ratings['creativity'] = min(10, assists_per_90 * 5 + stats.get('key_passes', 0) * per_90 * 0.5)
        else:
            ratings['creativity'] = min(10, assists_per_90 * 8)
        
        # === PASSING RATINGS ===
        
        passes = stats.get('passes', 0)
        passes_completed = stats.get('passes_completed', 0)
        progressive_passes = stats.get('progressive_passes', 0)
        
        pass_completion = passes_completed / max(passes, 1) if passes > 0 else 0.75
        progressive_per_90 = progressive_passes * per_90
        
        if position_group == 'MID':
            # Midfielders: passing is critical (progressive passes matter!)
            ratings['passing'] = min(10, (pass_completion * 5) + (progressive_per_90 * 0.3) + (passes * per_90 * 0.01))
        elif position_group == 'DEF':
            # Defenders: accurate passing important
            ratings['passing'] = min(10, (pass_completion * 6) + (progressive_per_90 * 0.2))
        else:
            # Forwards: less emphasis
            ratings['passing'] = min(10, pass_completion * 8)
        
        # === DEFENSIVE RATINGS ===
        
        tackles_per_90 = stats.get('tackles', 0) * per_90
        interceptions_per_90 = stats.get('interceptions', 0) * per_90
        clearances_per_90 = stats.get('clearances', 0) * per_90
        blocks_per_90 = stats.get('blocks', 0) * per_90
        
        if position_group == 'DEF':
            # Defenders: defending is primary role
            ratings['defending'] = min(10, (tackles_per_90 * 0.8) + (interceptions_per_90 * 0.6) + 
                                      (clearances_per_90 * 0.3) + (blocks_per_90 * 0.4))
        elif position_group == 'MID':
            # Midfielders: tackles and interceptions matter
            ratings['defending'] = min(10, (tackles_per_90 * 0.6) + (interceptions_per_90 * 0.5))
        else:
            # Forwards: minimal defensive contribution
            ratings['defending'] = min(10, (tackles_per_90 + interceptions_per_90) * 0.3)
        
        # === GOALKEEPER-SPECIFIC ===
        
        if position_group == 'GK':
            save_pct = stats.get('save_percentage', 0) / 100.0 if stats.get('save_percentage', 0) > 1 else stats.get('save_percentage', 0.7)
            saves_per_90 = stats.get('saves', 0) * per_90
            clean_sheets_pct = stats.get('clean_sheet_percentage', 0) / 100.0 if stats.get('clean_sheet_percentage', 0) > 1 else stats.get('clean_sheet_percentage', 0.3)
            
            ratings['shot_stopping'] = min(10, (save_pct * 8) + (saves_per_90 * 0.2) + (clean_sheets_pct * 2))
        else:
            ratings['shot_stopping'] = 5.0
        
        # === PHYSICAL & DISCIPLINE ===
        
        # Aerial duels (important for defenders and target forwards)
        aerial_won_per_90 = stats.get('aerials_won', 0) * per_90
        if position_group in ['DEF', 'FWD']:
            ratings['aerial'] = min(10, aerial_won_per_90 * 0.8)
        else:
            ratings['aerial'] = min(10, aerial_won_per_90 * 0.5)
        
        # Discipline (cards hurt rating)
        yellow_cards = stats.get('yellow_cards', 0)
        red_cards = stats.get('red_cards', 0)
        fouls_per_90 = stats.get('fouls', 0) * per_90
        
        discipline_penalty = (yellow_cards * 0.3) + (red_cards * 2) + (fouls_per_90 * 0.1)
        ratings['discipline'] = max(0, 10 - discipline_penalty)
        
        # === CALCULATE POSITION-SPECIFIC OVERALL ===
        
        if position_group == 'GK':
            weights = {
                'shot_stopping': 0.50,
                'passing': 0.15,
                'discipline': 0.10,
                'defending': 0.25
            }
        elif position_group == 'DEF':
            weights = {
                'defending': 0.40,
                'passing': 0.20,
                'aerial': 0.15,
                'discipline': 0.10,
                'goal_scoring': 0.05,
                'creativity': 0.10
            }
        elif position_group == 'MID':
            weights = {
                'passing': 0.30,
                'creativity': 0.25,
                'defending': 0.15,
                'goal_scoring': 0.15,
                'discipline': 0.10,
                'aerial': 0.05
            }
        else:  # FWD
            weights = {
                'goal_scoring': 0.45,
                'creativity': 0.20,
                'passing': 0.15,
                'aerial': 0.10,
                'discipline': 0.05,
                'defending': 0.05
            }
        
        overall = sum(ratings.get(k, 5.0) * w for k, w in weights.items())
        ratings['overall'] = min(10.0, max(1.0, overall))
        
        return ratings


