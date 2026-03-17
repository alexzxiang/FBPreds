"""
Transfermarkt Data Parser
Parses game data from transfermarkt format to integrate with our prediction system.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TransfermarktParser:
    """Parse Transfermarkt data into format compatible with our pipeline."""
    
    def __init__(self, games_dir: str = "games", player_stats_dir: str = "PlayerStats"):
        self.games_dir = Path(games_dir)
        self.player_stats_dir = Path(player_stats_dir)
        
        # Elite competitions mapping (from transfermarkt to our system)
        self.ELITE_COMPETITIONS = {
            'GB1': 'Premier League',
            'ES1': 'La Liga', 
            'L1': 'Bundesliga',
            'IT1': 'Serie A',
            'FR1': 'Ligue 1',
            'CL': 'Champions League',
            'EL': 'Europa League',
            'UCOL': 'Conference League'
        }
        
        # Load all base data
        self.games_df = None
        self.club_games_df = None
        self.appearances_df = None
        self.game_lineups_df = None
        self.game_events_df = None
        self.players_df = None
        self.clubs_df = None
        self.competitions_df = None
        self.current_season_players_df = None
        
    def load_all_data(self):
        """Load all CSV files into memory."""
        print("📂 Loading Transfermarkt data...")
        
        # Load games data
        self.games_df = pd.read_csv(self.games_dir / "games.csv")
        self.club_games_df = pd.read_csv(self.games_dir / "club_games.csv")
        self.appearances_df = pd.read_csv(self.games_dir / "appearances.csv")
        self.game_lineups_df = pd.read_csv(self.games_dir / "game_lineups.csv")
        self.game_events_df = pd.read_csv(self.games_dir / "game_events.csv")
        self.players_df = pd.read_csv(self.games_dir / "players.csv")
        self.clubs_df = pd.read_csv(self.games_dir / "clubs.csv")
        self.competitions_df = pd.read_csv(self.games_dir / "competitions.csv")
        
        # Load current season player stats
        self.current_season_players_df = pd.read_csv(
            self.player_stats_dir / "players_data-2025_2026.csv"
        )
        
        print(f"✅ Loaded {len(self.games_df):,} games from {self.games_df['season'].min()}-{self.games_df['season'].max()}")
        print(f"✅ Loaded {len(self.players_df):,} players")
        print(f"✅ Loaded {len(self.clubs_df):,} clubs")
        print(f"✅ Loaded {len(self.current_season_players_df):,} current season player stats")
        
    def filter_elite_competitions(self, season: Optional[int] = None) -> pd.DataFrame:
        """Filter games to only elite competitions."""
        elite_games = self.games_df[
            self.games_df['competition_id'].isin(self.ELITE_COMPETITIONS.keys())
        ].copy()
        
        if season:
            elite_games = elite_games[elite_games['season'] == season]
            
        print(f"🏆 Filtered to {len(elite_games):,} elite competition games")
        return elite_games
    
    def parse_match_to_statsbomb_format(self, game_row: pd.Series) -> Dict:
        """
        Convert a Transfermarkt game to StatsBomb-like format.
        This allows us to reuse our existing pipeline.
        """
        match_id = game_row['game_id']
        
        # Get lineups for this match
        lineups = self.game_lineups_df[self.game_lineups_df['game_id'] == match_id]
        home_lineup = lineups[lineups['club_id'] == game_row['home_club_id']]
        away_lineup = lineups[lineups['club_id'] == game_row['away_club_id']]
        
        # Get appearances (player stats for this match)
        match_appearances = self.appearances_df[self.appearances_df['game_id'] == match_id]
        
        # Get events for this match
        match_events = self.game_events_df[self.game_events_df['game_id'] == match_id]
        
        # Build StatsBomb-like structure
        match_data = {
            'match_id': match_id,
            'match_date': game_row['date'],
            'season': {
                'season_id': game_row['season'],
                'season_name': f"{game_row['season']}/{game_row['season']+1}"
            },
            'competition': {
                'competition_id': game_row['competition_id'],
                'competition_name': self.ELITE_COMPETITIONS.get(
                    game_row['competition_id'], 
                    game_row['competition_id']
                )
            },
            'home_team': {
                'home_team_id': game_row['home_club_id'],
                'home_team_name': game_row.get('home_club_name', f"Club_{game_row['home_club_id']}"),
                'home_team_gender': 'male',
                'managers': [{'name': game_row.get('home_club_manager_name', 'Unknown')}],
                'lineup': self._parse_lineup(home_lineup, match_appearances, game_row['home_club_id'])
            },
            'away_team': {
                'away_team_id': game_row['away_club_id'],
                'away_team_name': game_row.get('away_club_name', f"Club_{game_row['away_club_id']}"),
                'away_team_gender': 'male',
                'managers': [{'name': game_row.get('away_club_manager_name', 'Unknown')}],
                'lineup': self._parse_lineup(away_lineup, match_appearances, game_row['away_club_id'])
            },
            'home_score': int(game_row['home_club_goals']) if pd.notna(game_row['home_club_goals']) else 0,
            'away_score': int(game_row['away_club_goals']) if pd.notna(game_row['away_club_goals']) else 0,
            'match_status': 'available',
            'metadata': {
                'stadium': game_row.get('stadium', 'Unknown'),
                'attendance': game_row.get('attendance'),
                'referee': game_row.get('referee'),
                'home_formation': game_row.get('home_club_formation'),
                'away_formation': game_row.get('away_club_formation')
            }
        }
        
        return match_data
    
    def _parse_lineup(self, lineup_df: pd.DataFrame, appearances_df: pd.DataFrame, 
                      club_id: int) -> List[Dict]:
        """Parse lineup data into StatsBomb-like format."""
        lineup = []
        
        # Get appearances for this club
        club_appearances = appearances_df[appearances_df['player_club_id'] == club_id]
        
        for _, player_row in lineup_df.iterrows():
            player_id = player_row['player_id']
            
            # Get appearance stats for this player
            player_appearance = club_appearances[
                club_appearances['player_id'] == player_id
            ]
            
            # Get player info from master players table
            player_info = self.players_df[self.players_df['player_id'] == player_id]
            
            player_data = {
                'player_id': player_id,
                'player_name': player_row.get('player_name', f"Player_{player_id}"),
                'player_nickname': player_row.get('player_name', '').split()[-1] if pd.notna(player_row.get('player_name')) else '',
                'position': player_row.get('position', 'Unknown'),
                'jersey_number': player_row.get('number'),
                'captain': bool(player_row.get('team_captain', 0)),
                'starting': player_row.get('type') == 'starting_lineup',
            }
            
            # Add match statistics if available
            if len(player_appearance) > 0:
                app = player_appearance.iloc[0]
                player_data.update({
                    'minutes_played': app.get('minutes_played', 0),
                    'goals': app.get('goals', 0),
                    'assists': app.get('assists', 0),
                    'yellow_cards': app.get('yellow_cards', 0),
                    'red_cards': app.get('red_cards', 0),
                })
            
            # Add player bio info if available
            if len(player_info) > 0:
                info = player_info.iloc[0]
                player_data.update({
                    'country': info.get('country_of_citizenship'),
                    'date_of_birth': info.get('date_of_birth'),
                    'position_primary': info.get('position'),
                    'foot': info.get('foot'),
                    'height_cm': info.get('height_in_cm'),
                    'market_value': info.get('market_value_in_eur'),
                })
            
            lineup.append(player_data)
        
        return lineup
    
    def get_player_aggregated_stats(self, player_id: int, as_of_date: str, 
                                     season: int = None) -> Dict:
        """
        Get aggregated season stats for a player up to a specific date.
        This provides cumulative performance data for player profiling.
        """
        # Filter appearances for this player
        player_apps = self.appearances_df[self.appearances_df['player_id'] == player_id].copy()
        
        # Join with games to get match dates
        player_apps = player_apps.merge(
            self.games_df[['game_id', 'date', 'season']], 
            on='game_id', 
            how='left'
        )
        
        # Filter to matches before the given date
        player_apps = player_apps[player_apps['date'] < as_of_date]
        
        # Filter to specific season if provided
        if season is not None:
            player_apps = player_apps[player_apps['season'] == season]
        
        if len(player_apps) == 0:
            return {}
        
        # Get player info for position
        player_info = self.players_df[self.players_df['player_id'] == player_id]
        position = player_info.iloc[0].get('position', 'Unknown') if len(player_info) > 0 else 'Unknown'
        
        # Aggregate stats
        stats = {
            'games_played': len(player_apps),
            'minutes_played': int(player_apps['minutes_played'].sum()),
            'goals': int(player_apps['goals'].sum()),
            'assists': int(player_apps['assists'].sum()),
            'yellow_cards': int(player_apps['yellow_cards'].sum()),
            'red_cards': int(player_apps['red_cards'].sum()),
            'position': position,
        }
        
        return stats
    
    def get_current_season_player_stats(self, player_name: str, team_name: str = None) -> Dict:
        """
        Get current season stats for a player from the PlayerStats data.
        This provides the most up-to-date performance metrics.
        """
        # Try to find player in current season data
        player_matches = self.current_season_players_df[
            self.current_season_players_df['Player'].str.contains(player_name, case=False, na=False)
        ]
        
        if team_name:
            player_matches = player_matches[
                player_matches['Squad'].str.contains(team_name, case=False, na=False)
            ]
        
        if len(player_matches) == 0:
            return {}
        
        # Take the first match (or sum if multiple entries)
        if len(player_matches) > 1:
            player_stats = player_matches.iloc[0]
        else:
            player_stats = player_matches.iloc[0]
        
        # Extract comprehensive statistics with position-specific focus
        stats = {
            # Basic playing time
            'games_played': player_stats.get('MP', 0),
            'starts': player_stats.get('Starts', 0),
            'minutes_played': player_stats.get('Min', 0),
            
            # Attacking stats
            'goals': player_stats.get('Gls', 0),
            'assists': player_stats.get('Ast', 0),
            'shots': player_stats.get('Sh', 0),
            'shots_on_target': player_stats.get('SoT', 0),
            
            # Discipline
            'yellow_cards': player_stats.get('CrdY', 0),
            'red_cards': player_stats.get('CrdR', 0),
            
            # Defensive stats (from misc stats)
            'tackles': player_stats.get('TklW', 0),  # Tackles won
            'interceptions': player_stats.get('Int', 0),
            'fouls': player_stats.get('Fls', 0),
            'fouls_drawn': player_stats.get('Fld', 0),
            'clearances': player_stats.get('Crs', 0),  # Note: this might be crosses, check
            
            # Goalkeeper stats
            'saves': player_stats.get('Saves', 0),
            'save_percentage': player_stats.get('Save%', 0),
            'clean_sheets': player_stats.get('CS', 0),
            'clean_sheet_percentage': player_stats.get('CS%', 0),
            'goals_against': player_stats.get('GA', 0),
            
            # Additional metadata
            'position': player_stats.get('Pos', ''),
            'squad': player_stats.get('Squad', ''),
            'competition': player_stats.get('Comp', ''),
            'age': player_stats.get('Age', 0),
        }
        
        # Remove None values and convert to proper types
        stats = {k: v for k, v in stats.items() if pd.notna(v)}
        
        # Convert percentages to decimals if needed
        if 'save_percentage' in stats and stats['save_percentage'] > 1:
            stats['save_percentage'] = stats['save_percentage'] / 100.0
        if 'clean_sheet_percentage' in stats and stats['clean_sheet_percentage'] > 1:
            stats['clean_sheet_percentage'] = stats['clean_sheet_percentage'] / 100.0
        
        return stats
    
    def build_training_dataset(self, seasons: List[int] = None, 
                               elite_only: bool = True) -> List[Dict]:
        """
        Build a complete training dataset from Transfermarkt data.
        Returns list of matches in StatsBomb-like format.
        """
        if seasons is None:
            # Use recent seasons (2020-2025)
            seasons = [2020, 2021, 2022, 2023, 2024, 2025]
        
        print(f"\n🏗️  Building training dataset for seasons: {seasons}")
        
        # Filter games
        if elite_only:
            games = self.filter_elite_competitions()
        else:
            games = self.games_df.copy()
        
        games = games[games['season'].isin(seasons)]
        
        # Filter out games without results
        games = games[
            (games['home_club_goals'].notna()) & 
            (games['away_club_goals'].notna())
        ]
        
        print(f"📊 Processing {len(games):,} games...")
        
        matches = []
        for idx, game_row in games.iterrows():
            try:
                match_data = self.parse_match_to_statsbomb_format(game_row)
                matches.append(match_data)
                
                if len(matches) % 1000 == 0:
                    print(f"   Processed {len(matches):,} matches...")
                    
            except Exception as e:
                print(f"⚠️  Error processing game {game_row['game_id']}: {e}")
                continue
        
        print(f"✅ Built dataset with {len(matches):,} matches")
        return matches
    
    def get_team_recent_form(self, team_id: int, as_of_date: str, 
                             num_games: int = 5) -> Dict:
        """Get a team's recent form before a specific date."""
        # Get all games for this team before the date
        team_games = self.club_games_df[
            (self.club_games_df['club_id'] == team_id) &
            (pd.to_datetime(self.games_df.set_index('game_id').loc[
                self.club_games_df['game_id']
            ]['date']) < pd.to_datetime(as_of_date))
        ].tail(num_games)
        
        if len(team_games) == 0:
            return {'games_played': 0, 'wins': 0, 'draws': 0, 'losses': 0}
        
        wins = team_games['is_win'].sum()
        total = len(team_games)
        draws = total - wins - (team_games['is_win'] == 0).sum()
        
        return {
            'games_played': total,
            'wins': int(wins),
            'draws': int(draws),
            'losses': int(total - wins - draws),
            'goals_scored': int(team_games['own_goals'].sum()),
            'goals_conceded': int(team_games['opponent_goals'].sum()),
            'win_rate': wins / total if total > 0 else 0
        }
    
    def get_h2h_history(self, team1_id: int, team2_id: int, 
                       as_of_date: str = None, num_games: int = 10) -> Dict:
        """Get head-to-head history between two teams."""
        # Find all games between these teams
        h2h_games = self.games_df[
            ((self.games_df['home_club_id'] == team1_id) & 
             (self.games_df['away_club_id'] == team2_id)) |
            ((self.games_df['home_club_id'] == team2_id) & 
             (self.games_df['away_club_id'] == team1_id))
        ]
        
        if as_of_date:
            h2h_games = h2h_games[
                pd.to_datetime(h2h_games['date']) < pd.to_datetime(as_of_date)
            ]
        
        h2h_games = h2h_games.tail(num_games)
        
        if len(h2h_games) == 0:
            return {'games_played': 0, 'team1_wins': 0, 'team2_wins': 0, 'draws': 0}
        
        team1_wins = 0
        team2_wins = 0
        draws = 0
        
        for _, game in h2h_games.iterrows():
            if game['home_club_id'] == team1_id:
                if game['home_club_goals'] > game['away_club_goals']:
                    team1_wins += 1
                elif game['home_club_goals'] < game['away_club_goals']:
                    team2_wins += 1
                else:
                    draws += 1
            else:
                if game['away_club_goals'] > game['home_club_goals']:
                    team1_wins += 1
                elif game['away_club_goals'] < game['home_club_goals']:
                    team2_wins += 1
                else:
                    draws += 1
        
        return {
            'games_played': len(h2h_games),
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'draws': draws,
            'team1_win_rate': team1_wins / len(h2h_games) if len(h2h_games) > 0 else 0
        }


if __name__ == "__main__":
    # Test the parser
    parser = TransfermarktParser()
    parser.load_all_data()
    
    # Build a small test dataset
    print("\n" + "="*60)
    print("Testing dataset building...")
    matches = parser.build_training_dataset(seasons=[2024, 2025], elite_only=True)
    
    if len(matches) > 0:
        print(f"\n✅ Successfully parsed {len(matches)} matches")
        print("\nSample match:")
        sample = matches[0]
        print(f"  Match ID: {sample['match_id']}")
        print(f"  Date: {sample['match_date']}")
        print(f"  Competition: {sample['competition']['competition_name']}")
        print(f"  {sample['home_team']['home_team_name']} vs {sample['away_team']['away_team_name']}")
        print(f"  Score: {sample['home_score']} - {sample['away_score']}")
        print(f"  Home manager: {sample['home_team']['managers'][0]['name']}")
        print(f"  Away manager: {sample['away_team']['managers'][0]['name']}")
        print(f"  Home lineup size: {len(sample['home_team']['lineup'])}")
        print(f"  Away lineup size: {len(sample['away_team']['lineup'])}")
