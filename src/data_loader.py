"""
Data loader for StatsBomb event data
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm


class DataLoader:
    """Load and parse StatsBomb JSON data"""
    
    def __init__(self, base_path: str = "open-data-master/data"):
        self.base_path = Path(base_path)
        self.events_path = self.base_path / "events"
        self.matches_path = self.base_path / "matches"
        self.lineups_path = self.base_path / "lineups"
        self.competitions_path = self.base_path / "competitions.json"
        
    def load_competitions(self) -> pd.DataFrame:
        """Load competitions data"""
        with open(self.competitions_path, 'r') as f:
            competitions = json.load(f)
        return pd.DataFrame(competitions)
    
    def load_matches(self, competition_id: int = None, season_id: int = None) -> pd.DataFrame:
        """Load match data for specific competition/season or all matches"""
        all_matches = []
        
        if competition_id and season_id:
            match_file = self.matches_path / str(competition_id) / f"{season_id}.json"
            if match_file.exists():
                with open(match_file, 'r') as f:
                    matches = json.load(f)
                all_matches.extend(matches)
        else:
            # Load all matches
            for comp_dir in self.matches_path.iterdir():
                if comp_dir.is_dir():
                    for match_file in comp_dir.glob("*.json"):
                        with open(match_file, 'r') as f:
                            matches = json.load(f)
                        all_matches.extend(matches)
        
        return pd.DataFrame(all_matches)
    
    def load_match_events(self, match_id: int) -> List[Dict]:
        """Load events for a specific match"""
        event_file = self.events_path / f"{match_id}.json"
        
        if not event_file.exists():
            return []
        
        with open(event_file, 'r') as f:
            events = json.load(f)
        
        return events
    
    def load_all_events(self, limit: int = None) -> pd.DataFrame:
        """Load all event files (can be memory intensive)"""
        all_events = []
        event_files = list(self.events_path.glob("*.json"))
        
        if limit:
            event_files = event_files[:limit]
        
        for event_file in tqdm(event_files, desc="Loading events"):
            match_id = event_file.stem
            with open(event_file, 'r') as f:
                events = json.load(f)
            
            for event in events:
                event['match_id'] = match_id
                all_events.append(event)
        
        return pd.DataFrame(all_events)
    
    def get_available_match_ids(self) -> List[str]:
        """Get list of all available match IDs with event data"""
        return [f.stem for f in self.events_path.glob("*.json")]
