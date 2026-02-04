"""
Profile Store

Versioned storage for corridor profiles with support for
history tracking and rollback capabilities.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging

from .corridor_profiler import CorridorProfile

logger = logging.getLogger(__name__)


class ProfileStore:
    """
    Manages storage and retrieval of corridor profiles with versioning.
    
    Supports:
    - Saving profiles with automatic versioning
    - Retrieving current or historical profiles
    - Rolling back to previous versions
    - Audit trail for profile changes
    
    Usage:
    ------
    store = ProfileStore('/path/to/profiles')
    store.save_profile(profile)
    
    current = store.get_current_profile('UK_NGN')
    history = store.get_profile_history('UK_NGN', n_versions=5)
    """
    
    CURRENT_FILENAME = 'current.json'
    HISTORY_DIR = 'history'
    
    def __init__(self, base_path: str):
        """
        Initialise the profile store.
        
        Parameters:
        -----------
        base_path : str
            Base directory for profile storage
        """
        self.base_path = Path(base_path)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directory structure."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / self.HISTORY_DIR).mkdir(exist_ok=True)
    
    def _corridor_path(self, corridor_code: str) -> Path:
        """Get path for corridor-specific storage."""
        path = self.base_path / corridor_code
        path.mkdir(exist_ok=True)
        (path / self.HISTORY_DIR).mkdir(exist_ok=True)
        return path
    
    def save_profile(self, profile: CorridorProfile) -> str:
        """
        Save a corridor profile with automatic versioning.
        
        Parameters:
        -----------
        profile : CorridorProfile
            The profile to save
            
        Returns:
        --------
        str: The version string of the saved profile
        """
        corridor_path = self._corridor_path(profile.corridor_code)
        
        # Archive current profile to history if it exists
        current_file = corridor_path / self.CURRENT_FILENAME
        if current_file.exists():
            self._archive_current(profile.corridor_code)
        
        # Save new profile as current
        profile_dict = profile.to_dict()
        with open(current_file, 'w') as f:
            json.dump(profile_dict, f, indent=2)
        
        logger.info(f"Saved profile for {profile.corridor_code} version {profile.version}")
        
        return profile.version
    
    def _archive_current(self, corridor_code: str):
        """Move current profile to history."""
        corridor_path = self._corridor_path(corridor_code)
        current_file = corridor_path / self.CURRENT_FILENAME
        
        if not current_file.exists():
            return
        
        # Load current profile to get its version
        with open(current_file, 'r') as f:
            current_data = json.load(f)
        
        version = current_data['metadata']['version']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save to history with timestamp
        history_file = corridor_path / self.HISTORY_DIR / f"{version}_{timestamp}.json"
        with open(history_file, 'w') as f:
            json.dump(current_data, f, indent=2)
        
        logger.debug(f"Archived {corridor_code} version {version} to history")
    
    def get_current_profile(self, corridor_code: str) -> Optional[CorridorProfile]:
        """
        Retrieve the current profile for a corridor.
        
        Parameters:
        -----------
        corridor_code : str
            The corridor identifier
            
        Returns:
        --------
        CorridorProfile or None if not found
        """
        corridor_path = self._corridor_path(corridor_code)
        current_file = corridor_path / self.CURRENT_FILENAME
        
        if not current_file.exists():
            logger.warning(f"No profile found for corridor {corridor_code}")
            return None
        
        with open(current_file, 'r') as f:
            data = json.load(f)
        
        return CorridorProfile.from_dict(data)
    
    def get_profile_history(self, 
                            corridor_code: str, 
                            n_versions: int = 10) -> List[CorridorProfile]:
        """
        Retrieve historical profiles for a corridor.
        
        Parameters:
        -----------
        corridor_code : str
            The corridor identifier
        n_versions : int
            Maximum number of historical versions to retrieve
            
        Returns:
        --------
        List of CorridorProfile, most recent first
        """
        corridor_path = self._corridor_path(corridor_code)
        history_path = corridor_path / self.HISTORY_DIR
        
        if not history_path.exists():
            return []
        
        # Get all history files, sorted by timestamp (newest first)
        history_files = sorted(
            history_path.glob('*.json'),
            key=lambda x: x.stem,
            reverse=True
        )[:n_versions]
        
        profiles = []
        for file_path in history_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            profiles.append(CorridorProfile.from_dict(data))
        
        return profiles
    
    def rollback(self, corridor_code: str, steps: int = 1) -> Optional[CorridorProfile]:
        """
        Rollback to a previous profile version.
        
        Parameters:
        -----------
        corridor_code : str
            The corridor identifier
        steps : int
            Number of versions to roll back
            
        Returns:
        --------
        The restored CorridorProfile, or None if rollback not possible
        """
        history = self.get_profile_history(corridor_code, n_versions=steps + 1)
        
        if len(history) < steps:
            logger.error(f"Cannot rollback {steps} steps: only {len(history)} versions available")
            return None
        
        # Get the target profile from history
        target_profile = history[steps - 1]
        
        # Save it as current (this will archive the current one)
        self.save_profile(target_profile)
        
        logger.info(f"Rolled back {corridor_code} to version {target_profile.version}")
        
        return target_profile
    
    def list_corridors(self) -> List[str]:
        """List all corridors with stored profiles."""
        corridors = []
        for item in self.base_path.iterdir():
            if item.is_dir() and item.name != self.HISTORY_DIR:
                current_file = item / self.CURRENT_FILENAME
                if current_file.exists():
                    corridors.append(item.name)
        return sorted(corridors)
    
    def get_profile_metadata(self, corridor_code: str) -> Optional[Dict]:
        """
        Get metadata for a corridor's current profile without loading full profile.
        
        Returns:
        --------
        Dict with version, profile_date, transaction_count, or None
        """
        profile = self.get_current_profile(corridor_code)
        if profile is None:
            return None
        
        return {
            'corridor_code': profile.corridor_code,
            'version': profile.version,
            'profile_date': profile.profile_date,
            'transaction_count': profile.transaction_count,
            'baseline_fraud_rate': profile.baseline_fraud_rate
        }
