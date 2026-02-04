"""
Corridor Profiler Module

Provides production-ready corridor profiling functionality for
context-aware fraud detection in cross-border payment systems.

Components:
- CorridorProfiler: Main class for profile generation and management
- ProfileStore: Versioned storage for corridor profiles
- ProfileValidator: Validation utilities for profile integrity
"""

from .corridor_profiler import CorridorProfiler, CorridorProfile
from .profile_store import ProfileStore
from .validators import ProfileValidator

__all__ = [
    'CorridorProfiler',
    'CorridorProfile', 
    'ProfileStore',
    'ProfileValidator'
]

__version__ = '0.1.0'
