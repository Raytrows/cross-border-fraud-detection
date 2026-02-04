"""
Corridor Profiler

Production-ready corridor profiling for context-aware fraud detection.
Maintains statistical baselines updated weekly for each payment route.

This module addresses the first component of the dynamic signal weighting
methodology: establishing corridor-specific norms against which transaction
behaviour is evaluated.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorridorProfile:
    """
    Statistical profile for a payment corridor.
    
    Captures the distribution characteristics of legitimate transactions
    in a specific corridor, enabling context-aware risk assessment.
    """
    corridor_code: str
    
    # Amount distribution
    median_amount: float
    mean_amount: float
    std_amount: float
    p25_amount: float
    p75_amount: float
    p95_amount: float
    p99_amount: float
    min_amount: float
    max_amount: float
    
    # Velocity distribution (transactions per sender per 24h)
    median_velocity_24h: float
    mean_velocity_24h: float
    p95_velocity_24h: float
    
    # Temporal patterns
    peak_hours: List[int] = field(default_factory=list)
    peak_days: List[int] = field(default_factory=list)  # 0=Monday, 6=Sunday
    
    # Population statistics
    transaction_count: int = 0
    unique_senders: int = 0
    unique_beneficiaries: int = 0
    
    # Risk baseline
    baseline_fraud_rate: float = 0.0
    
    # Metadata
    profile_date: str = ""
    version: str = ""
    data_window_days: int = 28
    
    def to_dict(self) -> Dict:
        """Convert profile to dictionary for serialisation."""
        return {
            'corridor_code': self.corridor_code,
            'amount_distribution': {
                'median': self.median_amount,
                'mean': self.mean_amount,
                'std': self.std_amount,
                'p25': self.p25_amount,
                'p75': self.p75_amount,
                'p95': self.p95_amount,
                'p99': self.p99_amount,
                'min': self.min_amount,
                'max': self.max_amount
            },
            'velocity_distribution': {
                'median_24h': self.median_velocity_24h,
                'mean_24h': self.mean_velocity_24h,
                'p95_24h': self.p95_velocity_24h
            },
            'temporal_patterns': {
                'peak_hours': self.peak_hours,
                'peak_days': self.peak_days
            },
            'population': {
                'transaction_count': self.transaction_count,
                'unique_senders': self.unique_senders,
                'unique_beneficiaries': self.unique_beneficiaries
            },
            'risk': {
                'baseline_fraud_rate': self.baseline_fraud_rate
            },
            'metadata': {
                'profile_date': self.profile_date,
                'version': self.version,
                'data_window_days': self.data_window_days
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CorridorProfile':
        """Create profile from dictionary."""
        return cls(
            corridor_code=data['corridor_code'],
            median_amount=data['amount_distribution']['median'],
            mean_amount=data['amount_distribution']['mean'],
            std_amount=data['amount_distribution']['std'],
            p25_amount=data['amount_distribution']['p25'],
            p75_amount=data['amount_distribution']['p75'],
            p95_amount=data['amount_distribution']['p95'],
            p99_amount=data['amount_distribution']['p99'],
            min_amount=data['amount_distribution']['min'],
            max_amount=data['amount_distribution']['max'],
            median_velocity_24h=data['velocity_distribution']['median_24h'],
            mean_velocity_24h=data['velocity_distribution']['mean_24h'],
            p95_velocity_24h=data['velocity_distribution']['p95_24h'],
            peak_hours=data['temporal_patterns']['peak_hours'],
            peak_days=data['temporal_patterns']['peak_days'],
            transaction_count=data['population']['transaction_count'],
            unique_senders=data['population']['unique_senders'],
            unique_beneficiaries=data['population']['unique_beneficiaries'],
            baseline_fraud_rate=data['risk']['baseline_fraud_rate'],
            profile_date=data['metadata']['profile_date'],
            version=data['metadata']['version'],
            data_window_days=data['metadata']['data_window_days']
        )


class CorridorProfiler:
    """
    Generates and manages corridor profiles for context-aware fraud detection.
    
    The profiler analyses historical transaction data to establish
    corridor-specific baselines that inform risk scoring decisions.
    
    Usage:
    ------
    profiler = CorridorProfiler()
    profile = profiler.generate_profile(transactions_df, 'UK_NGN')
    
    # Weekly update
    updated_profile = profiler.update_profile(new_transactions_df, existing_profile)
    """
    
    # Configuration
    DEFAULT_WINDOW_DAYS = 28  # 4 weeks of data
    MIN_TRANSACTIONS = 1000   # Minimum transactions for reliable profile
    OUTLIER_PERCENTILE = 99.5  # Percentile for outlier detection
    
    def __init__(self, window_days: int = None):
        """
        Initialise the corridor profiler.
        
        Parameters:
        -----------
        window_days : int, optional
            Number of days of historical data to use for profiling.
            Defaults to 28 days (4 weeks).
        """
        self.window_days = window_days or self.DEFAULT_WINDOW_DAYS
        
    def generate_profile(self, 
                         transactions: pd.DataFrame,
                         corridor_code: str,
                         fraud_labels: Optional[pd.Series] = None) -> CorridorProfile:
        """
        Generate a corridor profile from transaction data.
        
        Parameters:
        -----------
        transactions : pd.DataFrame
            Transaction data with columns: amount, sender_id, beneficiary_id,
            timestamp. Must be filtered to the target corridor.
        corridor_code : str
            Identifier for the corridor (e.g., 'UK_NGN', 'UK_PLN')
        fraud_labels : pd.Series, optional
            Boolean series indicating fraud (True) or legitimate (False).
            Used to calculate baseline fraud rate.
            
        Returns:
        --------
        CorridorProfile with calculated statistics
        """
        if len(transactions) < self.MIN_TRANSACTIONS:
            logger.warning(
                f"Corridor {corridor_code} has only {len(transactions)} transactions. "
                f"Minimum recommended is {self.MIN_TRANSACTIONS}."
            )
        
        # Filter outliers for robust statistics
        amounts = self._filter_outliers(transactions['amount'])
        
        # Calculate amount distribution
        amount_stats = self._calculate_amount_statistics(amounts)
        
        # Calculate velocity distribution
        velocity_stats = self._calculate_velocity_statistics(transactions)
        
        # Calculate temporal patterns
        temporal_patterns = self._calculate_temporal_patterns(transactions)
        
        # Calculate population statistics
        population_stats = self._calculate_population_statistics(transactions)
        
        # Calculate fraud baseline if labels provided
        if fraud_labels is not None:
            baseline_fraud_rate = fraud_labels.mean()
        else:
            baseline_fraud_rate = 0.0
        
        # Generate version string (ISO week format)
        now = datetime.now()
        version = f"{now.year}-W{now.isocalendar()[1]:02d}"
        
        return CorridorProfile(
            corridor_code=corridor_code,
            **amount_stats,
            **velocity_stats,
            **temporal_patterns,
            **population_stats,
            baseline_fraud_rate=baseline_fraud_rate,
            profile_date=now.isoformat(),
            version=version,
            data_window_days=self.window_days
        )
    
    def update_profile(self,
                       new_transactions: pd.DataFrame,
                       existing_profile: CorridorProfile,
                       fraud_labels: Optional[pd.Series] = None,
                       blend_factor: float = 0.3) -> CorridorProfile:
        """
        Update an existing profile with new transaction data.
        
        Uses exponential smoothing to blend new statistics with existing
        profile, providing stability while adapting to changing patterns.
        
        Parameters:
        -----------
        new_transactions : pd.DataFrame
            New transaction data since last profile update
        existing_profile : CorridorProfile
            Current corridor profile to update
        fraud_labels : pd.Series, optional
            Fraud labels for new transactions
        blend_factor : float
            Weight for new data (0-1). Higher values adapt faster.
            Default 0.3 means 30% new data, 70% existing profile.
            
        Returns:
        --------
        Updated CorridorProfile
        """
        # Generate fresh profile from new data
        new_profile = self.generate_profile(
            new_transactions, 
            existing_profile.corridor_code,
            fraud_labels
        )
        
        # Blend statistics using exponential smoothing
        blended = CorridorProfile(
            corridor_code=existing_profile.corridor_code,
            
            # Amount distribution - blended
            median_amount=self._blend(existing_profile.median_amount, 
                                       new_profile.median_amount, blend_factor),
            mean_amount=self._blend(existing_profile.mean_amount,
                                     new_profile.mean_amount, blend_factor),
            std_amount=self._blend(existing_profile.std_amount,
                                    new_profile.std_amount, blend_factor),
            p25_amount=self._blend(existing_profile.p25_amount,
                                    new_profile.p25_amount, blend_factor),
            p75_amount=self._blend(existing_profile.p75_amount,
                                    new_profile.p75_amount, blend_factor),
            p95_amount=self._blend(existing_profile.p95_amount,
                                    new_profile.p95_amount, blend_factor),
            p99_amount=self._blend(existing_profile.p99_amount,
                                    new_profile.p99_amount, blend_factor),
            min_amount=min(existing_profile.min_amount, new_profile.min_amount),
            max_amount=max(existing_profile.max_amount, new_profile.max_amount),
            
            # Velocity distribution - blended
            median_velocity_24h=self._blend(existing_profile.median_velocity_24h,
                                             new_profile.median_velocity_24h, blend_factor),
            mean_velocity_24h=self._blend(existing_profile.mean_velocity_24h,
                                           new_profile.mean_velocity_24h, blend_factor),
            p95_velocity_24h=self._blend(existing_profile.p95_velocity_24h,
                                          new_profile.p95_velocity_24h, blend_factor),
            
            # Temporal patterns - use new data
            peak_hours=new_profile.peak_hours,
            peak_days=new_profile.peak_days,
            
            # Population - cumulative
            transaction_count=existing_profile.transaction_count + new_profile.transaction_count,
            unique_senders=existing_profile.unique_senders,  # Would need dedup in production
            unique_beneficiaries=existing_profile.unique_beneficiaries,
            
            # Fraud rate - blended
            baseline_fraud_rate=self._blend(existing_profile.baseline_fraud_rate,
                                             new_profile.baseline_fraud_rate, blend_factor),
            
            # Metadata - updated
            profile_date=new_profile.profile_date,
            version=new_profile.version,
            data_window_days=self.window_days
        )
        
        return blended
    
    def _filter_outliers(self, values: pd.Series) -> pd.Series:
        """Remove extreme outliers that could skew statistics."""
        upper_bound = values.quantile(self.OUTLIER_PERCENTILE / 100)
        return values[values <= upper_bound]
    
    def _calculate_amount_statistics(self, amounts: pd.Series) -> Dict:
        """Calculate amount distribution statistics."""
        return {
            'median_amount': float(amounts.median()),
            'mean_amount': float(amounts.mean()),
            'std_amount': float(amounts.std()),
            'p25_amount': float(amounts.quantile(0.25)),
            'p75_amount': float(amounts.quantile(0.75)),
            'p95_amount': float(amounts.quantile(0.95)),
            'p99_amount': float(amounts.quantile(0.99)),
            'min_amount': float(amounts.min()),
            'max_amount': float(amounts.max())
        }
    
    def _calculate_velocity_statistics(self, transactions: pd.DataFrame) -> Dict:
        """Calculate velocity distribution (transactions per sender per 24h)."""
        # Group by sender and date to get daily transaction counts
        transactions = transactions.copy()
        transactions['date'] = pd.to_datetime(transactions['timestamp']).dt.date
        
        daily_counts = transactions.groupby(['sender_id', 'date']).size()
        
        return {
            'median_velocity_24h': float(daily_counts.median()),
            'mean_velocity_24h': float(daily_counts.mean()),
            'p95_velocity_24h': float(daily_counts.quantile(0.95))
        }
    
    def _calculate_temporal_patterns(self, transactions: pd.DataFrame) -> Dict:
        """Identify peak hours and days for transaction activity."""
        transactions = transactions.copy()
        transactions['hour'] = pd.to_datetime(transactions['timestamp']).dt.hour
        transactions['day'] = pd.to_datetime(transactions['timestamp']).dt.dayofweek
        
        # Find hours with above-average activity
        hourly_counts = transactions['hour'].value_counts()
        mean_hourly = hourly_counts.mean()
        peak_hours = hourly_counts[hourly_counts > mean_hourly].index.tolist()
        
        # Find days with above-average activity
        daily_counts = transactions['day'].value_counts()
        mean_daily = daily_counts.mean()
        peak_days = daily_counts[daily_counts > mean_daily].index.tolist()
        
        return {
            'peak_hours': sorted(peak_hours),
            'peak_days': sorted(peak_days)
        }
    
    def _calculate_population_statistics(self, transactions: pd.DataFrame) -> Dict:
        """Calculate population-level statistics."""
        return {
            'transaction_count': len(transactions),
            'unique_senders': transactions['sender_id'].nunique(),
            'unique_beneficiaries': transactions['beneficiary_id'].nunique()
        }
    
    @staticmethod
    def _blend(old_value: float, new_value: float, blend_factor: float) -> float:
        """Exponential smoothing blend of old and new values."""
        return (1 - blend_factor) * old_value + blend_factor * new_value


def calculate_normalised_score(value: float, 
                                profile: CorridorProfile,
                                feature: str = 'amount') -> float:
    """
    Calculate normalised feature score relative to corridor profile.
    
    Returns a score from 0 to 1 indicating how anomalous the value is
    within the corridor's normal distribution.
    
    Parameters:
    -----------
    value : float
        The observed value (amount, velocity, etc.)
    profile : CorridorProfile
        The corridor profile to compare against
    feature : str
        The feature type: 'amount' or 'velocity'
        
    Returns:
    --------
    float between 0 and 1, where higher indicates more anomalous
    """
    if feature == 'amount':
        median = profile.median_amount
        p95 = profile.p95_amount
        p99 = profile.p99_amount
    elif feature == 'velocity':
        median = profile.median_velocity_24h
        p95 = profile.p95_velocity_24h
        p99 = p95 * 1.5  # Approximate p99 from p95
    else:
        raise ValueError(f"Unknown feature: {feature}")
    
    if value <= median:
        return 0.0
    elif value <= p95:
        # Linear interpolation from 0 to 0.5
        return 0.5 * (value - median) / (p95 - median)
    elif value <= p99:
        # Linear interpolation from 0.5 to 0.9
        return 0.5 + 0.4 * (value - p95) / (p99 - p95)
    else:
        # Asymptotic approach to 1.0 for extreme values
        excess_ratio = (value - p99) / p99
        return min(0.9 + 0.1 * (1 - np.exp(-excess_ratio)), 1.0)
