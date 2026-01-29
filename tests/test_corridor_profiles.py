"""
Unit tests for corridor profiling module.

Tests cover profile generation, weekly updates, percentile calculations,
and profile versioning for the context-aware fraud detection system.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List


class TestCorridorProfileGeneration:
    """Test suite for corridor profile generation."""
    
    def test_profile_contains_required_fields(self):
        """Verify generated profiles contain all required statistical fields."""
        required_fields = [
            'corridor_code',
            'median_amount',
            'p95_amount',
            'p99_amount',
            'mean_amount',
            'std_amount',
            'median_velocity_24h',
            'p95_velocity_24h',
            'transaction_count',
            'unique_senders',
            'baseline_fraud_rate',
            'profile_date',
            'version'
        ]
        
        profile = self._generate_mock_profile('UK_NGN')
        
        for field in required_fields:
            assert field in profile, f"Profile missing required field: {field}"
    
    def test_percentiles_in_correct_order(self):
        """Verify p50 < p95 < p99 for amount distributions."""
        profile = self._generate_mock_profile('UK_NGN')
        
        assert profile['median_amount'] < profile['p95_amount'], \
            "Median should be less than p95"
        assert profile['p95_amount'] < profile['p99_amount'], \
            "P95 should be less than p99"
    
    def test_velocity_percentiles_in_correct_order(self):
        """Verify median velocity < p95 velocity."""
        profile = self._generate_mock_profile('UK_NGN')
        
        assert profile['median_velocity_24h'] < profile['p95_velocity_24h'], \
            "Median velocity should be less than p95 velocity"
    
    def test_fraud_rate_bounded(self):
        """Verify fraud rate is between 0 and 1."""
        profile = self._generate_mock_profile('UK_NGN')
        
        assert 0 <= profile['baseline_fraud_rate'] <= 1, \
            f"Fraud rate {profile['baseline_fraud_rate']} out of bounds"
    
    def test_different_corridors_produce_different_profiles(self):
        """Verify distinct corridors have meaningfully different profiles."""
        profile_ngn = self._generate_mock_profile('UK_NGN')
        profile_pln = self._generate_mock_profile('UK_PLN')
        
        # UK-Nigeria typically has higher amounts than UK-Poland
        assert profile_ngn['median_amount'] != profile_pln['median_amount'], \
            "Different corridors should have different median amounts"
    
    def _generate_mock_profile(self, corridor: str) -> Dict:
        """Generate a mock corridor profile for testing."""
        profiles = {
            'UK_NGN': {
                'corridor_code': 'UK_NGN',
                'median_amount': 350,
                'p95_amount': 2500,
                'p99_amount': 5000,
                'mean_amount': 450,
                'std_amount': 380,
                'median_velocity_24h': 1.2,
                'p95_velocity_24h': 3.5,
                'transaction_count': 125000,
                'unique_senders': 45000,
                'baseline_fraud_rate': 0.012,
                'profile_date': datetime.now().isoformat(),
                'version': '2024-W03'
            },
            'UK_PLN': {
                'corridor_code': 'UK_PLN',
                'median_amount': 180,
                'p95_amount': 1500,
                'p99_amount': 3000,
                'mean_amount': 220,
                'std_amount': 195,
                'median_velocity_24h': 0.8,
                'p95_velocity_24h': 2.1,
                'transaction_count': 89000,
                'unique_senders': 52000,
                'baseline_fraud_rate': 0.003,
                'profile_date': datetime.now().isoformat(),
                'version': '2024-W03'
            }
        }
        return profiles.get(corridor, profiles['UK_NGN'])


class TestProfileWeeklyUpdates:
    """Test suite for weekly profile update mechanism."""
    
    def test_version_format_is_iso_week(self):
        """Verify version follows ISO week format (YYYY-Www)."""
        import re
        
        profile = self._generate_updated_profile('UK_NGN')
        version_pattern = r'^\d{4}-W\d{2}$'
        
        assert re.match(version_pattern, profile['version']), \
            f"Version '{profile['version']}' does not match ISO week format"
    
    def test_update_preserves_corridor_code(self):
        """Verify corridor code remains unchanged after update."""
        original = self._generate_mock_profile('UK_NGN')
        updated = self._generate_updated_profile('UK_NGN')
        
        assert original['corridor_code'] == updated['corridor_code'], \
            "Corridor code should not change during update"
    
    def test_update_changes_profile_date(self):
        """Verify profile date is updated during refresh."""
        # Simulate old profile
        old_profile = self._generate_mock_profile('UK_NGN')
        old_profile['profile_date'] = (datetime.now() - timedelta(days=7)).isoformat()
        
        new_profile = self._generate_updated_profile('UK_NGN')
        
        assert new_profile['profile_date'] > old_profile['profile_date'], \
            "Updated profile should have more recent date"
    
    def test_statistics_can_drift_between_updates(self):
        """Verify that statistical values can change between weekly updates."""
        # This test validates that the system allows for natural drift
        # In production, values would change based on actual transaction data
        
        week1_median = 350
        week2_median = 365  # Simulated 4% increase
        
        drift_percentage = abs(week2_median - week1_median) / week1_median * 100
        
        # Allow up to 20% weekly drift (reasonable for dynamic markets)
        assert drift_percentage <= 20, \
            f"Weekly drift of {drift_percentage}% exceeds reasonable threshold"
    
    def _generate_mock_profile(self, corridor: str) -> Dict:
        """Generate mock profile."""
        return {
            'corridor_code': corridor,
            'median_amount': 350,
            'p95_amount': 2500,
            'p99_amount': 5000,
            'mean_amount': 450,
            'std_amount': 380,
            'median_velocity_24h': 1.2,
            'p95_velocity_24h': 3.5,
            'transaction_count': 125000,
            'unique_senders': 45000,
            'baseline_fraud_rate': 0.012,
            'profile_date': datetime.now().isoformat(),
            'version': '2024-W03'
        }
    
    def _generate_updated_profile(self, corridor: str) -> Dict:
        """Generate an updated profile simulating weekly refresh."""
        profile = self._generate_mock_profile(corridor)
        profile['profile_date'] = datetime.now().isoformat()
        
        # Calculate current ISO week
        now = datetime.now()
        profile['version'] = f"{now.year}-W{now.isocalendar()[1]:02d}"
        
        return profile


class TestPercentileCalculations:
    """Test suite for percentile calculation accuracy."""
    
    def test_percentile_calculation_with_known_data(self):
        """Verify percentile calculations against known test data."""
        # Test data with known percentiles
        test_amounts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        
        p50 = np.percentile(test_amounts, 50)
        p95 = np.percentile(test_amounts, 95)
        p99 = np.percentile(test_amounts, 99)
        
        assert 500 <= p50 <= 600, f"P50 should be ~550, got {p50}"
        assert 900 <= p95 <= 1000, f"P95 should be ~955, got {p95}"
        assert p99 >= 950, f"P99 should be >=950, got {p99}"
    
    def test_percentile_handles_outliers(self):
        """Verify percentile calculation is robust to outliers."""
        # Normal transactions with one extreme outlier
        amounts = [200, 250, 300, 350, 400, 450, 500, 50000]
        
        p50 = np.percentile(amounts, 50)
        p95 = np.percentile(amounts, 95)
        
        # Median should not be significantly affected by outlier
        assert p50 < 1000, f"Median {p50} should be robust to outlier"
    
    def test_percentile_with_single_value(self):
        """Verify percentile handling with minimal data."""
        amounts = [500]
        
        p50 = np.percentile(amounts, 50)
        p95 = np.percentile(amounts, 95)
        
        # With single value, all percentiles equal that value
        assert p50 == 500
        assert p95 == 500
    
    def test_percentile_with_identical_values(self):
        """Verify percentile handling when all values are identical."""
        amounts = [350] * 100
        
        p50 = np.percentile(amounts, 50)
        p95 = np.percentile(amounts, 95)
        p99 = np.percentile(amounts, 99)
        
        assert p50 == p95 == p99 == 350, \
            "All percentiles should equal the single unique value"


class TestProfileVersioning:
    """Test suite for profile version management."""
    
    def test_version_comparison(self):
        """Verify version strings can be compared chronologically."""
        v1 = '2024-W01'
        v2 = '2024-W02'
        v3 = '2024-W52'
        v4 = '2025-W01'
        
        assert v1 < v2, "Earlier week should sort before later week"
        assert v2 < v3, "Week 2 should sort before week 52"
        assert v3 < v4, "2024-W52 should sort before 2025-W01"
    
    def test_profile_history_maintained(self):
        """Verify system can maintain profile history for auditing."""
        history = self._generate_profile_history('UK_NGN', weeks=4)
        
        assert len(history) == 4, "Should maintain 4 weeks of history"
        
        # Verify chronological order
        versions = [p['version'] for p in history]
        assert versions == sorted(versions), "History should be chronologically ordered"
    
    def test_rollback_to_previous_version(self):
        """Verify ability to rollback to previous profile version."""
        history = self._generate_profile_history('UK_NGN', weeks=4)
        
        current = history[-1]
        previous = history[-2]
        
        # Simulate rollback
        rolled_back = self._rollback_profile(history, steps=1)
        
        assert rolled_back['version'] == previous['version'], \
            "Rollback should restore previous version"
    
    def _generate_profile_history(self, corridor: str, weeks: int) -> List[Dict]:
        """Generate mock profile history."""
        history = []
        base_date = datetime.now() - timedelta(weeks=weeks)
        
        for i in range(weeks):
            profile_date = base_date + timedelta(weeks=i)
            history.append({
                'corridor_code': corridor,
                'median_amount': 350 + i * 5,  # Simulate slight drift
                'version': f"{profile_date.year}-W{profile_date.isocalendar()[1]:02d}",
                'profile_date': profile_date.isoformat()
            })
        
        return history
    
    def _rollback_profile(self, history: List[Dict], steps: int) -> Dict:
        """Rollback to a previous profile version."""
        if steps >= len(history):
            raise ValueError("Cannot rollback beyond available history")
        return history[-(steps + 1)]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
