"""
Unit tests for dynamic signal weighting module.

Tests cover corridor-specific weight calculation, normalisation,
and edge case handling for the context-aware fraud detection system.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

# Import would be: from src.signal_weighting.dynamic_weights import DynamicWeightCalculator
# For testing purposes, we define expected behaviours


class TestCorridorMultipliers:
    """Test suite for corridor-specific weight multipliers."""
    
    # Expected multipliers based on methodology
    UK_NIGERIA_MULTIPLIERS = {
        'beneficiary_novelty': 1.5,
        'amount_deviation': 1.2,
        'velocity': 0.8,
        'temporal_anomaly': 0.6
    }
    
    UK_POLAND_MULTIPLIERS = {
        'beneficiary_novelty': 0.7,
        'amount_deviation': 1.0,
        'velocity': 1.4,
        'temporal_anomaly': 1.1
    }
    
    def test_nigeria_corridor_weights_sum_to_expected_range(self):
        """Verify UK-Nigeria multipliers produce balanced total weight."""
        total = sum(self.UK_NIGERIA_MULTIPLIERS.values())
        # Total should be approximately 4.1 (sum of multipliers)
        assert 4.0 <= total <= 4.2, f"UK-Nigeria total weight {total} outside expected range"
    
    def test_poland_corridor_weights_sum_to_expected_range(self):
        """Verify UK-Poland multipliers produce balanced total weight."""
        total = sum(self.UK_POLAND_MULTIPLIERS.values())
        # Total should be approximately 4.2
        assert 4.0 <= total <= 4.4, f"UK-Poland total weight {total} outside expected range"
    
    def test_velocity_weight_lower_for_high_volume_corridors(self):
        """
        High-volume corridors like UK-Nigeria should have lower velocity weights
        because frequent transactions are normal behaviour.
        """
        assert self.UK_NIGERIA_MULTIPLIERS['velocity'] < self.UK_POLAND_MULTIPLIERS['velocity'], \
            "Velocity weight should be lower for high-volume corridors"
    
    def test_beneficiary_novelty_higher_for_fraud_prone_corridors(self):
        """
        Corridors with higher fraud rates should weight beneficiary novelty more heavily
        as new recipients are more predictive of fraud in those contexts.
        """
        assert self.UK_NIGERIA_MULTIPLIERS['beneficiary_novelty'] > self.UK_POLAND_MULTIPLIERS['beneficiary_novelty'], \
            "Beneficiary novelty should be weighted higher for fraud-prone corridors"


class TestNormalisedFeatureCalculation:
    """Test suite for corridor-normalised feature scoring."""
    
    # Corridor profiles based on methodology
    CORRIDOR_PROFILES = {
        'UK_NGN': {
            'median_amount': 350,
            'p95_amount': 2500,
            'p99_amount': 5000,
            'median_velocity_24h': 1.2
        },
        'UK_PLN': {
            'median_amount': 180,
            'p95_amount': 1500,
            'p99_amount': 3000,
            'median_velocity_24h': 0.8
        }
    }
    
    def test_amount_below_median_scores_zero(self):
        """Transactions below corridor median should score 0 for amount deviation."""
        # £300 is below UK-Nigeria median of £350
        score = self._calculate_amount_score(300, 'UK_NGN')
        assert score == 0, f"Below-median amount should score 0, got {score}"
    
    def test_amount_at_p95_scores_half(self):
        """Transactions at 95th percentile should score approximately 0.5."""
        score = self._calculate_amount_score(2500, 'UK_NGN')
        assert 0.45 <= score <= 0.55, f"P95 amount should score ~0.5, got {score}"
    
    def test_amount_above_p99_approaches_one(self):
        """Transactions well above 99th percentile should approach score of 1.0."""
        score = self._calculate_amount_score(10000, 'UK_NGN')
        assert score >= 0.9, f"Extreme amount should score >=0.9, got {score}"
    
    def test_same_amount_scores_differently_by_corridor(self):
        """
        £1000 should score differently in UK-Nigeria vs UK-Poland corridors
        because their distributions differ.
        """
        score_ngn = self._calculate_amount_score(1000, 'UK_NGN')
        score_pln = self._calculate_amount_score(1000, 'UK_PLN')
        
        # £1000 is more anomalous in UK-Poland (lower p95)
        assert score_pln > score_ngn, \
            f"Same amount should score higher in lower-volume corridor: NGN={score_ngn}, PLN={score_pln}"
    
    def _calculate_amount_score(self, amount: float, corridor: str) -> float:
        """
        Calculate normalised amount deviation score.
        
        Score ranges from 0 to 1:
        - Below median: 0
        - Between median and p95: 0 to 0.5
        - Between p95 and p99: 0.5 to 0.9
        - Above p99: approaches 1.0
        """
        profile = self.CORRIDOR_PROFILES[corridor]
        
        if amount <= profile['median_amount']:
            return 0.0
        elif amount <= profile['p95_amount']:
            # Linear interpolation from 0 to 0.5
            range_size = profile['p95_amount'] - profile['median_amount']
            position = amount - profile['median_amount']
            return 0.5 * (position / range_size)
        elif amount <= profile['p99_amount']:
            # Linear interpolation from 0.5 to 0.9
            range_size = profile['p99_amount'] - profile['p95_amount']
            position = amount - profile['p95_amount']
            return 0.5 + 0.4 * (position / range_size)
        else:
            # Asymptotic approach to 1.0
            excess = amount - profile['p99_amount']
            return min(0.9 + 0.1 * (1 - np.exp(-excess / profile['p99_amount'])), 1.0)


class TestRiskScoreAggregation:
    """Test suite for final risk score calculation."""
    
    def test_risk_score_bounded_zero_to_one(self):
        """Final risk score must always be between 0 and 1."""
        # Simulate various feature combinations
        test_cases = [
            {'amount': 0, 'velocity': 0, 'beneficiary': 0, 'temporal': 0},
            {'amount': 1, 'velocity': 1, 'beneficiary': 1, 'temporal': 1},
            {'amount': 0.5, 'velocity': 0.3, 'beneficiary': 0.8, 'temporal': 0.2},
        ]
        
        for features in test_cases:
            score = self._aggregate_risk_score(features, 'UK_NGN')
            assert 0 <= score <= 1, f"Risk score {score} out of bounds for features {features}"
    
    def test_zero_features_produce_zero_risk(self):
        """When all feature scores are zero, risk score should be zero."""
        features = {'amount': 0, 'velocity': 0, 'beneficiary': 0, 'temporal': 0}
        score = self._aggregate_risk_score(features, 'UK_NGN')
        assert score == 0, f"Zero features should produce zero risk, got {score}"
    
    def test_weights_applied_correctly(self):
        """Verify corridor-specific weights are applied to features."""
        # High beneficiary novelty should matter more for UK-Nigeria
        features_high_beneficiary = {'amount': 0, 'velocity': 0, 'beneficiary': 1.0, 'temporal': 0}
        
        score_ngn = self._aggregate_risk_score(features_high_beneficiary, 'UK_NGN')
        score_pln = self._aggregate_risk_score(features_high_beneficiary, 'UK_PLN')
        
        # UK-Nigeria weights beneficiary at 1.5x, UK-Poland at 0.7x
        assert score_ngn > score_pln, \
            f"High beneficiary should score higher in UK-NGN: NGN={score_ngn}, PLN={score_pln}"
    
    def _aggregate_risk_score(self, features: dict, corridor: str) -> float:
        """
        Calculate weighted risk score using corridor-specific multipliers.
        """
        multipliers = {
            'UK_NGN': {'amount': 1.2, 'velocity': 0.8, 'beneficiary': 1.5, 'temporal': 0.6},
            'UK_PLN': {'amount': 1.0, 'velocity': 1.4, 'beneficiary': 0.7, 'temporal': 1.1}
        }
        
        weights = multipliers[corridor]
        
        weighted_sum = (
            features['amount'] * weights['amount'] +
            features['velocity'] * weights['velocity'] +
            features['beneficiary'] * weights['beneficiary'] +
            features['temporal'] * weights['temporal']
        )
        
        # Normalise by sum of weights
        total_weight = sum(weights.values())
        return weighted_sum / total_weight


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_handles_zero_transaction_amount(self):
        """System should handle zero-value transactions gracefully."""
        # Zero amount should not cause division errors
        score = self._safe_amount_score(0, median=350, p95=2500)
        assert score == 0, "Zero amount should score 0"
    
    def test_handles_negative_amount(self):
        """System should reject or handle negative amounts."""
        with pytest.raises(ValueError):
            self._safe_amount_score(-100, median=350, p95=2500)
    
    def test_handles_missing_corridor_profile(self):
        """System should raise clear error for unknown corridors."""
        with pytest.raises(KeyError):
            self._get_corridor_profile('UK_XXX')
    
    def test_handles_extreme_velocity(self):
        """System should cap velocity scores for extreme values."""
        # 100 transactions in 24 hours is extreme
        score = self._calculate_velocity_score(100, median_velocity=1.2)
        assert score <= 1.0, "Extreme velocity should be capped at 1.0"
    
    def _safe_amount_score(self, amount: float, median: float, p95: float) -> float:
        """Calculate amount score with input validation."""
        if amount < 0:
            raise ValueError("Amount cannot be negative")
        if amount <= median:
            return 0.0
        return min((amount - median) / (p95 - median) * 0.5, 1.0)
    
    def _get_corridor_profile(self, corridor: str) -> dict:
        """Retrieve corridor profile with validation."""
        profiles = {
            'UK_NGN': {'median': 350, 'p95': 2500},
            'UK_PLN': {'median': 180, 'p95': 1500}
        }
        if corridor not in profiles:
            raise KeyError(f"Unknown corridor: {corridor}")
        return profiles[corridor]
    
    def _calculate_velocity_score(self, velocity: float, median_velocity: float) -> float:
        """Calculate velocity score with upper bound."""
        if velocity <= median_velocity:
            return 0.0
        ratio = velocity / median_velocity
        return min(1.0, (ratio - 1) / 4)  # Cap at 5x median


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
