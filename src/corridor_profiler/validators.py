"""
Profile Validators

Validation utilities to ensure corridor profile integrity
and detect anomalies in profile updates.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .corridor_profiler import CorridorProfile

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a profile validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self) -> bool:
        return self.is_valid


class ProfileValidator:
    """
    Validates corridor profiles for integrity and consistency.
    
    Checks include:
    - Required fields present
    - Values within reasonable bounds
    - Statistical consistency (e.g., median < p95 < p99)
    - Drift detection between profile versions
    """
    
    # Validation thresholds
    MAX_WEEKLY_DRIFT_PERCENT = 25.0  # Maximum allowed weekly change
    MIN_TRANSACTION_COUNT = 100
    MAX_FRAUD_RATE = 0.10  # 10% - anything higher is suspicious
    
    def validate_profile(self, profile: CorridorProfile) -> ValidationResult:
        """
        Run all validation checks on a profile.
        
        Parameters:
        -----------
        profile : CorridorProfile
            The profile to validate
            
        Returns:
        --------
        ValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        
        # Check required fields
        field_errors = self._validate_required_fields(profile)
        errors.extend(field_errors)
        
        # Check value bounds
        bound_errors, bound_warnings = self._validate_bounds(profile)
        errors.extend(bound_errors)
        warnings.extend(bound_warnings)
        
        # Check statistical consistency
        consistency_errors = self._validate_consistency(profile)
        errors.extend(consistency_errors)
        
        # Check for suspicious values
        suspicious_warnings = self._check_suspicious_values(profile)
        warnings.extend(suspicious_warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_update(self, 
                        old_profile: CorridorProfile,
                        new_profile: CorridorProfile) -> ValidationResult:
        """
        Validate a profile update for excessive drift.
        
        Parameters:
        -----------
        old_profile : CorridorProfile
            The previous profile version
        new_profile : CorridorProfile
            The new profile to validate
            
        Returns:
        --------
        ValidationResult with drift warnings
        """
        errors = []
        warnings = []
        
        # First validate the new profile itself
        base_validation = self.validate_profile(new_profile)
        errors.extend(base_validation.errors)
        warnings.extend(base_validation.warnings)
        
        # Check corridor code matches
        if old_profile.corridor_code != new_profile.corridor_code:
            errors.append(
                f"Corridor code mismatch: {old_profile.corridor_code} vs {new_profile.corridor_code}"
            )
            return ValidationResult(False, errors, warnings)
        
        # Check for excessive drift in key metrics
        drift_warnings = self._check_drift(old_profile, new_profile)
        warnings.extend(drift_warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_required_fields(self, profile: CorridorProfile) -> List[str]:
        """Check that all required fields are present and non-null."""
        errors = []
        
        required_fields = [
            ('corridor_code', profile.corridor_code),
            ('median_amount', profile.median_amount),
            ('p95_amount', profile.p95_amount),
            ('median_velocity_24h', profile.median_velocity_24h),
            ('version', profile.version),
        ]
        
        for field_name, value in required_fields:
            if value is None:
                errors.append(f"Required field '{field_name}' is missing")
            elif isinstance(value, str) and not value.strip():
                errors.append(f"Required field '{field_name}' is empty")
        
        return errors
    
    def _validate_bounds(self, profile: CorridorProfile) -> Tuple[List[str], List[str]]:
        """Check that values are within reasonable bounds."""
        errors = []
        warnings = []
        
        # Amount bounds
        if profile.median_amount <= 0:
            errors.append(f"Median amount must be positive: {profile.median_amount}")
        
        if profile.median_amount > 100000:
            warnings.append(f"Unusually high median amount: £{profile.median_amount:,.0f}")
        
        # Velocity bounds
        if profile.median_velocity_24h < 0:
            errors.append(f"Velocity cannot be negative: {profile.median_velocity_24h}")
        
        if profile.median_velocity_24h > 50:
            warnings.append(
                f"Unusually high median velocity: {profile.median_velocity_24h} txns/day"
            )
        
        # Fraud rate bounds
        if profile.baseline_fraud_rate < 0:
            errors.append(f"Fraud rate cannot be negative: {profile.baseline_fraud_rate}")
        
        if profile.baseline_fraud_rate > self.MAX_FRAUD_RATE:
            warnings.append(
                f"Unusually high fraud rate: {profile.baseline_fraud_rate:.1%}"
            )
        
        # Transaction count
        if profile.transaction_count < self.MIN_TRANSACTION_COUNT:
            warnings.append(
                f"Low transaction count ({profile.transaction_count}) may produce unreliable profile"
            )
        
        return errors, warnings
    
    def _validate_consistency(self, profile: CorridorProfile) -> List[str]:
        """Check statistical consistency of the profile."""
        errors = []
        
        # Percentiles must be in order
        if not (profile.p25_amount <= profile.median_amount <= profile.p75_amount):
            errors.append(
                f"Amount percentiles out of order: p25={profile.p25_amount}, "
                f"median={profile.median_amount}, p75={profile.p75_amount}"
            )
        
        if not (profile.median_amount <= profile.p95_amount <= profile.p99_amount):
            errors.append(
                f"Upper percentiles out of order: median={profile.median_amount}, "
                f"p95={profile.p95_amount}, p99={profile.p99_amount}"
            )
        
        # Min/max consistency
        if profile.min_amount > profile.median_amount:
            errors.append(
                f"Minimum amount ({profile.min_amount}) exceeds median ({profile.median_amount})"
            )
        
        if profile.max_amount < profile.p99_amount:
            errors.append(
                f"Maximum amount ({profile.max_amount}) less than p99 ({profile.p99_amount})"
            )
        
        # Velocity consistency
        if profile.median_velocity_24h > profile.p95_velocity_24h:
            errors.append(
                f"Median velocity ({profile.median_velocity_24h}) exceeds "
                f"p95 velocity ({profile.p95_velocity_24h})"
            )
        
        return errors
    
    def _check_suspicious_values(self, profile: CorridorProfile) -> List[str]:
        """Check for values that might indicate data issues."""
        warnings = []
        
        # Check for suspiciously round numbers (might indicate placeholder data)
        if profile.median_amount == round(profile.median_amount, -2):
            if profile.median_amount in [100, 200, 500, 1000]:
                warnings.append(
                    f"Suspiciously round median amount: £{profile.median_amount}"
                )
        
        # Check for zero standard deviation (all same amount)
        if profile.std_amount == 0:
            warnings.append("Zero standard deviation - all transactions same amount?")
        
        # Check unique senders vs transaction count ratio
        if profile.transaction_count > 0 and profile.unique_senders > 0:
            ratio = profile.transaction_count / profile.unique_senders
            if ratio < 1.0:
                warnings.append(
                    f"More unique senders than transactions - data issue?"
                )
            elif ratio > 100:
                warnings.append(
                    f"Very high transactions per sender ratio ({ratio:.1f})"
                )
        
        return warnings
    
    def _check_drift(self, 
                     old_profile: CorridorProfile,
                     new_profile: CorridorProfile) -> List[str]:
        """Check for excessive drift between profile versions."""
        warnings = []
        
        metrics_to_check = [
            ('median_amount', old_profile.median_amount, new_profile.median_amount),
            ('p95_amount', old_profile.p95_amount, new_profile.p95_amount),
            ('median_velocity_24h', old_profile.median_velocity_24h, 
             new_profile.median_velocity_24h),
            ('baseline_fraud_rate', old_profile.baseline_fraud_rate,
             new_profile.baseline_fraud_rate),
        ]
        
        for metric_name, old_value, new_value in metrics_to_check:
            if old_value == 0:
                continue
            
            drift_percent = abs(new_value - old_value) / old_value * 100
            
            if drift_percent > self.MAX_WEEKLY_DRIFT_PERCENT:
                warnings.append(
                    f"High drift in {metric_name}: {drift_percent:.1f}% "
                    f"({old_value:.2f} → {new_value:.2f})"
                )
        
        return warnings
