# Methodology: Dynamic Signal Weighting for Corridor-Aware Fraud Detection

## Overview

This document details the technical approach to building a context-aware fraud detection system that adapts to the behavioural characteristics of specific payment corridors.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Transaction Input                             │
│  (amount, sender, beneficiary, corridor, timestamp, device)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Corridor Classification                          │
│         Identify origin-destination pair risk tier              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               Feature Engineering Layer                          │
│  • Transaction velocity (corridor-normalised)                   │
│  • Amount deviation (corridor-specific percentile)              │
│  • Beneficiary novelty score                                    │
│  • Device consistency index                                     │
│  • Temporal pattern matching                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Dynamic Weight Adjustment                           │
│     weights = base_weights × corridor_multipliers               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Infrastructure Health Check                         │
│    Cross-reference against payment rail status                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Risk Scoring Engine                            │
│          Final score + explainability output                    │
└─────────────────────────────────────────────────────────────────┘
```

## Component 1: Corridor Classification

### Risk Tier Assignment

Payment corridors are classified into risk tiers based on historical fraud rates and transaction characteristics:

| Tier | Fraud Rate Range | Example Corridors | Baseline Sensitivity |
|------|------------------|-------------------|---------------------|
| 1 (Low) | <0.1% | UK→EU, UK→US | Standard |
| 2 (Medium) | 0.1% - 0.5% | UK→India, UK→Philippines | Elevated |
| 3 (High) | 0.5% - 2.0% | UK→Nigeria, UK→Ghana | High |
| 4 (Very High) | >2.0% | Emerging corridors, new routes | Maximum |

**Important**: High-tier corridors are not inherently "bad"—they often represent underserved markets with legitimate high-volume remittance flows. The tier informs calibration, not blocking.

### Corridor Profile Storage

Each corridor maintains a statistical profile updated weekly:

```python
corridor_profile = {
    'corridor_id': 'GBP_NGN',
    'tier': 3,
    'statistics': {
        'median_amount': 350.00,
        'p95_amount': 2500.00,
        'median_velocity_24h': 1.2,  # transactions per sender
        'p95_velocity_24h': 4.0,
        'peak_hours': [9, 10, 11, 18, 19],  # UTC
        'peak_days': [0, 4, 5],  # Monday, Friday, Saturday
    },
    'fraud_patterns': {
        'common_vectors': ['account_takeover', 'new_beneficiary_rush'],
        'avg_fraud_amount': 890.00,
        'fraud_velocity_multiplier': 3.2,  # fraudsters transact 3.2x faster
    }
}
```

## Component 2: Feature Engineering

### 2.1 Transaction Velocity (Corridor-Normalised)

Standard velocity calculation:
```
velocity_24h = count(transactions in last 24 hours)
```

Corridor-normalised velocity:
```
normalised_velocity = velocity_24h / corridor_median_velocity
```

**Why this matters**: A sender making 4 transactions in 24 hours is suspicious in UK→Poland (where median is 0.8) but normal in UK→Philippines during peak remittance periods (where median is 3.5).

### 2.2 Amount Deviation Score

Rather than absolute thresholds, we calculate percentile position within the corridor:

```python
def amount_deviation_score(amount, corridor_profile):
    median = corridor_profile['statistics']['median_amount']
    p95 = corridor_profile['statistics']['p95_amount']
    
    if amount <= median:
        return 0.0  # Below median = low risk signal
    elif amount <= p95:
        # Linear interpolation between median and p95
        return (amount - median) / (p95 - median) * 0.5
    else:
        # Above p95 = elevated signal, capped at 1.0
        excess_ratio = (amount - p95) / p95
        return min(0.5 + excess_ratio, 1.0)
```

### 2.3 Beneficiary Novelty Score

New beneficiaries are a key fraud signal, but the weight varies by corridor:

```python
def beneficiary_novelty_score(sender_id, beneficiary_id, corridor_id):
    sender_history = get_sender_history(sender_id)
    
    # Has sender used this beneficiary before?
    if beneficiary_id in sender_history['beneficiaries']:
        return 0.0
    
    # First-time beneficiary: how many does sender typically have?
    existing_count = len(sender_history['beneficiaries'])
    corridor_avg = get_corridor_avg_beneficiaries(corridor_id)
    
    if existing_count < corridor_avg:
        return 0.3  # Normal to add beneficiaries early
    else:
        return 0.7  # Unusual to add new beneficiary for established sender
```

### 2.4 Device Consistency Index

Tracks device fingerprint stability:

```python
def device_consistency_index(sender_id, current_device):
    known_devices = get_sender_devices(sender_id)
    
    if current_device in known_devices:
        return 0.0  # Known device
    
    # New device: how often does this sender change devices?
    device_change_rate = len(known_devices) / sender_account_age_days
    corridor_avg_change_rate = get_corridor_device_change_rate()
    
    if device_change_rate > corridor_avg_change_rate * 2:
        return 0.9  # Unusually frequent device changes
    else:
        return 0.4  # New device but within normal range
```

### 2.5 Temporal Pattern Matching

Compares transaction timing against corridor norms:

```python
def temporal_anomaly_score(timestamp, corridor_profile):
    hour = timestamp.hour
    day = timestamp.weekday()
    
    peak_hours = corridor_profile['statistics']['peak_hours']
    peak_days = corridor_profile['statistics']['peak_days']
    
    hour_score = 0.0 if hour in peak_hours else 0.3
    day_score = 0.0 if day in peak_days else 0.2
    
    return hour_score + day_score
```

## Component 3: Dynamic Weight Adjustment

### Base Weights

Starting weights for each feature (learned from global fraud data):

```python
base_weights = {
    'velocity': 0.25,
    'amount_deviation': 0.20,
    'beneficiary_novelty': 0.25,
    'device_consistency': 0.20,
    'temporal_anomaly': 0.10,
}
```

### Corridor Multipliers

Each corridor has learned multipliers that adjust base weights:

```python
# Example: UK→Nigeria corridor
corridor_multipliers_GBP_NGN = {
    'velocity': 0.8,           # Velocity less predictive (high legitimate volume)
    'amount_deviation': 1.2,   # Amount more predictive (fraud tends to be higher value)
    'beneficiary_novelty': 1.5, # New beneficiary highly predictive
    'device_consistency': 1.3,  # Device changes more suspicious
    'temporal_anomaly': 0.6,   # Timing less predictive (24/7 remittance culture)
}

# Example: UK→Poland corridor
corridor_multipliers_GBP_PLN = {
    'velocity': 1.4,           # Velocity highly predictive (regular monthly pattern)
    'amount_deviation': 0.9,   # Amount less predictive (narrow range)
    'beneficiary_novelty': 0.7, # New beneficiary less suspicious (workers change)
    'device_consistency': 1.0,  # Standard weight
    'temporal_anomaly': 1.2,   # Timing more predictive (weekday payroll pattern)
}
```

### Weight Calculation

```python
def calculate_adjusted_weights(corridor_id):
    multipliers = get_corridor_multipliers(corridor_id)
    
    adjusted = {}
    for feature, base_weight in base_weights.items():
        adjusted[feature] = base_weight * multipliers[feature]
    
    # Normalise to sum to 1.0
    total = sum(adjusted.values())
    return {k: v/total for k, v in adjusted.items()}
```

## Component 4: Infrastructure Health Check

### The Problem

Local payment infrastructure in emerging markets experiences frequent outages, timeouts, and partial failures. Without adjustment, these appear as:
- Rapid retry attempts (high velocity)
- Multiple failed transactions followed by success (suspicious pattern)
- Device/IP changes as users try different methods

### The Solution

Cross-reference transaction patterns against real-time infrastructure status:

```python
def infrastructure_adjustment(transaction, raw_score):
    corridor = transaction['corridor']
    timestamp = transaction['timestamp']
    
    # Check infrastructure health for destination
    infra_status = get_infrastructure_status(corridor, timestamp)
    
    if infra_status['health'] < 0.7:  # Degraded service
        # Check if transaction pattern matches infrastructure issues
        if transaction['is_retry'] and infra_status['common_error'] == 'timeout':
            # Likely legitimate retry due to infrastructure
            adjustment = -0.3
        else:
            adjustment = 0.0
    else:
        adjustment = 0.0
    
    return max(0, raw_score + adjustment)
```

### Infrastructure Status Sources

- Payment processor API health endpoints
- Historical success rate tracking (rolling 1-hour window)
- Third-party monitoring services (where available)
- Manual status flags for known outages

## Component 5: Risk Scoring Engine

### Final Score Calculation

```python
def calculate_fraud_score(transaction):
    corridor = transaction['corridor']
    
    # Step 1: Calculate raw feature scores
    features = {
        'velocity': calculate_velocity_score(transaction),
        'amount_deviation': calculate_amount_score(transaction),
        'beneficiary_novelty': calculate_beneficiary_score(transaction),
        'device_consistency': calculate_device_score(transaction),
        'temporal_anomaly': calculate_temporal_score(transaction),
    }
    
    # Step 2: Get corridor-adjusted weights
    weights = calculate_adjusted_weights(corridor)
    
    # Step 3: Weighted sum
    raw_score = sum(features[f] * weights[f] for f in features)
    
    # Step 4: Infrastructure adjustment
    adjusted_score = infrastructure_adjustment(transaction, raw_score)
    
    # Step 5: Apply corridor baseline offset
    corridor_baseline = get_corridor_baseline(corridor)
    final_score = adjusted_score + corridor_baseline
    
    return {
        'score': final_score,
        'features': features,
        'weights': weights,
        'adjustments': {
            'infrastructure': adjusted_score - raw_score,
            'baseline': corridor_baseline,
        }
    }
```

### Decision Thresholds

```python
def make_decision(score_result):
    score = score_result['score']
    
    if score < 0.3:
        return 'APPROVE'
    elif score < 0.6:
        return 'REVIEW'  # Manual review queue
    else:
        return 'BLOCK'
```

### Explainability Output

For regulatory compliance and analyst review, each decision includes:

```python
explanation = {
    'decision': 'REVIEW',
    'score': 0.47,
    'primary_factors': [
        'New beneficiary for established sender (novelty_score: 0.7)',
        'Transaction amount in 92nd percentile for corridor (amount_score: 0.4)',
    ],
    'mitigating_factors': [
        'Known device used (device_score: 0.0)',
        'Transaction during peak hours (temporal_score: 0.0)',
    ],
    'corridor_context': 'UK→Nigeria: High-volume remittance corridor with elevated baseline monitoring',
}
```

## Model Training and Updates

### Initial Training

1. Historical transaction data (12+ months) with fraud labels
2. Group by corridor, calculate baseline statistics
3. Train gradient-boosted model on global data for base weights
4. Calculate corridor multipliers using held-out validation set

### Continuous Learning

- **Weekly**: Update corridor statistics (medians, percentiles)
- **Monthly**: Retrain corridor multipliers using recent fraud cases
- **Quarterly**: Full model retraining with expanded feature set

### Cold Start for New Corridors

New corridors with insufficient data inherit from similar corridors:

```python
def get_similar_corridor(new_corridor):
    # Match by: destination region, currency, historical relationship
    # Example: New UK→Kenya corridor inherits from UK→Nigeria initially
    return find_nearest_corridor(new_corridor, similarity_features)
```

## Performance Monitoring

### Key Metrics (Daily)

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Fraud Recall | ≥90% | <85% |
| False Positive Rate | <4% | >6% |
| Review Queue Volume | <5% of transactions | >8% |
| P99 Latency | <200ms | >300ms |

### Corridor-Level Monitoring

Each corridor is monitored independently to catch drift:

```python
def check_corridor_health(corridor_id, date_range):
    metrics = calculate_corridor_metrics(corridor_id, date_range)
    
    alerts = []
    if metrics['false_positive_rate'] > corridor_threshold * 1.5:
        alerts.append(f'FPR spike in {corridor_id}')
    if metrics['fraud_recall'] < 0.85:
        alerts.append(f'Recall drop in {corridor_id}')
    
    return alerts
```

---

## Summary

The dynamic signal weighting approach achieves corridor-aware fraud detection through:

1. **Corridor profiling**: Statistical baselines for each payment route
2. **Normalised features**: Scores relative to corridor norms, not global averages
3. **Adaptive weights**: Feature importance varies by corridor characteristics
4. **Infrastructure awareness**: Distinguishes system failures from fraud signals
5. **Explainable outputs**: Clear reasoning for each decision

This methodology reduced fraud losses by 12% while maintaining 90%+ recall and significantly reducing false positives that previously blocked legitimate remittance users.

---

*Technical implementation details. Specific thresholds and multipliers are illustrative.*
