# Contributing to Cross-Border Fraud Detection

Thank you for your interest in contributing to this project. This document provides guidelines for contributing to the context-aware fraud detection system for cross-border payments.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

This project is committed to providing a welcoming and inclusive environment. All contributors are expected to:

- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/cross-border-fraud-detection.git
   cd cross-border-fraud-detection
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/cross-border-fraud-detection.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behaviour
- **Expected behaviour** vs actual behaviour
- **Environment details** (Python version, OS, dependencies)
- **Sample data** if applicable (anonymised)

### Suggesting Enhancements

Enhancement suggestions are welcome. Please include:

- **Use case** - describe the problem you're trying to solve
- **Proposed solution** - how you envision the enhancement working
- **Alternatives considered** - other approaches you've thought about
- **Impact assessment** - which components would be affected

### Contributing Code

We welcome contributions in the following areas:

#### High Priority
- Additional corridor profiles for underserved payment routes
- Performance optimisations for real-time scoring
- Enhanced infrastructure awareness for new payment rails
- Test coverage improvements

#### Medium Priority
- Documentation improvements
- Example notebooks for new use cases
- Visualisation utilities
- CLI tools for profile management

#### Research Contributions
- Novel feature engineering approaches
- Alternative weighting schemes
- Comparative analysis with other methods

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If available

# Install in editable mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_dynamic_weights.py -v
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use descriptive variable names

### Documentation

- All public functions must have docstrings (Google style)
- Include type information in docstrings
- Provide examples for complex functions

```python
def calculate_normalised_score(value: float, profile: CorridorProfile) -> float:
    """
    Calculate normalised feature score relative to corridor profile.
    
    Parameters:
    -----------
    value : float
        The observed value to normalise
    profile : CorridorProfile
        The corridor profile containing baseline statistics
        
    Returns:
    --------
    float
        Normalised score between 0 and 1
        
    Examples:
    ---------
    >>> profile = CorridorProfile(corridor_code='UK_NGN', median_amount=350, ...)
    >>> calculate_normalised_score(500, profile)
    0.35
    """
```

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or modifications
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

Example:
```
feat(profiler): add support for weekly profile blending

Implements exponential smoothing for profile updates to balance
stability with adaptability to changing corridor patterns.

Closes #42
```

## Testing

### Test Requirements

- All new features must include tests
- Bug fixes should include regression tests
- Maintain or improve code coverage

### Test Structure

```
tests/
├── __init__.py
├── test_dynamic_weights.py    # Unit tests for weighting module
├── test_corridor_profiles.py  # Unit tests for profiling module
├── test_integration.py        # Integration tests
└── fixtures/                  # Test data and fixtures
    └── sample_profiles.json
```

### Writing Tests

```python
class TestFeatureName:
    """Test suite for feature name."""
    
    def test_normal_case(self):
        """Test behaviour under normal conditions."""
        # Arrange
        input_data = ...
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        assert result == expected_output
    
    def test_edge_case(self):
        """Test behaviour at boundaries."""
        ...
    
    def test_error_handling(self):
        """Test that errors are handled appropriately."""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```

## Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add tests** for new features
3. **Ensure all tests pass** locally
4. **Update CHANGELOG.md** with your changes
5. **Create the pull request** with:
   - Clear description of changes
   - Link to related issue(s)
   - Screenshots for UI changes (if applicable)

### PR Review Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No sensitive data included
- [ ] Commit messages follow conventions

## Issue Guidelines

### Issue Templates

When creating issues, please use the appropriate template:

#### Bug Report
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behaviour:
1. Load profile for '...'
2. Call function '...'
3. See error

**Expected behaviour**
What you expected to happen.

**Environment**
- Python version:
- OS:
- Package versions:
```

#### Feature Request
```markdown
**Problem statement**
Describe the problem you're trying to solve.

**Proposed solution**
Describe how you'd like to see this solved.

**Alternatives**
Any alternative solutions you've considered.

**Additional context**
Any other context about the request.
```

## Questions?

If you have questions about contributing, please:

1. Check existing documentation
2. Search closed issues for similar questions
3. Open a new issue with the `question` label

---

Thank you for contributing to making cross-border payments fairer and more accessible.
