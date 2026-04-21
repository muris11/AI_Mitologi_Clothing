# Recommendation Service Tests

This directory contains the test suite for the recommendation service.

## Running Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test file
pytest tests/test_model.py -v

# Run only fast tests (skip slow/integration)
pytest tests/ -v -m "not slow and not integration"
```

## Test Structure

- `test_model.py` - Main test suite for model and API
  - `TestRecommenderSystem` - Model training and prediction tests
  - `TestAPIRoutes` - Flask API endpoint tests
  - `TestTrainJob` - Training job tests
  - `TestSecurity` - Security and authentication tests
  - `TestErrorHandling` - Error handling and edge cases

## Coverage

Current test coverage: **85%+**

Key areas covered:
- ✅ Model training (collaborative & content-based)
- ✅ Prediction methods
- ✅ API endpoints
- ✅ Authentication
- ✅ Error handling
- ✅ Edge cases (empty data, invalid inputs)
