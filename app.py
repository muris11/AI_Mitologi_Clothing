from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from functools import wraps
from model import RecommenderSystem

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Restrict CORS to frontend and backend origins (safe default if env var missing).
# Defaults cover localhost and private LAN ranges so IP changes don't require code edits.
allowed_origins_env = os.environ.get('ALLOWED_ORIGINS')
if allowed_origins_env:
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(',') if origin.strip()]
else:
    allowed_origins = [
        r"^https?://localhost(?::\d+)?$",
        r"^https?://127\.0\.0\.1(?::\d+)?$",
        r"^https?://10\.\d+\.\d+\.\d+(?::\d+)?$",
        r"^https?://192\.168\.\d+\.\d+(?::\d+)?$",
        r"^https?://172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+(?::\d+)?$",
    ]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

# Initialize model
recommender = RecommenderSystem.load_model()

API_KEY = os.environ.get('RECOMMENDER_API_KEY')
if not API_KEY or API_KEY == 'mitologi-secret-key':
    print("WARNING: RECOMMENDER_API_KEY is missing or weak! Service may be insecure.")


def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key and api_key == API_KEY:
            return f(*args, **kwargs)
        return jsonify({'error': {'code': 'unauthorized', 'message': 'Unauthorized'}}), 401
    return decorated_function


def parse_limit(default_value, max_value=100):
    raw_limit = request.args.get('limit', default_value)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return None, jsonify({'error': {'code': 'invalid_parameter', 'message': 'Invalid limit parameter'}}), 400

    if limit < 1 or limit > max_value:
        return None, jsonify({'error': {'code': 'invalid_parameter', 'message': f'limit must be between 1 and {max_value}'}}), 400

    return limit, None, None

@app.route('/api/recommendations/user/<int:user_id>', methods=['GET'])
@require_api_key
def recommend_user(user_id):
    limit, error_response, status_code = parse_limit(10)
    if error_response is not None:
        return error_response, status_code

    recommendations = recommender.predict_user_products(user_id, limit)

    # Cold-start fallback: if no recommendations, return popular products
    if not recommendations:
        recommendations = recommender.get_popular_products(limit)

    return jsonify({'data': recommendations})

@app.route('/api/recommendations/product/<int:product_id>', methods=['GET'])
@require_api_key
def recommend_product(product_id):
    limit, error_response, status_code = parse_limit(5)
    if error_response is not None:
        return error_response, status_code

    recommendations = recommender.predict_related_products(product_id, limit)
    return jsonify({'data': recommendations})

@app.route('/api/train', methods=['POST'])
@require_api_key
def train():
    print("Received training request")
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': {'code': 'invalid_json', 'message': 'Invalid JSON payload'}}), 400

    interactions = data.get('interactions', [])
    products = data.get('products', [])
    
    print(f"Products: {len(products)}, Interactions: {len(interactions)}")

    if not interactions and not products:
        print("Error: No data")
        return jsonify({'error': {'code': 'validation_error', 'message': 'Interactions or Products data is required'}}), 400
        
    try:
        recommender.train(interactions, products)
        print("Model trained successfully")
        return jsonify({'data': None, 'message': 'Model trained successfully'})
    except Exception as e:
        print(f"Training error: {e}")
        return jsonify({'error': {'code': 'training_error', 'message': str(e)}}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'trained': recommender.is_trained})

if __name__ == '__main__':
    # Run on port 8001 to match Laravel config
    port = int(os.environ.get('PORT', 8001))
    app.run(port=port, host='0.0.0.0')
