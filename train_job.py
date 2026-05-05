import requests
import os
import json
from datetime import datetime
from app import recommender

LARAVEL_API_URL = os.environ.get('LARAVEL_URL', 'http://127.0.0.1:8011')
EXPORT_ENDPOINT = f"{LARAVEL_API_URL}/api/ml/export-data"

def run_train_job():
    """
    Cron job function that extracts interaction and product data from the
    Laravel backend, and triggers the RecommenderSystem training natively.
    """
    print(f"[{datetime.now()}] MLOps: Starting autonomous nightly training job...")
    
    try:
        print(f"[{datetime.now()}] MLOps: Downloading dataset from {EXPORT_ENDPOINT}...")
        response = requests.get(EXPORT_ENDPOINT, timeout=60)
        
        if response.status_code != 200:
            print(f"[{datetime.now()}] MLOps Error: Failed to fetch data. Laravel responded with status: {response.status_code}")
            return False
            
        try:
            data = response.json()
        except json.JSONDecodeError:
            print(f"[{datetime.now()}] MLOps Error: Failed to parse JSON from Laravel. Response snippet: {response.text[:200]}")
            return False
            
        interactions = data.get('interactions', [])
        products = data.get('products', [])
        
        print(f"[{datetime.now()}] MLOps: Downloaded {len(interactions)} interactions and {len(products)} products.")
        
        if not interactions and not products:
            print(f"[{datetime.now()}] MLOps Warning: No data available to train on. Aborting.")
            return False
            
        print(f"[{datetime.now()}] MLOps: Retraining Sklearn collaborative filtering model...")
        
        recommender.train(interactions, products)
        
        print(f"[{datetime.now()}] MLOps: Success! Model retrained and saved to disk.")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now()}] MLOps Network Error: Could not connect to Laravel API. Is the backend running? ({e})")
        return False
        
    except Exception as e:
        print(f"[{datetime.now()}] MLOps Fatal Error: Exception during training process: {e}")
        return False

if __name__ == "__main__":
    run_train_job()
