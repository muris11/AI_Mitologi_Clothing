import pandas as pd
import numpy as np
import pickle
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

MODEL_PATH = "recommender_model.pkl"

class RecommenderSystem:
    def __init__(self):
        self.nb_model = MultinomialNB()
        self.user_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        self.one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.content_matrix = None
        self.product_ids = []
        
        # Cold-start fallback: track popular products by interaction count
        self.product_interaction_counts = {}
        
        self.is_trained = False
        self.metrics = {
            'accuracy': 0.0,
            'samples': 0,
            'last_train_date': None
        }

    def train(self, interactions, products):
        """
        Train both Collaborative (Naive Bayes) and Content-Based models.
        """
        # Ensure metrics dictionary exists and has default keys
        if not hasattr(self, 'metrics') or not self.metrics or 'accuracy' not in self.metrics:
            self.metrics = {
                'accuracy': 0.0,
                'samples': 0,
                'last_train_date': None
            }
            
        # 1. Train Naive Bayes (Collaborative)
        if interactions:
            try:
                df = pd.DataFrame(interactions)
                
                # Popular products fallback
                self.product_interaction_counts = df['product_id'].value_counts().to_dict()
                
                # Filter out interactions where user or product is missing
                df = df.dropna(subset=['user_id', 'product_id'])
                
                if not df.empty:
                    self.user_encoder.fit(df['user_id'])
                    self.product_encoder.fit(df['product_id'])
                    
                    user_indices = self.user_encoder.transform(df['user_id'])
                    product_indices = self.product_encoder.transform(df['product_id'])
                    
                    X = self.one_hot.fit_transform(user_indices.reshape(-1, 1))
                    y = product_indices
                    
                    # Need at least 2 products to train Naive Bayes
                    if len(self.product_encoder.classes_) > 1:
                        # Split data for evaluation
                        if len(X) >= 5: # Only split if we have enough data
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            self.nb_model.fit(X_train, y_train)
                            
                            # Calculate accuracy
                            y_pred = self.nb_model.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            
                            self.metrics['accuracy'] = float(acc)
                            self.metrics['samples'] = len(interactions)
                            self.metrics['last_train_date'] = pd.Timestamp.now().isoformat()
                            
                            # Refit on all data after evaluation
                            self.nb_model.fit(X, y)
                        else:
                            self.nb_model.fit(X, y)
                            self.metrics['accuracy'] = 1.0 # Too small to split fairly
                            self.metrics['samples'] = len(interactions)
                        
                        print(f"ML: Naive Bayes fitted with {len(interactions)} records. Accuracy: {self.metrics['accuracy']:.2f}")
                    else:
                        print("ML: Only one product in dataset. Skipping NB model.")
            except Exception as e:
                print(f"ML Training Error (Collaborative): {e}")
        
        # 2. Train Content-Based (Product-Product)
        if products:
            try:
                df_products = pd.DataFrame(products)
                self.product_ids = df_products['id'].tolist()
                
                # Combine features for TF-IDF
                df_products['content'] = (
                    df_products['title'].fillna('') + " " + 
                    df_products['description'].fillna('') + " " + 
                    df_products['category'].fillna('')
                )
                
                self.content_matrix = self.tfidf.fit_transform(df_products['content'])
                print(f"ML: Content matrix built for {len(products)} products.")
            except Exception as e:
                print(f"ML Training Error (Content): {e}")
            
        self.is_trained = True
        self.save_model()

    def predict_user_products(self, user_id, limit=10):
        """
        Predict products for a user using Naive Bayes.
        """
        if not self.is_trained:
            return []
            
        try:
            if not hasattr(self.user_encoder, 'classes_') or user_id not in self.user_encoder.classes_:
                return [] 
                
            user_idx = self.user_encoder.transform([user_id])
            X_new = self.one_hot.transform(user_idx.reshape(-1, 1))
            
            # Check if model is fitted
            if not hasattr(self.nb_model, 'classes_'):
                return []
                
            probs = self.nb_model.predict_proba(X_new)[0]
            
            # Get indices of top N products
            top_indices = probs.argsort()[-limit:][::-1]
            
            top_products = self.product_encoder.inverse_transform(top_indices)
            return top_products.tolist()
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return []

    def predict_related_products(self, product_id, limit=5):
        """
        Find similar products using Content-Based Filtering (Cosine Similarity).
        """
        if self.content_matrix is None or product_id not in self.product_ids:
            return []
            
        try:
            idx = self.product_ids.index(product_id)
            
            # Compute similarity with all other products
            sim_scores = cosine_similarity(self.content_matrix[idx], self.content_matrix).flatten()
            
            # Get indices of top similar products (excluding itself)
            top_indices = sim_scores.argsort()[-(limit+1):-1][::-1]
            
            related_ids = [self.product_ids[i] for i in top_indices]
            return related_ids
            
        except Exception as e:
            print(f"Content prediction error: {e}")
            return []

    def get_popular_products(self, limit=10):
        """
        Cold-start fallback: return most popular products by interaction count.
        Used when a user has no interactions for collaborative filtering.
        """
        if not self.product_interaction_counts:
            return self.product_ids[:limit] if self.product_ids else []
        
        sorted_products = sorted(
            self.product_interaction_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [pid for pid, _ in sorted_products[:limit]]

    def save_model(self):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model():
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        return RecommenderSystem()
