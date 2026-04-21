# Mitologi Clothing - AI Recommendation Service

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-yellow?style=for-the-badge&logo=python" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Flask-3.0-lightgrey?style=for-the-badge&logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Render-Deployment-blue?style=for-the-badge&logo=render" alt="Render">
</p>

This is a standalone microservice that provides machine learning-powered product recommendations for the **Mitologi Clothing** platform. It uses a hybrid approach of Collaborative Filtering and Content-Based Filtering to deliver personalized shopping experiences.

## 🧠 Recommendation Engine

- **Collaborative Filtering:** Uses a Naive Bayes model to predict product interests based on user interaction history.
- **Content-Based Filtering:** Uses TF-IDF Vectorization and Cosine Similarity to find products with similar titles, categories, and descriptions.
- **Cold-Start Fallback:** Automatically serves popular products when user data is insufficient.
- **Autonomous Training:** Includes a background scheduler that periodically fetches new data from the Laravel backend and retrains the models.

## 🚀 API Endpoints

- `GET /api/recommendations/user/<user_id>`: Get personalized products for a specific user.
- `GET /api/recommendations/product/<product_id>`: Get related products for a specific product page.
- `POST /api/train`: Manually trigger a model retraining session.
- `GET /api/health`: Monitor the service status and model training state.

*Note: All endpoints (except health) require an `X-API-Key` header for security.*

## 🛠 Tech Stack

- **Language:** Python 3.11+
- **Framework:** Flask
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **Production Server:** Waitress (WSGI)
- **Scheduling:** Schedule library

## 📦 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/muris11/AI_Mitologi_Clothing.git
   cd AI_Mitologi_Clothing
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup:**
   Create a `.env` file based on your local requirements:
   ```env
   PORT=8001
   RECOMMENDER_API_KEY=your_secret_key
   LARAVEL_URL=http://localhost:8011
   ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8011
   ```

5. **Run the server:**
   ```bash
   python server.py
   ```

## ☁️ Deployment (Render.com)

This service is optimized for **Render**:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python server.py`
- **Health Check Path:** `/api/health`

## 📄 License

The Mitologi Clothing AI Service is open-sourced software licensed under the [MIT license](https://opensource.org/licenses/MIT).
