from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os
import json
from datetime import datetime

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
models_dir = os.path.join(project_dir, 'models')

app = Flask(__name__, template_folder=os.path.join(project_dir, 'templates'))

# Load the model and feature names
try:
    with open(os.path.join(models_dir, 'movie_rating_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(models_dir, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    print("Model loaded successfully!")
    print(f"Using {len(feature_names)} features")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Please run train.py first to train the model.")

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        data = request.get_json()
        
        # Create feature array with zeros
        features = np.zeros(len(feature_names))
        
        # Get release month and season
        release_date = datetime.strptime(data['release_year'], '%Y')
        month = release_date.month
        season = get_season(month)
        
        # Basic numeric features
        runtime = float(data['runtime'])
        budget = float(data['budget'])
        
        # Create feature map
        feature_map = {
            'budget': budget,
            'popularity': float(data['popularity']),
            'runtime': runtime,
            'vote_count': float(data['vote_count']),
            'release_year': float(data['release_year']),
            'budget_per_minute': budget / runtime if runtime > 0 else 0,
            'production_company_count': 1,  # Default value
            'is_english': 1,  # Assume English
            f'season_{season}': 1
        }
        
        # Set genre features
        for genre in data.get('genres', []):
            genre_key = f'genre_{genre}'
            if genre_key in feature_names:
                feature_map[genre_key] = 1.0
        
        # Map features to correct positions
        for i, feature_name in enumerate(feature_names):
            features[i] = feature_map.get(feature_name, 0.0)
        
        # Make prediction (scaling is now handled by the pipeline)
        prediction = float(model.predict(features.reshape(1, -1))[0])
        
        # Round to 1 decimal place
        prediction = round(prediction, 1)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000) 