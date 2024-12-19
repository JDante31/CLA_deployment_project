import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from xgboost import XGBRegressor
from sklearn.svm import SVR
import pickle
import os
import json
from datetime import datetime

def extract_top_k_from_json(json_str, k=3, key='name'):
    """Extract top k items from a JSON string list of dictionaries."""
    try:
        items = json.loads(json_str)
        return [item[key] for item in items[:k]]
    except:
        return []

def load_and_preprocess_data():
    print("Loading TMDB dataset...")
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    
    # Load movies dataset
    movies = pd.read_csv(os.path.join(data_dir, 'tmdb_5000_movies.csv'))
    
    # Print rating distribution statistics
    print("\nRating Distribution:")
    print(f"Mean Rating: {movies['vote_average'].mean():.2f}")
    print(f"Median Rating: {movies['vote_average'].median():.2f}")
    print(f"Std Dev: {movies['vote_average'].std():.2f}")
    print(f"Min Rating: {movies['vote_average'].min():.2f}")
    print(f"Max Rating: {movies['vote_average'].max():.2f}")
    
    # Count ratings in different ranges
    ranges = [(0,4), (4,6), (6,7), (7,8), (8,10)]
    print("\nRating Ranges:")
    for start, end in ranges:
        count = len(movies[(movies['vote_average'] >= start) & (movies['vote_average'] < end)])
        percent = count / len(movies) * 100
        print(f"{start}-{end}: {count} movies ({percent:.1f}%)")
    
    # Extract year and month from release_date
    movies['release_date'] = pd.to_datetime(movies['release_date'])
    movies['release_year'] = movies['release_date'].dt.year
    movies['release_month'] = movies['release_date'].dt.month
    
    # Create season feature
    movies['release_season'] = movies['release_month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Extract genres
    movies['genres'] = movies['genres'].fillna('[]')
    # Create genre columns
    all_genres = set()
    for genres in movies['genres'].apply(json.loads):
        for genre in genres:
            all_genres.add(genre['name'])
    
    # Create binary columns for genres
    for genre in all_genres:
        movies[f'genre_{genre}'] = movies['genres'].apply(
            lambda x: 1 if any(g['name'] == genre for g in json.loads(x)) else 0
        )
    
    # Extract production companies
    movies['production_companies'] = movies['production_companies'].fillna('[]')
    movies['production_company_count'] = movies['production_companies'].apply(
        lambda x: len(json.loads(x))
    )
    
    # Create budget features
    movies['budget'] = movies['budget'].fillna(0)
    movies['budget_per_minute'] = movies['budget'] / movies['runtime'].replace(0, np.nan)
    
    # Create language feature
    movies['is_english'] = (movies['original_language'] == 'en').astype(int)
    
    # Create season dummies
    season_dummies = pd.get_dummies(movies['release_season'], prefix='season')
    
    # Basic numeric features
    numeric_features = [
        'budget', 'popularity', 'runtime', 'vote_count', 'release_year',
        'budget_per_minute', 'production_company_count', 'is_english'
    ]
    
    # Get genre columns
    genre_columns = [col for col in movies.columns if col.startswith('genre_')]
    
    # Combine all features
    X = pd.concat([
        movies[numeric_features],
        movies[genre_columns],
        season_dummies
    ], axis=1)
    
    # Target variable
    y = movies['vote_average']
    
    # Handle missing values
    for feature in X.columns:
        X[feature] = X[feature].fillna(X[feature].median())
    
    # Remove entries with very few votes
    mask = movies['vote_count'] > 10
    X = X[mask]
    y = y[mask]
    
    print(f"Using {len(X.columns)} features")
    return X, y, list(X.columns)

def train_model():
    X, y, features = load_and_preprocess_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to try
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        ),
        'Ridge': RidgeCV(
            alphas=[0.01, 0.1, 1.0, 10.0],
            cv=5
        ),
        'Lasso': LassoCV(
            cv=5,
            random_state=42
        ),
        'SVR': SVR(
            kernel='rbf',
            C=1.0,
            epsilon=0.1
        )
    }
    
    # Compare models using cross-validation
    print("\nComparing different models using 5-fold cross-validation:")
    print("-" * 60)
    best_score = -np.inf
    best_model_name = None
    cv_results = {}
    
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=5, scoring='r2'
        )
        mean_score = scores.mean()
        std_score = scores.std()
        cv_results[name] = {
            'mean_score': mean_score,
            'std_score': std_score
        }
        print(f"{name:20s}: R² = {mean_score:.4f} (+/- {std_score:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
    
    print("-" * 60)
    print(f"Best model: {best_model_name} (R² = {best_score:.4f})")
    
    # Create and train the stacking ensemble
    base_models = [
        (name.lower().replace(' ', '_'), model)
        for name, model in models.items()
        if name != 'SVR'  # Exclude SVR as it's slow for large datasets
    ]
    
    # Create stacking model
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=LassoCV(cv=5),
        cv=5,
        n_jobs=-1
    )
    
    # Train the stacking model
    print("\nTraining stacking ensemble...")
    stacking_model.fit(X_train_scaled, y_train)
    
    # Evaluate all models on test set
    print("\nTest set performance:")
    print("-" * 60)
    
    # Train and evaluate individual models
    test_results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        test_results[name] = test_score
        print(f"{name:20s}: R² = {test_score:.4f}")
    
    # Evaluate stacking model
    stacking_score = stacking_model.score(X_test_scaled, y_test)
    test_results['Stacking Ensemble'] = stacking_score
    print(f"{'Stacking Ensemble':20s}: R² = {stacking_score:.4f}")
    
    # Find best model on test set
    best_test_model = max(test_results.items(), key=lambda x: x[1])
    print("-" * 60)
    print(f"Best model on test set: {best_test_model[0]} (R² = {best_test_model[1]:.4f})")
    
    # Use the best performing model
    if best_test_model[0] == 'Stacking Ensemble':
        final_model = stacking_model
    else:
        final_model = models[best_test_model[0]]
        final_model.fit(X_train_scaled, y_train)
    
    # Feature importance (if available)
    if hasattr(final_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': final_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
    
    # Create models directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the model and scaler
    print("\nSaving model and scaler...")
    with open(os.path.join(models_dir, 'movie_rating_model.pkl'), 'wb') as f:
        pickle.dump(final_model, f)
    
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names for the Flask app
    with open(os.path.join(models_dir, 'feature_names.json'), 'w') as f:
        json.dump(features, f)
    
    print("Training completed!")

if __name__ == '__main__':
    train_model() 