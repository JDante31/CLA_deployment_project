# Movie Rating Predictor

## Project Overview
This project implements a machine learning model to predict movie ratings based on various features from the TMDB 5000 Movies dataset. The system uses an ensemble of machine learning models to make predictions about a movie's potential rating based on characteristics like budget, genre, release timing, and other metadata.

## Data Analysis

### Dataset Description
- Source: TMDB 5000 Movies dataset
- Features: Multiple characteristics including budget, popularity, runtime, genres, release date, production companies, and more
- Target Variable: Movie rating (vote_average)

### Rating Distribution
- Mean Rating: ~6.9
- Median Rating: ~6.8
- Standard Deviation: ~0.8
- Range: 0-10

### Key Observations
1. Rating Distribution Pattern:
   - Majority of ratings cluster between 6-7
   - Very few movies rated below 4 or above 8
   - Distribution shows slight right skew

2. Feature Importance:
   - Vote count (number of ratings) significantly impacts prediction reliability
   - Budget and popularity are strong predictors
   - Genre combinations show meaningful correlations with ratings
   - Seasonal release patterns have minor but notable effects

## Model Architecture

### Feature Engineering
1. Temporal Features:
   - Release year and month extraction
   - Seasonal categorization (Spring, Summer, Fall, Winter)

2. Genre Processing:
   - One-hot encoding for all genres
   - Multiple genre combinations supported

3. Production Features:
   - Company count analysis
   - Budget per minute calculations
   - Language classification (English vs Non-English)

### Model Implementation
- Ensemble Architecture using scikit-learn Pipeline:
  - XGBoost (Best Individual Model)
  - Random Forest
  - Gradient Boosting
  - Ridge Regression
  - Lasso Regression
- Each model includes integrated StandardScaler
- Final Stacking Ensemble with Lasso meta-learner

### Model Performance
Latest Training Results (5-fold cross-validation):
- XGBoost: R² = 0.5483 (±0.0191)
- Gradient Boosting: R² = 0.5251 (±0.0258)
- Ridge: R² = 0.4522 (±0.0196)
- Lasso: R² = 0.4528 (±0.0178)

Final Stacking Ensemble:
- Test Set R² = 0.5831
- Best performing model overall
- Improved performance through model combination

Key Insights:
- XGBoost shows strongest individual performance
- Stacking ensemble provides ~3.5% improvement over best individual model
- Consistent prediction patterns across different model architectures

## Project Structure
```
movie_predictor/
├── data/               # Dataset storage
├── models/            # Trained model files
├── src/               # Source code
│   ├── train.py      # Model training script
│   └── app.py        # Flask web application
├── templates/         # HTML templates
└── requirements.txt   # Project dependencies
```

## Setup and Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python src/train.py
   ```
3. Run the web application:
   ```bash
   python src/app.py
   ```

## Limitations and Potential Improvements

### Current Limitations
1. Prediction Range:
   - Model tends to predict conservatively around the mean
   - Limited accuracy for extreme ratings (very high or very low)

2. Feature Coverage:
   - Limited use of textual data (descriptions, reviews)
   - No consideration of cast and crew information
   - Temporal trends might be oversimplified

### Potential Improvements

1. Data Enhancements:
   - Incorporate additional datasets (e.g., cast information, reviews)
   - Use text analysis on movie descriptions
   - Add more detailed financial data (marketing budget, worldwide earnings)

2. Model Improvements:
   - Implement deep learning models for text processing
   - Create separate models for different genres or budget ranges
   - Add time series components for temporal trends
   - Use more sophisticated ensemble techniques

3. Feature Engineering:
   - Create more interaction features between genres and other variables
   - Develop more sophisticated budget-related features
   - Include market condition indicators
   - Add social media sentiment analysis

4. Validation Enhancements:
   - Implement k-fold cross-validation with stratification
   - Add confidence intervals for predictions
   - Create more sophisticated evaluation metrics

## Conclusions
The current model provides reasonable predictions within the most common rating range but has limitations with extreme values. The implementation successfully demonstrates the use of ensemble methods and feature engineering, while leaving room for significant improvements through additional data sources and more sophisticated modeling techniques.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.