# Food Product Review Intelligence Dashboard

A Streamlit-based dashboard for analyzing sentiment and customer insights from the Amazon Fine Food Reviews dataset.

**Dashboard Link** : https://food-review-intelligence-dashboard.streamlit.app/

## Features
- **Product Search**: Search for reviews of specific food products (e.g., coffee, chocolate).
- **Sentiment Analysis**: Sentiment is derived directly from user scores (1-2: Negative, 3: Neutral, 4-5: Positive).
- **Sentiment Distribution**: Interactive donut chart showing the breakdown of sentiments.
- **Customer Insights**: Top pros, cons, and most discussed features extracted from reviews.
- **Word Cloud**: Visual representation of common words in reviews.
- **Buy Recommendation**: Rule-based recommendation engine.

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Refactoring for Python 3.14 Compatibility
The application has been refactored to remove TensorFlow and Keras dependencies to ensure compatibility with Python 3.14 on Streamlit Cloud. 
- Sentiment is now computed using rule-based mapping from the `Score` column.
- The model loading and prediction logic has been removed from the production dashboard.
- RNN training code is still available in `train_model.py` for reference and documentation.

## Requirements
- streamlit>=1.24
- pandas>=1.5
- numpy>=1.25
- scikit-learn>=1.2
- nltk>=3.7
- plotly>=5.0
- wordcloud>=1.8
