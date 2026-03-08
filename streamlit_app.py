import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os
from preprocessing import clean_text, derive_sentiment

# Page Configuration
st.set_page_config(page_title="Food Product Review Intelligence", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Dark Theme and Card Layout
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_dataset():
    if os.path.exists("Reviews.csv"):
        return pd.read_csv("Reviews.csv")
    return pd.DataFrame()

def get_insights(df, sentiment):
    # Filter by sentiment
    sentiment_df = df[df['Sentiment'] == sentiment]
    
    # Combine cleaned reviews
    all_text = " ".join(sentiment_df['Cleaned_Review'].astype(str))
    
    # Simple word frequency
    words = all_text.split()
    # Filter out short or common filler words that might have escaped cleaning
    words = [w for w in words if len(w) > 3]
    common_words = [pair[0] for pair in Counter(words).most_common(5)]
    
    return common_words

def main():
    st.title("🍔 Food Product Review Intelligence Dashboard")
    st.markdown("Analyze customer feedback and sentiment for food products using AI.")
    
    # Sidebar Search
    st.sidebar.header("Search")
    product_query = st.sidebar.text_input("Search Food Product", placeholder="e.g., coffee, chocolate")
    analyze_button = st.sidebar.button("Analyze Reviews")
    
    # Load data
    df_raw = load_dataset()
    
    if analyze_button and product_query:
        # Filter reviews
        filtered_df = df_raw[df_raw["Text"].str.contains(product_query, case=False, na=False)].copy()
        
        if filtered_df.empty:
            st.error(f"No reviews found for '{product_query}'.")
            return
            
        # Limit analysis to 500 reviews
        analysis_df = filtered_df.head(500).copy()
        
        with st.spinner(f"Analyzing {len(analysis_df)} reviews for '{product_query}'..."):
            # Use Score column to assign sentiment
            analysis_df['Sentiment'] = analysis_df['Score'].apply(derive_sentiment)
            analysis_df['Confidence'] = 1.0 # Rule-based mapping has full confidence
            analysis_df['Cleaned_Review'] = analysis_df['Text'].apply(clean_text)
            
        # Metrics Section
        col1, col2, col3, col4 = st.columns(4)
        total_reviews = len(analysis_df)
        counts = analysis_df['Sentiment'].value_counts()
        
        pos_pct = (counts.get('Positive', 0) / total_reviews) * 100
        neu_pct = (counts.get('Neutral', 0) / total_reviews) * 100
        neg_pct = (counts.get('Negative', 0) / total_reviews) * 100
        
        col1.metric("Total Reviews", total_reviews)
        col2.metric("Positive", f"{pos_pct:.1f}%")
        col3.metric("Neutral", f"{neu_pct:.1f}%")
        col4.metric("Negative", f"{neg_pct:.1f}%")
        
        # Main Content Area
        m_col1, m_col2 = st.columns([1, 1])
        
        with m_col1:
            st.subheader("Sentiment Distribution")
            fig = go.Figure(data=[go.Pie(labels=['Positive', 'Neutral', 'Negative'], 
                                        values=[pos_pct, neu_pct, neg_pct],
                                        hole=.5,
                                        marker_colors=['#28a745', '#ffc107', '#dc3545'])])
            fig.update_layout(showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)
            
        with m_col2:
            st.subheader("Buy Recommendation")
            if pos_pct >= 60:
                recommendation = "BUY"
                rec_color = "green"
                rec_text = "Most customers report positive experiences with taste and quality."
            elif neg_pct >= 40:
                recommendation = "NOT RECOMMENDED"
                rec_color = "red"
                rec_text = "A significant number of customers report issues with this product."
            else:
                recommendation = "MIXED REVIEWS"
                rec_color = "orange"
                rec_text = "Customer experiences vary significantly. Check individual reviews."
                
            st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; border-left: 10px solid {rec_color}; background-color: #1e2130;">
                    <h2 style="color: {rec_color}; margin-top: 0;">Recommendation: {recommendation}</h2>
                    <p style="font-size: 1.1em;">{rec_text}</p>
                </div>
            """, unsafe_allow_html=True)
            
        # Customer Insights Section
        st.divider()
        st.subheader("Customer Insights")
        i_col1, i_col2, i_col3 = st.columns(3)
        
        with i_col1:
            st.markdown("**✅ Top Pros**")
            pros = get_insights(analysis_df, 'Positive')
            for pro in pros:
                st.write(f"- {pro}")
                
        with i_col2:
            st.markdown("**❌ Top Cons**")
            cons = get_insights(analysis_df, 'Negative')
            for con in cons:
                st.write(f"- {con}")
                
        with i_col3:
            st.markdown("**🏷️ Most Discussed Features**")
            all_text = " ".join(analysis_df['Cleaned_Review'].astype(str)).split()
            # More specific feature extraction (simple frequency of nouns/adj would be better but let's stick to word count)
            features = [pair[0] for pair in Counter([w for w in all_text if len(w) > 4]).most_common(10)]
            st.write(", ".join(features))
            
        # Word Cloud
        st.divider()
        st.subheader("Review Word Cloud")
        all_text = " ".join(analysis_df['Cleaned_Review'].astype(str))
        if all_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='#0e1117', colormap='viridis').generate(all_text)
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            fig_wc.patch.set_facecolor('#0e1117')
            st.pyplot(fig_wc)
        
        # Review Table
        st.divider()
        st.subheader("Sample Analyzed Reviews")
        display_df = analysis_df[['Text', 'Sentiment', 'Confidence']].head(20).copy()
        # Clean display text (shorten if too long)
        display_df['Text'] = display_df['Text'].apply(lambda x: x[:200] + "..." if len(str(x)) > 200 else x)
        st.table(display_df)

    else:
        # Landing Page
        st.info("Enter a food product name in the sidebar to begin analysis.")
        st.image("https://images.unsplash.com/photo-1542838132-92c53300491e?auto=format&fit=crop&q=80&w=1000", use_container_width=True)

if __name__ == "__main__":
    main()
