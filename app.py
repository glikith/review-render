import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import plotly.express as px
import copy
import re  # <-- CHANGE 1: Import the regular expression library

# --- NLTK Setup ---
# Download necessary NLTK data (only runs once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- CONFIGURATION: Set the file to load ---
DATASET_FILEPATH = "Restaurants_Test_Data.csv"
TEXT_COLUMN_NAME = "Sentence"


# --- Aspect Keywords (The "Simple" ABSA Model) ---
# --- UPDATED AND EXPANDED KEYWORDS ---
ASPECT_KEYWORDS = {
    "food": ['food', 'dish', 'menu', 'curry', 'pizza', 'burger', 'sushi', 'pasta', 'steak', 'salad', 'taste', 'flavor', 'delicious', 'tasty', 'portions',
             'chicken', 'beef', 'pork', 'fish', 'seafood', 'drink', 'beverage', 'coffee', 'dessert', 'appetizer', 'appetizers',
             'fresh', 'hot', 'cold', 'bland', 'overcooked', 'undercooked', 'eatable', 'bread'],
    "service": ['service', 'waiter', 'waitress', 'staff', 'host', 'attendant', 'slow', 'fast', 'friendly', 'rude', 'attentive', 'waiting',
                'server', 'manager', 'hostess', 'reservation', 'order', 'wait', 'ignored', 'helpful', 'polite', 'unbelievably'],
    "ambiance": ['ambiance', 'atmosphere', 'decor', 'music', 'loud', 'quiet', 'vibe', 'setting', 'romantic', 'cleanliness',
                 'noise', 'noisy', 'dirty', 'clean', 'seating', 'table', 'lighting', 'crowded', 'sterile'],
    "price": ['price', 'cheap', 'expensive', 'value', 'cost', 'bill', 'paid', 'affordable', 'overpriced',
              'costly', 'bargain', '$$$', '$-', 'over-priced', 'prices']
}
# --- END OF UPDATE ---

# --- Sentiment Analyzer ---
sid = SentimentIntensityAnalyzer()

# --- Helper Functions ---

@st.cache_data
def load_data(filepath):
    """Loads the reviews from CSV."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=[TEXT_COLUMN_NAME])
    df[TEXT_COLUMN_NAME] = df[TEXT_COLUMN_NAME].astype(str)
    return df

def analyze_review_aspects(text):
    """
    Analyzes a single review.
    1. Splits review into sentences.
    2. Classifies sentiment of each sentence.
    3. Assigns each sentence to an aspect based on keywords.
    """
    aspect_sentiments = {aspect: [] for aspect in ASPECT_KEYWORDS.keys()}
    aspect_sentiments["other"] = []
    
    # <-- CHANGE 2: Pre-process text to split contrastive conjunctions
    # This turns "A but B" into "A. but B", allowing sent_tokenize to split it.
    text = re.sub(r' (but|however|while|though) ', r'. \1 ', text, flags=re.IGNORECASE)

    sentences = sent_tokenize(text)
    
    for sentence in sentences:
        score = sid.polarity_scores(sentence)['compound']
        
        # --- THIS IS THE FIX FOR THE NEUTRAL BUG ---
        # Ensure your thresholds are set to 0.05, not 0.5
        if score > 0.05:
            sentiment = "positive"
        elif score < -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        # --- END OF FIX ---
            
        found_aspect = False
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(keyword in sentence.lower() for keyword in keywords):
                aspect_sentiments[aspect].append((sentence, sentiment, score))
                found_aspect = True
                break # Assign to first matching aspect
                
        if not found_aspect:
            aspect_sentiments["other"].append((sentence, sentiment, score))
            
    return aspect_sentiments

def get_aggregated_stats(reviews_list):
    """
    Processes a list of review texts and aggregates sentiment counts for all aspects.
    """
    agg_data = {
        aspect: {"positive": 0, "negative": 0, "neutral": 0} 
        for aspect in list(ASPECT_KEYWORDS.keys()) + ["other"]
    }
    
    for review in reviews_list:
        if isinstance(review, str):
            analysis = analyze_review_aspects(review)
            for aspect, sentences in analysis.items():
                for (sentence, sentiment, score) in sentences:
                    agg_data[aspect][sentiment] += 1
                
    return agg_data

def update_agg_data(agg_data, new_analysis):
    """
    Efficiently updates the aggregated data with a single new analysis.
    """
    updated_data = copy.deepcopy(agg_data)
    for aspect, sentences in new_analysis.items():
        for (sentence, sentiment, score) in sentences:
            if aspect in updated_data and sentiment in updated_data[aspect]:
                updated_data[aspect][sentiment] += 1
    return updated_data

def plot_aspect_sentiment(aspect_name, data):
    """
    Creates a Plotly bar chart for a single aspect's sentiment.
    """
    df = pd.DataFrame(data.items(), columns=['Sentiment', 'Count'])
    colors = {'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#ffc107'}
    
    fig = px.bar(df, 
                 x='Sentiment', 
                 y='Count', 
                 color='Sentiment', 
                 color_discrete_map=colors,
                 title=f"{aspect_name.capitalize()} Sentiment")
    
    fig.update_layout(showlegend=False, 
                      yaxis_title="Review Count", 
                      xaxis_title=None)
    return fig

# --- Streamlit App Initialization ---

st.set_page_config(layout="wide", page_title="ABSA Feedback Tool")

if 'all_reviews' not in st.session_state:
    try:
        df = load_data(DATASET_FILEPATH)
        st.session_state.all_reviews = list(df[TEXT_COLUMN_NAME])
        st.session_state.agg_data = get_aggregated_stats(st.session_state.all_reviews)
        st.session_state.last_analysis = {}
    except FileNotFoundError:
        st.error(f"FATAL: '{DATASET_FILEPATH}' not found. Please make sure it's in the same folder.")
        st.session_state.all_reviews = []
        st.session_state.agg_data = get_aggregated_stats([])
        st.session_state.last_analysis = {}

# --- App Layout ---

st.markdown("""
<style>
/* This targets the h1 element specifically for the title */
.stApp h1 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* This targets the h1 element specifically for the title */
.stApp h2 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("Aspect-Based Feedback Analysis Tool")

st.markdown("---")

# --- Main Dashboard: Aggregated Analysis ---

st.header("Aggregated Feedback Dashboard")
st.markdown(f"<div style='text-align: center;'><b>Total Reviews Analyzed: {len(st.session_state.all_reviews)}<b></div>",
    unsafe_allow_html=True)

if not st.session_state.all_reviews:
    st.warning("No reviews loaded. Add a review below to begin.")
else:
    cols = st.columns(4)
    aspects_to_plot = list(ASPECT_KEYWORDS.keys())
    
    for i, aspect in enumerate(aspects_to_plot):
        with cols[i]:
            data = st.session_state.agg_data[aspect]
            fig = plot_aspect_sentiment(aspect, data)
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Live Demo Section ---

st.header("Add Your Own Review")
# st.markdown("Enter a review below. The analysis will appear, and the dashboard above will update **live**.")

col1, col2 = st.columns([0.6, 0.4])

with col1:
    user_input = st.text_area("Enter a new review:", 
                             height=150)
    
    b_col1, b_col2, _ = st.columns([0.5, 0.5, 1.0]) # Add some spacing

    with b_col1:
        if st.button("Analyze", type="primary", use_container_width=True):
            if user_input.strip():
                # 1. Analyze the new review
                new_analysis = analyze_review_aspects(user_input)
                st.session_state.last_analysis = new_analysis
                
                # 2. Add new review to the total list
                st.session_state.all_reviews.append(user_input.strip())
                
                # 3. Update the aggregated data
                st.session_state.agg_data = update_agg_data(st.session_state.agg_data, new_analysis)
                
                # Rerun to update the dashboard plots
                st.rerun()
            else:
                st.warning("Please enter a review.")

    with b_col2:
        if st.button("Reset Dashboard", use_container_width=True):
            # 1. CHANGE: Reload original data from the new file
            df = load_data(DATASET_FILEPATH)
            # 2. CHANGE: Get list from 'Sentence' column
            st.session_state.all_reviews = list(df[TEXT_COLUMN_NAME])
            # 3. Recalculate original stats
            st.session_state.agg_data = get_aggregated_stats(st.session_state.all_reviews)
            # 4. Clear last analysis
            st.session_state.last_analysis = {}
            st.toast("Dashboard has been reset to its original state!")
            # 5. Rerun
            st.rerun()

with col2:
    st.markdown("Analysis of Last Review")
    if not st.session_state.last_analysis:
        st.info("Results for the new review will appear here.")
    else:
        for aspect, sentences in st.session_state.last_analysis.items():
            if sentences:
                st.markdown(f"**{aspect.capitalize()}:**")
                for (sentence, sentiment, score) in sentences:
                    
                    # --- THIS IS THE FIX FOR THE DISPLAY BUG ---
                    # Clean up the sentence for display (remove the " . but " artifact)
                    display_sentence = re.sub(r'^\.? (but|however|while|though) ', r'\1 ', sentence.strip(), flags=re.IGNORECASE)
                    # --- END OF FIX ---

                    if sentiment == "positive":
                        st.success(f"[+] {display_sentence} (Score: {score:.2f})")
                    elif sentiment == "negative":
                        st.error(f"[-] {display_sentence} (Score: {score:.2f})")
                    else:
                        st.info(f"[~] {display_sentence} (Score: {score:.2f})")

st.markdown("---")

st.markdown(f"<div style='text-align: center;'><b>Made by Team 16 - AIML Section 1 </b> </div>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align: center;'>G Likith - 2420030056 | K Navya - 2420030180 | K Chaitanya Srinivas - 2420030765</div>", unsafe_allow_html=True)