import streamlit as st
import pandas as pd
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import plotly.express as px
import copy
import re

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

DATASET_FILEPATH = "Restaurants_Test_Data.csv"
TEXT_COLUMN_NAME = "Sentence"

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

sid = SentimentIntensityAnalyzer()

@st.cache_data
def load_data(filepath):
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, 'Restaurants_Test_Data.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(base_path, '..', 'data', 'Restaurants_Test_Data.csv')
    
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[TEXT_COLUMN_NAME])
    df[TEXT_COLUMN_NAME] = df[TEXT_COLUMN_NAME].astype(str)
    return df

def analyze_review_aspects(text):
    aspect_sentiments = {aspect: [] for aspect in ASPECT_KEYWORDS.keys()}
    aspect_sentiments["other"] = []
    text = re.sub(r' (but|however|while|though) ', r'. \1 ', text, flags=re.IGNORECASE)
    sentences = sent_tokenize(text)
    
    for sentence in sentences:
        score = sid.polarity_scores(sentence)['compound']
        sentiment = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
        found_aspect = False
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(keyword in sentence.lower() for keyword in keywords):
                aspect_sentiments[aspect].append((sentence, sentiment, score))
                found_aspect = True
                break
        if not found_aspect:
            aspect_sentiments["other"].append((sentence, sentiment, score))
    return aspect_sentiments

def get_aggregated_stats(reviews_list):
    agg_data = {aspect: {"positive": {"count": 0, "reviews": []}, "negative": {"count": 0, "reviews": []}, "neutral": {"count": 0, "reviews": []}} 
                for aspect in list(ASPECT_KEYWORDS.keys()) + ["other"]}
    for review in reviews_list:
        if isinstance(review, str):
            analysis = analyze_review_aspects(review)
            for aspect, sentences in analysis.items():
                for (sentence, sentiment, score) in sentences:
                    agg_data[aspect][sentiment]["count"] += 1
                    agg_data[aspect][sentiment]["reviews"].append(sentence)
    return agg_data

def update_agg_data(agg_data, new_analysis):
    updated_data = copy.deepcopy(agg_data)
    for aspect, sentences in new_analysis.items():
        for (sentence, sentiment, score) in sentences:
            if aspect in updated_data and sentiment in updated_data[aspect]:
                updated_data[aspect][sentiment]["count"] += 1
                updated_data[aspect][sentiment]["reviews"].append(sentence)
    return updated_data

def plot_aspect_sentiment(aspect_name, data):
    plot_dict = {sent: info["count"] for sent, info in data.items()}
    df = pd.DataFrame(plot_dict.items(), columns=['Sentiment', 'Count'])
    colors = {'positive': '#10B981', 'negative': '#EF4444', 'neutral': '#F59E0B'}
    fig = px.bar(df, x='Sentiment', y='Count', color='Sentiment', color_discrete_map=colors,
                 title=f"<b>{aspect_name.upper()}</b>")
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E2E8F0"),
        showlegend=False,
        yaxis_title=None,
        xaxis_title=None,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

st.set_page_config(layout="wide", page_title="Review Render", page_icon="📊")

st.markdown("""
    <style>
    .stApp {
        background-color: #0F172A;
        color: #E2E8F0;
    }
    .review-card {
        background-color: #1E293B;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #10B981;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    h1 {
        text-align: center;
        color: #10B981 !important;
        font-family: 'Inter', sans-serif;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.2rem;
        font-style: italic;
        margin-bottom: 2rem;
    }
    h2, h3 {
        color: #10B981 !important;
        font-family: 'Inter', sans-serif;
    }
    .total-reviews {
        font-size: 1.1rem;
        color: #94A3B8;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    hr {
        border: 0;
        height: 1px;
        background: #334155;
        margin: 2rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #94A3B8;
        font-size: 0.9rem;
    }
    .footer a {
        color: #10B981;
        text-decoration: none;
    }
    </style>
""", unsafe_allow_html=True)

if 'all_reviews' not in st.session_state:
    try:
        df = load_data(DATASET_FILEPATH)
        st.session_state.all_reviews = list(df[TEXT_COLUMN_NAME])
        st.session_state.agg_data = get_aggregated_stats(st.session_state.all_reviews)
        st.session_state.last_analysis = {}
    except Exception:
        st.session_state.all_reviews, st.session_state.agg_data, st.session_state.last_analysis = [], get_aggregated_stats([]), {}

st.markdown("<h1>Review Render - Aspect Based Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown(f"<div class='total-reviews'>Analyzing <b>{len(st.session_state.all_reviews)}</b> Customer Experiences</div>", unsafe_allow_html=True)

if st.session_state.all_reviews:
    cols = st.columns(4)
    aspects = list(ASPECT_KEYWORDS.keys())
    for i, aspect in enumerate(aspects):
        with cols[i]:
            st.plotly_chart(plot_aspect_sentiment(aspect, st.session_state.agg_data[aspect]), use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.header("📥 Submit Feedback")
col_in, col_res = st.columns([0.5, 0.5])

with col_in:
    user_input = st.text_area("What was your experience like?", placeholder="The food was amazing, but the service was slow...", height=150)
    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        if st.button("Analyze Review", type="primary", use_container_width=True):
            if user_input.strip():
                new_analysis = analyze_review_aspects(user_input)
                st.session_state.last_analysis = new_analysis   
                st.session_state.all_reviews.append(user_input.strip())
                st.session_state.agg_data = update_agg_data(st.session_state.agg_data, new_analysis)
                st.rerun()
    with btn_c2:
        if st.button("Reset System", use_container_width=True):
            st.session_state.clear()
            st.rerun()

with col_res:
    if st.session_state.last_analysis:
        st.markdown("### Real-time Classification")
        for aspect, sentences in st.session_state.last_analysis.items():
            if sentences:
                with st.expander(f"{aspect.upper()}", expanded=True):
                    for (sent, sentiment, score) in sentences:
                        if sentiment == "positive": st.success(f"✅ {sent}")
                        elif sentiment == "negative": st.error(f"❌ {sent}")
                        else: st.info(f"⚪ {sent}")
    else:
        st.info("Awaiting input for real-time analysis...")

st.markdown("<hr>", unsafe_allow_html=True)

st.header("💬 Recent Activity")
if st.session_state.all_reviews:
    for rev in reversed(st.session_state.all_reviews[-5:]):
        st.markdown(f"""<div class="review-card">{rev}</div>""", unsafe_allow_html=True)
else:
    st.write("No activity recorded yet.")

st.markdown("<hr>", unsafe_allow_html=True)

st.header("🔍 Targeted Sentiment Explorer")
d_col1, d_col2 = st.columns(2)
with d_col1:
    sel_aspect = st.selectbox("Select Category:", list(ASPECT_KEYWORDS.keys()), index=0)
with d_col2:
    sel_sentiment = st.selectbox("Select Sentiment:", ["negative", "positive", "neutral"], index=0)

feedback_list = st.session_state.agg_data[sel_aspect][sel_sentiment]["reviews"]

if feedback_list:
    st.markdown(f"Found **{len(feedback_list)}** mentions regarding **{sel_aspect}**:")
    for f in reversed(feedback_list[-15:]):
        if sel_sentiment == "negative": st.error(f"• {f}")
        elif sel_sentiment == "positive": st.success(f"• {f}")
        else: st.info(f"• {f}")
else:
    st.write(f"No {sel_sentiment} feedback found for {sel_aspect} yet.")

st.markdown("""
    <div class="footer">
        <hr>
        Made by <b>Gummadi Likith</b> | 
        <a href="https://github.com/glikith/ABSA" target="_blank">View on GitHub</a>
    </div>
""", unsafe_allow_html=True)