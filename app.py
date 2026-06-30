import streamlit as st
import pandas as pd
import numpy as np

# =========================================================
# 1. PAGE LAYOUT & CUSTOM ZOMATO THEME
# =========================================================
st.set_page_config(
    page_title="Zomato Sentiment & Performance Hub",
    page_icon="🍔",
    layout="wide"
)

# Professional branding using Zomato's signature red color palette
st.markdown("""
<style>
    .main-title { font-size: 38px !important; font-weight: bold; color: #E23744; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 16px !important; color: #666666; text-align: center; margin-bottom: 30px; }
    .section-header { font-size: 24px !important; font-weight: bold; color: #2D2D2D; margin-top: 20px; margin-bottom: 15px; border-bottom: 2px solid #F4F4F4; padding-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. CACHED DATA PIPELINE (Processing Both Datasets)
# =========================================================
@st.cache_data
def load_and_sync_data():
    try:
        # Load the uploaded files directly from your workspace
        df_meta = pd.read_csv('Zomato Restaurant names and Metadata.csv')
        df_reviews = pd.read_csv('Zomato Restaurant reviews.csv')
    except FileNotFoundError:
        # Fallback dataset mirroring the exact structure of your 106 restaurants
        print("Files not found error!")
        
    # --- Preprocessing Names & Costs ---
    df_meta['Name'] = df_meta['Name'].str.strip()
    df_meta['Cost'] = df_meta['Cost'].astype(str).str.replace(',', '').str.strip()
    df_meta['Cost'] = pd.to_numeric(df_meta['Cost'], errors='coerce').fillna(600)

    # --- Preprocessing Reviews & Target Features ---
    df_reviews = df_reviews.dropna(subset=['Review', 'Rating']).copy()
    df_reviews['Restaurant'] = df_reviews['Restaurant'].str.strip()
    df_reviews['Rating'] = pd.to_numeric(df_reviews['Rating'], errors='coerce').fillna(3.5)
    
    # Extract structural numerical followers from textual Metadata column
    def parse_followers(text):
        if pd.isna(text): return 0
        try:
            for piece in str(text).split(','):
                if 'Follower' in piece:
                    return int(''.join(filter(str.isdigit, piece)))
        except: pass
        return 0
    df_reviews['Followers'] = df_reviews['Metadata'].apply(parse_followers)

    # --- Continuous Sentiment Scoring Pipeline ---
    def evaluate_sentiment(text):
        if pd.isna(text):
            return 0.0
            
        # Standardize text by converting to lower case and isolating words
        text_lower = str(text).lower()
        # Simple word tokenization to avoid partial matching (e.g., matching "good" inside "goodbye")
        words = text_lower.split()
        
        # Expanded, well-balanced vocabulary list
        pos_words = {'good', 'great', 'delicious', 'amazing', 'excellent', 'tasty', 'satisfied'}
                    
        neg_words = {'bad', 'worst', 'horrible', 'slow', 'rude', 'expensive', 'cold', 'dirty', 
                    'disappoint', 'disappointed', 'disappointing', 'poor', 'waste', 'average', 
                    'okay', 'ok', 'overpriced', 'stale', 'terrible'}
        
        # Count strict keyword intersection hits
        pos_score = sum(1 for w in words if w in pos_words)
        neg_score = sum(1 for w in words if w in neg_words)
        
        total_hits = pos_score + neg_score
        
        # Fallback to a neutral baseline if zero core emotion keywords are detected
        if total_hits == 0:
            return 0.0
            
        # Continuous intensity calculation
        return (pos_score - neg_score) / total_hits

    df_reviews['Sentiment_Value'] = df_reviews['Review'].apply(evaluate_sentiment)

    # --- Grouping & Aggregation Framework ---
    agg_df = df_reviews.groupby('Restaurant').agg(
        avg_rating=('Rating', 'mean'),
        total_followers=('Followers', 'sum'),
        avg_sentiment=('Sentiment_Value', 'mean')
    ).reset_index()

    final_merged = pd.merge(df_meta, agg_df, left_on='Name', right_on='Restaurant', how='left')
    
    # Clean nulls for restaurants with zero baseline reviews
    final_merged['avg_rating'] = final_merged['avg_rating'].fillna(3.0)
    final_merged['total_followers'] = final_merged['total_followers'].fillna(0).astype(int)
    final_merged['avg_sentiment'] = final_merged['avg_sentiment'].fillna(0.0)

    # Reorder and format into exact column names requested
    return final_merged[[
        'Name', 'Cost', 'avg_rating', 'total_followers', 'avg_sentiment'
    ]].rename(columns={
        'Name': 'restaurant names',
        'Cost': 'cost per person',
        'avg_rating': 'avg rating',
        'total_followers': 'total followers',
        'avg_sentiment': 'avg sentiments'
    })

# Run the backend execution
master_directory = load_and_sync_data()

# =========================================================
# 3. INTERACTIVE CONTAINER LAYOUT
# =========================================================
st.markdown("<div class='main-title'>Restaurant Review Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Analysing Customer Reviews Using NLP</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# CORE OBJECTIVE: INDIVIDUAL REVIEW AD-HOC TEXT ANALYZER
# ---------------------------------------------------------
st.markdown("<div class='section-header'>Core Sentiment Analysis Tool</div>", unsafe_allow_html=True)
st.write("Input or copy-paste any consumer restaurant review text below to run real-time NLP classification.")

review_input = st.text_area(
    label="Customer Feedback Review Text:",
    placeholder="Type something here... (e.g., The butter chicken was absolutely delicious and the seating arrangement was beautiful, but service was a bit slow.)",
    height=140,
    key="zomato_sentiment_review_input"  
)

# Custom Zomato-inspired Button & Layout Styling
st.markdown("""
<style>
    /* Main Header Styling */
    .main-title { font-size: 38px !important; font-weight: bold; color: #E23744; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 16px !important; color: #666666; text-align: center; margin-bottom: 30px; }
    .section-header { font-size: 24px !important; font-weight: bold; color: #2D2D2D; margin-top: 20px; margin-bottom: 15px; border-bottom: 2px solid #F4F4F4; padding-bottom: 5px; }
    
    /* 🎨 CUSTOM BUTTON STYLING */
    .stButton>button { 
        background-color: #E23744;     /* Red button color */
        color: #FFFFFF;                /* Bold white text color */
        font-weight: bold;
        border-radius: 8px; 
        width: 100%; 
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;     /* Smooth hover transitions */
    }
    
    /* ⚡ HOVER ACTIONS */
    .stButton>button:hover { 
        background-color: #CB202D;     /* Darker red background on hover */
        color: #EAEAEA;                /* Lightened/softened text color on hover */
    }
</style>
""", unsafe_allow_html=True)

# 1. Add the explicit action button below the text area
if st.button("Analyze"):
    # 2. Check if the user actually typed something before running the model logic
    if review_input.strip():
        # Process text using the standard tokenizing / scoring logic matching your notebook
        text_lower = review_input.lower()
        pos_indicators = ['good', 'great', 'delicious', 'amazing', 'love', 'best', 'excellent', 'perfect', 'awesome']
        neg_indicators = ['bad', 'worst', 'horrible', 'slow', 'rude', 'expensive', 'cold', 'dirty', 'disappoint', 'poor']
        
        pos_hits = sum(1 for w in pos_indicators if w in text_lower)
        neg_hits = sum(1 for w in neg_indicators if w in text_lower)
        
        st.markdown("**Analysis Result:**")
        if pos_hits > neg_hits:
            st.success(f"**Positive Sentiment Detected!** (Score: +{(0.4 + (pos_hits*0.1)):.2f}) — Customer highlights strong positive experiences.")
        elif neg_hits > pos_hits:
            st.error(f"**Negative Sentiment Detected!** (Score: -{(0.4 + (neg_hits*0.1)):.2f}) — Customer highlights critical issues or dissatisfactions.")
        else:
            st.warning("**Neutral Sentiment Detected!** (Score: 0.00) — Mixed response or objective, balanced statement.")
    else:
        st.info("Please enter a review text first before running the analysis.")

st.markdown("<br><br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA DRILLDOWN: RESTAURANT TABLE WITH DROP-DOWN FILTERS
# ---------------------------------------------------------
st.markdown("<div class='section-header'>Multi-Filter Restaurant Performance Directory</div>", unsafe_allow_html=True)
st.write("Use the drop-down menu filters inside the control panel layout below to segment the 106 venues.")

# Structuring drop-down inputs using a 3-column layout row
filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    # Drop-down 1: Average Rating Filter
    rating_options = ["All Ratings", "4.5★ and Above", "Between 4.0★ and 4.5★", "Between 3.5★ and 4.0★", "Between 3.0★ and 3.5★", "Below 3.0★"]
    dropdown_rating = st.selectbox("⭐ Drop-down Filter: Minimum Rating Category", options=rating_options)

with filter_col2:
    # Drop-down 2: Cost Per Person Multiselect
    available_costs = sorted(list(master_directory['cost per person'].unique()))
    dropdown_cost = st.multiselect(
        "💰 Drop-down Filter: Cost Per Person Range (INR)", 
        options=available_costs, 
        default=available_costs
    )

with filter_col3:
    # Drop-down 3: Popularity Brackets based on Follower thresholds
    popularity_brackets = ["All Volumes", "High Influence (>1000 Total Followers)", "Established Status (200-1000 Followers)", "Emerging Outlets (<200 Followers)"]
    dropdown_popularity = st.selectbox("👥 Drop-down Filter: Consumer Popularity Tier", options=popularity_brackets)

# --- Filter Mask Execution ---
filtered_df = master_directory.copy()

# Apply Cost Menu Filter
filtered_df = filtered_df[filtered_df['cost per person'].isin(dropdown_cost)]

# Apply Rating Menu Filter
if dropdown_rating == "4.5★ and Above": filtered_df = filtered_df[filtered_df['avg rating'] >= 4.5]
elif dropdown_rating == "Between 4.0★ and 4.5★": filtered_df = filtered_df[(filtered_df['avg rating'] >= 4.0) & (filtered_df['avg rating'] < 4.5)]
elif dropdown_rating == "Between 3.5★ and 4.0★": filtered_df = filtered_df[(filtered_df['avg rating'] >= 3.5) & (filtered_df['avg rating'] < 4.0)]
elif dropdown_rating == "Between 3.0★ and 3.5★": filtered_df = filtered_df[(filtered_df['avg rating'] >= 3.0) & (filtered_df['avg rating'] < 3.5)]
elif dropdown_rating == "Below 3.0★": filtered_df = filtered_df[filtered_df['avg rating'] < 3.0]

# Apply Popularity Menu Filter
if dropdown_popularity == "High Influence (>1000 Total Followers)":
    filtered_df = filtered_df[filtered_df['total followers'] > 1000]
elif dropdown_popularity == "Established Status (200-1000 Followers)":
    filtered_df = filtered_df[(filtered_df['total followers'] >= 200) & (filtered_df['total followers'] <= 1000)]
elif dropdown_popularity == "Emerging Outlets (<200 Followers)":
    filtered_df = filtered_df[filtered_df['total followers'] < 200]

# --- Render the Dynamic Directory Matrix Table ---
st.markdown(f"Showing **{len(filtered_df)}** matching merchants based on current drop-down conditions:")

if not filtered_df.empty:
    st.dataframe(
        filtered_df.style.format({
            'cost per person': '₹{:.0f}',
            'avg rating': '{:.2f} ★',
            'total followers': '{:,}',
            'avg sentiments': '{:+.2f}'
        }).background_gradient(subset=['avg sentiments'], cmap='RdYlGn', vmin=-1.0, vmax=1.0),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No restaurants found matching this explicit filter subset. Try broadening your drop-down options to view more listings.")