import streamlit as st
import pandas as pd
import praw
import re
import emoji
import nltk
import matplotlib.pyplot as plt
import torch
import time
import json
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Reddit Sentiment Pro", layout="wide")

# ===================================
# ULTRA MODERN CSS (FULL SCREEN BUBBLES & WHITE METRICS)
# ===================================
st.markdown("""
<style>
/* Arka Plan */
.stApp {
    background: #0f172a !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(147, 51, 234, 0.15) 0px, transparent 50%) !important;
    color: #f8fafc !important;
}

/* TÜM EKRANA YAYILAN BALONCUKLAR */
.bg-bubbles {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    z-index: 0; pointer-events: none;
}
.bg-bubbles div {
    position: absolute; list-style: none; display: block;
    width: 20px; height: 20px; background-color: rgba(255, 255, 255, 0.1);
    bottom: -160px; border-radius: 50%; animation: bubbleUp 25s infinite linear;
}

/* Baloncuk Dağılımı */
.bg-bubbles div:nth-child(1) { left: 5%; width: 80px; height: 80px; animation-delay: 0s; }
.bg-bubbles div:nth-child(2) { left: 15%; width: 30px; height: 30px; animation-delay: 2s; animation-duration: 17s; }
.bg-bubbles div:nth-child(3) { left: 25%; width: 50px; height: 50px; animation-delay: 4s; }
.bg-bubbles div:nth-child(4) { left: 35%; width: 90px; height: 90px; animation-duration: 22s; background-color: rgba(255, 255, 255, 0.15); }
.bg-bubbles div:nth-child(5) { left: 50%; width: 40px; height: 40px; animation-delay: 1s; }
.bg-bubbles div:nth-child(6) { left: 65%; width: 120px; height: 120px; animation-delay: 3s; }
.bg-bubbles div:nth-child(7) { left: 75%; width: 25px; height: 25px; animation-delay: 7s; }
.bg-bubbles div:nth-child(8) { left: 85%; width: 60px; height: 60px; animation-delay: 5s; animation-duration: 18s; }
.bg-bubbles div:nth-child(9) { left: 95%; width: 45px; height: 45px; animation-delay: 2s; }

@keyframes bubbleUp {
    0% { transform: translateY(0) rotate(0deg); opacity: 0; }
    20% { opacity: 0.3; }
    100% { transform: translateY(-1300px) rotate(600deg); opacity: 0; }
}

/* KUTUCUK (CARD) YAPISI */
div[data-testid="stMetric"], .stDataFrame, div.stButton, .css-1r6p78m {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    padding: 15px !important;
    z-index: 10;
}

/* Dashboard Başlığı */
.white-header { color: white !important; font-weight: bold !important; margin-bottom: 15px; }

/* METRİKLERİ BEYAZ YAPMA */
[data-testid="stMetricLabel"] p { color: white !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] div { color: white !important; font-size: 2rem !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"] div { color: white !important; }
[data-testid="stMetricDelta"] svg { fill: white !important; }

/* Başlık */
.main-title {
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #60a5fa, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-top: 20px;
}
.sub-title { text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 30px; }

/* Buton */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    color: white !important; border-radius: 12px !important;
    font-weight: 700 !important; width: 100% !important; height: 50px !important;
}
</style>

<div class="bg-bubbles">
    <div></div><div></div><div></div><div></div><div></div>
    <div></div><div></div><div></div><div></div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD ASSETS
# -----------------------------------
@st.cache_resource
def load_assets():
    nltk.download('stopwords')
    nltk.download('wordnet')
    sent_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    emot_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
    return sent_model, emot_model

sent_pipe, emot_pipe = load_assets()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'http\S+|[^a-zA-Z\s]', ' ', str(text).lower())
    return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2])

EMOJI_MAP = {
    "joy": "😂", "anger": "🔥", "fear": "😰", "sadness": "🌊", 
    "surprise": "✨", "disgust": "🍄", "neutral": "💎", "love": "❤️"
}

# -----------------------------------
# UI
# -----------------------------------
st.markdown('<h1 class="main-title">🚀 Reddit Sentiment Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced NLP Analytics Dashboard</p>', unsafe_allow_html=True)

col_url_1, col_url_2, col_url_3 = st.columns([1, 2, 1])
with col_url_2:
    post_url = st.text_input("", placeholder="Paste Reddit URL here...")
    analyze_btn = st.button("Start Analysis")

# -----------------------------------
# LOGIC
# -----------------------------------
reddit = praw.Reddit(
    client_id="BGsm61Dv8hM14-5pLarLmQ",
    client_secret="zBk4n0vyucPgDsRnZpieiZ4oqF2-jA",
    user_agent="ai-project by u/Altruistic_Pitch_783",
)

if analyze_btn:
    if not post_url:
        st.warning("Please provide a URL.")
    else:
        with st.spinner("🔍 Scanning Community Insights..."):
            try:
                submission = reddit.submission(url=post_url)
                submission.comments.replace_more(limit=0)
                comments = [c.body for c in submission.comments.list() if c.body not in ["[deleted]", "[removed]"]]
                
                df = pd.DataFrame(comments, columns=["Comment"])
                df["Cleaned"] = df["Comment"].apply(clean_text)
                df = df[df["Cleaned"].str.len() > 10].reset_index(drop=True)

                sent_results = sent_pipe(df["Cleaned"].tolist())
                emot_results = emot_pipe(df["Cleaned"].tolist())
                
                df["Sentiment"] = [r["label"].capitalize() for r in sent_results]
                df["Confidence"] = [round(r["score"]*100, 1) for r in sent_results]
                df["Emotion"] = [r[0]["label"] for r in emot_results]

                # --- DASHBOARD ---
                st.markdown('<h3 class="white-header">📊 Deep Insights Dashboard</h3>', unsafe_allow_html=True)
                
                counts = df["Sentiment"].value_counts()
                emot_counts = df["Emotion"].value_counts().head(6)
                
                m1, m2, m3 = st.columns(3)
                # Metriklerin yanına okları manuel ekledik çünkü CSS bazen ok renklerini eziyor
                m1.metric("Positive", f"{counts.get('Positive', 0)}", "↑")
                m2.metric("Negative", f"{counts.get('Negative', 0)}", "↓")
                m3.metric("Neutral", f"{counts.get('Neutral', 0)}", "↔")

                st.markdown("<br>", unsafe_allow_html=True)
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("<p style='font-weight: bold; color: white;'>🎯 Sentiment Share</p>", unsafe_allow_html=True)
                    fig1, ax1 = plt.subplots(figsize=(6, 5))
                    fig1.patch.set_alpha(0)
                    colors = ['#3b82f6', '#ef4444', '#94a3b8']
                    ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=colors,
                            pctdistance=0.75, textprops={'color':"w", 'weight':'bold'},
                            wedgeprops={'width': 0.4, 'edgecolor': '#0f172a', 'linewidth': 2})
                    st.pyplot(fig1)

                with col_right:
                    st.markdown("<p style='font-weight: bold; color: white;'>🎭 Emotional Tone</p>", unsafe_allow_html=True)
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    fig2.patch.set_alpha(0)
                    ax2.set_facecolor('none')
                    
                    labels = [f"{EMOJI_MAP.get(e, '❓')} {e.capitalize()}" for e in emot_counts.index]
                    emot_colors = ['#a855f7', '#ec4899', '#f43f5e', '#fb923c', '#facc15', '#22d3ee']
                    
                    bars = ax2.barh(labels, emot_counts.values, color=emot_colors, height=0.7)
                    ax2.set_xticks([]); ax2.invert_yaxis()
                    for spine in ax2.spines.values(): spine.set_visible(False)
                    for bar in bars:
                        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                                f'{int(bar.get_width())}', va='center', color='white', fontweight='bold')
                    ax2.tick_params(axis='y', colors='white', labelsize=11)
                    st.pyplot(fig2)

                st.markdown('<h3 class="white-header">📝 Insights Library</h3>', unsafe_allow_html=True)
                st.dataframe(df[["Comment", "Sentiment", "Emotion", "Confidence"]].head(25), use_container_width=True)

            except Exception as e:
                st.error(f"Analysis failed: {e}")