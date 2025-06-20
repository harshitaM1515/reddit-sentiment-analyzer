import streamlit as st
import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------- Reddit Auth ----------
reddit = praw.Reddit(
    client_id="rYJHvdLxUNPf0kf3HnUkZA",
    client_secret="nkyai6O6dT9OkrKSogCxHn61B6j7uQ",
    user_agent="streamlit:test:v1.0 (by u/	PlentySad4623)"
)

# ---------- Sentiment Functions ----------
def analyze_vader(text):
    return SentimentIntensityAnalyzer().polarity_scores(text)['compound']

def analyze_textblob(text):
    return TextBlob(text).sentiment.polarity

def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def generate_wordcloud(texts, title):
    text = ' '.join(texts)
    wordcloud = WordCloud(background_color='black', colormap='viridis', width=800, height=400).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Reddit Sentiment Dashboard", layout="wide", page_icon="👽")
st.title("👽 Reddit Sentiment Dashboard")

choice = st.radio("Choose input mode:", ["Search keyword", "Subreddit"])
query = st.text_input("Enter your keyword or subreddit:")
post_limit = st.slider("Number of posts to analyze", 10, 200, 50)

if st.button("Analyze"):
    if not query:
        st.warning("Please enter a value.")
    else:
        with st.spinner("Fetching posts..."):
            posts = []

            if choice == "Search keyword":
                for submission in reddit.subreddit("all").search(query, limit=post_limit):
                    posts.append(submission.title + " " + (submission.selftext or ""))
            else:
                for submission in reddit.subreddit(query).hot(limit=post_limit):
                    posts.append(submission.title + " " + (submission.selftext or ""))

        df = pd.DataFrame(posts, columns=['Post'])
        df['VADER Score'] = df['Post'].apply(analyze_vader)
        df['TextBlob Score'] = df['Post'].apply(analyze_textblob)
        df['Sentiment'] = df['VADER Score'].apply(classify_sentiment)

        st.success(f"Fetched and analyzed {len(df)} posts.")

        st.dataframe(df)

        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = df['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

        st.subheader("☁ WordClouds")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Positive Posts**")
            generate_wordcloud(df[df['Sentiment'] == 'Positive']['Post'], "Positive")
        with col2:
            st.markdown("**Negative Posts**")
            generate_wordcloud(df[df['Sentiment'] == 'Negative']['Post'], "Negative")
