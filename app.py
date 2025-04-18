import streamlit as st
import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob
from collections import Counter
from pymongo import MongoClient
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# MongoDB setup
client = MongoClient("mongodb+srv://vgugan16:gugan2004@cluster0.qyh1fuo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["flask_app"]
queries_collection = db["queries"]
db2 = client["Cloud"]
courses_collection = db2["kn"]

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load course data
courses_cursor = courses_collection.find()
courses_df = pd.DataFrame(list(courses_cursor))

# Clean and format DataFrame
if not courses_df.empty:
    courses_df.columns = courses_df.columns.astype(str).str.strip()
    if '_id' in courses_df.columns:
        courses_df.drop(columns=['_id'], inplace=True)

    # Rename for consistency
    courses_df.rename(columns={
        'Name Of The Course': 'Name Of The Course',
        'Description': 'Description',
        'Institute': 'Institute',
        'Link': 'Link'
    }, inplace=True)

# Keyword mappings
keyword_mappings = {
    'ai': 'artificial intelligence',
    'ml': 'machine learning',
    'aiml': 'artificial intelligence and machine learning'
}

# Vectorizer and classifier setup
vectorizer = TfidfVectorizer()

def preprocess_data(df):
    df.columns = df.columns.astype(str).str.strip()  # Ensure all column names are strings
    if 'Name Of The Course' in df.columns and 'Description' in df.columns:
        df['Name Of The Course'] = df['Name Of The Course'].astype(str).str.lower().str.strip()
        df['Description'] = df['Description'].astype(str).str.lower().str.strip()
        df['Label'] = df['Description'].apply(lambda x: 1 if 'machine learning' in x else 0)
    else:
        st.error("❌ Required columns 'Name Of The Course' or 'Description' not found in the dataset.")
        st.stop()
    return df

def train_classifier(df):
    X = vectorizer.fit_transform(df['Description'].fillna(''))
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    return clf

# Prepare data
courses_df = preprocess_data(courses_df)
classifier = train_classifier(courses_df)

# Keyword extraction function
def extract_noun_phrases(text):
    text = text.lower()
    for abbr, full in keyword_mappings.items():
        text = text.replace(abbr, full)
    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    phrase_counts = Counter(noun_phrases)

    filtered = []
    for phrase in phrase_counts:
        if "certificate" not in phrase:
            parts = [part.strip() for part in phrase.split('and')]
            filtered.extend(parts)

    tokens = [
        token.text for token in doc
        if token.pos_ in ["NOUN", "PROPN", "ADJ"] and "certificate" not in token.text
    ]
    named_entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT", "EVENT"]]
    return list(set(filtered + tokens + named_entities))

# Course recommendation logic
def recommend_courses(keywords):
    if not keywords:
        return pd.DataFrame()
    input_vector = vectorizer.transform([' '.join(keywords)])
    predicted = classifier.predict(input_vector)

    if predicted[0] == 1:
        recs = courses_df[courses_df['Description'].str.contains('machine learning', na=False)]
    else:
        sim_scores = cosine_similarity(input_vector, vectorizer.transform(courses_df['Description'].fillna(''))).flatten()
        top_indices = np.argsort(sim_scores)[-10:][::-1]
        recs = courses_df.iloc[top_indices]

    return recs[['Name Of The Course', 'Description', 'Institute', 'Link']].drop_duplicates()

# Save user query and recommendations
def save_to_mongo(query, recommendations):
    query = query.lower().strip()
    if not queries_collection.find_one({"query": query}):
        record = {
            "query": query,
            "recommendations": recommendations,
            "timestamp": datetime.now()
        }
        queries_collection.insert_one(record)

# NLP and syntax analysis display
def display_syntax(text):
    doc = nlp(text)
    st.subheader("🔍 Syntax and Semantics")
    st.write(f"**Original Text:** {text}")

    st.write("**Part of Speech Tagging:**")
    for token in doc:
        st.markdown(f"`{token.text}` → POS: {token.pos_}, Lemma: {token.lemma_}, Dependency: {token.dep_}")

    st.write("**Named Entities:**")
    for ent in doc.ents:
        st.markdown(f"`{ent.text}` → {ent.label_}")

    st.write("**Noun Phrases:**")
    for chunk in doc.noun_chunks:
        st.markdown(f"- {chunk.text}")

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    sentiment = "Positive" if polarity >= 0 else "Negative"
    st.write("**Sentiment Analysis:**")
    st.markdown(f"Polarity: `{polarity:.2f}` | Subjectivity: `{subjectivity:.2f}` → **{sentiment}**")

# Streamlit UI
st.title("📚 Course Recommendation & NLP Assistant")

# Main UI
tab_option = st.radio("Choose a feature:", ["🧠 Course Recommender"])

if tab_option == "🧠 Course Recommender":
    st.subheader("What topic are you curious about?")
    user_topic = st.text_input("Enter topic", placeholder="e.g. data science in agriculture")

    if st.button("Get Recommendations"):
        if user_topic.strip():
            display_syntax(user_topic)
            keywords = extract_noun_phrases(user_topic)
            recs = recommend_courses(keywords)

            if not recs.empty:
                st.success("🎯 Recommended Courses:")
                for _, row in recs.iterrows():
                    st.markdown(f"### 🔹 {row['Name Of The Course']}")
                    st.write(f"**Description:** {row['Description']}")
                    st.write(f"**Institute:** {row['Institute']}")
                    st.markdown(f"[📎 Course Link]({row['Link']})\n")
                save_to_mongo(user_topic, recs.to_dict("records"))
            else:
                st.warning("😕 Sorry, we couldn't find relevant courses for your topic.")
