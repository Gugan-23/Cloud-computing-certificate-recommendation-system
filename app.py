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

# Load data and models
nlp = spacy.load("en_core_web_sm")
courses_df = pd.read_csv("alison.csv", encoding="ISO-8859-1")
courses_df.columns = courses_df.columns.str.strip()

keyword_mappings = {
    'ai': 'artificial intelligence',
    'ml': 'machine learning',
    'aiml': 'artificial intelligence and machine learning'
}

vectorizer = TfidfVectorizer()

def preprocess_data(df):
    df['Name Of The Course'] = df['Name Of The Course'].str.lower().str.strip()
    df['Description'] = df['Description'].str.lower().str.strip()
    df['Label'] = df['Description'].apply(lambda x: 1 if 'machine learning' in x else 0)
    return df

def train_classifier(df):
    X = vectorizer.fit_transform(df['Description'].fillna(''))
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    return clf

courses_df = preprocess_data(courses_df)
classifier = train_classifier(courses_df)

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

def save_to_mongo(query, recommendations):
    query = query.lower().strip()
    if not queries_collection.find_one({"query": query}):
        record = {
            "query": query,
            "recommendations": recommendations,
            "timestamp": datetime.now()
        }
        queries_collection.insert_one(record)

def display_syntax(text):
    doc = nlp(text)
    st.subheader("Syntax and Semantics")
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

# Load transformers

def remove_repeats(text):
    words = text.split()
    seen = set()
    return " ".join([word for word in words if not (word.lower() in seen or seen.add(word.lower()))])

# Streamlit UI
# Streamlit UI
st.title("📚 Course Recommendation & NLP Assistant")

# Create tabs using radio buttons
tab_option = st.radio("Choose a tab:", ["🧠 Course Recommender"])

if tab_option == "🧠 Course Recommender":
    st.subheader("Enter a topic you're interested in:")
    user_topic = st.text_input("Topic", placeholder="e.g. machine learning for healthcare")

    if st.button("Get Recommendations"):
        if user_topic.strip():
            display_syntax(user_topic)
            keywords = extract_noun_phrases(user_topic)
            recs = recommend_courses(keywords)

            if not recs.empty:
                st.success("Here are some recommended courses:")
                for _, row in recs.iterrows():
                    st.markdown(f"**{row['Name Of The Course']}**")
                    st.write(f"{row['Description']}")
                    st.write(f"Institution: {row['Institute']}")
                    st.markdown(f"[Course Link]({row['Link']})\n")
                save_to_mongo(user_topic, recs.to_dict("records"))
            else:
                st.warning("No courses found for your input.")
