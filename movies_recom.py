import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("D:/DS/ML/ML Dataset/movies_content.csv")
    df = df[['name', 'description']].dropna().drop_duplicates().reset_index(drop=True)
    df['name'] = df['name'].str.lower()
    return df

df = load_data()

# TF-IDF Vectorizer and Nearest Neighbors Model
@st.cache_data
def train_model(data):
    tv = TfidfVectorizer(stop_words="english", lowercase=True)
    vectors = tv.fit_transform(data.description).toarray()
    model = NearestNeighbors(metric="cosine")
    model.fit(vectors)
    return tv, vectors, model

tv, vectors, model = train_model(df)

# Streamlit App
st.title("🎬 Movie Recommendation System")
st.write("Find similar movies based on their descriptions!")

# Sidebar information
st.sidebar.title("About")
st.sidebar.write("**Manik Kamboj**")
st.sidebar.write("📞 +91-9996541776")

# User input for movie name via dropdown
movie_name = st.selectbox("Select a movie", options=df['name'].tolist())

if movie_name:
    # Find recommendations
    index = df[df.name == movie_name].index[0]
    distances, indexes = model.kneighbors([vectors[index]], n_neighbors=5)
    
    st.success(f"Movies similar to **{movie_name}**:")
    for i in indexes[0][1:]:
        st.write(f"- {df.iloc[i]['name'].title()}")

# Option to display the dataset
if st.checkbox("Show dataset"):
    st.write(df)
