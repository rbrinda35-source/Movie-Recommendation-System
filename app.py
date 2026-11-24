import streamlit as st

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #10101c, #191329);
    color: white;
}

/* Main content width */
.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

/* Headings */
h1, h2, h3 {
    color: #ffffff !important;
}

/* Movie cards */
.movie-card {
    background: #252137;
    border: 1px solid #40365d;
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 15px;
    min-height: 125px;
}

/* Movie title */
.movie-title {
    font-size: 18px;
    font-weight: 600;
    color: #ffffff;
}

/* Similarity score */
.movie-score {
    color: #c4a5ff;
    font-size: 14px;
    margin-top: 12px;
}

/* Small subtitle */
.subtitle {
    color: #aaa4bd;
    font-size: 16px;
}

/* Button */
.stButton > button {
    background: #8b5cf6;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 25px;
    font-weight: 600;
    width: 100%;
}

.stButton > button:hover {
    background: #7042d8;
    color: white;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# Load the movies dataset
movies = pd.read_csv('dataset/movies.csv')

# Load the tags dataset
tags = pd.read_csv('dataset/tags.csv')

# Combine tags for each movie
movie_tags = tags.groupby('movieId')['tag'].apply(
    lambda x: ' '.join(x.dropna().astype(str).unique())
).reset_index()

# Merge movies and tags
movies = movies.merge(movie_tags, on='movieId', how='left')  # left join to keep all movies even if they don't have tags

# Replace missing tags with an empty string
movies['tag'] = movies['tag'].fillna('')

# clean genres
movies['genres'] = movies['genres'].replace('(no genres listed)', '')

# Combine genres and tags into a single feature
movies['combined_features'] = (
    movies['genres'].str.replace('|', ' ', regex=False)
    + ' ' + movies['tag']
).str.lower().str.strip()

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Convert movie combined features into a TF-IDF matrix vector
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['combined_features'])

print("TF-IDF matrix shape:", tfidf_matrix.shape)

# Calculate the cosine similarity matrix between all movies
similarity_matrix = cosine_similarity(tfidf_matrix)


# ==============================
# CINEMATCH USER INTERFACE
# ==============================

# Header
st.markdown(
    "<h1 style='text-align:center;'>🎬 CineMatch</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle' style='text-align:center;'>"
    "Discover your next favourite movie"
    "</p>",
    unsafe_allow_html=True
)

st.write("")

# Movie selection
st.subheader("🍿 Find Your Next Movie")

selected_movie = st.selectbox(
    "Choose a movie you love",
    movies['title'].tolist()
)

# Recommendation button
if st.button("✨ Discover Similar Movies"):

    movie_index = movies[
        movies['title'] == selected_movie
    ].index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    sorted_movies = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove selected movie
    sorted_movies = [
        movie for movie in sorted_movies
        if movie[0] != movie_index
    ]

    # Store recommendations
    st.session_state['recommendations'] = sorted_movies[:5]
    st.session_state['selected_movie'] = selected_movie


# Display recommendations
if 'recommendations' in st.session_state:

    st.divider()

    st.subheader(
        f"🎥 Movies similar to {st.session_state['selected_movie']}"
    )

    recommendations = st.session_state['recommendations']

    # Display two movie cards per row
    for i in range(0, len(recommendations), 2):

        cols = st.columns(2)

        for j, col in enumerate(cols):

            if i + j < len(recommendations):

                movie_index, score = recommendations[i + j]

                movie_title = movies.iloc[movie_index]['title']

                with col:

                    st.markdown(
                        f"""
                    <div class="movie-card">
                        <div class="movie-title">
                            🎬 {movie_title}
                        </div>
                        <div class="movie-score">
                            Similarity Score: {score:.4f}
                        </div>
                    </div>  
                    """,
                     unsafe_allow_html=True
                    )


# Footer
st.markdown(
    "<p style='text-align:center;color:#777;margin-top:50px;'>"
    "Built with Python | Machine Learning | Streamlit"
    "</p>",
    unsafe_allow_html=True
)