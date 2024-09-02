import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
movies = pd.read_csv('movies_metadata.csv', low_memory=False)

# Select relevant columns
movies = movies[['title', 'overview']]

# Fill missing values in 'overview' with an empty string
movies['overview'] = movies['overview'].fillna('')

# Create a TF-IDF Vectorizer to convert the text data into feature vectors
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the data
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping of movie titles to indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend_movies(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies['title'].iloc[movie_indices]

# Example: Recommend movies similar to 'The Dark Knight'
print(recommend_movies('The Dark Knight'))