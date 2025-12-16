import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data
movies = pd.read_csv("movies.csv")

# Combine genre and keywords for content-based similarity
movies['features'] = movies['genre'] + " " + movies['keywords']

# Convert text features into TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(movies['features'])

# Compute cosine similarity between movies
similarity = cosine_similarity(feature_matrix, feature_matrix)

# Function to get recommendations
def recommend(movie_title):
    if movie_title not in movies['title'].values:
        return "Movie not found in database."
    
    # Get the index of the movie
    idx = movies[movies['title'] == movie_title].index[0]
    
    # Get similarity scores for all movies
    scores = list(enumerate(similarity[idx]))
    
    # Sort by similarity score (highest first)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Get top 5 recommendations (excluding itself)
    top_movies = [movies.iloc[i[0]]['title'] for i in scores[1:6]]
    
    return top_movies

# Example usage
movie_name = input("Enter a movie you like: ")
recommendations = recommend(movie_name)

print("\nRecommended movies:")
if isinstance(recommendations, list):
    for m in recommendations:
        print("🎥", m)
else:
    print(recommendations)
