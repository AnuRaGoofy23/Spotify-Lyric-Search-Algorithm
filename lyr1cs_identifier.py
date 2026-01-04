import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

# Configuration
DATASET_PATH = 'lyrics_10k.csv'  # Switch to the larger dataset
SAMPLE_SIZE = 50000  # Number of songs to use (max)

def setup_nltk():
    """Download necessary NLTK data."""
    print("Setting up NLTK...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

def load_data(path):
    """Load and preprocess the dataset."""
    if not os.path.exists(path):
        print(f"Error: Dataset not found at {path}")
        print("Please ensure the 'lyrics_10k.csv' or 'spotify_millsongdata.csv' file is in the current directory.")
        return None

    print(f"Loading dataset from {path}...")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    # Column mapping to standardize names
    # lyrics_10k.csv has 'artists', 'song', 'lyrics'
    # spotify_millsongdata.csv has 'artist', 'song', 'text'
    column_mapping = {
        'artists': 'artist',
        'lyrics': 'text',
        'song_name': 'song', # potential variation
        'track_name': 'song' # potential variation
    }
    df.rename(columns=column_mapping, inplace=True)

    # Check for required columns
    required_cols = ['artist', 'song', 'text']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print("Available columns:", df.columns)
        return None

    print(f"Dataset loaded. Shape: {df.shape}")
    
    # Check if dataset is dangerously small
    if len(df) < 100:
        print("\n[WARNING] Dataset is very small (fewer than 100 songs).")
        print("Search results might be poor or irrelevant.")
        print("Please ensure you have a full dataset file.\n")
    
    # Sample if too large to speed up demo
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"Sampled down to {SAMPLE_SIZE} songs for performance.")

    return df

def preprocess_text(text):
    """Clean and preprocess lyrics text."""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Replace hyphens with spaces to preserve words in "jewel-studded" etc.
    text = text.replace('-', ' ')
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Fallback if punk not found (though setup_nltk should handle it)
        tokens = text.split()
        
    # Remove stopwords and non-alphanumeric characters
    # Get stopwords once or use global if performance is critical, but this is fine for now
    stop_words = set(stopwords.words('english'))
    
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    return " ".join(filtered_tokens)

def build_model(df):
    """Build TF-IDF model."""
    print("Building TF-IDF model...")
    # Use our processed text, so we can turn off internal preprocessing/tokenization if we wanted,
    # but keeping default settings (except stop_words since we did it) is often fine.
    # However, since we already removed stopwords, we can remove that arg or keep it as safety.
    vectorizer = TfidfVectorizer(
        max_features=10000, 
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix

def search_song(query, vectorizer, tfidf_matrix, df, top_n=3):
    """Search for a song using a lyrics snippet."""
    print(f"\nSearching for snippet: '{query}'")
    
    # Preprocess query
    processed_query = preprocess_text(query)
    
    query_vec = vectorizer.transform([processed_query])
    
    # Calculate similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top N indices
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        score = similarities[idx]
        song = df.iloc[idx]
        results.append({
            'artist': song['artist'],
            'song': song['song'],
            'score': score,
            'snippet': song['text'][:100] + "..." # Preview
        })
        
    return results

def main():
    setup_nltk()
    
    df = load_data(DATASET_PATH)
    if df is None:
        return

    # Basic cleaning
    print("Preprocessing data (this might take a moment)...")
    df['text'] = df['text'].astype(str)
    # Apply preprocessing to create the column expected by build_model
    df['processed_text'] = df['text'].apply(preprocess_text)

    vectorizer, tfidf_matrix = build_model(df)
    
    print("\nModel ready!")
    
    # Interactive loop
    print("\n--- Lyrics Search Engine ---")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            query = input("\nEnter lyrics snippet: ")
            if query.lower().strip() in ['exit', 'quit']:
                break
            if not query.strip():
                continue
                
            results = search_song(query, vectorizer, tfidf_matrix, df)
            
            found_match = False
            for i, res in enumerate(results):
                if res['score'] > 0:
                    found_match = True
                    print(f"{i+1}. {res['song']} by {res['artist']} (Score: {res['score']:.4f})")
            
            if not found_match:
                print("No matches found.")
        except KeyboardInterrupt:
            break
            
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
