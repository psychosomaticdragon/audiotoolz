import sqlite3
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import mplcursors
import ast

def fetch_embeddings(db_path):
    """
    Fetch audio embeddings from an SQLite database.

    Args:
        db_path: Path to the SQLite database

    Returns:
        tuple: (embeddings array, list of file names)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the embedding_shape column exists
    cursor.execute("PRAGMA table_info(embeddings)")
    columns = [col[1] for col in cursor.fetchall()]
    has_shape_column = 'embedding_shape' in columns

    if has_shape_column:
        cursor.execute("SELECT file_name, embedding, embedding_shape FROM embeddings")
    else:
        cursor.execute("SELECT file_name, embedding FROM embeddings")

    rows = cursor.fetchall()
    
    if len(rows) == 0:
        print("No embeddings found in the database.")
        conn.close()
        return np.array([]), []
    
    labels = [row[0] for row in rows]

    # Handle embeddings with or without shape information
    if has_shape_column:
        embeddings = []
        for row in rows:
            binary_data = row[1]
            shape_str = row[2]

            try:
                # Parse shape from string like "(512,)"
                shape = ast.literal_eval(shape_str)
                embedding = np.frombuffer(binary_data, dtype=np.float32).reshape(shape)
                embeddings.append(embedding)
            except (ValueError, SyntaxError):
                # Fallback: assume it's a 1D array with 512 elements
                embedding = np.frombuffer(binary_data, dtype=np.float32)
                embeddings.append(embedding)
    else:
        # No shape information available, assume all are same size and 1D
        embeddings = []
        for row in rows:
            binary_data = row[1]
            embedding = np.frombuffer(binary_data, dtype=np.float32)
            embeddings.append(embedding)
    embeddings = np.array(embeddings)
    print(f"Retrieved {len(embeddings)} embeddings.")
    print(f"Embeddings shape: {embeddings.shape}")
    
    conn.close()

    return embeddings, labels

def generate_tsne_plot(embeddings, labels):
    """
    Generate a t-SNE plot of audio embeddings.

    Args:
        embeddings: Array of embedding vectors
        labels: List of file names corresponding to the embeddings
    """
    # Choose appropriate perplexity based on number of samples
    perplexity = min(30, max(3, len(embeddings) // 3))

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set(text=labels[sel.index])

    plt.title('t-SNE Plot of Audio Embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    plt.show(block=True)

def main(db_path):
    """
    Main function to fetch embeddings and generate a t-SNE plot.

    Args:
        db_path: Path to the SQLite database
    """
    embeddings, labels = fetch_embeddings(db_path)
    
    if len(embeddings) < 2:
        print("Not enough audio files to generate a t-SNE plot. Please ensure the database has at least two entries.")
        return

    generate_tsne_plot(embeddings, labels)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python plot_database.py <db_path>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    main(db_path)

