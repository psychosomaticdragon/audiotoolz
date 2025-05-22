import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from muq import MuQMuLan
import torch
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

def generate_custom_axes_plot(embeddings, labels, axis1_text, axis2_text):
    """
    Generate a plot using text embeddings as custom axes.
    
    Args:
        embeddings: Array of audio embedding vectors
        labels: List of file names corresponding to the embeddings
        axis1_text: Text description for the x-axis
        axis2_text: Text description for the y-axis
    """
    print(f"Generating plot with axes: '{axis1_text}' vs '{axis2_text}'")
    
    # Initialize MuQMuLan
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan = mulan.to(device).eval()
    
    # Generate text embeddings for the axis labels
    texts = [axis1_text, axis2_text]
    with torch.no_grad():
        text_embeds = mulan(texts=texts)
    
    print(f"Text embeddings shape: {text_embeds.shape}")
    
    # Ensure embeddings are the right shape for similarity calculation
    audio_embeds = torch.from_numpy(embeddings)
    if len(audio_embeds.shape) == 3:
        # If embeddings are 3D (batch, sequence, features), flatten or select first sequence
        audio_embeds = audio_embeds.reshape(audio_embeds.shape[0], -1)
    
    print(f"Audio embeddings shape for similarity: {audio_embeds.shape}")
    
    # Calculate similarity to each text embedding
    similarities = mulan.calc_similarity(audio_embeds.cuda(), text_embeds)
    print(f"Similarities shape: {similarities.shape}")
    
    # Extract similarities to each axis concept
    x_similarities = similarities[:, 0].cpu().numpy()
    y_similarities = similarities[:, 1].cpu().numpy()
    
    # Plot the embeddings on the custom axes
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_similarities, y_similarities)
    
    # Add interactive hover labels
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set(text=labels[sel.index])

    plt.title('Audio Embeddings - Semantic Similarity Plot')
    plt.xlabel(f"Similarity to '{axis1_text}'")
    plt.ylabel(f"Similarity to '{axis2_text}'")
    
    # Add min/max annotations
    top_x = np.argmax(x_similarities)
    plt.annotate(
        f"Most '{axis1_text}': {labels[top_x]}", 
        xy=(x_similarities[top_x], y_similarities[top_x]),
        xytext=(10, 10), textcoords='offset points'
    )
    
    top_y = np.argmax(y_similarities)
    plt.annotate(
        f"Most '{axis2_text}': {labels[top_y]}", 
        xy=(x_similarities[top_y], y_similarities[top_y]),
        xytext=(10, -15), textcoords='offset points'
    )
    
    plt.tight_layout()
    plt.show(block=True)

def main(db_path, axis1_text, axis2_text):
    """
    Main function to fetch embeddings and generate a custom axes plot.
    
    Args:
        db_path: Path to the SQLite database
        axis1_text: Text description for the x-axis
        axis2_text: Text description for the y-axis
    """
    embeddings, labels = fetch_embeddings(db_path)
    
    if len(embeddings) < 2:
        print("Not enough audio files to generate a plot. Please ensure the database has at least two entries.")
        return

    generate_custom_axes_plot(embeddings, labels, axis1_text, axis2_text)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python plot_database_custom_axes.py <db_path> <axis1_text> <axis2_text>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    axis1_text = sys.argv[2]
    axis2_text = sys.argv[3]
    
    main(db_path, axis1_text, axis2_text)