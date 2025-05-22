import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from muq import MuQMuLan
import torch
import ast
import os
import librosa
from sklearn.manifold import TSNE

def extract_audio_embedding(audio_path):
    """
    Extract embedding from an audio file using MuQMuLan.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        numpy.ndarray: Audio embedding vector
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan = mulan.to(device).eval()
    
    print(f"Extracting embedding from: {audio_path}")
    wav, sr = librosa.load(audio_path, sr=24000)
    wavs = torch.tensor(wav).unsqueeze(0).to(device)
    
    with torch.no_grad():
        audio_embed = mulan(wavs=wavs)
    
    return audio_embed.cpu().numpy().flatten()

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

def generate_tsne_with_new_song(embeddings, labels, new_song_path):
    """
    Generate a t-SNE plot including a new song highlighted differently.
    
    Args:
        embeddings: Array of audio embedding vectors from the database
        labels: List of file names corresponding to the embeddings
        new_song_path: Path to the new audio file to add to the visualization
    """
    new_song_name = os.path.basename(new_song_path)
    print(f"Generating t-SNE plot with new song: '{new_song_name}'")
    
    # Extract embedding for the new song
    new_song_embedding = extract_audio_embedding(new_song_path)
    
    # Combine embeddings (database + new song)
    all_embeddings = np.vstack([embeddings, new_song_embedding.reshape(1, -1)])
    all_labels = labels + [new_song_name]
    
    # Choose appropriate perplexity based on number of samples
    perplexity = min(30, max(5, len(all_embeddings) // 5))
    
    # Calculate t-SNE for the combined embeddings
    print(f"Calculating t-SNE with perplexity {perplexity}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Split back into database and new song for plotting
    db_embeddings_2d = embeddings_2d[:-1]
    new_song_2d = embeddings_2d[-1]
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot database songs
    scatter_db = plt.scatter(db_embeddings_2d[:, 0], db_embeddings_2d[:, 1], 
                           alpha=0.7, label='Database Songs')
    
    # Plot new song with different color and shape
    scatter_new = plt.scatter(new_song_2d[0], new_song_2d[1], 
                            color='red', s=150, marker='*', 
                            label=f'New Song: {new_song_name}')
    
    # Find and label the nearest neighbors to the new song
    distances = np.sqrt(np.sum((db_embeddings_2d - new_song_2d)**2, axis=1))
    nearest_indices = np.argsort(distances)[:5]  # Get 5 nearest neighbors
    
    # Add annotations for nearest neighbors
    for i, idx in enumerate(nearest_indices):
        plt.annotate(
            f"{i+1}. {labels[idx]}", 
            xy=(db_embeddings_2d[idx, 0], db_embeddings_2d[idx, 1]),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, color='blue',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
        )
        
        # Draw a line connecting the new song to its nearest neighbors
        plt.plot([new_song_2d[0], db_embeddings_2d[idx, 0]], 
                [new_song_2d[1], db_embeddings_2d[idx, 1]], 
                'k--', alpha=0.3)
    
    # Add interactive hover labels for all points
    cursor = mplcursors.cursor([scatter_db, scatter_new], hover=True)
    @cursor.connect("add")
    def on_add(sel):
        if sel.artist == scatter_db:
            sel.annotation.set(text=all_labels[sel.index])
        else:
            sel.annotation.set(text=all_labels[-1])
    
    plt.title('t-SNE Visualization with New Song')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    
    # Add annotation for the new song
    plt.annotate(
        f"NEW: {new_song_name}", 
        xy=(new_song_2d[0], new_song_2d[1]),
        xytext=(15, 15), textcoords='offset points',
        fontsize=12, fontweight='bold', color='red',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='red')
    )
    
    # Calculate similarity scores using MuQMuLan
    print("Calculating similarity scores for nearest neighbors...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan = mulan.to(device).eval()
    
    new_embedding_tensor = torch.from_numpy(new_song_embedding).unsqueeze(0).to(device)
    db_embeddings_tensor = torch.from_numpy(embeddings).to(device)
    
    with torch.no_grad():
        similarities = mulan.calc_similarity(db_embeddings_tensor, new_embedding_tensor)
        similarities = similarities.cpu().numpy().flatten()
    
    # Get the indices of the most similar songs based on actual embedding similarity
    most_similar_indices = np.argsort(similarities)[::-1][:5]  # Top 5 most similar
    
    # Add a table showing the most similar songs based on embedding similarity
    table_text = "Most Similar Songs by Embedding:\n"
    for i, idx in enumerate(most_similar_indices):
        sim_score = similarities[idx]
        table_text += f"{i+1}. {labels[idx]} ({sim_score:.3f})\n"
    
    plt.figtext(0.02, 0.02, table_text, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show(block=True)

def main(db_path, new_song_path):
    """
    Main function to fetch embeddings and generate a t-SNE plot with a new song.
    
    Args:
        db_path: Path to the SQLite database
        new_song_path: Path to the new audio file to add to the visualization
    """
    # Validate inputs
    if not os.path.exists(new_song_path):
        print(f"Error: New song file not found: {new_song_path}")
        return
    
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        return
        
    embeddings, labels = fetch_embeddings(db_path)
    
    if len(embeddings) < 2:
        print("Not enough audio files in the database. Please ensure the database has at least two entries.")
        return

    generate_tsne_with_new_song(embeddings, labels, new_song_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python plot_new_song_tsne.py <db_path> <new_song_path>")
        print("This will generate a t-SNE plot showing where the new song fits relative to the database songs.")
        sys.exit(1)
    
    db_path = sys.argv[1]
    new_song_path = sys.argv[2]
    
    main(db_path, new_song_path)