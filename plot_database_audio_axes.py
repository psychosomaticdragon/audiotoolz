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

def calculate_tsne_dimension(embeddings):
    """
    Calculate t-SNE for the second dimension.
    
    Args:
        embeddings: Array of audio embedding vectors
        
    Returns:
        numpy.ndarray: 1D array of t-SNE values for the second dimension
    """
    print("Calculating t-SNE for second dimension...")
    # Choose appropriate perplexity based on number of samples
    perplexity = min(30, max(5, len(embeddings) // 5))
    
    # Use only 1 component since we need just the second dimension
    tsne = TSNE(n_components=1, random_state=42, perplexity=perplexity)
    tsne_values = tsne.fit_transform(embeddings).flatten()
    
    # Normalize to 0-1 range similar to similarity scores
    tsne_min, tsne_max = tsne_values.min(), tsne_values.max()
    tsne_values = (tsne_values - tsne_min) / (tsne_max - tsne_min)
    
    return tsne_values

def generate_audio_custom_axes_plot(embeddings, labels, reference_audio_path, concept_text=None):
    """
    Generate a plot using one audio and one custom axis (text or t-SNE).
    
    Args:
        embeddings: Array of audio embedding vectors
        labels: List of file names corresponding to the embeddings
        reference_audio_path: Path to the reference audio file for one axis
        concept_text: Optional text description for the second axis. If None, t-SNE is used
    """
    if concept_text:
        print(f"Generating plot with axes: '{os.path.basename(reference_audio_path)}' vs '{concept_text}'")
    else:
        print(f"Generating plot with axes: '{os.path.basename(reference_audio_path)}' vs 't-SNE'")
    
    # Initialize MuQMuLan
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan = mulan.to(device).eval()
    
    # Extract embedding for the reference audio
    reference_embedding = extract_audio_embedding(reference_audio_path)
    reference_embedding_tensor = torch.from_numpy(reference_embedding).unsqueeze(0).to(device)
    
    # Ensure embeddings are the right shape for similarity calculation
    audio_embeds = torch.from_numpy(embeddings)
    if len(audio_embeds.shape) == 3:
        # If embeddings are 3D (batch, sequence, features), flatten or select first sequence
        audio_embeds = audio_embeds.reshape(audio_embeds.shape[0], -1)
    audio_embeds = audio_embeds.to(device)
    
    # Calculate similarity to reference audio for first axis
    audio_similarities = mulan.calc_similarity(audio_embeds, reference_embedding_tensor)
    audio_similarities = audio_similarities.cpu().numpy().flatten()
    
    # Get values for second axis - either concept similarity or t-SNE
    if concept_text:
        # Generate text embedding for the concept and calculate similarities
        with torch.no_grad():
            concept_embedding = mulan(texts=[concept_text])
        
        y_values = mulan.calc_similarity(audio_embeds, concept_embedding)
        y_values = y_values.cpu().numpy().flatten()
        y_axis_label = f"Similarity to '{concept_text}'"
    else:
        # Calculate t-SNE for second dimension
        y_values = calculate_tsne_dimension(embeddings)
        y_axis_label = "t-SNE Dimension"
    
    # Plot the embeddings on the custom axes
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(audio_similarities, y_values, alpha=0.7)
    
    # Add interactive hover labels
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set(text=labels[sel.index])

    if concept_text:
        plt.title('Audio Embeddings - Audio-Concept Similarity Plot')
    else:
        plt.title('Audio Embeddings - Audio Similarity vs t-SNE')
    
    plt.xlabel(f"Similarity to '{os.path.basename(reference_audio_path)}'")
    plt.ylabel(y_axis_label)
    
    # Add reference line for audio similarity
    plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.3)
    
    # Add min/max annotations
    top_audio = np.argmax(audio_similarities)
    plt.annotate(
        f"Most similar to reference: {labels[top_audio]}", 
        xy=(audio_similarities[top_audio], y_values[top_audio]),
        xytext=(10, 10), textcoords='offset points',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
    )
    
    if concept_text:
        # Add concept-specific annotations
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        
        top_concept = np.argmax(y_values)
        plt.annotate(
            f"Most '{concept_text}': {labels[top_concept]}", 
            xy=(audio_similarities[top_concept], y_values[top_concept]),
            xytext=(10, -15), textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
        )
        
        # Add a legend explaining the quadrants
        plt.text(0.98, 0.98, "Similar to both", ha='right', va='top', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.3))
        
        plt.text(0.02, 0.98, f"High '{concept_text}'\nLow similarity to reference", 
                ha='left', va='top', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.3))
        
        plt.text(0.98, 0.02, f"High similarity to reference\nLow '{concept_text}'", 
                ha='right', va='bottom', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.3))
        
        plt.text(0.02, 0.02, "Low similarity to both", 
                ha='left', va='bottom', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", alpha=0.3))
    else:
        # Add t-SNE specific annotations - highlight extremes
        top_tsne = np.argmax(y_values)
        bottom_tsne = np.argmin(y_values)
        
        plt.annotate(
            f"t-SNE high: {labels[top_tsne]}", 
            xy=(audio_similarities[top_tsne], y_values[top_tsne]),
            xytext=(10, -15), textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.3)
        )
        
        plt.annotate(
            f"t-SNE low: {labels[bottom_tsne]}", 
            xy=(audio_similarities[bottom_tsne], y_values[bottom_tsne]),
            xytext=(-10, 10), textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.3)
        )
    
    # Highlight the reference song if it's in the database
    reference_name = os.path.basename(reference_audio_path)
    if reference_name in labels:
        idx = labels.index(reference_name)
        plt.scatter(audio_similarities[idx], y_values[idx], 
                   color='red', s=100, marker='*', label='Reference Song')
        plt.legend()
    
    plt.tight_layout()
    plt.show(block=True)

def main(db_path, reference_audio_path, concept_text=None):
    """
    Main function to fetch embeddings and generate an audio-concept axes plot.
    
    Args:
        db_path: Path to the SQLite database
        reference_audio_path: Path to the reference audio file for one axis
        concept_text: Optional text description for the second axis. If None, t-SNE is used
    """
    # Validate inputs
    if not os.path.exists(reference_audio_path):
        print(f"Error: Reference audio file not found: {reference_audio_path}")
        return
    
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        return
        
    embeddings, labels = fetch_embeddings(db_path)
    
    if len(embeddings) < 2:
        print("Not enough audio files to generate a plot. Please ensure the database has at least two entries.")
        return

    generate_audio_custom_axes_plot(embeddings, labels, reference_audio_path, concept_text)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python plot_database_audio_axes.py <db_path> <reference_audio_path> [concept_text]")
        print("If concept_text is omitted, t-SNE will be used for the second axis")
        sys.exit(1)
    
    db_path = sys.argv[1]
    reference_audio_path = sys.argv[2]
    concept_text = sys.argv[3] if len(sys.argv) == 4 else None
    
    main(db_path, reference_audio_path, concept_text)