import os
import torch
import librosa
from muq import MuQMuLan
import sqlite3
import numpy as np
import pathlib

# This will automatically fetch checkpoints from huggingface
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
mulan = mulan.to(device).eval()

def extract_embeddings(audio_path):
    """
    Extract audio embeddings from an audio file.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        numpy.ndarray: Audio embedding vector
    """
    wav, sr = librosa.load(audio_path, sr=24000)
    wavs = torch.tensor(wav).unsqueeze(0).to(device) 
    with torch.no_grad():
        audio_embeds = mulan(wavs=wavs)
    
    # Ensure we return a flattened 1D array for consistent storage
    return audio_embeds.cpu().numpy().flatten()

def setup_database(db_path):
    """
    Set up the SQLite database with necessary tables if they don't exist.
    Args:
        db_path: Path to the SQLite database

    Returns:
        bool: True if the database was created, False if it already existed
    """
    db_file = pathlib.Path(db_path)
    db_dir = db_file.parent

    # Create parent directories if they don't exist
    if not db_dir.exists():
        print(f"Creating directory: {db_dir}")
        db_dir.mkdir(parents=True, exist_ok=True)

    db_existed = db_file.exists()

    # Connect to the database (this will create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        print(f"Creating new embeddings table in database: {db_path}")
        cursor.execute("""
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                file_name TEXT, 
                embedding BLOB,
                embedding_shape TEXT
            )
        """)
    
    conn.commit()
    conn.close()

    if not db_existed:
        print(f"Created new database at: {db_path}")
        return True
    else:
        print(f"Using existing database at: {db_path}")
        return False

def add_to_database(db_path, file_name, embedding):
    """
    Store the audio embedding in an SQLite database.
    Args:
        db_path: Path to the SQLite database
        file_name: Name of the audio file
        embedding: Numpy array containing the embedding
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Store the embedding shape as JSON string
    embedding_shape = str(embedding.shape)

    # Insert the new entry
    print(f"Storing embedding of shape: {embedding.shape}, dtype: {embedding.dtype}")
    cursor.execute(
        "INSERT INTO embeddings (file_name, embedding, embedding_shape) VALUES (?, ?, ?)",
        (file_name, sqlite3.Binary(embedding.tobytes()), embedding_shape)
    )

    conn.commit()
    conn.close()
        
def get_existing_files(db_path):
    """
    Get a list of files already in the database.

    Args:
        db_path: Path to the SQLite database

    Returns:
        set: Set of file names already in the database
    """
    if not os.path.exists(db_path):
        return set()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT file_name FROM embeddings")
        existing_files = set(row[0] for row in cursor.fetchall())
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        existing_files = set()

    conn.close()
    return existing_files

def main(audio_folder: str, db_path: str, update_db: bool):
    """
    Process audio files and store their embeddings in a database.

    Args:
        audio_folder: Path to the folder containing audio files
        db_path: Path to the SQLite database
        update_db: Whether to update existing entries in the database
    """
    # Ensure the database exists with the correct schema
    setup_database(db_path)

    # Get list of audio files
    audio_files = []
    for root, dirs, files in os.walk(audio_folder):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                audio_path = os.path.join(root, file)
                audio_files.append((audio_path, file))

    if not audio_files:
        print(f"No audio files found in {audio_folder}")
        return
    print(f"Found {len(audio_files)} audio files")

    # If we're not updating, check which files are already in DB
    if not update_db:
        existing_files = get_existing_files(db_path)

        # Filter out files that are already in the database
        new_files = [(path, name) for path, name in audio_files if name not in existing_files]

        print(f"Skipping {len(audio_files) - len(new_files)} files already in database")
        audio_files = new_files
    print(f"Processing {len(audio_files)} audio files...")
    for i, (audio_file, file_name) in enumerate(audio_files):
        print(f"Processing {i+1}/{len(audio_files)}: {file_name}")
        embedding = extract_embeddings(audio_file)
        add_to_database(db_path, file_name, embedding)
    
    print(f"Completed processing. Database updated at {db_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python analyze_and_store.py <audio_folder> <db_path> <update_db>")
        print("  audio_folder: Path to folder containing audio files (.wav, .mp3, .flac)")
        print("  db_path: Path where the SQLite database should be stored")
        print("  update_db: 'true' to reprocess existing files, 'false' to process only new files")
        sys.exit(1)
    
    audio_folder = sys.argv[1]
    db_path = sys.argv[2]
    update_db = sys.argv[3].lower() in ('true', 'yes', '1', 't')
    main(audio_folder, db_path, update_db)
