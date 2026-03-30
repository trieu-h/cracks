"""
Simple storage module - persists to SQLite database.
"""
from database import load_datasets, save_dataset, delete_dataset

# Storage dictionary - datasets loaded from SQLite on startup
storage = {
    'datasets': {},           # id -> dataset_info (loaded from SQLite)
    'training_sessions': {},  # id -> session_data
    'models': {},            # id -> model_info
    'predictions': {},       # id -> prediction_result
    'gpu_history': [],       # list of gpu stats over time
    'active_websockets': {}  # session_id -> websocket connections
}

def init_storage():
    """Initialize storage by loading datasets from SQLite database."""
    storage['datasets'] = load_datasets()
    print(f"Storage initialized with {len(storage['datasets'])} datasets")

def get_storage():
    """Get the storage dict."""
    return storage

# Database helper functions for datasets
def save_dataset_to_db(dataset_info: dict) -> bool:
    """Save a dataset to both memory and database."""
    # Save to database first
    success = save_dataset(dataset_info)
    if success:
        # Update in-memory storage
        storage['datasets'][dataset_info['id']] = dataset_info
    return success

def delete_dataset_from_db(dataset_id: str) -> bool:
    """Delete a dataset from both memory and database."""
    # Delete from database first
    success = delete_dataset(dataset_id)
    if success and dataset_id in storage['datasets']:
        # Remove from in-memory storage
        del storage['datasets'][dataset_id]
    return success
