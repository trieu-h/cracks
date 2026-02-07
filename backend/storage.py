"""
Simple in-memory storage - just a dict, no classes.
"""

# Storage dictionary - persists in memory while server is running
storage = {
    'datasets': {},           # id -> dataset_info
    'training_sessions': {},  # id -> session_data
    'models': {},            # id -> model_info
    'predictions': {},       # id -> prediction_result
    'gpu_history': [],       # list of gpu stats over time
    'active_websockets': {}  # session_id -> websocket connections
}

def get_storage():
    """Get the storage dict."""
    return storage
