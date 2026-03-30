"""
SQLite database module for persistent storage of datasets.
Uses Python's built-in sqlite3 - no additional dependencies required.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Database file path
DB_DIR = Path('./data')
DB_PATH = DB_DIR / 'app.db'


def init_db():
    """Initialize the database and create tables if they don't exist."""
    # Create data directory if it doesn't exist
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                yaml_path TEXT NOT NULL,
                num_classes INTEGER NOT NULL,
                class_names TEXT NOT NULL,  -- JSON array stored as text
                train_images INTEGER NOT NULL DEFAULT 0,
                val_images INTEGER NOT NULL DEFAULT 0,
                test_images INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        print(f"Database initialized at {DB_PATH}")
    finally:
        conn.close()


def save_dataset(dataset_info: Dict[str, Any]) -> bool:
    """Save or update a dataset in the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO datasets 
                (id, name, path, yaml_path, num_classes, class_names, train_images, val_images, test_images, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(
                    (SELECT created_at FROM datasets WHERE id = ?), 
                    CURRENT_TIMESTAMP
                ))
            ''', (
                dataset_info['id'],
                dataset_info['name'],
                dataset_info['path'],
                dataset_info['yaml_path'],
                dataset_info['num_classes'],
                json.dumps(dataset_info['class_names']),
                dataset_info.get('train_images', 0),
                dataset_info.get('val_images', 0),
                dataset_info.get('test_images', 0),
                dataset_info['id']  # For the COALESCE subquery
            ))
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception as e:
        print(f"Error saving dataset to database: {e}")
        return False


def delete_dataset(dataset_id: str) -> bool:
    """Delete a dataset from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.execute('DELETE FROM datasets WHERE id = ?', (dataset_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    except Exception as e:
        print(f"Error deleting dataset from database: {e}")
        return False


def load_datasets() -> Dict[str, Dict[str, Any]]:
    """Load all datasets from the database into a dictionary."""
    datasets = {}
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('SELECT * FROM datasets')
            rows = cursor.fetchall()
            
            for row in rows:
                dataset = {
                    'id': row['id'],
                    'name': row['name'],
                    'path': row['path'],
                    'yaml_path': row['yaml_path'],
                    'num_classes': row['num_classes'],
                    'class_names': json.loads(row['class_names']),
                    'train_images': row['train_images'],
                    'val_images': row['val_images'],
                    'test_images': row['test_images'],
                    'created_at': row['created_at']
                }
                datasets[row['id']] = dataset
                
            print(f"Loaded {len(datasets)} datasets from database")
        finally:
            conn.close()
    except Exception as e:
        print(f"Error loading datasets from database: {e}")
    
    return datasets


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Get a single dataset by ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'path': row['path'],
                    'yaml_path': row['yaml_path'],
                    'num_classes': row['num_classes'],
                    'class_names': json.loads(row['class_names']),
                    'train_images': row['train_images'],
                    'val_images': row['val_images'],
                    'test_images': row['test_images'],
                    'created_at': row['created_at']
                }
        finally:
            conn.close()
    except Exception as e:
        print(f"Error getting dataset from database: {e}")
    
    return None


def list_datasets() -> List[Dict[str, Any]]:
    """Get all datasets as a list."""
    return list(load_datasets().values())
