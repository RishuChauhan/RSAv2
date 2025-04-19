import sqlite3
import json
import os
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class DataStorage:
    """
    Manages data storage for rifle shooting analysis application.
    Stores user profiles, sessions, shots, and metrics in SQLite database.
    """
    
    def __init__(self, db_path: str = 'data/rifle_shot_analysis.db'):
        """
        Initialize the data storage manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        # Connect to database and initialize tables
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish connection to the SQLite database with improved error handling."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Access rows by column name
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            # Create a fallback in-memory database if file connection fails
            self.conn = sqlite3.connect(':memory:')
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            # This will recreate the tables in memory
            self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        # Users table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Sessions table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Shots table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS shots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            subjective_score INTEGER NOT NULL,
            metrics TEXT NOT NULL,  -- JSON string for flexibility
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
        ''')
        
        # Baselines table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS baselines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            metrics TEXT NOT NULL,  -- JSON string for flexibility
            subjective_score REAL NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        self.conn.commit()
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def create_user(self, name: str, email: str, password_hash: str) -> int:
        """
        Create a new user profile.
        
        Args:
            name: User's full name
            email: User's email address
            password_hash: Hashed password for security
            
        Returns:
            User ID of the created user
        """
        try:
            self.cursor.execute(
                'INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)',
                (name, email, password_hash)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            # Email already exists
            return -1
    
    def authenticate_user(self, email: str, password_hash: str) -> Optional[Dict]:
        """
        Authenticate a user.
        
        Args:
            email: User's email address
            password_hash: Hashed password to check
            
        Returns:
            User data dictionary if authenticated, None otherwise
        """
        self.cursor.execute(
            'SELECT * FROM users WHERE email = ? AND password_hash = ?',
            (email, password_hash)
        )
        user = self.cursor.fetchone()
        
        if user:
            return dict(user)
        return None
    
    def get_user(self, user_id: int) -> Optional[Dict]:
        """
        Get user data by ID.
        
        Args:
            user_id: ID of the user to retrieve
            
        Returns:
            User data dictionary if found, None otherwise
        """
        self.cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = self.cursor.fetchone()
        
        if user:
            return dict(user)
        return None
    
    def create_session(self, user_id: int, name: str) -> int:
        """
        Create a new shooting session.
        
        Args:
            user_id: ID of the user
            name: Name of the session
            
        Returns:
            Session ID of the created session
        """
        self.cursor.execute(
            'INSERT INTO sessions (user_id, name) VALUES (?, ?)',
            (user_id, name)
        )
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_sessions(self, user_id: int) -> List[Dict]:
        """
        Get all sessions for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of session dictionaries
        """
        self.cursor.execute(
            '''SELECT s.*, COUNT(sh.id) as shot_count 
               FROM sessions s 
               LEFT JOIN shots sh ON s.id = sh.session_id 
               WHERE s.user_id = ? 
               GROUP BY s.id 
               ORDER BY s.created_at DESC''',
            (user_id,)
        )
        sessions = self.cursor.fetchall()
        return [dict(session) for session in sessions]
    
    def store_shot(self, session_id: int, metrics: Dict, subjective_score: int) -> int:
        """
        Store shot data with metrics and subjective score.
        
        Args:
            session_id: ID of the session
            metrics: Dictionary of stability metrics
            subjective_score: User's subjective score (1-10)
            
        Returns:
            Shot ID of the stored shot
        """
        # Convert metrics to JSON string
        metrics_json = json.dumps(metrics)
        
        # Get current timestamp in ISO8601 format
        timestamp = datetime.now().isoformat()
        
        self.cursor.execute(
            'INSERT INTO shots (session_id, timestamp, subjective_score, metrics) VALUES (?, ?, ?, ?)',
            (session_id, timestamp, subjective_score, metrics_json)
        )
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_shots(self, session_id: int) -> List[Dict]:
        """
        Get all shots for a session with improved error handling.
        
        Args:
            session_id: ID of the session
            
        Returns:
            List of shot dictionaries
        """
        if not session_id or session_id <= 0:
            return []
            
        try:
            # Ensure connection is active
            if self.conn is None or self.cursor is None:
                self._connect()
                
            self.cursor.execute(
                'SELECT * FROM shots WHERE session_id = ? ORDER BY timestamp',
                (session_id,)
            )
            shots = self.cursor.fetchall()
            
            # Parse metrics JSON
            shot_list = []
            for shot in shots:
                shot_dict = dict(shot)
                try:
                    shot_dict['metrics'] = json.loads(shot_dict['metrics'])
                except json.JSONDecodeError:
                    # Handle corrupt JSON gracefully
                    shot_dict['metrics'] = {}
                shot_list.append(shot_dict)
                    
            return shot_list
        except sqlite3.Error as e:
            print(f"Error fetching shots: {e}")
            return []
    
    def get_shot(self, shot_id: int) -> Optional[Dict]:
        """
        Get a specific shot by ID.
        
        Args:
            shot_id: ID of the shot to retrieve
            
        Returns:
            Shot dictionary if found, None otherwise
        """
        self.cursor.execute('SELECT * FROM shots WHERE id = ?', (shot_id,))
        shot = self.cursor.fetchone()
        
        if shot:
            shot_dict = dict(shot)
            shot_dict['metrics'] = json.loads(shot_dict['metrics'])
            return shot_dict
        return None
    
    def update_baseline(self, user_id: int, metrics: Dict, subjective_score: float) -> bool:
        """
        Update user's baseline best shot metrics.
        
        Args:
            user_id: ID of the user
            metrics: Dictionary of stability metrics
            subjective_score: Subjective score of the shot
            
        Returns:
            Boolean indicating if update was successful
        """
        # Convert metrics to JSON string
        metrics_json = json.dumps(metrics)
        
        # Check if baseline exists for user
        self.cursor.execute('SELECT id FROM baselines WHERE user_id = ?', (user_id,))
        baseline = self.cursor.fetchone()
        
        if baseline:
            # Update existing baseline
            self.cursor.execute(
                'UPDATE baselines SET metrics = ?, subjective_score = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?',
                (metrics_json, subjective_score, user_id)
            )
        else:
            # Create new baseline
            self.cursor.execute(
                'INSERT INTO baselines (user_id, metrics, subjective_score) VALUES (?, ?, ?)',
                (user_id, metrics_json, subjective_score)
            )
        
        self.conn.commit()
        return True
    
    def get_baseline(self, user_id: int) -> Optional[Dict]:
        """
        Get user's baseline best shot metrics.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Baseline dictionary if found, None otherwise
        """
        self.cursor.execute('SELECT * FROM baselines WHERE user_id = ?', (user_id,))
        baseline = self.cursor.fetchone()
        
        if baseline:
            baseline_dict = dict(baseline)
            baseline_dict['metrics'] = json.loads(baseline_dict['metrics'])
            return baseline_dict
        return None
    
    def get_sessions(self, user_id: int) -> List[Dict]:
        """
        Get all sessions for a user with improved error handling.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of session dictionaries
        """
        try:
            # Ensure connection is active
            if self.conn is None or self.cursor is None:
                self._connect()
                
            self.cursor.execute(
                '''SELECT s.*, COUNT(sh.id) as shot_count 
                FROM sessions s 
                LEFT JOIN shots sh ON s.id = sh.session_id 
                WHERE s.user_id = ? 
                GROUP BY s.id 
                ORDER BY s.created_at DESC''',
                (user_id,)
            )
            sessions = self.cursor.fetchall()
            return [dict(session) for session in sessions] if sessions else []
        except sqlite3.Error as e:
            print(f"Error fetching sessions: {e}")
            return []
        
    def get_session_stats(self, session_id: int) -> Dict:
        """
        Get statistical summary of a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Dictionary of session statistics
        """
        # Handle invalid session ID
        if not session_id or session_id <= 0:
            return {
                'avg_subjective_score': 0,
                'max_subjective_score': 0,
                'min_subjective_score': 0,
                'shot_count': 0
            }
        
        try:
            # Ensure connection is active
            if self.conn is None or self.cursor is None:
                self._connect()
                
            # Get all shots for the session
            shots = self.get_shots(session_id)
            
            if not shots:
                return {
                    'avg_subjective_score': 0,
                    'max_subjective_score': 0,
                    'min_subjective_score': 0,
                    'shot_count': 0
                }
            
            # Extract subjective scores
            subjective_scores = [shot.get('subjective_score', 0) for shot in shots if 'subjective_score' in shot]
            
            # Calculate statistics
            if subjective_scores:
                return {
                    'avg_subjective_score': sum(subjective_scores) / len(subjective_scores),
                    'max_subjective_score': max(subjective_scores),
                    'min_subjective_score': min(subjective_scores),
                    'shot_count': len(shots)
                }
            else:
                return {
                    'avg_subjective_score': 0,
                    'max_subjective_score': 0,
                    'min_subjective_score': 0,
                    'shot_count': len(shots)
                }
        except Exception as e:
            print(f"Error getting session stats: {e}")
            return {
                'avg_subjective_score': 0,
                'max_subjective_score': 0,
                'min_subjective_score': 0,
                'shot_count': 0,
                'error': str(e)
            }