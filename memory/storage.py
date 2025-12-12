import sqlite3
import time
import numpy as np
from config import DB_FILE, ENABLE_LOGGING

class StorageService:
    def __init__(self):
        self.conn = sqlite3.connect(DB_FILE)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS qa_memory (
            id INTEGER PRIMARY KEY, question TEXT, question_norm TEXT UNIQUE, 
            answer TEXT, embedding BLOB, created_at REAL, usage_count INTEGER DEFAULT 0, flags TEXT)''')
        cur.execute('CREATE TABLE IF NOT EXISTS user_facts (key TEXT PRIMARY KEY, value TEXT, confidence REAL, last_updated REAL)')
        cur.execute('CREATE TABLE IF NOT EXISTS conversation_log (id INTEGER PRIMARY KEY, role TEXT, message TEXT, embedding BLOB, timestamp REAL)')
        cur.execute('CREATE TABLE IF NOT EXISTS intent_training_data (id INTEGER PRIMARY KEY, text TEXT, label TEXT, timestamp REAL)')
        
        # Migrations
        cur.execute("PRAGMA table_info(qa_memory)")
        cols = [c[1] for c in cur.fetchall()]
        if 'flags' not in cols: cur.execute("ALTER TABLE qa_memory ADD COLUMN flags TEXT")
        self.conn.commit()

    def load_cache(self, brain):
        if not brain.is_cache_dirty and brain.vector_cache['vectors'] is not None: return
        cur = self.conn.cursor()
        cur.execute("SELECT id, embedding, answer, created_at, usage_count FROM qa_memory WHERE flags IS NULL OR flags != 'flagged'")
        rows = cur.fetchall()
        vecs, ans, ids, ts, usage = [], [], [], [], []
        for row in rows:
            if row['embedding']:
                vecs.append(np.frombuffer(row['embedding'], dtype=np.float32))
                ans.append(row['answer'])
                ids.append(row['id'])
                ts.append(row['created_at'])
                usage.append(row['usage_count'])
        
        if vecs:
            raw_matrix = np.vstack(vecs)
            brain.vector_cache = {
                "vectors": brain.apply_pca(raw_matrix),
                "answers": ans, "ids": ids,
                "timestamps": np.array(ts), "usage": np.array(usage)
            }
        else:
            brain.vector_cache = {"vectors": None, "answers": [], "ids": [], "timestamps": [], "usage": []}
        brain.is_cache_dirty = False

    def save_qa(self, q, a, vec):
        try:
            self.conn.execute("INSERT INTO qa_memory (question, question_norm, answer, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
                (q, q.lower().strip(), a, vec.tobytes(), time.time()))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError: return False

    def save_fact(self, k, v, conf=1.0):
        self.conn.execute("INSERT OR REPLACE INTO user_facts (key, value, confidence, last_updated) VALUES (?, ?, ?, ?)", 
                          (k, v, conf, time.time()))
        self.conn.commit()

    def get_user_facts(self):
        cur = self.conn.cursor()
        cur.execute("SELECT key, value FROM user_facts")
        return {row['key']: row['value'] for row in cur.fetchall()}

    def flag_memory(self, mem_id):
        self.conn.execute("UPDATE qa_memory SET flags = 'flagged' WHERE id = ?", (mem_id,))
        self.conn.commit()

    def log(self, role, msg, vec=None):
        if not ENABLE_LOGGING: return
        blob = vec.tobytes() if vec is not None else None
        self.conn.execute("INSERT INTO conversation_log (role, message, embedding, timestamp) VALUES (?, ?, ?, ?)", 
                          (role, msg, blob, time.time()))
        self.conn.commit()

    def wipe(self):
        self.conn.execute("DELETE FROM user_facts")
        self.conn.execute("DELETE FROM conversation_log")
        self.conn.commit()