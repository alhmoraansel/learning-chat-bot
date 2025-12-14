import re
import random
import shutil
import os
from datetime import datetime
from config import DB_FILE

class SafetyGuard:
    PROFANITY_LIST = ["badword1", "badword2"] 
    @staticmethod
    def is_safe(text):
        if len(text) > 500: return False 
        if any(bad in text.lower() for bad in SafetyGuard.PROFANITY_LIST): return False
        return True
    @staticmethod
    def normalize_text(text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text)

class Linguistics:
    STARTERS = {"who", "what", "where", "when", "why", "how", "is", "are", "do", "does", "can", "could", "would", "should", "whats", "what's"}
    GREETINGS = {"hey", "hi", "hello", "sup", "yo", "greetings", "morning"}
    FOLLOWUP_MARKERS = {"and", "but", "so", "then", "also", "plus"}
    PRONOUNS = {"it", "that", "this", "those", "he", "she", "they"}
    
    SYNONYMS = {
        "basically": ["essentially", "in short", "fundamentally"],
        "think": ["believe", "suspect", "guess"],
        "usually": ["typically", "generally", "often"],
        "correct": ["accurate", "right", "spot on"],
        "wrong": ["incorrect", "off", "inaccurate"]
    }

    @staticmethod
    def is_teachable(text):
        text = text.strip().lower()
        if text.endswith("?"): return True
        parts = text.split(' ')
        if not parts: return False
        if parts[0] in Linguistics.STARTERS: return True
        if parts[0] in Linguistics.GREETINGS: return True
        return False

    @staticmethod
    def paraphrase(text):
        words = text.split()
        new_words = []
        for w in words:
            clean_w = re.sub(r'[^\w]', '', w.lower())
            if clean_w in Linguistics.SYNONYMS and random.random() > 0.6:
                replacement = random.choice(Linguistics.SYNONYMS[clean_w])
                if w[0].isupper(): replacement = replacement.capitalize()
                new_words.append(replacement)
            else:
                new_words.append(w)
        return " ".join(new_words)

class Personality:
    TEMPLATES = {
        "GIVE_INFO": ["Here's what I know: {}.", "Basically: {}.", "The answer is: {}.", "As far as I recall: {}."],
        "UNCERTAIN": ["I'm not 100% sure, but: {}.", "It might be related to this: {}.", "Take this with a grain of salt: {}."],
        "CLARIFY": ["Do you mean {A} or {B}?", "Wait, are we talking about {topic}?", "Could you rephrase that?"],
        "REPAIR": ["My bad. Let me relearn that.", "I misunderstood. What is the correct answer?", "Thanks for the correction. What should it be?"],
        "SMALLTALK": ["Glad to hear it!", "Awesome.", "Great!", "Happy to help.", "Anytime!", "Good to know."]
    }
    
    @staticmethod
    def format(act, content):
        if act not in Personality.TEMPLATES: return content
        tmpl = random.choice(Personality.TEMPLATES[act])
        if len(content.split()) > 3 and act != "SMALLTALK":
            content = Linguistics.paraphrase(content)
        if "{}" in tmpl: return tmpl.format(content)
        return tmpl 

class FactExtractor:
    PATTERNS = [
        (r"(?:my|the)\s+(favorite\s+\w+|\w+)\s+is\s+(.+)", "standard"),
        (r"i am\s+(.+)", "identity"),
        (r"i like\s+(.+)", "likes"),
        (r"call me\s+(.+)", "name")
    ]
    @staticmethod
    def extract(text):
        text = text.strip().lower()
        for pat, type_ in FactExtractor.PATTERNS:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                if type_ == "standard": return match.group(1).strip(), match.group(2).strip()
                elif type_ == "identity": return "identity", match.group(1).strip()
                elif type_ == "likes": return "likes", match.group(1).strip()
                elif type_ == "name": return "name", match.group(1).strip()
        return None, None

class BackupManager:
    @staticmethod
    def create_backup():
        if os.path.exists(DB_FILE):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{DB_FILE}.{timestamp}.bak"
            try:
                backups = sorted([f for f in os.listdir('.') if f.startswith(DB_FILE) and f.endswith('.bak')])
                if len(backups) >= 3: os.remove(backups[0])
                shutil.copy2(DB_FILE, backup_name)
            except: pass