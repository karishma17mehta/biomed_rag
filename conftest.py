# conftest.py
import sys
from pathlib import Path

# Ensure repo root is on the path so `from app.xxx import` works in tests
sys.path.insert(0, str(Path(__file__).parent))