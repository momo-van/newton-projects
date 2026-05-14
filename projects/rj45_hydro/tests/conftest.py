"""Add the project root to sys.path so tests can import rj45_hydro and hydro_contact_viz."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
