"""
Script for setting up essential project paths.
"""

from pathlib import Path

# Define the root absolute path
ROOT_ABSPATH = Path(__file__).resolve().parent.parent

# Define and create the path to the 'pdf' directory
PDF_DIR = ROOT_ABSPATH / "res" / "pdf"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# Define and create the path to the 'chunks' directory
CHUNKS_DIR = ROOT_ABSPATH / "res" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Define the filepath where the chunk documents get saved
CHUNKS_REPORT = CHUNKS_DIR / "chunks_report.json"
