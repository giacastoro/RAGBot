import os
import sys

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Working directory:", os.getcwd())

try:
    from langchain_chroma import Chroma
    print("Import langchain_chroma.Chroma: SUCCESS")
except ImportError as e:
    print("Import langchain_chroma.Chroma: FAILED")
    print(f"Error: {e}")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("Import langchain_huggingface.HuggingFaceEmbeddings: SUCCESS")
except ImportError as e:
    print("Import langchain_huggingface.HuggingFaceEmbeddings: FAILED")
    print(f"Error: {e}") 