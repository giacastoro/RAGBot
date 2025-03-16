@echo off
echo Avvio dell'applicazione RAG Chatbot...

REM Attiva l'ambiente virtuale
call .venv\Scripts\activate.bat

REM Imposta variabili d'ambiente per evitare problemi con torch e asyncio
set TOKENIZERS_PARALLELISM=false
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set STREAMLIT_SERVER_WATCH_FOLDERS=false

REM Avvia l'applicazione con le opzioni ottimali
streamlit run app.py --server.fileWatcherType none

pause 