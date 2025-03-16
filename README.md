# Chatbot RAG Locale

Questo progetto implementa un chatbot basato su RAG (Retrieval-Augmented Generation) utilizzando:
- ChromaDB (database vettoriale)
- LangChain (framework per LLM)
- Streamlit (interfaccia utente)
- Ollama (modelli linguistici locali)

## Installazione

1. Installa Ollama seguendo le istruzioni su [ollama.ai](https://ollama.ai/)
2. Scarica un modello supportato, ad esempio:
   ```
   ollama pull llama2
   ```
3. Installa le dipendenze Python:
   ```
   pip install -r requirements.txt
   ```

## Esecuzione

Per avviare l'applicazione:
```
streamlit run app.py
```

Questo aprirà l'interfaccia nel browser predefinito. 