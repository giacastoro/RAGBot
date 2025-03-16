# Guida Completa all'Utilizzo del RAG Chatbot

## Indice
1. [Introduzione](#introduzione)
2. [Requisiti e Installazione](#requisiti-e-installazione)
3. [Configurazione di Ollama](#configurazione-di-ollama)
4. [Avvio dell'Applicazione](#avvio-dellapplicazione)
5. [Gestione dei Chatbot Multipli](#gestione-dei-chatbot-multipli)
6. [Caricamento e Gestione dei Documenti](#caricamento-e-gestione-dei-documenti)
7. [Elaborazione dei Documenti](#elaborazione-dei-documenti)
8. [Interfaccia di Chat](#interfaccia-di-chat)
9. [Personalizzazione dei Template](#personalizzazione-dei-template)
10. [Impostazioni Avanzate](#impostazioni-avanzate)
11. [Risoluzione dei Problemi](#risoluzione-dei-problemi)

## Introduzione

Il RAG Chatbot è un'applicazione che combina la potenza dei modelli linguistici con una base di conoscenza creata dai tuoi documenti personali. Grazie all'architettura RAG (Retrieval-Augmented Generation), il chatbot è in grado di rispondere a domande basandosi esclusivamente sui documenti che hai caricato, fornendo risposte accurate e contestualizzate.

### Principali Funzionalità
- Gestione di più chatbot specializzati su diversi argomenti
- Supporto per vari formati di documenti (PDF, DOCX, TXT, MD, CSV, XLSX)
- Elaborazione OCR per estrarre testo dalle immagini
- Creazione di database vettoriali per ricerche semantiche
- Interfaccia di chat intuitiva
- Personalizzazione dei template di prompt
- Selezione di diversi modelli LLM

## Requisiti e Installazione

### Requisiti di Sistema
- Python 3.8 o superiore
- Almeno 4GB di RAM
- Connessione Internet per il download dei modelli e l'OCR
- **Ollama installato e funzionante** (vedi sezione dedicata)

### Dipendenze
Tutte le dipendenze necessarie sono elencate nel file `requirements.txt` e includono:
- streamlit
- langchain e relativi componenti
- chromadb
- sentence-transformers
- torch
- pdfplumber
- e altre librerie per l'elaborazione dei documenti

### Installazione

1. Clona o scarica il repository
2. Crea un ambiente virtuale:
   ```
   python -m venv .venv
   ```
3. Attiva l'ambiente virtuale:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
4. Installa le dipendenze:
   ```
   pip install -r requirements.txt
   ```

## Configurazione di Ollama

Questa applicazione utilizza Ollama per eseguire modelli linguistici localmente. È necessario installarlo e configurarlo **prima** di avviare l'applicazione RAG Chatbot.

### Installazione di Ollama

1. Scarica Ollama dal sito ufficiale: [https://ollama.com/download](https://ollama.com/download)
2. Segui le istruzioni di installazione per il tuo sistema operativo
3. Verifica che Ollama sia stato installato correttamente eseguendo `ollama --version` nel terminale

### Download di Almeno Un Modello

Prima di utilizzare l'applicazione, è necessario scaricare almeno un modello:

1. Apri un terminale
2. Esegui il comando per scaricare il modello predefinito (gemma3:4b):
   ```
   ollama pull gemma3:4b
   ```
3. Attendi il completamento del download (può richiedere alcuni minuti)

### Verifica che Ollama sia in Esecuzione

Prima di avviare l'applicazione:

1. Assicurati che il servizio Ollama sia in esecuzione
2. Su Windows, puoi verificarlo cercando "Ollama" nei processi attivi
3. Puoi testare che Ollama funzioni correttamente eseguendo:
   ```
   ollama run gemma3:4b "Ciao!"
   ```

Se ricevi una risposta, Ollama è configurato correttamente e l'applicazione potrà utilizzare i modelli.

> **Nota importante**: L'applicazione RAG Chatbot si basa su Ollama per il funzionamento dei modelli LLM. Se Ollama non è installato o non è in esecuzione, riceverai errori durante l'utilizzo dell'applicazione.

## Avvio dell'Applicazione

Per avviare l'applicazione, esegui il file batch incluso:
```
.\start_app.bat
```

Questo script:
1. Attiva l'ambiente virtuale
2. Imposta le variabili d'ambiente necessarie per evitare conflitti con torch e asyncio
3. Avvia Streamlit con le opzioni ottimali

L'applicazione sarà disponibile nel browser all'indirizzo: http://localhost:8501

## Gestione dei Chatbot Multipli

L'applicazione supporta la creazione e gestione di più chatbot specializzati per diversi ambiti o collezioni di documenti.

### Creazione di un Nuovo Chatbot

1. Nella sezione "Gestione Chatbot Multipli", espandi "➕ Crea Nuovo Chatbot"
2. Inserisci un nome significativo per il tuo chatbot
3. Clicca su "Crea Chatbot"

Ogni chatbot avrà la sua struttura di directory:
```
chatbots/nome_chatbot/
├── data/         # Documenti originali
├── markdown/     # File convertiti in markdown
├── chroma_db/    # Database vettoriale
└── templates.json # Template personalizzati
```

### Selezione di un Chatbot

1. Utilizza il menu a tendina "Seleziona Chatbot" per passare da un chatbot all'altro
2. Il sistema caricherà automaticamente lo stato e i documenti del chatbot selezionato

### Gestione dei Chatbot Esistenti

1. Espandi la sezione "⚙️ Gestisci 'nome_chatbot'"
2. Qui puoi:
   - Rinominare il chatbot (cambia solo il nome visualizzato, non la directory)
   - Eliminare il chatbot (richiede conferma)

## Caricamento e Gestione dei Documenti

### Caricamento dei Documenti

1. Vai nella scheda "Gestione File"
2. Apri le impostazioni (⚙️) e seleziona la tab "Documenti"
3. Utilizza il caricatore di file per selezionare uno o più documenti
4. Clicca su "Salva documenti" per caricarli nel chatbot attivo

### Formati Supportati
- PDF (con supporto OCR)
- DOCX
- TXT
- MD (Markdown)
- CSV
- XLSX (Excel)

### Gestione dei File

Nella scheda "Gestione File" puoi:
1. Visualizzare l'elenco dei documenti PDF caricati
2. Visualizzare i file Markdown generati
3. Visualizzare il contenuto dei file Markdown
4. Eliminare documenti specifici
5. Gestire il database vettoriale

## Elaborazione dei Documenti

Dopo aver caricato i documenti, devi elaborarli per creare il database vettoriale che il chatbot utilizzerà per rispondere alle domande.

### Elaborazione Completa

1. Vai nella scheda "Gestione File"
2. Clicca su "Elabora tutti i documenti"

Questo processo:
- Carica tutti i documenti dalla cartella `data/`
- Li converte in markdown se necessario
- Divide i documenti in chunk
- Crea embedding vettoriali
- Salva il database in `chroma_db/`

### Elaborazione Solo Markdown

Se hai già convertito i documenti in markdown:
1. Clicca su "Usa solo file Markdown"

Questa opzione è più veloce e può essere utile se hai già elaborato i file ma devi ricreare il database.

### Caricamento Database Esistente

Se il database è già stato creato in precedenza:
1. Clicca su "Carica database esistente"

Questa opzione carica il database vettoriale senza rielaborare i documenti.

## Interfaccia di Chat

### Utilizzo della Chat

1. Vai nella scheda "Chat"
2. Assicurati che un database sia stato caricato (altrimenti apparirà un messaggio informativo)
3. Digita la tua domanda nella casella di input in basso
4. Riceverai una risposta basata sui contenuti dei tuoi documenti
5. La risposta includerà anche un elenco dei documenti consultati come fonte

### Funzionalità della Chat

- L'interfaccia mostra la cronologia completa della conversazione
- Puoi cancellare la cronologia con il pulsante "🗑️ Cancella chat"
- Le risposte includono riferimenti ai documenti di origine
- La casella di input rimane fissa in basso per facilità d'uso

## Personalizzazione dei Template

I template controllano come il modello linguistico interpreta le domande e genera le risposte.

### Accesso ai Template

1. Apri le impostazioni (⚙️)
2. Seleziona la tab "Template"

### Personalizzazione

Puoi modificare due template principali:
1. **Template per la query iniziale**: Controlla come il modello formula la prima risposta
2. **Template per il raffinamento**: Controlla come il modello migliora la risposta con ulteriori informazioni

Variabili disponibili:
- `{context}`: Il testo estratto dai documenti
- `{question}`: La domanda dell'utente
- `{existing_answer}`: La risposta precedente (solo per il template di raffinamento)

Dopo aver modificato i template, clicca su "Salva template".

## Impostazioni Avanzate

### OCR (Riconoscimento Ottico dei Caratteri)

L'OCR permette di estrarre testo dalle immagini contenute nei PDF:
1. Nelle impostazioni generali, puoi attivare/disattivare l'OCR
2. Quando attivo, utilizza l'API Mistral per l'elaborazione delle immagini

### Selezione del Modello LLM

Puoi cambiare il modello LLM utilizzato dal chatbot:
1. Usa il selettore "Modello LLM" nell'intestazione dell'applicazione
2. I modelli disponibili vengono rilevati automaticamente (basati su Ollama)
3. Il modello predefinito è "gemma3:4b"

### Modalità Debug

1. Nelle impostazioni generali, attiva "Modalità debug"
2. Questo mostrerà informazioni aggiuntive utili per diagnosticare problemi

## Risoluzione dei Problemi

### Database Bloccato

Se ricevi errori relativi al database bloccato:
1. Vai nella scheda "Gestione File"
2. Nella tab "Database", clicca su "Elimina in background"
3. In alternativa, riavvia l'applicazione

### Problemi con Torch/Event Loop

Se riscontri errori relativi a "no running event loop":
1. Assicurati di avviare l'applicazione usando `.\start_app.bat`
2. Questo file imposta le variabili d'ambiente necessarie per evitare conflitti

### Problemi di Importazione

Se vedi errori di importazione:
1. Verifica che tutte le dipendenze siano installate: `pip install -r requirements.txt`
2. Assicurati di utilizzare l'ambiente virtuale corretto

### Memoria Insufficiente

Se l'applicazione si arresta per problemi di memoria:
1. Riduci la dimensione dei chunk nei template (default: 1000)
2. Utilizza modelli LLM più leggeri
3. Elabora meno documenti alla volta

### Altri Problemi

In caso di altri problemi:
1. Verifica i log nel terminale
2. Riavvia l'applicazione
3. Controlla che l'ambiente virtuale sia attivato correttamente

---

## Nota sulla Sicurezza

Tutti i dati rimangono in locale sul tuo computer. L'applicazione non invia i tuoi documenti a server esterni, ad eccezione dell'OCR che utilizza l'API Mistral per elaborare le immagini contenute nei PDF.

---

Questa guida è stata generata il 16/03/2025. 