import os
import streamlit as st
import tempfile
import shutil
import atexit
import time
import gc
import threading
import json

# Importazione delle classi dai components
from components.document_processor import DocumentProcessor
from components.chatbot import Chatbot
from components.vector_store import VectorStore

# Importazioni per il text processing
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configurazioni per prevenire problemi con torch e asyncio
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import nest_asyncio
nest_asyncio.apply()

# Configurazione dell'applicazione
st.set_page_config(
    page_title="RAG Chatbot Locale",
    page_icon="🤖",
    layout="wide"
)

# Directory principale dei chatbot
CHATBOTS_DIR = "chatbots"
os.makedirs(CHATBOTS_DIR, exist_ok=True)

# File di configurazione per i chatbot
CHATBOTS_CONFIG = os.path.join(CHATBOTS_DIR, "chatbots_config.json")

# Percorsi delle directory
PDF_DIR = "data"
DB_DIR = "chroma_db"
MD_DIR = "markdown_data"

# Chiave API di Mistral
MISTRAL_API_KEY = "your_api"

# Funzione per caricare la configurazione dei chatbot
def load_chatbots_config():
    """Carica la configurazione dei chatbot dal file JSON"""
    if os.path.exists(CHATBOTS_CONFIG):
        try:
            with open(CHATBOTS_CONFIG, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Errore nel caricamento della configurazione dei chatbot: {str(e)}")
    
    # Se il file non esiste o c'è un errore, restituisci una configurazione di default
    default_config = {
        "chatbots": [],
        "active_chatbot": None
    }
    save_chatbots_config(default_config)
    return default_config

# Funzione per salvare la configurazione dei chatbot
def save_chatbots_config(config):
    """Salva la configurazione dei chatbot nel file JSON"""
    try:
        with open(CHATBOTS_CONFIG, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Errore nel salvataggio della configurazione dei chatbot: {str(e)}")
        return False

# Funzione per creare un nuovo chatbot
def create_chatbot(name):
    """
    Crea un nuovo chatbot con la struttura di directory richiesta
    
    :param name: Nome del chatbot
    :return: Dizionario con le informazioni del chatbot
    """
    # Normalizza il nome per usarlo come nome di directory
    dir_name = name.lower().replace(" ", "_")
    chatbot_path = os.path.join(CHATBOTS_DIR, dir_name)
    
    # Verifica se esiste già un chatbot con lo stesso nome
    if os.path.exists(chatbot_path):
        return None, f"Esiste già un chatbot con il nome '{name}'"
    
    # Crea le directory per il chatbot
    os.makedirs(chatbot_path, exist_ok=True)
    os.makedirs(os.path.join(chatbot_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(chatbot_path, "markdown"), exist_ok=True)
    os.makedirs(os.path.join(chatbot_path, "chroma_db"), exist_ok=True)
    
    # Crea un file templates.json di default
    default_templates = {
        "qa_template": """
            Sei un assistente AI utile ed esperto che aiuta le persone a trovare informazioni.
            Utilizza SOLO le informazioni di contesto fornite per rispondere alla domanda.
            Se le informazioni necessarie non sono presenti nel contesto, rispondi onestamente che non lo sai.
            
            Contesto: {context}
            
            Domanda: {question}
            
            Risposta:
            """,
        "refine_template": """
            Hai una risposta esistente: 
            {existing_answer}
            
            Abbiamo il seguente nuovo contesto da considerare:
            {context}
            
            Perfeziona la risposta originale se ci sono nuove o migliori informazioni.
            Se il nuovo contesto non cambia o non aggiunge nulla alla risposta originale, mantienila invariata.
            Se la risposta non è nei dati di origine o è incompleta, dì:
            "Mi dispiace, ma non ho trovato queste informazioni nei dati forniti."
            
            Domanda: {question}
            
            Risposta perfezionata:
            """
    }
    
    # Salva i template nel file
    with open(os.path.join(chatbot_path, "templates.json"), "w", encoding="utf-8") as f:
        json.dump(default_templates, f, indent=4, ensure_ascii=False)
    
    # Crea il dizionario del chatbot
    chatbot_info = {
        "id": dir_name,
        "name": name,
        "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "path": chatbot_path,
        "data_dir": os.path.join(chatbot_path, "data"),
        "markdown_dir": os.path.join(chatbot_path, "markdown"),
        "chroma_db_dir": os.path.join(chatbot_path, "chroma_db"),
        "templates_file": os.path.join(chatbot_path, "templates.json")
    }
    
    # Aggiorna la configurazione
    config = load_chatbots_config()
    config["chatbots"].append(chatbot_info)
    save_chatbots_config(config)
    
    return chatbot_info, f"Chatbot '{name}' creato con successo"

# Funzione per rinominare un chatbot
def rename_chatbot(chatbot_id, new_name):
    """
    Rinomina un chatbot esistente
    
    :param chatbot_id: ID del chatbot
    :param new_name: Nuovo nome del chatbot
    :return: Esito dell'operazione e messaggio
    """
    config = load_chatbots_config()
    
    # Trova il chatbot
    for i, chatbot in enumerate(config["chatbots"]):
        if chatbot["id"] == chatbot_id:
            # Aggiorna solo il nome (non la directory)
            config["chatbots"][i]["name"] = new_name
            save_chatbots_config(config)
            return True, f"Chatbot rinominato in '{new_name}'"
    
    return False, f"Chatbot con ID '{chatbot_id}' non trovato"

# Funzione per eliminare un chatbot
def delete_chatbot(chatbot_id):
    """
    Elimina un chatbot e tutti i suoi dati
    
    :param chatbot_id: ID del chatbot da eliminare
    :return: Esito dell'operazione e messaggio
    """
    config = load_chatbots_config()
    
    # Trova il chatbot
    for i, chatbot in enumerate(config["chatbots"]):
        if chatbot["id"] == chatbot_id:
            chatbot_path = chatbot["path"]
            chatbot_name = chatbot["name"]
            
            # Rimuovi il chatbot dalla configurazione
            del config["chatbots"][i]
            
            # Se il chatbot attivo è quello che stiamo eliminando, imposta active_chatbot a None
            if config["active_chatbot"] == chatbot_id:
                config["active_chatbot"] = None
            
            save_chatbots_config(config)
            
            # Elimina la directory del chatbot
            try:
                shutil.rmtree(chatbot_path)
                return True, f"Chatbot '{chatbot_name}' eliminato con successo"
            except Exception as e:
                return False, f"Errore nell'eliminazione della directory: {str(e)}"
    
    return False, f"Chatbot con ID '{chatbot_id}' non trovato"

# Funzione per impostare il chatbot attivo
def set_active_chatbot(chatbot_id):
    """
    Imposta il chatbot attivo
    
    :param chatbot_id: ID del chatbot da attivare
    :return: Dizionario con le informazioni del chatbot
    """
    config = load_chatbots_config()
    
    # Imposta il chatbot attivo
    config["active_chatbot"] = chatbot_id
    save_chatbots_config(config)
    
    # Trova e restituisci le informazioni del chatbot
    for chatbot in config["chatbots"]:
        if chatbot["id"] == chatbot_id:
            return chatbot
    
    return None

# Funzione per ottenere il chatbot attivo
def get_active_chatbot():
    """
    Ottiene le informazioni del chatbot attivo
    
    :return: Dizionario con le informazioni del chatbot
    """
    config = load_chatbots_config()
    active_id = config.get("active_chatbot")
    
    if active_id:
        for chatbot in config["chatbots"]:
            if chatbot["id"] == active_id:
                return chatbot
    
    return None

# Funzione per pulire le risorse alla chiusura dell'applicazione
def cleanup_resources():
    """Pulisce le risorse alla chiusura dell'applicazione"""
    print("Pulizia delle risorse in corso...")
    
    # Chiudi il vector store se è aperto
    if "vector_store" in st.session_state and st.session_state.vector_store is not None:
        try:
            print("Chiusura del database vettoriale...")
            st.session_state.vector_store.close()
        except Exception as e:
            print(f"Errore nella chiusura del database: {str(e)}")
    
    # Forza la garbage collection
    gc.collect()
    print("Pulizia completata.")

# Registra la funzione cleanup per l'esecuzione all'uscita
atexit.register(cleanup_resources)

# Funzione per eliminare un file in modo sicuro, anche se bloccato
def safe_delete(path, max_attempts=5, delay=1):
    """
    Tenta di eliminare un file o una directory più volte
    
    :param path: Percorso del file o directory da eliminare
    :param max_attempts: Numero massimo di tentativi
    :param delay: Ritardo tra i tentativi in secondi
    :return: True se l'eliminazione ha avuto successo, False altrimenti
    """
    for attempt in range(max_attempts):
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
            return True
        except PermissionError:
            print(f"File in uso, tentativo {attempt+1}/{max_attempts}. Attesa di {delay} secondi...")
            # Forza la garbage collection prima di riprovare
            gc.collect()
            time.sleep(delay)
    
    print(f"Impossibile eliminare {path} dopo {max_attempts} tentativi")
    return False

# Funzione per eliminare il database in un thread separato
def delete_db_async():
    """Elimina il database in un thread separato"""
    try:
        # Chiudi il vector store se è aperto
        if "vector_store" in st.session_state and st.session_state.vector_store is not None:
            st.session_state.vector_store.close()
            st.session_state.vector_store = None
        
        # Rimuovi anche il riferimento al chatbot
        if "chatbot" in st.session_state:
            st.session_state.chatbot = None
        
        # Forza la garbage collection
        gc.collect()
        
        # Attendi che il sistema rilasci le risorse
        time.sleep(1)
        
        # Elimina il database
        success = safe_delete(DB_DIR)
        
        if success:
            print("Database eliminato con successo in background")
        else:
            print("Impossibile eliminare il database in background")
    except Exception as e:
        print(f"Errore nell'eliminazione del database in background: {str(e)}")

# Inizializza le sessioni di stato
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

if "current_model" not in st.session_state:
    st.session_state.current_model = "gemma3:4b"

if "available_models" not in st.session_state:
    st.session_state.available_models = []

if "document_processor" not in st.session_state:
    st.session_state.document_processor = None

if "ocr_enabled" not in st.session_state:
    st.session_state.ocr_enabled = True

if "deletion_requested" not in st.session_state:
    st.session_state.deletion_requested = False

if "show_settings" not in st.session_state:
    st.session_state.show_settings = False

if "active_chatbot_info" not in st.session_state:
    st.session_state.active_chatbot_info = get_active_chatbot()

if "new_chatbot_name" not in st.session_state:
    st.session_state.new_chatbot_name = ""

if "rename_chatbot_name" not in st.session_state:
    st.session_state.rename_chatbot_name = ""

# Funzione per resettare lo stato quando si cambia chatbot
def reset_chatbot_state():
    """Resetta lo stato del chatbot nella sessione"""
    st.session_state.messages = []
    st.session_state.chatbot = None
    st.session_state.vector_store = None
    
    # Inizializza il document processor con il chatbot attivo
    if st.session_state.active_chatbot_info:
        st.session_state.document_processor = DocumentProcessor(
            mistral_api_key=MISTRAL_API_KEY, 
            md_dir=st.session_state.active_chatbot_info["markdown_dir"]
        )
    else:
        st.session_state.document_processor = None

# Inizializza il document processor se è None ma esiste un chatbot attivo
if st.session_state.document_processor is None and st.session_state.active_chatbot_info:
    st.session_state.document_processor = DocumentProcessor(
        mistral_api_key=MISTRAL_API_KEY, 
        md_dir=st.session_state.active_chatbot_info["markdown_dir"]
    )

# Funzione per ottenere i modelli disponibili
def refresh_available_models():
    # Crea temporaneamente un chatbot solo per ottenere i modelli
    try:
        temp_chatbot = Chatbot(vector_store=None, model_name=st.session_state.current_model)
        models = temp_chatbot.get_available_models()
        if not models:
            models = ["gemma3:4b"]  # Fallback se non riesce a ottenere i modelli
        st.session_state.available_models = models
        return models
    except Exception as e:
        st.error(f"Errore nel recupero dei modelli: {str(e)}")
        return ["gemma3:4b"]  # Fallback in caso di errore

# Funzione per cambiare modello
def change_model():
    selected_model = st.session_state.model_selector
    
    if selected_model != st.session_state.current_model:
        # Aggiorna il modello nella sessione
        st.session_state.current_model = selected_model
        
        # Se il chatbot è già inizializzato, aggiorna il suo modello
        if st.session_state.chatbot is not None:
            with st.spinner(f"Cambiando modello a {selected_model}..."):
                success, message = st.session_state.chatbot.change_model(selected_model)
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
        else:
            # Se non è inizializzato, verifica se abbiamo un database
            if st.session_state.vector_store is not None:
                # Inizializza il chatbot con il modello selezionato
                st.session_state.chatbot = Chatbot(vector_store=st.session_state.vector_store, 
                                                 model_name=selected_model)
                st.success(f"Inizializzato chatbot con modello {selected_model}")
            else:
                # Solo aggiorna il modello selezionato per uso futuro
                st.info(f"Modello cambiato a {selected_model}, ma sarà attivo solo dopo aver caricato un database.")

# Funzione per attivare/disattivare OCR
def toggle_ocr():
    st.session_state.ocr_enabled = not st.session_state.ocr_enabled
    
    if st.session_state.active_chatbot_info:
        # Verifica che document_processor sia inizializzato
        if st.session_state.document_processor is None:
            try:
                st.session_state.document_processor = DocumentProcessor(
                    mistral_api_key=MISTRAL_API_KEY, 
                    md_dir=st.session_state.active_chatbot_info["markdown_dir"]
                )
            except Exception as e:
                st.error(f"Impossibile inizializzare Document Processor: {str(e)}")
                return
        
        if st.session_state.ocr_enabled:
            # Riattiva l'OCR
            st.session_state.document_processor.set_mistral_api_key(
                MISTRAL_API_KEY, 
                md_dir=st.session_state.active_chatbot_info["markdown_dir"]
            )
            st.success("OCR Mistral attivato")
        else:
            # Disattiva l'OCR
            st.session_state.document_processor.ocr_processor = None
            st.info("OCR Mistral disattivato")
    else:
        st.warning("Nessun chatbot attivo")

# Funzione per eliminare completamente un documento dal RAG
def delete_document(document_name):
    """
    Elimina un documento dal sistema RAG, inclusi file PDF, Markdown e dati vettoriali
    
    :param document_name: Nome del documento da eliminare (senza estensione)
    """
    if not st.session_state.active_chatbot_info:
        st.warning("Nessun chatbot attivo selezionato")
        return
    
    # Nome completo del documento
    base_name = os.path.splitext(document_name)[0]
    
    # Ottieni i percorsi dai dati del chatbot attivo
    PDF_DIR = st.session_state.active_chatbot_info["data_dir"]
    MD_DIR = st.session_state.active_chatbot_info["markdown_dir"]
    DB_DIR = st.session_state.active_chatbot_info["chroma_db_dir"]
    
    with st.spinner(f"Rimozione di {document_name} dal sistema..."):
        # 1. Elimina il file PDF se esiste
        pdf_path = os.path.join(PDF_DIR, document_name)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            st.info(f"File PDF eliminato: {pdf_path}")
        
        # 2. Elimina il file Markdown se esiste
        md_path = os.path.join(MD_DIR, f"{base_name}.md")
        if os.path.exists(md_path):
            os.remove(md_path)
            st.info(f"File Markdown eliminato: {md_path}")
        
        # 3. Ricrea il database vettoriale senza il documento
        if os.path.exists(DB_DIR):
            try:
                # Chiudi qualsiasi connessione al database prima di eliminarlo
                if st.session_state.vector_store is not None:
                    # Chiudi il database utilizzando il metodo dedicato
                    st.session_state.vector_store.close()
                    # Rimuovi il riferimento
                    st.session_state.vector_store = None
                
                # Rimuovi anche il riferimento al chatbot
                st.session_state.chatbot = None
                
                # Forza la garbage collection
                gc.collect()
                
                # Attendi un attimo che il sistema rilasci le risorse
                time.sleep(0.5)
                
                # Elimina completamente il database
                if safe_delete(DB_DIR):
                    st.info(f"Database vettoriale eliminato")
                    
                    # Ricrea il database con i documenti rimanenti
                    st.info("Ricostruzione del database vettoriale in corso...")
                    process_documents()
                else:
                    st.warning("Impossibile eliminare completamente il database. Verrà tentata l'eliminazione in background.")
                    # Imposta il flag per eliminare il database al prossimo avvio
                    st.session_state.deletion_requested = True
                    threading.Thread(target=delete_db_async).start()
            except Exception as e:
                st.error(f"Impossibile eliminare il database: {str(e)}")
                st.warning("Il database è ancora in uso. Prova a riavviare l'applicazione per eliminarlo completamente.")
        
        st.success(f"Documento '{document_name}' rimosso dal sistema")
        st.rerun()

# Funzione per caricare file
def upload_files(uploaded_files):
    if not st.session_state.active_chatbot_info:
        st.warning("Nessun chatbot attivo selezionato")
        return [], []
    
    # Verifica che document_processor sia inizializzato
    if st.session_state.document_processor is None:
        # Tenta di inizializzarlo
        try:
            st.session_state.document_processor = DocumentProcessor(
                mistral_api_key=MISTRAL_API_KEY, 
                md_dir=st.session_state.active_chatbot_info["markdown_dir"]
            )
            st.success("Document Processor inizializzato automaticamente")
        except Exception as e:
            st.error(f"Impossibile inizializzare Document Processor: {str(e)}")
            return [], []
    
    # Ottieni le directory dal chatbot attivo
    PDF_DIR = st.session_state.active_chatbot_info["data_dir"]
    
    # Ottieni i tipi di file supportati
    supported_extensions = get_supported_extensions()
    supported_str = ", ".join(ext[1:] for ext in supported_extensions)  # Rimuovi il punto iniziale
    
    # Crea la directory dei dati se non esiste
    os.makedirs(PDF_DIR, exist_ok=True)
    
    processed_files = []
    md_files = []
    
    # Salva i file caricati nella directory
    for uploaded_file in uploaded_files:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_ext not in supported_extensions:
            st.warning(f"Il file {uploaded_file.name} non è supportato. Formati supportati: {supported_str}")
            continue
            
        try:
            # Processa il file caricato
            file_path, md_file_path = st.session_state.document_processor.process_uploaded_file(
                uploaded_file, PDF_DIR
            )
            
            processed_files.append(file_path)
            if md_file_path:
                md_files.append(md_file_path)
                st.success(f"File salvato e convertito in Markdown: {uploaded_file.name}")
            else:
                st.success(f"File salvato: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Errore nel processare {uploaded_file.name}: {str(e)}")
    
    return processed_files, md_files

# Funzione per elaborare i documenti e creare il database vettoriale
def process_documents():
    if not st.session_state.active_chatbot_info:
        st.warning("Nessun chatbot attivo selezionato")
        return
    
    # Verifica che document_processor sia inizializzato
    if st.session_state.document_processor is None:
        try:
            st.session_state.document_processor = DocumentProcessor(
                mistral_api_key=MISTRAL_API_KEY, 
                md_dir=st.session_state.active_chatbot_info["markdown_dir"]
            )
            st.success("Document Processor inizializzato automaticamente")
        except Exception as e:
            st.error(f"Impossibile inizializzare Document Processor: {str(e)}")
            return
    
    # Ottieni le directory dal chatbot attivo
    PDF_DIR = st.session_state.active_chatbot_info["data_dir"]
    DB_DIR = st.session_state.active_chatbot_info["chroma_db_dir"]
    
    # Elabora i documenti con il document processor
    with st.spinner("Caricamento e divisione dei documenti..."):
        documents = st.session_state.document_processor.load_and_split_documents(PDF_DIR)
    
    if not documents:
        st.warning("Nessun documento trovato o elaborato.")
        return
    
    # Inizializza il database vettoriale
    vector_store = VectorStore(persist_directory=DB_DIR)
    
    # Crea il database vettoriale dai documenti
    with st.spinner("Creazione del database vettoriale..."):
        db = vector_store.create_from_documents(documents)
    
    if db is None:
        st.error("Errore nella creazione del database vettoriale.")
        return
    
    # Salva il database in session_state
    st.session_state.vector_store = db
    
    # Inizializza o aggiorna il chatbot
    if st.session_state.chatbot is None:
        # Usa il modello corrente
        templates_file = st.session_state.active_chatbot_info["templates_file"]
        st.session_state.chatbot = Chatbot(
            vector_store=db, 
            model_name=st.session_state.current_model,
            templates_file=templates_file
        )
        st.success(f"Chatbot inizializzato con modello {st.session_state.current_model}")
    else:
        st.session_state.chatbot.set_vector_store(db)
    
    st.success(f"Database creato con successo: {len(documents)} chunks.")

# Funzione per elaborare solo i file Markdown
def process_markdown_only():
    if not st.session_state.active_chatbot_info:
        st.warning("Nessun chatbot attivo selezionato")
        return
    
    # Ottieni le directory dal chatbot attivo
    MD_DIR = st.session_state.active_chatbot_info["markdown_dir"]
    DB_DIR = st.session_state.active_chatbot_info["chroma_db_dir"]
    
    if not os.path.exists(MD_DIR):
        st.warning("Nessuna directory Markdown trovata. Impossibile procedere.")
        return
    
    md_files = [f for f in os.listdir(MD_DIR) if f.endswith('.md')]
    
    if not md_files:
        st.warning("Nessun file Markdown trovato. Impossibile procedere.")
        return
    
    with st.spinner(f"Elaborazione di {len(md_files)} file Markdown..."):
        documents = []
        
        for md_file in md_files:
            md_path = os.path.join(MD_DIR, md_file)
            try:
                # Carica il file Markdown
                loader = TextLoader(md_path, encoding="utf-8")
                content = loader.load()
                
                # Dividi il documento in chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                split_content = text_splitter.split_documents(content)
                documents.extend(split_content)
                
                st.info(f"Elaborato {md_file}: {len(split_content)} chunks")
            except Exception as e:
                st.error(f"Errore nell'elaborazione di {md_file}: {str(e)}")
    
    if not documents:
        st.warning("Nessun documento è stato elaborato correttamente.")
        return
    
    # Inizializza il database vettoriale
    vector_store = VectorStore(persist_directory=DB_DIR)
    
    # Crea il database vettoriale dai documenti
    with st.spinner("Creazione del database vettoriale..."):
        db = vector_store.create_from_documents(documents)
    
    if db is None:
        st.error("Errore nella creazione del database vettoriale.")
        return
    
    # Salva il database in session_state
    st.session_state.vector_store = db
    
    # Inizializza o aggiorna il chatbot
    if st.session_state.chatbot is None:
        # Usa il modello corrente
        st.session_state.chatbot = Chatbot(vector_store=db, model_name=st.session_state.current_model)
        st.success(f"Chatbot inizializzato con modello {st.session_state.current_model}")
    else:
        st.session_state.chatbot.set_vector_store(db)
    
    st.success(f"Database creato con successo usando solo file Markdown: {len(documents)} chunks totali.")

# Funzione per caricare il database esistente
def load_database():
    if not st.session_state.active_chatbot_info:
        st.warning("Nessun chatbot attivo selezionato")
        return
    
    # Ottieni la directory del database dal chatbot attivo
    DB_DIR = st.session_state.active_chatbot_info["chroma_db_dir"]
    
    # Inizializza il database vettoriale
    vector_store = VectorStore(persist_directory=DB_DIR)
    
    # Carica il database
    with st.spinner("Caricamento del database vettoriale..."):
        db = vector_store.load()
    
    if db is None:
        st.error("Nessun database trovato.")
        return
    
    # Salva il database in session_state
    st.session_state.vector_store = db
    
    # Inizializza o aggiorna il chatbot
    if st.session_state.chatbot is None:
        templates_file = st.session_state.active_chatbot_info["templates_file"]
        
        # Usa il modello corrente
        st.session_state.chatbot = Chatbot(
            vector_store=db, 
            model_name=st.session_state.current_model,
            templates_file=templates_file
        )
        st.success(f"Chatbot inizializzato con modello {st.session_state.current_model}")
    else:
        st.session_state.chatbot.set_vector_store(db)
    
    st.success("Database caricato con successo.")

# Funzione per gestire la chat
def handle_chat():
    """Gestisce l'interfaccia di chat e le domande dell'utente"""
    # Aggiungo CSS per migliorare l'interfaccia di chat
    st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 500px;
        overflow-y: auto;
        padding-bottom: 80px;
        margin-bottom: 100px;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 10px;
        border-top: 1px solid #e0e0e0;
        z-index: 1000;
    }
    .stChatInputContainer {
        position: sticky !important;
        bottom: 20px !important;
        background-color: white !important;
        padding: 10px !important;
        border-radius: 10px !important;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1) !important;
        margin-top: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Crea un container per la chat con altezza fissa e scrollabile
    with st.container(height=500, border=False):
        # Visualizza tutti i messaggi memorizzati
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Verifica se il chatbot è inizializzato
    if not st.session_state.chatbot:
        st.info("ℹ️ Carica un database o elabora dei documenti per iniziare a chattare")
        return

    # Gestione dell'input utente
    prompt = st.chat_input("Fai una domanda sui tuoi documenti...")
    
    if prompt:
        # Aggiungi la domanda dell'utente alla cronologia
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Visualizza la domanda dell'utente
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Mostra la risposta dell'assistente con animazione
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Sto pensando..."):
                try:
                    # Ottieni la risposta dal chatbot
                    response = st.session_state.chatbot.get_answer(prompt)
                    
                    # Debug dell'output per diagnosticare problemi
                    if st.session_state.debug_mode:
                        st.write("Debug - Tipo di risposta:", type(response))
                        st.write("Debug - Contenuto:", response)
                    
                    # Estrai testo e documenti di origine
                    if isinstance(response, dict):
                        answer = response.get("answer", "")
                        source_docs = response.get("source_documents", [])
                    elif isinstance(response, str):
                        answer = response
                        source_docs = []
                    else:
                        answer = str(response)
                        source_docs = []
                    
                    # Se l'answer è vuoto, imposta un messaggio di errore
                    if not answer or answer.strip() == "":
                        answer = "Mi dispiace, non sono riuscito a generare una risposta. Prova a riformulare la domanda."
                    
                    # Formatta i documenti consultati
                    formatted_docs = ""
                    if source_docs and len(source_docs) > 0:
                        formatted_docs = "\n\n---\n\n**Documenti consultati:**\n"
                        for i, doc in enumerate(source_docs, 1):
                            # Estrai nome del file dalla metadata
                            source = doc.metadata.get("source", "Fonte sconosciuta")
                            if isinstance(source, str) and os.path.isfile(source):
                                source = os.path.basename(source)
                            
                            # Aggiungi frammento utilizzato
                            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                            formatted_docs += f"\n{i}. **{source}**: {content_preview}\n"
                    
                    # Mostra sempre i documenti consultati
                    final_answer = answer
                    if formatted_docs:
                        final_answer += formatted_docs
                    
                    # Visualizza la risposta
                    message_placeholder.markdown(final_answer)
                    
                    # Aggiungi la risposta dell'assistente alla cronologia
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                except Exception as e:
                    # Gestisci eventuali errori
                    error_msg = f"Si è verificato un errore: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Bottone per cancellare la chat
    if st.button("🗑️ Cancella chat"):
        st.session_state.messages = []
        st.rerun()

# Funzione per visualizzare la lista dei file nei vari componenti
def list_files_and_data():
    if not st.session_state.active_chatbot_info:
        st.warning("Nessun chatbot attivo selezionato")
        return
    
    # Ottieni le directory dal chatbot attivo
    PDF_DIR = st.session_state.active_chatbot_info["data_dir"]
    MD_DIR = st.session_state.active_chatbot_info["markdown_dir"]
    DB_DIR = st.session_state.active_chatbot_info["chroma_db_dir"]
    
    # Crea tabs per i diversi tipi di file
    tab1, tab2, tab3 = st.tabs(["Documenti PDF", "File Markdown", "Database"])
    
    # Tab 1: Documenti PDF
    with tab1:
        st.subheader("Documenti PDF")
        if not os.path.exists(PDF_DIR):
            st.info("Cartella dei documenti non trovata")
        else:
            pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                st.info("Nessun documento PDF trovato")
            else:
                st.write(f"{len(pdf_files)} documenti trovati:")
                
                for pdf_file in pdf_files:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"📄 {pdf_file}")
                    with col2:
                        if st.button("Elimina", key=f"del_{pdf_file}"):
                            delete_document(pdf_file)
    
    # Tab 2: File Markdown
    with tab2:
        st.subheader("File Markdown")
        if not os.path.exists(MD_DIR):
            st.info("Cartella Markdown non trovata")
        else:
            md_files = [f for f in os.listdir(MD_DIR) if f.endswith('.md')]
            
            if not md_files:
                st.info("Nessun file Markdown trovato")
            else:
                st.write(f"{len(md_files)} file Markdown trovati:")
                
                for md_file in md_files:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"📝 {md_file}")
                    with col2:
                        if st.button("Visualizza", key=f"view_{md_file}"):
                            with open(os.path.join(MD_DIR, md_file), "r", encoding="utf-8") as f:
                                content = f.read()
                            
                            with st.expander(f"Contenuto di {md_file}", expanded=True):
                                st.markdown(content)
                    with col3:
                        if st.button("Elimina", key=f"del_md_{md_file}"):
                            # Estrai il nome base (senza estensione)
                            base_name = os.path.splitext(md_file)[0]
                            # Cerca il PDF corrispondente
                            pdf_candidates = [f for f in os.listdir(PDF_DIR) if os.path.splitext(f)[0] == base_name]
                            if pdf_candidates:
                                delete_document(pdf_candidates[0])
                            else:
                                # Se non c'è un PDF, elimina solo il file Markdown
                                os.remove(os.path.join(MD_DIR, md_file))
                                st.success(f"File Markdown {md_file} eliminato")
                                st.rerun()
    
    # Tab 3: Database
    with tab3:
        st.subheader("Database vettoriale")
        if not os.path.exists(DB_DIR):
            st.info("Database vettoriale non trovato")
        else:
            st.success("Database vettoriale trovato")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Elimina database"):
                    if os.path.exists(DB_DIR):
                        try:
                            # Chiudi esplicitamente qualsiasi connessione al database
                            if st.session_state.vector_store is not None:
                                # Chiudi il database utilizzando il metodo dedicato
                                st.session_state.vector_store.close()
                                # Rimuovi il riferimento
                                st.session_state.vector_store = None
                            
                            # Rimuovi anche il riferimento al chatbot
                            st.session_state.chatbot = None
                            
                            # Forza la garbage collection per liberare risorse
                            gc.collect()
                            
                            # Attendi un attimo che il sistema rilasci le risorse
                            time.sleep(0.5)
                            
                            # Elimina il database
                            if safe_delete(DB_DIR):
                                st.success("Database vettoriale eliminato completamente")
                            else:
                                st.warning("Impossibile eliminare completamente il database. Verrà tentata l'eliminazione in background.")
                                # Imposta il flag per eliminare il database al prossimo avvio
                                st.session_state.deletion_requested = True
                                threading.Thread(target=delete_db_async).start()
                            
                            st.rerun()
                        except Exception as e:
                            st.error(f"Impossibile eliminare il database: {str(e)}")
                            st.warning("Il database è ancora in uso. Per eliminarlo, riavvia l'applicazione Streamlit e poi prova di nuovo.")
                            st.info("Alternativa: chiudi completamente l'applicazione Streamlit (CTRL+C nel terminale) e cancella manualmente la cartella 'chroma_db'")
            
            with col2:
                if st.button("Elimina in background"):
                    st.session_state.deletion_requested = True
                    st.warning("Eliminazione del database pianificata. Il database verrà eliminato in background.")
                    # Avvia un thread per eliminare il database in background
                    threading.Thread(target=delete_db_async).start()
                    time.sleep(1)  # Attendi un secondo prima di ricaricare
                    st.rerun()

# Verifica se è stata richiesta l'eliminazione del database
if st.session_state.deletion_requested and os.path.exists(DB_DIR):
    st.warning("Eliminazione del database in corso...")
    threading.Thread(target=delete_db_async).start()
    st.session_state.deletion_requested = False
    time.sleep(1)  # Attendi un secondo prima di ricaricare
    st.rerun()

# Funzione per ottenere le estensioni supportate in modo sicuro
def get_supported_extensions():
    """
    Ottiene le estensioni supportate dal document processor o restituisce valori di default
    se il document processor non è disponibile
    
    :return: Lista di estensioni supportate
    """
    if st.session_state.document_processor is not None:
        return st.session_state.document_processor.get_supported_extensions()
    # Valori di default se document_processor non è disponibile
    return ['.pdf', '.docx', '.txt', '.md', '.csv', '.xlsx']

# Interfaccia utente principale
def main():
    # Carica la configurazione dei chatbot
    chatbots_config = load_chatbots_config()
    
    # Header con logo e titolo
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/4616/4616271.png", width=100)
    
    with col2:
        st.title("🤖 RAG Chatbot")
        st.markdown("Un chatbot che risponde alle domande basandosi sui tuoi documenti")
    
    with col3:
        # Container per il selettore di modello nell'header
        with st.container():
            # Aggiorna l'elenco dei modelli se necessario
            if not st.session_state.available_models:
                refresh_available_models()
                
            model_options = st.session_state.available_models
            model_index = model_options.index(st.session_state.current_model) if st.session_state.current_model in model_options else 0
            
            st.selectbox(
                "Modello LLM",
                options=model_options,
                index=model_index,
                key="model_selector",
                on_change=change_model
            )
            
            model_status = "🟢" if st.session_state.chatbot is not None else "⚠️"
            st.caption(f"{model_status} {st.session_state.current_model}")
            
            # Pulsante per aprire/chiudere le impostazioni
            if st.button("⚙️ Impostazioni"):
                st.session_state.show_settings = not st.session_state.show_settings
    
    # Sezione per la gestione dei chatbot multipli
    st.subheader("🤖 Gestione Chatbot Multipli")
    
    chatbot_cols = st.columns([2, 3, 2])
    
    with chatbot_cols[0]:
        # Creazione nuovo chatbot
        with st.expander("➕ Crea Nuovo Chatbot", expanded=False):
            st.text_input("Nome del nuovo chatbot", key="new_chatbot_name")
            
            if st.button("Crea Chatbot") and st.session_state.new_chatbot_name:
                chatbot_info, message = create_chatbot(st.session_state.new_chatbot_name)
                
                if chatbot_info:
                    st.success(message)
                    # Imposta il nuovo chatbot come attivo
                    st.session_state.active_chatbot_info = set_active_chatbot(chatbot_info["id"])
                    # Resetta lo stato
                    reset_chatbot_state()
                    # Aggiorna la pagina
                    st.rerun()
                else:
                    st.error(message)
    
    with chatbot_cols[1]:
        # Lista chatbot disponibili
        chatbots = chatbots_config.get("chatbots", [])
        
        if not chatbots:
            st.info("Non ci sono chatbot disponibili. Creane uno nuovo per iniziare.")
        else:
            # Crea una lista di nomi dei chatbot
            chatbot_names = [c["name"] for c in chatbots]
            chatbot_ids = [c["id"] for c in chatbots]
            
            # Trova l'indice del chatbot attivo
            active_id = chatbots_config.get("active_chatbot")
            selected_index = chatbot_ids.index(active_id) if active_id in chatbot_ids else 0
            
            # Selettore di chatbot
            selected_chatbot = st.selectbox(
                "Seleziona Chatbot",
                options=chatbot_names,
                index=selected_index,
                key="selected_chatbot_name"
            )
            
            # Trova l'ID del chatbot selezionato
            selected_index = chatbot_names.index(selected_chatbot)
            selected_id = chatbot_ids[selected_index]
            
            # Se il chatbot selezionato è diverso da quello attivo, cambialo
            if selected_id != active_id:
                st.session_state.active_chatbot_info = set_active_chatbot(selected_id)
                reset_chatbot_state()
                st.success(f"Chatbot '{selected_chatbot}' attivato")
                st.rerun()
    
    with chatbot_cols[2]:
        # Gestione chatbot esistente
        if st.session_state.active_chatbot_info:
            active_name = st.session_state.active_chatbot_info["name"]
            
            with st.expander(f"⚙️ Gestisci '{active_name}'", expanded=False):
                # Rinomina chatbot
                st.text_input("Nuovo nome", key="rename_chatbot_name")
                
                if st.button("Rinomina") and st.session_state.rename_chatbot_name:
                    success, message = rename_chatbot(
                        st.session_state.active_chatbot_info["id"], 
                        st.session_state.rename_chatbot_name
                    )
                    
                    if success:
                        st.success(message)
                        # Aggiorna i dati del chatbot attivo
                        st.session_state.active_chatbot_info = get_active_chatbot()
                        st.rerun()
                    else:
                        st.error(message)
                
                st.divider()
                
                # Elimina chatbot con conferma
                st.warning("⚠️ Questa operazione è irreversibile!")
                delete_confirm = st.checkbox("Conferma eliminazione")
                
                if st.button("Elimina Chatbot", type="primary", disabled=not delete_confirm):
                    success, message = delete_chatbot(st.session_state.active_chatbot_info["id"])
                    
                    if success:
                        st.success(message)
                        # Resetta lo stato
                        st.session_state.active_chatbot_info = get_active_chatbot()
                        reset_chatbot_state()
                        st.rerun()
                    else:
                        st.error(message)
    
    # Verifica se c'è un chatbot attivo
    if not st.session_state.active_chatbot_info:
        st.warning("Nessun chatbot attivo. Seleziona o crea un chatbot per iniziare.")
        # Nascondi il resto dell'interfaccia se non c'è un chatbot attivo
        return
    
    # Verifica se document_processor è None e inizializzalo se necessario
    if st.session_state.document_processor is None:
        st.session_state.document_processor = DocumentProcessor(
            mistral_api_key=MISTRAL_API_KEY, 
            md_dir=st.session_state.active_chatbot_info["markdown_dir"]
        )
    
    # Finestra modale per le impostazioni
    if st.session_state.show_settings:
        with st.sidebar:
            st.header("⚙️ Impostazioni")
            
            # Tabs per organizzare le impostazioni
            settings_tabs = st.tabs(["Generale", "Documenti", "Template", "Stato"])
            
            # Tab Generale
            with settings_tabs[0]:
                st.subheader("OCR Mistral")
                
                ocr_status = "Attivo 🟢" if st.session_state.ocr_enabled else "Disattivato 🔴"
                st.info(f"Stato OCR: {ocr_status}")
                
                if st.button("Attiva/Disattiva OCR"):
                    toggle_ocr()
                
                # Tipi di file supportati - Usa la funzione sicura
                supported_extensions = get_supported_extensions()
                supported_str = ", ".join(ext[1:] for ext in supported_extensions)
                st.info(f"Formati supportati: {supported_str}")
                
                # Debug mode
                st.checkbox("Modalità debug", value=st.session_state.debug_mode, key="debug_mode")
            
            # Tab Documenti
            with settings_tabs[1]:
                st.subheader("Caricamento Documenti")
                
                # Usa la funzione sicura per ottenere le estensioni supportate
                supported_extensions = get_supported_extensions()
                
                uploaded_files = st.file_uploader(
                    "Carica i tuoi documenti", 
                    type=[ext[1:] for ext in supported_extensions], 
                    accept_multiple_files=True
                )
                
                if uploaded_files:
                    if st.button("Salva documenti"):
                        upload_files(uploaded_files)
                
                # Elaborazione documenti
                st.subheader("Elaborazione Documenti")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Elabora tutti i documenti"):
                        process_documents()
                with col2:
                    if st.button("Usa solo file Markdown"):
                        process_markdown_only()
                
                # Caricamento database esistente
                if st.button("Carica database esistente"):
                    load_database()
            
            # Tab Template
            with settings_tabs[2]:
                st.subheader("Template RAG")
                
                # Funzione per salvare i template
                def save_template_settings():
                    if st.session_state.chatbot is not None:
                        if not hasattr(st.session_state.chatbot, 'templates_file') or not st.session_state.chatbot.templates_file:
                            # Se manca il percorso del file, prova a recuperarlo dal chatbot attivo
                            if st.session_state.active_chatbot_info:
                                st.session_state.chatbot.templates_file = st.session_state.active_chatbot_info["templates_file"]
                                st.info(f"Percorso template impostato a: {st.session_state.chatbot.templates_file}")
                            else:
                                st.error("Impossibile salvare: nessun percorso template disponibile")
                                return
                    
                    templates = {
                        "qa_template": st.session_state.qa_template,
                        "refine_template": st.session_state.refine_template
                    }
                    success = st.session_state.chatbot.save_templates(templates)
                    if success:
                        st.success("Template salvati con successo")
                    else:
                        st.error("Errore nel salvataggio dei template")
                
                # Carica i template correnti
                if st.session_state.chatbot is not None:
                    current_templates = st.session_state.chatbot.templates
                    
                    # Template per la query iniziale
                    st.text_area(
                        "Template per la query iniziale",
                        value=current_templates["qa_template"],
                        height=200,
                        key="qa_template",
                        help="Template utilizzato per la query iniziale. Usa {context} e {question} come variabili."
                    )
                    
                    # Template per il raffinamento
                    st.text_area(
                        "Template per il raffinamento",
                        value=current_templates["refine_template"],
                        height=200,
                        key="refine_template",
                        help="Template utilizzato per raffinare la risposta. Usa {existing_answer}, {context} e {question} come variabili."
                    )
                    
                    # Pulsante per salvare
                    if st.button("Salva template"):
                        save_template_settings()
                        
                    # Info sul funzionamento
                    with st.expander("Come funziona"):
                        st.info("""
                        1. Il template iniziale viene utilizzato per generare una prima risposta basata sul contesto.
                        2. Il template di raffinamento viene utilizzato per migliorare la risposta utilizzando ulteriori frammenti di contesto.
                        3. Questo approccio produce risposte più complete e accurate, strettamente basate sui documenti.
                        """)
                else:
                    st.warning("Inizializza prima un chatbot caricando un database per personalizzare i template")
            
            # Tab Stato
            with settings_tabs[3]:
                st.subheader("Stato Sistema")
                
                status_col1, status_col2 = st.columns(2)
                with status_col1:
                    if st.session_state.vector_store is not None:
                        st.success("Database vettoriale")
                    else:
                        st.error("Database vettoriale")
                with status_col2:
                    if st.session_state.chatbot is not None:
                        st.success("Chatbot")
                    else:
                        st.error("Chatbot")
                
                # Stato OCR
                ocr_col1, ocr_col2 = st.columns(2)
                with ocr_col1:
                    if st.session_state.ocr_enabled:
                        st.success("OCR Attivo")
                    else:
                        st.warning("OCR Disattivato")
                with ocr_col2:
                    if st.session_state.debug_mode:
                        st.info("Debug Attivo")
                    else:
                        st.info("Debug Disattivato")
                
                # Aggiorna modelli
                if st.button("Aggiorna modelli disponibili"):
                    refresh_available_models()
                
                # Stato del Document Processor
                doc_processor_status = "Attivo 🟢" if st.session_state.document_processor is not None else "Non inizializzato ⚠️"
                st.info(f"Document Processor: {doc_processor_status}")
                
                # Pulsante per reinizializzare il document processor
                if st.button("Reinizializza Document Processor"):
                    if st.session_state.active_chatbot_info:
                        st.session_state.document_processor = DocumentProcessor(
                            mistral_api_key=MISTRAL_API_KEY, 
                            md_dir=st.session_state.active_chatbot_info["markdown_dir"]
                        )
                        st.success("Document Processor reinizializzato")
                    else:
                        st.warning("Nessun chatbot attivo, impossibile inizializzare")
            
            # Pulsante per chiudere le impostazioni
            if st.button("Chiudi impostazioni"):
                st.session_state.show_settings = False
    
    # Area principale dell'applicazione
    tab1, tab2 = st.tabs(["Chat", "Gestione File"])
    
    with tab1:
        # Interfaccia di chat
        handle_chat()
    
    with tab2:
        # Nota informativa sulla priorità dei file
        st.info("📝 **NOTA SULLA PRIORITÀ DEI FILE**: Il sistema utilizza prioritariamente i file Markdown quando disponibili. Se un file PDF ha un corrispondente file Markdown nella cartella del chatbot, verrà utilizzato quest'ultimo per il database vettoriale.")
        
        # Lista dei file e gestione
        list_files_and_data()

if __name__ == "__main__":
    main() 
