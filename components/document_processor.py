import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional, Tuple
from components.ocr_processor import MistralOCRProcessor

class DocumentProcessor:
    def __init__(self, mistral_api_key=None, md_dir="markdown_data"):
        """
        Inizializza il processore di documenti
        
        :param mistral_api_key: Chiave API di Mistral per l'OCR
        :param md_dir: Directory per i file Markdown
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Salva il percorso della directory Markdown
        self.md_dir = md_dir
        
        # Inizializza il processore OCR se è stata fornita la chiave API
        self.ocr_processor = None
        if mistral_api_key:
            self.ocr_processor = MistralOCRProcessor(api_key=mistral_api_key, md_dir=md_dir)
    
    def set_mistral_api_key(self, api_key, md_dir="markdown_data"):
        """
        Imposta la chiave API di Mistral e inizializza l'OCR processor
        
        :param api_key: Chiave API di Mistral
        :param md_dir: Directory per i file Markdown
        """
        self.md_dir = md_dir
        self.ocr_processor = MistralOCRProcessor(api_key=api_key, md_dir=md_dir)
        
    def get_supported_extensions(self):
        """
        Restituisce le estensioni di file supportate
        
        :return: Lista di estensioni supportate
        """
        if self.ocr_processor:
            # Se OCR è disponibile, supporta più formati
            return ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.docx', '.doc', '.xlsx', '.xls', '.pptx']
        else:
            # Altrimenti supporta solo PDF
            return ['.pdf']
    
    def _process_with_ocr(self, file_path):
        """
        Processa un file con OCR e restituisce il percorso del file Markdown
        
        :param file_path: Percorso del file da processare
        :return: Percorso del file Markdown generato
        """
        _, md_file_path = self.ocr_processor.process_document_from_path(file_path)
        return md_file_path
    
    def _is_supported_extension(self, file_path):
        """
        Verifica se l'estensione del file è supportata
        
        :param file_path: Percorso del file
        :return: True se l'estensione è supportata, False altrimenti
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.get_supported_extensions()
    
    def _find_matching_md_file(self, file_path):
        """
        Cerca un file Markdown corrispondente al file originale
        
        :param file_path: Percorso del file originale
        :return: Percorso del file Markdown se esiste, None altrimenti
        """
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        md_file_path = os.path.join(self.md_dir, f"{base_name}.md")
        
        if os.path.exists(md_file_path):
            print(f"Trovato file Markdown corrispondente: {md_file_path}")
            return md_file_path
        
        return None
    
    def _load_markdown_file(self, md_file_path):
        """
        Carica un file Markdown e lo divide in chunks
        
        :param md_file_path: Percorso del file Markdown
        :return: Lista di documenti divisi in chunks
        """
        try:
            loader = TextLoader(md_file_path, encoding="utf-8")
            md_content = loader.load()
            
            # Dividi il documento in chunks
            split_content = self.text_splitter.split_documents(md_content)
            print(f"File Markdown caricato: {len(split_content)} chunks generati")
            
            return split_content
        except Exception as e:
            print(f"Errore nel caricare il file Markdown {md_file_path}: {str(e)}")
            return []
    
    def load_and_split_documents(self, documents_directory):
        """
        Carica i documenti dalla directory specificata e li divide in chunks.
        Dà priorità ai file Markdown esistenti rispetto ai PDF originali.
        
        :param documents_directory: Percorso alla directory contenente i file
        :return: Lista di documenti divisi in chunks
        """
        documents = []
        
        # Verifica se la directory esiste
        if not os.path.exists(documents_directory):
            print(f"La directory {documents_directory} non esiste")
            return documents
        
        # Verifica se la directory Markdown esiste
        if not os.path.exists(self.md_dir):
            os.makedirs(self.md_dir, exist_ok=True)
            print(f"Creata directory Markdown: {self.md_dir}")
        
        # Ottieni la lista dei file supportati nella directory
        supported_exts = self.get_supported_extensions()
        supported_files = []
        
        for f in os.listdir(documents_directory):
            file_path = os.path.join(documents_directory, f)
            if os.path.isfile(file_path) and any(f.lower().endswith(ext) for ext in supported_exts):
                supported_files.append(f)
        
        if not supported_files:
            print(f"Nessun file supportato trovato in {documents_directory}")
            return documents
        
        # Carica e processa ogni file
        for file_name in supported_files:
            file_path = os.path.join(documents_directory, file_name)
            print(f"Elaborazione di {file_name}...")
            
            try:
                # 1. Verifica se esiste un file Markdown corrispondente
                md_file_path = self._find_matching_md_file(file_path)
                
                if md_file_path:
                    # Se esiste un file Markdown, caricalo
                    print(f"Utilizzando file Markdown esistente per {file_name}")
                    split_content = self._load_markdown_file(md_file_path)
                    documents.extend(split_content)
                elif file_name.lower().endswith('.pdf') and not self.ocr_processor:
                    # Carica direttamente il PDF se non è disponibile l'OCR e non esiste un MD
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    
                    # Dividi il documento in chunks
                    split_pages = self.text_splitter.split_documents(pages)
                    documents.extend(split_pages)
                    
                    print(f"Processato {file_name}: {len(split_pages)} chunks generati")
                else:
                    # Usa OCR per altri formati o se OCR è richiesto anche per PDF
                    if self.ocr_processor:
                        print(f"Utilizzo OCR per {file_name}...")
                        md_file_path = self._process_with_ocr(file_path)
                        
                        # Carica il file Markdown generato
                        split_content = self._load_markdown_file(md_file_path)
                        documents.extend(split_content)
                    else:
                        print(f"Il file {file_name} non può essere elaborato senza OCR. Fornire una chiave API Mistral.")
                
            except Exception as e:
                print(f"Errore nel processare {file_name}: {str(e)}")
                continue
        
        return documents
    
    def process_uploaded_file(self, uploaded_file, save_dir):
        """
        Processa un file caricato tramite Streamlit
        
        :param uploaded_file: File caricato tramite Streamlit
        :param save_dir: Directory dove salvare il file
        :return: Percorso del file salvato e del file Markdown generato (se applicabile)
        """
        # Crea la directory se non esiste
        os.makedirs(save_dir, exist_ok=True)
        
        # Salva il file caricato
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Processa con OCR se disponibile e supportato
        md_file_path = None
        if self.ocr_processor and self._is_supported_extension(file_path):
            try:
                _, md_file_path = self.ocr_processor.process_document_from_path(file_path)
            except Exception as e:
                print(f"Errore nell'elaborazione OCR: {str(e)}")
        
        return file_path, md_file_path 