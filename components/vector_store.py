import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStore:
    def __init__(self, persist_directory="chroma_db"):
        """
        Inizializza un database vettoriale
        
        :param persist_directory: Directory in cui persistere il database
        """
        self.persist_directory = persist_directory
        
        # Inizializza il modello di embedding
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self._client = None
        self._db = None
        
        # Crea la directory per il database se non esiste
        os.makedirs(persist_directory, exist_ok=True)
    
    def create_from_documents(self, documents):
        """
        Crea un nuovo database vettoriale dai documenti
        
        :param documents: Lista di documenti
        :return: Il database vettoriale
        """
        if not documents:
            return None
            
        # Crea la directory se non esiste
        os.makedirs(self.persist_directory, exist_ok=True)
        
        print(f"Creazione del database vettoriale con {len(documents)} documenti...")
        
        try:
            # Crea il database vettoriale
            self._db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                persist_directory=self.persist_directory
            )
            
            # Estrai il client e persistilo
            if hasattr(self._db, "_client"):
                self._client = self._db._client
            
            # Persisti il database
            if hasattr(self._db, "persist"):
                self._db.persist()
                
            print(f"Database vettoriale creato e salvato in {self.persist_directory}")
            
            return self._db
        except Exception as e:
            print(f"Errore nella creazione del database vettoriale: {str(e)}")
            return None
    
    def load(self):
        """
        Carica un database vettoriale esistente
        
        :return: Il database vettoriale o None se non esiste
        """
        if not os.path.exists(self.persist_directory):
            print(f"Directory {self.persist_directory} non trovata.")
            return None
            
        print(f"Caricamento del database vettoriale da {self.persist_directory}...")
        
        try:
            # Carica il database vettoriale
            self._db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding
            )
            
            # Estrai il client
            if hasattr(self._db, "_client"):
                self._client = self._db._client
                
            print(f"Database vettoriale caricato")
            
            return self._db
        except Exception as e:
            print(f"Errore nel caricamento del database vettoriale: {str(e)}")
            return None
    
    def close(self):
        """
        Chiude la connessione al database
        """
        try:
            # Persisti eventuali modifiche
            if self._db and hasattr(self._db, "persist"):
                self._db.persist()
            
            # Chiudi il client
            if self._client and hasattr(self._client, "close"):
                self._client.close()
                
            # Resetta le variabili
            self._client = None
            self._db = None
            
            print("Database vettoriale chiuso correttamente")
            
            return True
        except Exception as e:
            print(f"Errore nella chiusura del database: {str(e)}")
            return False 