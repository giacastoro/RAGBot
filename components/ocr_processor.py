import os
import base64
import json
import requests
import tempfile
from pathlib import Path

class MistralOCRProcessor:
    """
    Classe per gestire la conversione di documenti in formato Markdown utilizzando Mistral OCR
    """
    def __init__(self, api_key, md_dir="markdown_data"):
        """
        Inizializza il processore OCR
        
        :param api_key: Chiave API di Mistral
        :param md_dir: Directory dove salvare i file Markdown
        """
        self.api_key = api_key
        self.md_dir = md_dir
        
        # Crea la directory per i file Markdown se non esiste
        os.makedirs(self.md_dir, exist_ok=True)
        
    def _encode_file(self, file_path):
        """
        Codifica un file in base64
        
        :param file_path: Percorso del file da codificare
        :return: Stringa base64 del file
        """
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")
            
    def _get_file_extension(self, file_path):
        """
        Ottiene l'estensione del file
        
        :param file_path: Percorso del file
        :return: Estensione del file (senza il punto)
        """
        return Path(file_path).suffix[1:].lower()
    
    def _get_media_type(self, file_extension):
        """
        Ottiene il tipo MIME in base all'estensione del file
        
        :param file_extension: Estensione del file
        :return: Tipo MIME
        """
        mime_types = {
            "pdf": "application/pdf",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "tiff": "image/tiff",
            "tif": "image/tiff",
            "bmp": "image/bmp",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "doc": "application/msword",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xls": "application/vnd.ms-excel",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        }
        return mime_types.get(file_extension, "application/octet-stream")
    
    def _upload_file_to_mistral(self, file_path):
        """
        Carica un file su Mistral
        
        :param file_path: Percorso del file da caricare
        :return: ID del file caricato
        """
        file_name = os.path.basename(file_path)
        
        with open(file_path, "rb") as file_content:
            files = {
                "file": (file_name, file_content, "application/pdf")
            }
            
            data = {"purpose": "ocr"}
            
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            print(f"Caricamento del file {file_name} su Mistral...")
            response = requests.post(
                "https://api.mistral.ai/v1/files",
                headers=headers,
                files=files,
                data=data
            )
            
            if response.status_code != 200:
                print(f"Errore durante il caricamento del file: {response.status_code}, {response.text}")
                response.raise_for_status()
            
            result = response.json()
            file_id = result.get("id")
            print(f"File caricato con ID: {file_id}")
            
            return file_id
    
    def _get_signed_url(self, file_id):
        """
        Ottiene l'URL firmato per un file caricato
        
        :param file_id: ID del file
        :return: URL firmato
        """
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        print(f"Ottenimento dell'URL firmato per il file {file_id}...")
        response = requests.get(
            f"https://api.mistral.ai/v1/files/{file_id}/url",
            headers=headers
        )
        
        if response.status_code != 200:
            print(f"Errore durante l'ottenimento dell'URL firmato: {response.status_code}, {response.text}")
            response.raise_for_status()
        
        result = response.json()
        signed_url = result.get("url")
        print(f"URL firmato ottenuto")
        
        return signed_url
    
    def process_document_from_path(self, file_path):
        """
        Processa un documento a partire dal percorso del file
        
        :param file_path: Percorso del file da processare
        :return: Testo Markdown estratto, percorso del file MD salvato
        """
        # Verifica che il file esista
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Il file {file_path} non esiste")
        
        # Ottieni nome ed estensione del file
        file_name = os.path.basename(file_path)
        file_extension = self._get_file_extension(file_path)
        
        # Genera il percorso del file MD di output
        output_file = os.path.join(self.md_dir, f"{os.path.splitext(file_name)[0]}.md")
        
        # Controlla se il file MD esiste già
        if os.path.exists(output_file):
            print(f"File Markdown già esistente per {file_name}, caricamento da disco")
            with open(output_file, "r", encoding="utf-8") as f:
                return f.read(), output_file
        
        # Ottieni il tipo MIME
        media_type = self._get_media_type(file_extension)
        
        try:
            print(f"Elaborazione OCR di {file_name} con Mistral...")
            
            # Gestisci diversamente i PDF e le immagini
            if file_extension.lower() == "pdf":
                # Per i PDF, usa l'approccio file upload -> URL -> OCR
                # 1. Carica il file
                file_id = self._upload_file_to_mistral(file_path)
                
                # 2. Ottieni l'URL firmato
                signed_url = self._get_signed_url(file_id)
                
                # 3. Richiedi l'OCR usando l'URL firmato
                payload = {
                    "model": "mistral-ocr-latest",
                    "document": {
                        "type": "document_url",
                        "document_url": signed_url
                    }
                }
            else:
                # Per le immagini, usa l'approccio base64
                file_base64 = self._encode_file(file_path)
                
                payload = {
                    "model": "mistral-ocr-latest",
                    "document": {
                        "type": "image_url",
                        "image_url": f"data:{media_type};base64,{file_base64}"
                    }
                }
            
            # Effettua la richiesta all'API Mistral OCR
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(
                "https://api.mistral.ai/v1/ocr",
                headers=headers,
                json=payload
            )
            
            # Debug: stampa l'errore se presente
            if response.status_code != 200:
                print(f"Risposta API: {response.status_code}, {response.text}")
            
            # Verifica la risposta
            response.raise_for_status()
            ocr_result = response.json()
            
            # Estrai il contenuto Markdown
            markdown_content = ocr_result.get("content", "")
            
            # Se il contenuto è vuoto ma ci sono pagine, prova a estrarre il markdown dalle pagine
            if not markdown_content and "pages" in ocr_result:
                pages_content = []
                for page in ocr_result["pages"]:
                    if "markdown" in page:
                        pages_content.append(page["markdown"])
                markdown_content = "\n\n".join(pages_content)
            
            # Salva il contenuto Markdown su file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            print(f"File Markdown salvato in {output_file}")
            return markdown_content, output_file
            
        except Exception as e:
            print(f"Errore durante l'elaborazione OCR: {str(e)}")
            raise
    
    def process_document_from_data(self, file_data, file_name, file_extension):
        """
        Processa un documento a partire dai dati binari
        
        :param file_data: Dati binari del file
        :param file_name: Nome del file
        :param file_extension: Estensione del file (senza il punto)
        :return: Testo Markdown estratto, percorso del file MD salvato
        """
        # Genera il percorso del file MD di output
        output_file = os.path.join(self.md_dir, f"{os.path.splitext(file_name)[0]}.md")
        
        # Controlla se il file MD esiste già
        if os.path.exists(output_file):
            print(f"File Markdown già esistente per {file_name}, caricamento da disco")
            with open(output_file, "r", encoding="utf-8") as f:
                return f.read(), output_file
                
        # Ottieni il tipo MIME
        media_type = self._get_media_type(file_extension)
        
        try:
            print(f"Elaborazione OCR di {file_name} con Mistral...")
            
            # Gestisci diversamente i PDF e le immagini
            if file_extension.lower() == "pdf":
                # Per i PDF, usa l'approccio file upload -> URL -> OCR
                # Salva temporaneamente i dati in un file
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                    temp_file.write(file_data)
                    temp_file_path = temp_file.name
                
                try:
                    # 1. Carica il file
                    file_id = self._upload_file_to_mistral(temp_file_path)
                    
                    # 2. Ottieni l'URL firmato
                    signed_url = self._get_signed_url(file_id)
                    
                    # 3. Richiedi l'OCR usando l'URL firmato
                    payload = {
                        "model": "mistral-ocr-latest",
                        "document": {
                            "type": "document_url",
                            "document_url": signed_url
                        }
                    }
                finally:
                    # Rimuovi il file temporaneo
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
            else:
                # Per le immagini, usa l'approccio base64
                file_base64 = base64.b64encode(file_data).decode("utf-8")
                
                payload = {
                    "model": "mistral-ocr-latest",
                    "document": {
                        "type": "image_url",
                        "image_url": f"data:{media_type};base64,{file_base64}"
                    }
                }
            
            # Effettua la richiesta all'API Mistral OCR
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(
                "https://api.mistral.ai/v1/ocr",
                headers=headers,
                json=payload
            )
            
            # Debug: stampa l'errore se presente
            if response.status_code != 200:
                print(f"Risposta API: {response.status_code}, {response.text}")
            
            # Verifica la risposta
            response.raise_for_status()
            ocr_result = response.json()
            
            # Estrai il contenuto Markdown
            markdown_content = ocr_result.get("content", "")
            
            # Se il contenuto è vuoto ma ci sono pagine, prova a estrarre il markdown dalle pagine
            if not markdown_content and "pages" in ocr_result:
                pages_content = []
                for page in ocr_result["pages"]:
                    if "markdown" in page:
                        pages_content.append(page["markdown"])
                markdown_content = "\n\n".join(pages_content)
                
            # Salva il contenuto Markdown su file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            print(f"File Markdown salvato in {output_file}")
            return markdown_content, output_file
            
        except Exception as e:
            print(f"Errore durante l'elaborazione OCR: {str(e)}")
            raise
            
    def get_markdown_path(self, file_name):
        """
        Ottiene il percorso del file Markdown per un dato file
        
        :param file_name: Nome del file originale
        :return: Percorso del file Markdown o None se non esiste
        """
        md_path = os.path.join(self.md_dir, f"{os.path.splitext(file_name)[0]}.md")
        return md_path if os.path.exists(md_path) else None 