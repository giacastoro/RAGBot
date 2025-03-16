from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import sys
import traceback
import requests
import os
import json

class Chatbot:
    def __init__(self, vector_store=None, model_name="gemma3:4b", templates_file=None):
        """
        Inizializza il chatbot
        
        :param vector_store: Database vettoriale per il retrieval
        :param model_name: Nome del modello Ollama da utilizzare
        :param templates_file: Percorso del file con i template personalizzati
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.templates_file = templates_file
        
        # Inizializza il modello LLM
        self.llm = OllamaLLM(model=self.model_name, temperature=0.1)
        
        # Carica i template personalizzati o usa quelli di default
        self.templates = self.load_templates()
        
        # Crea i prompt template
        self.qa_prompt = PromptTemplate(
            template=self.templates["qa_template"],
            input_variables=["context", "question"]
        )
        
        self.refine_prompt = PromptTemplate(
            template=self.templates["refine_template"],
            input_variables=["existing_answer", "context", "question"]
        )
        
        # Inizializza la chain QA
        self.initialize_qa_chain()
    
    def load_templates(self):
        """
        Carica i template da file o utilizza quelli di default
        
        :return: Dizionario con i template
        """
        default_templates = {
            "qa_template": """
            Sei un assistente AI utile ed esperto che aiuta le persone a trovare informazioni.
            Utilizza SOLO le informazioni di contesto fornite per rispondere alla domanda.
            Se le informazioni necessarie non sono presenti nel contesto, rispondi onestamente che non lo sai.
            
            Considera che:
            - Quando la domanda chiede informazioni su un ruolo (es. "presidente", "direttore"), cerca sia il ruolo che la persona che lo ricopre.
            - Cerca di riconoscere entità come nomi, ruoli e organizzazioni, anche se ci sono piccole variazioni nei nomi.
            - Se la domanda chiede "chi è X" e X è un ruolo, fornisci il nome della persona che ricopre quel ruolo.
            - Se la domanda chiede "chi è X" e X è una persona, fornisci informazioni sul ruolo di quella persona.
            - Se non trovi informazioni esatte, verifica se ci sono informazioni simili o correlate che potrebbero rispondere all'intento della domanda.
            
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
        
        try:
            if self.templates_file and os.path.exists(self.templates_file):
                with open(self.templates_file, "r", encoding="utf-8") as f:
                    loaded_templates = json.load(f)
                    print(f"Template personalizzati caricati da {self.templates_file}")
                    return loaded_templates
        except Exception as e:
            print(f"Errore nel caricamento dei template da {self.templates_file}: {str(e)}")
        
        print("Utilizzo dei template predefiniti")
        return default_templates
    
    def save_templates(self, templates):
        """
        Salva i template in un file
        
        :param templates: Dizionario con i template da salvare
        :return: True se il salvataggio è avvenuto con successo, False altrimenti
        """
        if not self.templates_file:
            print("Nessun file dei template specificato")
            return False
        
        try:
            # Crea la directory se non esiste
            os.makedirs(os.path.dirname(self.templates_file), exist_ok=True)
            
            with open(self.templates_file, "w", encoding="utf-8") as f:
                json.dump(templates, f, indent=4, ensure_ascii=False)
            
            # Aggiorna i template correnti
            self.templates = templates
            
            # Aggiorna i prompt
            self.qa_prompt = PromptTemplate(
                template=self.templates["qa_template"],
                input_variables=["context", "question"]
            )
            
            self.refine_prompt = PromptTemplate(
                template=self.templates["refine_template"],
                input_variables=["existing_answer", "context", "question"]
            )
            
            # Reinizializza la chain QA
            self.initialize_qa_chain()
            
            return True
        except Exception as e:
            print(f"Errore nel salvataggio dei template: {str(e)}")
            return False
    
    def initialize_qa_chain(self):
        """
        Inizializza la chain QA con il vector store
        """
        if self.vector_store is None:
            self.qa_chain = None
            print("ATTENZIONE: Vector store non impostato, QA chain non inizializzata")
            return
            
        try:
            # Crea il retriever dal vector store
            print(f"Creazione retriever con {self.vector_store._collection.count()} documenti")
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Recupera 5 documenti per avere più contesto
            )
            
            # Configura le opzioni per la chain "refine"
            chain_type_kwargs = {
                "question_prompt": self.qa_prompt,
                "refine_prompt": self.refine_prompt,
                "document_variable_name": "context",
            }
            
            # Crea la chain QA con il tipo "refine"
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="refine",  # Utilizzo tipo "refine"
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs
            )
            print("QA chain inizializzata con successo usando la strategia 'refine'")
        except Exception as e:
            print(f"ERRORE nell'inizializzazione della QA chain: {str(e)}")
            traceback.print_exc()
            self.qa_chain = None
    
    def set_vector_store(self, vector_store):
        """
        Imposta un nuovo database vettoriale e reinizializza la chain QA
        
        :param vector_store: Nuovo database vettoriale
        """
        self.vector_store = vector_store
        self.initialize_qa_chain()
    
    def change_model(self, new_model_name):
        """
        Cambia il modello Ollama utilizzato
        
        :param new_model_name: Nome del nuovo modello
        :return: True se il cambio è avvenuto con successo, False altrimenti
        """
        try:
            print(f"Cambiando modello da {self.model_name} a {new_model_name}")
            
            # Aggiorna il modello LLM
            self.model_name = new_model_name
            self.llm = OllamaLLM(model=self.model_name, temperature=0.1)
            
            # Reinizializza la chain QA
            if self.vector_store:
                self.initialize_qa_chain()
                
            return True, f"Modello cambiato con successo a {new_model_name}"
        except Exception as e:
            error_msg = f"Errore nel cambio del modello: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return False, error_msg
            
    @staticmethod
    def get_available_models():
        """
        Ottiene la lista dei modelli Ollama disponibili
        
        :return: Lista dei nomi dei modelli
        """
        try:
            # Prova a contattare l'API Ollama per ottenere i modelli
            response = requests.get("http://localhost:11434/api/tags")
            
            if response.status_code == 200:
                models_data = response.json()
                # Estrai i nomi dei modelli
                models = [model["name"] for model in models_data.get("models", [])]
                return models
            else:
                print(f"Errore nella richiesta API Ollama: {response.status_code}")
                return []
        except Exception as e:
            print(f"Errore nel recupero dei modelli Ollama: {str(e)}")
            return []
    
    def get_answer(self, question):
        """
        Ottiene la risposta alla domanda utilizzando la chain QA
        
        :param question: Domanda da porre
        :return: Risposta e documenti di origine
        """
        if self.qa_chain is None:
            return {
                "answer": "Non ho accesso a documenti. Carica prima dei PDF o controlla il database vettoriale.",
                "source_documents": []
            }
            
        try:
            print(f"Elaborazione domanda: {question}")
            
            # Ottieni la risposta dalla chain QA
            result = self.qa_chain.invoke({"query": question})
            
            print(f"Risultato ottenuto: {type(result)}")
            print(f"Contenuto risultato: {result}")
            
            # Verifica il formato della risposta
            if isinstance(result, dict):
                if "result" in result:
                    answer = result["result"]
                    source_docs = result.get("source_documents", [])
                elif "answer" in result:
                    answer = result["answer"]
                    source_docs = result.get("source_documents", [])
                else:
                    # In caso di formato sconosciuto, mostra direttamente
                    answer = str(result)
                    # Ottieni documenti rilevanti direttamente
                    source_docs = self.vector_store.similarity_search(question, k=3) if self.vector_store else []
            else:
                # Se result non è un dizionario, trattalo come risposta diretta
                answer = str(result)
                # Ottieni documenti rilevanti direttamente
                source_docs = self.vector_store.similarity_search(question, k=3) if self.vector_store else []
            
            # Genera una risposta formattata
            return {
                "answer": answer,
                "source_documents": source_docs
            }
        except Exception as e:
            print(f"ERRORE durante la generazione della risposta: {str(e)}")
            traceback.print_exc()
            
            # In caso di errore, tenta un approccio diretto
            try:
                if self.vector_store:
                    # Ottieni documenti rilevanti direttamente
                    docs = self.vector_store.similarity_search(question, k=3)
                    context = "\n\n".join([d.page_content for d in docs])
                    
                    # Usa direttamente il prompt e il modello
                    prompt_text = self.qa_prompt.format(context=context, question=question)
                    direct_response = self.llm.invoke(prompt_text)
                    
                    return {
                        "answer": direct_response,
                        "source_documents": docs
                    }
                else:
                    raise Exception("Vector store non inizializzato")
            except Exception as e2:
                return {
                    "answer": f"Si è verificato un errore: {str(e)}. Secondo tentativo fallito: {str(e2)}",
                    "source_documents": []
                } 