import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Any

class RAGPipeline:
    def __init__(self, model_embedding: str = 'BAAI/bge-small-en-v1.5'):
        """
        Initialise le pipeline RAG avec le modèle d'embedding.
        """
        print("Initialisation du pipeline RAG...")
        self.embedding_model = SentenceTransformer(model_embedding)
        self.index = None
        self.chunks = []
        print("Modèle d'embedding chargé.")

    def construire_index_faiss(self, chunks: List[str]):
        """
        Crée les embeddings pour les chunks et construit l'index FAISS.
        """
        self.chunks = chunks
        print("Génération des embeddings pour les chunks...")
        embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print("Index FAISS construit avec succès.")

    def recuperer_documents_pertinents(self, requete: str, k: int = 5) -> List[str]:
        """
        Récupère les 'k' chunks les plus pertinents pour une requête donnée.
        """
        if self.index is None:
            raise ValueError("L'index FAISS n'a pas été construit. Appelez d'abord 'construire_index_faiss'.")
            
        print(f"Récupération des {k} documents les plus pertinents...")
        requete_embedding = self.embedding_model.encode([requete], convert_to_tensor=False)
        _, indices = self.index.search(np.array(requete_embedding).astype('float32'), k)
        
        return [self.chunks[i] for i in indices[0]]