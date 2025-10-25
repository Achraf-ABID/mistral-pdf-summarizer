import fitz  # PyMuPDF
from typing import List
import re

def extraire_texte_de_pdf(chemin_pdf: str) -> str:
    """
    Extrait le texte brut d'un fichier PDF.
    """
    texte_complet = ""
    try:
        document = fitz.open(chemin_pdf)
        for page in document:
            texte_complet += page.get_text()
        document.close()
        print(f"Texte extrait avec succès de {chemin_pdf}")
    except Exception as e:
        print(f"Erreur lors de l'extraction du texte du PDF: {e}")
    return texte_complet

def decouper_texte_en_chunks(texte: str, taille_chunk: int = 1000, chevauchement_chunk: int = 200) -> List[str]:
    """
    Découpe un long texte en segments plus petits (chunks) avec chevauchement.
    """
    # Nettoyer le texte et séparer en phrases
    texte = re.sub(r'\s+', ' ', texte).strip()
    phrases = re.split(r'(?<=[.!?])\s+', texte)
    
    chunks = []
    chunk_actuel = ""
    
    for phrase in phrases:
        if len(chunk_actuel) + len(phrase) <= taille_chunk:
            chunk_actuel += " " + phrase if chunk_actuel else phrase
        else:
            if chunk_actuel:
                chunks.append(chunk_actuel.strip())
            # Commencer un nouveau chunk avec le chevauchement
            mots = chunk_actuel.split()
            mots_chevauchement = " ".join(mots[-int(chevauchement_chunk/10):])  # Approximatif: 10 caractères par mot
            chunk_actuel = mots_chevauchement + " " + phrase
    
    if chunk_actuel:
        chunks.append(chunk_actuel.strip())
    
    print(f"Texte découpé en {len(chunks)} chunks.")
    return chunks