import os
from dotenv import load_dotenv

from core.data_processor import extraire_texte_de_pdf, decouper_texte_en_chunks
from core.rag_pipeline import RAGPipeline
from core.text_generator import GenerateurTexte

def main():
    # Charger les variables d'environnement (token Hugging Face)
    load_dotenv()
    hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")
    
    if not hugging_face_token:
        print("Erreur : Le token Hugging Face n'est pas défini. Veuillez le créer dans un fichier .env.")
        return

    # --- Étape 1 : Traitement du document ---
    chemin_pdf = "documents/exemple.pdf" # Assurez-vous d'avoir un PDF ici
    if not os.path.exists(chemin_pdf):
        print(f"Erreur : Le fichier PDF '{chemin_pdf}' n'a pas été trouvé.")
        return
        
    texte_brut = extraire_texte_de_pdf(chemin_pdf)
    chunks = decouper_texte_en_chunks(texte_brut)

    # --- Étape 2 : Pipeline RAG ---
    pipeline_rag = RAGPipeline()
    pipeline_rag.construire_index_faiss(chunks)
    
    # Récupérer le contexte global du document pour une génération de haute qualité
    contexte_pertinent = pipeline_rag.recuperer_documents_pertinents("Donne-moi un aperçu global de ce document", k=10)

    # --- Étape 3 : Génération de texte avec Mistral 7B ---
    generateur = GenerateurTexte()

    # Générer et afficher le résumé
    print("\n" + "="*50)
    print("                 RÉSUMÉ DU DOCUMENT")
    print("="*50)
    resume = generateur.generer_resume(contexte_pertinent, langue="Français")
    print(resume)

    # Générer et afficher le QCM
    print("\n" + "="*50)
    print("        QUESTIONNAIRE À CHOIX MULTIPLES (QCM)")
    print("="*50)
    qcm = generateur.generer_qcm(contexte_pertinent, nombre_questions=5)
    print(qcm)
    print("\n" + "="*50)


if __name__ == "__main__":
    main()