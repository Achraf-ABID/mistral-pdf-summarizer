# 📝 Générateur de Résumés et QCM avec Mistral 7B

Ce projet fournit un pipeline complet pour analyser des documents PDF, en extraire les informations clés, et générer automatiquement des résumés concis ainsi que des questionnaires à choix multiples (QCM) pertinents.

Il s'appuie sur le modèle de langage de pointe **Mistral 7B** et une architecture de **Génération Augmentée par Récupération (RAG)** pour garantir des résultats de haute qualité, même sur des documents longs et complexes.

## ✨ Fonctionnalités

-   **Extraction de Texte depuis PDF** : Gère un ou plusieurs fichiers PDF en entrée.
-   **Génération de Résumés** : Crée des résumés clairs, structurés et multilingues (Français, Anglais).
-   **Génération de QCM** : Produit des questionnaires pertinents avec 4 options de réponse pour évaluer la compréhension.
-   **Analyse Sémantique** : Utilise des embeddings et une base de données vectorielle (FAISS) pour trouver les passages les plus pertinents avant la génération.
-   **Modèle Optimisé** : Profite de la puissance de Mistral-7B-Instruct, optimisé avec la quantification 4-bits pour s'exécuter sur des GPU avec des ressources limitées.

## 🏛️ Architecture du projet (RAG)

Le système suit une approche de Génération Augmentée par Récupération (RAG) pour fournir des réponses contextuellement riches et précises.

1.  **Chargement et Découpage** : Le texte est d'abord extrait des fichiers PDF. Il est ensuite découpé en plus petits segments (chunks) pour être plus facilement traitable par les modèles.
2.  **Création d'Embeddings** : Chaque segment de texte est transformé en une représentation numérique (vecteur d'embedding) à l'aide du modèle `BAAI/bge-small-en-v1.5`. Ces vecteurs capturent le sens sémantique du texte.
3.  **Indexation (FAISS)** : Les vecteurs sont stockés et indexés dans une base de données vectorielle FAISS. Cet index permet une recherche de similarité ultra-rapide.
4.  **Récupération (Retrieval)** : Lorsqu'une requête est faite (par exemple, "génère un résumé"), le système recherche dans l'index FAISS les segments de texte les plus pertinents sémantiquement.
5.  **Génération (Mistral 7B)** : Les segments pertinents récupérés sont injectés dans un prompt structuré et envoyés au modèle Mistral 7B. Le modèle utilise ce contexte pour générer un résumé ou un QCM précis et fidèle au document source.

## 🛠️ Technologies Utilisées

-   **Modèle LLM** : `mistralai/Mistral-7B-Instruct-v0.1`
-   **Modèle d'Embedding** : `BAAI/bge-small-en-v1.5`
-   **Frameworks IA/ML** : `Hugging Face Transformers`, `PyTorch`, `LangChain` (pour le découpage de texte)
-   **Base de Données Vectorielle** : `FAISS (Facebook AI Similarity Search)`
-   **Traitement de PDF** : `PyMuPDF (fitz)`
-   **Optimisation** : `bitsandbytes` (pour la quantification 4-bits)

## 🚀 Démarrage Rapide

### 1. Prérequis

-   Python 3.8+
-   Un environnement avec GPU est fortement recommandé (NVIDIA T4, V100, A100...).
-   Un compte Hugging Face et un token d'accès.

### 2. Installation

Clonez ce dépôt et installez les dépendances :

```bash
# Cloner le projet (exemple)
git clone https://votre-url-de-projet.git
cd votre-projet

# Installer les packages requis
pip install -U torch torchvision torchaudio
pip install -q -U transformers accelerate sentence-transformers langchain faiss-cpu pymupdf bitsandbytes huggingface_hub
