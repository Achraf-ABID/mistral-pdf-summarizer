import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, Any, List

class GenerateurTexte:
    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        """
        Charge le modèle Mistral 7B avec quantification 4-bits.
        """
        print("Chargement du modèle Mistral 7B...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("Modèle Mistral 7B chargé avec succès.")

    def generer_reponse(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Génère une réponse textuelle à partir d'un prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )
        reponse = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return reponse.replace(prompt, "").strip()

    def generer_resume(self, contexte: List[str], langue: str = "Français") -> str:
        """
        Génère un résumé structuré à partir d'un contexte.
        """
        contexte_str = "\n".join(contexte)
        prompt = f"""
        [INST]
        En te basant sur le contexte suivant, fournis un résumé clair et concis en {langue}.
        Le résumé doit capturer les idées principales et les points clés du document.

        Contexte :
        {contexte_str}
        [/INST]
        Résumé :
        """
        print("Génération du résumé...")
        return self.generer_reponse(prompt)

    def generer_qcm(self, contexte: List[str], nombre_questions: int = 5) -> str:
        """
        Génère un QCM à partir d'un contexte.
        """
        contexte_str = "\n".join(contexte)
        prompt = f"""
        [INST]
        En te basant strictement sur le contexte fourni, génère un questionnaire à choix multiples (QCM) de {nombre_questions} questions.
        Pour chaque question, propose 4 options de réponse (A, B, C, D) où une seule est correcte.
        Indique clairement la bonne réponse après chaque question (par exemple : "Réponse : C").

        Contexte :
        {contexte_str}
        [/INST]
        QCM :
        """
        print("Génération du QCM...")
        return self.generer_reponse(prompt, max_new_tokens=1024)