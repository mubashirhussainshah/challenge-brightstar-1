import re
import torch
import logging
from tqdm.auto import tqdm
from typing import List, Dict, Optional
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM


class ItalianSentenceNormalizer:
    """
    LLM-based normalizer for Italian voicebot transcriptions.
    Extracts core B2B service requests from noisy call center transcripts.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        intent_list: Optional[List[Dict]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the normalizer with 4-bit quantized model.

        Args:
            model_id: HuggingFace model identifier
            intent_list: List of intent dictionaries with 'intent' and 'description' keys
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Initializing 4-bit quantized model: {model_id}")

        self.intent_map = {}
        if intent_list:
            self.intent_map = {
                item["intent"]: item["description"] for item in intent_list
            }

        # Configure 4-bit quantization for efficient inference
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            self.logger.error(f"Error: Could not load tokenizer for {model_id}.")
            raise e

        # Load model with quantization
        print("Loading model with 4-bit quantization...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=self.quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self.logger.info("Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error: Could not load model {model_id}.")
            raise e

        self._define_static_assets()
        print("Normalizer initialized successfully.")
        # Build system prompt with intent descriptions
        # self.intent_descriptions = intent_data or []
        # self._build_system_prompt()
        # self._define_few_shot_examples()

    def _define_static_assets(self) -> None:
        """Define regex patterns and few-shot examples."""

        # Regex patterns for pre-cleaning
        self.patterns = {
            "greetings": r"\b(buongiorno|buonasera|salve|ciao|pronto|addio|arrivederci|grazie|prego)\b",
            "fillers": r"\b(allora|dunque|cioè|praticamente|sostanzialmente|diciamo|ecco|insomma|quindi|scusa|scusi|senta|senti|guarda|ascolta)\b",
            "meta_talk": r"\b(volevo sapere|volevo chiedere|posso chiedere|avrei bisogno di|mi serve|chiamo per|sto chiamando per|desideravo)\b",
        }

        self.base_system_instruction = """Sei un assistente esperto in normalizzazione di testi per call center B2B (Lotto, POS, Servizi).
Il tuo compito è trasformare il parlato rumoroso in una frase sintetica e formale."""

        # Few-shot examples
        self.few_shot_examples = [
            {
                "input": "cambiando società Brightstar Spa non è passato il pagamento e quindi è saltato il pagamento di 1355 perché cè il mandato nuovo",
                "output": "non è passato il pagamento di 1355 per mandato nuovo",
            },
            {
                "input": "Mi scusi volevo sapere Come ottenere una nuova carta POS dopo il furto che l'ho persa",
                "output": "ottenere una nuova carta POS dopo il furto",
            },
            {
                "input": "Senta scusi il rotolino dellotto non entra nel terminale è difettoso cioè non gira",
                "output": "il rotolino dellotto non entra nel terminale è difettoso",
            },
            {
                "input": "Cho la lisprinter il nuovo terminale per le ricariche per liste qua non mi funziona bene",
                "output": "la lisprinter il nuovo terminale per le ricariche non funziona",
            },
            {
                "input": "Buongiorno non ho ancora ricevuto un ordine di Gratta e Vinci fatto settimana scorsa",
                "output": "non ho ricevuto un ordine di Gratta e Vinci",
            },
            {
                "input": "Praticamente ho il monitor della macchina del Lotto spento tutto nero",
                "output": "monitor della macchina del Lotto spento",
            },
            {
                "input": "Cosa significa il messaggio codice confezione errato che mi appare sul display",
                "output": "messaggio codice confezione errato",
            },
            {
                "input": "non mi funziona il lettore dei Gratta e Vinci nella macchinetta igt non legge il codice",
                "output": "non funziona il lettore dei Gratta e Vinci nella macchinetta igt",
            },
            {
                "input": "ho il terminale bloccato da sabato mattina non riesco a fare nulla",
                "output": "terminale bloccato",
            },
            {
                "input": "Dovrei registrare i pacchi che mi sono arrivati della della bstack",
                "output": "Dovrei registrare i pacchi arrivati della bstack",
            },
            {
                "input": "Sì sì sì 14 331 71 14 685 da 10 non parte il terminale stamattina",
                "output": "non parte il terminale stamattina",
            },
        ]

    def _preprocess_text(self, text: str) -> str:
        """
        Rule-based cleaning to remove obvious noise before LLM processing.
        """
        text = text.lower()
        text = re.sub(self.patterns["greetings"], "", text)
        text = re.sub(self.patterns["fillers"], "", text)
        text = re.sub(self.patterns["meta_talk"], "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _build_system_prompt(self, predicted_intent: Optional[str] = None) -> str:
        """
        Constructs a dynamic system prompt.
        If a specific intent is predicted, we inject its description to focus the LLM.
        """
        # Default Context
        context_instruction = """
**OBIETTIVO:**
Estrai SOLO la richiesta tecnica o commerciale principale.
Rimuovi esitazioni, ripetizioni e preamboli.
MANTIENI: Codici errore, nomi prodotti, importi, verbi chiave.
""".strip()

        # Dynamic Injection: Logic to focus on specific intent
        if predicted_intent and predicted_intent in self.intent_map:
            description = self.intent_map[predicted_intent]
            specific_instruction = f"""
**INTENTO SPECIFICO:**
L'utente sta parlando di: "{predicted_intent}" ({description}).
Estrai ESCLUSIVAMENTE i dettagli relativi a questo argomento. Ignora altri discorsi non pertinenti.
"""
        else:
            # Fallback if no intent or unknown intent
            specific_instruction = """
**FOCUS GENERALE:**
L'utente sta richiedendo assistenza su un servizio B2B. Identifica il problema principale.
"""

        return f"{self.base_system_instruction}\n{context_instruction}\n{specific_instruction}"

    def _build_chat_messages(
        self,
        raw_sentence: str,
        preprocessed_text: str,
        predicted_intent: Optional[str] = None,
    ) -> List[Dict]:
        """Build the chat template."""

        system_content = self._build_system_prompt(predicted_intent)
        messages = [{"role": "system", "content": system_content}]

        for example in self.few_shot_examples:
            messages.append({"role": "user", "content": f"Input: {example['input']}"})
            messages.append({"role": "assistant", "content": example["output"]})

        final_input = preprocessed_text if len(preprocessed_text) > 3 else raw_sentence
        messages.append({"role": "user", "content": f"Input: {final_input}"})

        return messages

    def _validate_output(self, llm_output: str, fallback_text: str) -> str:
        """
        Checks if the LLM output is a refusal or explanation.
        If so, returns the fallback text (Regex cleaned input).
        """

        # If output is > 200% of input length and input was short, it's likely an explanation
        if len(fallback_text) > 5 and len(llm_output) > len(fallback_text) * 1.5:
            return fallback_text

        return llm_output

    def normalize_sentence(
        self,
        raw_sentence: str,
        predicted_intent: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 64,
    ) -> str:
        """
        Normalize a single sentence.

        Args:
            raw_sentence: The original transcription.
            predicted_intent: (Optional) The intent predicted by BERT.
            temperature: Lower is more deterministic.
        """
        if not raw_sentence or not isinstance(raw_sentence, str):
            return ""

        # Rule-based preprocessing
        clean_text = self._preprocess_text(raw_sentence)

        # If regex stripped everything, return raw_sentence
        if not clean_text:
            return raw_sentence

        if len(clean_text.split()) < 3:
            return clean_text

        try:
            # Prepare inputs with dynamic prompt
            messages = self._build_chat_messages(
                raw_sentence, clean_text, predicted_intent
            )

            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.model.device)

            # Generate
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode
            start_idx = inputs["input_ids"].shape[1]
            output = self.tokenizer.decode(
                generated[0][start_idx:], skip_special_tokens=True
            )

            return self._validate_output(output, clean_text)

        except Exception as e:
            print(f"Error in normalization: {e}")
            return raw_sentence  # Fallback to original

    def normalize_batch(
        self,
        sentences: List[str],
        predicted_intents: Optional[List[str]] = None,
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Normalize a list of sentences.

        Args:
            sentences: List of text strings.
            predicted_intents: List of intent strings (must be same length as sentences).
        """
        results = []

        # Handle case where no intents are provided
        if predicted_intents is None:
            predicted_intents = [None] * len(sentences)

        if len(sentences) != len(predicted_intents):
            raise ValueError("Length mismatch between sentences and predicted intents")

        iterator = zip(sentences, predicted_intents)
        if show_progress:
            iterator = tqdm(
                iterator, total=len(sentences), desc="Normalizing", unit="msg"
            )

        for text, intent in iterator:
            res = self.normalize_sentence(text, predicted_intent=intent)
            results.append(res)

        return results
