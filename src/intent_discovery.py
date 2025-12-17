import gc
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, TextGeneration
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)
# from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

from .config import DiscoveryConfig
from .logging_config import LoggingConfig

logger = LoggingConfig.get_logger(__name__)


class IntentCleaner:
    """
    Text preprocessing specifically for intent discovery.
    """

    def __init__(self):
        # Domain-specific stopwords
        self.stopwords = {
            "il",
            "lo",
            "la",
            "di",
            "in",
            "con",
            "al",
            "alla",
            "del",
            "i",
            "gli",
            "le",
            "un",
            "uno",
            "una",
            "di",
            "a",
            "da",
            "in",
            "con",
            "su",
            "per",
            "tra",
            "fra",
            "che",
            "chi",
            "cui",
            "sulla",
            "ad",
            "no",
        }

        # Voice fillers to remove
        self.voice_fillers = {
            "senti",
            "scusa",
            "pronto",
            "guarda",
            "allora",
            "diciamo",
            "tipo",
            "vabbè",
            "buongiorno",
            "buonasera",
            "salve",
            "ciao",
            "arrivederci",
            "grazie",
            "prego",
            "scusami",
            "cortesia",
            "perfavore",
            "favore",
            "mh",
            "eh",
            "ehm",
            "ok",
            "eccolo",
            "ecco",
            "certo",
            "dunque",
            "aspetta",
            "dimmi",
            "sì",
            "va bene",
        }

        # Domain normalizations
        self.replacements = {
            r"\b(gratta e vinci|gratta vinci|gratta\s?e\s?vinci)\b": "grattaevinci",
            r"\b(10 e lotto|dieci e lotto|10elotto)\b": "diecielotto",
            r"\b(million day|millionday)\b": "millionday",
            r"\b(grattini|gratta)\b": "grattaevinci",
            r"\b(l'otto|l otto)\b": "lotto",
            r"\b(tv|televisione|televisore|schermo|display)\b": "monitor",
            r"\b(stampantina|stampatrice)\b": "stampante",
            r"\b(human|umano|persona|cristiano|collega)\b": "operatore",
            r"\b(ribon|ribbon|nastri|rulli)\b": "cartucce",
            r"\b(vincite)\b": "vincita",
            r"\b(time out)\b": "timeout",
        }

    def clean_text(self, text: str) -> str:
        """Clean a single text for clustering"""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        text = text.lower()

        # Apply domain normalizations
        for pattern, replacement in self.replacements.items():
            text = re.sub(pattern, replacement, text)

        # Remove special characters but keep spaces
        text = re.sub(r"[^a-zàèéìòù0-9\s]", " ", text)

        # Tokenize and filter
        words = text.split()
        words = [
            w
            for w in words
            if w not in self.stopwords
            and w not in self.voice_fillers
            and len(w) > 1
            and not w.isdigit()
        ]

        return " ".join(words)

    def preprocess_batch(
        self, texts: List[str], min_words: int = 3
    ) -> Tuple[List[str], List[int]]:
        """
        Preprocess a batch of texts with quality filtering.

        Returns:
            cleaned_texts: List of cleaned texts
            valid_indices: Original indices of valid texts
        """
        logger.info(f"Preprocessing {len(texts)} texts for intent discovery")
        cleaned_texts = []
        valid_indices = []

        for i, text in enumerate(texts):
            cleaned = self.clean_text(text)
            # Quality filter: at least min_words
            if len(cleaned.split()) >= min_words:
                cleaned_texts.append(cleaned)
                valid_indices.append(i)

        logger.info(
            f"Kept {len(cleaned_texts)}/{len(texts)} texts after preprocessing "
            f"({len(cleaned_texts)/len(texts)*100:.1f}%)"
        )

        return cleaned_texts, valid_indices


class LocalLLMLabeler:
    """
    Generates topic labels using your existing local LLM (Llama).
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        max_new_tokens: int = 30,
        temperature: float = 0.1,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logger.info(f"Initializing Local LLM for topic labeling: {model_id}")
        self._load_model()

    def _load_model(self):
        """Load 4-bit quantized LLM"""
        # Quantization config (same as your normalizer)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

            # Create pipeline
            self.generator = pipeline(
                task="text-generation",
                model=model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                repetition_penalty=1.1,
                return_full_text=False,
            )

            logger.info("LLM loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise

    def get_bertopic_representation(self) -> TextGeneration:
        """
        Returns a BERTopic-compatible representation model.
        This gets plugged directly into BERTopic.
        """
        prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Sei un analista esperto che etichetta gruppi di richieste clienti B2B (Lotto, POS, Servizi).
Genera un'etichetta CONCISA (massimo 3 parole) e TECNICA che descriva il problema comune.
Rispondi SOLO con l'etichetta. Niente spiegazioni o testo aggiuntivo.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Questo topic contiene i seguenti documenti rappresentativi:
[DOCUMENTS]

Parole chiave del topic: '[KEYWORDS]'

Genera un'etichetta breve (massimo 3 parole) per questo gruppo di richieste.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

        return TextGeneration(
            self.generator,
            prompt=prompt_template,
            doc_length=100,
            tokenizer=self.tokenizer,
            diversity=None,
        )


class IntentDiscoveryPipeline:
    """
    Main pipeline for discovering intents in unlabeled data.
    """

    def __init__(self, config: DiscoveryConfig, enable_llm_labeling: bool):
        self.config = config
        self.enable_llm_labeling = enable_llm_labeling
        self.cleaner = IntentCleaner()
        self.topic_model = None
        self.embeddings = None
        self.embedding_model = None
        self.llm_labeler = None

    def build_model(self) -> BERTopic:
        """Build BERTopic model with local LLM"""
        logger.info("=" * 70)
        logger.info("BUILDING BERTOPIC MODEL")
        logger.info("=" * 70)

        # Embedding model
        embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)

        # Dimensionality reduction (UMAP)
        umap_model = UMAP(
            n_neighbors=self.config.UMAP_N_NEIGHBORS,
            n_components=self.config.UMAP_N_COMPONENTS,
            min_dist=self.config.UMAP_MIN_DIST,
            metric=self.config.UMAP_METRIC,
            random_state=self.config.SEED,
        )

        # Clustering (HDBSCAN)
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.config.MIN_CLUSTER_SIZE,
            min_samples=self.config.MIN_SAMPLES,
            metric=self.config.HDBSCAN_METRIC,
            cluster_selection_method="eom",
            prediction_data=True,
        )

        # Vectorizer
        vectorizer_model = CountVectorizer(
            stop_words=list(self.cleaner.stopwords),
            ngram_range=(1, 2),
            min_df=self.config.MIN_DF,
            max_df=0.95,
        )

        representation_model = {}
        representation_model["KeyBERT"] = MaximalMarginalRelevance(diversity=0.4)

        if self.enable_llm_labeling:
            try:
                llm_labeler = LocalLLMLabeler(
                    model_id=self.config.LLM_MODEL_ID,
                    max_new_tokens=self.config.LLM_MAX_NEW_TOKENS,
                    temperature=self.config.LLM_TEMPERATURE,
                )
                representation_model["Main"] = llm_labeler.get_bertopic_representation()
                logger.info("LLM Labeling Enabled.")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}. Falling back to KeyBERT.")
        else:
            logger.info("LLM Labeling Disabled. Using KeyBERT.")

        # Initialize BERTopic
        self.topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            top_n_words=self.config.TOP_N_WORDS,
            min_topic_size=self.config.MIN_CLUSTER_SIZE,
            nr_topics=None,
            verbose=True,
            calculate_probabilities=True,
        )

        logger.info("BERTopic model initialized successfully")
        return self.topic_model

    def discover_intents(
        self, texts: List[str]
    ) -> Tuple[List[int], np.ndarray, List[str], List[int]]:
        """
        Main discovery pipeline.

        Args:
            texts: List of raw texts to cluster

        Returns:
            topics: Topic assignments (-1 for outliers)
            probs: Probability matrix
            cleaned_texts: Preprocessed texts
            valid_indices: Indices of valid texts in original list
        """
        logger.info("-" * 70)
        logger.info("STARTING INTENT DISCOVERY")
        logger.info("-" * 70)

        # Preprocess
        cleaned_texts, valid_indices = self.cleaner.preprocess_batch(texts)

        if len(cleaned_texts) < self.config.MIN_CLUSTER_SIZE:
            raise ValueError(
                f"Not enough valid texts ({len(cleaned_texts)}) for clustering. "
                f"Need at least {self.config.MIN_CLUSTER_SIZE}"
            )

        if self.embeddings is None:
            logger.info("Generating sentence embeddings...")
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            self.embeddings = self.embedding_model.encode(
                cleaned_texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=32,
            )
            logger.info(f"Embeddings shape: {self.embeddings.shape}")

        # Build model (if not already built)
        if self.topic_model is None:
            self.build_model()

        # Fit and transform
        logger.info(f"Clustering {len(cleaned_texts)} texts...")
        topics, probs = self.topic_model.fit_transform(
            cleaned_texts, embeddings=self.embeddings
        )

        # Log results
        unique_topics = set(topics)
        n_clusters = len([t for t in unique_topics if t != -1])
        n_outliers = sum(1 for t in topics if t == -1)

        logger.info("-" * 70)
        logger.info("CLUSTERING COMPLETE")
        logger.info("-" * 70)
        logger.info(f"Discovered {n_clusters} intent clusters")
        logger.info(f"Outliers: {n_outliers} ({n_outliers/len(topics)*100:.1f}%)")
        logger.info(f"Coverage: {(len(topics)-n_outliers)/len(topics)*100:.1f}%")

        return topics, probs, cleaned_texts, valid_indices

    def get_topic_info(self) -> pd.DataFrame:
        """Get detailed topic information"""
        if self.topic_model is None:
            raise ValueError("Model not trained yet. Call discover_intents() first.")

        return self.topic_model.get_topic_info()

    def _get_topic_name(self, topic_id: int) -> str:
        """
        Centralized topic name extraction with proper cleaning.
        """
        if topic_id == -1:
            return "NOISE"

        try:
            topic_info = self.get_topic_info()
            row = topic_info[topic_info["Topic"] == topic_id]

            if row.empty:
                return f"Topic {topic_id}"

            # Prefer LLM label, fallback to KeyBERT
            source_col = "Main" if "Main" in topic_info.columns else "Name"
            raw_label = row.iloc[0][source_col]

            # Handle list/string formats
            if isinstance(raw_label, list):
                label_text = " ".join(str(x) for x in raw_label[:3])
            else:
                label_text = str(raw_label)

            # Clean: remove topic ID prefix, quotes, underscores
            clean = re.sub(r"^\d+\s*[_-]\s*", "", label_text)
            clean = re.sub(r'[_\'"]+', " ", clean)
            clean = re.sub(r"\s+", " ", clean).strip()

            return clean

        except Exception as e:
            logger.warning(f"Could not extract name for topic {topic_id}: {e}")
            return f"Topic {topic_id}"

    def get_representative_sentences(
        self,
        topic_id: int,
        sentences: List[str],
        topics: List[int],
        probs: np.ndarray,
        n: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Get most representative sentences with deterministic tie-breaking.
        """
        try:
            indices = [i for i, t in enumerate(topics) if t == topic_id]
            if len(indices) == 0:
                return []

            topic_probs = (
                probs[indices, topic_id]
                if topic_id >= 0 and probs.shape[1] > topic_id
                else np.zeros(len(indices))
            )

            # Sort by probability (primary) and length (secondary)
            lengths = np.array([len(sentences[i]) for i in indices])

            # Lexicographic sort: primary by prob, secondary by length
            sort_keys = np.lexsort((lengths, topic_probs))
            sorted_indices = sort_keys[-n:][::-1]

            return [
                (sentences[indices[i]], float(topic_probs[i])) for i in sorted_indices
            ]

        except Exception as e:
            logger.warning(f"Could not get representative sentences: {e}")
            return []

    def _calculate_coherence(
        self, topics: List[int], metric: str = "cosine"
    ) -> pd.DataFrame:
        """
        Vectorized coherence calculation with configurable metric.
        """
        logger.info(f"Calculating semantic coherence (metric={metric})...")

        topics_arr = np.array(topics)
        unique_labels = sorted(list(set(topics)))
        coherence_scores = {}

        # Calculate centroids
        centroids = []
        valid_labels = []

        for label in unique_labels:
            if label == -1:
                continue

            mask = topics_arr == label
            if sum(mask) == 0:
                continue

            cluster_embeds = self.embeddings[mask]
            centroid = np.mean(cluster_embeds, axis=0)
            centroids.append(centroid)
            valid_labels.append(label)

        if not centroids:
            return pd.DataFrame()

        centroids = np.array(centroids)

        # Vectorized coherence computation
        for label, centroid in zip(valid_labels, centroids):
            cluster_embeds = self.embeddings[topics_arr == label]

            if metric == "cosine":
                # Vectorized cosine similarity
                sims = np.dot(cluster_embeds, centroid) / (
                    np.linalg.norm(cluster_embeds, axis=1) * np.linalg.norm(centroid)
                    + 1e-8
                )
            elif metric == "euclidean":
                # Convert distance to similarity
                dists = np.linalg.norm(cluster_embeds - centroid, axis=1)
                sims = 1 / (1 + dists)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            avg_sim = np.mean(sims)

            coherence_scores[label] = {
                "name": self._get_topic_name(label),
                "coherence_score": float(avg_sim),
                "size": len(cluster_embeds),
                "std_coherence": float(np.std(sims)),
            }

        return pd.DataFrame(coherence_scores).T.sort_values(
            "coherence_score", ascending=False
        )

    def evaluate_clustering(
        self,
        cleaned_texts: List[str],
        topics: List[int],
        coherence_metric: str = "cosine",
    ) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.

        Returns:
            Dictionary with silhouette_score and davies_bouldin_score
        """
        logger.info("Calculating clustering quality metrics...")
        metrics = {
            "silhouette_score": 0.0,
            "davies_bouldin_score": 0.0,
            "avg_coherence_score": 0.0,
            "num_clusters": len(set(topics)) - (1 if -1 in topics else 0),
            "coverage": sum(1 for t in topics if t != -1) / len(topics),
        }

        # Generate embeddings if not cached
        if self.embeddings is None:
            logger.warning("Embeddings not cached, regenerating...")
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            self.embeddings = self.embedding_model.encode(
                cleaned_texts, show_progress_bar=True, convert_to_numpy=True
            )

        # Filter out outliers for metrics
        mask = np.array(topics) != -1
        filtered_embeddings = self.embeddings[mask]
        filtered_topics = np.array(topics)[mask]

        if len(np.unique(filtered_topics)) >= 2:
            try:
                sil_score = silhouette_score(
                    filtered_embeddings,
                    filtered_topics,
                    sample_size=min(10000, len(filtered_topics)),
                )
                db_score = davies_bouldin_score(filtered_embeddings, filtered_topics)

                metrics["silhouette_score"] = float(sil_score)
                metrics["davies_bouldin_score"] = float(db_score)

                logger.info(f"Silhouette Score: {sil_score:.4f}")
                logger.info(f"Davies-Bouldin Score: {db_score:.4f}")
            except Exception as e:
                logger.error(f"Failed to calculate geometric metrics: {e}")
        else:
            logger.warning("Not enough clusters for geometric metrics.")

        try:
            coherence_df = self._calculate_coherence(topics, metric=coherence_metric)

            if not coherence_df.empty:
                avg_coherence = coherence_df["coherence_score"].mean()
                metrics["avg_coherence_score"] = float(avg_coherence)
                logger.info(f"Average Coherence Score: {avg_coherence:.4f}")

                # Save detailed report
                output_path = f"{self.config.OUTPUT_DIR}/report_topic_coherence.csv"
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                coherence_df.to_csv(output_path)
                logger.info(f"Detailed coherence report saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to calculate coherence: {e}")

        return metrics

    def generate_visualizations(self):
        """
        Generate BERTopic's built-in visualizations.
        """
        if not self.config.CREATE_VISUALIZATIONS:
            return

        logger.info("Generating visualizations...")
        viz_dir = Path(self.config.VIZ_OUTPUT_DIR)
        viz_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Topic distance map
            fig = self.topic_model.visualize_topics()
            fig.write_html(str(viz_dir / "topic_map.html"))
            logger.info(f"Topic map saved to {viz_dir / 'topic_map.html'}")

            # Document clusters (if embeddings available)
            if self.embeddings is not None:
                fig = self.topic_model.visualize_documents(
                    self.cleaned_texts,
                    embeddings=self.embeddings,
                    hide_annotations=True,
                )
                fig.write_html(str(viz_dir / "document_clusters.html"))
                logger.info(f"Document clusters saved")

            # Topic hierarchy
            fig = self.topic_model.visualize_hierarchy()
            fig.write_html(str(viz_dir / "hierarchy.html"))
            logger.info(f"Hierarchy saved")

        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")

    def save_results(
        self,
        df_original: pd.DataFrame,
        valid_indices: List[int],
        topics: List[int],
        output_path: str,
    ) -> pd.DataFrame:
        """
        Save discovery results to CSV.

        Args:
            df_original: Original dataframe
            valid_indices: Indices of valid texts
            topics: Topic assignments
            output_path: Where to save results
        """
        logger.info(f"Saving results to {output_path}")

        # Get topic labels
        topic_info = self.get_topic_info()
        source_col = "Main" if "Main" in topic_info.columns else "Name"

        topic_labels = {}
        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]

            if topic_id == -1:
                topic_labels[topic_id] = "NOISE"
                continue

            # Clean LLM-generated label
            raw_label = row[source_col]
            if isinstance(raw_label, list):
                label_text = " ".join(raw_label[:4])
            else:
                label_text = str(raw_label)

            # Normalize label
            clean_label = (
                label_text.replace("_", " ").replace('"', "").replace("'", "").strip()
            )
            clean_label = re.sub(r"^\d+\s*", "", clean_label)
            clean_label = re.sub(r"\s+", " ", clean_label)

            topic_labels[topic_id] = clean_label

        # Create results dataframe
        df_results = df_original.iloc[valid_indices].copy()
        df_results["discovered_topic_id"] = topics
        df_results["discovered_intent"] = df_results["discovered_topic_id"].map(
            topic_labels
        )

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_path, index=False, encoding="utf-8-sig")

        logger.info(f"Results saved: {len(df_results)} rows")

        return df_results

    def cleanup(self):
        """
        Proper cleanup of all GPU resources.
        """
        logger.info("Cleaning up GPU memory...")

        if self.llm_labeler:
            try:
                if hasattr(self.llm_labeler, "generator"):
                    del self.llm_labeler.generator
                if hasattr(self.llm_labeler, "tokenizer"):
                    del self.llm_labeler.tokenizer
                del self.llm_labeler
            except Exception as e:
                logger.warning(f"Error cleaning LLM: {e}")

        if self.embedding_model:
            del self.embedding_model

        if self.topic_model:
            del self.topic_model

        self.embeddings = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleanup complete")
