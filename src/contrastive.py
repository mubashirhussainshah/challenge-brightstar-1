import logging
import torch
import random
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    models,
    util,
)
from .config import Config
from tqdm import tqdm


def train_contrastive_embeddings(
    config: Config,
    df_train: pd.DataFrame,
    text_col: str,
    label_col: str,
    output_path: str = "contrastive_bert_base",
) -> str:
    """
    Stage 1: Fine-tune the BERT body using Contrastive Learning.
    This aligns embeddings so that same-intents are close and diff-intents are far.
    """
    logging.info("=" * 70)
    logging.info("STAGE 1: CONTRASTIVE EMBEDDING TRAINING")
    logging.info("=" * 70)

    train_examples = []

    # Group by intent to create positive pairs
    intent_groups = df_train.groupby(label_col)[text_col].apply(list).to_dict()

    import random

    random.seed(42)

    # Generate pairs (Anchor, Positive)
    for intent, texts in intent_groups.items():
        if len(texts) < 2:
            continue

        # Create pairs: (Text A, Text B) where both have same intent
        for i in range(len(texts)):
            anchor = texts[i]
            # Pick a random positive that isn't the anchor
            positive_idx = (i + 1) % len(texts)
            positive = texts[positive_idx]

            train_examples.append(InputExample(texts=[anchor, positive], label=1))

            # Add another pair with a different positive
            if len(texts) > 2:
                pos_idx_2 = (i + 2) % len(texts)
                train_examples.append(
                    InputExample(texts=[anchor, texts[pos_idx_2]], label=1)
                )

    logging.info(f"Generated {len(train_examples)} contrastive pairs.")

    # Define Model Architecture for Sentence Transformers
    word_embedding_model = models.Transformer(
        config.MODEL_NAME, max_seq_length=config.MAX_LEN
    )
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=config.BATCH_SIZE
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=int(len(train_dataloader) * 0.1),
        show_progress_bar=True,
        output_path=output_path,
    )

    logging.info(f"Contrastive Model saved to {output_path}")
    return output_path
