import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from . import config
import logging


class IntentDataset(Dataset):
    """
    Dataset that holds raw text and tokenizes on-the-fly.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_len: int,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize only this specific sample
        # padding=False. DataCollator will pad the batch later.
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,  # standard lists, not tensor
        )

        item = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": label,
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"]

        return item


def load_and_prep_data(
    config: config.Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, int]:
    """Load and prepare dataset with stratified split"""
    logging.info("=" * 70)
    logging.info("LOADING DATASET")
    logging.info("=" * 70)

    # Read and sanitize dataset
    df = pd.read_csv(config.MAIN_DATA_FILE)
    df = df.dropna(subset=[config.RAW_TEXT_COLUMN, config.LABEL_COLUMN])
    df[config.RAW_TEXT_COLUMN] = df[config.RAW_TEXT_COLUMN].astype(str).str.strip()
    df.reset_index(drop=True, inplace=True)
    logging.info(f"Loaded {len(df)} samples")

    # Stratified split
    df_trainval, df_test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=df[config.LABEL_COLUMN],
    )

    # Create label mappings
    target_intents = sorted(df[config.LABEL_COLUMN].unique())
    num_labels = len(target_intents)
    label2id = {label: i for i, label in enumerate(target_intents)}
    id2label = {i: label for i, label in enumerate(target_intents)}
    logging.info(f"Model will train on {num_labels} valid intents.")

    # Show distribution
    intent_counts = df[config.LABEL_COLUMN].value_counts()
    logging.info(f"{num_labels} intents found:")
    for intent, count in intent_counts.head(10).items():
        logging.info(f"  {intent[:50]:50s} : {count:4d} ({count/len(df)*100:5.2f}%)")

    df_trainval["labels"] = df_trainval[config.LABEL_COLUMN].map(label2id)
    df_test["labels"] = df_test[config.LABEL_COLUMN].map(label2id)

    # Save splits
    df_trainval.to_csv(config.TRAINVAL_SET_PATH, index=False)
    df_test.to_csv(config.TEST_SET_PATH, index=False)

    logging.info(f"Train/Val: {len(df_trainval)} | Test: {len(df_test)}")

    return df_trainval, df_test, label2id, id2label, num_labels


def debug_dataset(
    df: pd.DataFrame, label_col: str, n_per_class: int = 10, seed: int = 42
) -> pd.DataFrame:
    logging.info(f"Creating debug dataset: {n_per_class} samples per class...")

    # Group by label and sample
    debug_df = (
        df.groupby(label_col, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), n_per_class), random_state=seed))
        .reset_index(drop=True)
    )

    logging.info(f"Debug dataset created: {len(debug_df)} total samples.")
    return debug_df
