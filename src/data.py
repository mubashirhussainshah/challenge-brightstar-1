import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from .config import TrainingConfig
from pathlib import Path


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
    config: TrainingConfig, logger
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, int]:
    """Load and prepare dataset with stratified split"""
    logger.info("=" * 70)
    logger.info("LOADING DATASET")
    logger.info("=" * 70)

    # Read and sanitize dataset
    df = pd.read_csv(config.MAIN_DATA_FILE)
    df = df.dropna(subset=[config.RAW_TEXT_COLUMN, config.LABEL_COLUMN])
    df[config.RAW_TEXT_COLUMN] = df[config.RAW_TEXT_COLUMN].astype(str).str.strip()
    df.reset_index(drop=True, inplace=True)

    # Stratified split
    df_trainval, df_test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=df[config.LABEL_COLUMN],
    )

    df_test = df_test.sample(frac=1, random_state=config.SEED).reset_index(drop=True)

    # Create label mappings
    target_intents = sorted(df[config.LABEL_COLUMN].unique())
    num_labels = len(target_intents)

    label2id = {label: i for i, label in enumerate(target_intents)}
    id2label = {i: label for i, label in enumerate(target_intents)}

    logger.info(f"Model will train on {num_labels} valid intents.")

    # Show distribution
    intent_counts = df[config.LABEL_COLUMN].value_counts()
    logger.info(f"{num_labels} intents found:")
    for intent, count in intent_counts.head(10).items():
        logger.info(f"  {intent[:50]:50s} : {count:4d} ({count/len(df)*100:5.2f}%)")

    df_trainval["labels"] = df_trainval[config.LABEL_COLUMN].map(label2id)
    df_test["labels"] = (
        df_test[config.LABEL_COLUMN].map(label2id).fillna(-1).astype(int)
    )

    # Save splits
    Path(config.TRAINVAL_SET_PATH).parent.mkdir(parents=True, exist_ok=True)
    df_trainval.to_csv(config.TRAINVAL_SET_PATH, index=False)
    Path(config.TEST_SET_PATH).parent.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(config.TEST_SET_PATH, index=False)

    return df_trainval, df_test, label2id, id2label, num_labels
