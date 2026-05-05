import pandas as pd
import torch
from transformers import AutoTokenizer, T5EncoderModel


def load_data(csv_path):
    """
    Load texts and labels from a CSV file.

    The CSV file is expected to contain two columns:
      - text: input text
      - label: integer class label
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])

    texts = df["text"].tolist()
    labels = df["label"].astype(int).tolist()

    return texts, labels


def get_gtr_token_embeddings(texts, model_path, device="cuda"):
    """
    Obtain mean-pooled text embeddings using a T5 encoder.

    Args:
        texts: A list of input texts.
        model_path: Local model path or Hugging Face model name.
        device: Device used for inference, e.g., "cuda" or "cpu".

    Returns:
        A tensor of shape [B, H], where B is the number of texts and
        H is the hidden size of the encoder.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5EncoderModel.from_pretrained(model_path).to(device)
    model.eval()

    all_embeddings = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            encoder_outputs = model(**inputs).last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)

            masked_hidden = encoder_outputs * attention_mask
            summed = masked_hidden.sum(dim=1)
            counts = attention_mask.sum(dim=1)
            mean_pooled = summed / counts

        all_embeddings.append(mean_pooled.squeeze(0))

    return torch.stack(all_embeddings)