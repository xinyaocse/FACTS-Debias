import pandas as pd
import torch
from transformers import AutoTokenizer, T5EncoderModel


def load_data(csv_path):

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])

    texts = df["text"].tolist()
    labels = df["label"].astype(int).tolist()

    return texts, labels


def get_gtr_token_embeddings(texts, model_path, device="cuda"):

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