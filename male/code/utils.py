import pandas as pd
import torch
from transformers import AutoTokenizer, T5EncoderModel

def load_data(csv_path):
    """
    从 CSV 文件中加载文本和标签
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['text', 'label'])  # 确保 text 和 label 列没有缺失值
    texts = df['text'].tolist()
    labels = df['label'].astype(int).tolist()
    return texts, labels

def get_gtr_token_embeddings(texts, model_path, device="cuda"):
    """
    使用 gtr-t5-base 模型获取文本的 mean-pooled embedding（768维）

    Args:
        texts: List[str] - 文本列表
        model_path: str - 本地路径或模型名（如 sentence-transformers/gtr-t5-base）
        device: str - "cuda" 或 "cpu"

    Returns:
        Tensor: [B, 768] embedding 向量
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5EncoderModel.from_pretrained(model_path).to(device)
    model.eval()

    all_embeddings = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            encoder_outputs = model(**inputs).last_hidden_state  # [1, seq_len, 768]
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [1, seq_len, 1]
            masked_hidden = encoder_outputs * attention_mask  # [1, seq_len, 768]
            summed = masked_hidden.sum(dim=1)  # [1, 768]
            counts = attention_mask.sum(dim=1)  # [1, 1]
            mean_pooled = summed / counts  # [1, 768]
        all_embeddings.append(mean_pooled.squeeze(0))  # [768]

    return torch.stack(all_embeddings)  # [B, 768]
