"""
model_utils.py
----------------
Builds the shared bert-base-uncased + LoRA severity-classification model
used identically by all three conditions (centralized, DP-FedAvg,
CipherTune), and provides:

  * build_model()              -- fresh model + tokenizer
  * trainable_param_names()    -- fixed, deterministic order of the
                                  trainable tensors (LoRA A/B matrices +
                                  classifier head) so that every client's
                                  flattened delta vector lines up
                                  element-for-element across clients and
                                  across rounds.
  * get_trainable_vector() / set_trainable_vector()
                                -- flatten <-> load, used to compute and
                                  apply "deltas" in the federated loop.
  * train_local() / evaluate() -- one local-training pass / a full
                                  evaluation pass that also returns
                                  PER-SAMPLE losses (needed by the
                                  membership-inference attack).

Only the LoRA adapter matrices and the classification head are ever
trainable; the pretrained BERT backbone is frozen throughout, which is
exactly what keeps the encrypted update small enough for Paillier to be
tractable.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

from . import config as C


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class CVEDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_dataloader(texts, labels, tokenizer, batch_size, shuffle):
    ds = CVEDataset(texts, labels, tokenizer, C.MAX_TOKEN_LENGTH)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ----------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------
def _target_module_names() -> List[str]:
    """
    peft's `target_modules` matches by substring against module *names*.
    We restrict LoRA to the query/value projections of a fixed subset of
    encoder layers (config.LORA_TARGET_LAYER_INDICES) to keep the
    trainable-parameter count small enough for Paillier encryption to be
    tractable on commodity hardware (see config.py "SCALE UP" notes).
    """
    names = []
    for layer_idx in C.LORA_TARGET_LAYER_INDICES:
        for module in C.LORA_TARGET_MODULES:
            names.append(f"encoder.layer.{layer_idx}.attention.self.{module}")
    return names


def build_model(device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL_NAME)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        C.BASE_MODEL_NAME, num_labels=C.NUM_LABELS
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=C.LORA_R,
        lora_alpha=C.LORA_ALPHA,
        lora_dropout=C.LORA_DROPOUT,
        target_modules=_target_module_names(),
        modules_to_save=["classifier"],  # classifier head trains fully (small) and travels with the delta too
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    return model, tokenizer


# ----------------------------------------------------------------------
# Flatten / load trainable parameters in a fixed, deterministic order
# ----------------------------------------------------------------------
def trainable_param_names(model) -> List[str]:
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    return sorted(names)  # deterministic order shared by every client


def get_trainable_vector(model) -> Tuple[np.ndarray, List[Tuple[str, torch.Size]]]:
    names = trainable_param_names(model)
    state = dict(model.named_parameters())
    shapes = [(n, state[n].shape) for n in names]
    flat = torch.cat([state[n].detach().reshape(-1) for n in names]).cpu().numpy().astype(np.float32)
    return flat, shapes


def set_trainable_vector(model, vector: np.ndarray, shapes: List[Tuple[str, torch.Size]]):
    state = dict(model.named_parameters())
    offset = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for name, shape in shapes:
            numel = int(np.prod(shape))
            chunk = vector[offset: offset + numel]
            state[name].copy_(torch.tensor(chunk, dtype=state[name].dtype, device=device).reshape(shape))
            offset += numel
    assert offset == len(vector), "vector length does not match the model's trainable-parameter count"


def num_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------
# Local training / evaluation
# ----------------------------------------------------------------------
def train_local(model, dataloader, epochs: int, lr: float, device: str = "cpu"):
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, n_batches = 0.0, 0
    for _ in range(epochs):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device: str = "cpu"):
    """Returns (accuracy, macro_f1, per_sample_losses, all_preds, all_labels)."""
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    all_preds, all_labels, all_losses = [], [], []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        outputs = model(**batch)
        losses = loss_fn(outputs.logits, labels)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_losses.extend(losses.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1, np.array(all_losses), np.array(all_preds), np.array(all_labels)
