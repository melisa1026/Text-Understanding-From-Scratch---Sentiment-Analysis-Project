import pandas as pd
import numpy as np
import torch

def get_one_hots():

    character_list = list("abcdefghijklmnopqrstuvwxyz.!? 1234567890")

    # One-hot encodings for each character
    one_hot_characters = pd.get_dummies(character_list).astype(int)
    mapping_characters = {
        ch: one_hot_characters.loc[i].values
        for i, ch in enumerate(character_list)
    }

    classes = ["positive", "neutral", "negative"]

    # One-hot encodings for sentiment labels
    one_hot_classes = pd.get_dummies(classes).astype(int)
    mapping_classes = {
        label: one_hot_classes.loc[i].values
        for i, label in enumerate(classes)
    }

    return mapping_characters, mapping_classes

def encode_and_pad_one(text_vecs, max_len=1014, char_dim=40):
    rows = []

    # Truncate
    for v in text_vecs[:max_len]:
        v = np.asarray(v, dtype=np.float32)
        rows.append(v)

    # Pad with zeros if shorter than MAX_LEN
    while len(rows) < max_len:
        rows.append(np.zeros(char_dim, dtype=np.float32))

    # Shape (max_len, char_dim) -> transpose to (char_dim, max_len)
    mat = np.stack(rows, axis=0)
    mat = mat.T
    return mat


def text_to_tensor(text, max_len=1014, char_dim=40):

    mapping_characters = get_one_hots()[0]

    text = str(text).lower().strip()
    char_vecs = []

    for ch in text:
        if ch in mapping_characters:
            char_vecs.append(mapping_characters[ch])
    if not char_vecs:
        char_vecs = [mapping_characters[" "]]

    mat = encode_and_pad_one(char_vecs, max_len=max_len, char_dim=char_dim)  # (char_dim, max_len)
    x = torch.tensor(mat, dtype=torch.float32).unsqueeze(0)  # (1, char_dim, max_len)
    return x