import torch
from model import CNN
from oneHotEncoding import text_to_tensor
import numpy as np


def predict_sentiment(text, model):

    classes = ["negative", "neutral", "positive"]
    idx_to_class = {i: c for i, c in enumerate(classes)}

    model.eval()
    x = text_to_tensor(text)

    with torch.no_grad():
        logits = model(x)                     # (1, 3)
        probs = torch.softmax(logits, dim=1)  # (1, 3)
        probs = probs.cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]

    return pred_label

def init_model():
    model = CNN()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()

    print('Model Initialized')

    return model
