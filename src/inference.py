# /usr/bin/env python3

import argparse

import torch
from torchtext.data.metrics import bleu_score

from .dataset_nlp import load_multi30k, lookup_token_en, translate_sentence
from .model import get_model

device = "cuda:0"


def calculate_bleu(dataloader, model, device, max_len=50):

    tgts = []
    pred_tgts = []

    for data in dataloader:
        src_data, tgt_data = data
        assert src_data.shape[1] == tgt_data.shape[1]
        for i in range(src_data.shape[1]):
            src, tgt = src_data[:, i], tgt_data[:, i]
            pred_tgt = translate_sentence(src, model, device, max_len)[1:-1]
            pred_tgts.append(pred_tgt)
            tgt = lookup_token_en(tgt)
            tgts.append([tgt])
    return bleu_score(pred_tgts, tgts)


def main(args=None):
    _, _, test_dataloader, dataset_info = load_multi30k(device="cpu")

    INPUT_DIM = dataset_info["input_dim"]
    OUTPUT_DIM = dataset_info["output_dim"]
    model = get_model(INPUT_DIM, OUTPUT_DIM, device=device)
    model.load_state_dict(torch.load("tut1-model.pt"))

    src = "ein schwarzer hund und ein gefleckter hund kämpfen."
    tgt = translate_sentence(src, model, device)
    print(tgt)

    bleu_score = calculate_bleu(test_dataloader, model, device)
    print(f"BLEU score = {bleu_score*100:.2f}")


if __name__ == "__main__":
    main()
