# /usr/bin/env python3

import argparse
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .dataset_nlp import load_multi30k
from .model import get_model

SEED = 1234
device = "cuda:0"

N_EPOCHS = 10
CLIP = 1


def init_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, clip):

    model.train()
    epoch_loss = 0
    for i, batch in tqdm(enumerate(iterator)):
        src, trg = batch
        optimizer.zero_grad()
        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


@torch.no_grad()
def evaluate(model, iterator, criterion):

    model.eval()
    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src, trg = batch

        output = model(src, trg, 0)  # turn off teacher forcing

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


@torch.no_grad()
def test(model, iterator, criterion):
    model.load_state_dict(torch.load("tut1-model.pt"))

    test_loss = evaluate(model, iterator, criterion)

    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")


def main(args):
    train_dataloader, valid_dataloader, test_dataloader, dataset_info = load_multi30k(
        device=device
    )
    # dataloader produce (src, tgt)
    # src: [Seq_len, batch size, id], for DNA strand, id in [4 * max cluster size]
    # tgt: [Seq_len, batch size, id], for DNA strand, id in [4]

    INPUT_DIM = dataset_info["input_dim"]
    OUTPUT_DIM = dataset_info["output_dim"]
    TRG_PAD_IDX = dataset_info["output_pad"]
    model = get_model(INPUT_DIM, OUTPUT_DIM, device=device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_dataloader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "tut1-model.pt")

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")

    test(model, test_dataloader, criterion)


def parseargs(arg=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="default", type=str, help="Name of experiments.")
    args = parser.parse_args(arg)
    return args


if __name__ == "__main__":
    args = parseargs()
    main(args)
