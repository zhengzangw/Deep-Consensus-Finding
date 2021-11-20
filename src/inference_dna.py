# /usr/bin/env python3

import torch
from tqdm import tqdm

from .dna_dataset import get_DNA_loader
from .model import get_model

device = "cuda:0"
keys = ["A", "C", "G", "T", "<B>", "<E>"]


def ids_to_strand(ids):
    # ids: [120]
    s = ""
    for i in ids.tolist():
        s += keys[i]
    return s


def val(model, test_dataloader):
    metric = {
        "total_strand": 0,
        "total_char": 0,
        "acc_strand": 0,
        "acc_char": 0,
    }
    for t, data in enumerate(test_dataloader):
        src_data, tgt_data = data
        src_data = src_data.to(device)
        tgt_data = tgt_data.to(device)
        outputs = model(src_data, tgt_data, teacher_forcing_ratio=0)
        pred = outputs.argmax(dim=-1)

        pred = pred[1:-1].t()
        label = tgt_data[1:-1].t()

        for i in range(len(pred)):
            pred_s = ids_to_strand(pred[i])
            lbl_s = ids_to_strand(label[i])

            if t == 0 and i == 0:
                print("An example:")
                print(f"GT: {lbl_s}")
                print(f"PD: {pred_s}")
                print("====================")

            metric["total_strand"] += 1
            metric["total_char"] += len(lbl_s)
            metric["acc_strand"] += pred_s == lbl_s
            metric["acc_char"] += sum([x == y for x, y in zip(pred_s, lbl_s)])

    print(
        f"""\
Total strand: {metric["total_strand"]}, length of strand: {len(lbl_s)}
Acc (strand): {metric["acc_strand"] / metric["total_strand"] * 100:.4f}% ({metric["acc_strand"]})
Acc (char): {metric["acc_char"] / metric["total_char"] * 100:.4f}% ({metric["acc_char"]})
    """
    )


def main(args=None):
    _, _, test_dataloader, dataset_info = get_DNA_loader(root="./data", device="cpu")

    INPUT_DIM = dataset_info["input_dim"]
    OUTPUT_DIM = dataset_info["output_dim"]
    model = get_model(INPUT_DIM, OUTPUT_DIM, device=device)
    model.load_state_dict(torch.load("tut1-model.pt"))
    model.eval()

    val(model, test_dataloader)


if __name__ == "__main__":
    main()
