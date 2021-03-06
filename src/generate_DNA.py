import argparse
import os
import random

import torch

DICT = "ACGT"
DICT2 = {"A": 0, "C": 1, "G": 2, "T": 3}


def gen_sample_dj(gt, sub_p, del_p, ins_p):
    res = []
    for w in gt:
        r = random.random()
        if r < sub_p:
            res.append(random.choice(DICT))
        elif r < sub_p + ins_p:
            res.append(random.choice(DICT))
            res.append(w)
        elif r > sub_p + ins_p + del_p:
            res.append(w)
    return "".join(res)


def generate_gt(args):
    for i, split in enumerate(["train", "val", "test"]):
        if not os.path.exists(f"./data/{split}"):
            os.makedirs(f"./data/{split}")

        outf = open(f"./data/{split}/clusters.txt", "w")
        for k in range(args.num_strands[i]):
            s = "".join([random.choice(DICT) for _ in range(args.max_len)]) + "\n"
            outf.write(s)


def generate_noise_strands(args):
    for i, split in enumerate(["train", "val", "test"]):
        inf = open(f"./data/{split}/clusters.txt", "r")
        outf = open(f"./data/{split}/noisy_strands.txt", "w")
        lines = inf.readlines()
        for j, line in enumerate(lines, start=1):
            line = line.strip()
            # num_strands = random.randint(args.max_t - 2, args.max_t)
            num_strands = args.max_t
            outf.write(f"# {j} {num_strands}\n")
            for _ in range(num_strands):
                s = gen_sample_dj(line, args.subp, args.delp, args.insp) + "\n"
                outf.write(s)


def build_meta(max_len, max_t):
    meta = {"max_len": max_len, "max_t": max_t, "dict": DICT2}

    # build vin
    vin = {"<BOS>": 0, "<EOS>": 1}
    cnt = 2
    mul = torch.tensor([(len(DICT2) + 1) ** k for k in range(max_t)])
    for split in ["train", "val", "test"]:
        root_dir = os.path.join("./data", split)
        noisy_f = open(os.path.join(root_dir, "noisy_strands.txt"), "r")

        noise_lines = noisy_f.readlines()
        idx = 0
        while idx < len(noise_lines):
            line = noise_lines[idx].strip()
            assert line.startswith("#")
            num_strands = int(line.split()[-1])
            idx += 1
            strands = noise_lines[idx : idx + num_strands]

            # update vin
            # left align
            paddings = len(DICT2) * torch.ones(max_t, max_len, dtype=torch.int)
            for i, s in enumerate(strands):
                s = s.strip()[:max_len]
                paddings[i][: len(s)] = torch.tensor(list(map(DICT2.get, s)))
            for pos in range(max_len):
                num = int(sum(mul * paddings[:, pos]))
                if num not in vin.keys():
                    vin[num] = cnt
                    cnt += 1
            # right align
            paddings = len(DICT2) * torch.ones(max_t, max_len, dtype=torch.int)
            for i, s in enumerate(strands):
                s = s.strip()[:max_len]
                paddings[i][-len(s) :] = torch.tensor(list(map(DICT2.get, s)))
            for pos in range(max_len):
                num = int(sum(mul * paddings[:, pos]))
                if num not in vin.keys():
                    vin[num] = cnt
                    cnt += 1

            idx += num_strands
    meta["vin"] = vin

    # vout
    vout = {"A": 0, "C": 1, "G": 2, "T": 3, "<BOS>": 4, "<EOS>": 5}
    meta["vout"] = vout

    torch.save(meta, "./data/meta_info.pth")


def build_meta_v2(max_len, max_t):
    meta = {"max_len": max_len, "max_t": max_t, "dict": DICT2}
    # vin
    # vin = {"<BOS>": 0, "<EOS>": 1, 'A': 2, 'C': 3, 'G': 4, 'T': 5, '<PAD>': 6}
    vin = {"A": 0, "C": 1, "G": 2, "T": 3}
    meta["vin"] = vin

    # vout
    vout = {"A": 0, "C": 1, "G": 2, "T": 3}
    meta["vout"] = vout

    torch.save(meta, "./data/meta_info_v2.pth")


def main():
    # python -m src.generate_DNA --subp 0.013 --delp 0.013 --insp 0.013
    parser = argparse.ArgumentParser("Noise generation")
    parser.add_argument("--max-len", type=int, default=120)
    parser.add_argument("--max-t", type=int, default=8)
    parser.add_argument(
        "--num-strands",
        type=list,
        nargs="+",
        default=[50000, 1000, 1000],
        help="[num_train, num_val, num_test]",
    )

    parser.add_argument("--subp", type=float, default=0.01)
    parser.add_argument("--delp", type=float, default=0.01)
    parser.add_argument("--insp", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    random.seed(args.seed)

    # # generate GT
    generate_gt(args)

    # # generate noisy strands
    generate_noise_strands(args)

    # build meta; include max_len, max_t, vin, DICT
    # Add a const to max_len in case insertion > max_len
    # build_meta(args.max_len, args.max_t)
    build_meta_v2(args.max_len, args.max_t)


if __name__ == "__main__":
    main()
