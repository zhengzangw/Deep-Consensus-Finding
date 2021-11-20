import os

import torch
from torch.utils.data import DataLoader, Dataset

meta = torch.load("./data/meta_info.pth")
DICT = meta["dict"]  # {'A': 0, 'C': 1, 'G': 2, 'T': 3}
MAX_LEN = meta["max_len"]  # max length of a strand; typically 120; add a const for insertion length > 120
MAX_T = meta["max_t"]  # max num of noisy strands in a cluster; typically 8
vin = meta["vin"]
vout = meta["vout"]  # {'A': 0, 'C': 1, 'G': 2, 'T': 3, '<BOS>': 4, '<EOS>': 5}


class DNA_dataset(Dataset):
    def __init__(self, root, split, *args, **kwargs):
        super(DNA_dataset, self).__init__(*args, **kwargs)
        self.data_left = []  # [num_samples, MAX_LEN]
        self.data_right = []  # [num_samples, MAX_LEN]
        self.labels = []  # [num_samples, MAX_LEN]

        split_dir = os.path.join(root, split)
        self.load_data(split_dir)

    def load_data(self, split_dir):
        noisy_f = open(os.path.join(split_dir, "noisy_strands.txt"), "r")
        cluster_f = open(os.path.join(split_dir, "clusters.txt"), "r")

        noise_lines = noisy_f.readlines()
        idx = 0
        while idx < len(noise_lines):
            line = noise_lines[idx].strip()
            assert line.startswith("#")
            num_strands = int(line.split()[-1])
            idx += 1
            strands = noise_lines[idx: idx + num_strands]
            left_word_ids = self.strands_to_id(strands, idx, align='left')  # tensor: [MAX_LEN]
            right_word_ids = self.strands_to_id(strands, idx, align='right')  # tensor: [MAX_LEN]
            self.data_left.append(left_word_ids)
            self.data_right.append(right_word_ids)

            idx += num_strands

        # gt strand
        cluster_lines = cluster_f.readlines()
        for line in cluster_lines:
            self.labels.append(self.strand_to_id(line.strip()))

    def strands_to_id(self, strands, idx, align='left'):
        assert align in ['left', 'right']
        paddings = len(DICT) * torch.ones(MAX_T, MAX_LEN, dtype=torch.int)
        for i, s in enumerate(strands):
            s = s.strip()[:MAX_LEN]
            if align == 'left':
                paddings[i][: len(s)] = torch.tensor(list(map(DICT.get, s)))
            else:
                paddings[i][-len(s):] = torch.tensor(list(map(DICT.get, s)))

        ids = []
        mul = torch.tensor([(len(DICT) + 1) ** k for k in range(MAX_T)])
        for pos in range(MAX_LEN):
            num = int(sum(mul * paddings[:, pos]))
            # assert num in vin.keys()
            # if num not in vin.keys():
            #     print(idx, pos)
            ids.append(vin[num])
        return torch.tensor(ids)

    def strand_to_id(self, s):
        ids = [vout[c] for c in s]
        return torch.tensor(ids)

    def __getitem__(self, idx):
        return self.data_left[idx], self.data_right[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def get_DNA_loader(root, device="cpu", batch_size=128):
    def collate_batch(batch):
        src_left, src_right, tgt = [], [], []
        for data_left, data_right, label in batch:
            data_left = [torch.tensor([vin["<BOS>"]]), data_left, torch.tensor([vin["<EOS>"]])]
            data_right = [torch.tensor([vin["<BOS>"]]), data_right, torch.tensor([vin["<EOS>"]])]
            label = [torch.tensor([vout["<BOS>"]]), label, torch.tensor([vout["<EOS>"]])]
            src_left.append(torch.cat(data_left))
            src_right.append(torch.cat(data_right))
            tgt.append(torch.cat(label))
        src_left = torch.stack(src_left, dim=0).T.contiguous().to(device)
        src_right = torch.stack(src_right, dim=0).T.contiguous().to(device)
        tgt = torch.stack(tgt, dim=0).T.contiguous().to(device)
        return src_left, src_right, tgt

    train_ds = DNA_dataset(root=root, split="train")
    val_ds = DNA_dataset(root=root, split="val")
    test_ds = DNA_dataset(root=root, split="test")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    dataset_info = dict(input_dim=len(vin), output_dim=len(vout))
    return train_dl, val_dl, test_dl, dataset_info


if __name__ == "__main__":
    train_dl = get_DNA_loader(root="./data", batch_size=2)[0]
    data = next(iter(train_dl))
