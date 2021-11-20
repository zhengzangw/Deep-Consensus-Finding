import os

import torch
from torch.utils.data import DataLoader, Dataset

meta = torch.load("./data/meta_info_v2.pth")
# DICT = meta["dict"]  # {'A': 0, 'C': 1, 'G': 2, 'T': 3}
char_dict = meta["dict"]  # {'A': 0, 'C': 1, 'G': 2, 'T': 3}
MAX_LEN = meta["max_len"]  # max length of a strand; typically 120; add a const for insertion length > 120
MAX_T = meta["max_t"]  # max num of noisy strands in a cluster; typically 8
vin = meta["vin"]  # {'A': 0, 'C': 1, 'G': 2, 'T': 3}
vout = meta["vout"]  # {'A': 0, 'C': 1, 'G': 2, 'T': 3}


class DNA_dataset_v2(Dataset):
    def __init__(self, root, split, *args, **kwargs):
        super(DNA_dataset_v2, self).__init__(*args, **kwargs)
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
            onehot_left = self.strands_to_onehot(strands, align='left')  # tensor: (MAX_T, MAX_LEN, len(char_dict))
            onehot_right = self.strands_to_onehot(strands, align='right')  # tensor: (MAX_T, MAX_LEN, len(char_dict))
            self.data_left.append(onehot_left)
            self.data_right.append(onehot_right)

            idx += num_strands

        # gt strand
        cluster_lines = cluster_f.readlines()
        for line in cluster_lines:
            self.labels.append(self.strand_to_id(line.strip()))

    def strands_to_onehot(self, strands, align='left'):
        assert align in ['left', 'right']
        paddings = torch.zeros(MAX_T, MAX_LEN, len(vin))
        for i, s in enumerate(strands):
            s = s.strip()[:MAX_LEN]
            len_s = len(s)
            if align == 'left':
                ids = torch.tensor(list(map(vin.get, s)))
                paddings[i][torch.arange(len_s), ids] = 1
            else:
                ids = torch.tensor(list(map(vin.get, s[::-1])))
                paddings[i][torch.arange(len_s), ids] = 1

        return paddings

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
            # label = [torch.tensor([vout["<BOS>"]]), label, torch.tensor([vout["<EOS>"]])]
            src_left.append(data_left)
            src_right.append(data_right)
            tgt.append(label)
        src_left = torch.stack(src_left, dim=-1).contiguous().to(device)  # (max_t, max_len, 4, bsz)
        src_right = torch.stack(src_right, dim=-1).contiguous().to(device)
        tgt = torch.stack(tgt, dim=-1).contiguous().to(device)  # (max_len, bsz)
        return src_left, src_right, tgt

    train_ds = DNA_dataset_v2(root=root, split="train")
    val_ds = DNA_dataset_v2(root=root, split="val")
    test_ds = DNA_dataset_v2(root=root, split="test")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    dataset_info = dict(input_dim=len(vin), output_dim=len(vout))
    return train_dl, val_dl, test_dl, dataset_info


if __name__ == "__main__":
    train_dl = get_DNA_loader(root="./data", batch_size=2)[0]
    data = next(iter(train_dl))
