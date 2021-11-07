import spacy
import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k

BATCH_SIZE = 128


def load_multi30k(device="cpu"):
    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")
    tokenize_de = lambda text: [tok.text.lower() for tok in spacy_de.tokenizer(text)][::-1]
    tokenize_en = lambda text: [tok.text.lower() for tok in spacy_en.tokenizer(text)]
    train_data_de, train_data_en = zip(*Multi30k(split="train"))
    vocab_de = torchtext.vocab.build_vocab_from_iterator(
        map(tokenize_de, train_data_de), min_freq=2, specials=["<unk>", "<BOS>", "<EOS>", "<PAD>"]
    )
    vocab_de.set_default_index(vocab_de["<unk>"])
    vocab_en = torchtext.vocab.build_vocab_from_iterator(
        map(tokenize_en, train_data_en), min_freq=2, specials=["<unk>", "<BOS>", "<EOS>", "<PAD>"]
    )
    vocab_en.set_default_index(vocab_en["<unk>"])

    def text_transform(x, vocab, tokenizer):
        return [vocab["<BOS>"]] + [vocab[token] for token in tokenizer(x)] + [vocab["<EOS>"]]

    def collate_batch(batch):
        de_list, en_list = [], []
        for (de, en) in batch:
            de_list.append(torch.tensor(text_transform(de, vocab_de, tokenize_de)))
            en_list.append(torch.tensor(text_transform(en, vocab_en, tokenize_en)))
        de_list = pad_sequence(de_list, padding_value=vocab_de["<PAD>"]).to(device)
        en_list = pad_sequence(en_list, padding_value=vocab_en["<PAD>"]).to(device)
        return de_list, en_list

    def bucket_iterator(split):
        return DataLoader(
            list(Multi30k(split=split)),
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_batch,
        )

    train_dataloader, valid_dataloader, test_dataloader = map(
        bucket_iterator, ["train", "valid", "test"]
    )
    dataset_info = dict(
        input_dim=len(vocab_de), output_dim=len(vocab_en), output_pad=vocab_de["<PAD>"]
    )

    return train_dataloader, valid_dataloader, test_dataloader, dataset_info
