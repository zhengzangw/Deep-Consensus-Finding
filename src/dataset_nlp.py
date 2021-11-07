import spacy
import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k

BATCH_SIZE = 128

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


def load_multi30k(device="cpu"):
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


@torch.no_grad()
def translate_sentence(sentence, model, device, max_len=50, keep_indices=False):
    model.eval()
    if isinstance(sentence, str):
        src_indexes = text_transform(sentence, vocab_de, tokenize_de)
    else:
        src_indexes = sentence

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    tgt_tensor = torch.LongTensor([vocab_en["<BOS>"]] * 50).unsqueeze(1).to(device)

    outputs = model(src_tensor, tgt_tensor, teacher_forcing_ratio=0)
    tgt_indexes = [vocab_de["<BOS>"]]
    for y in outputs:
        pred_token = y.argmax().item()
        tgt_indexes.append(pred_token)
        if pred_token == vocab_de["<EOS>"]:
            break

    tgt_tokens = [vocab_en.lookup_token(i) for i in tgt_indexes]
    ret = tgt_indexes if keep_indices else tgt_tokens

    return ret


def lookup_token_en(indices):
    tgt_indices = []
    for y in indices:
        tgt_indices.append(y)
        if y == vocab_en["<EOS>"]:
            break
    tgt_tokens = [vocab_en.lookup_token(i) for i in tgt_indices]
    return tgt_tokens
