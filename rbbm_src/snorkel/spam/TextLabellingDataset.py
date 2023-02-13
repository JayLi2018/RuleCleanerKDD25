import torch
import numpy as np
from torch.utils import data


class TextLabellingDataset(data.Dataset):
    def __init__(self, dataset, vocab, tokenizer, max_len=512):
        self.vocab = vocab
        self.max_len = max_len
        self.tokenizer = tokenizer
        # read path
        if isinstance(dataset, str):
            return
        else:
            (texts, lf_results, pred_labels) = dataset
        sents = []
        lfrs = []
        labels = []
        for i, x in enumerate(zip(texts, lf_results, pred_labels)):
            e = x[0]
            lfr = x[1]
            l = x[2]
            sents.append(e)
            # lfr_str = '[LFR]'.join(self.tokenizer(lfr))
            # lfrs.append(['[LFR]'] + lfr_str.split('[LFR]'))
            lfrs.append(lfr)
            labels.append(l)
        self.sents = sents
        self.lfrs = lfrs
        self.labels = labels
        # vocab
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.vocab)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.vocab)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Return the ith item of in the dataset.
        Args:
            idx (int): the element index
        Returns (TODO):
            words, x, is_heads, tags, mask, y, seqlen, self.taskname
        """
        words, lfr, label = self.sents[idx], self.lfrs[idx], self.labels[idx]        
        tokens = ["[CLS]"] + self.tokenizer.tokenize(words) +  ["[SEP]"]
        tokens += self.tokenizer.tokenize(' '.join(lfr)) +  ["[SEP]"]
        x = tokenizer.convert_tokens_to_ids(tokens)[:self.max_len]
        y = self.tag2idx[label] # label
        mask = [1] * len(x)

        assert len(x)==len(mask), \
          f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"
        # seqlen
        seqlen = len(mask)

        return words, x, label, mask, y, seqlen

    @staticmethod
    def pad(batch):
        '''Pads to the longest sample
        Args:
            batch:
        Returns (TODO):
            return words, f(x), is_heads, tags, f(mask), f(y), seqlens, name
        '''
        f = lambda x: [sample[x] for sample in batch]
        g = lambda x, seqlen, val: \
              [sample[x] + [val] * (seqlen - len(sample[x])) \
               for sample in batch] # 0: <pad>

        # get maximal sequence length
        seqlens = f(5)
        maxlen = np.array(seqlens).max()

        words = f(0)
        x = g(1, maxlen, 0)
        tags = f(2)
        mask = g(3, maxlen, 1)
        y = f(4)

        f = torch.LongTensor
        if isinstance(y[0], float):
            y = torch.Tensor(y)
        else:
            y = torch.LongTensor(y)
        return words, f(x), tags, f(mask), y, seqlens

   