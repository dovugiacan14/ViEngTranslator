import torch 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TranslationDataset(Dataset): 
    def __init__(self, data): 
        """
        data: list of tuples
        """
        self.data = data 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        src, tgt = self.data[index]
        return torch.tensor(src, dtype= torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch):
    """
    batch: list of (src_tensor, tgt_tensor)
    Returns:
        src_padded: (batch_size, src_len_max)
        tgt_padded: (batch_size, tgt_len_max)
        src_mask: (batch_size, src_len_max) -> mask cho attention
    Note: use look-ahead mask in case process with Transformer 
    """
    src_batch, tgt_batch = zip(*batch) 

    # pad sequences to max length in batch 
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    # Mask: True in pad position
    src_mask = (src_padded == 0)
    tgt_mask = (tgt_padded == 0)
    return src_padded, tgt_padded, src_mask, tgt_mask