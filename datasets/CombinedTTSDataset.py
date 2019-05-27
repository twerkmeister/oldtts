from torch.utils.data import Dataset
import numpy as np
import random


class CombinedTTSDataset(Dataset):
    """A Dataset instance that combines multiple underlying datasets.

    The underlying datasets can have different characteristics like languages,
    phoneme use, audio parameters, etc."""

    def __init__(self,
                 datasets,
                 batch_group_size=0,
                 min_seq_len=0,
                 max_seq_len=float("inf"),
                 verbose=False):
        """
            Args:
                datasets (list): list of TTSDataset instances.
                batch_group_size (int): (0) range of batch randomization after
                                        sorting sequences by length.
                min_seq_len (int): (0) minimum sequence length to be processed
                by the loader.
                max_seq_len (int): (float("inf")) maximum sequence length.
        """
        self.datasets = datasets
        self.batch_group_size = batch_group_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.verbose = verbose
        self.index = self.generate_index()

    def _shuffle_batch_groups(self, length_idx):
        """Shuffles the data in local batch groups."""
        length_idx = np.copy(length_idx)
        if self.batch_group_size > 0:
            for i in range(len(length_idx) // self.batch_group_size):
                offset = i * self.batch_group_size
                end_offset = offset + self.batch_group_size
                batch_group = length_idx[offset:end_offset]
                random.shuffle(batch_group)
                length_idx[offset:end_offset] = batch_group
        return length_idx

    def _filter_lengths(self, index, text_lengths):
        """Filters out examples outside min and max seq len."""
        joined = zip(index, text_lengths)
        joined = [x for x in joined
                  if self.min_seq_len < x[1] < self.max_seq_len]
        index, text_lengths = zip(*joined)
        return index, text_lengths

    def generate_index(self):
        """Creates a mapping from idx to dataset & that dataset's idx."""
        index = []
        text_lengths = []
        for i, ds in enumerate(self.datasets):
            ds_idx_broadcasted = [i] * len(ds)
            ds_item_idx = range(len(ds))
            index.extend(zip(ds_idx_broadcasted, ds_item_idx))
            text_lengths.extend([len(it[0]) for it in ds.items])

        num_total_items = len(index)
        index, text_lengths = self._filter_lengths(index, text_lengths)
        num_items_filtered = len(index)

        if self.verbose:
            print("Combined dataset has {} items. {} items were filtered "
                  "out because they were either too short or too long."
                  "".format(num_items_filtered,
                            num_total_items-num_items_filtered))
            print("Max text len: {}, min text len: {}, avg text len: {}"
                  "".format(np.max(text_lengths),
                            np.min(text_lengths),
                            np.average(text_lengths)))

        length_idx = np.argsort(text_lengths)
        length_idx = self._shuffle_batch_groups(length_idx)

        return np.array(index)[length_idx]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ds_idx, item_idx = self.index[idx]
        return self.datasets[ds_idx][item_idx]
