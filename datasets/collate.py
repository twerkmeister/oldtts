import collections
import torch
import numpy as np
from utils.data import prepare_data, prepare_tensor, prepare_stop_target


def make_collate_func(outputs_per_step):
    """
    Args:
        outputs_per_step (int): number of time frames predicted per step.
    """
    def collate_func(batch):
        r"""
            Perform preprocessing and create a final data batch:
            1. PAD sequences with the longest sequence in the batch
            2. Convert Audio signal to Spectrograms.
            3. PAD sequences that can be divided by r.
            4. Convert Numpy to Torch tensors.
        """

        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.Mapping):
            text_lenghts = np.array([len(d["text"]) for d in batch])
            text_lenghts, ids_sorted_decreasing = torch.sort(
                torch.LongTensor(text_lenghts), dim=0, descending=True)

            item_idxs = [
                batch[idx]['item_idx'] for idx in ids_sorted_decreasing
            ]
            text = [batch[idx]['text'] for idx in ids_sorted_decreasing]

            mel = [batch[idx]['mel'] for idx in ids_sorted_decreasing]
            # linear = [batch[idx]['linear'] for idx in ids_sorted_decreasing]
            speaker_ids = [batch[idx]['speaker_id']
                           for idx in ids_sorted_decreasing]

            mel_lengths = [m.shape[1] + 1 for m in mel]  # +1 for zero-frame


            # compute 'stop token' targets
            stop_targets = [
                np.array([0.] * (mel_len - 1)) for mel_len in mel_lengths
            ]

            # PAD stop targets
            stop_targets = prepare_stop_target(stop_targets,
                                               outputs_per_step)

            # PAD sequences with largest length of the batch
            text = prepare_data(text).astype(np.int32)

            # PAD features with largest length + a zero frame
            # linear = prepare_tensor(linear, outputs_per_step)
            mel = prepare_tensor(mel, outputs_per_step)
            # assert mel.shape[2] == linear.shape[2]

            # B x T x D
            # linear = linear.transpose(0, 2, 1)
            mel = mel.transpose(0, 2, 1)

            # convert things to pytorch
            text_lenghts = torch.LongTensor(text_lenghts)
            text = torch.LongTensor(text)
            # linear = torch.FloatTensor(linear).contiguous()
            mel = torch.FloatTensor(mel).contiguous()
            mel_lengths = torch.LongTensor(mel_lengths)
            stop_targets = torch.FloatTensor(stop_targets)
            speaker_ids = torch.LongTensor(speaker_ids)

            return text, text_lenghts, mel, mel_lengths, stop_targets, \
                speaker_ids, item_idxs

        raise TypeError(("batch must contain tensors, numbers, dicts or lists;\
                         found {}".format(type(batch[0]))))

    return collate_func
