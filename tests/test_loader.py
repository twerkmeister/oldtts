import os
import unittest
import shutil

from datasets.loading import _setup_dataset, _setup_loader
from utils.generic_utils import load_config
from utils.audio import AudioProcessor


class TestTTSDataset(unittest.TestCase):
    def setUp(self):
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.OUTPATH = os.path.join(file_path, "outputs/loader_tests/")
        os.makedirs(self.OUTPATH, exist_ok=True)
        self.c = load_config(os.path.join(file_path, 'test_config.json'))
        self.ap = AudioProcessor(**self.c.audio)

    def __init__(self, *args, **kwargs):
        super(TestTTSDataset, self).__init__(*args, **kwargs)
        self.max_loader_iter = 4

    def test_loader(self):
        dataset = _setup_dataset(self.ap, self.c, self.c['datasets'])
        data_loader = _setup_loader(self.c, dataset, 0)

        for i, data in enumerate(data_loader):
            if i == self.max_loader_iter:
                break
            text_input = data[0]
            text_lengths = data[1]
            linear_input = data[2]
            mel_input = data[3]
            mel_lengths = data[4]
            stop_target = data[5]
            item_idx = data[6]

            neg_values = text_input[text_input < 0]
            check_count = len(neg_values)
            assert check_count == 0, \
                " !! Negative values in text_input: {}".format(check_count)
            # TODO: more assertion here
            assert linear_input.shape[0] == self.c.batch_size
            assert linear_input.shape[2] == self.ap.num_freq
            assert mel_input.shape[0] == self.c.batch_size
            assert mel_input.shape[2] == self.c.audio['num_mels']
            # check normalization ranges
            if self.ap.symmetric_norm:
                assert mel_input.max() <= self.ap.max_norm
                assert mel_input.min() >= -self.ap.max_norm
                assert mel_input.min() < 0
            else:
                assert mel_input.max() <= self.ap.max_norm
                assert mel_input.min() >= 0

    def test_batch_group_shuffle(self):
        dataset = _setup_dataset(self.ap, self.c, self.c['datasets'])
        data_loader = _setup_loader(self.c, dataset, 0)
        last_length = 0
        for i, data in enumerate(data_loader):
            if i == self.max_loader_iter:
                break
            text_input = data[0]
            text_lengths = data[1]
            linear_input = data[2]
            mel_input = data[3]
            mel_lengths = data[4]
            stop_target = data[5]
            item_idx = data[6]

            avg_length = mel_lengths.numpy().mean()
            assert avg_length >= last_length
        # data_loader.dataset.sort_items()
        # assert frames[0] != data_loader.dataset.items[0]

    def test_padding_and_spec(self):
        self.c.batch_size = 1
        dataset = _setup_dataset(self.ap, self.c, self.c['datasets'])
        data_loader = _setup_loader(self.c, dataset, 0)

        for i, data in enumerate(data_loader):
            if i == self.max_loader_iter:
                break
            text_input = data[0]
            text_lengths = data[1]
            linear_input = data[2]
            mel_input = data[3]
            mel_lengths = data[4]
            stop_target = data[5]
            item_idx = data[6]

            # check mel_spec consistency
            wav = self.ap.load_wav(item_idx[0])
            mel = self.ap.melspectrogram(wav)
            mel_dl = mel_input[0].cpu().numpy()
            assert (abs(mel.T).astype("float32")
                    - abs(mel_dl[:-1])
                    ).sum() == 0

            # check mel-spec correctness
            mel_spec = mel_input[0].cpu().numpy()
            wav = self.ap.inv_mel_spectrogram(mel_spec.T)
            self.ap.save_wav(wav, self.OUTPATH + '/mel_inv_dataloader.wav')
            shutil.copy(item_idx[0], self.OUTPATH + '/mel_target_dataloader.wav')

            # check linear-spec
            linear_spec = linear_input[0].cpu().numpy()
            wav = self.ap.inv_spectrogram(linear_spec.T)
            self.ap.save_wav(wav, self.OUTPATH + '/linear_inv_dataloader.wav')
            shutil.copy(item_idx[0],
                        self.OUTPATH + '/linear_target_dataloader.wav')

            # check the last time step to be zero padded
            assert linear_input[0, -1].sum() == 0
            assert linear_input[0, -2].sum() != 0
            assert mel_input[0, -1].sum() == 0
            assert mel_input[0, -2].sum() != 0
            assert stop_target[0, -1] == 1
            assert stop_target[0, -2] == 0
            assert stop_target.sum() == 1
            assert len(mel_lengths.shape) == 1
            assert mel_lengths[0] == linear_input[0].shape[0]
            assert mel_lengths[0] == mel_input[0].shape[0]

        # Test for batch size 2
        self.c.batch_size = 2
        dataset = _setup_dataset(self.ap, self.c, self.c['datasets'])
        data_loader = _setup_loader(self.c, dataset, 0)

        for i, data in enumerate(data_loader):
            if i == self.max_loader_iter:
                break
            text_input = data[0]
            text_lengths = data[1]
            linear_input = data[2]
            mel_input = data[3]
            mel_lengths = data[4]
            stop_target = data[5]
            item_idx = data[6]

            if mel_lengths[0] > mel_lengths[1]:
                idx = 0
            else:
                idx = 1

            # check the first item in the batch
            assert linear_input[idx, -1].sum() == 0
            assert linear_input[idx, -2].sum() != 0, linear_input
            assert mel_input[idx, -1].sum() == 0
            assert mel_input[idx, -2].sum() != 0, mel_input
            assert stop_target[idx, -1] == 1
            assert stop_target[idx, -2] == 0
            assert stop_target[idx].sum() == 1
            assert len(mel_lengths.shape) == 1
            assert mel_lengths[idx] == mel_input[idx].shape[0]
            assert mel_lengths[idx] == linear_input[idx].shape[0]

            # check the second itme in the batch
            assert linear_input[1 - idx, -1].sum() == 0
            assert mel_input[1 - idx, -1].sum() == 0
            assert stop_target[1 - idx, -1] == 1
            assert len(mel_lengths.shape) == 1

            # check batch conditions
            assert (linear_input * stop_target.unsqueeze(2)).sum() == 0
            assert (mel_input * stop_target.unsqueeze(2)).sum() == 0
