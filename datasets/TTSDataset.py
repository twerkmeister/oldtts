import os
import numpy as np
from torch.utils.data import Dataset

from utils.text import text_to_sequence, phoneme_to_sequence


class TTSDataset(Dataset):
    def __init__(self,
                 root_path,
                 meta_file,
                 text_cleaner,
                 ap,
                 preprocessor,
                 use_phonemes=True,
                 phoneme_cache_path=None,
                 phoneme_language="en-us",
                 enable_eos_bos=False,
                 verbose=False):
        """
        Args:
            root_path (str): root path for the data folder.
            meta_file (str): name for dataset file including audio transcripts 
                and file names (or paths in cached mode).
            text_cleaner (str): text cleaner used for the dataset.
            ap (TTS.utils.AudioProcessor): audio processor object.
            preprocessor (dataset.preprocess.Class): preprocessor for the
            dataset.
                Create your own if you need to run a new dataset.
            use_phonemes (bool): (true) if true, text converted to phonemes.
            phoneme_cache_path (str): path to cache phoneme features. 
            phoneme_language (str): one the languages from 
                https://github.com/bootphon/phonemizer#languages
            enable_eos_bos (bool): enable end of sentence and beginning of
            sentences characters.
            verbose (bool): print diagnostic information.
        """
        self.root_path = root_path
        self.meta_file = meta_file
        self.items = preprocessor(root_path, meta_file)
        self.sample_rate = ap.sample_rate
        self.cleaners = text_cleaner
        self.ap = ap
        self.use_phonemes = use_phonemes
        self.phoneme_cache_path = phoneme_cache_path
        self.phoneme_language = phoneme_language
        self.enable_eos_bos = enable_eos_bos
        self._make_phn_cache_path()
        if verbose:
            self.print_stats()

    def _make_phn_cache_path(self):
        if self.use_phonemes and not os.path.isdir(self.phoneme_cache_path):
            os.makedirs(self.phoneme_cache_path, exist_ok=True)

    def print_stats(self):
        lengths = np.array([len(ins[0]) for ins in self.items])
        print("Dataset at {}, use phonemes: {}, lang: {}, num instances: {}, "
              "max text len: {}, min text len: {}, avg text len: {}".format(
                os.path.join(self.root_path, self.meta_file),
                self.use_phonemes,
                self.phoneme_language,
                len(self.items),
                np.max(lengths),
                np.min(lengths),
                np.average(lengths)))

    def load_wav(self, filename):
        audio = self.ap.load_wav(filename)
        return audio

    def load_np(self, filename):
        data = np.load(filename).astype('float32')
        return data

    def generate_phoneme_sequence(self, text):
        phonemes = phoneme_to_sequence(text, [self.cleaners],
                                       language=self.phoneme_language,
                                       enable_eos_bos=self.enable_eos_bos)
        phonemes = np.asarray(phonemes, dtype=np.int32)
        return phonemes

    def load_or_generate_phoneme_sequence(self, wav_file, text):
        file_name = os.path.basename(wav_file).split('.')[0]
        tmp_path = os.path.join(self.phoneme_cache_path,
                                file_name + '_phoneme.npy')
        try:
            phonemes = np.load(tmp_path)
        except FileNotFoundError:
            phonemes = self.generate_phoneme_sequence(text)
        except (ValueError, IOError):
            print(" > ERROR: failed loading phonemes for {}. "
                  "Recomputing.".format(wav_file))
            phonemes = self.generate_phoneme_sequence(text)

        np.save(tmp_path, phonemes)
        return phonemes

    def load_data(self, idx):
        text, wav_file = self.items[idx]
        wav = np.asarray(self.load_wav(wav_file), dtype=np.float32)
        mel = self.ap.melspectrogram(wav).astype('float32')
        linear = self.ap.spectrogram(wav).astype('float32')

        if self.use_phonemes:
            text = self.load_phoneme_sequence(wav_file, text)
        else:
            text = np.asarray(
                text_to_sequence(text, [self.cleaners]), dtype=np.int32)

        assert text.size > 0, self.items[idx][1]
        assert wav.size > 0, self.items[idx][1]

        sample = {
            'text': text,
            'mel': mel,
            'linear': linear,
            'item_idx': self.items[idx][1]
        }
        return sample

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.load_data(idx)