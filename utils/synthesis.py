import io
import time
import librosa
import torch
import numpy as np
from .text import text_to_sequence, phoneme_to_sequence, sequence_to_phoneme
from .visual import visualize
from matplotlib import pylab as plt


def synthesis(model, text, CONFIG, use_cuda, ap, token_scores, truncated=False,
              enable_eos_bos_chars=False, trim_silence=False):
    """Synthesize voice for the given text.

        Args:
            model (TTS.models): model to synthesize.
            text (str): target text
            CONFIG (dict): config dictionary to be loaded from config.json.
            use_cuda (bool): enable cuda.
            ap (TTS.utils.audio.AudioProcessor): audio processor to process
                model outputs.
            token_scores (np array): scores/weights for style tokens
            truncated (bool): keep model states after inference. It can be used
                for continuous inference at long texts.
            enable_eos_bos_chars (bool): enable special chars for end of
            sentence and start of sentence.
            trim_silence (bool): trim silence after synthesis.
    """
    # preprocess the given text
    text_cleaner = [CONFIG.text_cleaner]
    if CONFIG.use_phonemes:
        seq = np.asarray(
            phoneme_to_sequence(text, text_cleaner, CONFIG.phoneme_language,
                                enable_eos_bos_chars),
            dtype=np.int32)
    else:
        seq = np.asarray(text_to_sequence(text, text_cleaner), dtype=np.int32)
    chars_var = torch.from_numpy(seq).unsqueeze(0)
    token_scores_var = torch.from_numpy(token_scores).unsqueeze(0)
    # synthesize voice
    if use_cuda:
        chars_var = chars_var.cuda()
        token_scores_var = token_scores_var.cuda()
    if truncated:
        decoder_output, postnet_output, alignments, stop_tokens = \
            model.inference_truncated(
            chars_var.long())
    else:
        decoder_output, postnet_output, alignments, stop_tokens = \
            model.inference(chars_var.long(), token_scores_var.float())
    # convert outputs to numpy
    postnet_output = postnet_output[0].data.cpu().numpy()
    decoder_output = decoder_output[0].data.cpu().numpy()
    alignment = alignments[0].cpu().data.numpy()
    # plot results
    if CONFIG.model == "Tacotron":
        wav = ap.inv_mel_spectrogram(postnet_output.T)
    else:
        wav = ap.inv_mel_spectrogram(postnet_output.T)
    # trim silence
    if trim_silence:
        wav = wav[:ap.find_endpoint(wav)]
    return wav, alignment, decoder_output, postnet_output, stop_tokens


def get_token_scores(model, filename, ap, use_cuda):
    wav = ap.load_wav(filename)
    mel = ap.melspectrogram(wav).astype('float32')
    mel = torch.FloatTensor(mel).contiguous()
    if use_cuda:
        mel = mel.cuda()
    mel = mel.unsqueeze(0)
    return model.get_token_scores(mel)
