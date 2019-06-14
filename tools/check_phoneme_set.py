from collections import Counter
from tqdm import tqdm
from utils.text import text2phone, _clean_text, _phonemes_to_id


def main(text_line_file, language):
    keep = []
    drop = []
    with open(text_line_file) as f:
        for line in tqdm(f):
            line = line.strip()
            text = line.split("|")[2]
            text = text.replace(":", "")
            clean_text = _clean_text(text, ["phoneme_cleaners"])
            phonemes = text2phone(clean_text, language)
            for phonemes in filter(None, phonemes.split('|')):
                for p in phonemes:
                    if p in _phonemes_to_id and p is not '_' and p is not '~':
                        keep.append(p)
                    else:
                        drop.append(p)
    print(Counter(keep))
    print(Counter(drop))


if __name__ == "__main__":
    import plac
    plac.call(main)
