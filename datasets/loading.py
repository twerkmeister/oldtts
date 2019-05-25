import importlib

from .CombinedTTSDataset import CombinedTTSDataset
from .TTSDataset import TTSDataset
from .collate import make_collate_func
from torch.utils.data import DataLoader
from distribute import DistributedSampler


def preprocessor_factory(dataset_type):
    """Returns the preprocessor function for the given dataset type name."""
    preprocessor_module = importlib.import_module('.preprocess')
    return getattr(preprocessor_module, dataset_type.lower())


def _setup_dataset(ap, conf, dataset_confs, is_val=False, verbose=False):
    """Sets up the different datasets and combines them."""
    single_datasets = []
    for ds in dataset_confs:
        single_datasets.append(
            TTSDataset(
                ds['data_path'],
                ds['meta_file'],
                ds['text_cleaner'],
                preprocessor=preprocessor_factory(ds['type']),
                ap=ap,
                phoneme_cache_path=ds.get('phoneme_cache_path', None),
                use_phonemes=conf.use_phonemes,
                phoneme_language=ds.get('phoneme_language', "en-us"),
                enable_eos_bos=conf.enable_eos_bos_chars,
                verbose=verbose,
            )
        )
    batch_group_size = 0 if is_val else conf.batch_group_size * conf.batch_size
    combined_dataset = CombinedTTSDataset(single_datasets,
                                          batch_group_size,
                                          conf.min_seq_len,
                                          conf.max_seq_len)

    return combined_dataset


def _setup_loader(conf, dataset, num_gpus, is_val=False):
    sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    batch_size = conf.eval_batch_size if is_val else conf.batch_size
    num_workers = conf.num_val_loader_workers if is_val \
        else conf.num_loader_workers
    collate_fn = make_collate_func(conf.r)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False)
    return loader


def setup(conf, ap, num_gpus, is_val=False, verbose=False):
    dataset_confs = conf["eval_datasets"] if is_val else conf["train_datasets"]
    dataset = _setup_dataset(ap, conf, dataset_confs, is_val, verbose)
    loader = _setup_loader(conf, dataset, num_gpus, is_val)
    return loader
