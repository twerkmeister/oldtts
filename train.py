import os
import sys
import time
import shutil
import torch
import argparse
import importlib
import traceback
import numpy as np

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from distribute import init_distributed, apply_gradient_allreduce, reduce_tensor
from distribute import DistributedSampler

from utils.generic_utils import (
    remove_experiment_folder, create_experiment_folder, save_checkpoint,
    save_best_model, load_config, lr_decay, count_parameters, check_update,
    get_commit_hash, sequence_mask, AnnealLR, weight_decay)
from utils.visual import plot_alignment, plot_spectrogram
from models.tacotron import Tacotron
from layers.losses import L1LossMasked
from utils.audio import AudioProcessor
from utils.synthesis import synthesis

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(" > Using CUDA: ", use_cuda)
print(" > Number of GPUs: ", num_gpus)


def setup_loader(is_val=False):
    global ap
    if is_val and not c.run_eval:
        loader = None
    else:
        dataset = MyDataset(
            c.data_path,
            c.meta_file_val if is_val else c.meta_file_train,
            c.r,
            c.text_cleaner,
            preprocessor=preprocessor,
            ap=ap,
            batch_group_size=0 if is_val else 8 * c.batch_size,
            min_seq_len=0 if is_val else c.min_seq_len)
        sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=c.eval_batch_size if is_val else c.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            sampler=sampler,
            num_workers=c.num_val_loader_workers if is_val else c.num_loader_workers,
            pin_memory=False)
    return loader


def train(model, criterion, criterion_st, optimizer, optimizer_st, scheduler,
          ap, epoch):
    data_loader = setup_loader(is_val=False)
    model = model.train()
    epoch_time = 0
    avg_linear_loss = 0
    avg_mel_loss = 0
    avg_stop_loss = 0
    avg_step_time = 0
    print(" | > Epoch {}/{}".format(epoch, c.epochs), flush=True)
    n_priority_freq = int(
        3000 / (c.audio['sample_rate'] * 0.5) * c.audio['num_freq'])

    if num_gpus > 0:
        batch_n_iter = int(len(data_loader.dataset) / (c.batch_size * num_gpus))
    else:
        batch_n_iter = int(len(data_loader.dataset) / c.batch_size) 

    for num_iter, data in enumerate(data_loader):
        model.zero_grad()
        start_time = time.time()

        # setup input data
        text_input = data[0]
        text_lengths = data[1]
        linear_input = data[2]
        mel_input = data[3]
        mel_lengths = data[4]
        stop_targets = data[5]

        avg_text_len = torch.mean(text_lengths.float())
        avg_mel_len = torch.mean(mel_lengths.float())

        # set stop targets view, we predict a single stop token per r frames prediction
        stop_targets = stop_targets.view(text_input.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float()

        current_step = num_iter + args.restore_step + \
            epoch * len(data_loader) + 1

        # setup lr
        if c.lr_decay:
            scheduler.step()

        # dispatch data to GPU
        if use_cuda:
            text_input = text_input.cuda(non_blocking=True)
            text_lengths = text_lengths.cuda(non_blocking=True)
            mel_input = mel_input.cuda(non_blocking=True)
            mel_lengths = mel_lengths.cuda(non_blocking=True)
            linear_input = linear_input.cuda(non_blocking=True)
            stop_targets = stop_targets.cuda(non_blocking=True)

        # compute mask for padding
        mask = sequence_mask(text_lengths)

        # forward pass
        mel_output, linear_output, alignments, stop_tokens = model(
            text_input, mel_input, mask)

        # loss computation
        stop_loss = criterion_st(stop_tokens, stop_targets)
        mel_loss = criterion(mel_output, mel_input, mel_lengths)
        linear_loss = 0.5 * criterion(linear_output, linear_input, mel_lengths)\
            + 0.5 * criterion(linear_output[:, :, :n_priority_freq],
                              linear_input[:, :, :n_priority_freq],
                              mel_lengths)
        loss = mel_loss + linear_loss

        # backpass and check the grad norm for spec losses
        loss.backward(retain_graph=True)
        optimizer, current_lr = weight_decay(optimizer, c.wd)
        grad_norm, _ = check_update(model, 1)
        optimizer.step()

        # backpass and check the grad norm for stop loss
        stop_loss.backward()
        optimizer_st, _ = weight_decay(optimizer_st, c.wd)
        grad_norm_st, _ = check_update(model.decoder.stopnet, 0.5)
        optimizer_st.step()

        step_time = time.time() - start_time
        epoch_time += step_time

        # logging
        if current_step % c.print_step == 0:
            print(
                "   | > Step:{}/{}  GlobalStep:{}  TotalLoss:{:.5f}  LinearLoss:{:.5f}  "
                "MelLoss:{:.5f}  StopLoss:{:.5f}  GradNorm:{:.5f}  "
                "GradNormST:{:.5f}  StepTime:{:.2f}  AvgTextLen:{:.2f}  AvgSpecLen:{:.2f}  LR:{:.6f}".format(
                    num_iter, batch_n_iter, current_step, loss.item(),
                    linear_loss.item(), mel_loss.item(), stop_loss.item(),
                    grad_norm, grad_norm_st, step_time, avg_text_len, avg_mel_len, current_lr),
                flush=True)

        # aggregate losses from processes
        if num_gpus > 1:
            linear_loss = reduce_tensor(linear_loss.data, num_gpus)
            mel_loss = reduce_tensor(mel_loss.data, num_gpus)
            loss = reduce_tensor(loss.data, num_gpus)
            stop_loss = reduce_tensor(stop_loss.data, num_gpus)

        avg_linear_loss += float(linear_loss.item())
        avg_mel_loss += float(mel_loss.item())
        avg_stop_loss += float(stop_loss.item())
        avg_step_time += step_time

        if args.rank == 0:
            # Plot iter stats
            tb.add_scalar('TrainIterLoss/TotalLoss', loss.item(), current_step)
            tb.add_scalar('TrainIterLoss/LinearLoss', linear_loss.item(),
                          current_step)
            tb.add_scalar('TrainIterLoss/MelLoss', mel_loss.item(),
                          current_step)
            tb.add_scalar('Params/LearningRate',
                          optimizer.param_groups[0]['lr'], current_step)
            tb.add_scalar('Params/GradNorm', grad_norm, current_step)
            tb.add_scalar('Params/GradNormSt', grad_norm_st, current_step)
            tb.add_scalar('Time/StepTime', step_time, current_step)

            if current_step % c.save_step == 0 and args.rank == 0:
                if c.checkpoint:
                    # save model
                    save_checkpoint(model, optimizer, optimizer_st,
                                    linear_loss.item(), OUT_PATH, current_step,
                                    epoch)

                # Diagnostic visualization
                const_spec = linear_output[0].data.cpu().numpy()
                gt_spec = linear_input[0].data.cpu().numpy()

                const_spec = plot_spectrogram(const_spec, ap)
                gt_spec = plot_spectrogram(gt_spec, ap)
                tb.add_figure('Visual/Reconstruction', const_spec,
                              current_step)
                tb.add_figure('Visual/GroundTruth', gt_spec, current_step)

                align_img = alignments[0].data.cpu().numpy()
                align_img = plot_alignment(align_img)
                tb.add_figure('Visual/Alignment', align_img, current_step)

                # Synthesis audio
                audio_signal = linear_output[0].data.cpu().numpy()
                ap.griffin_lim_iters = 60
                audio_signal = ap.inv_spectrogram(audio_signal.T)
                try:
                    tb.add_audio(
                        'SampleAudio',
                        audio_signal,
                        current_step,
                        sample_rate=c.sample_rate)
                except:
                    traceback.print_exc()
                    pass

    avg_linear_loss /= (num_iter + 1)
    avg_mel_loss /= (num_iter + 1)
    avg_stop_loss /= (num_iter + 1)
    avg_total_loss = avg_mel_loss + avg_linear_loss + avg_stop_loss
    avg_step_time /= (num_iter + 1)

    # print epoch stats
    print(
        "   | > EPOCH END -- GlobalStep:{}  AvgTotalLoss:{:.5f}  "
        "AvgLinearLoss:{:.5f}  AvgMelLoss:{:.5f}  "
        "AvgStopLoss:{:.5f}  EpochTime:{:.2f}  "
        "AvgStepTime:{:.2f}".format(current_step, avg_total_loss,
                                    avg_linear_loss, avg_mel_loss,
                                    avg_stop_loss, epoch_time, avg_step_time),
        flush=True)

    # Plot Epoch Stats
    if args.rank == 0:
        tb.add_scalar('TrainEpochLoss/TotalLoss', avg_total_loss, current_step)
        tb.add_scalar('TrainEpochLoss/LinearLoss', avg_linear_loss,
                      current_step)
        tb.add_scalar('TrainEpochLoss/MelLoss', avg_mel_loss, current_step)
        tb.add_scalar('TrainEpochLoss/StopLoss', avg_stop_loss, current_step)
        tb.add_scalar('Time/EpochTime', epoch_time, epoch)
        epoch_time = 0
    return avg_linear_loss, current_step


def evaluate(model, criterion, criterion_st, ap, current_step):
    global c
    data_loader = setup_loader(is_val=True)
    model = model.eval()
    epoch_time = 0
    avg_linear_loss = 0
    avg_mel_loss = 0
    avg_stop_loss = 0
    print(" | > Validation")
    test_sentences = [
        "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
        "Be a voice, not an echo.",
        "I'm sorry Dave. I'm afraid I can't do that.",
        "This cake is great. It's so delicious and moist."
    ]
    n_priority_freq = int(
        3000 / (c.audio['sample_rate'] * 0.5) * c.audio['num_freq'])
    with torch.no_grad():
        if data_loader is not None:
            for num_iter, data in enumerate(data_loader):
                start_time = time.time()

                # setup input data
                text_input = data[0]
                text_lengths = data[1]
                linear_input = data[2]
                mel_input = data[3]
                mel_lengths = data[4]
                stop_targets = data[5]

                # set stop targets view, we predict a single stop token per r frames prediction
                stop_targets = stop_targets.view(text_input.shape[0],
                                                 stop_targets.size(1) // c.r,
                                                 -1)
                stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float()

                # dispatch data to GPU
                if use_cuda:
                    text_input = text_input.cuda(non_blocking=True)
                    mel_input = mel_input.cuda(non_blocking=True)
                    mel_lengths = mel_lengths.cuda(non_blocking=True)
                    linear_input = linear_input.cuda(non_blocking=True)
                    stop_targets = stop_targets.cuda(non_blocking=True)

                # forward pass
                mel_output, linear_output, alignments, stop_tokens =\
                    model.forward(text_input, mel_input)

                # loss computation
                stop_loss = criterion_st(stop_tokens, stop_targets)
                mel_loss = criterion(mel_output, mel_input, mel_lengths)
                linear_loss = 0.5 * criterion(linear_output, linear_input, mel_lengths) \
                    + 0.5 * criterion(linear_output[:, :, :n_priority_freq],
                                    linear_input[:, :, :n_priority_freq],
                                    mel_lengths)
                loss = mel_loss + linear_loss + stop_loss

                step_time = time.time() - start_time
                epoch_time += step_time

                if num_iter % c.print_step == 0:
                    print(
                        "   | > TotalLoss: {:.5f}   LinearLoss: {:.5f}   MelLoss:{:.5f}  "
                        "StopLoss: {:.5f}  ".format(loss.item(),
                                                    linear_loss.item(),
                                                    mel_loss.item(),
                                                    stop_loss.item()),
                        flush=True)

                # aggregate losses from processes
                if num_gpus > 1:
                    linear_loss = reduce_tensor(linear_loss.data, num_gpus)
                    mel_loss = reduce_tensor(mel_loss.data, num_gpus)
                    stop_loss = reduce_tensor(stop_loss.data, num_gpus)

                avg_linear_loss += linear_loss.item()
                avg_mel_loss += mel_loss.item()
                avg_stop_loss += stop_loss.item()

            if args.rank == 0:
                # Diagnostic visualizations
                idx = np.random.randint(mel_input.shape[0])
                const_spec = linear_output[idx].data.cpu().numpy()
                gt_spec = linear_input[idx].data.cpu().numpy()
                align_img = alignments[idx].data.cpu().numpy()

                const_spec = plot_spectrogram(const_spec, ap)
                gt_spec = plot_spectrogram(gt_spec, ap)
                align_img = plot_alignment(align_img)

                tb.add_figure('ValVisual/Reconstruction', const_spec,
                              current_step)
                tb.add_figure('ValVisual/GroundTruth', gt_spec, current_step)
                tb.add_figure('ValVisual/ValidationAlignment', align_img,
                              current_step)

                # Sample audio
                audio_signal = linear_output[idx].data.cpu().numpy()
                ap.griffin_lim_iters = 60
                audio_signal = ap.inv_spectrogram(audio_signal.T)
                try:
                    tb.add_audio(
                        'ValSampleAudio',
                        audio_signal,
                        current_step,
                        sample_rate=c.audio["sample_rate"])
                except:
                    traceback.print_exc()
                    # sometimes audio signal is out of boundaries
                    pass

                # compute average losses
                avg_linear_loss /= (num_iter + 1)
                avg_mel_loss /= (num_iter + 1)
                avg_stop_loss /= (num_iter + 1)
                avg_total_loss = avg_mel_loss + avg_linear_loss + avg_stop_loss

                # Plot Learning Stats
                tb.add_scalar('ValEpochLoss/TotalLoss', avg_total_loss,
                              current_step)
                tb.add_scalar('ValEpochLoss/LinearLoss', avg_linear_loss,
                              current_step)
                tb.add_scalar('ValEpochLoss/MelLoss', avg_mel_loss,
                              current_step)
                tb.add_scalar('ValEpochLoss/Stop_loss', avg_stop_loss,
                              current_step)

    # test sentences
    if c.run_test_synthesis and args.rank == 0:
        ap.griffin_lim_iters = 60
        for idx, test_sentence in enumerate(test_sentences):
            try:
                wav, alignment, linear_spec, _, stop_tokens = synthesis(
                    model, test_sentence, c, use_cuda, ap)

                file_path = os.path.join(AUDIO_PATH, str(current_step))
                os.makedirs(file_path, exist_ok=True)
                file_path = os.path.join(file_path,
                                         "TestSentence_{}.wav".format(idx))
                ap.save_wav(wav, file_path)

                wav_name = 'TestSentences/{}'.format(idx)
                tb.add_audio(
                    wav_name,
                    wav,
                    current_step,
                    sample_rate=c.audio['sample_rate'])
                linear_spec = plot_spectrogram(linear_spec, ap)
                align_img = plot_alignment(alignment)
                tb.add_figure('TestSentences/{}_Spectrogram'.format(idx),
                              linear_spec, current_step)
                tb.add_figure('TestSentences/{}_Alignment'.format(idx),
                              align_img, current_step)
            except:
                print(" !! Error as creating Test Sentence -", idx)
                traceback.print_exc()
                pass
    return avg_linear_loss


def main(args):
    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(args.rank, num_gpus, args.group_id,
                         c.distributed["backend"], c.distributed["url"])

    model = Tacotron(c.embedding_size, ap.num_freq, ap.num_mels, c.r)
    print(" | > Num output units : {}".format(ap.num_freq), flush=True)

    optimizer = optim.Adam(model.parameters(), lr=c.lr, weight_decay=0)
    optimizer_st = optim.Adam(
        model.decoder.stopnet.parameters(), lr=c.lr, weight_decay=0)

    criterion = L1LossMasked()
    criterion_st = nn.BCELoss()

    if len(args.restore_path) > 1:
        checkpoint = torch.load(args.restore_path)
        model.load_state_dict(checkpoint['model'])
        if use_cuda:
            model = model.cuda()
            criterion.cuda()
            criterion_st.cuda()
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(
            " > Model restored from step %d" % checkpoint['step'], flush=True)
        start_epoch = checkpoint['epoch'] 
        best_loss = checkpoint['linear_loss']
        args.restore_step = checkpoint['step']
    else:
        args.restore_step = 0
        print("\n > Starting a new training", flush=True)
        if use_cuda:
            model = model.cuda()
            criterion.cuda()
            criterion_st.cuda()

    # DISTRUBUTED
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    if c.lr_decay:
        scheduler = AnnealLR(
            optimizer,
            warmup_steps=c.warmup_steps,
            last_epoch=args.restore_step - 1)
    else:
        scheduler = None

    num_params = count_parameters(model)
    print(" | > Model has {} parameters".format(num_params), flush=True)

    if 'best_loss' not in locals():
        best_loss = float('inf')

    for epoch in range(0, c.epochs):
        train_loss, current_step = train(model, criterion, criterion_st,
                                         optimizer, optimizer_st,
                                         scheduler, ap, epoch)
        val_loss = evaluate(model, criterion, criterion_st, ap,
                            current_step)
        print(
            " | > Train Loss: {:.5f}   Validation Loss: {:.5f}".format(
                train_loss, val_loss),
            flush=True)
        best_loss = save_best_model(model, optimizer, train_loss, best_loss,
                                    OUT_PATH, current_step, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Folder path to checkpoints',
        default='')
    parser.add_argument(
        '--config_path',
        type=str,
        help='path to config file for training',
    )
    parser.add_argument(
        '--debug',
        type=bool,
        default=False,
        help='do not ask for git has before run.')
    parser.add_argument(
        '--data_path', type=str, help='dataset path.', default='')
    parser.add_argument(
        '--output_path',
        type=str,
        help='path for training outputs.',
        default='')

    # DISTRUBUTED
    parser.add_argument(
        '--rank',
        type=int,
        default=0,
        help='DISTRIBUTED: process rank for distributed training.')
    parser.add_argument(
        '--group_id',
        type=str,
        default="",
        help='DISTRIBUTED: process group id.')
    args = parser.parse_args()

    # setup output paths and read configs
    c = load_config(args.config_path)
    _ = os.path.dirname(os.path.realpath(__file__))

    if args.data_path != '':
        c.data_path = args.data_path

    if args.output_path == '':
        OUT_PATH = os.path.join(_, c.output_path)
    else:
        OUT_PATH = args.output_path

    if args.group_id == '':
        OUT_PATH = create_experiment_folder(OUT_PATH, c.model_name, args.debug)

    AUDIO_PATH = os.path.join(OUT_PATH, 'test_audios')

    if args.rank == 0:
        os.makedirs(AUDIO_PATH, exist_ok=True)
        shutil.copyfile(args.config_path, os.path.join(OUT_PATH,
                                                       'config.json'))
        os.chmod(AUDIO_PATH, 0o775)
        os.chmod(OUT_PATH, 0o775)

    LOG_DIR = OUT_PATH
    tb = SummaryWriter(LOG_DIR)

    # Conditional imports
    preprocessor = importlib.import_module('datasets.preprocess')
    preprocessor = getattr(preprocessor, c.dataset.lower())
    MyDataset = importlib.import_module('datasets.' + c.data_loader)
    MyDataset = getattr(MyDataset, "MyDataset")
    audio = importlib.import_module('utils.' + c.audio['audio_processor'])
    AudioProcessor = getattr(audio, 'AudioProcessor')

    # Audio processor
    ap = AudioProcessor(**c.audio)

    # try:
    main(args)
    # except KeyboardInterrupt:
    #     remove_experiment_folder(OUT_PATH)
    #     try:
    #         sys.exit(0)
    #     except SystemExit:
    #         os._exit(0)
    # except Exception:
    #     remove_experiment_folder(OUT_PATH)
    #     traceback.print_exc()
    #     sys.exit(1)
