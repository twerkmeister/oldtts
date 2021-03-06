import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim

from datasets.loading import setup
from distribute import (apply_gradient_allreduce,
                        init_distributed, reduce_tensor)
from layers.losses import L1LossMasked, MSELossMasked
from utils.audio import AudioProcessor
from utils.generic_utils import (NoamLR, check_update, count_parameters,
                                 create_experiment_folder, get_git_branch,
                                 load_config, lr_decay,
                                 remove_experiment_folder, save_best_model,
                                 save_checkpoint, sequence_mask, weight_decay,
                                 set_init_dict, copy_config_file, setup_model,
                                 get_max_speaker_id)
from utils.logger import Logger
from utils.speakers import load_speaker_mapping
from utils.synthesis import synthesis
from utils.text.symbols import phonemes, symbols
from utils.visual import plot_alignment, plot_spectrogram, plot_like_spectrogram

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(" > Using CUDA: ", use_cuda)
print(" > Number of GPUs: ", num_gpus)


def train(model, criterion, criterion_st, optimizer, optimizer_st, scheduler,
          ap, epoch, c):
    is_first_epoch = epoch == 0
    data_loader = setup(c, ap, num_gpus, verbose=is_first_epoch)
    model.train()
    epoch_time = 0
    avg_postnet_loss = 0
    avg_decoder_loss = 0
    avg_stop_loss = 0
    avg_step_time = 0
    avg_token_loss = 0
    print("\n > Epoch {}/{}".format(epoch, c.epochs), flush=True)
    batch_n_iter = int(len(data_loader.dataset) / (c.batch_size * max(1, num_gpus)))
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # setup input data
        text_input = data[0]
        text_lengths = data[1]
        mel_input = data[2]
        mel_lengths = data[3]
        stop_targets = data[4]
        speaker_ids = data[5]
        avg_text_length = torch.mean(text_lengths.float())
        avg_spec_length = torch.mean(mel_lengths.float())

        # set stop targets view, we predict a single stop token per r frames prediction
        stop_targets = stop_targets.view(text_input.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze(2)

        current_step = num_iter + args.restore_step + \
            epoch * len(data_loader) + 1

        # setup lr
        if c.lr_decay:
            scheduler.step()
        optimizer.zero_grad()
        if optimizer_st: optimizer_st.zero_grad();

        # dispatch data to GPU
        if use_cuda:
            text_input = text_input.cuda(non_blocking=True)
            text_lengths = text_lengths.cuda(non_blocking=True)
            mel_input = mel_input.cuda(non_blocking=True)
            mel_lengths = mel_lengths.cuda(non_blocking=True)
            stop_targets = stop_targets.cuda(non_blocking=True)
            speaker_ids = speaker_ids.cuda(non_blocking=True)

        # forward pass model
        decoder_output, postnet_output, alignments, \
        stop_tokens, token_scores = model(text_input, text_lengths,
                                          mel_input, speaker_ids)

        # loss computation
        if c.stopnet:
            stop_loss = c.stop_loss_adjustment * \
                        criterion_st(stop_tokens, stop_targets)
        else:
            stop_loss = torch.zeros(1)

        if c.loss_masking:
            decoder_loss = criterion(decoder_output, mel_input, mel_lengths)
            if c.model == "Tacotron":
                postnet_loss = criterion(postnet_output, mel_input, mel_lengths)
            else:
                postnet_loss = criterion(postnet_output, mel_input, mel_lengths)
        else:
            decoder_loss = criterion(decoder_output, mel_input)
            if c.model == "Tacotron":
                postnet_loss = criterion(postnet_output, mel_input)
            else:
                postnet_loss = criterion(postnet_output, mel_input)

        # style_token_loss = 1e-5 * model.global_style_tokens.style_token_layer.style_tokens.norm(1)
        style_token_loss = c.token_score_reg * token_scores.norm(1)
        loss = decoder_loss + postnet_loss + style_token_loss
        if not c.separate_stopnet and c.stopnet:
            loss += stop_loss

        # backpass and check the grad norm for spec losses
        if c.separate_stopnet:
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        optimizer, current_lr = weight_decay(optimizer, c.wd)
        grad_norm, _ = check_update(model, c.grad_clip)
        optimizer.step()

        # backpass and check the grad norm for stop loss
        if c.separate_stopnet:
            stop_loss.backward()
            optimizer_st, _ = weight_decay(optimizer_st, c.wd)
            grad_norm_st, _ = check_update(model.decoder.stopnet, 1.0)
            optimizer_st.step()
        else:
            grad_norm_st = 0

        step_time = time.time() - start_time
        epoch_time += step_time

        if current_step % c.print_step == 0:
            print(
                "   | > Step:{}/{}  GlobalStep:{}  TotalLoss:{:.5f}  PostnetLoss:{:.5f}  "
                "DecoderLoss:{:.5f}  StopLoss:{:.5f} TokenLoss:{:.5f} GradNorm:{:.5f}  "
                "GradNormST:{:.5f}  AvgTextLen:{:.1f}  AvgSpecLen:{:.1f}  StepTime:{:.2f}  LR:{:.6f}".format(
                    num_iter, batch_n_iter, current_step, loss.item(),
                    postnet_loss.item(), decoder_loss.item(), stop_loss.item(), style_token_loss.item(),
                    grad_norm, grad_norm_st, avg_text_length, avg_spec_length, step_time, current_lr),
                flush=True)

        # aggregate losses from processes
        if num_gpus > 1:
            postnet_loss = reduce_tensor(postnet_loss.data, num_gpus)
            decoder_loss = reduce_tensor(decoder_loss.data, num_gpus)
            loss = reduce_tensor(loss.data, num_gpus)
            stop_loss = reduce_tensor(stop_loss.data, num_gpus) if c.stopnet else stop_loss

        if args.rank == 0:
            avg_postnet_loss += float(postnet_loss.item())
            avg_decoder_loss += float(decoder_loss.item())
            avg_token_loss += style_token_loss.item()
            avg_stop_loss += stop_loss if type(stop_loss) is float else float(stop_loss.item())
            avg_step_time += step_time

            # Plot Training Iter Stats
            iter_stats = {"loss_posnet": postnet_loss.item(),
                        "loss_decoder": decoder_loss.item(),
                        "token_loss": style_token_loss.item(),
                        "lr": current_lr,
                        "grad_norm": grad_norm,
                        "grad_norm_st": grad_norm_st,
                        "step_time": step_time}
            tb_logger.tb_train_iter_stats(current_step, iter_stats)

            if current_step % c.save_step == 0:
                if c.checkpoint:
                    # save model
                    stop_optimizer = optimizer_st if c.separate_stopnet else None
                    save_checkpoint(model, optimizer, stop_optimizer,
                                    postnet_loss.item(), OUT_PATH, current_step,
                                    epoch)

                # Diagnostic visualizations
                decoder_spec = decoder_output[0].data.cpu().numpy()
                const_spec = postnet_output[0].data.cpu().numpy()
                gt_spec = mel_input[0].data.cpu().numpy()
                align_img = alignments[0].data.cpu().numpy()
                loss_spec = np.abs(gt_spec - const_spec)
                loss_spec_sqr = np.square(loss_spec)

                figures = {
                    "prediction_decoder": plot_spectrogram(decoder_spec, ap),
                    "prediction": plot_spectrogram(const_spec, ap),
                    "ground_truth": plot_spectrogram(gt_spec, ap),
                    "alignment": plot_alignment(align_img),
                    "loss_spec": plot_like_spectrogram(loss_spec),
                    "loss_spec_sqr": plot_like_spectrogram(loss_spec_sqr)
                }
                tb_logger.tb_train_figures(current_step, figures)

                sample_training_audios = data_loader.dataset.load_random_samples(2)
                sample_training_audios = {name: audio for name, audio
                                          in sample_training_audios}
                # Sample audio
                postnet_audio = ap.inv_mel_spectrogram(const_spec.T)
                decoder_audio = ap.inv_mel_spectrogram(decoder_spec.T)
                tb_logger.tb_train_audios(current_step,
                                          {'postnet_audio': postnet_audio,
                                           'decoder_audio': decoder_audio,
                                           **sample_training_audios},
                                          c.audio["sample_rate"])

    avg_postnet_loss /= (num_iter + 1)
    avg_decoder_loss /= (num_iter + 1)
    avg_stop_loss /= (num_iter + 1)
    avg_token_loss /= (num_iter + 1)
    avg_total_loss = avg_decoder_loss + avg_postnet_loss + avg_stop_loss \
                     + avg_token_loss
    avg_step_time /= (num_iter + 1)

    # print epoch stats
    print(
        "   | > EPOCH END -- GlobalStep:{}  AvgTotalLoss:{:.5f}  "
        "AvgPostnetLoss:{:.5f}  AvgDecoderLoss:{:.5f}  "
        "AvgStopLoss:{:.5f} AvgTokenLoss:{:.5f} EpochTime:{:.2f}  "
        "AvgStepTime:{:.2f}".format(current_step, avg_total_loss,
                                    avg_postnet_loss, avg_decoder_loss,
                                    avg_stop_loss, avg_token_loss, epoch_time,
                                    avg_step_time),
        flush=True)

    # Plot Epoch Stats
    if args.rank == 0:
        # Plot Training Epoch Stats
        epoch_stats = {"loss_postnet": avg_postnet_loss,
                    "loss_decoder": avg_decoder_loss,
                    "token_loss": avg_token_loss,
                    "stop_loss": avg_stop_loss,
                    "epoch_time": epoch_time}
        tb_logger.tb_train_epoch_stats(current_step, epoch_stats)
        if c.tb_model_param_stats:
            tb_logger.tb_model_weights(model, current_step) 
    return avg_postnet_loss, current_step


def evaluate(model, criterion, criterion_st, ap, current_step, epoch, c):
    """Evaluate the model based on validation set."""
    data_loader = setup(c, ap, num_gpus, is_val=True)
    model.eval()
    epoch_time = 0
    avg_postnet_loss = 0
    avg_decoder_loss = 0
    avg_stop_loss = 0
    avg_token_loss = 0
    print("\n > Validation")

    with torch.no_grad():
        for num_iter, data in enumerate(data_loader):
            start_time = time.time()

            # setup input data
            text_input = data[0]
            text_lengths = data[1]
            # linear_input = data[2] if c.model == "Tacotron" else None
            mel_input = data[2]
            mel_lengths = data[3]
            stop_targets = data[4]
            speaker_ids = data[5]

            # set stop targets view, we predict a single stop token per r frames prediction
            stop_targets = stop_targets.view(text_input.shape[0],
                                             stop_targets.size(1) // c.r,
                                             -1)
            stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze(2)

            # dispatch data to GPU
            if use_cuda:
                text_input = text_input.cuda()
                mel_input = mel_input.cuda()
                mel_lengths = mel_lengths.cuda()
                # linear_input = linear_input.cuda() if c.model == "Tacotron" else None
                stop_targets = stop_targets.cuda()
                speaker_ids = speaker_ids.cuda()

            # forward pass
            decoder_output, postnet_output, alignments, \
            stop_tokens, token_scores = model.forward(text_input,
                                                      text_lengths,
                                                      mel_input,
                                                      speaker_ids)

            # loss computation
            if c.stopnet:
                stop_loss = c.stop_loss_adjustment * \
                            criterion_st(stop_tokens, stop_targets)
            else:
                stop_loss = torch.zeros(1)
            if c.loss_masking:
                decoder_loss = criterion(decoder_output, mel_input, mel_lengths)
                if c.model == "Tacotron":
                    postnet_loss = criterion(postnet_output, mel_input, mel_lengths)
                else:
                    postnet_loss = criterion(postnet_output, mel_input, mel_lengths)
            else:
                decoder_loss = criterion(decoder_output, mel_input)
                if c.model == "Tacotron":
                    postnet_loss = criterion(postnet_output, mel_input)
                else:
                    postnet_loss = criterion(postnet_output, mel_input)
            style_token_loss = c.token_score_reg * token_scores.norm(1)
            loss = decoder_loss + postnet_loss + \
                   stop_loss + style_token_loss

            step_time = time.time() - start_time
            epoch_time += step_time

            if num_iter % c.print_step == 0:
                print(
                    "   | > TotalLoss: {:.5f}   PostnetLoss: {:.5f}   DecoderLoss:{:.5f}  "
                    "StopLoss: {:.5f}  ".format(loss.item(),
                                                postnet_loss.item(),
                                                decoder_loss.item(),
                                                stop_loss.item()),
                    flush=True)

            # aggregate losses from processes
            if num_gpus > 1:
                postnet_loss = reduce_tensor(postnet_loss.data, num_gpus)
                decoder_loss = reduce_tensor(decoder_loss.data, num_gpus)
                if c.stopnet:
                    stop_loss = reduce_tensor(stop_loss.data, num_gpus)

            avg_postnet_loss += float(postnet_loss.item())
            avg_decoder_loss += float(decoder_loss.item())
            avg_stop_loss += stop_loss.item()
            avg_token_loss += float(style_token_loss.item())

        if args.rank == 0:
            # Diagnostic visualizations
            idx = np.random.randint(mel_input.shape[0])
            postnet_spec = postnet_output[idx].data.cpu().numpy()
            decoder_spec = decoder_output[idx].data.cpu().numpy()

            gt_spec = mel_input[idx].data.cpu().numpy()
            align_img = alignments[idx].data.cpu().numpy()

            loss_spec = np.abs(gt_spec - postnet_spec)
            loss_spec_sqr = np.square(loss_spec)

            eval_figures = {
                "decoder_spec": plot_spectrogram(decoder_spec, ap),
                "post_net": plot_spectrogram(postnet_spec, ap),
                "ground_truth": plot_spectrogram(gt_spec, ap),
                "alignment": plot_alignment(align_img),
                "loss_spec": plot_like_spectrogram(loss_spec),
                "loss_spec_sqr": plot_like_spectrogram(loss_spec_sqr)
            }
            tb_logger.tb_eval_figures(current_step, eval_figures)

            # Sample audio
            eval_audio = ap.inv_mel_spectrogram(postnet_spec.T)
            eval_decoder_audio = ap.inv_mel_spectrogram(decoder_spec.T)
            tb_logger.tb_eval_audios(current_step,
                                     {"ValAudio": eval_audio,
                                      "ValDecAudio": eval_decoder_audio},
                                     c.audio["sample_rate"])

            # compute average losses
            avg_postnet_loss /= (num_iter + 1)
            avg_decoder_loss /= (num_iter + 1)
            avg_stop_loss /= (num_iter + 1)
            avg_token_loss /= (num_iter + 1)

            # Plot Validation Stats
            epoch_stats = {"loss_postnet": avg_postnet_loss,
                        "loss_decoder": avg_decoder_loss,
                        "stop_loss": avg_stop_loss,
                        "token_loss": avg_token_loss}
            tb_logger.tb_eval_stats(current_step, epoch_stats)
        return avg_postnet_loss


def test(model, ap, current_step, epoch, c):
    test_sentences = [
        "Die Erfolge der Grünen bringen eine Reihe "
        "Unerfahrener in die Parlamente.",
        "Andrea Nahles will in der Fraktion die Vertrauensfrage stellen.",
        "Die Luftfahrtbranche arbeite daran, CO2-neutral zu werden",
        "Michael Kretschmer versucht seit Monaten, die Bürger zu umgarnen.",
        "Nun ist der Spieltempel pleite, und manchen Dorfbewohnern "
        "fehlt das Geld zum Essen."
    ]

    if args.rank == 0 and epoch > c.test_delay_epochs:
        # test sentences
        test_audios = {}
        test_figures = {}
        print(" | > Synthesizing test sentences")
        for idx, test_sentence in enumerate(test_sentences):
            try:
                token_scores = np.random.normal(0, 0.3, c.num_style_tokens)
                speaker_id = np.random.randint(0, get_max_speaker_id(c) + 1, 1)[0]
                wav, alignment, decoder_output, postnet_output, stop_tokens = \
                    synthesis(model, test_sentence, c, use_cuda,
                              ap, token_scores, speaker_id, "de")
                file_path = os.path.join(AUDIO_PATH, str(current_step))
                os.makedirs(file_path, exist_ok=True)
                file_path = os.path.join(file_path,
                                        "TestSentence_{}.wav".format(idx))
                ap.save_wav(wav, file_path)
                test_audios['{}-audio'.format(idx)] = wav
                test_figures['{}-prediction'.format(idx)] = plot_spectrogram(postnet_output, ap)
                test_figures['{}-alignment'.format(idx)] = plot_alignment(alignment)
            except:
                print(" !! Error creating Test Sentence -", idx)
                traceback.print_exc()
        tb_logger.tb_test_audios(current_step, test_audios, c.audio['sample_rate'])
        tb_logger.tb_test_figures(current_step, test_figures)


def main(args, c):
    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(args.rank, num_gpus, args.group_id,
                         c.distributed["backend"], c.distributed["url"])
    model = setup_model(c)

    # Audio processor
    ap = AudioProcessor(**c.audio)

    print(" | > Num output units : {}".format(ap.num_freq), flush=True)

    optimizer = optim.Adam(model.parameters(), lr=c.lr, weight_decay=0)
    if c.stopnet and c.separate_stopnet:
        optimizer_st = optim.Adam(
            model.decoder.stopnet.parameters(), lr=c.lr, weight_decay=0)
    else:
        optimizer_st = None

    if c.loss_masking:
        if c.loss == "l1":
            criterion = L1LossMasked()
        else:
            criterion = MSELossMasked()
    else:
        if c.loss == "l1":
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
    criterion_st = nn.BCEWithLogitsLoss() if c.stopnet else None

    if args.restore_path:
        checkpoint = torch.load(args.restore_path)
        try:
            # TODO: fix optimizer init, model.cuda() needs to be called before
            # optimizer restore
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if len(c.reinit_layers) > 0:
                raise RuntimeError
            model.load_state_dict(checkpoint['model'])
        except:
            print(" > Partial model initialization.")
            partial_init_flag = True
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint, c)
            model.load_state_dict(model_dict)
            del model_dict
        for group in optimizer.param_groups:
            group['lr'] = c.lr
        print(
            " > Model restored from step %d" % checkpoint['step'], flush=True)
        start_epoch = checkpoint['epoch']
        args.restore_step = checkpoint['step']
    else:
        args.restore_step = 0

    if use_cuda:
        model = model.cuda()
        criterion.cuda()
        if criterion_st: criterion_st.cuda();

    # DISTRUBUTED
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    if c.lr_decay:
        scheduler = NoamLR(
            optimizer,
            warmup_steps=c.warmup_steps,
            last_epoch=args.restore_step - 1)
    else:
        scheduler = None

    num_params = count_parameters(model)
    print("\n > Model has {} parameters".format(num_params), flush=True)

    if 'best_loss' not in locals():
        best_loss = float('inf')

    for epoch in range(0, c.epochs):
        train_loss, current_step = train(model, criterion, criterion_st,
                                         optimizer, optimizer_st, scheduler,
                                         ap, epoch, c)
        print(" | > Training Loss: {:.5f}".format(train_loss), flush=True)
        target_loss = train_loss

        if c.run_eval:
            val_loss = evaluate(model, criterion, criterion_st,
                                ap, current_step, epoch, c)
            print(" | > Validation Loss: {:.5f}".format(val_loss), flush=True)
            target_loss = val_loss

            test(model, ap, current_step, epoch, c)

        best_loss = save_best_model(model, optimizer, target_loss, best_loss,
                                    OUT_PATH, current_step, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Path to model outputs (checkpoint, tensorboard etc.).',
        default=0)
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training.',
    )
    parser.add_argument(
        '--debug',
        type=bool,
        default=True,
        help='Do not verify commit integrity to run training.')
    parser.add_argument(
        '--data_path',
        type=str,
        default='',
        help='Defines the data path. It overwrites config.json.')
    parser.add_argument(
        '--output_path',
        type=str,
        help='path for training outputs.',
        default='')
    parser.add_argument(
        '--output_folder',
        type=str,
        default='',
        help='folder name for traning outputs.'
    )
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
    base_dir = os.path.dirname(os.path.realpath(__file__))
    if args.data_path != '':
        c.data_path = args.data_path

    if args.output_path == '':
        OUT_PATH = os.path.join(base_dir, c.output_path)
    else:
        OUT_PATH = args.output_path

    if args.group_id == '' and args.output_folder == '':
        OUT_PATH = create_experiment_folder(OUT_PATH, c.run_name, args.debug)
    else:
        OUT_PATH = os.path.join(OUT_PATH, args.output_folder)

    AUDIO_PATH = os.path.join(OUT_PATH, 'test_audios')

    if args.rank == 0:
        os.makedirs(AUDIO_PATH, exist_ok=True)
        new_fields = {}
        if args.restore_path:
            new_fields["restore_path"] = args.restore_path
        new_fields["github_branch"] = get_git_branch()
        copy_config_file(args.config_path,
                         os.path.join(OUT_PATH, 'config.json'), new_fields)
        os.chmod(AUDIO_PATH, 0o775)
        os.chmod(OUT_PATH, 0o775)
        LOG_DIR = OUT_PATH
        tb_logger = Logger(LOG_DIR)

    try:
        main(args, c)
    except KeyboardInterrupt:
        remove_experiment_folder(OUT_PATH)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception:
        remove_experiment_folder(OUT_PATH)
        traceback.print_exc()
        sys.exit(1)
