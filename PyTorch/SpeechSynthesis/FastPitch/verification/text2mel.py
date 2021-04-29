import json
import argparse
# import time
# import tqdm
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import models
from inference import load_and_setup_model, load_model_from_ckpt, parse_args
import data_functions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PitchNormalizer():
    def __init__(self, stats):
        with open(stats) as fp:
            stats = json.load(fp)
        self.mean = stats["mean"]
        self.std = stats["std"]
    def denormalize_voiced(self, utt_pitch):
        """
        param utt_pitch: token pitch sequence of an utterance
        """
        voiced = utt_pitch != 0
        voiced_pitch = self.std * utt_pitch[voiced] + self.mean
        utt_pitch[voiced] = voiced_pitch
        return utt_pitch


def strech_pitch(pitch, duration):
    """
    Stretch pitch countour according to durations
    
    :param pitch: (T,)
    :param duration: (T,) `int`
    """
    assert len(pitch) == len(duration)
    pitch_output = []
    for p, d in zip(pitch, duration):
        pitch_seg = p * np.ones(d)
        pitch_output.append(pitch_seg)
    pitch_output = np.concatenate(pitch_output)
    return np.asarray(pitch_output)


def batch_to_device(batch):
    batch_to_gpu = data_functions.get_batch_to_gpu('FastPitch')
    if torch.cuda.is_available():
        x, y, num_frames = batch_to_gpu(batch)
    else:
        text_padded, input_lengths, mel_padded, output_lengths, \
            len_x, dur_padded, dur_lens, pitch_padded, speaker = batch
        x = [text_padded, input_lengths, mel_padded, output_lengths,
            dur_padded, dur_lens, pitch_padded, speaker]
        y = [mel_padded, dur_padded, dur_lens, pitch_padded]
        num_frames = torch.sum(output_lengths)
    return x, y, num_frames


def load_model_from_ckpt(checkpoint_path, ema, model):
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    status = ''
    if 'state_dict' in checkpoint_data:
        sd = checkpoint_data['state_dict']
        if ema and 'ema_state_dict' in checkpoint_data:
            sd = checkpoint_data['ema_state_dict']
            status += ' (EMA)'
        elif ema and not 'ema_state_dict' in checkpoint_data:
            print(f'WARNING: EMA weights missing for {checkpoint_data}')
        if any(key.startswith('module.') for key in sd):
            sd = {k.replace('module.', ''): v for k,v in sd.items()}
        status += ' ' + str(model.load_state_dict(sd, strict=False))
    else:
        model = checkpoint_data['model']
    print(f'Loaded {checkpoint_path}{status}')
    return model

def to_numpy(torch_tensor):
    return torch_tensor.detach().to("cpu").numpy()

def plot_mels(mels, fname):
    mel_padded, mel_out_gt, mel_out = mels

    fig, axs = plt.subplots(3, 1, figsize=(4, 7))

    im = axs[0].imshow(np.flipud(mel_padded), aspect="auto")
    axs[0].title.set_text("Ground-truth")
    fig.colorbar(im, ax=axs[0])

    im = axs[1].imshow(np.flipud(mel_out_gt), aspect="auto")
    axs[1].title.set_text("Prediction from true dur/pitch")
    fig.colorbar(im, ax=axs[1])

    im = axs[2].imshow(np.flipud(mel_out), aspect="auto")
    axs[2].title.set_text("Prediction from text")
    fig.colorbar(im, ax=axs[2])

    fig.tight_layout()
    fig.savefig(fname)
    plt.close()


def plot_pitch(f0_gt, f0_pred, fname):
    plt.figure()
    plt.plot(f0_gt, "r", alpha=0.75)
    plt.plot(f0_pred, "b", alpha=0.75)
    plt.legend(["ground-truth", "prediction"])
    plt.ylim(0.39, 0.415)
    plt.xlim(0, len(f0_pred))
    plt.title("Pitch")
    plt.savefig(fname)
    plt.close()


def plot_duration(duration_gt, duration_pred, fname):
    index = np.arange(len(duration_gt))
    plt.figure()
    plt.bar(index, duration_gt, color="r", alpha=0.5)
    plt.bar(index, duration_pred, color="b", alpha=0.5)
    plt.xlim(-0.5, index[-1] + 0.5)
    plt.title("Duration")
    plt.legend(["ground-truth", "prediction"])
    plt.savefig(fname)
    plt.close()



checkpoint_path = "output/FastPitch_checkpoint_900.pt"
AUDIOPATHS_AND_TEXT = "filelists/ljs_mel_dur_pitch_text_test_filelist.txt"
BATCH_SIZE = 1

# Model =======================
checkpoint = torch.load(checkpoint_path, map_location=device)
# model_config = ckpt["config"]

forward_is_infer = False   # FIXME test
generator = models.get_model(
    model_name="FastPitch",
    model_config=checkpoint["config"],
    device=device,
    forward_is_infer=forward_is_infer,
    jitable=False)
generator = load_model_from_ckpt(checkpoint_path, False, generator)
generator.eval()
# =======================


# Dataloader ======================
kwargs = checkpoint["config"]
kwargs.update({
    "dataset_path": "FastPitch_LJSpeech-1.1",  # FIXME
    "symbol_set": "IPA",  # FIXME
    "text_cleaners": ["as_is"],  # FIXME
    "n_speakers": 1
})
collate_fn = data_functions.get_collate_function('FastPitch')
val_sampler = None
valset = data_functions.get_data_loader(
    'FastPitch',
    audiopaths_and_text=AUDIOPATHS_AND_TEXT,
    **kwargs)
val_loader = DataLoader(
    valset, 
    num_workers=0,
    shuffle=False,
    sampler=val_sampler,
    batch_size=BATCH_SIZE,  # FIXME 
    pin_memory=False,
    collate_fn=collate_fn)
# =-======================

batch = next(iter(val_loader))
x, y, num_frames = batch_to_device(batch)
mel_padded, duration_gt, pitch_padded = (x[2], x[4], x[6])

with torch.no_grad():
    y_pred = generator(x)  #, use_gt_durations=True)
    mel_out_gt, dec_mask, dur_pred_gt, log_dur_pred_gt, \
        pitch_pred_gt = y_pred
    y_pred = generator(x, use_gt_durations=False, use_gt_pitch=False)
    mel_out, dec_mask, duration_pred, log_dur_pred, \
        pitch_pred = y_pred


stat_file = "FastPitch_LJSpeech-1.1/pitch_char_stats__ljs_audio_text_val_filelist.json"
pitch_normalizer = PitchNormalizer(stat_file)

f0_gt = pitch_normalizer.denormalize_voiced(pitch_padded[0])
f0_pred = pitch_normalizer.denormalize_voiced(pitch_pred[0])

f0_gt = strech_pitch(f0_gt, duration_gt[0])
f0_pred = strech_pitch(f0_pred, duration_gt[0])

mels = (
    to_numpy(mel_padded[0]),
    to_numpy(mel_out_gt[0]).T,
    to_numpy(mel_out[0]).T
)

logdir = Path(checkpoint_path).parent

plot_mels(mels, logdir / f"mels.png")
plot_pitch(f0_gt, f0_pred, logdir / f"pitch.png")
plot_duration(duration_gt[0], duration_pred[0], logdir / f"duration.png")
