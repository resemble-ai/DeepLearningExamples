from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch


if __name__ == "__main__":
    FP_ROOT = Path("LJSpeech-1.1/mels")
    TACO_ROOT = Path("Tacotron_LJSpeech-1.1/mels")  # LJ001-0001.npy

    files = list(TACO_ROOT.glob("*.npy"))
    for fname in tqdm(files):
        stem = fname.stem
        taco_mel = np.load(fname)
        taco_f0 = np.load(fname.parent / "f0_frames" / f"{stem}.npy")
        fp_mel = torch.load(FP_ROOT / f"{stem}.pt")

        # # # The shapes of f0 are different:
        # # - Resembletron: [T, 2], values are normalized somehow.
        # # - FastPitch: mean pitch of characters. 
        # #     shape=[T'], values seem to be Z-normalized. 
        # #     (unvoiced chars have F0 = 0.0)  

        # taco_f0 = np.load(fname.parent.parent / "f0_frames" / f"{stem}.npy")
        # fp_f0 = torch.load(FP_ROOT.parent / "pitch_char" / f"{stem}.pt" )
        # dur = torch.load(FP_ROOT.parent / "durations" / f"{stem}.pt")
        assert taco_mel.shape[1] == fp_mel.shape[1]

    print(f"All {len(files)} (resemble, fastpitch) mel pairs have the same lengths.")


# # import matplotlib.pyplot as plt

# fig, axs = plt.subplots(2, 1)
# for ax, title, mel in zip(
#     axs, 
#     ["Resembletron", "FastPitch"],
#     [taco_mel, fp_mel]
# ):
#     ax.title.set_text(title)
#     im = ax.imshow(mel, aspect="auto")
#     fig.colorbar(im, ax=ax)

# fig.tight_layout()
# fig.savefig("test.png")
# plt.close()


# ====
# # The range is different: Taco=[0, 1], FP=[<-10, >0]
# # fp_mel = taco_mel * v +u
# T = taco_mel.reshape(-1)
# T = np.stack([T, np.ones_like(T)], -1)
# y = fp_mel.reshape(-1).numpy()
# v, u = np.linalg.solve(T.T @ T, T.T @ y)
# ====


# # Rescaled Mel
# fig, ax = plt.subplots(2, 1)
# ax[0].title.set_text("Resembletron")
# im = ax[0].imshow(taco_mel, aspect="auto")
# fig.colorbar(im, ax=ax[0])

# ax[1].title.set_text("FastPitch (rescaled)")
# im = ax[1].imshow((fp_mel -u) / v, aspect="auto")
# fig.colorbar(im, ax=ax[1])

# fig.tight_layout()
# fig.savefig("test-rescaled.png")
# plt.close()

# Conclusion: 
#   1. it appears that FastPitch did not do pre-emphasis 
#      and used a different rescaling scheme (min power threshold).
#   2. We can just ignore the mels extracted by FastPitch anyway 
#      because the number of frames is the same.
#      Just use Resembletron's mel and the (duration, pitch) from FastPitch. 
#   3. Duration is a problem because input to FP is char...

# After I killed the zombies:
# 1. the resources (RAM, GPU RAMs) are released!
# 2. data iteration in my FastPitch docker is still slow: 500/13100 [01:03<5:53:30,  1.68s/it]

# OK, the problem is that previous experiments were executed on HDD.
# On the main disk (SSD), the speed is like:
# (docker): 13100/13100 [00:07<00:00, 1768.47it/s]

# However, training on SSD is only as fast as the training on HDD
# 31728.98 frames/s | took 0.54 s
