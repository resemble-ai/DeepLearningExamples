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
        fp_mel = torch.load(FP_ROOT / f"{stem}.pt")
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

# After I killed the zombies:
# 1. the resources (RAM, GPU RAMs) are released!
# 2. data iteration is still slow:  | 500/13100 [01:03<5:53:30,  1.68s/it]