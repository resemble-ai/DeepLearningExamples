import numpy as np
from scipy.stats import norm, gamma

if __name__ == "__main__":
    n_token = 10
    dur = gamma.rvs(2, scale=2, size=n_token).round()
    precision = gamma.rvs(1, scale=2, size=n_token)
    # FIXME: `s` is actually learnable per token
    s = np.sqrt(1/ precision)
    len_output = int(dur.cumsum()[-1])
    center =  dur / 2 
    cumsum = dur.cumsum()
    cumsum[1:] = cumsum[:-1]
    cumsum[0] = 0
    center += cumsum

    I = np.arange(n_token)
    T = np.arange(len_output)
    alignment = np.zeros([n_token, len_output])
    for i in I:
        for t in T:
            alignment[i, t] = norm.pdf(t, loc=center[i], scale=s[i])

    attention = alignment / alignment.sum(0, keepdims=True)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1)
    for ax, img, title in zip(
        axs, [alignment, attention],
        ["Gaussian PDF", "Weight $w_{ti}$"]
    ):
        ax.imshow(img, aspect="auto")
        ax.set_ylabel("token index")
        if "Gaussian" in title:
            ax.axis("off")
        ax.title.set_text(title)

    axs[-1].plot(dur)
    axs[-1].plot(s)
    axs[-1].set_xlabel("token index")
    axs[-1].legend(["duration", "std"])

    ax.set_xlabel("frame/mel index")

    fig.tight_layout()
    plt.savefig("gaussian_upsampling.png")
    plt.close()
