import os
import os.path as osp
import errno
import matplotlib.pyplot as plt
import numpy as np


def mkdir_p(dirname):
    """Like "mkdir -p", make a dir recursively, but do nothing if the dir
    exists.

    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == "" or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def grid_show(ims, titles=None, row=1, col=3, dpi=200, save_path=None, title_fontsize=5, show=False):
    if row * col < len(ims):
        print("_____________row*col < len(ims)___________")
        col = int(np.ceil(len(ims) / row))
    if titles is not None:
        assert len(ims) == len(titles), "{} != {}".format(len(ims), len(titles))
    fig = plt.figure(dpi=dpi, figsize=plt.figaspect(row / float(col)))
    k = 0
    for i in range(row):
        for j in range(col):
            if k >= len(ims):
                break
            plt.subplot(row, col, k + 1)
            plt.axis("off")
            plt.imshow(ims[k].astype('uint8'))
            if ims[k].ndim == 2:
                plt.colorbar()

            if titles is not None:
                plt.text(
                    0.5,
                    1.08,
                    titles[k],
                    horizontalalignment="center",
                    fontsize=title_fontsize,
                    transform=plt.gca().transAxes,
                )
            k += 1

    # plt.tight_layout()
    if show:
        plt.show()
    else:
        if save_path is not None:
            mkdir_p(osp.dirname(save_path))
            plt.savefig(save_path)
    plt.close()

