# Vendored from so-vits-svc 4.1-Stable vdecoder/hifigan/utils.py (@ 730930d),
# trimmed to what models.py imports (init_weights / get_padding) plus
# apply_weight_norm. Dropped: plot_spectrogram, load/save_checkpoint,
# del_old_checkpoints, scan_checkpoint — unused by the training closure, and the
# upstream file's top-level `import matplotlib.pylab` made matplotlib a hard
# import dependency of the vocoder module.
from torch.nn.utils import weight_norm


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)
