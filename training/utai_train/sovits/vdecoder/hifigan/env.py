# Vendored verbatim from so-vits-svc 4.1-Stable (vdecoder/hifigan/env.py @ 730930d).
# Changes vs upstream: package-relative imports only.
import os
import shutil


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))
