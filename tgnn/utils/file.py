# -*- coding: utf-8 -*-
# Copyright (c) 2022, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/10/21 17:19
import os
import gzip
import sys
import re

import requests
from zipfile import ZipFile
from pathlib import Path
import shutil
from tqdm import tqdm


def download_file(local_file, url):
    """ downloads remote_file to local_file if necessary """
    if not os.path.isfile(local_file):
        print(f"downloading {url} to {local_file}")
        response = requests.get(url)
        open(local_file, "wb").write(response.content)


def get_cache_dir(name="tgnn"):
    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, '.cache', name)
    os.makedirs(cache_dir, exist_ok=True)

    return cache_dir


class File:
    """
    Small class for downloading models and training assets.
    """

    def __init__(self, path, url, force=False):
        self.path = Path(path)
        self.force = force
        self.url = str(url)

    def location(self, filename):
        return self.path / filename

    def exists(self, filename):
        return self.location(filename).exists()

    def download(self):
        """
        Download the remote file
        """
        req = requests.get(self.url, stream=True)
        total = int(req.headers.get('content-length', 0))
        fname = re.findall('filename="([^"]+)', req.headers['content-disposition'])[0]
        # skip download if local file is found
        if self.exists(fname.strip('.zip')) and not self.force:
            print("[skipping %s]" % fname, file=sys.stderr)

            return

        if self.exists(fname.strip('.zip')) and self.force:
            shutil.rmtree(self.location(fname.strip('.zip')))

        # download the file
        with tqdm(total=total, unit='iB', ascii=True, ncols=100, unit_scale=True, leave=False) as t:
            with open(self.location(fname), 'wb') as f:
                for data in req.iter_content(1024):
                    f.write(data)
                    t.update(len(data))

        print("[downloaded %s]" % fname, file=sys.stderr)

        if fname.endswith('.zip'):
            with ZipFile(self.location(fname), 'r') as zfile:
                zfile.extractall(self.path)
            os.remove(self.location(fname))


def unzip(filename):
    with gzip.GzipFile(filename) as gz:
        return gz.read()
