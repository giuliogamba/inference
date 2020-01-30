# Copyright (c) 2020, Xilinx, Inc.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:

# 1.  Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the following disclaimer.

# 2.  Redistributions in binary form must reproduce the above copyright 
#   notice, this list of conditions and the following disclaimer in the 
#   documentation and/or other materials provided with the distribution.

# 3.  Neither the name of the copyright holder nor the names of its 
#   contributors may be used to endorse or promote products derived from 
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
helper to load and preprocess cifar10 testset
"""

__author__ = "Ussama Zahid"
__copyright__ = "Copyright 2020, Xilinx"
__email__ = "ussamaz@xilinx.com"

import time
import pickle
import dataset
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cifar10")

class CIFAR10(dataset.Dataset):
    def __init__(self, data_path, image_list, name, use_cache=0, image_size=None,
                 image_format="BIN", pre_process=None, count=None, cache_dir=None, use_label_map=False):
        super(CIFAR10, self).__init__()
        if image_size is None:
	        self.image_size = [1, 32*32*3]
        else:
            self.image_size = image_size

        self.data_path = data_path
        self.name = name
        self.count = count
        self.use_cache = use_cache
        self.pre_process = pre_process
        self.classes = 10
        self.np_data_type = np.float64
        
        start = time.time()
        images, labels = self.load_cifar10_test_set(self.data_path, self.count)
        images = self.pre_process(images)
        self.image_list = images.tolist()        
        self.label_list = labels.tolist()
        time_taken = time.time() - start

        log.info("loaded {} images, cache={}, took={:.1f}sec".format(
            len(self.image_list), use_cache, time_taken))

    def load_cifar10_test_set(self, path, count=None):
        test_batch = path + "/test_batch"
        with open(test_batch, "rb") as f:
            dict = pickle.load(f, encoding='bytes')
            imgs, labels = dict[b'data'], np.asarray(dict[b'labels'], dtype=np.uint8)
            image_count = imgs.shape[0]
        if count is None:
            count = image_count
        return imgs[:count], labels[:count]

    def get_item(self, nbr):
        img = self.image_list[nbr]
        label = self.label_list[nbr]
        return img, label

    def get_item_loc(self, nr):
        pass