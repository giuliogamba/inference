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