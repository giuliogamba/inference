"""
helper to load and preprocess mnist testset
"""

__author__ = "Ussama Zahid"
__copyright__ = "Copyright 2020, Xilinx"
__email__ = "ussamaz@xilinx.com"

import dataset
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mnist")


class MNIST(dataset.Dataset):
    def __init__(self, data_path, image_list, name, use_cache=0, image_size=None,
                 image_format="BIN", pre_process=None, count=None, cache_dir=None, use_label_map=False):
        super(MNIST, self).__init__()
        if image_size is None:
	        self.image_size = [1, 784]
        else:
            self.image_size = image_size

        self.data_path = data_path
        self.name = name
        self.count = count
        self.use_cache = use_cache
        self.pre_process = pre_process
        self.classes = 10
        self.use_cma = True
        start = time.time()
        images, labels = self.load_mnist_test_set(self.data_path, self.count)
        images = self.pre_process(images)
        self.image_list = images.tolist()        
        self.label_list = labels.tolist()
        time_taken = time.time() - start

        log.info("loaded {} images, cache={}, took={:.1f}sec".format(
            len(self.image_list), use_cache, time_taken))

    def load_mnist_test_set(self, path, count=None):
        imgs_file = path + "/t10k-images-idx3-ubyte"
        label_file = path + "/t10k-labels-idx1-ubyte"

        with open(imgs_file, 'rb') as f:
            magic_number = int.from_bytes(f.read(4), byteorder="big")
            image_count = int.from_bytes(f.read(4), byteorder="big")
            dim = int.from_bytes(f.read(4), byteorder="big")
            dim = int.from_bytes(f.read(4), byteorder="big")
            imgs = np.frombuffer(f.read(), dtype=np.uint8)
            imgs = imgs.reshape(image_count, dim*dim)

        with open(label_file, 'rb') as f:
            magic_number = int.from_bytes(f.read(4), byteorder="big")
            label_count = int.from_bytes(f.read(4), byteorder="big")
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        if count is None:
            count = image_count

        return imgs[:count], labels[:count]


    def get_item(self, nbr):
        img = self.image_list[nbr]
        label = self.label_list[nbr]
        return img, label

    def get_item_loc(self, nr):
        pass