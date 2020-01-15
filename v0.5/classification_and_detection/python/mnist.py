"""
helper to load mnist testset
"""

__author__ = "Ussama Zahid"
__copyright__ = "Copyright 2020, Xilinx"
__email__ = "ussamaz@xilinx.com"

import gzip
import dataset

class MNIST(dataset.Dataset):
    def __init__(self, data_path, image_list, name, use_cache=0, image_size=None,
                 image_format="NHWC", pre_process=None, count=None, cache_dir=None,use_label_map=False):
        super().__init__()
        pass