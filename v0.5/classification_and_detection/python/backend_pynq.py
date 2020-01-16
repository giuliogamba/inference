"""
pynq backend
"""

__author__ = "Ussama Zahid"
__copyright__ = "Copyright 2020, Xilinx"
__email__ = "ussamaz@xilinx.com"


import pynq
import backend
from bnn import lfcWrapper, cnvWrapper


class BackendPynq(backend.Backend):
    def __init__(self):
        super(BackendPynq, self).__init__()
        self.sess = None
        self.model = None
        self.device = "fpga"
    def version(self):
        return pynq.__version__

    def name(self):
        return "pynq"

    def image_format(self):
        return "BIN"

    def load(self, model_path, inputs=None, outputs=None, name=None):
        self.outputs = ["output"]
        self.inputs = ["input"]
        model_path = model_path.replace('//', '/')
        if name.startswith('lfc'):
            self.model = lfcWrapper(network=name, bitstream_path=model_path, download_bitstream=True)
        elif name.startswith('cnv'):
            self.model = cnvWrapper(network=name, bitstream_path=model_path)
        return self

    def predict(self, feed):
        # print(feed["input"])
        # exit(0)
        # key=[key for key in feed.keys()][0]    
        output = self.model.inference(feed["input"])    
        return output