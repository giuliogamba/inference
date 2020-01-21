"""
pynq backend
"""

__author__ = "Ussama Zahid"
__copyright__ = "Copyright 2020, Xilinx"
__email__ = "ussamaz@xilinx.com"


import pynq
import backend
from threading import Lock


class BackendPynq(backend.Backend):
    def __init__(self):
        super(BackendPynq, self).__init__()
        self.sess = None
        self.model = None
        self.device = "fpga"
        self.lock = Lock()

    def version(self):
        return pynq.__version__

    def name(self):
        return "pynq"

    def image_format(self):
        return "BIN"

    def load(self, model_path, inputs=None, outputs=None, name=None):
        self.inputs = ["input"]
        self.outputs = ["output"]
        model_path = model_path.replace('//', '/')
        if name.startswith('lfc'):
            from lfc_wrapper import lfcWrapper
            self.model = lfcWrapper(network=name, bitstream_path=model_path, download_bitstream=True)
        elif name.startswith('cnv'):
            from cnv_wrapper import cnvWrapper
            self.model = cnvWrapper(network=name, bitstream_path=model_path, download_bitstream=True)
        elif name.startswith('resnet50'):
            from resnet_wrapper import resnet_Wrapper
            self.model = resnet_Wrapper(network=name, bitstream_path=model_path, download_bitstream=True)
        return self

    def predict(self, feed):
        self.lock.acquire()    
        output = self.model.inference(feed["input"])    
        self.lock.release()        
        return output