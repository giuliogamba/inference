"""
pynq backend
"""

__author__ = "Ussama Zahid"
__copyright__ = "Copyright 2020, Xilinx"
__email__ = "ussamaz@xilinx.com"


import backend
import pynq
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

    def load(self):
    	pass