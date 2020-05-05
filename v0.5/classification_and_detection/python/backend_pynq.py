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
pynq backend
"""

__author__ = "Ussama Zahid"
__copyright__ = "Copyright 2020, Xilinx"
__email__ = "ussamaz@xilinx.com"


import pynq
import backend


class BackendPynq(backend.Backend):
    def __init__(self):
        super(BackendPynq, self).__init__()
        self.model = None
        self.device = "fpga"

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
            self.model = lfcWrapper(network=name, bitstream_path=model_path)
        elif name.startswith('cnv'):
            from cnv_wrapper import cnvWrapper
            self.model = cnvWrapper(network=name, bitstream_path=model_path)
        elif name.startswith('resnet50'):
            from resnet_wrapper import resnet_Wrapper
            self.model = resnet_Wrapper(network=name, bitstream_path=model_path)
        return self

    def transfer_buffer(self, data, idx):
        self.model.transfer_buffer(data, idx)

    def init_cma_buffers(self, count, shape):
        self.model.init_cma_buffers(count, shape)


    def predict(self, i):
        output = self.model.inference(i)
        return output