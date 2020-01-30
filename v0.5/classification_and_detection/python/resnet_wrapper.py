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
python wrapper for resnet50 on alveo U250
"""

__author__ = "Ussama Zahid, Lucian Petrica"
__copyright__ = "Copyright 2020, Xilinx"
__email__ = "{ussamaz, lucianp}@xilinx.com"

import os
import numpy as np
from pynq import Overlay, PL, allocate


class resnet_Wrapper():
    def __init__(self, network, params_path=None, bitstream_path=None, download_bitstream=True):
        self.bitstream_path = bitstream_path
        self.overlay = Overlay(self.bitstream_path, download=download_bitstream)
        if PL.bitfile_name != self.bitstream_path:
            raise RuntimeError("Incorrect Overlay loaded")
        self.accelerator = self.overlay.resnet50_1
        # loading fc weight in DDR of alveo
        wfile, _  = os.path.split(self.bitstream_path)
        wfile = wfile + "/fcweights.csv"
        print("Loading fc weights...")
        fcweights = np.genfromtxt(wfile, delimiter=',', dtype=np.int8)
        self.fcbuf = allocate((1000,2048), dtype=np.int8, target=self.overlay.bank0)
        #csv reader erroneously adds one extra element to the end, so remove, then reshape
        fcweights = fcweights[:-1].reshape(1000,2048)
        self.fcbuf[:] = fcweights
        self.fcbuf.flush()
        self.accel_input_buffer = None
        self.accel_output_buffer = None
        self.psl = 1

    def inference(self, imgs):
        # transfer to accelerator
        self.accel_input_buffer[:] = imgs
        self.accel_input_buffer.flush()     
        # do inference
        self.accelerator.call(self.accel_input_buffer, self.accel_output_buffer, self.fcbuf, imgs.shape[0])
        # get results
        self.accel_output_buffer.invalidate()
        results = np.copy(self.accel_output_buffer)
        return results

    def allocate_io_buffers(self, data):
        self.accel_input_buffer = allocate(shape=data.shape, dtype=np.uint8, target=self.overlay.bank0)
        # returns top 5 classes
        self.accel_output_buffer = allocate(shape=(data.shape[0]*self.psl, 5), dtype=np.uint32, target=self.overlay.bank0)
