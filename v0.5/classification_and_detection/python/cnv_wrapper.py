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
python wrapper for cnv networks
"""

__author__ = "Ussama Zahid"
__copyright__ = "Copyright 2020, Xilinx"
__email__ = "ussamaz@xilinx.com"

import os
import time
import cffi
import pickle
import numpy as np
from pynq import Overlay, PL, allocate

bitsPerExtMemWord = 64

if os.environ['BOARD'] == 'Ultra96':
    PLATFORM="ultra96"
elif os.environ['BOARD'] == 'Pynq-Z1' or os.environ['BOARD'] == 'Pynq-Z2':
    PLATFORM="pynqZ1-Z2"
else:
    raise RuntimeError("Board not supported")

_ffi = cffi.FFI()

_ffi.cdef("""
void load_layer(const char* path, unsigned int layer, unsigned int PEs, unsigned int Wtiles, unsigned int Ttiles, unsigned int API, unsigned int addr);
void deinit();
"""
)

_libraries = {}

def paddedSize(inp, padTo):
    if inp % padTo == 0: 
        return inp
    else:
        return inp + padTo - (inp % padTo)

class cnvWrapper:
    def __init__(self, network, params_path=None, bitstream_path=None, download_bitstream=True):
        self.bitstream_path = bitstream_path
        self.overlay = Overlay(self.bitstream_path, download=download_bitstream)
        if PL.bitfile_name != self.bitstream_path:
            raise RuntimeError("Incorrect Overlay loaded")
        self.bbj = self.overlay.BlackBoxJam_0.register_map
        self.base_addr = self.overlay.BlackBoxJam_0.mmio.base_addr
        self.network_name = network
        
        dllname = "./python/pynq-lib/layer_loader.so"
        if dllname not in _libraries:
            _libraries[dllname] = _ffi.dlopen(dllname)
        self.interface = _libraries[dllname]

        self.accel_input_buffer = None
        self.accel_output_buffer = None

        # from network's config.h
        self.layers = 9
        self.pe   = [16, 32, 16, 16,  4,  1, 1, 1, 4]
        self.Wtiles = [36, 36, 144, 288, 2304, 18432, 32768, 32768, 8192]
        self.Ttiles = [ 4,  2,   8,   8,   64,   256,   512,   512,   16]   
        if self.network_name == "cnvW1A1":
            self.out_prec_int_layer  = [1 , 1 , 1 , 1 , 1 , 1 , 1, 1,  0]
        elif self.network_name == "cnvW1A2":
            self.out_prec_int_layer  = [2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 0]
        elif self.network_name == "cnvW2A2":
            self.pe   = [8, 16,  8,  8, 4, 1, 1, 2, 4]
            self.Wtiles = [72, 144, 576, 1152, 9216, 73728, 65536, 65536, 8192]
            self.Ttiles = [ 8,   4,  16,   16,   64,   256,   512,   256,   16]
            self.out_prec_int_layer  = [2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 0]
        
        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        paddedclasses = paddedSize(len(self.classes), bitsPerExtMemWord)
        self.psl = paddedSize(paddedclasses*16, bitsPerExtMemWord) // bitsPerExtMemWord

        # else baked in weights will be used
        params_path = None #"/home/xilinx/BNN-PYNQ/bnn/params/cifar10/" + self.network_name
        if params_path is not None:
            self.load_parameters(params_path)
        else:
            print("Using baked in weights and thresholds (if any) of the accelerator...")
        
    # function to set weights and activation thresholds of specific network
    def load_parameters(self, params_path):
        if os.path.isdir(params_path):
            start = time.time()
            self.params_loader(params_path)
            end = time.time() - start
            print("Parameter loading took {:.2f} sec...\n".format(end))
            self.classes = []
            with open (os.path.join(params_path, "classes.txt")) as f:
                self.classes = [c.strip() for c in f.readlines()]
            filter(None, self.classes)
        else:
            print("\nERROR: No such parameter directory \"" + params_path + "\"")

    def params_loader(self, params):
        print("Setting network weights and thresholds in accelerator...")
        for layer in range(self.layers):
            self.interface.load_layer(params.encode(), layer, self.pe[layer], \
                self.Wtiles[layer], self.Ttiles[layer], self.out_prec_int_layer[layer], self.base_addr)

    def ExecAccel(self):
        self.bbj.CTRL.AP_START = 1
        while not self.bbj.CTRL.AP_DONE:
            pass

    def inference(self, imgs, count=None):
        self.ExecAccel()
        predictions = np.copy(np.frombuffer(self.accel_output_buffer, dtype=np.uint64))
        predictions = predictions.reshape(imgs.shape[0], -1).view(np.int16)
        predictions = predictions[:,:len(self.classes)]
        return predictions

    def allocate_io_buffers(self, data):

        self.accel_input_buffer  = allocate(shape=data.shape, dtype=np.uint64)
        self.accel_output_buffer = allocate(shape=(data.shape[0]*self.psl,), dtype=np.uint64)
        
        np.copyto(self.accel_input_buffer, data)        
        
        self.bbj.in_V_1 = self.accel_input_buffer.physical_address & 0xffffffff
        self.bbj.in_V_2 = (self.accel_input_buffer.physical_address >> 32) & 0xffffffff

        self.bbj.out_V_1 = self.accel_output_buffer.physical_address & 0xffffffff
        self.bbj.out_V_2 = (self.accel_output_buffer.physical_address >> 32) & 0xffffffff

        self.bbj.doInit = 0
        self.bbj.numReps = data.shape[0]

    def __del__(self):
        self.interface.deinit()

