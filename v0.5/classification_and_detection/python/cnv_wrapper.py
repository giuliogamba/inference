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

BNN_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
BNN_LIB_DIR = os.path.join(BNN_ROOT_DIR, 'libraries', PLATFORM)
BNN_BIT_DIR = os.path.join(BNN_ROOT_DIR, 'bitstreams', PLATFORM)
BNN_PARAM_DIR = os.path.join(BNN_ROOT_DIR, 'params')

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
		self.bitstream_name="{0}-{1}.bit".format(network,PLATFORM)
		if bitstream_path is None:
			self.bitstream_path=os.path.join(BNN_BIT_DIR, self.bitstream_name)
		else:
			self.bitstream_path = bitstream_path
		self.overlay = Overlay(self.bitstream_path, download=download_bitstream)
		if PL.bitfile_name != self.bitstream_path:
			raise RuntimeError("Incorrect Overlay loaded")
		self.bbj = self.overlay.BlackBoxJam_0.register_map
		self.base_addr = self.overlay.BlackBoxJam_0.mmio.base_addr
		self.network_name = network
		
		dllname = "./lib/layer_loader.so"
		if dllname not in _libraries:
			_libraries[dllname] = _ffi.dlopen(os.path.join(BNN_LIB_DIR, dllname))
		self.interface = _libraries[dllname]

		self.accel_input_buffer = None
		self.accel_output_buffer = None

		# from network's config.h
		self.layers = 9
		self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		
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
		
		paddedclasses = paddedSize(len(self.classes), bitsPerExtMemWord)
		self.psl = paddedSize(paddedclasses*16, bitsPerExtMemWord) // bitsPerExtMemWord
		
		# self.simd = [3 , 32, 32, 32, 32, 32, 4, 8, 1]
		# self.wei_prec_int_layer  = [1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1]
		# self.inp_prec_int_layer  = [1 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2]
		# self.wei_prec_frac_layer = [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,  0]
		# self.inp_prec_frac_layer = [7 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,  0]
		# self.out_prec_frac_layer = [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,  0]


		# else baked in weights will be used
		params_path = "/home/xilinx/BNN-PYNQ/bnn/params/cifar10/" + self.network_name
		if params_path is not None:
			self.load_parameters(params_path)
		else:
			print("Using baked in weights and thresholds (if any) of the accelerator...")

	def __del__(self):
		self.interface.deinit()
		
	# function to set weights and activation thresholds of specific network
	def load_parameters(self, params_path):
		if not os.path.isabs(params_path):
			params_path = os.path.join(BNN_PARAM_DIR, params_path, self.network_name)
		if os.path.isdir(params_path):
			start = time.time()
			self.params_loader(params_path)
			end = time.time() - start
			# print("Parameter count = {}".format(73792))
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
			# .so driver to load param. 1300x faster than python loops
			self.interface.load_layer(params.encode(), layer, self.pe[layer], \
				self.Wtiles[layer], self.Ttiles[layer], self.out_prec_int_layer[layer], self.base_addr)
			# # python for loops to load weights. Too too slow.
			# self.load_layer(layer, params)
			
	def load_layer(self, layer, params):
		self.load_layer_weights(layer, params)	
		self.load_layer_thresholds(layer, params)	

	def load_layer_weights(self, layer, params):
		for pe in range(self.pe[layer]):
			weight_file = params + "/" + str(layer) + "-" + str(pe) + "-weights.bin"
			binary_file = open(weight_file, "rb")
			temp = binary_file.read()
			arry = np.copy(np.frombuffer(temp, dtype=np.uint64))
			for tile in range(self.Wtiles[layer]):
				self.set_param(2*layer, pe, tile, 0, int(arry[tile]))
			binary_file.close()

	def load_layer_thresholds(self, layer, params):
		if self.out_prec_int_layer[layer] > 0:
			for pe in range(self.pe[layer]):
				threshold_file = params + "/" + str(layer) + "-" + str(pe) + "-thres.bin"
				binary_file = open(threshold_file, "rb")
				for tile in range(self.Ttiles[layer]):
					for j in range(self.out_prec_int_layer[layer]):
						temp = binary_file.read(8)
						temp = int.from_bytes(temp, byteorder="little")
						self.set_param(2*layer+1, pe, tile, j, temp)
				binary_file.close()

	def set_param(self, layer, pe, tile, threshold, param):
		self.bbj.doInit = 1

		self.bbj.targetLayer = layer
		self.bbj.targetMem = pe
		self.bbj.targetInd = tile
		self.bbj.targetThresh = threshold

		self.bbj.val_V_1 = param & 0xffffffff
		self.bbj.val_V_2 = (param >> 32) & 0xffffffff

		self.ExecAccel()

		self.bbj.doInit = 0

	def ExecAccel(self):
		self.bbj.CTRL.AP_START = 1
		while not self.bbj.CTRL.AP_DONE:
			pass

	def inference(self, imgs, count=None):
		# start = time.time()
		# self.allocate_io_buffers(input_shape=imgs.shape, output_shape=(imgs.shape[0]*self.psl,))
		# np.copyto(self.accel_input_buffer, imgs)
		
		# self.bbj.in_V_1 = imgs.physical_address & 0xffffffff
		# self.bbj.in_V_2 = (imgs.physical_address >> 32) & 0xffffffff

		# self.bbj.numReps = imgs.shape[0]
		self.ExecAccel()
		predictions = np.copy(np.frombuffer(self.accel_output_buffer, dtype=np.uint64))
		predictions = predictions.reshape(imgs.shape[0], -1).view(np.int16)
		predictions = predictions[:,:len(self.classes)]
		# self.free_io_buffers()
		# end = time.time() - start
		# print("Inference time = {}".format(end))
		return predictions


	def load_cifar10_test_set(self, path, count=None):
		test_batch = path + "/test_batch"
		with open(test_batch, "rb") as f:
			dict = pickle.load(f, encoding='bytes')
			imgs, labels = dict[b'data'], np.asarray(dict[b'labels'], dtype=np.uint8)
			image_count = imgs.shape[0]
		if count is None:
			count = image_count
		return imgs[:count], labels[:count]

	def preprocess_cifar10_images(self, imgs):
		bytes_paded = (paddedSize(imgs.shape[1]*8, bitsPerExtMemWord) - (imgs.shape[1]*8))//8
		imgs = np.pad(imgs, ((0,0),(0,bytes_paded)), 'constant', constant_values=0)
		imgs = 2*(imgs/255.)-1
		
		imgs = self.interleave_channels(imgs, 32, 32)
		binImages = self.quantizeAndPack(imgs)
		binImages = binImages.reshape(-1,)
		return binImages

	def quantizeAndPack(self, imgs):
		imgs = np.clip(imgs, -1, 1-(2**-7))
		imgs = np.round(imgs*2**7)
		imgs = imgs.astype(np.uint8).view(np.uint64)
		return imgs

	def interleave_channels(self, imgs, dim1, dim2):
		imgs = imgs.reshape(imgs.shape[0], -1, dim1*dim2)
		imgs = np.swapaxes(imgs, -1, 1).reshape(imgs.shape[0], -1)
		return imgs


	def allocate_io_buffers(self, input_shape, output_shape):

		# self.accel_input_buffer  = allocate(shape=input_shape, dtype=np.uint64)
		self.accel_output_buffer = allocate(shape=(output_shape,), dtype=np.uint64)
				
		# self.bbj.in_V_1 = self.accel_input_buffer.physical_address & 0xffffffff
		# self.bbj.in_V_2 = (self.accel_input_buffer.physical_address >> 32) & 0xffffffff

		self.bbj.out_V_1 = self.accel_output_buffer.physical_address & 0xffffffff
		self.bbj.out_V_2 = (self.accel_output_buffer.physical_address >> 32) & 0xffffffff
		
		self.bbj.doInit = 0


	def evaluate_cnv_results(self, labels, predictions, time):
		predictions = predictions.reshape(len(labels), -1).view(np.int16)
		predictions = predictions[:,:len(self.classes)]
		
		# # uncomment this when using with pytorch trained network
		# pred = np.zeros((10000,))
		# for i in range(predictions.shape[0]):
		# 	maxscr = 0
		# 	for j in range(predictions.shape[1]):
		# 		if predictions[i][j] >= maxscr:
		# 			pred[i] = j
		# 			maxscr = predictions[i][j]
		# predictions = pred

		# uncomment this when using with theano trained network
		predictions = np.argmax(predictions, axis=1)

		# np.savetxt('/home/xilinx/hw_res.csv', predictions)
		
		errors = np.count_nonzero(labels-predictions)
		accuracy = (len(labels) - errors)/float(len(labels))*100
		print("Accuracy: {:.2f}%".format(accuracy))
		print("Latency : {:.9f} seconds".format(time))		
		print("FPS     : {:.0f}".format(len(labels)/time))
		return predictions

	def free_io_buffers(self):
		# del self.accel_input_buffer
		del self.accel_output_buffer

