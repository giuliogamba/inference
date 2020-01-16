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

import os
import time
import cffi
import numpy as np
from pynq import Overlay, PL, allocate


__author__ = "Ussama Zahid"
__copyright__ = "Copyright 2020, Xilinx"
__email__ = "ussamaz@xilinx.com"

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

# for LfcWxAy
class lfcWrapper:
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

		dllname = "/home/xilinx/BNN-PYNQ/bnn/libraries/ultra96/layer_loader.so"
		if dllname not in _libraries:
			_libraries[dllname] = _ffi.dlopen(os.path.join(BNN_LIB_DIR, dllname))
		self.interface = _libraries[dllname]

		self.accel_input_buffer = None
		self.accel_output_buffer = None

		# from network's config.h
		self.layers = 4
		self.pe   = [32, 64, 32, 16]
		self.Wtiles = [416, 512, 512, 512]
		self.Ttiles = [32, 16, 32, 4]

		if self.network_name == "lfcW1A1":
			self.out_prec_int_layer = [1,1,1,1]
		elif self.network_name == "lfcW1A2":
			self.out_prec_int_layer = [2,2,2,1]
		self.classes = [0,1,2,3,4,5,6,7,8,9]
		self.psl = paddedSize(len(self.classes), bitsPerExtMemWord) // bitsPerExtMemWord
		
		# self.simd = [64, 32, 64,  8]
		# self.wei_prec_int_layer = [1,1,1,1]
		# self.inp_prec_int_layer = [1,1,1,1]
		# self.wei_prec_frac_layer = [0,0,0,0]
		# self.inp_prec_frac_layer = [0,0,0,0]
		# self.out_prec_frac_layer = [0,0,0,0]

		# else baked in weights will be used
		if params_path is not None:
			self.load_parameters(params_path)
		else:
			print("Using baked in weights and thresholds (if any) of the accelerator...")
		
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
		# imgs = imgs.astype(np.uint64)
		self.allocate_io_buffers(input_shape=imgs.shape, output_shape=(imgs.shape[0]*self.psl,))
		np.copyto(self.accel_input_buffer, imgs)
		self.bbj.numReps = imgs.shape[0]
		start = time.time()
		self.ExecAccel()
		end = time.time() - start
		predictions = np.copy(np.frombuffer(self.accel_output_buffer, dtype=np.uint64))
		self.free_io_buffers()
		return predictions

	# dataset specific
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

	def preprocess_mnist_images(self, imgs):
		psl = paddedSize(len(self.classes), bitsPerExtMemWord) // bitsPerExtMemWord
		imgs = 2*(imgs/255.)-1
		binImages = self.binarizeAndPack(imgs)
		return binImages, psl


	def binarizeAndPack(self, imgs):
		values_paded = paddedSize(imgs.shape[1], bitsPerExtMemWord) - imgs.shape[1]
		imgs = np.where(imgs < 0, False, True).astype(np.bool)
		imgs = np.pad(imgs, ((0,0),(0,values_paded)), 'constant', constant_values=False)
		binImages = np.packbits(imgs, axis=1, bitorder='little').view(np.uint64).reshape(-1,)
		return binImages


	def allocate_io_buffers(self, input_shape, output_shape):

		self.accel_input_buffer  = allocate(shape=input_shape, dtype=np.uint64)
		self.accel_output_buffer = allocate(shape=output_shape, dtype=np.uint64)
		
		np.copyto(self.accel_output_buffer, np.zeros(output_shape, dtype=np.uint64))
		
		self.bbj.in_V_1 = self.accel_input_buffer.physical_address & 0xffffffff
		self.bbj.in_V_2 = (self.accel_input_buffer.physical_address >> 32) & 0xffffffff

		self.bbj.out_V_1 = self.accel_output_buffer.physical_address & 0xffffffff
		self.bbj.out_V_2 = (self.accel_output_buffer.physical_address >> 32) & 0xffffffff
		
		self.bbj.doInit = 0

	# from hinge loss settings for cross entropy use argmax instead of log2.
	def evaluate_results(self, labels, predictions, time):
		# set the value to 1 for bit accurate with theano
		# set the value to 2**9 for bit accurate with pytorch
		predictions = np.where(predictions==0, 1, predictions)
		pred = np.log2(predictions).astype(np.uint)
		errors = np.count_nonzero(labels-pred[:len(labels)])
		accuracy = (len(labels) - errors)/float(len(labels))*100
		print("Accuracy: {:.2f}%".format(accuracy))
		print("Latency : {:.9f} seconds".format(time))		
		print("FPS     : {:.0f}".format(len(labels)/time))
		return pred


	def free_io_buffers(self):
		del self.accel_input_buffer
		del self.accel_output_buffer
		self.interface.deinit()