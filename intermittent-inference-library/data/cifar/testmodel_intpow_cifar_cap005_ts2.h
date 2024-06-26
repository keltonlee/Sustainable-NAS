/* 
* TestModel_INTPOW_cifar_cap005_ts2.h 
* (Auto-generated)
* Created on: 02/26/21,18:34:59 
* Label : TestModel_INTPOW_cifar_cap005_ts2 
*/

#ifndef TESTMODEL_INTPOW_CIFAR_CAP005_TS2_H_
#define TESTMODEL_INTPOW_CIFAR_CAP005_TS2_H_

#include "../cnn/cnn_types.h"
#include "../cnn/cnn_conv_tiled_std.h"
#include "../cnn/cnn_fc.h"
#include "../cnn/cnn_pool.h"



#pragma PERSISTENT(TestModel_INTPOW_cifar_cap005_ts2)
CNNLayer_t TestModel_INTPOW_cifar_cap005_ts2[7] = {
{
	.lix = 0,
	.fun = CNN_Intermittent_LayerConv_Tiled_Std,
	.weights = (Mat_t){
		.data = 38500,
		.n = 4,
		.ch = 3,
		.h = 5,
		.w = 5
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100,
		.n = 1,
		.ch = 3,
		.h = 32,
		.w = 32
	},
	.ofm = (Mat_t){
		.data = 19300,
		.n = 1,
		.ch = 4,
		.h = 28,
		.w = 28
	},
	.parE = (ExeParams_t){
		.Tn = 1,
		.Tm = 4,
		.Tr = 4,
		.Tc = 7,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	.lix = 1,
	.fun = CNN_Intermittent_LayerConv_Tiled_Std,
	.weights = (Mat_t){
		.data = 39100,
		.n = 16,
		.ch = 4,
		.h = 7,
		.w = 7
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 19300,
		.n = 1,
		.ch = 4,
		.h = 28,
		.w = 28
	},
	.ofm = (Mat_t){
		.data = 100,
		.n = 1,
		.ch = 16,
		.h = 22,
		.w = 22
	},
	.parE = (ExeParams_t){
		.Tn = 4,
		.Tm = 2,
		.Tr = 2,
		.Tc = 11,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	.lix = 2,
	.fun = CNN_Intermittent_LayerConv_Tiled_Std,
	.weights = (Mat_t){
		.data = 45372,
		.n = 24,
		.ch = 16,
		.h = 3,
		.w = 3
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100,
		.n = 1,
		.ch = 16,
		.h = 22,
		.w = 22
	},
	.ofm = (Mat_t){
		.data = 19300,
		.n = 1,
		.ch = 24,
		.h = 20,
		.w = 20
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 1,
		.Tr = 2,
		.Tc = 5,
		.str = 1,
		.pad = 0,
		.lpOdr = IFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 24,
	},
	.idxBuf = 0
},
{
	.lix = 3,
	.fun = CNN_Intermittent_LayerConv_Tiled_Std,
	.weights = (Mat_t){
		.data = 52284,
		.n = 24,
		.ch = 24,
		.h = 5,
		.w = 5
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 19300,
		.n = 1,
		.ch = 24,
		.h = 20,
		.w = 20
	},
	.ofm = (Mat_t){
		.data = 100,
		.n = 1,
		.ch = 24,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 12,
		.Tm = 1,
		.Tr = 2,
		.Tc = 4,
		.str = 1,
		.pad = 0,
		.lpOdr = IFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 12,
	},
	.idxBuf = 0
},
{
	.lix = 4,
	.fun = CNN_Intermittent_LayerConv_Tiled_Std,
	.weights = (Mat_t){
		.data = 81084,
		.n = 16,
		.ch = 24,
		.h = 3,
		.w = 3
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100,
		.n = 1,
		.ch = 24,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 19300,
		.n = 1,
		.ch = 16,
		.h = 14,
		.w = 14
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 1,
		.Tr = 1,
		.Tc = 7,
		.str = 1,
		.pad = 0,
		.lpOdr = IFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 16,
	},
	.idxBuf = 0
},
{
	.lix = 5,
	.fun = CNN_Intermittent_GlobalAveragePool,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 19300,
		.n = 1,
		.ch = 16,
		.h = 14,
		.w = 14
	},
	.ofm = (Mat_t){
		.data = 100,
		.n = 1,
		.ch = 16,
		.h = 1,
		.w = 1
	},
	.parE = (ExeParams_t){
		.Tn = 1,
		.Tm = 16,
		.Tr = 1,
		.Tc = 14,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 14,
	},
	.idxBuf = 0
},
{
	.lix = 6,
	.fun = CNN_Intermittent_LayerConv_Tiled_Std,
	.weights = (Mat_t){
		.data = 87996,
		.n = 10,
		.ch = 16,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100,
		.n = 1,
		.ch = 16,
		.h = 1,
		.w = 1
	},
	.ofm = (Mat_t){
		.data = 19300,
		.n = 1,
		.ch = 10,
		.h = 1,
		.w = 1
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 10,
		.Tr = 1,
		.Tc = 1,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
}
};

#pragma PERSISTENT(network)
CNNModel_t network={
	.Layers       = TestModel_INTPOW_cifar_cap005_ts2,
	.numLayers = 7,
	.name = "TestModel_INTPOW_cifar_cap005_ts2"
};

#endif /* TESTMODEL_INTPOW_CIFAR_CAP005_TS2_H_ */
