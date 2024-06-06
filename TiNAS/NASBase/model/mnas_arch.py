import numpy as np
from pprint import pprint
import math
import sys

import torch.nn as nn

from collections import OrderedDict

from .mnas_ops import ReLUConvBN, MBConv, MBConv1D
from .common_types import LAYERTYPES

# import train_search
# from controllers.environment import envs
# from helpers import evaluate
# from mnasnet.evolution import evolution
# from mnasnet.operations import *
# from quantization.quantz import quantize_rl

'''
taken from : 
https://github.com/yukang2017/NAS-quantization
'''

class Block(nn.Module):
    def __init__(self, Bix, C_in, C_out, kernel_size, num_layers, conv_op, exp_factor, stride_first=2, support_skip=True, affine=True, name='',
                 batchnorm_epsilon=1e-5): #  C_out
        super(Block, self).__init__()
        self._name = name
        self._layers = nn.ModuleList()
                
        # -- Validations        
        if conv_op == LAYERTYPES.L_CONV:      #'conv'
            conv = ReLUConvBN
        # elif conv_op == LAYERTYPES.L_SEPCONV:    #'sep_conv'
        #     conv = SepConv        
        
        
        elif conv_op == LAYERTYPES.L_MBCONV:    #'mib_conv'
            conv = MBConv

        elif conv_op == LAYERTYPES.L_MBCONV_1D:
            conv = MBConv1D
        
                
        # elif conv_op == LAYERTYPES.L_MBCONV1:    #'mib_conv1'
        #     conv = MBConv1
        # elif conv_op == LAYERTYPES.L_MBCONV2:    #'mib_conv2'
        #     conv = MBConv2
        # elif conv_op == LAYERTYPES.L_MBCONV3:    #'mib_conv3'
        #     conv = MBConv3
        # elif conv_op == LAYERTYPES.L_MBCONV4:    #'mib_conv4'
        #     conv = MBConv4
        # elif conv_op == LAYERTYPES.L_MBCONV5:    #'mib_conv5'
        #     conv = MBConv5
        # elif conv_op == LAYERTYPES.L_MBCONV6:    #'mib_conv6'
        #     conv = MBConv6
        # elif conv_op == LAYERTYPES.L_MBCONV7:    #'mib_conv7'
        #     conv = MBConv7    
        # elif conv_op == LAYERTYPES.L_MBCONV8:    #'mib_conv8'
        #     conv = MBConv8
        
        else:
            raise ValueError('Wrong conv layer type {}'.format(conv_op))
        
        C_i = C_in; C_o = C_out
                
        if((C_i < 1) or (C_o < 1)): raise ValueError('Wrong Cin,Cout : {},{}'.format(C_i, C_o))        
        if (num_layers < 1): raise ValueError('Wrong num_layers : {}'.format(num_layers))

        #padding=int((kernel_size-1)/2)
        #padding = 1
        
        for layer in range(num_layers):            
            if layer == 0:
                stride = stride_first; 
                in_channels = C_i
            else:
                stride = 1; 
                in_channels = C_o
            
            padding=int((kernel_size-1)/2)
            conv_name = name_prefix='{}_'.format(Bix)
            op = conv(conv_name, C_in=in_channels, C_out=C_o, 
                        kernel_size=kernel_size, expansion_factor=exp_factor, stride=stride, padding=padding, support_skip=support_skip, 
                        affine=affine, batchnorm_epsilon=batchnorm_epsilon)
            
            self._layers.append(op)
            
            

    def forward(self, x):        
        #print('---------------> Block:start ', self._name)
        #for op, skip in zip(self._layers, self._skips):                        # skip disabled
        for lix, op in enumerate(self._layers):
            #print('---------------> layer_start - ',lix)            
            x = op(x)
            #print('Block:end ', self._name, ' op_res_size=', op_res.size(), ' skip_res_size=', skip_res.size())
            #print("layer-output size =", x.size())
            #print('---------------> layer_end - ',lix)
        
        #print('---------------> Block:end ', self._name)
        
        return x

    
class MNASSuperNet(nn.Module):
    SUPERNET_OBJTYPE = 'mnas'

    def __init__(self, global_settings, dataset, block_out_channels, net_choices = None, blk_choices=None, search_options=None,
                 name='mnas_supernet_test'):
        super(MNASSuperNet, self).__init__()
        
        self.global_settings = global_settings
        self.dataset         = dataset                        
        self.output_channels = block_out_channels
        self.name            = name

        settings_per_dataset = global_settings.NAS_SETTINGS_PER_DATASET[dataset]
        if self.dataset in ("CIFAR10", "HAR", "KWS"):
            self.num_blocks  = settings_per_dataset['NUM_BLOCKS']
            self.num_classes = settings_per_dataset['NUM_CLASSES']
            self.stem_c_out  = settings_per_dataset['STEM_C_OUT']
            self.input_channels = settings_per_dataset['INPUT_CHANNELS']            
            self.stride_first = settings_per_dataset['STRIDE_FIRST']   
            self.downsample_blocks = settings_per_dataset['DOWNSAMPLE_BLOCKS']            
        else:
            raise ValueError('Class Network::dataset {} unknown'.format(dataset))
        
        # == BUILD SUPERNET ===

        self.use_1d_conv = settings_per_dataset['USE_1D_CONV']

        if self.use_1d_conv:
            conv_class = nn.Conv1d
            batchnorm_class = nn.BatchNorm1d
            pooling_class = nn.AdaptiveAvgPool1d
        else:
            conv_class = nn.Conv2d
            batchnorm_class = nn.BatchNorm2d
            pooling_class = nn.AdaptiveAvgPool2d

        batchnorm_epsilon = global_settings.NAS_SETTINGS_GENERAL['TRAIN_BATCHNORM_EPSILON']
        
        # -- initial layer
        self.stem = nn.Sequential(
            OrderedDict([
                ("conv0_stem", conv_class(in_channels=self.input_channels, out_channels=self.stem_c_out, kernel_size=3, stride=1, padding=1, bias=False)),
                ("bn0_stem", batchnorm_class(self.stem_c_out))
        ]))
        
        self.net_choices = net_choices
        
        # get different choice permutations for each block, or use whats given
        if blk_choices == None:
            self.blk_choices = self.supernet_block_choices(self.num_blocks)
        else:
            self.blk_choices = blk_choices
        
        # -- blocks part of the search space
        self.choice_blocks = nn.ModuleList()
        
        prev_c_output = -1
        for bix in range(self.num_blocks):
            
            c_output = block_out_channels[bix] # get c_out for current block
                        
            # stride for first layer changes for downsample blocks
            stride_first = 2 if (bix in self.downsample_blocks) else 1            
            #stride_first = 2
            
            # c_out of prev block becomes c_in of current block
            if bix == 0: # first block
                c_input  = self.stem_c_out                    
            else:
                c_input  =  prev_c_output # c_cout for prev block
            
            #print("block:", bix, " c_in=",c_input, ", c_output=", c_output)
            
            cb_list = nn.ModuleList([])
            for cix, each_choice in enumerate(self.blk_choices): # here each choice is a list of different blk_choices per block
                
                # get parms for this subnet
                if self.use_1d_conv:
                    conv_op = LAYERTYPES.L_MBCONV_1D
                else:
                    conv_op = LAYERTYPES.L_MBCONV
                exp_factor = each_choice[0]
                kernel_size = each_choice[1]
                num_layers = each_choice[2]
                supp_skip = each_choice[3]
                
                # if search_options is None:
                #     conv_op = each_choice[0]
                #     kernel_size = each_choice[1]
                #     num_layers = each_choice[2]
                #     supp_skip = each_choice[3]
                # else:
                #     conv_op = each_choice[0]
                #     kernel_size = search_options['KERNEL_SIZES'][each_choice[1]]
                #     num_layers = search_options['MOBILENET_NUM_LAYERS_EXPLICIT'][each_choice[2]]
                #     supp_skip = search_options['SUPPORT_SKIP'][each_choice[3]]
                                                
                if (num_layers < 1): num_layers = 1 # must have at least one layer
                                
                block_name = "Block-B{}-{}".format(bix,cix)
                block = Block(Bix = bix, C_in=c_input, C_out=c_output, 
                              kernel_size=kernel_size, num_layers=num_layers, conv_op=conv_op, exp_factor=exp_factor, 
                              stride_first=stride_first, support_skip=supp_skip, affine=True, name=block_name, batchnorm_epsilon=batchnorm_epsilon)
                cb_list.append(block)                
        
            prev_c_output = c_output
            
            self.choice_blocks.append(cb_list)
                
        # -- last layers
        self.global_pooling = pooling_class(1)
        self.classifier = nn.Linear(c_output, self.num_classes)
                
        self._initialize_weights()
                
    def forward(self, x, rnd_choice_per_block):        
        #self._debug_supernet_block_choices(rnd_choice_per_block)                
        x = self.stem(x); #print('stem fin, x_size=', x.size())
        
        # choice only one of the permulations
        for bix in range(self.num_blocks):
            x = self.choice_blocks[bix][rnd_choice_per_block[bix]](x)
            #print('finished Block ',bix, ' x_size=', x.size())
        out = self.global_pooling(x)
        #print('finished gpool out_size=', out.size())
        logits = self.classifier(out.view(out.size(0),-1))
        return logits


    
    def get_search_space_size(self):        
        sss = len(self.blk_choices)**self.num_blocks
        return sss

    def get_all_subnets(self):
        return 

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            else:
                #print('_initialize_weights::skipping unknown layer type {}'.format(type(m)))
                pass


    def supernet_block_choices(self, num_blocks, search_options=None):
        # Import here to avoid circular import
        from .common_utils import parametric_supernet_blk_choices
        
        # get required params blk_choices
        
        
        # if search_options is None:
        #     k_convtypes = sorted(list(CONV_TYPES.keys()))
        #     k_kernelsizes = sorted(list(KERNEL_SIZES.keys()))
        #     k_num_layers_explicit = sorted(list(MOBILENET_NUM_LAYERS_EXPLICIT.keys()))
        #     k_support_skip = sorted(list(SUPPORT_SKIP.keys()))
        # else:
        #     k_convtypes = sorted(list(search_options['CONV_TYPES'].keys()))
        #     k_kernelsizes = sorted(list(search_options['KERNEL_SIZES'].keys()))
        #     k_num_layers_explicit = sorted(list(search_options['MOBILENET_NUM_LAYERS_EXPLICIT'].keys()))
        #     k_support_skip = sorted(list(search_options['SUPPORT_SKIP'].keys()))
        
        blk_choices = parametric_supernet_blk_choices(search_options=search_options, global_settings=self.global_settings)
                        
                    #for each_nld in k_num_layers_mbnet_delta:
                        #for k_ncr in k_num_out_channels_mbnet_ratio:  # for now, remove this from the supernet search space                            
                        #blk_choices.append([each_cnv, each_ksz, each_skp, each_nld])
                        #blk_choices.append([each_cnv, each_ksz, each_skp])
                        
                        
        
        #choices_per_block = itertools.product(blk_choices, repeat=num_blocks)        
        #pprint([x for x in itertools.product(blk_choices, repeat=num_blocks)])        
        
        # choices_ixs = np.arange(len(blk_choices))
        # print ("finished getting blk_choices : ", len(blk_choices))

        # # from https://itecnote.com/tecnote/python-itertools-product-speed-up/
        # choices_per_block = np.array(choices_ixs)[np.rollaxis(
        #                     np.indices((len(choices_ixs),) * num_blocks), 0, num_blocks + 1)
        #                     .reshape(-1, num_blocks)]
        
        #return  choices_per_block, blk_choices
        
        # report full search space
        #print("Full search space size = ", len(blk_choices)**num_blocks)
        
        return blk_choices
    
    
    # =========== debug related =========    
    def _debug_get_tot_num_layers(self, choice_per_blk):
        sys.exit("_debug_get_tot_num_layers:: not implemented")
        # exlucde the layers in the stem and last
        # tot_nl = 0; all_blk_nl = []
        # for bix, c in enumerate(choice_per_blk):            
        #     nl = self.blk_choices[c][3] + MOBILENET_V2_NUM_LAYERS[bix]
        #     all_blk_nl.append(nl)
        #     tot_nl+=nl       
        # return tot_nl
    
    def _debug_supernet_block_choices(self, choices_per_blk):
        print("rnd_choice_per_block => ")
        pprint([self.blk_choices[cix] for cix in choices_per_blk])
        print("\n")
        

# a single subnet
class MNASSubNet(nn.Module):
    
    def __init__(self, name, num_blocks, num_classes, stem_c_out, input_channels, stride_first, downsample_blocks, block_out_channels,
                       subnet_id, single_choice_per_block, net_choices=None, search_options=None, use_1d_conv=False,
                       ):
        super(MNASSubNet, self).__init__()

        self.id = subnet_id
        self.name = name
        self.output_channels = block_out_channels

        self.num_blocks  = num_blocks
        self.num_classes = num_classes
        self.stem_c_out  = stem_c_out
        self.input_channels = input_channels
        self.stride_first = stride_first
        self.downsample_blocks = downsample_blocks
        self.choice_per_block = single_choice_per_block
        self.net_choices = net_choices
        self.use_1d_conv = use_1d_conv
        
        # == BUILD SUBNET ===
        if self.use_1d_conv:
            conv_class = nn.Conv1d
            batchnorm_class = nn.BatchNorm1d
            pooling_class = nn.AdaptiveAvgPool1d
        else:
            conv_class = nn.Conv2d
            batchnorm_class = nn.BatchNorm2d
            pooling_class = nn.AdaptiveAvgPool2d
        
        # -- initial layer
        self.stem = nn.Sequential(
            OrderedDict([
                ("conv0", conv_class(in_channels=self.input_channels, out_channels=self.stem_c_out, kernel_size=3, stride=1, padding=1, bias=False)),
                ("bn0", batchnorm_class(self.stem_c_out))
        ]))

        c_output = self.stem_c_out
                
        # -- blocks of the subnet
        self.choice_blocks = nn.ModuleList()
        
        prev_c_output = -1
        for bix in range(self.num_blocks):
            
            c_output = block_out_channels[bix] # get c_out for current block
                        
            # stride for first layer changes for downsample blocks            
            stride_first = 2 if (bix in self.downsample_blocks) else 1            
            #stride_first = 2
            
            # c_out of prev block becomes c_in of current block
            if bix == 0: c_input  = self.stem_c_out  # first block                
            else: c_input = prev_c_output # c_cout for prev block
            
            
            # get parms for this subnet            
            if self.use_1d_conv:
                conv_op = LAYERTYPES.L_MBCONV_1D
            else:
                conv_op = LAYERTYPES.L_MBCONV
            exp_factor = single_choice_per_block[bix][0]
            kernel_size = single_choice_per_block[bix][1]
            num_layers = single_choice_per_block[bix][2]
            supp_skip = single_choice_per_block[bix][3]            
            
            # # get parms for this subnet
            # if search_options is None:
            #     conv_op = single_choice_per_block[bix][0]
            #     kernel_size = KERNEL_SIZES[single_choice_per_block[bix][1]]            
            #     num_layers = MOBILENET_NUM_LAYERS_EXPLICIT[single_choice_per_block[bix][2]]
            #     supp_skip = SUPPORT_SKIP[single_choice_per_block[bix][3]]
            # else:
            #     conv_op = single_choice_per_block[bix][0]
            #     kernel_size = search_options['KERNEL_SIZES'][single_choice_per_block[bix][1]]
            #     num_layers = search_options['MOBILENET_NUM_LAYERS_EXPLICIT'][single_choice_per_block[bix][2]]
            #     supp_skip = search_options['SUPPORT_SKIP'][single_choice_per_block[bix][3]]
                        
            if (num_layers < 1): num_layers = 1
            
            #print("Blk={}, t=conv_op={}, ks={}, ch_out={}, nl={}".format(bix, conv_op, kernel_size, c_output, num_layers))
                            
            block_name = "Block-B{}-{}".format(bix, subnet_id)
            block = Block(Bix = bix, C_in=c_input, C_out=c_output, 
                            kernel_size=kernel_size, num_layers=num_layers, conv_op=conv_op, exp_factor=exp_factor,
                            stride_first=stride_first, support_skip=supp_skip, affine=True, name=block_name)
            
            self.choice_blocks.append(block)        
            
            prev_c_output = c_output
                
        # -- last layers
        self.global_pooling = pooling_class(1)
        self.classifier = nn.Linear(c_output, self.num_classes)
                
        self._initialize_weights()
                
    def forward(self, x):                
        x = self.stem(x); #print('stem out, x_size=', x.size())
        # choice only one of the permulations
        for bix in range(self.num_blocks):
            x = self.choice_blocks[bix](x)
            #print('finished Block ',bix, ' x_size=', x.size())
        out = self.global_pooling(x); #print('finished pooling ',bix, ' out_size=', out.size())        
        logits = self.classifier(out.view(out.size(0),-1)); #print('finished classifier ',bix, ' logits_size=', logits.size())    
        return logits
    
    
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)




    
                    
                    
