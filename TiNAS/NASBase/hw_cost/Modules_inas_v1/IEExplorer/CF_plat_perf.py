
import math
import sys
import random
import os
from os.path import dirname, realpath
import sys
import csv
import time
import copy
import json 
from typing import Any, Tuple, List


from pprint import pprint
import torch
import numpy as np
import traceback
import numpy as np

from .CF_handler import (
    find_best_solution,
    get_layer_data_access_cost,
    find_layer_cost_fixed_solution,
    find_layer_flops_fixed_solution,
    find_layer_vm_usage_fixed_solution,
    get_minimum_solution,
)
from ..CostModel.plat_energy_costs import PlatformCostModel
from ..CostModel import common
from ..CostModel.cnn import pass_constraint_storage

#sys.path.append("..")
from ....model.common_types import LAYERTYPES, OPTYPES, Mat
from ....model.common_utils import get_network_obj, get_network_dimension, netobj_to_pyobj

sys.path.append(dirname(dirname(dirname(dirname(dirname(realpath(__file__)))))))
from settings import Settings

#sys.path.append("../..")
#from file_utils import NumpyEncoder


# Copy PLATFORM_SETTINGS in a constructor to get overridden values towards PLATFORM_SETTINGS
class SettingsINTpow(Settings):
    def __init__(self):
        self.PLATFORM_SETTINGS = copy.deepcopy(Settings.PLATFORM_SETTINGS)
        self.PLATFORM_SETTINGS["POW_TYPE"] = "INT"
        
class SettingsCONTpow(Settings):
    def __init__(self):
        self.PLATFORM_SETTINGS = copy.deepcopy(Settings.PLATFORM_SETTINGS)
        self.PLATFORM_SETTINGS["POW_TYPE"] = "CONT"


class PlatPerf:
    def __init__(self, NAS_SETTINGS, PLAT_SETTINGS):

        self.PLAT_SETTINGS = PLAT_SETTINGS        
        self.NAS_SETTINGS = NAS_SETTINGS
        self.PLAT_COST_PROFILE = self.get_cost_model()
        self.Ecf = 100          # embodied carbon footprint for whole MCU

    
    # get cost profile per platform (using the benchmark measurements)                
    def get_cost_model(self):
        if self.PLAT_SETTINGS['MCU_TYPE'] == "MSP430":
            plat_cost_profile = PlatformCostModel.PLAT_MSP430_EXTNVM
        
        # elif self.PLAT_SETTINGS['MCU_TYPE'] == "MSP432":
        #     plat_cost_profile = PlatformCostModel.PLAT_MSP432_EXTNVM
        
        else:
            sys.exit("NASBase::PlatPerf: ERROR - unsupported platform cost model")
        
        return plat_cost_profile

    

    def get_network_inasV1(self, predic_actions, model_fn_type='MODEL_FN_CONV2D'):
        network = []

        ifm_h = self.NAS_SETTINGS['IMG_SIZE']
        ifm_w = self.NAS_SETTINGS['IMG_SIZE']
        ifm_ch= self.NAS_SETTINGS['IMG_CHANNEL']
        ofm_ch= 0
        layers= self.NAS_SETTINGS['NUM_LAYERS']
        num_classes= self.NAS_SETTINGS['NUM_CLASS']

        # -- get network depending on model function type
        if model_fn_type == "MODEL_FN_CONV2D":
            # cov layers
            for i in range(layers):
                kernel_h = predic_actions[i*2]; kernel_w = predic_actions[i*2]
                filter_size = predic_actions[i*2+1]
                ofm_ch = filter_size
                ofm_h = ifm_h - kernel_h + 1; ofm_w = ifm_w - kernel_w + 1
                item = {
                    'name' : "CONV_"+str(i), 'type' : "CONV", 'stride' : 1,
                    'K' : Mat(None, filter_size, ifm_ch, kernel_h, kernel_w), # n, ch, h, w
                    'IFM' : Mat(None, 1, ifm_ch, ifm_h, ifm_w), 'OFM' : Mat(None, 1, filter_size, ofm_h, ofm_w),
                }
                network.append(item)
                ifm_h = ofm_h; ifm_w = ofm_w
                ifm_ch= filter_size

            # append global pooling layer
            item={
                'name' : "GAVGPOOL_0", 'type' : "GAVGPOOL", 'stride' : 1,
                'K' : Mat(None, ifm_ch, ifm_ch, ifm_h, ifm_w),
                'IFM' : Mat(None, 1, ifm_ch, ifm_h, ifm_w), 'OFM' : Mat(None, 1, ifm_ch, 1, 1),
            }
            network.append(item)
        
        elif model_fn_type == "MODEL_FN_CONV1D":            
            ifm_w = 1
            # cov layers
            for i in range(layers):
                kernel_h = predic_actions[i*2]; kernel_w = 1
                filter_size = predic_actions[i*2+1]
                ofm_ch = filter_size
                ofm_h = ifm_h - kernel_h + 1; ofm_w = 1
                item = {
                    'name' : "CONV_"+str(i), 'type' : "CONV", 'stride' : 1,
                    'K' : Mat(None, filter_size, ifm_ch, kernel_h, kernel_w), # n, ch, h, w
                    'IFM' : Mat(None, 1, ifm_ch, ifm_h, ifm_w), 'OFM' : Mat(None, 1, filter_size, ofm_h, ofm_w),
                }
                network.append(item)
                ifm_h = ofm_h; ifm_w = ofm_w
                ifm_ch= filter_size

            # append global pooling layer
            item={
                'name' : "GAVGPOOL_0", 'type' : "GAVGPOOL", 'stride' : 1,
                'K' : Mat(None, ifm_ch, ifm_ch, ifm_h, ifm_w),
                'IFM' : Mat(None, 1, ifm_ch, ifm_h, ifm_w), 'OFM' : Mat(None, 1, ifm_ch, 1, 1),
            }
            network.append(item)

        elif model_fn_type == "MODEL_FN_FC":    
            ifm_w = 1        
            # fc layers
            for i in range(layers):
                kernel_h = ifm_h; kernel_w = 1
                filter_size = predic_actions[i]
                ofm_ch = filter_size
                ofm_h = 1; ofm_w = 1
                item = {
                    'name' : "FC_"+str(i), 'type' : "FC", 'stride' : 1,
                    'K' : Mat(None, filter_size, ifm_ch, kernel_h, kernel_w), # n, ch, h, w
                    'IFM' : Mat(None, 1, ifm_ch, ifm_h, ifm_w), 'OFM' : Mat(None, 1, filter_size, ofm_h, ofm_w),
                }
                network.append(item)
                ifm_h = ofm_h; ifm_w = ofm_w
                ifm_ch= filter_size
        
        else:
            sys.exit("get_network:: unknown")

        
        # -- append last fc layer
        item={
            'name' : "FC_END", 'type' : "FC", 'stride' : 1,
            'K' : Mat(None, num_classes, ifm_ch, 1, 1),
            'IFM' : Mat(None, 1, ifm_ch, 1, 1), 'OFM' : Mat(None, 1, num_classes, 1, 1),
        }
        network.append(item)

        return network
        

    
    @staticmethod
    def get_inference_latency_verbose(perf_model_intpow, perf_mode_contpow, net_obj, net_config, fixed_params=None):
        
        try:
            sb_blk_choice_key = "<" + ','.join([str(c) for c in net_config]) + ">"                           
                
            # -- get perf for INT pow                                
            e2e_lat_intpow, exec_design_intpow, error_intpow = perf_model_intpow.get_inference_latency(net_obj, fixed_params=fixed_params)
            
            #if (e2e_lat_intpow == -1):
                #return error_intpow
                        
            ip_tot_npc = np.sum([l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1        # total power cycles
            ip_tot_rc = np.sum([l['L_rc_tot'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1    # total recharge time
            ip_tot_rb = np.sum([l['cost_brk']['rb'][1] * l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1
                                    
            # -- get perf for CONT pow                 
            # cont pow performance - fixed params : same as intpow        
            e2e_lat_contpow_fp, exec_design_contpow_fp, error_contpow_fp = perf_mode_contpow.get_inference_latency(net_obj, fixed_params=exec_design_intpow)
            
            # error reporting            
            if any(x == -1 for x in [ip_tot_npc, ip_tot_rc, ip_tot_rb, e2e_lat_intpow]):                
                error_net_perf = True

                int_mng_cost_proportion_cpfp = -1
            else:
                error_net_perf = False    

                # Calc IMC (vs. contpow_fp)
                active_time = (e2e_lat_intpow - ip_tot_rc)            
                int_mng_cost_proportion_cpfp = ((active_time-e2e_lat_contpow_fp)/active_time)*100
            
            subnet_latency_info = { 
                "sn_cpb"  :  net_config,                                 
                "subnet_obj": netobj_to_pyobj(net_obj),
                
                # perf int pow 
                "error_codes"  : [error_net_perf, error_intpow, error_contpow_fp],       
                "perf_e2e_intpow_lat": e2e_lat_intpow,
                "perf_exec_design_intpow": exec_design_intpow,
                #"perf_error_intpow": error_intpow,
                "ip_tot_npc": ip_tot_npc,
                
                # perf cont pow - fixed params
                "perf_e2e_contpow_fp_lat": e2e_lat_contpow_fp,
                "perf_exec_design_contpow_fp": exec_design_contpow_fp,                     
                            
                "active_time" : active_time,
                "recharge_time": ip_tot_rc,
                # proportion
                "imc_prop" :  int_mng_cost_proportion_cpfp,
            } 
                                        
        except Exception as e:
            print("here --")
            error_net_perf = True
            pprint(e)
            tb = traceback.format_exc(); print(tb); print("subnet_cpb: ", sb_blk_choice_key)
            #sys.exit()
            subnet_latency_info = None
            
        return subnet_latency_info   
        
    
    
    def get_inference_latency(self, network, fixed_params=None, power_type=None):
        #network = self.get_network(predic_actions, model_fn_type=self.NAS_SETTINGS['MODEL_FN_TYPE']) 
        #network = get_network_obj(subnet)
        error = None
        latency = 0
        network_exec_design = []        

        if not power_type:
            power_type = self.PLAT_SETTINGS['POW_TYPE']
        
        # exec and pres params not given, so have to find best sol
        if fixed_params == None:
        
            # find_best_solution找tile size/loop order/... 一般來說NAS不會給
            sols = find_best_solution(network, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE, power_type=power_type)
            if 'NET_ERROR' in sols:
                #print("NET_ERROR !!! ")
                return -1, None, sols['NET_ERROR']['error_code']
                
            
            for each_layer in network:             
                # if any one of the layers do not have a best solution, then return -1
                if sols[each_layer['name']]['best_sol'] == None: # cannot find a feasible execution design
                    return -1, None, sols[each_layer['name']]['error_code']
                else:
                    latency = latency + sols[each_layer['name']]['best_sol'][0]['Le2e']
                    network_exec_design.append({'layer' : each_layer['name'], 
                                                'alias' : each_layer['alias'],
                                                'params': sols[each_layer['name']]['best_sol'][0]['params'],
                                                'params_exec': sols[each_layer['name']]['best_sol'][0]['params_exec'],
                                                'params_pres': sols[each_layer['name']]['best_sol'][0]['params_pres'],
                                                'cost_brk': sols[each_layer['name']]['best_sol'][0]['cost_brk'],
                                                'Epc' : sols[each_layer['name']]['best_sol'][0]['Epc'],
                                                'Eu' : sols[each_layer['name']]['best_sol'][0]['Eu'],
                                                'Le2e' : sols[each_layer['name']]['best_sol'][0]['Le2e'],
                                                'npc' : sols[each_layer['name']]['best_sol'][0]['npc'][0],
                                                'L_rc_tot' : sols[each_layer['name']]['best_sol'][0]['L_rc_tot'],
                                                'vm' : sols[each_layer['name']]['best_sol'][0]['vm'],                                            
                                                })
            
            return latency, network_exec_design, error
        
        else:     
            # exec and pres params given
                   
            for lix, each_layer in enumerate(network): 
                cost_stats = find_layer_cost_fixed_solution(each_layer, fixed_params[lix], self.PLAT_SETTINGS, self.PLAT_COST_PROFILE, power_type=power_type)
                if cost_stats['Le2e'] != None:
                    latency = latency + cost_stats['Le2e']
                    network_exec_design.append({'layer' : each_layer['name'], 'alias' : each_layer['alias'],
                                                'params': fixed_params[lix]['params'],
                                                'params_exec': fixed_params[lix].get('params_exec'),
                                                'params_pres': fixed_params[lix].get('params_pres'),
                                                'cost_brk': cost_stats['cost_brk'],
                                                'Epc' : cost_stats['Epc'], 'Eu' : cost_stats['Eu'], 
                                                'Le2e' : cost_stats['Le2e'], 'npc' : cost_stats['npc'][0], 
                                                'L_rc_tot' : cost_stats['L_rc_tot'],
                                                'vm' : cost_stats['vm'],                                            
                                                })
                else:
                    return -1, None, cost_stats['reason']
            
            return latency, network_exec_design, error
            
            
            
    def get_network_flops(self, network, fixed_params=None, layer_based_cals=False):
        error = None
        network_flops = []
        network_exec_design = []        
        
        # exec and pres params not given, so have to find best sol
        if fixed_params == None and not layer_based_cals:
            sols = find_best_solution(network, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE, power_type=self.PLAT_SETTINGS['POW_TYPE'])
            for each_layer in network: 
                # if any one of the layers do not have a best solution, then return -1
                if sols[each_layer['name']]['best_sol'] == None: # cannot find a feasible execution design
                    return -1, None, sols[each_layer['name']]['error_code']
                else:
                    solution = {'layer' : each_layer['name'],
                                'alias' : each_layer['alias'],
                                'params': sols[each_layer['name']]['best_sol'][0]['params'],
                                }
                    network_exec_design.append(solution)
                layer_flops = find_layer_flops_fixed_solution(each_layer, solution, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE, power_type=self.PLAT_SETTINGS['POW_TYPE'], layer_based_cals=layer_based_cals)
                network_flops.append(layer_flops)

            return network_flops, network_exec_design, error

        else:
            # exec and pres params given

            for lix, each_layer in enumerate(network):
                solution = {'layer' : each_layer['name'],
                            'alias' : each_layer['alias'],
                            }
                if not layer_based_cals:
                    solution['params'] = fixed_params[lix]['params']
                else:
                    solution['params'] = None
                layer_flops = find_layer_flops_fixed_solution(each_layer, solution, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE, power_type=self.PLAT_SETTINGS['POW_TYPE'], layer_based_cals=layer_based_cals)
                network_exec_design.append(solution)
                network_flops.append(layer_flops)
            return network_flops, network_exec_design, error
        
        
    
    def get_network_data_access_cost(self, predic_actions, network_exec_design):
        network = self.get_network(predic_actions, model_fn_type=self.NAS_SETTINGS['MODEL_FN_TYPE']) 
        error = None
        latency = 0 
        
        net_cost = []
        for lix, each_layer in enumerate(network):
            exec_design = network_exec_design[lix]
            [rd_cost_L, wr_cost_L, rd_cost_E, wr_cost_E] = get_layer_data_access_cost(each_layer, exec_design, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE, power_type='INT')
            net_cost.append([rd_cost_L, wr_cost_L, rd_cost_E, wr_cost_E])
        return net_cost

    def get_vm_usage(self, network, fixed_params=None) -> Tuple[bool, List, List, Any]:

        error = None
        network_vm_usage = []
        network_exec_design = []        
        all_layers_fit_vm = True
        
        # exec and pres params not given, so have to find best sol
        if fixed_params == None:
            if self.PLAT_SETTINGS['LAT_E2E_REQ'] > 0:
                # need the best solution for latency constraint
                sols = find_best_solution(network, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE, power_type=self.PLAT_SETTINGS['POW_TYPE'])
            else:
                # if only memory constraints are considered, only the smallest tile size should be checked, not the best one
                sols = get_minimum_solution(network)

            if 'NET_ERROR' in sols:
                return False, [], [], sols['NET_ERROR']['error_code']

            for each_layer in network: 
                # if any one of the layers do not have a best solution, then return -1
                if sols[each_layer['name']]['error_code'] != None: # cannot find a feasible execution design
                    return False, None, None, sols[each_layer['name']]['error_code']
                else:
                    solution = {'layer' : each_layer['name'],
                                'alias' : each_layer['alias'],
                                'params': sols[each_layer['name']]['best_sol'][0]['params'],
                                }
                    network_exec_design.append(solution)
                layer_vm_usage = find_layer_vm_usage_fixed_solution(each_layer, solution, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE, power_type=self.PLAT_SETTINGS['POW_TYPE'])

                cur_layer_fit_vm = layer_vm_usage[0]
                all_layers_fit_vm = all_layers_fit_vm and cur_layer_fit_vm

                network_vm_usage.append(layer_vm_usage)

            return all_layers_fit_vm, network_vm_usage, network_exec_design, error

        else:
            # exec and pres params given

            for lix, each_layer in enumerate(network):
                solution = {'layer' : each_layer['name'],
                            'alias' : each_layer['alias'],
                            }
                solution['params'] = fixed_params[lix]['params']
                network_exec_design.append(solution)
                layer_vm_usage = find_layer_vm_usage_fixed_solution(each_layer, solution, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE, power_type=self.PLAT_SETTINGS['POW_TYPE'])

                cur_layer_fit_vm = layer_vm_usage[0]
                all_layers_fit_vm = all_layers_fit_vm and cur_layer_fit_vm

                network_vm_usage.append(layer_vm_usage)

            return all_layers_fit_vm, network_vm_usage, network_exec_design, error

    def get_nvm_usage(self, network) -> Tuple[bool, List, Any]:
        error = None
        all_layers_fit_nvm, _, network_nvm_usage = pass_constraint_storage(network, self.PLAT_SETTINGS)
        return all_layers_fit_nvm, network_nvm_usage, error

    @classmethod
    def get_latency_info(cls, net_obj, net_config, fixed_params=None):

        global_settings_intpow = SettingsINTpow() # default settings
        global_settings_contpow = SettingsCONTpow() # default settings

        performance_model_intpow = PlatPerf(global_settings_intpow.NAS_SETTINGS_GENERAL, global_settings_intpow.PLATFORM_SETTINGS)
        performance_model_contpow = PlatPerf(global_settings_contpow.NAS_SETTINGS_GENERAL, global_settings_contpow.PLATFORM_SETTINGS)

        subnet_latency_info = cls.get_inference_latency_verbose(performance_model_intpow, performance_model_contpow, net_obj, net_config, fixed_params=fixed_params)

        return subnet_latency_info

    @classmethod
    def get_imc(cls, net_obj, net_config):
        subnet_latency_info = cls.get_latency_info(net_obj, net_config)

        if subnet_latency_info:
            return subnet_latency_info['imc_prop']
        else:
            return -1
    



    # **NEW** calculate total inference times
    def get_inference_times(self, T_flash, energy):
        # need to change LT for each component from plat_cost_profile
        LT_flash = 100000

        psu_energy_capacity = 1000 #may set global
        psu_charge_cycle = 500 #may set global

        times = min(LT_flash // T_flash, (psu_energy_capacity * psu_charge_cycle) // energy)
        return times
    
    # **NEW** get carbon for CONT pow
    def get_inference_carbon(self, network, carbon_intensity, fixed_params=None, power_type=None):
        #network = self.get_network(predic_actions, model_fn_type=self.NAS_SETTINGS['MODEL_FN_TYPE']) 
        #network = get_network_obj(subnet)
        error = None
        latency, energy, T_flash = 0, 0, 0
        network_exec_design = []        

        if not power_type:
            power_type = self.PLAT_SETTINGS['POW_TYPE']
        
        # exec and pres params not given, so have to find best sol
        if fixed_params == None:
        
            # find_best_solution找tile size/loop order/... 一般來說NAS不會給
            sols = find_best_solution(network, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE, power_type=power_type)
            if 'NET_ERROR' in sols:
                #print("NET_ERROR !!! ")
                return -1, -1, -1, None, sols['NET_ERROR']['error_code']
                
            
            for each_layer in network:             
                # if any one of the layers do not have a best solution, then return -1
                if sols[each_layer['name']]['best_sol'] == None: # cannot find a feasible execution design
                    return -1, -1, -1, None, sols[each_layer['name']]['error_code']
                else:
                    latency = latency + sols[each_layer['name']]['best_sol'][0]['Le2e']
                    T_flash = T_flash + sols[each_layer['name']]['best_sol'][0]['T_flash']
                    energy = energy + sols[each_layer['name']]['best_sol'][0]['Epc'][0]
                    network_exec_design.append({'layer' : each_layer['name'],    
                                                'alias' : each_layer['alias'],
                                                'params': sols[each_layer['name']]['best_sol'][0]['params'],
                                                'params_exec': sols[each_layer['name']]['best_sol'][0]['params_exec'],
                                                'params_pres': sols[each_layer['name']]['best_sol'][0]['params_pres'],
                                                'cost_brk': sols[each_layer['name']]['best_sol'][0]['cost_brk'],
                                                'Epc' : sols[each_layer['name']]['best_sol'][0]['Epc'],
                                                'Eu' : sols[each_layer['name']]['best_sol'][0]['Eu'],
                                                'Le2e' : sols[each_layer['name']]['best_sol'][0]['Le2e'],
                                                'T_flash': sols[each_layer['name']]['best_sol'][0]['T_flash'],
                                                'npc' : sols[each_layer['name']]['best_sol'][0]['npc'][0],
                                                'L_rc_tot' : sols[each_layer['name']]['best_sol'][0]['L_rc_tot'],
                                                'vm' : sols[each_layer['name']]['best_sol'][0]['vm'],                                            
                                                })
            
            # return latency, network_exec_design, error
        
        else:     
            # exec and pres params given
            # new
            network_size = 0
            for each_layer in network: 
                if each_layer['K'].n == None:
                    continue
                network_size += each_layer['K'].n * each_layer['K'].ch * each_layer['K'].h * each_layer['K'].w

            flash_capacity = 10000 #may set global
            B_ofm = flash_capacity - (network_size * 4) #fp32

            for lix, each_layer in enumerate(network): 
                cost_stats = find_layer_cost_fixed_solution(each_layer, fixed_params[lix], self.PLAT_SETTINGS, self.PLAT_COST_PROFILE, B_ofm, power_type=power_type)
                if cost_stats['Le2e'] != None:
                    latency = latency + cost_stats['Le2e']
                    T_flash = T_flash + cost_stats['T_flash']
                    energy = energy + cost_stats['Epc'][0]
                    network_exec_design.append({'layer' : each_layer['name'], 'alias' : each_layer['alias'],
                                                'params': fixed_params[lix]['params'],
                                                'params_exec': fixed_params[lix].get('params_exec'),
                                                'params_pres': fixed_params[lix].get('params_pres'),
                                                'cost_brk': cost_stats['cost_brk'],
                                                'Epc' : cost_stats['Epc'], 'Eu' : cost_stats['Eu'], 
                                                'Le2e' : cost_stats['Le2e'], 
                                                'T_flash': sols[each_layer['name']]['best_sol'][0]['T_flash'],
                                                'npc' : cost_stats['npc'][0], 
                                                'L_rc_tot' : cost_stats['L_rc_tot'],
                                                'vm' : cost_stats['vm'],                                            
                                                })
                else:
                    return -1, -1, -1, None, cost_stats['reason']
            
        # get overall carbon
        Ocf = energy * carbon_intensity     # Operational Carbon Footprint
        inference_per_lifetime = self.get_inference_times(T_flash, energy)
        Ecf_per_inference = self.Ecf / inference_per_lifetime     # Embodied Carbon Footprint per inference

        carbon = Ocf + Ecf_per_inference
        # print(f'carbon: {carbon}')
        
        return latency, carbon, inference_per_lifetime, network_exec_design, error
    

    # test used
    @staticmethod
    def get_inference_carbon_verbose(perf_model_intpow, perf_mode_contpow, net_obj, net_config, fixed_params=None):
        try:
            sb_blk_choice_key = "<" + ','.join([str(c) for c in net_config]) + ">"                           
                
            # -- get perf for INT pow                                
            # e2e_lat_intpow, exec_design_intpow, error_intpow = perf_model_intpow.get_inference_latency(net_obj, fixed_params=fixed_params)
            
            #if (e2e_lat_intpow == -1):
                #return error_intpow
                        
            ip_tot_npc = 0
            ip_tot_rc = 0
            ip_tot_rb = 0
                                    
            # -- get perf for CONT pow                 
            # cont pow performance - fixed params : same as intpow        
            e2e_lat_contpow_fp, e2e_carbon_contpow_fp, ipl_contpow_fp, exec_design_contpow_fp, error_contpow_fp = perf_mode_contpow.get_inference_carbon(net_obj, carbon_intensity = 400, fixed_params=fixed_params)
            
            # error reporting            
            if any(x == -1 for x in [ip_tot_npc, ip_tot_rc, ip_tot_rb]):                
                error_net_perf = True

                int_mng_cost_proportion_cpfp = -1
            else:
                error_net_perf = False    

                # Calc IMC (vs. contpow_fp)
                active_time = 10            
                int_mng_cost_proportion_cpfp = ((active_time-e2e_lat_contpow_fp)/active_time)*100
            
            subnet_latency_info = { 
                "sn_cpb"  :  net_config,                                 
                "subnet_obj": netobj_to_pyobj(net_obj),
                
                # perf int pow 
                "error_codes"  : [error_net_perf, 0, error_contpow_fp],       
                "perf_e2e_intpow_lat": 0,
                "perf_exec_design_intpow": 0,
                #"perf_error_intpow": error_intpow,
                "ip_tot_npc": ip_tot_npc,
                
                # perf cont pow - fixed params
                "perf_e2e_contpow_fp_lat": e2e_lat_contpow_fp,
                "perf_e2e_contpow_fp_carbon": e2e_carbon_contpow_fp,
                "perf_e2e_contpow_fp_inf_per_lifetime": ipl_contpow_fp,
                "perf_exec_design_contpow_fp": exec_design_contpow_fp,                     
                            
                "active_time" : active_time,
                "recharge_time": ip_tot_rc,
                # proportion
                "imc_prop" :  int_mng_cost_proportion_cpfp,
            } 
                                        
        except Exception as e:
            print("here --")
            error_net_perf = True
            pprint(e)
            tb = traceback.format_exc(); print(tb); print("subnet_cpb: ", sb_blk_choice_key)
            #sys.exit()
            subnet_latency_info = None
            
        return subnet_latency_info, e2e_carbon_contpow_fp, ipl_contpow_fp


    
    @classmethod
    def get_carbon_info(cls, net_obj, net_config, fixed_params=None):

        global_settings_intpow = SettingsINTpow() # default settings
        global_settings_contpow = SettingsCONTpow() # default settings

        performance_model_intpow = PlatPerf(global_settings_intpow.NAS_SETTINGS_GENERAL, global_settings_intpow.PLATFORM_SETTINGS)
        performance_model_contpow = PlatPerf(global_settings_contpow.NAS_SETTINGS_GENERAL, global_settings_contpow.PLATFORM_SETTINGS)

        subnet_latency_info, CF, inference_per_lifetime = cls.get_inference_carbon_verbose(performance_model_intpow, performance_model_contpow, net_obj, net_config, fixed_params=fixed_params)

        return subnet_latency_info, CF, inference_per_lifetime








# convert exec design into string
def exec_design_to_string(exec_design):

    if exec_design != None:        
        s = ""
        for each_layer in exec_design:        
            #s += ','.join(['='.join(i) for i in each_layer.items()])
            s += str(each_layer)
            s += ","
        return s
    else:
        return ""

