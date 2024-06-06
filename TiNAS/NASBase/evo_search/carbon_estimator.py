#import yaml
#from ofa.utils import download_url

import sys, os
from pprint import pprint
import torch
import numpy as np
import traceback
import inspect
import math

#sys.path.append("..")
from NASBase.model.mnas_arch import MNASSuperNet, MNASSubNet
from NASBase.model.common_utils import (
    get_network_dimension, get_network_obj, get_subnet, parametric_supernet_blk_choices, blkchoices_to_blkchoices_ixs, get_subnet_from_config,
    get_dummy_net_input_tensor,
)
from NASBase.hw_cost.Modules_inas_v1.IEExplorer.CF_plat_perf import PlatPerf

#sys.path.append("../..")
from NASBase import utils as utils
from NASBase import file_utils as file_utils
from settings import Settings, arg_parser, load_settings



CI=100


class CarbonEstimator(object):

    def __init__(self, global_settings):
        self.global_settings = global_settings
        self.dataset = global_settings.NAS_SETTINGS_GENERAL['DATASET']
        
    
    def _get_net_perf(self, subnet_obj, settings_inst, fixed_params=None, power_type='CONT'):        
        #print("get_network_perf::Enter")
        performance_model = PlatPerf(settings_inst.NAS_SETTINGS_GENERAL, settings_inst.PLATFORM_SETTINGS)
        if power_type == 'INT':
            time_performance, exec_design, error = performance_model.get_inference_latency(subnet_obj, fixed_params=fixed_params, power_type=power_type)
            return time_performance, exec_design, error
        else:
            time_performance, carbon_performance, inf_per_lt, exec_design, error = performance_model.get_inference_carbon(subnet_obj, carbon_intensity=CI,fixed_params=fixed_params, power_type=power_type)
            return time_performance, carbon_performance, inf_per_lt, exec_design, error
        #subnet_cust_net_obj = 
        return time_performance, carbon_performance, exec_design, error   
        
    
    # get both latency under cont/int pow
    # might just need CONT pow
    def predict_network_carbon(self, net_config, net_obj, input_size, input_ch):
        #print("predict_network_latency:: Enter subnet_config {}".format(net_config))
        
            
        net_input = get_dummy_net_input_tensor(self.global_settings, input_size)
                    
        try:     
            sb_blk_choice_key = "<" + ','.join([str(c) for c in net_config]) + ">"                           
            
            # -- get subnet costs
            subnet_dims = get_network_dimension(net_obj, input_tensor = net_input)         
            subnet_obj = get_network_obj(subnet_dims)       
                        
            # get perf for INT pow         
            # Delete?   
            # e2e_lat_intpow, exec_design_intpow, error_intpow = self._get_net_perf(subnet_obj, self.global_settings, power_type='INT')
                        
            ip_tot_npc = 0      # total power cycles
            ip_tot_rc = 0    # total recharge time
            ip_tot_rb = 0
                                    
            # get perf for CONT pow                 
            # cont pow performance - fixed params : same as intpow
            e2e_lat_contpow_fp, e2e_carbon_contpow_fp, inf_per_lt, exec_design_contpow_fp, error_contpow_fp = self._get_net_perf(subnet_obj, self.global_settings, power_type='CONT')                        

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
                
                # perf int pow 
                "error_net_perf"  : error_net_perf,          
                "perf_e2e_intpow_lat": None,
                "perf_exec_design_intpow": None,
                "perf_error_intpow": None,
                
                # perf cont pow - fixed params
                "perf_e2e_contpow_fp_lat": e2e_lat_contpow_fp,
                "perf_e2e_contpow_fp_carbon": e2e_carbon_contpow_fp,
                "perf_inf_per_lt_e2e_contpow_fp": inf_per_lt,
                "perf_exec_design_contpow_fp": exec_design_contpow_fp,
                "perf_error_contpow_fp": error_contpow_fp,          
                            
                # proportion
                "imc_prop" :  int_mng_cost_proportion_cpfp,
            }        
                                            
        except Exception as e:            
            error_net_perf = True
            pprint(e)
            tb = traceback.format_exc(); print(tb); print("subnet_cpb: ", sb_blk_choice_key)                        
        
        return subnet_latency_info

    
    
    
    def predict_carbon_efficiency_and_imc(self, net_config, supernet_config, input_ch):
        width_multiplier, input_resolution = supernet_config

        pow_type = self.global_settings.PLATFORM_SETTINGS['POW_TYPE']
        # supernet = get_supernet(self.global_settings, self.dataset, load_state=False)        
        blk_choices = parametric_supernet_blk_choices(global_settings=self.global_settings)
        # get_subnet() wants block choice *indices*
        subnet_choice_per_blk = blkchoices_to_blkchoices_ixs(blk_choices, net_config)
        subnet = get_subnet(self.global_settings, self.dataset, blk_choices, subnet_choice_per_blk, -1, width_multiplier, input_resolution)
        net_latency_info = self.predict_network_carbon(net_config, subnet, input_resolution, input_ch)
        
        if pow_type == 'CONT':
            lat = net_latency_info['perf_e2e_contpow_fp_lat']
            carbon = net_latency_info['perf_e2e_contpow_fp_carbon']
            inf_per_lt = net_latency_info['perf_inf_per_lt_e2e_contpow_fp']
            imc = net_latency_info['imc_prop']
            return lat, carbon, inf_per_lt, imc
        elif pow_type == 'INT':
            lat = net_latency_info['perf_e2e_intpow_lat']
            imc = net_latency_info['imc_prop']
            return lat, imc
        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown pow_type, " + pow_type)

        imc = net_latency_info['imc_prop']
        
        return lat, imc
        
    def predict_nvm_usage(self, net_config, supernet_config):
        performance_model = PlatPerf(self.global_settings.NAS_SETTINGS_GENERAL, self.global_settings.PLATFORM_SETTINGS)

        subnet_obj, _ = get_subnet_from_config(self.global_settings, self.dataset, net_config, supernet_config)

        all_layers_fit_nvm, _, _ = performance_model.get_nvm_usage(subnet_obj)

        return all_layers_fit_nvm

    def predict_network_carbon_and_latency_verbose(self, subnet_choice_per_blk, supernet_config):
        performance_model = PlatPerf(self.global_settings.NAS_SETTINGS_GENERAL, self.global_settings.PLATFORM_SETTINGS)

        subnet_obj, _ = get_subnet_from_config(self.global_settings, self.dataset, subnet_choice_per_blk, supernet_config)

        subnet_latency_info, CF, inference_per_lifetime = performance_model.get_carbon_info(subnet_obj, subnet_choice_per_blk)

        return subnet_latency_info


    


