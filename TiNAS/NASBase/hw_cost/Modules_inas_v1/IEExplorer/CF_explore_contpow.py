import sys, os
from pprint import pprint
import numpy as np
from time import perf_counter 
import inspect
import warnings
import math


# local import
from ..CostModel import cnn as cnn
from ..CostModel import common as common 
from ..CostModel.capacitor import cap_energy, cal_cap_recharge_time_custom
from ....model.common_types import Mat
from ....file_utils import json_dump, json_load



# no constraints
def get_energy_all_params_contpow(layer, plat_settings, plat_cost_profile):

    R = layer['OFM'].h; C = layer['OFM'].w; M = layer['OFM'].ch; N = layer['IFM'].ch
    H = layer['IFM'].h; W = layer['IFM'].w
    Kh = layer['K'].h; Kw = layer['K'].w
    stride = layer['stride']

    # get search space
    tr_lst, tc_lst, tm_lst, tn_lst = common.filter_legal_tilesizes(None, None, None, None, H, W, R, C, M, N, layer_type = layer['type'], op_type=layer['optype'])
    inter_tile_order = ['reuse_I', 'reuse_W', 'reuse_O']

    results = []
    search_space_size = 0

    # -- all tile size permutations
    for Tr in tr_lst:
        for Tc in tc_lst:
            Tri, Tci = common._calc_conv_ifm_tile_size(Tr, Tc, Kh, Kw, stride=stride)
            for Tm in tm_lst:
                for Tn in tn_lst:
                    # -- all loop orders
                    for inter_lo in inter_tile_order:                        
                        params_exec = {'tile_size': [Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn], 'inter_lo': inter_lo}                                                
                        search_space_size+=1
                        #lay_E, lay_L, lay_T_cpu, lay_T_mem, lay_T_disk = cnn.est_cost_layer_contpow(layer, params_exec, plat_settings, plat_cost_profile); common.check_infnan([lay_E, lay_L])                        
                        lay_E, lay_L = cnn.est_cost_layer_contpow(layer, params_exec, plat_settings, plat_cost_profile); common.check_infnan([lay_E, lay_L])                        
                        B_in, B_w, B_out = common._vm_buff_size(Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn, inter_lo, S=1, layer_type = layer['type'], op_type=layer['optype']); common.check_infnan([B_in, B_w, B_out])  
                        total_vm_req = (B_in + B_w + B_out) * plat_settings['DATA_SZ']

                        results.append({
                            'params' : common.to_string_params_all(params_exec, None),                                                                                
                            'lay_E': lay_E, "lay_L": lay_L,                                
                            'vm' : total_vm_req
                        })             
    print("Layer [%s] eval. complete" % (layer['name']))

    return results

# NSGA functions
def dominates(item_a, item_b):
    # a dominates b
    if item_a['T_flash'] <= item_b['T_flash'] and item_a['Epc'][0] <= item_b['Epc'][0]:     
        return 1
    # b dominates a
    elif item_a['T_flash'] >= item_b['T_flash'] and item_a['Epc'][0] >= item_b['Epc'][0]:
        return -1
    return 0

# no need to sort here, just pick the set that don't get dominated
def pareto_front(population):
    S = []  # S[p] = set of solutions; the solution p dominates.
    N = []  # N[p] = domination count; the number of solutions which dominate p.

    # assign ranks to each solution
    for p, p_idx in zip(population, range(len(population))):
        # initialize
        S.append(set())
        N.append(0)
        for q, q_idx in zip(population, range(len(population))):
            if p_idx == q_idx: continue
            judge = dominates(p, q)
            if judge == 1:
                # p dominates q
                S[p_idx].add(q_idx)
            elif judge == -1:
                # q dominates p
                N[p_idx] += 1

    best_population = [population[index] for index, rank in enumerate(N) if rank == min(N)]
    return best_population, min(N)

#Calculate T_flash
def cal_lay_T_flash(B_ofm, R, C, M, N, Tr, Tc, Tm, Tn):
    FWS = 8 #may get from global
    N_write = (R / Tr) * (C / Tc) * (M / Tm) * (N / Tn)
    b_out = Tr * Tc * Tm
    N_fws = math.ceil((N_write * b_out) / FWS)
    T_flash = math.ceil(N_fws / math.floor(B_ofm / FWS))
    return T_flash

# with constraints
def explore_full_param_sweep_contpow(layer, plat_settings, plat_cost_profile, B_ofm, report_topN=0.5, best_selection='first'):
    
    R = layer['OFM'].h; C = layer['OFM'].w; M = layer['OFM'].ch; N = layer['IFM'].ch
    H = layer['IFM'].h; W = layer['IFM'].w
    Kh = layer['K'].h; Kw = layer['K'].w
    stride = layer['stride']

    # get search space
    tr_lst, tc_lst, tm_lst, tn_lst = common.filter_legal_tilesizes(None, None, None, None, H, W, R, C, M, N, layer_type = layer['type'], op_type=layer['optype'])
    inter_tile_order = common.filter_legal_reuseschems(layer_type = layer['type'], op_type=layer['optype'])
    
    
    #print(layer)
    #pprint(tr_lst); pprint(tc_lst); pprint(tm_lst); pprint(tn_lst)
    #sys.exit()

    result_pass = []
    result_fail = []    
    search_space_size = 0

    # -- all tile size permutations
    for Tr in tr_lst:
        for Tc in tc_lst:
            Tri, Tci = common._calc_conv_ifm_tile_size(Tr, Tc, Kh, Kw, stride = layer['stride'], layer_type = layer['type'])
            for Tm in tm_lst:
                for Tn in tn_lst:

                    # -- all loop orders
                    for inter_lo in inter_tile_order:                        
                            # print (R, C, M, N, Tr, Tc, Tm, Tn, inter_lo)
                        
                            search_space_size+=1

                            # -- check if passes the initial constraints ?                                                        
                            params_exec = {'tile_size': [Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn], 'inter_lo': inter_lo, 'full_feature_map': [R, C, M, N], 'B_ofm': B_ofm}
                            params_pres = {'backup_batch_size': 1} # using S=1 for constant power
                                                      
                            res_cons_c0 = cnn.pass_constraint_spatial(layer, plat_settings, params_exec, params_pres); common.check_infnan(res_cons_c0)  
                            npc, npc_n0, npc_ngt0 = cnn.est_npc_layer_intpow(layer, params_exec, params_pres) # estmate the number of power cycles

                            if (res_cons_c0[0]):
                                lay_E, lay_L, cost_breakdown = cnn.est_cost_layer_contpow(layer, params_exec, plat_settings, plat_cost_profile); common.check_infnan([lay_E, lay_L]) 
                                T_flash = cal_lay_T_flash(B_ofm, R, C, M, N, Tr, Tc, Tm, Tn)    
                                #rb_E, rb_L = cnn.est_cost_onetime_reboot() # one time reboot cost (only once per network model) 
                                                           
                                # result_pass.append({
                                #     'params' : common.to_string_params_all(params_exec, None),                                                                                
                                #     'lay_E': lay_E, "lay_L": lay_L,                                                                    
                                #     'vm' : res_cons_c0[2]
                                # })  
                                result_pass.append({
                                        'params' : common.to_string_params_all(params_exec, params_pres),                                                                                
                                        'params_exec': params_exec,
                                        'params_pres': params_pres,
                                        'Epc': [lay_E, lay_E], 'Le2e': lay_L,
                                        'T_flash': T_flash,
                                        'Lpc': [None, None], 'Lrc': [None, None], 'Eu' : None, 'npc' : [npc, npc_n0, npc_ngt0],
                                        'L_rc_tot' : None,
                                        'cost_brk': cost_breakdown,
                                        'vm' : res_cons_c0[2]
                                })

                            else:
                                result_fail.append({
                                    'params' : common.to_string_params_all(params_exec, None),                                                                                
                                    'reason': 'FAILED_C0',
                                    'Epc': [lay_E, lay_E], 'Le2e': lay_L,
                                    'T_flash': T_flash,
                                    'npc' : [npc, npc_n0, npc_ngt0],     
                                    'cost_brk': None, 'L_rc_tot' : None,                                                      
                                    'vm' : res_cons_c0[2]
                                })                              
                                
    
    #print("Layer [%s] eval. complete. PASS= %d/%d = %.1f" % (layer['name'], len(result_pass), search_space_size, (len(result_pass)/search_space_size)*100.0 ))
    # print(f'execution design length: {len(result_pass)}')
    # print(f'best execution designs: {len(best_sols)} level: {level}')


    if (len(result_pass) > 0):
        # -- find best solution (lowest E2E latency)  
        min_lat = np.min([r['Le2e'] for r in result_pass])
        all_best_sols = [r for r in result_pass if r['Le2e'] == min_lat]
        best_solution = sorted(all_best_sols, key = lambda i: i['vm'])

        sorted_result_pass = sorted(result_pass, key = lambda i: i['Le2e'])
        
        # top N% solutions
        nperc = report_topN
        nnum = int(np.ceil(nperc* len(sorted_result_pass)))
        pass_topN = sorted_result_pass[0:nnum]

        # sort and save the top N% failed solutions    
        sorted_results_fail_c0 = sorted([f for f in result_fail if f['reason'] == 'FAILED_C0'], key = lambda i: i['vm'])
        nnum = int(np.ceil(nperc * len(sorted_results_fail_c0)))
        sorted_results_fail_c0 = sorted_results_fail_c0[0:nnum]
        sorted_results_fail_c1 = []

        return best_solution, result_pass, sorted_results_fail_c0, sorted_results_fail_c1, pass_topN
    
    else:

        warnings.warn("WARNING: Layer [%s] - unable to find a solution" % (layer['name']))
        best_solution = None
        result_pass = []
        sorted_results_fail_c0 = []
        sorted_results_fail_c1 = []
        pass_topN = []
        
        return best_solution, result_pass, sorted_results_fail_c0, sorted_results_fail_c1, pass_topN


# get end-to-end latency for a specific fixed param
def get_le2e_fixed_params_contpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile, B_ofm):
    result = {}
    res_cons_c0 = cnn.pass_constraint_spatial(layer, plat_settings, params_exec, params_pres)
    R = layer['OFM'].h; C = layer['OFM'].w; M = layer['OFM'].ch; N = layer['IFM'].ch
    Tr, Tc, Tm, Tn = params_exec['tile_size'][4:]
    if (res_cons_c0[0]):
        lay_E, lay_L, cost_breakdown = cnn.est_cost_layer_contpow(layer, params_exec, plat_settings, plat_cost_profile)
        T_flash = cal_lay_T_flash(B_ofm, R, C, M, N, Tr, Tc, Tm, Tn)
        result = {
            'params' : common.to_string_params_all(params_exec, params_pres),                                                                                
            'Epc': [lay_E, lay_E], 'Le2e': lay_L, 
            'T_flash':  T_flash,
            'Lpc': [None, None], 'Lrc': [None, None], 'Eu' : None, 'npc' : [1, 1, 0],
            'L_rc_tot' : None,
            'cost_brk': cost_breakdown,
            'vm' : res_cons_c0[2]
        }
    else:
        result = {
                'params' : common.to_string_params_all(params_exec, params_pres),                                        
                'reason': 'FAILED_C0',
                'Epc': [None, None], 'Le2e': None, 'Lpc': [None, None],
                'T_flash':  None,
                'npc' : [None, None, None],
                'L_rc_tot': None,
                'cost_brk': None,
                'vm' : res_cons_c0[2]
            }

    return result




def get_flops_fixed_params_contpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile, layer_based_cals):
    total_flops, total_macs = cnn.est_FLOPS_cost_layer(layer, params_exec, params_pres, layer_based_cals)
    return total_flops, total_macs


def get_vm_usage_fixed_params_contpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile):
    vm_usage = cnn.pass_constraint_spatial(layer, plat_settings, params_exec, params_pres)
    return vm_usage
