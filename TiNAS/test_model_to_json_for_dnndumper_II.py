from NASBase import file_utils
from NASBase.model.common_utils import get_subnet_from_config, netobj_to_pyobj
from NASBase.hw_cost.Modules_inas_v1.IEExplorer.plat_perf import PlatPerf
from NASBase.hw_cost.Modules_inas_v1.CostModel import common
from settings import Settings

# Large/small in terms of IMO
MODEL_CONFIGS = {
    # 'VERY_LARGE_SUBNET_CONFIG_CIFAR10': [[6, 1, 3, True], [6, 1, 3, True], [6, 1, 3, True], [6, 1, 3, True]], # (~58% IMO)
    # 'LARGE_SUBNET_CONFIG_CIFAR10':  [[3, 3, 2, True], [3, 3, 2, True], [3, 3, 2, True], [3, 3, 2, True]], # (~48% IMO)
    'MEDIUM_SUBNET_CONFIG_CIFAR10': [[6, 7, 3, True], [6, 7, 3, True], [6, 7, 3, True], [6, 7, 3, True]], # (~33% IMO)
    # 'SMALL_SUBNET_CONFIG_CIFAR10': [[1, 7, 1, False], [1, 7, 1, False], [1, 7, 1, False], [1, 7, 1, False]], # (~25% IMO)
}

REPEAT_SMALL_LAYERS = True

def exec_design_postprocessing(perf_exec_design_intpow, perf_exec_design_contpow_fp, old_perf_exec_design_contpow_fp=None):
    for layer in perf_exec_design_intpow:
        Tr, Tc, Tm, Tn, reuse_sch, S = common.string_to_params_all(layer['params'])
        layer['params'] = {'Tr': Tr, 'Tc': Tc, 'Tm': Tm, 'Tn': Tn, 'reuse_sch': reuse_sch, 'S': S}

    for idx, layer in enumerate(perf_exec_design_contpow_fp):
        Tr, Tc, Tm, Tn, reuse_sch, S = common.string_to_params_all(layer['params'])
        layer['params'] = {'Tr': Tr, 'Tc': Tc, 'Tm': Tm, 'Tn': Tn, 'reuse_sch': reuse_sch, 'S': S}

        repeat = None
        if old_perf_exec_design_contpow_fp:
            repeat = old_perf_exec_design_contpow_fp[idx].get('repeat')
        if not repeat:
            repeat = 1
            if REPEAT_SMALL_LAYERS:
                Le2e = layer['Le2e']
                while Le2e < 10:
                    repeat *= 2
                    Le2e *= 2
        layer['repeat'] = repeat

def dump_for_model_config(global_settings, subnet_config, subnet_config_name):
    CCAP = global_settings.PLATFORM_SETTINGS['CCAP']

    print(f'Dumping for {subnet_config_name} with {CCAP*1000}mF')

    subnet_obj, _ = get_subnet_from_config(global_settings, 'CIFAR10', subnet_config, supernet_config=(1.0, 32))

    performance_model = PlatPerf(global_settings.NAS_SETTINGS_GENERAL, global_settings.PLATFORM_SETTINGS)

    subnet_latency_info, carbon = performance_model.get_carbon_info(subnet_obj, subnet_config)

    perf_exec_design_intpow = subnet_latency_info["perf_exec_design_intpow"]
    perf_exec_design_contpow_fp = subnet_latency_info["perf_exec_design_contpow_fp"]

    exec_design_postprocessing(perf_exec_design_intpow, perf_exec_design_contpow_fp)

    best_solution = [
        [],
        {
            'subnet_config': subnet_config,
            'subnet_obj': netobj_to_pyobj(subnet_obj),
            'perf_exec_design_intpow': perf_exec_design_intpow,
            'perf_exec_design_contpow_fp': perf_exec_design_contpow_fp,
            'imc_prop': subnet_latency_info["imc_prop"],
            'active_time': subnet_latency_info["active_time"],
            'recharge_time': subnet_latency_info["recharge_time"],
            'total_power_cycle': subnet_latency_info["ip_tot_npc"],
        },
    ]

    print(f'Solution carbon footprint: {carbon}')
    file_utils.json_dump(f'best_solution-{subnet_config_name}-{CCAP*1000}mF.json', best_solution)

def main():
    global_settings = Settings()
    global_settings.NAS_SETTINGS_PER_DATASET['CIFAR10']['NUM_BLOCKS'] = 1

    for subnet_config_name, subnet_config in MODEL_CONFIGS.items():
        dump_for_model_config(global_settings, subnet_config, subnet_config_name)

if __name__ == '__main__':
    main()
