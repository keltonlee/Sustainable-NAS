from settings import Settings, SSOptPolicy
from NASBase.ss_optimization.ss_opt_flops import ss_optimization_by_flops
from NASBase.ss_optimization.ss_opt_imc import ss_optimization_by_imc

def ss_optimization(global_settings: Settings, dataset, supernet_choices, supernet_block_choices):

    ss_opt_policy = global_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_POLICY']

    if ss_opt_policy == SSOptPolicy.FLOPS:
        return ss_optimization_by_flops(global_settings, dataset, supernet_choices, supernet_block_choices)
    elif ss_opt_policy == SSOptPolicy.IMC:
        return ss_optimization_by_imc(global_settings, dataset, supernet_choices, supernet_block_choices)
