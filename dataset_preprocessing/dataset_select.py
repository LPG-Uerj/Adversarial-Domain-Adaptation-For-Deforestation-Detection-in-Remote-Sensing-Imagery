from . import Amazon_PA, Amazon_RO, Cerrado_MA

import sys
sys.path.append('..')
from parameters.dataset_parameters import PA, MA, RO


def select_domain(domain_str: str):
    if domain_str == "PA":
        domain = Amazon_PA.AM_PA(PA)
        domain_params = PA
    elif domain_str == "RO":
        domain = Amazon_RO.AM_RO(RO)
        domain_params = RO
    elif domain_str == "MA":
        domain = Cerrado_MA.CE_MA(MA)
        domain_params = MA
        
    return domain, domain_params