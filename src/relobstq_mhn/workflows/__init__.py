"""End-to-end, table-producing scientific workflows.

Workflow modules compose the pure method functions.  They do not import a
plotting library and always return their result tables to the caller.
"""

from .cross_sectional import CrossSectionalConfig, run_cross_sectional_cohort
from .longitudinal import LongitudinalConfig, evaluate_longitudinal_pairs
from .longitudinal_preparation import LongitudinalPreparationConfig, prepare_longitudinal_pairs
from .preparation import CrossSectionalPreparationConfig, prepare_cross_sectional_cohort
from .simulation import DwellGradientConfig, run_dwell_gradient
from .topology_robustness import TopologyRobustnessConfig, run_topology_robustness

__all__ = [
    "CrossSectionalConfig",
    "CrossSectionalPreparationConfig",
    "DwellGradientConfig",
    "LongitudinalConfig",
    "LongitudinalPreparationConfig",
    "TopologyRobustnessConfig",
    "evaluate_longitudinal_pairs",
    "prepare_longitudinal_pairs",
    "prepare_cross_sectional_cohort",
    "run_cross_sectional_cohort",
    "run_dwell_gradient",
    "run_topology_robustness",
]
