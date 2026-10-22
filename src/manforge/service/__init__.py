"""Web-independent contract layer for service consumers (APIs, job workers).

Exposes exactly what a deployment needs: a name→class registry with
serializable descriptors, and a run function with plain inputs/outputs.
"""

from manforge.core.material import ParamMeta
from manforge.service.registry import ModelRegistry, ModelDescriptor, default_registry
from manforge.service.runner import run_simulation, SimulationOutput

__all__ = [
    "ParamMeta",
    "ModelRegistry",
    "ModelDescriptor",
    "default_registry",
    "run_simulation",
    "SimulationOutput",
]
