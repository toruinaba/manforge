"""Web-independent contract layer for service consumers (APIs, job workers).

Exposes exactly what a deployment needs: a name→model registry with
serializable descriptors, and a run function with plain inputs/outputs.
"""

from returnmap.service.registry import ModelRegistry, ModelDescriptor, default_registry
from returnmap.service.runner import run_simulation, SimulationOutput

__all__ = [
    "ModelRegistry",
    "ModelDescriptor",
    "default_registry",
    "run_simulation",
    "SimulationOutput",
]
