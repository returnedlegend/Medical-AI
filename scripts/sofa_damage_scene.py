"""Explicit experimental material variant of the corrected LapGym scene.

Select stock vector-valued corotational FEM before scene initialization so each
organ element can soften irreversibly. Upstream files remain unchanged. This is
a distinct model variant, not the original FastTetrahedral solver or a validated
rupture/contact/thermal model.
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('retraction_corrected_scene', Path(__file__).with_name('sofa_scene.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def createScene(root_node, **kwargs):
    result = base.createScene(root_node, **kwargs)
    tissue = result['gallbladder']
    previous = tissue.deformation_force_field
    tissue.node.removeObject(previous)
    tissue.deformation_force_field = tissue.node.addObject(
        'TetrahedronFEMForceField', name='DamageAwareCorotatedFEM', method='large',
        youngModulus=[float(tissue.material.young_modulus)],
        poissonRatio=[float(tissue.material.poisson_ratio)],
    )
    return result
