#!/usr/bin/env python3
"""Check real SOFA constitutive softening against a restrained tetrahedron.

Use the optional SOFA Python runtime through scripts/check-sofa-damage.sh.
This is an elementary mechanics/software check, not human-tissue validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import numpy as np
from retraction.sofa_damage import TetraDamageModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/workspace/.sofa-runtime/damage-reference.json'))
    args = parser.parse_args()
    import Sofa
    import SofaRuntime
    errors = []

    class Diagnostics(Sofa.Helper.MessageHandler):
        def process(self, message):
            if message['type'] in ('Error', 'Fatal'):
                errors.append(str(message['message']))

    plugins = ['AnimationLoop', 'StateContainer', 'Topology.Container.Dynamic',
               'SolidMechanics.FEM.Elastic', 'ODESolver.Backward', 'LinearSolver.Iterative',
               'Constraint.Projective', 'MechanicalLoad', 'Mass']
    for plugin in plugins:
        SofaRuntime.importPlugin('Sofa.Component.' + plugin)
    points = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    tetrahedra = np.array([[0,1,2,3]])
    base_modulus, applied_force = 1000., .1
    root = Sofa.Core.Node('damage_reference')
    root.dt = .01
    root.gravity = [0,0,0]
    root.addObject('DefaultAnimationLoop')
    root.addObject('EulerImplicitSolver', rayleighMass=5, rayleighStiffness=.1)
    root.addObject('CGLinearSolver', iterations=100, tolerance=1e-12, threshold=1e-12)
    mechanical = root.addObject('MechanicalObject', name='tissue', position=points.tolist(), template='Vec3d')
    root.addObject('TetrahedronSetTopologyContainer', tetrahedra=tetrahedra.tolist())
    # Vector-valued stock FEM supports live per-element material updates.
    fem = root.addObject('TetrahedronFEMForceField', method='large', youngModulus=base_modulus, poissonRatio=0)
    root.addObject('UniformMass', totalMass=1)
    root.addObject('FixedProjectiveConstraint', indices=[0,1,2])
    root.addObject('ConstantForceField', indices=[3], forces=[[0,0,applied_force]])
    with Diagnostics():
        Sofa.Simulation.init(root)
        try:
            for _ in range(1500):
                Sofa.Simulation.animate(root, .01)
            intact_displacement = float(mechanical.position.array()[3,2] - 1)
            injury = TetraDamageModel(points, tetrahedra, base_young_modulus=base_modulus)
            preconditioning = points.copy()
            preconditioning[3,2] = 1.5
            injury.update(preconditioning, .1)
            # The forced extension is a controlled injury preconditioning, not
            # an inferred tool-force threshold or an asserted rupture point.
            injury.apply_to_sofa(fem)
            for _ in range(1500):
                Sofa.Simulation.animate(root, .01)
            injured_displacement = float(mechanical.position.array()[3,2] - 1)
            volume = 1 / 6
            # nu=0, base nodes fixed, affine uniaxial strain: k=E*V/L^2.
            expected_intact = applied_force / (base_modulus * volume)
            effective_modulus = float(injury.young_modulus[0])
            expected_injured = applied_force / (effective_modulus * volume)
            intact_error = abs(intact_displacement / expected_intact - 1)
            injured_error = abs(injured_displacement / expected_injured - 1)
            assert not errors, errors
            assert intact_error < .01 and injured_error < .01, (intact_error, injured_error)
            assert injured_displacement > intact_displacement * 1.5
            report = {
                'status': 'passed', 'scope': 'one-tetrahedron constitutive/software reference only',
                'sofa_errors': errors, 'force_field': 'TetrahedronFEMForceField',
                'force_scene_units': applied_force, 'reference_volume_scene_units_cubed': volume,
                'base_young_modulus_scene_units': base_modulus,
                'injured_young_modulus_scene_units': effective_modulus,
                'irreversible_damage': float(injury.damage[0]),
                'intact_displacement_scene_units': intact_displacement,
                'injured_displacement_scene_units': injured_displacement,
                'expected_intact_displacement': expected_intact,
                'expected_injured_displacement': expected_injured,
                'intact_relative_error': intact_error, 'injured_relative_error': injured_error,
                'limitations': ['Prescribed injury preconditioning, not calibrated tissue damage.',
                                'No cutting/remeshing, vessel flow, or anatomical validation.',
                                'A simple restrained element does not validate a full organ scene.'],
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(json.dumps(report, indent=2, allow_nan=False))
        finally:
            Sofa.Simulation.unload(root)


if __name__ == '__main__':
    main()
