#!/usr/bin/env python3
"""Upstream SOFA mesh sensitivity checks; not convergence proofs.

Uses one SOFA scene per subprocess, preserves upstream assets, and compares
original-node displacement/common-element incremental strain, never raw summed
internal force. Seed42 is development data; no reserved RL patient is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def createScene(root_node, **kwargs):
    """Project-owned scene override: change only gallbladder volume mesh path."""
    from sofa_env.scenes.grasp_lift_touch import scene_description as upstream
    original = upstream.Gallbladder
    selected = Path(os.environ['RETRACTION_RESOLUTION_MESH'])
    def gallbladder_factory(*args, **settings):
        settings['volume_mesh_path'] = selected
        return original(*args, **settings)
    upstream.Gallbladder = gallbladder_factory
    try:
        result = upstream.createScene(root_node, **kwargs)
    finally:
        upstream.Gallbladder = original
    for name in ('liver', 'gallbladder'):
        result[name].topology_container.triangles.value = []
        result[name].topology_container.edges.value = []
    return result


def strains(reference, current, tets):
    points = reference[tets]
    initial = np.transpose(points[:, 1:] - points[:, :1], (0, 2, 1))
    points = current[tets]
    deformed = np.transpose(points[:, 1:] - points[:, :1], (0, 2, 1))
    gradient = deformed @ np.linalg.inv(initial)
    principal = np.linalg.svd(gradient, compute_uv=False)
    tensile = np.maximum(0, principal.max(axis=1) - 1)
    volume = np.abs(np.linalg.det(initial)) / 6
    return {'maximum_principal_tensile_strain': float(tensile.max()),
            'volume_weighted_mean_principal_tensile_strain': float(np.sum(tensile * volume) / volume.sum()),
            'inverted_elements': int((np.linalg.det(gradient) <= 0).sum())}


def worker(args):
    import Sofa
    import SofaRuntime  # noqa: F401
    from PIL import Image
    from sofa_env.scenes.grasp_lift_touch.grasp_lift_touch_env import GraspLiftTouchEnv, ObservationType, RenderMode, Phase
    from refine_sofa_mesh import read_tetra_mesh
    diagnostics = {'errors': [], 'warnings': []}
    class Handler(Sofa.Helper.MessageHandler):
        def process(self, message):
            kind = message['type']
            if kind in ('Error', 'Fatal'):
                diagnostics['errors'].append({key: str(message[key]) for key in ('sender','message')})
            elif kind == 'Warning' and len(diagnostics['warnings']) < 40:
                diagnostics['warnings'].append({key: str(message[key]) for key in ('sender','message')})
    handler = Handler(); handler.__enter__()
    os.environ['RETRACTION_RESOLUTION_MESH'] = str(args.mesh)
    original_points, original_tets = read_tetra_mesh(args.original_mesh)
    start = time.monotonic()
    env = GraspLiftTouchEnv(scene_path=Path(__file__), observation_type=ObservationType.RGB,
                           render_mode=RenderMode.HEADLESS, image_shape=(128,128), time_step=0.01,
                           settle_steps=10, randomize_point_of_interest=False,
                           limit_grasping_attempts_in_reset=True, start_in_phase=Phase.LIFT)
    try:
        obs, _ = env.reset(seed=42)
        body = env.gallbladder
        rest = body.mechanical_object.rest_position.array().copy()
        initial = body.mechanical_object.position.array().copy()
        collision_rest = body.collision_model_node.MechanicalObject.rest_position.array().copy()
        all_tets = body.topology_container.tetrahedra.array().copy()
        initial_pose = np.asarray(env.gripper.get_pose()).copy()
        if diagnostics['errors']:
            raise RuntimeError('SOFA initialization reported errors')
        steps = []
        series = []
        for index in range(args.steps):
            action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            action[3] = -0.05  # same bounded normalized insertion velocity
            obs, reward, terminated, truncated, info = env.step(action)
            current = body.mechanical_object.position.array().copy()
            if not np.isfinite(current).all():
                raise RuntimeError('Nonfinite mesh positions')
            original = current[:len(original_points)]
            delta = original - initial[:len(original_points)]
            common = strains(initial[:len(original_points)], original, original_tets)
            native = strains(initial, current, all_tets)
            if native['inverted_elements'] or common['inverted_elements']:
                raise RuntimeError('Element inversion encountered')
            steps.append({'step': index+1, 'time_s': (index+1)*env.time_step*env.frame_skip,
                          'grasp_established': bool(env.gripper.grasp_established),
                          'terminated': bool(terminated), 'truncated': bool(truncated),
                          'mean_original_vertex_displacement': float(np.linalg.norm(delta,axis=1).mean()),
                          'max_original_vertex_displacement': float(np.linalg.norm(delta,axis=1).max()),
                          'common_coarse_tetra_incremental_strain': common,
                          'native_tetra_incremental_strain': native})
            series.append(original)
            if terminated or truncated:
                break
        if diagnostics['errors']:
            raise RuntimeError('SOFA trajectory reported errors')
        if len(steps) != args.steps or not all(row['grasp_established'] for row in steps):
            raise RuntimeError('Bounded trajectory did not retain the requested grasp/step count')
        image = np.asarray(env.render())
        if image.shape != (128,128,3) or image.max() == image.min():
            raise RuntimeError('Headless renderer returned an invalid image')
        Image.fromarray(image).save(args.output.with_suffix('.png'))
        arrays_path = args.output.with_suffix('.npz')
        np.savez_compressed(arrays_path, rest=rest, initial=initial,
                            series=np.asarray(series), collision_rest=collision_rest,
                            initial_gripper_pose=initial_pose, tetrahedra=all_tets)
        report = {'purpose': 'two-resolution-sensitivity-only', 'seed': 42,
                  'mesh_path': str(args.mesh), 'mesh_sha256': hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
                  'vertex_count': len(initial), 'tetrahedron_count': len(all_tets),
                  'topology_component_state': str(body.topology_container.componentState.value),
                  'original_vertex_count': len(original_points), 'steps': steps,
                  'time_step_s': env.time_step, 'frame_skip': env.frame_skip,
                  'total_mass': float(body.mass.totalMass.value), 'material_young_modulus': body.material.young_modulus,
                  'material_poisson_ratio': body.material.poisson_ratio,
                  'collision_attachment_indices': list(body.collision_spring_force_field_indices),
                  'grasp_spring_collision_indices': {name: spring.points.array().tolist()
                                                     for name,spring in env.gripper.grasping_force_field.items()},
                  'initial_gripper_pose': initial_pose.tolist(), 'diagnostics': diagnostics,
                  'elapsed_seconds': time.monotonic()-start, 'arrays_path': str(arrays_path),
                  'raw_internal_force_sum_compared': False}
        with args.output.open('x') as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
        return 0
    finally:
        env.close(); handler.__exit__(None,None,None)


def cantilever_mesh():
    from refine_sofa_mesh import signed_volumes
    points = np.array([[x, y, z] for x in np.linspace(0, 10, 5) for y in (0.,1.) for z in (0.,1.)])
    def index(i,y,z): return i*4+y*2+z
    tets=[]
    for i in range(4):
        a,b,c,d = index(i,0,0),index(i+1,0,0),index(i+1,1,0),index(i,1,0)
        e,f,g,h = index(i,0,1),index(i+1,0,1),index(i+1,1,1),index(i,1,1)
        tets.extend([(a,b,c,g),(a,c,d,g),(a,d,h,g),(a,h,e,g),(a,e,f,g),(a,f,b,g)])
    tets=np.array(tets,dtype=int)
    negative=signed_volumes(points,tets)<0
    tets[negative,0],tets[negative,1]=tets[negative,1].copy(),tets[negative,0].copy()
    return points,tets


def cantilever_worker(args):
    from collections import Counter
    from itertools import combinations
    import Sofa
    import SofaRuntime
    import Sofa.Simulation
    from refine_sofa_mesh import read_tetra_mesh,signed_volumes,mesh_geometry
    plugins=['Sofa.Component.AnimationLoop','Sofa.Component.ODESolver.Backward',
             'Sofa.Component.LinearSolver.Iterative','Sofa.Component.StateContainer',
             'Sofa.Component.Topology.Container.Dynamic','Sofa.Component.Mass',
             'Sofa.Component.SolidMechanics.FEM.Elastic','Sofa.Component.Constraint.Projective',
             'Sofa.Component.MechanicalLoad']
    for plugin in plugins: SofaRuntime.importPlugin(plugin)
    messages=[]
    class Handler(Sofa.Helper.MessageHandler):
        def process(self,message):
            if message['type'] in ('Error','Fatal','Warning'):
                messages.append({key:str(message[key]) for key in ('type','sender','message')})
    handler=Handler();handler.__enter__()
    points,tets=read_tetra_mesh(args.mesh)
    volumes=signed_volumes(points,tets)
    masses=np.zeros(len(points))
    for k in range(4):np.add.at(masses,tets[:,k],volumes*.001/4)
    faces=Counter(tuple(sorted(face)) for tet in tets for face in combinations(map(int,tet),3))
    end_faces=[face for face,count in faces.items() if count==1 and np.allclose(points[list(face),0],10)]
    weights=np.zeros(len(points))
    for face in end_faces:
        p=points[list(face)];area=np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0]))/2
        weights[list(face)]+=area/3
    forces=np.zeros_like(points);forces[:,1]=-.01*weights
    fixed=np.flatnonzero(np.isclose(points[:,0],0))
    root=Sofa.Core.Node('cantilever');root.gravity.value=[0.,0.,0.];root.dt.value=.01
    root.addObject('DefaultAnimationLoop')
    node=root.addChild('body')
    node.addObject('StaticSolver',newton_iterations=10,absolute_residual_tolerance_threshold=1e-11,
                   relative_residual_tolerance_threshold=1e-11)
    node.addObject('CGLinearSolver',iterations=10000,tolerance=1e-13,threshold=1e-15)
    mechanical=node.addObject('MechanicalObject',position=points.tolist())
    topo=node.addObject('TetrahedronSetTopologyContainer',position=points.tolist(),tetrahedra=tets.tolist())
    node.addObject('TetrahedronSetGeometryAlgorithms')
    mass=node.addObject('DiagonalMass',vertexMass=masses.tolist())
    node.addObject('TetrahedronFEMForceField',method='small',youngModulus=1500,poissonRatio=.3)
    node.addObject('FixedProjectiveConstraint',indices=fixed.tolist())
    node.addObject('ConstantForceField',indices=list(range(len(points))),forces=forces.tolist())
    start=time.monotonic()
    try:
        Sofa.Simulation.init(root)
        previous=mechanical.position.array().copy();changes=[]
        for _ in range(4):
            Sofa.Simulation.animate(root,.01)
            current=mechanical.position.array().copy()
            changes.append(float(np.max(np.linalg.norm(current-previous,axis=1))))
            previous=current
        if any(row['type'] in ('Error','Fatal') for row in messages):
            raise RuntimeError('Cantilever scene reported SOFA errors')
        if not np.isfinite(current).all():raise RuntimeError('Nonfinite cantilever state')
        incremental=strains(points,current,tets)
        if incremental['inverted_elements']:raise RuntimeError('Inverted cantilever elements')
        tip=float(np.sum((current[:,1]-points[:,1])*weights)/weights.sum())
        moment=np.cross(points-[0.,.5,.5],forces).sum(axis=0)
        classical=.01*10**3/(3*1500*(1/12))
        data={'purpose':'controlled-small-strain-cantilever-fixture',
              'vertex_count':len(points),'tetrahedra':len(tets),'geometry':mesh_geometry(points,tets)[0],
              'beam_dimensions':[10,1,1],'young_modulus':1500,'poisson_ratio':.3,
              'density':.001,'declared_lumped_mass':float(masses.sum()),
              'sofa_mass_sum':float(np.asarray(mass.vertexMass.value).sum()),
              'end_traction_area':float(weights.sum()),'total_force':forces.sum(axis=0).tolist(),
              'total_moment_about_fixed_face_center':moment.tolist(),
              'fixed_plane_x':0,'fixed_nodes':fixed.tolist(),
              'tip_area_weighted_y_displacement':tip,
              'euler_bernoulli_tip_deflection_magnitude':classical,
              'relative_difference_from_beam_approximation':abs(abs(tip)-classical)/classical,
              'last_static_solve_max_position_change':changes[-1],
              'successive_static_solve_changes':changes,'strain':incremental,
              'topology_component_state':str(topo.componentState.value),
              'messages':messages,'elapsed_seconds':time.monotonic()-start,
              'limitations':['Beam formula is a slender-beam approximation, not the exact3Dcontinuumsolution.',
                             'Three meshes show sensitivity; no asymptotic rate or spatial-error bound is established.',
                             'This fixture has no organ anatomy, camera, contact, injury, or surgical efficacy interpretation.']}
        with args.output.open('x') as stream:json.dump(data,stream,indent=2,allow_nan=False)
        np.savez_compressed(args.output.with_suffix('.npz'),reference=points,current=current,tetrahedra=tets,
                            nodal_forces=forces,nodal_masses=masses)
        return 0
    finally:
        Sofa.Simulation.unload(root);handler.__exit__(None,None,None)


def independent_linear_tetra_reference(array_path):
    """Independent small-strain P1 tetra assembly; not a continuum truth claim."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    with np.load(array_path) as data:
        points,tets,forces,current=[data[name].copy() for name in ['reference','tetrahedra','nodal_forces','current']]
    young,poisson=1500.,.3
    mu=young/(2*(1+poisson));lam=young*poisson/((1+poisson)*(1-2*poisson))
    constitutive=np.zeros((6,6));constitutive[:3,:3]=lam
    constitutive[np.arange(3),np.arange(3)]+=2*mu
    constitutive[np.arange(3,6),np.arange(3,6)]=mu
    rows=[];columns=[];entries=[]
    for tet in tets:
        x=points[tet]
        gradients=np.linalg.inv(np.column_stack([np.ones(4),x]))[1:].T
        b=np.zeros((6,12))
        for i,(dx,dy,dz) in enumerate(gradients):
            b[0,3*i]=dx;b[1,3*i+1]=dy;b[2,3*i+2]=dz
            b[3,3*i]=dy;b[3,3*i+1]=dx
            b[4,3*i+1]=dz;b[4,3*i+2]=dy
            b[5,3*i]=dz;b[5,3*i+2]=dx
        volume=abs(np.linalg.det((x[1:]-x[0]).T))/6
        local=volume*(b.T@constitutive@b)
        dofs=(3*tet[:,None]+np.arange(3)).ravel()
        rows.extend(np.repeat(dofs,12));columns.extend(np.tile(dofs,12));entries.extend(local.ravel())
    stiffness=coo_matrix((entries,(rows,columns)),shape=(points.size,points.size)).tocsr()
    fixed=np.flatnonzero(np.isclose(points[:,0],0))
    locked=(3*fixed[:,None]+np.arange(3)).ravel();free=np.setdiff1d(np.arange(points.size),locked)
    displacement=np.zeros(points.size)
    displacement[free]=spsolve(stiffness[free][:,free],forces.ravel()[free])
    residual=stiffness@displacement-forces.ravel()
    u=displacement.reshape(-1,3)
    return {'method':'independent SciPy isotropic P1 tetra assembly with engineering shear components',
            'max_nodal_displacement_difference_from_sofa':float(np.linalg.norm(u-(current-points),axis=1).max()),
            'free_dof_residual_l2':float(np.linalg.norm(residual[free])),
            'integrated_support_reaction':residual.reshape(-1,3)[fixed].sum(axis=0).tolist(),
            'tip_area_weighted_y_displacement':float(np.sum(u[:,1]*forces[:,1])/forces[:,1].sum()),
            'scope':'Same discrete linear-elastic equations independently assembled; not anatomical or continuum validation.'}


def run_cantilever(args,environment,python,output):
    from refine_sofa_mesh import refine_tetrahedra,write_tetra_vtk
    points,tets=cantilever_mesh()
    fine_points,fine_tets,refinement=refine_tetrahedra(points,tets)
    extra_points,extra_tets,extra_refinement=refine_tetrahedra(fine_points,fine_tets)
    meshes={'coarse':(points,tets),'fine':(fine_points,fine_tets),'extra_fine':(extra_points,extra_tets)}
    results={}
    for label,(p,t) in meshes.items():
        mesh=output/f'cantilever-{label}.vtk';write_tetra_vtk(mesh,p,t)
        result=output/f'{label}.json'
        command=[str(python),str(Path(__file__).resolve()),'--cantilever-worker','--mesh',str(mesh),'--output',str(result)]
        with (output/f'{label}.log').open('x') as log:
            process=subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=60)
        if process.returncode:raise RuntimeError(f'Cantilever {label} failed; inspect{output/label}.log')
        results[label]=json.loads(result.read_text())
        results[label]['independent_discrete_reference']=independent_linear_tetra_reference(result.with_suffix('.npz'))
    checks={key:all(results['coarse'][key]==row[key] for row in results.values()) for key in
            ['beam_dimensions','young_modulus','poisson_ratio','density','fixed_plane_x']}
    for key in ['declared_lumped_mass','sofa_mass_sum','end_traction_area','total_force','total_moment_about_fixed_face_center']:
        checks[key]=all(bool(np.allclose(results['coarse'][key],row[key],atol=1e-12,rtol=1e-12)) for row in results.values())
    checks['independent_discrete_assembly_agrees']=all(r['independent_discrete_reference']['max_nodal_displacement_difference_from_sofa']<1e-9 and r['independent_discrete_reference']['free_dof_residual_l2']<1e-9 for r in results.values())
    checks['both_static_states_settled']=all(r['last_static_solve_max_position_change']<1e-8 for r in results.values())
    report={'purpose':'controlled-cantilever-mesh-sensitivity-only','matching_conditions':checks,
            'conditions_verified':all(checks.values()),'coarse':results['coarse'],'fine':results['fine'],
            'extra_fine':results['extra_fine'],'refinement':refinement,'extra_refinement':extra_refinement,
            'tip_deflection_difference':results['fine']['tip_area_weighted_y_displacement']-results['coarse']['tip_area_weighted_y_displacement'],
            'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ['sofa_mesh_resolution_check.py','refine_sofa_mesh.py']}}
    with (output/'comparison.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({'conditions_verified':report['conditions_verified'],
                      'coarse_tip':results['coarse']['tip_area_weighted_y_displacement'],
                      'fine_tip':results['fine']['tip_area_weighted_y_displacement'],
                      'extra_fine_tip':results['extra_fine']['tip_area_weighted_y_displacement'],
                      'beam_approximation':results['fine']['euler_bernoulli_tip_deflection_magnitude']},indent=2))
    return 0 if report['conditions_verified'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,default=Path('/workspace/.sofa-runtime'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--cantilever',action='store_true')
    parser.add_argument('--cantilever-worker',action='store_true')
    parser.add_argument('--mesh',type=Path)
    parser.add_argument('--original-mesh',type=Path)
    args=parser.parse_args()
    if not 1<=args.steps<=30:
        parser.error('steps must be1..30')
    if args.cantilever_worker:
        return cantilever_worker(args)
    if args.worker:
        return worker(args)
    root=args.runtime.resolve(); output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    original=root/'source/sofa_env/scenes/grasp_lift_touch/meshes/gallbladder_volumetric.vtk'
    fine=root/'mesh-resolution/gallbladder-fine.vtk'
    if not args.cantilever and not fine.is_file():
        raise RuntimeError('Generate the audited refined mesh with refine_sofa_mesh.py first')
    python=root/'venv/bin/python'
    sofa=root/'SOFA_v24.06.00_Linux'
    toolchain=subprocess.check_output([str(python),'-c','import sys;print(sys.base_prefix+"/lib")'],text=True).strip()
    environment=os.environ.copy()
    environment.update(SOFA_ROOT=str(sofa),SOFAPYTHON3_ROOT=str(sofa/'plugins/SofaPython3'),
                       SOFA_ENV_SOURCE=str(root/'source'),
                       PYTHONPATH=f'{sofa}/plugins/SofaPython3/lib/python3/site-packages:{root}/source',
                       LD_LIBRARY_PATH=f'{sofa}/lib:{sofa}/plugins/SofaPython3/lib:{toolchain}'+(':'+os.environ['LD_LIBRARY_PATH'] if 'LD_LIBRARY_PATH' in os.environ else ''))
    if args.cantilever:
        return run_cantilever(args,environment,python,output)
    reports={}
    for label,mesh in [('coarse',original),('fine',fine)]:
        result_path=output/f'{label}.json'
        command=[str(python),str(Path(__file__).resolve()),'--worker','--mesh',str(mesh),
                 '--original-mesh',str(original),'--output',str(result_path),'--steps',str(args.steps)]
        with (output/f'{label}.log').open('x') as log:
            process=subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=150)
        if process.returncode!=0:
            raise RuntimeError(f'{label} worker failed; inspect {output/label}.log')
        reports[label]=json.loads(result_path.read_text())
    with np.load(output/'coarse.npz') as a,np.load(output/'fine.npz') as b:
        n=reports['coarse']['vertex_count']
        initial_difference=np.linalg.norm(a['initial']-b['initial'][:n],axis=1)
        increment_difference=(b['series']-b['initial'][:n])-(a['series']-a['initial'])
        final_difference=np.linalg.norm(increment_difference[-1],axis=1)
        collision_rest_equal=np.array_equal(a['collision_rest'],b['collision_rest'])
        rest_equal=np.array_equal(a['rest'],b['rest'][:n])
    comparison={'purpose':'two-resolution-sensitivity-only, not a convergence or clinical validation claim',
                'coarse':reports['coarse'],'fine':reports['fine'],
                'original_rest_positions_exactly_equal':bool(rest_equal),
                'collision_rest_positions_exactly_equal':bool(collision_rest_equal),
                'attachment_collision_indices_equal':reports['coarse']['collision_attachment_indices']==reports['fine']['collision_attachment_indices'],
                'grasp_spring_collision_indices_equal':reports['coarse']['grasp_spring_collision_indices']==reports['fine']['grasp_spring_collision_indices'],
                'initial_gripper_pose_equal':reports['coarse']['initial_gripper_pose']==reports['fine']['initial_gripper_pose'],
                'initial_original_vertex_rms_difference':float(np.sqrt(np.mean(initial_difference**2))),
                'final_incremental_original_vertex_rms_difference':float(np.sqrt(np.mean(final_difference**2))),
                'final_incremental_original_vertex_max_difference':float(final_difference.max()),
                'limitations':['UniformMass holds total mass fixed but redistributes per-vertex masses when resolution changes.',
                               'Collision/visual geometry and indexed attachment sites stay fixed, but their barycentric volume mappings and load localization change.',
                               'Scripted reset includes contact/settling; identical seeds do not guarantee identical preloaded tissue state.',
                               'Native maximum strain samples more tetrahedra on the fine mesh; compare common coarse-tetra estimates separately.',
                               'Raw summed nodal internal force is resolution dependent and is intentionally excluded.',
                               'One refinement and one bounded trajectory cannot establish asymptotic convergence or clinical fidelity.']}
    comparison['source_sha256'] = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['sofa_mesh_resolution_check.py','refine_sofa_mesh.py']}
    comparison['upstream_commit'] = (root/'source/.upstream_commit').read_text().strip()
    comparison['sofa_version'] = '24.06.00'
    comparison['controlled_comparison_ready'] = bool(comparison['grasp_spring_collision_indices_equal'] and comparison['collision_rest_positions_exactly_equal'] and comparison['attachment_collision_indices_equal'])
    comparison['comparison_status'] = 'limited_two_resolution_sensitivity' if comparison['controlled_comparison_ready'] else 'uncontrolled_contact_reset_changes'
    with (output/'comparison.json').open('x') as stream:
        json.dump(comparison,stream,indent=2,allow_nan=False)
    print(json.dumps({key:value for key,value in comparison.items() if key not in ('coarse','fine')},indent=2))
    return 0 if comparison['controlled_comparison_ready'] else 2


if __name__=='__main__':
    raise SystemExit(main())
