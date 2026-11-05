"""Geometric/refined-strain invariants without requiring SOFA."""
import sys
from pathlib import Path
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from refine_sofa_mesh import refine_tetrahedra, read_tetra_mesh, write_tetra_vtk
from sofa_mesh_resolution_check import strains


def shared_face_mesh():
    return (np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],dtype=float),
            np.array([[0,1,2,3],[0,2,1,4]],dtype=int))


def test_two_tetrahedra_share_midpoints_and_preserve_boundary_volume():
    points,tets=shared_face_mesh()
    fine,children,report=refine_tetrahedra(points,tets)
    np.testing.assert_array_equal(fine[:len(points)],points)
    assert len(fine)==14  # five original sites plus nine unique shared edges
    assert children.shape==(16,4)
    assert report['refined_boundary_matches_shared_face_subdivision']
    assert report['fine']['boundary_faces']==24
    assert report['fine']['nonpositive_tetrahedra']==0
    assert report['fine']['nonmanifold_face_count']==0
    assert report['fine']['nonmanifold_boundary_edge_count']==0
    assert report['fine']['total_volume_source_units_cubed']==pytest.approx(1/3)
    assert report['max_per_parent_relative_volume_error']<1e-12


def test_affine_strain_and_rigid_rotation_are_resolution_invariant():
    points,tets=shared_face_mesh()
    fine,children,_=refine_tetrahedra(points,tets)
    affine=np.diag([1.1,.95,1.0])
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    for reference,connectivity in [(points,tets),(fine,children)]:
        result=strains(reference,reference@affine.T+[2,3,4],connectivity)
        assert result['maximum_principal_tensile_strain']==pytest.approx(.1)
        assert result['volume_weighted_mean_principal_tensile_strain']==pytest.approx(.1)
        assert result['inverted_elements']==0
        rigid=strains(reference,reference@rotation.T+[2,3,4],connectivity)
        assert rigid['maximum_principal_tensile_strain']<1e-12


def test_generated_vtk_roundtrip_is_exact_and_never_overwrites(tmp_path):
    points,tets=shared_face_mesh()
    fine,children,_=refine_tetrahedra(points,tets)
    path=tmp_path/'fine.vtk'
    write_tetra_vtk(path,fine,children)
    restored,indices=read_tetra_mesh(path)
    np.testing.assert_array_equal(restored,fine)
    np.testing.assert_array_equal(indices,children)
    with pytest.raises(FileExistsError):write_tetra_vtk(path,points,tets)


def test_negative_or_degenerate_input_is_rejected():
    points,tets=shared_face_mesh()
    tets[0,[0,1]]=tets[0,[1,0]]
    with pytest.raises(ValueError,match='positive'):refine_tetrahedra(points,tets)
    with pytest.raises(ValueError,match='positive'):refine_tetrahedra(points,np.array([[0,0,1,2]]))
