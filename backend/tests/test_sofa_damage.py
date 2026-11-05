"""Constitutive invariants independent of a native SOFA installation."""
import numpy as np
import pytest

from retraction.sofa_damage import TetraDamageModel, DamageParameters

POINTS = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
TETS = np.array([[0,1,2,3]])


def test_rigid_body_motion_does_not_injure_tissue():
    model = TetraDamageModel(POINTS, TETS)
    rotation = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    result = model.update(POINTS @ rotation.T + [4,8,-2], .1)
    assert result['max_damage'] == 0
    assert result['max_incremental_tensile_strain'] == pytest.approx(0, abs=1e-12)
    assert model.young_modulus[0] == 1500


def test_overstretch_irreversibly_reduces_future_stiffness():
    model = TetraDamageModel(POINTS, TETS)
    stretched = POINTS.copy(); stretched[3, 2] = 1.5
    result = model.update(stretched, .1)
    assert result['max_damage'] == pytest.approx(.5)
    assert model.young_modulus[0] == pytest.approx(787.5)
    model.update(POINTS, .1)
    assert model.damage[0] == pytest.approx(.5)
    assert model.young_modulus[0] == pytest.approx(787.5)


def test_thermal_contact_accumulates_injury_and_cooling_cannot_heal():
    model = TetraDamageModel(POINTS, TETS)
    for _ in range(10):
        model.update(POINTS, .1, heat_position=POINTS.mean(axis=0), power=1)
    damage = model.damage.copy()
    assert damage[0] > .1
    temperature = model.temperature_c[0]
    model.update(POINTS, .1)
    assert model.temperature_c[0] < temperature
    assert model.damage[0] >= damage[0]
    assert model.young_modulus[0] > 0


def test_power_without_contact_has_no_effect():
    model = TetraDamageModel(POINTS, TETS)
    for _ in range(20):
        model.update(POINTS, .1, heat_position=[20,20,20], power=1)
    assert model.damage[0] == 0
    assert model.temperature_c[0] == 37


def test_material_floor_and_no_topology_surgery():
    model = TetraDamageModel(POINTS, TETS)
    result = model.update(POINTS * 3, .1)
    assert result['max_damage'] == 1
    assert model.young_modulus[0] == pytest.approx(75)
    assert np.array_equal(model.tetrahedra, TETS)
    assert np.array_equal(model.reference, POINTS)


@pytest.mark.parametrize('kwargs', [{'strain_failure':.1}, {'residual_stiffness':0}, {'heat_rate_c_per_s':-1}])
def test_invalid_model_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        DamageParameters(**kwargs)


def test_invalid_geometry_and_nonfinite_actions_rejected():
    with pytest.raises(ValueError):
        TetraDamageModel(POINTS * 0, TETS)
    model = TetraDamageModel(POINTS, TETS)
    with pytest.raises(ValueError):
        model.update(POINTS, float('nan'))
    with pytest.raises(ValueError):
        model.update(POINTS, .1, heat_position=[0,0,0], power=2)


def test_invalid_thermal_contact_cannot_partially_mutate_material():
    model = TetraDamageModel(POINTS, TETS)
    stretched = POINTS.copy(); stretched[3,2] = 2
    with pytest.raises(ValueError):
        model.update(stretched, .1, heat_position=[float('nan'),0,0], power=1)
    assert model.damage[0] == 0
    assert model.time_s == 0
    assert model.temperature_c[0] == 37
