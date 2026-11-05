"""Inspectable experimental element damage for the optional SOFA adapter.

No SOFA import is needed to test the constitutive bookkeeping. Scene length and
force units are inherited from the upstream scene and are not clinically
calibrated. Damage cannot heal, and residual stiffness keeps the numerical model
positive; this is softening, not mesh surgery or a calibrated rupture model.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DamageParameters:
    strain_onset: float = 0.25
    strain_failure: float = 0.75
    residual_stiffness: float = 0.05
    ambient_temperature_c: float = 37.0
    heat_rate_c_per_s: float = 60.0
    cooling_rate_per_s: float = 0.4
    thermal_onset_c: float = 50.0
    thermal_rate_per_c2_s: float = 0.002
    contact_radius_scene_units: float = 3.0

    def __post_init__(self):
        values = vars(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError('Damage parameters must be finite')
        if not 0 <= self.strain_onset < self.strain_failure:
            raise ValueError('Strain failure must exceed onset')
        if not 0 < self.residual_stiffness <= 1:
            raise ValueError('Residual stiffness must lie in (0,1]')
        if min(self.heat_rate_c_per_s, self.cooling_rate_per_s, self.thermal_rate_per_c2_s) < 0:
            raise ValueError('Heating/cooling/damage rates must be nonnegative')
        if self.contact_radius_scene_units <= 0:
            raise ValueError('Contact radius must be positive')


class TetraDamageModel:
    """Rotation-invariant, irreversible damage measured against a fixed reference."""

    def __init__(self, reference_positions, tetrahedra, base_young_modulus=1500.0,
                 parameters: DamageParameters | None = None):
        self.parameters = parameters or DamageParameters()
        self.reference = np.asarray(reference_positions, dtype=float).copy()
        raw_tets = np.asarray(tetrahedra)
        if raw_tets.ndim != 2 or raw_tets.shape[1] != 4 or not len(raw_tets):
            raise ValueError('Expected nonempty (n,4) tetrahedron indices')
        if not np.issubdtype(raw_tets.dtype, np.integer):
            raise ValueError('Tetrahedron indices must be integers')
        self.tetrahedra = raw_tets.astype(int, copy=True)
        if self.reference.ndim != 2 or self.reference.shape[1] != 3 or not np.isfinite(self.reference).all():
            raise ValueError('Expected finite (n,3) reference positions')
        if self.tetrahedra.min() < 0 or self.tetrahedra.max() >= len(self.reference):
            raise ValueError('Tetrahedron index out of bounds')
        points = self.reference[self.tetrahedra]
        # Columns are the three edge vectors in each reference tetrahedron.
        matrices = np.transpose(points[:, 1:] - points[:, :1], (0, 2, 1))
        determinant = np.linalg.det(matrices)
        if np.any(np.abs(determinant) < 1e-12):
            raise ValueError('Degenerate reference tetrahedron')
        self.inverse_reference = np.linalg.inv(matrices)
        self.volumes = np.abs(determinant) / 6.0
        self.base_modulus = np.broadcast_to(np.asarray(base_young_modulus, dtype=float), (len(points),)).copy()
        if not np.isfinite(self.base_modulus).all() or np.any(self.base_modulus <= 0):
            raise ValueError('Young modulus must be finite and positive')
        self.mechanical_damage = np.zeros(len(points))
        self.thermal_damage = np.zeros(len(points))
        self.temperature_c = np.full(len(points), self.parameters.ambient_temperature_c)
        self.peak_tensile_strain = np.zeros(len(points))
        self.time_s = 0.0

    @property
    def damage(self):
        return 1.0 - (1.0 - self.mechanical_damage) * (1.0 - self.thermal_damage)

    @property
    def young_modulus(self):
        return self.base_modulus * (1.0 - (1.0 - self.parameters.residual_stiffness) * self.damage)

    def update(self, positions, dt_s, *, heat_position=None, power=0.0):
        current = np.asarray(positions, dtype=float)
        if current.shape != self.reference.shape or not np.isfinite(current).all():
            raise ValueError('Current positions must match the finite reference array')
        if not np.isfinite(dt_s) or not 0 < dt_s <= 1:
            raise ValueError('dt_s must lie in (0,1]')
        if not np.isfinite(power) or not 0 <= power <= 1:
            raise ValueError('Power must lie in [0,1]')
        source = None
        if power:
            source = np.asarray(heat_position, dtype=float)
            if source.shape != (3,) or not np.isfinite(source).all():
                raise ValueError('Powered thermal contact needs a finite 3D source')
        p = self.parameters
        points = current[self.tetrahedra]
        matrices = np.transpose(points[:, 1:] - points[:, :1], (0, 2, 1))
        deformation = matrices @ self.inverse_reference
        stretches = np.linalg.svd(deformation, compute_uv=False)
        tensile = np.maximum(stretches.max(axis=1) - 1.0, 0.0)
        if not np.isfinite(tensile).all():
            raise ValueError('Nonfinite strain')
        self.peak_tensile_strain = np.maximum(self.peak_tensile_strain, tensile)
        target_damage = np.clip((self.peak_tensile_strain - p.strain_onset) /
                                (p.strain_failure - p.strain_onset), 0, 1)
        self.mechanical_damage = np.maximum(self.mechanical_damage, target_damage)
        # Exact exponential cooling plus a bounded explicit heating source.
        self.temperature_c = p.ambient_temperature_c + (self.temperature_c - p.ambient_temperature_c) * np.exp(-p.cooling_rate_per_s * dt_s)
        if power:
            distance = np.linalg.norm(points.mean(axis=1) - source, axis=1)
            # Compact support: no heating if the source does not touch the patch.
            contact = np.maximum(0, 1 - distance / p.contact_radius_scene_units)
            self.temperature_c += p.heat_rate_c_per_s * power * contact * dt_s
        excess = np.maximum(self.temperature_c - p.thermal_onset_c, 0)
        self.thermal_damage = 1 - (1 - self.thermal_damage) * np.exp(-p.thermal_rate_per_c2_s * excess**2 * dt_s)
        self.time_s += dt_s
        return {
            'time_s': self.time_s,
            'max_incremental_tensile_strain': float(tensile.max()),
            'max_peak_tensile_strain': float(self.peak_tensile_strain.max()),
            'max_damage': float(self.damage.max()),
            'mean_volume_weighted_damage': float(np.average(self.damage, weights=self.volumes)),
            'damaged_element_count': int((self.damage > 0).sum()),
            'element_count': len(self.tetrahedra),
            'max_temperature_c': float(self.temperature_c.max()),
            'minimum_young_modulus_scene_units': float(self.young_modulus.min()),
            'inverted_element_count': int((np.linalg.det(deformation) <= 0).sum()),
        }

    def apply_to_sofa(self, force_field):
        """Apply local softening using stock vector-valued FEM material updates.

        FastTetrahedralCorotationalForceField caches scalar material data and did
        not update compliance on reinit in the tested SOFA release. Reject it
        instead of claiming that changing a displayed parameter changes physics.
        The experimental damage scene selects TetrahedronFEMForceField before
        initialization; its live per-element response is independently checked.
        """
        current = np.asarray(force_field.youngModulus.value)
        if current.ndim == 0:
            raise ValueError('Live damage requires vector-valued TetrahedronFEMForceField; scalar cached FEM is unsupported')
        force_field.youngModulus.value = self.young_modulus.tolist()
        force_field.reinit()
        return {'application_mode': 'per-element-modulus',
                'minimum_requested_modulus': float(self.young_modulus.min()),
                'volume_weighted_modulus': float(np.average(self.young_modulus, weights=self.volumes))}
