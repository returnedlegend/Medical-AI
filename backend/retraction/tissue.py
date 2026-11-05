"""Stateful lumped tissue/heat/injury mechanics for an educational surrogate.

These illustrative equations are not calibrated tissue laws, a patient model,
SOFA/FEM, clinical thresholds, or instructions for performing surgery.
"""
from __future__ import annotations

from typing import Any
import numpy as np

TISSUE_VERSION = "lumped-tissue-1.0"
KINDS = ("gallbladder", "artery", "cystic_duct", "common_bile_duct")
# XY positions in an illustrative workspace, not reconstructed human anatomy.
NODE_SPECS = [
    ("gallbladder", 0, -3),
    *[("gallbladder", x, 3) for x in (-6, 0, 6)],
    *[("gallbladder", x, 9) for x in (-9, -3, 3, 9)],
    *[("gallbladder", x, 15) for x in (-9, -3, 3, 9)],
    *[("gallbladder", x, 21) for x in (-6, 0, 6)],
    ("gallbladder", 0, 27),
    ("artery", -4, -10), ("artery", -2, -16),
    ("cystic_duct", 2, -8), ("cystic_duct", 4, -14),
    ("common_bile_duct", 8, -12), ("common_bile_duct", 8, -20), ("common_bile_duct", 8, -28),
]
NODE_COUNT = len(NODE_SPECS)


def combined_damage(mechanical: np.ndarray, thermal: np.ndarray) -> np.ndarray:
    return 1 - (1 - mechanical) * (1 - thermal)


class TissueModel:
    """Overdamped spring network with irreversible injury and thermal memory."""
    def __init__(self, patient: dict[str, Any]):
        self.patient = dict(patient)
        self.kinds = [row[0] for row in NODE_SPECS]
        self.rest = np.array([[row[1], row[2]] for row in NODE_SPECS], dtype=float)
        self.rest[:16, 0] *= 1 + 0.15 * (patient["distension"] - 1)
        self.positions = self.rest.copy()
        self.mechanical_damage = np.zeros(NODE_COUNT)
        self.thermal_dose = np.zeros(NODE_COUNT)
        self.temperature = np.full(NODE_COUNT, 37.0)
        self.strain = np.zeros(NODE_COUNT)
        self.coagulation = np.zeros(NODE_COUNT)
        self.edges = np.array([(i, j) for i in range(NODE_COUNT) for j in range(i + 1, NODE_COUNT)
                               if self.kinds[i] == self.kinds[j] and
                               np.linalg.norm(self.rest[i] - self.rest[j]) <= 8.7], dtype=int)
        self.rest_lengths = np.linalg.norm(self.rest[self.edges[:, 1]] - self.rest[self.edges[:, 0]], axis=1)
        # Anatomical settings affect mechanics, not a prescribed optimal angle.
        base = 1.6 * patient["distension"] / patient["attachment_length_mm"]
        self.base_stiffness = np.full(NODE_COUNT, base)
        for i, (kind, _, y) in enumerate(NODE_SPECS):
            if kind != "gallbladder":
                self.base_stiffness[i] = 0.35
            elif self.rest[i, 0] < 0:
                self.base_stiffness[i] += patient["adhesion_count"] * patient["adhesion_stiffness_n_per_mm"] * 0.22
            if kind == "gallbladder" and y == 27:
                self.base_stiffness[i] = 2.0  # fixed-support surrogate at fundus
        self.contact_index: int | None = None
        self.contact_force_n = 0.0
        self.blood_loss_ml = 0.0
        self.bile_leak_ml = 0.0
        self.bleeding_rate = 0.0
        self.bile_leak_rate = 0.0
        self.heart_rate = 72.0
        self.mean_arterial_pressure = 90.0

    @property
    def thermal_damage(self) -> np.ndarray:
        return 1 - np.exp(-self.thermal_dose / 0.8)

    @property
    def damage(self) -> np.ndarray:
        return combined_damage(self.mechanical_damage, self.thermal_damage)

    @property
    def stiffness(self) -> np.ndarray:
        # Injury permanently reduces the load-bearing material, even after cooling.
        return self.base_stiffness * (1 - 0.9 * self.damage)

    def acquire_contact(self, grasper: np.ndarray, closure: float) -> bool:
        if self.contact_index is not None or closure < 0.25:
            return False
        distances = np.linalg.norm(self.positions - grasper, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= 3.0 and self.damage[nearest] < 0.9:
            self.contact_index = nearest
            return True
        return False

    def release_contact(self) -> bool:
        had_contact = self.contact_index is not None
        self.contact_index = None
        self.contact_force_n = 0.0
        return had_contact

    def step(self, dt: float, grasper: np.ndarray, closure: float,
             cautery: np.ndarray, power: float, force_limit: float,
             strain_limit: float) -> dict[str, Any]:
        """Advance one action interval; all dissipated injury persists."""
        previous_damage = self.damage.copy()
        previous_blood, previous_bile = self.blood_loss_ml, self.bile_leak_ml
        # Integrate the damped network in fixed substeps for reproducibility.
        for _ in range(5):
            h = dt / 5
            damage = self.damage
            forces = self.stiffness[:, None] * (self.rest - self.positions)
            edge_vectors = self.positions[self.edges[:, 1]] - self.positions[self.edges[:, 0]]
            lengths = np.linalg.norm(edge_vectors, axis=1)
            directions = edge_vectors / np.maximum(lengths[:, None], 1e-9)
            edge_k = 0.18 * (1 - 0.9 * (damage[self.edges[:, 0]] + damage[self.edges[:, 1]]) / 2)
            spring_forces = (edge_k * (lengths - self.rest_lengths))[:, None] * directions
            np.add.at(forces, self.edges[:, 0], spring_forces)
            np.add.at(forces, self.edges[:, 1], -spring_forces)
            if self.contact_index is not None:
                i = self.contact_index
                coupling = 1.1 * closure * (1 - 0.8 * damage[i])
                tool_force = coupling * (grasper - self.positions[i])
                norm = float(np.linalg.norm(tool_force))
                if norm > 20:
                    tool_force *= 20 / norm  # numerical saturation, not a safety limit
                forces[i] += tool_force
            self.positions += h * forces / 0.65  # overdamped mobility (mm/N/s)
            self.positions = np.clip(self.positions, [-35, -35], [35, 45])

        edge_vectors = self.positions[self.edges[:, 1]] - self.positions[self.edges[:, 0]]
        edge_strain = np.maximum(0, np.linalg.norm(edge_vectors, axis=1) / self.rest_lengths - 1)
        self.strain[:] = np.linalg.norm(self.positions - self.rest, axis=1) / self.patient["attachment_length_mm"]
        np.maximum.at(self.strain, self.edges[:, 0], edge_strain)
        np.maximum.at(self.strain, self.edges[:, 1], edge_strain)
        mechanical_increment = dt * 0.45 * np.maximum(0, self.strain / strain_limit - 1)**2
        if self.contact_index is not None:
            i = self.contact_index
            self.contact_force_n = min(20.0, float(np.linalg.norm(
                1.1 * closure * (1 - 0.8 * self.damage[i]) * (grasper - self.positions[i]))))
            mechanical_increment[i] += dt * 0.25 * max(0, self.contact_force_n / force_limit - 1)**2
            # Persistent excessive jaw compression creates injury even at rest.
            mechanical_increment[i] += dt * 0.22 * (max(0, closure - 0.75) / 0.25)**2
        else:
            self.contact_force_n = 0.0
        self.mechanical_damage = np.clip(self.mechanical_damage + mechanical_increment, 0, 1)

        distance = np.linalg.norm(self.positions - cautery, axis=1)
        heat_kernel = np.exp(-0.5 * (distance / 2.4)**2) * (distance <= 7.2)
        self.temperature += dt * (160 * power * heat_kernel - (self.temperature - 37) / 3)
        self.temperature = np.clip(self.temperature, 37, 180)
        # A deliberately simple degree-second surrogate, not CEM43 or Arrhenius.
        self.thermal_dose += dt * np.maximum(0, self.temperature - 43) / 100
        vessel_mask = np.array([kind == "artery" for kind in self.kinds])
        coagulating = vessel_mask & (self.temperature >= 50) & (self.temperature <= 70)
        self.coagulation[coagulating] = np.clip(self.coagulation[coagulating] + dt * 0.65 * power, 0, 1)
        # Thermal rupture reopens incompletely sealed injured vessels.
        self.coagulation[vessel_mask] *= np.maximum(0, 1 - dt * np.maximum(0, self.temperature[vessel_mask] - 85) / 30)

        damage = self.damage
        vessel_damage = damage[vessel_mask]
        # Flow is driven by remaining toy pressure, so lost volume feeds back.
        pressure_fraction = max(0.15, self.mean_arterial_pressure / 90)
        self.bleeding_rate = float(np.sum(np.maximum(0, vessel_damage - 0.08) *
                                         (1 - self.coagulation[vessel_mask])) * 55 * pressure_fraction)
        cystic = np.array([kind == "cystic_duct" for kind in self.kinds])
        common = np.array([kind == "common_bile_duct" for kind in self.kinds])
        self.bile_leak_rate = float(np.maximum(0, damage[cystic] - 0.08).sum() * 2.5 +
                                    np.maximum(0, damage[common] - 0.08).sum() * 5)
        self.blood_loss_ml += min(5000 - self.blood_loss_ml, self.bleeding_rate * dt)
        self.bile_leak_ml += self.bile_leak_rate * dt
        volume_fraction_lost = self.blood_loss_ml / 5000
        self.mean_arterial_pressure = float(np.clip(90 * (1 - 2.5 * volume_fraction_lost), 20, 90))
        self.heart_rate = float(np.clip(72 + 180 * volume_fraction_lost, 72, 180))
        torn = bool(self.contact_index is not None and damage[self.contact_index] >= 0.9)
        if torn:
            self.release_contact()
        return {"damage_increment": np.maximum(0, damage - previous_damage),
                "blood_loss_increment_ml": self.blood_loss_ml - previous_blood,
                "bile_leak_increment_ml": self.bile_leak_ml - previous_bile,
                "contact_torn": torn}

    def exposure(self) -> float:
        # Lateral movement of the lower-right neck marker opens an illustrative
        # projected triangle. Blood/bile obscure the view; damage is not progress.
        marker = 3
        opening = max(0, self.positions[marker, 0] - self.rest[marker, 0])
        geometric = opening / (10 * (0.8 + 0.2 * self.patient["distension"]))
        obscuration = min(0.8, self.blood_loss_ml / 250 + self.bile_leak_ml / 30)
        return float(np.clip(geometric - obscuration, 0, 1))

    def critical_damage(self) -> float:
        return float(max(self.damage[i] for i, kind in enumerate(self.kinds) if kind != "gallbladder"))

    def snapshot(self) -> dict[str, Any]:
        damage, thermal, stiffness = self.damage, self.thermal_damage, self.stiffness
        return {"nodes": [{"id": f"n{i:02d}", "kind": kind,
                           "rest_position_mm": self.rest[i].tolist(), "position_mm": self.positions[i].tolist(),
                           "damage": float(damage[i]), "mechanical_damage": float(self.mechanical_damage[i]),
                           "thermal_damage": float(thermal[i]), "temperature_c": float(self.temperature[i]),
                           "thermal_dose": float(self.thermal_dose[i]), "strain": float(self.strain[i]),
                           "stiffness_n_per_mm": float(stiffness[i]), "coagulation": float(self.coagulation[i]),
                           "torn": bool(damage[i] >= 0.9)} for i, kind in enumerate(self.kinds)],
                "edges": [[f"n{i:02d}", f"n{j:02d}"] for i, j in self.edges],
                "model": TISSUE_VERSION}
