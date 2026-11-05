"""Mechanistic invariants for the illustrative stateful tissue model."""
from __future__ import annotations
import unittest
import numpy as np
from retraction.tissue import NODE_COUNT, TissueModel

PATIENT = {"attachment_length_mm": 24.0, "adhesion_count": 2,
           "adhesion_stiffness_n_per_mm": 0.08, "distension": 1.2}


class TissueResponseTests(unittest.TestCase):
    def step_model(self, tissue, grasper=None, closure=0, cautery=None, power=0):
        return tissue.step(0.1, np.array([6, 3]) if grasper is None else grasper, closure,
                           np.array([-25, 35]) if cautery is None else cautery, power, 2.5, 0.6)

    def test_rest_is_mechanical_and_thermal_equilibrium_without_tools(self):
        tissue = TissueModel(PATIENT)
        for _ in range(50):
            self.step_model(tissue)
        np.testing.assert_array_equal(tissue.positions, tissue.rest)
        np.testing.assert_array_equal(tissue.damage, np.zeros(NODE_COUNT))
        np.testing.assert_array_equal(tissue.temperature, np.full(NODE_COUNT, 37.0))
        self.assertEqual(tissue.blood_loss_ml, 0)
        self.assertEqual(tissue.bile_leak_ml, 0)

    def test_contact_requires_proximity_and_sufficient_closure(self):
        tissue = TissueModel(PATIENT)
        self.assertFalse(tissue.acquire_contact(np.array([29, 39]), 1))
        self.assertFalse(tissue.acquire_contact(tissue.rest[3], 0.1))
        self.assertTrue(tissue.acquire_contact(tissue.rest[3], 0.55))
        self.assertEqual(tissue.contact_index, 3)
        self.assertTrue(tissue.release_contact())
        self.assertFalse(tissue.release_contact())

    def test_excessive_grip_causes_irreversible_crush_and_stiffness_loss(self):
        tissue = TissueModel(PATIENT)
        tissue.acquire_contact(tissue.rest[3], 1)
        original_k = tissue.stiffness[3]
        for _ in range(12):
            self.step_model(tissue, tissue.rest[3], closure=1)
        self.assertGreater(tissue.mechanical_damage[3], 0.2)
        self.assertLess(tissue.stiffness[3], original_k)
        injury = tissue.damage.copy()
        tissue.release_contact()
        for _ in range(20):
            self.step_model(tissue)
        self.assertTrue(np.all(tissue.damage >= injury))

    def test_injury_alters_the_next_mechanical_response(self):
        intact = TissueModel(PATIENT)
        injured = TissueModel(PATIENT)
        injured.acquire_contact(injured.rest[3], 1)
        for _ in range(12):
            self.step_model(injured, injured.rest[3], closure=1)
        injured.release_contact()
        for tissue in (intact, injured):
            tissue.acquire_contact(tissue.rest[3], 0.55)
            for _ in range(10):
                self.step_model(tissue, tissue.rest[3] + [3, 0], closure=0.55)
        self.assertLess(injured.contact_force_n, intact.contact_force_n)
        self.assertFalse(np.allclose(injured.positions, intact.positions))
        self.assertLess(injured.stiffness[3], intact.stiffness[3])

    def test_traction_overload_causes_mechanical_damage_without_heat(self):
        tissue = TissueModel(PATIENT)
        tissue.acquire_contact(tissue.rest[3], 0.6)
        peak_force, peak_strain = 0.0, 0.0
        for _ in range(30):
            self.step_model(tissue, np.array([30, 3]), closure=0.6)
            peak_force = max(peak_force, tissue.contact_force_n)
            peak_strain = max(peak_strain, float(tissue.strain.max()))
        self.assertGreater(peak_force, 2.5)
        self.assertGreater(tissue.mechanical_damage.max(), 0)
        self.assertTrue(np.all(tissue.thermal_dose == 0))
        self.assertGreater(peak_strain, 0)

    def test_cautery_dose_persists_after_cooling_and_softens_tissue(self):
        tissue = TissueModel(PATIENT)
        target = tissue.rest[8].copy()
        for _ in range(12):
            self.step_model(tissue, cautery=target, power=1)
        hot_temp = tissue.temperature[8]
        damage = tissue.damage.copy()
        self.assertGreater(hot_temp, 70)
        self.assertGreater(tissue.thermal_dose[8], 0)
        self.assertLess(tissue.stiffness[8], tissue.base_stiffness[8])
        for _ in range(60):
            self.step_model(tissue)
        self.assertLess(tissue.temperature[8], hot_temp)
        self.assertTrue(np.all(tissue.damage >= damage))
        self.assertGreater(tissue.thermal_damage[8], 0)

    def test_cautery_in_empty_space_does_not_heat_tissue(self):
        tissue = TissueModel(PATIENT)
        for _ in range(20):
            self.step_model(tissue, cautery=np.array([30, 40]), power=1)
        self.assertTrue(np.all(tissue.temperature == 37))
        self.assertTrue(np.all(tissue.thermal_dose == 0))

    def test_vessel_injury_causes_loss_pressure_drop_and_tachycardia(self):
        tissue = TissueModel(PATIENT)
        for _ in range(18):
            self.step_model(tissue, cautery=tissue.rest[16], power=1)
        self.assertGreater(tissue.damage[16], 0.5)
        self.assertGreater(tissue.bleeding_rate, 0)
        self.assertGreater(tissue.blood_loss_ml, 0)
        self.assertLess(tissue.mean_arterial_pressure, 90)
        self.assertGreater(tissue.heart_rate, 72)
        self.assertEqual(tissue.bile_leak_ml, 0)
        loss = tissue.blood_loss_ml
        self.step_model(tissue)
        self.assertGreaterEqual(tissue.blood_loss_ml, loss)

    def test_duct_injury_causes_bile_leak_and_visual_obscuration(self):
        tissue = TissueModel(PATIENT)
        tissue.positions[3, 0] += 5  # known marker opening for view comparison
        for _ in range(18):
            self.step_model(tissue, cautery=tissue.rest[21], power=1)
        self.assertGreater(tissue.damage[21], 0.5)
        self.assertGreater(tissue.bile_leak_rate, 0)
        self.assertGreater(tissue.bile_leak_ml, 0)
        actual_exposure = tissue.exposure()
        leaked = tissue.bile_leak_ml
        tissue.bile_leak_ml = 0
        self.assertGreater(tissue.exposure(), actual_exposure)
        tissue.bile_leak_ml = leaked
        self.assertEqual(tissue.blood_loss_ml, 0)

    def test_snapshot_contains_changed_geometry_and_irreversible_material_state(self):
        tissue = TissueModel(PATIENT)
        tissue.acquire_contact(tissue.rest[3], 0.6)
        self.step_model(tissue, tissue.rest[3] + [4, 0], closure=0.6)
        snapshot = tissue.snapshot()
        self.assertEqual(len(snapshot["nodes"]), NODE_COUNT)
        self.assertNotEqual(snapshot["nodes"][3]["position_mm"], snapshot["nodes"][3]["rest_position_mm"])
        import json
        json.dumps(snapshot, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
