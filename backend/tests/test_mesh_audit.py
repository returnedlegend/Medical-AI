"""Readiness-gate regressions using synthetic tetrahedra, without SOFA."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/audit_sofa_meshes.py"
SPEC = importlib.util.spec_from_file_location("mesh_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class MeshAuditTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
        self.tets = [(0, 1, 2, 3), (0, 2, 1, 4)]
        faces = Counter(tuple(sorted(face)) for tet in self.tets for face in combinations(tet, 3))
        self.source = {10: self.tets,
                       5: [triangle for face, count in faces.items() if count == 1 for triangle in (face, face[::-1])]}
        self.observed = {"component_state": "Undefined", "vertex_count": 5,
                         "tetrahedra": [list(tet) for tet in self.tets],
                         "triangles": [list(face) for face in faces],
                         "edges": [list(edge) for edge in sorted({tuple(sorted(edge)) for tet in self.tets for edge in combinations(tet, 2)})]}

    def report(self, observed=None, errors=0, topology_errors=0, mode="derived"):
        observed = observed if observed is not None else self.observed
        return {"source": {"liver": audit.connectivity_report(self.points, self.source)},
                "scene": {"topology_mode": mode,
                          "diagnostic_counts": {"error_count": errors, "topology_error_count": topology_errors},
                          "topology": {name: audit.compare_scene_topology(self.source, observed, 5)
                                       for name in ("liver", "gallbladder")}}}

    def test_complete_derivation_passes_without_relabeling_undefined(self):
        report = self.report()
        self.assertTrue(audit.validate_scene_audit(report))
        state = report["scene"]["topology"]["liver"]
        self.assertEqual(state["component_state"], "Undefined")
        self.assertFalse(state["component_explicitly_valid"])
        self.assertEqual(report["source"]["liver"]["missing_tetrahedral_faces_count"], 1)

    def test_missing_face_fails(self):
        broken = deepcopy(self.observed)
        broken["triangles"].remove([0, 1, 2])  # shared interior face
        report = self.report(broken)
        self.assertFalse(audit.validate_scene_audit(report))
        self.assertEqual(report["scene"]["topology"]["liver"]["missing_derived_face_count"], 1)

    def test_extra_edge_fails(self):
        broken = deepcopy(self.observed)
        broken["edges"].append([3, 4])  # tips never share a tetrahedron
        self.assertFalse(audit.validate_scene_audit(self.report(broken)))

    def test_explicitly_invalid_component_fails_even_with_complete_arrays(self):
        broken = {**self.observed, "component_state": "Invalid"}
        report = self.report(broken)
        self.assertTrue(report["scene"]["topology"]["liver"]["connectivity_matches_tetrahedral_derivation"])
        self.assertFalse(audit.validate_scene_audit(report))

    def test_scene_errors_fail(self):
        self.assertFalse(audit.validate_scene_audit(self.report(errors=1)))
        self.assertFalse(audit.validate_scene_audit(self.report(topology_errors=1)))

    def test_original_mode_and_unchanged_boundary_triangles_fail(self):
        self.assertFalse(audit.validate_scene_audit(self.report(mode="original")))
        original = {**self.observed, "triangles": [list(triangle) for triangle in self.source[5]]}
        report = self.report(original)
        self.assertFalse(audit.validate_scene_audit(report))
        self.assertGreater(report["scene"]["topology"]["liver"]["duplicate_triangle_count"], 0)

    def test_tetrahedral_orientation_change_fails_despite_same_vertex_sets(self):
        broken = deepcopy(self.observed)
        broken["tetrahedra"][0] = [1, 0, 2, 3]
        report = self.report(broken)
        self.assertTrue(report["scene"]["topology"]["liver"]["tetrahedral_connectivity_unchanged"])
        self.assertFalse(report["scene"]["topology"]["liver"]["tetrahedra_index_order_unchanged"])
        self.assertFalse(audit.validate_scene_audit(report))

    def test_missing_diagnostics_or_component_fail_closed(self):
        self.assertFalse(audit.validate_scene_audit({}))
        report = self.report()
        del report["scene"]["diagnostic_counts"]["error_count"]
        self.assertFalse(audit.validate_scene_audit(report))
        report = self.report()
        del report["scene"]["topology"]["gallbladder"]
        self.assertFalse(audit.validate_scene_audit(report))

    def write_vtk(self, path):
        cells = [(5, face) for face in self.source[5]] + [(10, tet) for tet in self.tets]
        encoded_cells = np.asarray([value for _, cell in cells for value in (len(cell), *cell)], dtype=">i4")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.write(b"# vtk DataFile Version 4.2\nsynthetic fixture\nBINARY\nDATASET UNSTRUCTURED_GRID\nPOINTS 5 float\n")
            stream.write(self.points.astype(">f4").tobytes())
            stream.write(f"\nCELLS {len(cells)} {len(encoded_cells)}\n".encode())
            stream.write(encoded_cells.tobytes())
            stream.write(f"\nCELL_TYPES {len(cells)}\n".encode())
            stream.write(np.asarray([kind for kind, _ in cells], dtype=">i4").tobytes())
            stream.write(b"\n")

    def test_cli_emits_report_and_nonzero_status_for_scene_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("liver", "gallbladder"):
                self.write_vtk(root / f"source/sofa_env/scenes/grasp_lift_touch/meshes/{name}_volumetric.vtk")
            scene_path, output_path = root / "scene.json", root / "report.json"
            scene = {"topology_mode": "derived", "diagnostics": {"error_count": 1, "topology_error_count": 0},
                     "topology": {name: self.observed for name in ("liver", "gallbladder")}}
            scene_path.write_text(json.dumps(scene))
            command = [sys.executable, str(SCRIPT), "--runtime", str(root), "--scene-json", str(scene_path),
                       "--output", str(output_path), "--strict-scene"]
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1, failed.stderr)
            self.assertFalse(json.loads(failed.stdout)["strict_scene_check"]["passed"])
            self.assertFalse(json.loads(output_path.read_text())["strict_scene_check"]["passed"])
            scene["diagnostics"]["error_count"] = 0
            scene_path.write_text(json.dumps(scene))
            passed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertTrue(json.loads(passed.stdout)["strict_scene_check"]["passed"])

    def test_cli_strict_requires_scene_json(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--strict-scene"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--strict-scene requires --scene-json", result.stderr)


if __name__ == "__main__":
    unittest.main()
