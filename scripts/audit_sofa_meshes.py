#!/usr/bin/env python3
"""Read-only connectivity audit of pinned upstream VTKs and optional scene output.

Requires NumPy, available in both project and optional SOFA virtualenvs. This is
an input/connectivity diagnostic, not a proof of physical model accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_legacy_binary_vtk(path: Path) -> tuple[np.ndarray, dict[int, list[tuple[int, ...]]]]:
    """Decode the specific legacy binary format used by pinned upstream assets."""
    with path.open("rb") as stream:
        header = [stream.readline().decode("ascii").strip() for _ in range(4)]
        if header[2:] != ["BINARY", "DATASET UNSTRUCTURED_GRID"]:
            raise ValueError(f"Unsupported VTK encoding in {path}")
        declaration = stream.readline().decode("ascii").split()
        if declaration[0] != "POINTS" or declaration[2] != "float":
            raise ValueError("Expected POINTS count float (big-endian float32)")
        count = int(declaration[1])
        points = np.frombuffer(stream.read(count * 3 * 4), dtype=">f4").reshape(count, 3).astype(float)
        if stream.readline() != b"\n":
            raise ValueError("Unexpected POINTS terminator")
        declaration = stream.readline().decode("ascii").split()
        if declaration[0] != "CELLS":
            raise ValueError("Expected legacy CELLS declaration")
        cell_count, size = map(int, declaration[1:])
        raw = np.frombuffer(stream.read(size * 4), dtype=">i4")
        if stream.readline() != b"\n":
            raise ValueError("Unexpected CELLS terminator")
        declaration = stream.readline().decode("ascii").split()
        if declaration != ["CELL_TYPES", str(cell_count)]:
            raise ValueError("Expected one CELL_TYPE per cell")
        types = np.frombuffer(stream.read(cell_count * 4), dtype=">i4")
    grouped: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    offset = 0
    for cell_type in types:
        width = int(raw[offset])
        grouped[int(cell_type)].append(tuple(map(int, raw[offset + 1:offset + 1 + width])))
        offset += width + 1
    if offset != size:
        raise ValueError("Inconsistent CELLS size")
    return points, grouped


def connectivity_report(points: np.ndarray, cells: dict[int, list[tuple[int, ...]]]) -> dict[str, Any]:
    tetrahedra, triangles, edges = cells.get(10, []), cells.get(5, []), cells.get(3, [])
    all_cells = [cell for group in cells.values() for cell in group]
    bad_indices = sum(any(vertex < 0 or vertex >= len(points) for vertex in cell) for cell in all_cells)
    if bad_indices:
        raise ValueError(f"Cannot audit geometry with {bad_indices} out-of-range cells")
    if not tetrahedra or any(len(tet) != 4 for tet in tetrahedra):
        raise ValueError("Expected nonempty four-index tetrahedral cells")
    faces = Counter(tuple(sorted(face)) for tet in tetrahedra for face in combinations(tet, 3))
    tet_edges = {tuple(sorted(edge)) for tet in tetrahedra for edge in combinations(tet, 2)}
    tri_counts = Counter(tuple(sorted(triangle)) for triangle in triangles)
    tri_edges = {tuple(sorted(edge)) for triangle in triangles for edge in combinations(triangle, 2)}
    boundary = {face for face, count in faces.items() if count == 1}
    boundary_edges = Counter(tuple(sorted(edge)) for face in boundary for edge in combinations(face, 2))
    signed_volumes = np.array([np.linalg.det(np.column_stack([
        points[tet[i]] - points[tet[0]] for i in range(1, 4)])) / 6 for tet in tetrahedra])
    adjacency: dict[int, set[int]] = defaultdict(set)
    for tet in tetrahedra:
        for a, b in combinations(tet, 2):
            adjacency[a].add(b); adjacency[b].add(a)
    seen: set[int] = set()
    components = []
    for vertex in adjacency:
        if vertex in seen:
            continue
        queue, component = [vertex], set()
        while queue:
            current = queue.pop()
            if current in component:
                continue
            component.add(current)
            queue.extend(adjacency[current] - component)
        seen.update(component)
        components.append(len(component))
    return {
        "point_count": len(points),
        "cell_types": {str(key): len(value) for key, value in cells.items()},
        "tetrahedron_count": len(tetrahedra),
        "duplicate_tetrahedron_count": len(tetrahedra) - len({tuple(sorted(tet)) for tet in tetrahedra}),
        "repeated_vertex_tetrahedron_count": sum(len(set(tet)) < 4 for tet in tetrahedra),
        "bad_index_count": bad_indices,
        "positive_signed_volume_count": int((signed_volumes > 0).sum()),
        "negative_signed_volume_count": int((signed_volumes < 0).sum()),
        "near_zero_volume_count": int((np.abs(signed_volumes) < 1e-10).sum()),
        "min_abs_volume_source_units_cubed": float(np.abs(signed_volumes).min()),
        "max_abs_volume_source_units_cubed": float(np.abs(signed_volumes).max()),
        "unique_supplied_triangle_count": len(tri_counts),
        "supplied_triangle_multiplicity_histogram": dict(Counter(tri_counts.values())),
        "derived_face_count": len(faces), "derived_edge_count": len(tet_edges),
        "derived_face_incidence_histogram": dict(Counter(faces.values())),
        "derived_boundary_face_count": len(boundary),
        "derived_boundary_edge_incidence_histogram": dict(Counter(boundary_edges.values())),
        "missing_tetrahedral_faces_count": len(set(faces) - set(tri_counts)),
        "missing_boundary_faces_count": len(boundary - set(tri_counts)),
        "extra_non_tetrahedral_triangles": [list(triangle) for triangle in sorted(set(tri_counts) - set(faces))],
        "non_boundary_supplied_triangles_count": len(set(tri_counts) - boundary),
        "triangle_edges_absent_from_tetrahedra": [list(edge) for edge in sorted(tri_edges - tet_edges)],
        "used_vertex_component_sizes": sorted(components, reverse=True),
        "unused_vertex_indices": sorted(set(range(len(points))) - seen),
        "coincident_point_surplus": len(points) - len(np.unique(points, axis=0)),
    }


def compare_scene_topology(source: dict[int, list[tuple[int, ...]]], observed: dict[str, Any],
                           vertex_count: int) -> dict[str, Any]:
    expected_tets = Counter(tuple(sorted(tet)) for tet in source[10])
    actual_tets = Counter(tuple(sorted(tet)) for tet in observed["tetrahedra"])
    expected_faces = {tuple(sorted(face)) for tet in source[10] for face in combinations(tet, 3)}
    expected_edges = {tuple(sorted(edge)) for tet in source[10] for edge in combinations(tet, 2)}
    actual_faces = Counter(tuple(sorted(face)) for face in observed["triangles"])
    actual_edges = Counter(tuple(sorted(edge)) for edge in observed["edges"])
    state = observed.get("component_state")
    result = {
        "component_state": state,
        "component_explicitly_invalid": state == "Invalid",
        "component_explicitly_valid": state == "Valid",
        "vertex_count_unchanged": observed["vertex_count"] == vertex_count,
        "tetrahedral_connectivity_unchanged": actual_tets == expected_tets,
        "tetrahedra_index_order_unchanged": [list(tet) for tet in source[10]] == observed["tetrahedra"],
        "triangle_count": len(observed["triangles"]), "edge_count": len(observed["edges"]),
        "missing_derived_face_count": len(expected_faces - set(actual_faces)),
        "extra_triangle_count": len(set(actual_faces) - expected_faces),
        "duplicate_triangle_count": sum(actual_faces.values()) - len(actual_faces),
        "missing_derived_edge_count": len(expected_edges - set(actual_edges)),
        "extra_edge_count": len(set(actual_edges) - expected_edges),
        "duplicate_edge_count": sum(actual_edges.values()) - len(actual_edges),
    }
    result["connectivity_matches_tetrahedral_derivation"] = bool(
        result["vertex_count_unchanged"] and result["tetrahedral_connectivity_unchanged"] and
        all(result[key] == 0 for key in ["missing_derived_face_count", "extra_triangle_count",
            "duplicate_triangle_count", "missing_derived_edge_count", "extra_edge_count", "duplicate_edge_count"]))
    return result


def validate_scene_audit(report: dict[str, Any]) -> bool:
    """Gate corrected-scene connectivity readiness, never physical validity.

    Undefined component states are retained as Undefined. They are accepted for
    this narrow check because SOFA's topology component does not set Valid in
    the successful initialization path; missing observations still fail closed.
    """
    scene = report.get("scene", {})
    if scene.get("topology_mode") != "derived":
        return False
    counts = scene.get("diagnostic_counts", {})
    for name in ("error_count", "topology_error_count"):
        if type(counts.get(name)) is not int or counts[name] != 0:
            return False
    topology = scene.get("topology", {})
    for name in ("liver", "gallbladder"):
        component = topology.get(name, {})
        if (component.get("connectivity_matches_tetrahedral_derivation") is not True
                or component.get("tetrahedra_index_order_unchanged") is not True
                or component.get("component_explicitly_invalid") is not False
                or component.get("component_state") not in ("Undefined", "Valid")):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path("/workspace/.sofa-runtime"))
    parser.add_argument("--scene-json", type=Path, help="Optional sofa_smoke.py result containing exported topology arrays")
    parser.add_argument("--output", type=Path, help="Write JSON report in addition to stdout")
    parser.add_argument("--baseline-scene-json", type=Path, help="Optional same-protocol trajectory for numerical comparison")
    parser.add_argument("--strict-scene", action="store_true",
                        help="Require a derived scene with matching connectivity and no observed SOFA errors; failure exits 1 after emitting the report")
    args = parser.parse_args()
    if args.strict_scene and not args.scene_json:
        parser.error("--strict-scene requires --scene-json")
    meshes = args.runtime / "source/sofa_env/scenes/grasp_lift_touch/meshes"
    report: dict[str, Any] = {
        "scope": "Read-only static input/connectivity checks; not physical or clinical validation.",
        "source": {},
    }
    scene = json.loads(args.scene_json.read_text()) if args.scene_json else None
    if scene:
        report["scene"] = {"topology_mode": scene.get("topology_mode"),
                           "diagnostic_counts": {key: scene.get("diagnostics", {}).get(key) for key in
                                                 ["error_count", "warning_count", "topology_error_count"]},
                           "topology": {}}
    for name in ["liver", "gallbladder"]:
        path = meshes / f"{name}_volumetric.vtk"
        points, cells = read_legacy_binary_vtk(path)
        report["source"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  **connectivity_report(points, cells)}
        if scene:
            report["scene"]["topology"][name] = compare_scene_topology(cells, scene["topology"][name], len(points))
    if args.baseline_scene_json:
        if not scene:
            parser.error("--baseline-scene-json requires --scene-json")
        baseline = json.loads(args.baseline_scene_json.read_text())
        protocol_keys = ["seed", "time_step_s", "commands", "start_grasped", "requested_steps"]
        same_protocol = all(scene.get(key) == baseline.get(key) for key in protocol_keys)
        comparison = {"protocol_fields_equal": same_protocol,
                      "protocol_fields_compared": protocol_keys,
                      "step_records_exactly_equal": scene["steps"] == baseline["steps"],
                      "initial_vertices_exactly_equal": scene["initial_vertices"] == baseline["initial_vertices"],
                      "final_vertices_exactly_equal": scene["final_vertices"] == baseline["final_vertices"]}
        if len(scene["final_vertices"]) == len(baseline["final_vertices"]):
            comparison["max_final_coordinate_difference_scene_units"] = float(np.max(np.abs(
                np.asarray(scene["final_vertices"]) - np.asarray(baseline["final_vertices"]))))
        report["trajectory_comparison"] = comparison
    if args.strict_scene:
        report["strict_scene_check"] = {
            "passed": validate_scene_audit(report),
            "scope": "Derived topology/connectivity readiness only; Undefined remains unvalidated, and this is not physical or clinical validation.",
        }
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    if args.strict_scene and not report["strict_scene_check"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
