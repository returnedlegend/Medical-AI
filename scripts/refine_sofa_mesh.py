#!/usr/bin/env python3
"""Read-only, one-level conforming 1-to-8 tetrahedral midpoint refinement.

Original points retain their indices. Outputs must be new files outside the
input asset directory. Refinement changes discretization, not tissue validity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np


def read_tetra_mesh(path):
    from audit_sofa_meshes import read_legacy_binary_vtk
    path = Path(path)
    with path.open('rb') as stream:
        header = [stream.readline().decode('ascii').strip() for _ in range(4)]
    if header[2] == 'BINARY':
        points, cells = read_legacy_binary_vtk(path)
        return points, np.asarray(cells[10], dtype=np.int64)
    if header[2:] != ['ASCII', 'DATASET UNSTRUCTURED_GRID']:
        raise ValueError('Unsupported VTK encoding')
    tokens = path.read_text().splitlines()[4:]
    fields = ' '.join(tokens).split()
    if fields[0] != 'POINTS':
        raise ValueError('Expected POINTS')
    n = int(fields[1]); offset = 3
    points = np.asarray(fields[offset:offset + n * 3], dtype=float).reshape(n, 3); offset += n * 3
    if fields[offset] != 'CELLS':
        raise ValueError('Expected CELLS')
    count, size = map(int, fields[offset + 1:offset + 3]); offset += 3
    raw = np.asarray(fields[offset:offset + size], dtype=np.int64).reshape(count, 5)
    if not np.all(raw[:, 0] == 4):
        raise ValueError('Only tetrahedral cells are supported in generated ASCII files')
    offset += size
    if fields[offset:offset + 2] != ['CELL_TYPES', str(count)] or not all(int(x) == 10 for x in fields[offset + 2:offset + 2 + count]):
        raise ValueError('Expected tetrahedral CELL_TYPES')
    return points, raw[:, 1:]


def signed_volumes(points, tets):
    p = np.asarray(points)[np.asarray(tets)]
    return np.linalg.det(np.transpose(p[:, 1:] - p[:, :1], (0, 2, 1))) / 6


def mesh_geometry(points, tets):
    volume = signed_volumes(points, tets)
    faces = Counter(tuple(sorted(face)) for tet in tets for face in combinations(map(int, tet), 3))
    boundary = {face for face, count in faces.items() if count == 1}
    boundary_edges = Counter(tuple(sorted(edge)) for face in boundary for edge in combinations(face, 2))
    p = points[tets]
    edge_sum = sum(np.sum((p[:, i] - p[:, j])**2, axis=1) for i, j in combinations(range(4), 2))
    quality = 12 * np.power(3 * np.abs(volume), 2 / 3) / edge_sum
    return {'points': len(points), 'tetrahedra': len(tets), 'total_volume_source_units_cubed': float(volume.sum()),
            'minimum_signed_volume': float(volume.min()), 'nonpositive_tetrahedra': int((volume <= 0).sum()),
            'unique_faces': len(faces), 'boundary_faces': len(boundary),
            'nonmanifold_face_count': sum(count > 2 for count in faces.values()),
            'nonmanifold_boundary_edge_count': sum(count != 2 for count in boundary_edges.values()),
            'unused_points': sorted(set(range(len(points))) - set(tets.ravel().tolist())),
            'quality_mean_ratio_min': float(quality.min()), 'quality_mean_ratio_median': float(np.median(quality)),
            'quality_mean_ratio_max': float(quality.max())}, boundary


def refine_tetrahedra(points, tets):
    points = np.asarray(points, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite three-dimensional points')
    if tets.ndim != 2 or tets.shape[1] != 4 or len(tets) == 0 or tets.min() < 0 or tets.max() >= len(points):
        raise ValueError('Expected nonempty valid tetrahedral indices')
    original_volumes = signed_volumes(points, tets)
    if (original_volumes <= 0).any():
        raise ValueError('Input tetrahedra must have positive nonzero orientation')
    edges = sorted({tuple(sorted(edge)) for tet in tets for edge in combinations(map(int, tet), 2)})
    midpoint_index = {edge: len(points) + index for index, edge in enumerate(edges)}
    midpoints = np.array([(points[a] + points[b]) / 2 for a, b in edges])
    refined_points = np.concatenate([points, midpoints])
    children = []
    for a, b, c, d in tets.tolist():
        ab, ac, ad, bc, bd, cd = [midpoint_index[tuple(sorted(pair))] for pair in [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]]
        children.extend([(a,ab,ac,ad),(b,ab,bc,bd),(c,ac,bc,cd),(d,ad,bd,cd),
                         (ab,ac,ad,cd),(ab,ac,bc,cd),(ab,bc,bd,cd),(ab,ad,bd,cd)])
    refined_tets = np.asarray(children, dtype=np.int64)
    negative = signed_volumes(refined_points, refined_tets) < 0
    refined_tets[negative, 0], refined_tets[negative, 1] = refined_tets[negative, 1].copy(), refined_tets[negative, 0].copy()
    volumes = signed_volumes(refined_points, refined_tets)
    rel = np.abs(volumes.reshape(-1,8).sum(axis=1) - original_volumes) / original_volumes
    coarse, coarse_boundary = mesh_geometry(points, tets)
    fine, fine_boundary = mesh_geometry(refined_points, refined_tets)
    expected_boundary = set()
    for a, b, c in coarse_boundary:
        ab, ac, bc = [midpoint_index[tuple(sorted(pair))] for pair in [(a,b),(a,c),(b,c)]]
        expected_boundary.update(tuple(sorted(face)) for face in [(a,ab,ac),(b,ab,bc),(c,ac,bc),(ab,ac,bc)])
    report = {'scheme': 'one-level edge-midpoint 1-to-8 red refinement; fixed central diagonal',
              'coarse': coarse, 'fine': fine,
              'original_vertices_preserved_exactly': bool(np.array_equal(refined_points[:len(points)], points)),
              'new_shared_midpoints': len(edges), 'max_per_parent_relative_volume_error': float(rel.max()),
              'refined_boundary_matches_shared_face_subdivision': expected_boundary == fine_boundary,
              'limitations': ['Fixed diagonal choice can worsen element shape; refinement does not guarantee mesh quality.',
                              'Original orphan vertices are retained to preserve indices.',
                              'Positive volume and manifold incidence do not prove anatomical/physical validity.']}
    if (fine['nonpositive_tetrahedra'] or fine['nonmanifold_face_count'] or fine['nonmanifold_boundary_edge_count']
            or rel.max() > 1e-9 or expected_boundary != fine_boundary):
        raise RuntimeError('Refinement failed geometric invariants')
    return refined_points, refined_tets, report


def write_tetra_vtk(path, points, tets):
    with Path(path).open('x', encoding='ascii') as stream:
        stream.write('# vtk DataFile Version 4.2\nProject-owned conforming tetrahedral refinement\nASCII\nDATASET UNSTRUCTURED_GRID\n')
        stream.write(f'POINTS {len(points)} double\n')
        for p in points:
            stream.write(' '.join(format(float(x), '.17g') for x in p) + '\n')
        stream.write(f'CELLS {len(tets)} {len(tets)*5}\n')
        for tet in tets:
            stream.write('4 ' + ' '.join(str(int(x)) for x in tet) + '\n')
        stream.write(f'CELL_TYPES {len(tets)}\n')
        stream.write('10\n' * len(tets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.input.resolve().parent):
        parser.error('Generated output must be outside the upstream asset directory')
    points, tets = read_tetra_mesh(args.input)
    fine_points, fine_tets, report = refine_tetrahedra(points, tets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_tetra_vtk(args.output, fine_points, fine_tets)
    parsed_points, parsed_tets = read_tetra_mesh(args.output)
    if not np.array_equal(fine_points, parsed_points) or not np.array_equal(fine_tets, parsed_tets):
        raise RuntimeError('Generated VTK failed exact roundtrip')
    report.update(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    with args.output.with_suffix('.refinement.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
