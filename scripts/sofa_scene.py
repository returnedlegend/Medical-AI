"""Runtime-only topology adapter for the pinned upstream LapGym scene.

The VTK files supply boundary triangles, not every tetrahedral face. SOFA's
TetrahedronSetTopologyContainer treats a nonempty triangle list as complete.
Clear the inherited lower-dimensional topology before initialization so SOFA
derives all faces and edges from the unchanged tetrahedra. Upstream files and
tetrahedral connectivity remain untouched. See docs/physics-diagnostics.md.
"""

from sofa_env.scenes.grasp_lift_touch.scene_description import createScene as upstream_create_scene


def createScene(root_node, **kwargs):
    result = upstream_create_scene(root_node, **kwargs)
    for name in ("liver", "gallbladder"):
        topology = result[name].topology_container
        topology.triangles.value = []
        topology.edges.value = []
    return result
