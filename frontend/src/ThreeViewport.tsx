import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { TissueState } from "./Procedure";
interface Props {
  state: TissueState;
  onTarget: (x: number, y: number) => void;
  onUnavailable: () => void;
}
const faces = [
  [0, 1, 2],
  [0, 2, 3],
  [1, 4, 5],
  [1, 5, 2],
  [2, 5, 6],
  [2, 6, 3],
  [3, 6, 7],
  [4, 8, 9],
  [4, 9, 5],
  [5, 9, 10],
  [5, 10, 6],
  [6, 10, 11],
  [6, 11, 7],
  [8, 12, 9],
  [9, 12, 13],
  [9, 13, 10],
  [10, 13, 14],
  [10, 14, 11],
  [12, 15, 13],
  [13, 15, 14],
];
const edge = [0, 1, 4, 8, 12, 15, 14, 11, 7, 3];
export default function ThreeViewport({
  state,
  onTarget,
  onUnavailable,
}: Props) {
  const host = useRef<HTMLDivElement>(null),
    resetCamera = useRef<(() => void) | null>(null),
    stateRef = useRef(state),
    targetRef = useRef(onTarget),
    failRef = useRef(onUnavailable);
  stateRef.current = state;
  targetRef.current = onTarget;
  failRef.current = onUnavailable;
  useEffect(() => {
    const mount = host.current;
    if (!mount) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        powerPreference: "low-power",
        preserveDrawingBuffer: true,
      });
    } catch {
      failRef.current();
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor("#14232b");
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    renderer.domElement.setAttribute(
      "aria-label",
      "Three-dimensional illustrative tissue surface. Depth is procedural; XY node positions and injury state come from the environment.",
    );
    renderer.domElement.setAttribute("role", "img");
    renderer.domElement.tabIndex = 0;
    const contextLost = (event: Event) => {
      event.preventDefault();
      failRef.current();
    };
    renderer.domElement.addEventListener("webglcontextlost", contextLost);
    mount.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog("#14232b", 100, 210);
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 500);
    camera.position.set(36, -54, 100);
    camera.up.set(0, 1, 0);
    const controls = new OrbitControls(camera, renderer.domElement);
    let cameraChanged = true;
    const invalidateCamera = () => {
      cameraChanged = true;
    };
    controls.addEventListener("change", invalidateCamera);
    controls.target.set(0, 6, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 45;
    controls.maxDistance = 170;
    controls.maxPolarAngle = Math.PI * 0.92;
    controls.update();
    controls.saveState();
    resetCamera.current = () => controls.reset();
    scene.add(new THREE.HemisphereLight("#eff5f4", "#3a4241", 2.3));
    const key = new THREE.DirectionalLight("#fff2df", 3.2);
    key.position.set(-25, 45, 70);
    scene.add(key);
    const fill = new THREE.PointLight("#9bc4dd", 4500);
    fill.position.set(35, -25, 45);
    scene.add(fill);
    const materials: THREE.Material[] = [],
      geometries: THREE.BufferGeometry[] = [];
    const material = (color: string, roughness = 0.5, metalness = 0) => {
      const m = new THREE.MeshStandardMaterial({ color, roughness, metalness });
      materials.push(m);
      return m;
    };
    const liverMat = material("#875b52", 0.67),
      shaftMat = material("#a8b8c0", 0.25, 0.8),
      toolMat = material("#4a7580", 0.3, 0.65),
      ductMat = material("#b0ae6e", 0.55),
      arteryMat = material("#ae6669", 0.5);
    const sphereGeo = new THREE.SphereGeometry(1, 32, 24);
    geometries.push(sphereGeo);
    const lobe = (pos: number[], scale: number[]) => {
      const mesh = new THREE.Mesh(sphereGeo, liverMat);
      mesh.position.set(pos[0], pos[1], pos[2]);
      mesh.scale.set(scale[0], scale[1], scale[2]);
      scene.add(mesh);
    };
    lobe([-10, 29, -8], [24, 12, 7]);
    lobe([17, 27, -9], [22, 10, 6]);
    lobe([-26, 22, -9], [9, 10, 5]);
    const bedGeo = new THREE.PlaneGeometry(150, 150);
    geometries.push(bedGeo);
    const bed = new THREE.Mesh(bedGeo, material("#243b37", 0.9));
    bed.position.z = -16;
    scene.add(bed);
    const grid = new THREE.GridHelper(100, 20, "#405e64", "#304c51");
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -15.9;
    scene.add(grid);
    geometries.push(grid.geometry);
    materials.push(
      ...(Array.isArray(grid.material) ? grid.material : [grid.material]),
    );
    const bodyGeo = new THREE.BufferGeometry(),
      positions = new Float32Array(32 * 3),
      colors = new Float32Array(32 * 3);
    bodyGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    bodyGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometries.push(bodyGeo);
    const baseIndices = faces.flat();
    baseIndices.push(
      ...faces.flatMap((f) => [f[2] + 16, f[1] + 16, f[0] + 16]),
    );
    for (let i = 0; i < edge.length; i++) {
      const a = edge[i],
        b = edge[(i + 1) % edge.length];
      baseIndices.push(a, b, b + 16, a, b + 16, a + 16);
    }
    bodyGeo.setIndex(baseIndices);
    const bodyMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.43,
      metalness: 0.05,
      side: THREE.DoubleSide,
    });
    materials.push(bodyMat);
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    scene.add(body);
    const nodeMeshes = Array.from({ length: 23 }, (_, i) => {
      const m = new THREE.Mesh(
        sphereGeo,
        material(i < 16 ? "#a5bc9a" : i < 18 ? "#bc6b70" : "#b9b873", 0.5),
      );
      m.scale.setScalar(i < 16 ? 0.6 : 0.9);
      scene.add(m);
      return m;
    });
    const cylinderGeo = new THREE.CylinderGeometry(1, 1, 1, 16);
    geometries.push(cylinderGeo);
    const segment = (mat: THREE.Material, radius: number) => {
      const mesh = new THREE.Mesh(cylinderGeo, mat);
      mesh.userData.radius = radius;
      scene.add(mesh);
      return mesh;
    };
    const grasper = segment(shaftMat, 0.7),
      cautery = segment(toolMat, 0.6),
      jaws = [segment(shaftMat, 0.22), segment(shaftMat, 0.22)],
      cauteryTip = segment(shaftMat, 0.2);
    const tubeMeshes: THREE.Mesh[] = [];
    for (let i = 0; i < 5; i++) {
      const mesh = new THREE.Mesh(
        new THREE.BufferGeometry(),
        i === 0 ? arteryMat : ductMat,
      );
      scene.add(mesh);
      tubeMeshes.push(mesh);
    }
    const damageMeshes = Array.from({ length: 23 }, () => {
      const m = new THREE.Mesh(sphereGeo, material("#5b2531", 0.8));
      scene.add(m);
      return m;
    });
    const heatMat = new THREE.MeshBasicMaterial({
      color: "#ef963d",
      transparent: true,
      opacity: 0.22,
      depthWrite: false,
    });
    materials.push(heatMat);
    const heat = new THREE.Mesh(sphereGeo, heatMat);
    scene.add(heat);
    const bloodMat = material("#ad2539", 0.25),
      bileMat = material("#c1bc45", 0.32);
    const particles = Array.from({ length: 34 }, (_, i) => {
      const m = new THREE.Mesh(sphereGeo, i < 22 ? bloodMat : bileMat);
      m.scale.setScalar(0.3 + (i % 3) * 0.13);
      scene.add(m);
      return m;
    });
    const lineGeo = new THREE.BufferGeometry();
    geometries.push(lineGeo);
    const lineMat = new THREE.LineBasicMaterial({
      color: "#bcc8b8",
      transparent: true,
      opacity: 0.22,
    });
    materials.push(lineMat);
    const line = new THREE.LineSegments(lineGeo, lineMat);
    scene.add(line);
    const nodeZ = (i: number) =>
      i < 16
        ? 2 +
          4 *
            Math.sqrt(
              Math.max(
                0,
                1 -
                  (stateRef.current.tissue.nodes[i].rest_position_mm[0] / 11) **
                    2,
              ),
            ) *
            Math.sin(
              (Math.PI *
                (stateRef.current.tissue.nodes[i].rest_position_mm[1] + 4)) /
                33,
            )
        : 1;
    const vec = (p: number[], z = 3) => new THREE.Vector3(p[0], p[1], z);
    const connect = (
      mesh: THREE.Mesh,
      a: THREE.Vector3,
      b: THREE.Vector3,
      radius?: number,
    ) => {
      const dir = b.clone().sub(a);
      mesh.position.copy(a).add(b).multiplyScalar(0.5);
      mesh.scale.set(
        radius ?? mesh.userData.radius,
        dir.length(),
        radius ?? mesh.userData.radius,
      );
      mesh.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir.normalize(),
      );
    };
    let lastState: TissueState | null = null;
    const update = () => {
      const s = stateRef.current,
        nodes = s.tissue.nodes;
      if (nodes.length < 23) return;
      if (lastState !== s) {
        for (let i = 0; i < 23; i++) {
          const n = nodes[i],
            z = nodeZ(i),
            position = vec(n.position_mm, z);
          nodeMeshes[i].position.copy(position);
          const c = new THREE.Color(
            i < 16 ? "#8ba178" : i < 18 ? "#b96166" : "#b1b26a",
          );
          if (n.temperature_c > 43)
            c.lerp(
              new THREE.Color("#d98a45"),
              Math.min(1, (n.temperature_c - 43) / 45),
            );
          c.lerp(new THREE.Color("#5c2530"), n.damage);
          (nodeMeshes[i].material as THREE.MeshStandardMaterial).color.copy(c);
          damageMeshes[i].visible = n.damage > 0.4;
          damageMeshes[i].position
            .copy(position)
            .add(new THREE.Vector3(0, 0, 0.3));
          damageMeshes[i].scale.set(
            1 + n.damage * 1.5,
            0.4 + n.damage * 0.8,
            0.15,
          );
          if (i < 16) {
            for (let back = 0; back < 2; back++) {
              const offset = (i + back * 16) * 3;
              positions[offset] = n.position_mm[0];
              positions[offset + 1] = n.position_mm[1];
              positions[offset + 2] = back ? -1.4 : z;
              c.toArray(colors, offset);
            }
          }
        }
        bodyGeo.attributes.position.needsUpdate = true;
        bodyGeo.attributes.color.needsUpdate = true;
        bodyGeo.computeVertexNormals();
        bodyGeo.computeBoundingSphere();
        const byId = new Map(nodes.map((n, i) => [n.id, { n, i }]));
        const points: number[] = [];
        for (const [a, b] of s.tissue.edges) {
          const n = byId.get(a),
            m = byId.get(b);
          if (!n || !m) continue;
          points.push(
            ...vec(n.n.position_mm, nodeZ(n.i) + 0.15).toArray(),
            ...vec(m.n.position_mm, nodeZ(m.i) + 0.15).toArray(),
          );
        }
        lineGeo.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(points, 3),
        );
        const paths = [
          [16, 17],
          [18, 19],
          [19, 20],
          [20, 21],
          [21, 22],
        ];
        paths.forEach((ids, i) => {
          const a = vec(nodes[ids[0]].position_mm, 1),
            b = vec(nodes[ids[1]].position_mm, 1);
          const curve = new THREE.CatmullRomCurve3([
            a,
            a
              .clone()
              .lerp(b, 0.5)
              .add(new THREE.Vector3(0.3, 0, 0.2)),
            b,
          ]);
          tubeMeshes[i].geometry.dispose();
          tubeMeshes[i].geometry = new THREE.TubeGeometry(
            curve,
            12,
            i === 0 ? 0.65 : 1.05,
            10,
            false,
          );
        });
        const g = s.tools.grasper,
          c = s.tools.cautery,
          gp = vec(g.position_mm, 7),
          cp = vec(c.position_mm, 5);
        connect(grasper, new THREE.Vector3(44, -44, 27), gp);
        connect(cautery, new THREE.Vector3(-44, -42, 25), cp);
        const spread = 2.2 * (1 - g.grip) + 0.25;
        connect(
          jaws[0],
          gp,
          gp.clone().add(new THREE.Vector3(-spread, 2.5, -1.5)),
        );
        connect(
          jaws[1],
          gp,
          gp.clone().add(new THREE.Vector3(spread, 2.5, -1.5)),
        );
        connect(cauteryTip, cp, cp.clone().add(new THREE.Vector3(0, 2.2, -1)));
        heat.visible = c.power > 0;
        heat.position.copy(cp);
        heat.scale.setScalar(1 + c.power * 4);
        lastState = s;
      }
      const clock = s.time_s;
      particles.forEach((p, i) => {
        const bleeding = i < 22,
          rate = bleeding
            ? s.physiology.bleeding_rate_ml_per_s
            : s.physiology.bile_leak_rate_ml_per_s;
        const source = bleeding
          ? nodes.slice(16, 18).sort((a, b) => b.damage - a.damage)[0]
          : nodes.slice(18).sort((a, b) => b.damage - a.damage)[0];
        p.visible = rate > 0;
        if (!p.visible) return;
        const phase = (clock * 0.5 + i * 0.137) % 1,
          spread = Math.min(8, 2 + rate * 0.5);
        p.position.set(
          source.position_mm[0] + Math.sin(i * 2.4) * phase * spread,
          source.position_mm[1] - phase * 12,
          1 + Math.sin(phase * Math.PI) * 2,
        );
        p.scale.setScalar((0.2 + phase * 0.5) * (bleeding ? 1 : 0.8));
      });
    };
    const resize = new ResizeObserver(() => {
      if (!mount.clientWidth || !mount.clientHeight) return;
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      update();
      controls.update();
      renderer.render(scene, camera);
    });
    resize.observe(mount);
    let frame = 0;
    let previousFrame = 0;
    const animate = (timestamp = 0) => {
      frame = requestAnimationFrame(animate);
      if (timestamp - previousFrame < 1000 / 30) return;
      previousFrame = timestamp;
      const stateChanged = lastState !== stateRef.current;
      if (stateChanged) update();
      controls.update();
      if (stateChanged || cameraChanged) renderer.render(scene, camera);
      cameraChanged = false;
    };
    animate();
    const pointer = new THREE.Vector2(),
      ray = new THREE.Raycaster(),
      plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    let down = [0, 0];
    const onDown = (e: PointerEvent) => {
      renderer.domElement.focus();
      down = [e.clientX, e.clientY];
    };
    const onUp = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      const box = renderer.domElement.getBoundingClientRect();
      pointer.set(
        ((e.clientX - box.left) / box.width) * 2 - 1,
        (-(e.clientY - box.top) / box.height) * 2 + 1,
      );
      ray.setFromCamera(pointer, camera);
      const point = new THREE.Vector3();
      if (ray.ray.intersectPlane(plane, point))
        targetRef.current(point.x, point.y);
    };
    renderer.domElement.addEventListener("pointerdown", onDown);
    renderer.domElement.addEventListener("pointerup", onUp);
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      controls.dispose();
      controls.removeEventListener("change", invalidateCamera);
      resetCamera.current = null;
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("pointerup", onUp);
      renderer.domElement.removeEventListener("webglcontextlost", contextLost);
      geometries.forEach((g) => g.dispose());
      tubeMeshes.forEach((m) => m.geometry.dispose());
      materials.forEach((m) => m.dispose());
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    };
  }, []);
  return (
    <div className="three-tissue-viewport" ref={host}>
      <button
        className="three-reset-view"
        onClick={() => resetCamera.current?.()}
        aria-label="Reset 3D camera"
      >
        Reset view
      </button>
      <div className="three-controls-caption">
        Drag: orbit · wheel: zoom · click: bounded tool motion
        <br />
        Depth and surface shading are illustrative. XY displacement,
        temperature, and damage follow environment state. Flow markers follow
        model time.
      </div>
    </div>
  );
}
