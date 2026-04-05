import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { createTextureCompareUI } from "./ui.js";

const canvas = document.querySelector("canvas.webgl");
const scene = new THREE.Scene();
const textureLoader = new THREE.TextureLoader();

const texturePaths = {
  color: "./predictions/albedo_input.png",
  originalRoughness: "./predictions/roughness_original.png",
  predictedRoughness: "./predictions/roughness_pred.png",
};

const requiredAssetPaths = [
  texturePaths.color,
  texturePaths.originalRoughness,
  texturePaths.predictedRoughness,
];

const applyTiling = (texture) => {
  texture.repeat.set(8, 8);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.needsUpdate = true;
  return texture;
};

const loadTexture = (path) =>
  new Promise((resolve, reject) => {
    textureLoader.load(path, resolve, undefined, () => {
      reject(new Error(`Could not load texture at ${path}. Run run_texture_model.py to export the prediction assets into static/predictions/.`));
    });
  });

const assetExists = async (path) => {
  try {
    const response = await fetch(path, { method: "HEAD" });
    return response.ok;
  } catch {
    return false;
  }
};

const floorMaterial = new THREE.MeshStandardMaterial({
  map: null,
  roughnessMap: null,
  roughness: 1.0,
  metalness: 0.0,
});

let floorColorTexture = null;
let floorOriginalRoughnessTexture = null;
let floorPredictedRoughnessTexture = null;

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(20, 20, 1, 1),
  floorMaterial,
);

floor.rotation.x = -Math.PI * 0.5;
scene.add(floor);

const ambientLight = new THREE.AmbientLight("#ffffff", 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight("#ffffff", 1.5);
directionalLight.position.set(3, 2, -8);
scene.add(directionalLight);

const sizes = {
  width: window.innerWidth,
  height: window.innerHeight,
};

const camera = new THREE.PerspectiveCamera(
  75,
  sizes.width / sizes.height,
  0.1,
  100,
);
camera.position.set(4, 2, 5);
scene.add(camera);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

const renderer = new THREE.WebGLRenderer({ canvas });
renderer.setSize(sizes.width, sizes.height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

window.addEventListener("resize", () => {
  sizes.width = window.innerWidth;
  sizes.height = window.innerHeight;

  camera.aspect = sizes.width / sizes.height;
  camera.updateProjectionMatrix();

  renderer.setSize(sizes.width, sizes.height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

const showOriginalRoughness = () => {
  if (!floorOriginalRoughnessTexture) {
    return;
  }

  floorMaterial.roughnessMap = floorOriginalRoughnessTexture;
  floorMaterial.needsUpdate = true;
};

const showPredictedRoughness = () => {
  if (!floorPredictedRoughnessTexture) {
    return;
  }

  floorMaterial.roughnessMap = floorPredictedRoughnessTexture;
  floorMaterial.needsUpdate = true;
};

const loadPredictedRoughness = async () => {
  if (!floorPredictedRoughnessTexture) {
    floorPredictedRoughnessTexture = applyTiling(
      await loadTexture(texturePaths.predictedRoughness),
    );
    floorPredictedRoughnessTexture.colorSpace = THREE.NoColorSpace;
  }

  showPredictedRoughness();
};

const ui = createTextureCompareUI({
  onPredict: loadPredictedRoughness,
  onShowOriginal: showOriginalRoughness,
  onShowNew: showPredictedRoughness,
  initialStatus: "Checking exported texture assets...",
});

const initializeExportedAssets = async () => {
  const missingAssets = [];

  for (const path of requiredAssetPaths) {
    const exists = await assetExists(path);
    if (!exists) {
      missingAssets.push(path);
    }
  }

  if (missingAssets.length > 0) {
    ui.setStatus(
      `Missing exported assets: ${missingAssets.join(", ")}. Run python run_texture_model.py from the portfolio folder.`,
    );
    return;
  }

  floorColorTexture = applyTiling(await loadTexture(texturePaths.color));
  floorColorTexture.colorSpace = THREE.SRGBColorSpace;

  floorOriginalRoughnessTexture = applyTiling(
    await loadTexture(texturePaths.originalRoughness),
  );
  floorOriginalRoughnessTexture.colorSpace = THREE.NoColorSpace;

  floorMaterial.map = floorColorTexture;
  floorMaterial.roughnessMap = floorOriginalRoughnessTexture;
  floorMaterial.needsUpdate = true;

  ui.setStatus("Exported assets found. Click Predict to load the new roughness map.");
};

initializeExportedAssets();

const tick = () => {
  controls.update();
  renderer.render(scene, camera);
  window.requestAnimationFrame(tick);
};

tick();
