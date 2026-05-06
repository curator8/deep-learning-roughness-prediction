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
const testSamples = [
  {
    id: "brick_pavement_03_1k",
    label: "Brick pavement",
    albedo: "./samples/brick_pavement_03_1k_albedo.png",
    roughness: "./samples/brick_pavement_03_1k_roughness.png",
  },
  {
    id: "dirt_01",
    label: "Dirt",
    albedo: "./samples/dirt_01_albedo.png",
    roughness: "./samples/dirt_01_roughness.png",
  },
  {
    id: "grassy_cobblestone_1k",
    label: "Grassy cobble",
    albedo: "./samples/grassy_cobblestone_1k_albedo.png",
    roughness: "./samples/grassy_cobblestone_1k_roughness.png",
  },
  {
    id: "mixed_rock_tiles_1k",
    label: "Rock tiles",
    albedo: "./samples/mixed_rock_tiles_1k_albedo.png",
    roughness: "./samples/mixed_rock_tiles_1k_roughness.png",
  },
  {
    id: "rock_wall_12_1k",
    label: "Rock wall",
    albedo: "./samples/rock_wall_12_1k_albedo.png",
    roughness: "./samples/rock_wall_12_1k_roughness.png",
  },
  {
    id: "sand_03_1k",
    label: "Sand",
    albedo: "./samples/sand_03_1k_albedo.png",
    roughness: "./samples/sand_03_1k_roughness.png",
  },
];
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const requiredAssetPaths = [
  texturePaths.color,
  texturePaths.originalRoughness,
];

let maxTextureAnisotropy = 1;

const applyTiling = (texture) => {
  texture.repeat.set(8, 8);
  texture.wrapS = THREE.MirroredRepeatWrapping;
  texture.wrapT = THREE.MirroredRepeatWrapping;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.anisotropy = maxTextureAnisotropy;
  texture.needsUpdate = true;
  return texture;
};

const applySingleTile = (texture) => {
  texture.repeat.set(1, 1);
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.anisotropy = maxTextureAnisotropy;
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
let predictedRoughnessObjectUrl = null;

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
maxTextureAnisotropy = renderer.capabilities.getMaxAnisotropy();

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

const disposeTexture = (texture) => {
  if (texture) {
    texture.dispose();
  }
};

const loadAlbedoFile = async (path, filename) => {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load sample albedo at ${path}.`);
  }

  const blob = await response.blob();
  return new File([blob], filename, { type: blob.type || "image/png" });
};

const selectSample = async (sample) => {
  const [colorTexture, originalRoughnessTexture, albedoFile] = await Promise.all([
    loadTexture(sample.albedo),
    loadTexture(sample.roughness),
    loadAlbedoFile(sample.albedo, `${sample.id}_albedo.png`),
  ]);

  disposeTexture(floorColorTexture);
  disposeTexture(floorOriginalRoughnessTexture);
  disposeTexture(floorPredictedRoughnessTexture);
  floorColorTexture = applyTiling(colorTexture);
  floorColorTexture.colorSpace = THREE.SRGBColorSpace;
  floorOriginalRoughnessTexture = applyTiling(originalRoughnessTexture);
  floorOriginalRoughnessTexture.colorSpace = THREE.NoColorSpace;
  floorPredictedRoughnessTexture = null;

  if (predictedRoughnessObjectUrl) {
    URL.revokeObjectURL(predictedRoughnessObjectUrl);
    predictedRoughnessObjectUrl = null;
  }

  floorMaterial.map = floorColorTexture;
  floorMaterial.roughnessMap = floorOriginalRoughnessTexture;
  floorMaterial.needsUpdate = true;

  return albedoFile;
};

const predictRoughness = async (selectedFile) => {
  const sourceFile = selectedFile ?? await fetchDefaultAlbedoFile();
  const formData = new FormData();
  formData.append("file", sourceFile);

  const response = await fetch(`${apiBaseUrl}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Prediction failed: ${message}`);
  }

  const predictionBlob = await response.blob();
  const nextObjectUrl = URL.createObjectURL(predictionBlob);
  const nextTexture = applySingleTile(await loadTexture(nextObjectUrl));
  nextTexture.colorSpace = THREE.NoColorSpace;

  if (floorPredictedRoughnessTexture) {
    disposeTexture(floorPredictedRoughnessTexture);
  }

  if (predictedRoughnessObjectUrl) {
    URL.revokeObjectURL(predictedRoughnessObjectUrl);
  }

  predictedRoughnessObjectUrl = nextObjectUrl;
  floorPredictedRoughnessTexture = nextTexture;
  showPredictedRoughness();
};

const fetchDefaultAlbedoFile = async () => {
  const response = await fetch(texturePaths.color);
  if (!response.ok) {
    throw new Error(`Could not load default albedo at ${texturePaths.color}.`);
  }

  const blob = await response.blob();
  return new File([blob], "albedo_input.png", { type: blob.type || "image/png" });
};

const ui = createTextureCompareUI({
  onPredict: predictRoughness,
  onSelectSample: selectSample,
  onShowOriginal: showOriginalRoughness,
  onShowNew: showPredictedRoughness,
  samples: testSamples,
  initialStatus: "Checking local texture assets and backend...",
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

  ui.setStatus("Assets found. Click Predict to call the FastAPI backend.");
};

initializeExportedAssets();

const tick = () => {
  controls.update();
  renderer.render(scene, camera);
  window.requestAnimationFrame(tick);
};

tick();
