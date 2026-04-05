# Texture Creator Theory Notes

This file is a cleaned reference for the ideas behind the Python model and the Three.js viewer.

## Project Goal

The project learns a mapping:

`albedo image -> roughness map`

The input is a color texture.
The output is a grayscale material-property texture that controls how glossy or matte the surface appears.

## PBR Map Basics

PBR means physically based rendering. A material is described with several texture maps that affect lighting behavior.

### Albedo / Base Color

- Stores the visible color of the surface
- Does not describe shape or reflectivity by itself
- In Three.js this is usually assigned to `material.map`

### Roughness Map

- Controls micro-surface scattering
- Black means smoother surface and sharper reflections
- White means rougher surface and blurrier reflections
- In Three.js this is assigned to `material.roughnessMap`

### Normal Map

- Changes lighting direction without changing the actual mesh geometry
- Makes a flat surface look bumpy

### Metalness Map

- Controls whether a surface behaves like a metal or non-metal

### Height / Displacement Map

- Height map stores elevation-like information
- Displacement map actually moves mesh vertices when the mesh has enough subdivisions

### ARM Map

An ARM texture packs multiple grayscale maps into RGB channels:

- `R`: ambient occlusion
- `G`: roughness
- `B`: metalness

This is efficient for real-time rendering because it reduces texture fetches.

## Why Roughness Matters

Roughness does not change the mesh itself.
It changes the BRDF lighting calculation, which controls how reflections spread across the surface.

- `roughness = 0.0`: mirror-like
- `roughness = 1.0`: very matte

For this project, the network is learning surface reflectivity behavior from appearance.

## Dataset Theory

Each sample folder contains paired textures:

```text
sample_name/
├── albedo.png
└── roughness.png
```

The critical idea is that these are paired examples, not classes.
That means:

- input = albedo
- target = roughness

This is an image-to-image regression task, not image classification.

## Why PNG Was Important

PNG is lossless, so pixel values are preserved exactly.
That matters because the model is learning from pixel-level correspondences.

For technical maps like roughness, depth, or masks, lossy formats like JPEG can distort the values enough to hurt training quality.

## 16-bit Roughness Issue

One major bug came from reading the roughness maps incorrectly.

The roughness files were `16-bit grayscale`, but they were initially forced into `8-bit` mode.

That caused:

- distorted min/max values
- loss of useful precision
- a nearly blank-looking display

The fix was to:

- load the roughness file without forcing `convert("L")`
- read it as a numpy array
- preserve the 16-bit range
- normalize by `65535.0`

## Model Theory

This project uses convolutional neural networks for image-to-image prediction.

### Encoder

The encoder gradually compresses the input image:

- spatial resolution gets smaller
- feature depth gets richer

This helps the network learn abstract patterns instead of memorizing raw pixels.

### Bottleneck Representation

The bottleneck is the most compressed internal representation of the input.

It acts like a summary of:

- texture type
- large-scale structure
- surface appearance cues

If the bottleneck is too weak, the decoder has trouble reconstructing a good roughness map.

### Decoder

The decoder upsamples the compressed representation back to the original image resolution.

Its job is to reconstruct a full roughness map from the learned features.

## The Two Models

### RoughnessUNet

This is the stronger model.

It is an encoder-decoder with skip connections.

Skip connections directly pass encoder feature maps to matching decoder stages.
That helps preserve fine spatial detail.

For texture prediction, this matters because:

- input and output are aligned pixel-for-pixel
- local texture structure should be preserved

### RoughnessAutoencoder

This is the simpler baseline.

It is also an encoder-decoder, but it does not use skip connections.

That means the decoder only reconstructs from the bottleneck.
This often produces blurrier or less accurate outputs.

## Skip Connections

Skip connections are shortcuts from encoder layers to decoder layers at the same resolution.

Why they help:

- the encoder can lose fine detail during downsampling
- the decoder gets that detail back through the skip path

This is why U-Net is a strong architecture for segmentation and image-to-image tasks.

## Loss and Optimization

The training loop follows the usual deep learning steps:

1. forward pass
2. compare prediction to target
3. compute loss
4. backpropagate gradients
5. update weights with the optimizer

In this project, training uses `L1` loss:

- `prediction = model(albedo)`
- `loss = mean(abs(prediction - target))`

L1 is a good default for grayscale map regression because it is stable and easy to interpret.

## Metrics Used

The assignment asked for metrics and ablations, so the project tracks several evaluation numbers.

### MAE

Mean Absolute Error

- average pixel-wise absolute error
- lower is better
- easiest metric to explain

### MSE

Mean Squared Error

- average squared pixel-wise error
- lower is better
- punishes larger mistakes more heavily

### RMSE

Root Mean Squared Error

- square root of MSE
- lower is better
- easier to read because it is back in the original scale

### Cosine Similarity

- compares the predicted map and target map as flattened vectors
- closer to `1.0` is better
- measures whether the overall pattern is similar

## Ablation Study

An ablation study means changing one important part of the system and measuring what happens.

In this project, the ablation compares:

- `RoughnessUNet`: with skip connections
- `RoughnessAutoencoder`: without skip connections

This lets you answer:

“Do skip connections improve roughness-map prediction?”

If the U-Net has better metrics, then the architectural change is useful.

## Three.js Theory

The frontend does not run the PyTorch model directly.

Instead, the Python pipeline exports texture files, and Three.js loads them as standard material maps.

### Material Setup

The viewer uses `THREE.MeshStandardMaterial`, which supports PBR inputs:

```js
new THREE.MeshStandardMaterial({
  map: albedoTexture,
  roughnessMap: roughnessTexture,
})
```

### Compare Workflow

The viewer loads:

- exported albedo input
- exported original roughness
- exported predicted roughness

The UI then switches the material’s `roughnessMap` between the original and predicted versions.

That means the “Predict” button is really:

- “load the predicted roughness texture and apply it”

It is not live browser inference.

## Full Pipeline

The current project flow is:

1. load paired albedo + roughness data
2. train the model in Python
3. compute metrics and ablation results
4. export:
   - input albedo
   - original roughness
   - predicted roughness
5. load those files in Three.js
6. compare original vs predicted material response visually

## Practical Limitations

Right now the dataset is very small.

That means:

- training metrics can improve quickly
- overfitting is likely
- results are useful for proving the pipeline works
- results are not yet strong evidence of generalization

For stronger evaluation, the next step would be:

- split into `train` and `val`
- report metrics on validation data
- keep the same ablation setup

## Files in This Folder

- `run_texture_model.py`
  Clean script for training, metrics, ablations, and texture export
- `texture_dataset.py`
  Paired texture dataset loader
- `texture_model.py`
  Model definitions and training helpers
