# LiteRT.js Image Upscaling POC

This project is a small browser-first foundation for PhotoIQ image tooling using
official LiteRT.js APIs. It currently supports one image workflow:

- 4x image upscaling with a tiled EDSR pipeline with GPU-first execution

The implementation is intentionally simple and extensible. The UI stays minimal,
and the model-specific behavior lives in one configuration file so you can swap
models later without rewriting the app.

## What This Uses

- `@litertjs/core` for model loading and compilation
- `@litertjs/tfjs-interop` for official TensorFlow.js tensor execution
- `@tensorflow/tfjs` for browser-side image preprocessing and postprocessing
- `vite` for local browser development and bundling

## Project Setup

### 1. Create the Python virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Python dependency installation

There are no third-party Python dependencies at this stage.

```powershell
python -m pip install -r requirements.txt
```

The virtual environment is still useful because it isolates any future Python
model-conversion utilities or helper scripts for this PhotoIQ pipeline.

### 3. Install JavaScript dependencies

PowerShell may block `npm.ps1`, so use `npm.cmd`:

```powershell
npm.cmd install
```

This project includes a `postinstall` step that copies the official LiteRT.js
Wasm runtime files from `node_modules/@litertjs/core/wasm/` into `public/wasm/`
so the browser can load them with `loadLiteRt("/wasm/")`.

### 4. Add your LiteRT models

Put your `.tflite` files in [public/models/README.md](public/models/README.md).

Expected default file name:

- `public/models/edsr.tflite`

If your model uses different tensor shapes, layouts, normalization, or output
semantics, update [src/config/model-config.js](src/config/model-config.js).

## Running The Project

### Browser dev server

```powershell
npm.cmd run dev
```

Open the printed local URL in a browser.

### Production build

```powershell
npm.cmd run build
npm.cmd run preview
```

### Optional Python static server

If you want a very small Python-only local server:

```powershell
python .\tools\serve.py
```

## Current Browser Features

- Upload an image
- Preview the original image
- Run 4x upscaling with EDSR
- Preview the processed image
- Download the processed result
- Display original and processed resolution
- Show friendly processing and model-loading errors

## LiteRT.js Architecture Overview

The app follows the same broad LiteRT.js flow documented by Google:

1. Load the official LiteRT.js Wasm runtime with `loadLiteRt`.
2. Compile a `.tflite` model in the browser with `loadAndCompile`.
3. Convert image pixels into TensorFlow.js tensors for preprocessing.
4. Run inference through `runWithTfjsTensors` from the official LiteRT TFJS interop package.
5. Split the image into `128x128` patches for tiled inference.
6. Reassemble the `512x512` per-tile results into a final 4x upscaled image.

Code structure:

- [src/lib/litert-runtime.js](src/lib/litert-runtime.js): runtime initialization and compiled-model caching
- [src/lib/pipelines/upscaling.js](src/lib/pipelines/upscaling.js): tiled EDSR upscaling pipeline
- [src/config/model-config.js](src/config/model-config.js): one place for model shape/layout configuration
- [src/app.js](src/app.js): simple browser UI wiring

## Model Contract Notes

This POC assumes:

- Models are browser-loadable `.tflite` files
- Inputs and outputs use supported LiteRT.js tensor types such as `float32`
- The upscaler accepts `128x128` RGB tiles and returns 4x RGB output tiles

The exact input/output tensor layout varies by model. This is why
[src/config/model-config.js](src/config/model-config.js) exists. Adjust:

- `layout`: `nhwc` or `nchw`
- `width` and `height`
- `normalize`: `zeroToOne` or `minusOneToOne`
- output thresholding or scale metadata

## Browser Compatibility

This project is browser-oriented and uses documented LiteRT.js web APIs.

- Works best in recent Chromium-based browsers for stable Wasm support.
- The default config uses `accelerator: "wasm"` to match the CPU/XNNPACK path.
- Google&apos;s LiteRT.js blog notes that CPU execution uses XNNPACK, while GPU and
  NPU paths can be significantly faster for demanding image manipulation.
- You can later switch models to `webgpu` or `webnn` in
  [src/config/model-config.js](src/config/model-config.js) if your target model
  and browser environment support them.

## Error Handling

The app currently handles these common failure cases:

- No uploaded image
- Invalid image file
- Missing `.tflite` model files
- Empty model outputs
- Browser canvas failures
- Download blob generation failures

If a model fails because of unsupported ops, tensor layout mismatch, or model
shape mismatch, update the model config or replace the model with one that
matches the expected tiled super-resolution pipeline.
