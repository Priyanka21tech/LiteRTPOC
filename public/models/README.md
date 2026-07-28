# Models Folder

Place your LiteRT `.tflite` files here.

Expected default:

- `realesrgan.tflite`

The current app is wired to a single 4x Real-ESRGAN upscaler through
`src/config/model-config.js`.

The active pipeline splits the source image into `128x128` tiles, runs the
model on each tile, and stitches the `512x512` outputs back together.
