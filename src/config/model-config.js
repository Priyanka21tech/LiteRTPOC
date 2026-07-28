export const MODEL_CONFIG = {
  upscaling: {
    id: "upscaling",
    label: "Real-ESRGAN x4 Upscaling",
    defaultModelPath: "/models/realesrgan.tflite",
    accelerator: "webgpu",
    input: {
      width: 128,
      height: 128,
      layout: "auto",
      channels: 3,
      normalize: "zeroToOne",
    },
    output: {
      layout: "auto",
    },
    tiling: {
      inputTileSize: 128,
      outputTileSize: 512,
      scale: 4,
    },
  },
};
