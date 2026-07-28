export const MODEL_CONFIG = {
  upscaling: {
    id: "upscaling",
    label: "EDSR x4 Upscaling",
    defaultModelPath: "/models/edsr.tflite",
    accelerator: "webgpu",
    input: {
      width: 128,
      height: 128,
      layout: "nchw",
      channels: 3,
      normalize: "zeroToOne",
    },
    output: {
      layout: "nchw",
    },
    tiling: {
      inputTileSize: 128,
      outputTileSize: 512,
      scale: 4,
    },
  },
};
