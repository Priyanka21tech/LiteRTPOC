import * as tf from "@tensorflow/tfjs";
import { runWithTfjsTensors } from "@litertjs/tfjs-interop";
import { getCompiledModel } from "../litert-runtime";
import {
  buildInputTensor,
  createHiddenCanvas,
  drawImageToCanvas,
  resolveInputConfigFromModel,
  tensorLayoutToNHWC,
} from "../image-utils";

function createOutputCanvas(width, height) {
  const { canvas } = createHiddenCanvas();
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function createTileCanvas(sourceCanvas, tileX, tileY, tileSize) {
  const { canvas, context } = createHiddenCanvas();
  canvas.width = tileSize;
  canvas.height = tileSize;
  context.drawImage(sourceCanvas, tileX, tileY, tileSize, tileSize, 0, 0, tileSize, tileSize);
  return canvas;
}

function createPaddedCanvas(sourceCanvas, tileSize) {
  const paddedWidth = Math.ceil(sourceCanvas.width / tileSize) * tileSize;
  const paddedHeight = Math.ceil(sourceCanvas.height / tileSize) * tileSize;
  const { canvas, context } = createHiddenCanvas();
  canvas.width = paddedWidth;
  canvas.height = paddedHeight;
  context.drawImage(sourceCanvas, 0, 0);

  if (sourceCanvas.width < paddedWidth) {
    context.drawImage(
      sourceCanvas,
      sourceCanvas.width - 1,
      0,
      1,
      sourceCanvas.height,
      sourceCanvas.width,
      0,
      paddedWidth - sourceCanvas.width,
      sourceCanvas.height,
    );
  }

  if (sourceCanvas.height < paddedHeight) {
    context.drawImage(
      canvas,
      0,
      sourceCanvas.height - 1,
      paddedWidth,
      1,
      0,
      sourceCanvas.height,
      paddedWidth,
      paddedHeight - sourceCanvas.height,
    );
  }

  return canvas;
}

async function upscaleTile(tileCanvas, model, modelConfig) {
  const inputConfig = resolveInputConfigFromModel(model, modelConfig.input);
  const inputTensor = buildInputTensor(tileCanvas, inputConfig);

  try {
    const outputs = await runWithTfjsTensors(model, [inputTensor]);

    if (!outputs.length) {
      throw new Error("The EDSR upscaling model did not return any output tensors.");
    }

    const tileTensor = tf.tidy(() => {
      const outputTensor = tensorLayoutToNHWC(outputs[0], modelConfig.output.layout);
      return outputTensor.squeeze().clipByValue(0, 1);
    });

    tf.dispose(outputs);
    inputTensor.dispose();
    return tileTensor;
  } catch (error) {
    inputTensor.dispose();
    throw error;
  }
}

export async function runUpscalingPipeline(imageElement, modelConfig, modelSource, options = {}) {
  const model = await getCompiledModel(modelConfig, modelSource);
  const sourceCanvas = drawImageToCanvas(imageElement);
  const paddedCanvas = createPaddedCanvas(sourceCanvas, modelConfig.tiling.inputTileSize);
  const outputCanvas = createOutputCanvas(
    paddedCanvas.width * modelConfig.tiling.scale,
    paddedCanvas.height * modelConfig.tiling.scale,
  );
  const outputContext = outputCanvas.getContext("2d");

  if (!outputContext) {
    throw new Error("Unable to create a 2D canvas context for the upscaled output.");
  }

  const tilesX = paddedCanvas.width / modelConfig.tiling.inputTileSize;
  const tilesY = paddedCanvas.height / modelConfig.tiling.inputTileSize;
  const totalTiles = tilesX * tilesY;
  let completedTiles = 0;

  for (let tileRow = 0; tileRow < tilesY; tileRow += 1) {
    for (let tileColumn = 0; tileColumn < tilesX; tileColumn += 1) {
      const tileX = tileColumn * modelConfig.tiling.inputTileSize;
      const tileY = tileRow * modelConfig.tiling.inputTileSize;
      const sourceTile = createTileCanvas(paddedCanvas, tileX, tileY, modelConfig.tiling.inputTileSize);
      const tileTensor = await upscaleTile(sourceTile, model, modelConfig);
      const tileCanvas = createOutputCanvas(
        modelConfig.tiling.outputTileSize,
        modelConfig.tiling.outputTileSize,
      );

      try {
        await tf.browser.toPixels(tileTensor, tileCanvas);
      } catch (error) {
        tileTensor.dispose();
        throw error;
      }
      tileTensor.dispose();
      outputContext.drawImage(
        tileCanvas,
        tileColumn * modelConfig.tiling.outputTileSize,
        tileRow * modelConfig.tiling.outputTileSize,
      );

      completedTiles += 1;
      options.onProgress?.({
        completedTiles,
        totalTiles,
      });
    }
  }

  const croppedOutput = createOutputCanvas(
    sourceCanvas.width * modelConfig.tiling.scale,
    sourceCanvas.height * modelConfig.tiling.scale,
  );
  const croppedContext = croppedOutput.getContext("2d");

  if (!croppedContext) {
    throw new Error("Unable to create a 2D canvas context for the cropped result.");
  }

  croppedContext.drawImage(
    outputCanvas,
    0,
    0,
    croppedOutput.width,
    croppedOutput.height,
    0,
    0,
    croppedOutput.width,
    croppedOutput.height,
  );

  return {
    canvas: croppedOutput,
    width: croppedOutput.width,
    height: croppedOutput.height,
    totalTiles,
  };
}
