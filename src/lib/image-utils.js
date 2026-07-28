import * as tf from "@tensorflow/tfjs";

export function createHiddenCanvas() {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d", { willReadFrequently: true });

  if (!context) {
    throw new Error("Unable to create a 2D canvas context in this browser.");
  }

  return { canvas, context };
}

export function drawImageToCanvas(image) {
  const { canvas, context } = createHiddenCanvas();
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas;
}

export function canvasToBlob(canvas, type = "image/png", quality = 0.92) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Failed to convert canvas output into a downloadable file."));
        return;
      }

      resolve(blob);
    }, type, quality);
  });
}

export function tensorLayoutToNHWC(tensor, layout) {
  const resolvedLayout = layout === "auto" ? inferTensorLayout(tensor.shape) : layout;

  if (resolvedLayout === "nchw") {
    return tensor.transpose([0, 2, 3, 1]);
  }

  return tensor;
}

export function tensorLayoutFromNHWC(tensor, layout) {
  if (layout === "nchw") {
    return tensor.transpose([0, 3, 1, 2]);
  }

  return tensor;
}

function inferTensorLayout(shape) {
  if (!shape || shape.length !== 4) {
    return "nhwc";
  }

  const [, dim1, , dim3] = shape;

  if (dim1 === 1 || dim1 === 3 || dim1 === 4) {
    return "nchw";
  }

  if (dim3 === 1 || dim3 === 3 || dim3 === 4) {
    return "nhwc";
  }

  return "nhwc";
}

export function resolveInputConfigFromModel(model, inputConfig, inputIndex = 0) {
  const details = model.getInputDetails?.()[inputIndex];

  if (!details?.shape || details.shape.length !== 4) {
    return {
      ...inputConfig,
      layout: inputConfig.layout === "auto" ? "nhwc" : inputConfig.layout,
    };
  }

  const shape = Array.from(details.shape);
  const [, dim1, dim2, dim3] = shape;
  const layout = inputConfig.layout === "auto" ? inferTensorLayout(shape) : inputConfig.layout;

  if (layout === "nchw") {
    return {
      ...inputConfig,
      layout,
      channels: dim1 > 0 ? dim1 : inputConfig.channels,
      height: dim2 > 0 ? dim2 : inputConfig.height,
      width: dim3 > 0 ? dim3 : inputConfig.width,
    };
  }

  return {
    ...inputConfig,
    layout,
    height: dim1 > 0 ? dim1 : inputConfig.height,
    width: dim2 > 0 ? dim2 : inputConfig.width,
    channels: dim3 > 0 ? dim3 : inputConfig.channels,
  };
}

export function normalizeTensor(tensor, mode) {
  if (mode === "zeroToOne") {
    return tensor.div(255);
  }

  if (mode === "minusOneToOne") {
    return tensor.div(127.5).sub(1);
  }

  return tensor;
}

export function denormalizeTensor(tensor, mode) {
  if (mode === "minusOneToOne") {
    return tensor.add(1).mul(127.5);
  }

  if (mode === "zeroToOne") {
    return tensor.mul(255);
  }

  return tensor;
}

export function tensorToCanvas(rgbTensor) {
  const [height, width, channels] = rgbTensor.shape;

  if (channels !== 3 && channels !== 4) {
    throw new Error(`Expected an RGB or RGBA image tensor, received shape [${rgbTensor.shape.join(", ")}].`);
  }

  const clipped = rgbTensor.clipByValue(0, 255).cast("int32");
  const { canvas, context } = createHiddenCanvas();
  canvas.width = width;
  canvas.height = height;
  const imageData = new ImageData(width, height);

  return clipped.data().then((data) => {
    if (channels === 4) {
      imageData.data.set(data);
    } else {
      for (let sourceIndex = 0, targetIndex = 0; sourceIndex < data.length; sourceIndex += 3, targetIndex += 4) {
        imageData.data[targetIndex] = data[sourceIndex];
        imageData.data[targetIndex + 1] = data[sourceIndex + 1];
        imageData.data[targetIndex + 2] = data[sourceIndex + 2];
        imageData.data[targetIndex + 3] = 255;
      }
    }

    context.putImageData(imageData, 0, 0);
    clipped.dispose();
    return canvas;
  });
}

export function rgbaTensorToCanvas(rgbaTensor) {
  const [height, width] = rgbaTensor.shape;
  const clipped = rgbaTensor.clipByValue(0, 255).cast("int32");
  const { canvas, context } = createHiddenCanvas();
  canvas.width = width;
  canvas.height = height;
  const imageData = new ImageData(width, height);

  return clipped.data().then((data) => {
    imageData.data.set(data);
    context.putImageData(imageData, 0, 0);
    clipped.dispose();
    return canvas;
  });
}

export function buildInputTensor(imageSource, inputConfig) {
  return tf.tidy(() => {
    const pixels = tf.browser.fromPixels(imageSource, inputConfig.channels);
    const resized = tf.image.resizeBilinear(pixels, [inputConfig.height, inputConfig.width], true);
    const normalized = normalizeTensor(resized.toFloat(), inputConfig.normalize);
    const withBatch = normalized.expandDims(0);
    return tensorLayoutFromNHWC(withBatch, inputConfig.layout);
  });
}
