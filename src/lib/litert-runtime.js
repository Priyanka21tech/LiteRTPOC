import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import { getWebGpuDevice, loadAndCompile, loadLiteRt } from "@litertjs/core";
import { WebGPUBackend } from "@tensorflow/tfjs-backend-webgpu";

let runtimePromise;
const compiledModels = new Map();
let activeAccelerator = "wasm";
let initializedTfjsBackend = null;

function isLikelyTfliteModel(bytes) {
  if (bytes.length < 8) {
    return false;
  }

  const signature = String.fromCharCode(bytes[4], bytes[5], bytes[6], bytes[7]);
  return signature === "TFL3";
}

function isLikelyHtmlResponse(bytes) {
  const prefix = String.fromCharCode(...bytes.slice(0, 64)).trimStart().toLowerCase();
  return prefix.startsWith("<!doctype html") || prefix.startsWith("<html");
}

async function fetchModelBytes(modelPath) {
  const response = await fetch(modelPath, { cache: "no-store" });

  if (!response.ok) {
    throw new Error(
      `Model file "${modelPath}" was not found or could not be read. Add a valid .tflite file under public/models.`,
    );
  }

  const bytes = new Uint8Array(await response.arrayBuffer());

  if (!isLikelyTfliteModel(bytes)) {
    if (isLikelyHtmlResponse(bytes)) {
      throw new Error(
        `Model file "${modelPath}" was not served as a .tflite file. Add a real TensorFlow Lite model at public${modelPath}, then restart or refresh the dev server.`,
      );
    }

    throw new Error(
      `Model file "${modelPath}" is not a valid TensorFlow Lite model. Make sure the file is a real .tflite artifact, not a placeholder or HTML response.`,
    );
  }

  return bytes;
}

function createModelCacheKey(modelConfig, modelSource) {
  if (modelSource?.kind === "file") {
    return `${modelConfig.id}:file:${modelSource.file.name}:${modelSource.file.size}:${modelSource.file.lastModified}`;
  }

  if (modelSource?.kind === "bytes") {
    return `${modelConfig.id}:bytes:${modelSource.cacheKey}`;
  }

  const path = modelSource?.path || modelConfig.defaultModelPath;
  return `${modelConfig.id}:path:${path}`;
}

async function resolveModelBytes(modelConfig, modelSource) {
  if (modelSource?.kind === "file") {
    return new Uint8Array(await modelSource.file.arrayBuffer());
  }

  if (modelSource?.kind === "bytes") {
    return modelSource.bytes;
  }

  const modelPath = modelSource?.path || modelConfig.defaultModelPath;

  if (!modelPath) {
    throw new Error(
      `No model is configured for "${modelConfig.label}". Upload a .tflite model file or set a default model path.`,
    );
  }

  return fetchModelBytes(modelPath);
}

export async function initializeLiteRt() {
  if (!runtimePromise) {
    runtimePromise = loadLiteRt("/wasm/");
  }

  return runtimePromise;
}

async function initializeTfjsWebGpuBackend() {
  await tf.setBackend("webgpu");
  await tf.ready();
  const device = getWebGpuDevice();
  tf.removeBackend("webgpu");
  tf.registerBackend("webgpu", () => new WebGPUBackend(device, device.adapterInfo));
  await tf.setBackend("webgpu");
  await tf.ready();
}

async function initializeCpuBackend() {
  await tf.setBackend("cpu");
  await tf.ready();
}

export async function initializeInferenceBackend(preferredAccelerator = "wasm") {
  await initializeLiteRt();

  if (preferredAccelerator === "webgpu" && typeof navigator !== "undefined" && navigator.gpu) {
    if (initializedTfjsBackend !== "webgpu") {
      try {
        await initializeTfjsWebGpuBackend();
        initializedTfjsBackend = "webgpu";
      } catch {
        await initializeCpuBackend();
        initializedTfjsBackend = "cpu";
        activeAccelerator = "wasm";
        return activeAccelerator;
      }
    }

    activeAccelerator = "webgpu";
    return activeAccelerator;
  }

  if (initializedTfjsBackend !== "cpu") {
    await initializeCpuBackend();
    initializedTfjsBackend = "cpu";
  }

  activeAccelerator = "wasm";
  return activeAccelerator;
}

function withAccelerator(modelConfig, accelerator) {
  return {
    ...modelConfig,
    accelerator,
  };
}

export function getActiveAccelerator() {
  return activeAccelerator;
}

export async function getCompiledModel(modelConfig, modelSource) {
  const preferredAccelerator = modelConfig.accelerator || "wasm";
  const resolvedAccelerator = await initializeInferenceBackend(preferredAccelerator);
  const resolvedConfig = withAccelerator(modelConfig, resolvedAccelerator);
  const cacheKey = createModelCacheKey(resolvedConfig, modelSource);

  if (!compiledModels.has(cacheKey)) {
    const compilePromise = resolveModelBytes(resolvedConfig, modelSource).then((modelBytes) => {
      if (!isLikelyTfliteModel(modelBytes)) {
        throw new Error(
          `The selected model for "${resolvedConfig.label}" is not a valid TensorFlow Lite file. Use a real .tflite model file.`,
        );
      }

      return loadAndCompile(modelBytes, {
        accelerator: resolvedConfig.accelerator,
      });
    }).catch(async (error) => {
      if (resolvedAccelerator === "webgpu") {
        compiledModels.delete(cacheKey);
        await initializeInferenceBackend("wasm");
        return getCompiledModel(withAccelerator(modelConfig, "wasm"), modelSource);
      }

      throw error;
    });
    compiledModels.set(cacheKey, compilePromise);
  }

  return compiledModels.get(cacheKey);
}
