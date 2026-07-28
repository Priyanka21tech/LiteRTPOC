import * as tf from "@tensorflow/tfjs";
import { MODEL_CONFIG } from "./config/model-config";
import { canvasToBlob } from "./lib/image-utils";
import { getActiveAccelerator } from "./lib/litert-runtime";
import { runUpscalingPipeline } from "./lib/pipelines/upscaling";

function formatResolution(width, height) {
  return width && height ? `${width} x ${height}` : "Not available";
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function getFriendlyError(error) {
  const message = error instanceof Error ? error.message : "Unknown processing error.";

  if (message.includes("not found or could not be read")) {
    return message;
  }

  if (message.includes("not a valid TensorFlow Lite model")) {
    return message;
  }

  if (message.includes("Failed to fetch")) {
    return "The LiteRT runtime or model file could not be loaded. Verify the files under public/wasm and public/models.";
  }

  return message;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function getRenderedImageBox(imageElement) {
  const boxWidth = imageElement.clientWidth;
  const boxHeight = imageElement.clientHeight;
  const imageRatio = imageElement.naturalWidth / imageElement.naturalHeight;
  const boxRatio = boxWidth / boxHeight;

  if (imageRatio > boxRatio) {
    const renderedHeight = boxWidth / imageRatio;
    return {
      width: boxWidth,
      height: renderedHeight,
      offsetX: 0,
      offsetY: (boxHeight - renderedHeight) / 2,
    };
  }

  const renderedWidth = boxHeight * imageRatio;
  return {
    width: renderedWidth,
    height: boxHeight,
    offsetX: (boxWidth - renderedWidth) / 2,
    offsetY: 0,
  };
}

export function createApp(root) {
  root.innerHTML = `
    <main class="shell">
      <section class="panel hero">
        <p class="eyebrow">PhotoIQ Foundation</p>
        <h1>LiteRT.js Image Upscaling POC</h1>
        <p class="summary">
          Browser-side 4x image upscaling using LiteRT.js with GPU-first execution.
          The current build uses a Real-ESRGAN x4 LiteRT model for visibly sharper detail on small images while keeping the 128x128 to 512x512 tiled pipeline.
        </p>
      </section>

      <section class="panel controls">
        <label class="upload">
          <span>Upload image</span>
          <input id="file-input" type="file" accept="image/*" />
        </label>
        <div class="actions">
          <button id="upscale-btn" type="button" disabled>Run 4x upscale</button>
          <button id="download-btn" type="button" disabled>Download result</button>
        </div>
        <p class="meta">Accelerator: <strong id="accelerator-label">Not initialized</strong></p>
        <p id="status" class="status">Select an image to begin.</p>
        <p id="error" class="error" hidden></p>
      </section>

      <section class="grid">
        <article class="panel preview-card">
          <h2>Original</h2>
          <div class="meta">Resolution: <strong id="original-resolution">Not available</strong></div>
          <div id="original-frame" class="preview-frame preview-interactive">
            <img id="original-preview" alt="Original preview" />
            <button id="original-focus" class="focus-indicator" type="button" aria-label="Select comparison focus"></button>
          </div>
        </article>

        <article class="panel preview-card">
          <h2>Upscaled</h2>
          <div class="meta">Resolution: <strong id="processed-resolution">Not available</strong></div>
          <div id="processed-frame" class="preview-frame preview-interactive">
            <img id="processed-preview" alt="Processed preview" />
            <button id="processed-focus" class="focus-indicator" type="button" aria-label="Select comparison focus"></button>
          </div>
        </article>
      </section>

      <section class="grid detail-grid">
        <article class="panel preview-card">
          <h2>Detail Comparison</h2>
          <p class="meta">Click either preview to move the comparison focus.</p>
          <p id="detail-focus-label" class="meta">Focus: center</p>
          <div class="detail-compare">
            <section>
              <h3>Original crop enlarged 4x</h3>
              <canvas id="original-detail" width="384" height="384"></canvas>
            </section>
            <section>
              <h3>Upscaled crop at native pixels</h3>
              <canvas id="processed-detail" width="384" height="384"></canvas>
            </section>
          </div>
        </article>
      </section>
    </main>
  `;

  const fileInput = root.querySelector("#file-input");
  const originalFrame = root.querySelector("#original-frame");
  const processedFrame = root.querySelector("#processed-frame");
  const originalPreview = root.querySelector("#original-preview");
  const processedPreview = root.querySelector("#processed-preview");
  const originalFocus = root.querySelector("#original-focus");
  const processedFocus = root.querySelector("#processed-focus");
  const originalDetail = root.querySelector("#original-detail");
  const processedDetail = root.querySelector("#processed-detail");
  const detailFocusLabel = root.querySelector("#detail-focus-label");
  const originalResolution = root.querySelector("#original-resolution");
  const processedResolution = root.querySelector("#processed-resolution");
  const upscaleButton = root.querySelector("#upscale-btn");
  const downloadButton = root.querySelector("#download-btn");
  const acceleratorLabel = root.querySelector("#accelerator-label");
  const status = root.querySelector("#status");
  const errorBox = root.querySelector("#error");

  let currentImage = null;
  let currentResultImage = null;
  let currentResultBlob = null;
  let currentOperationName = "processed";
  let originalPreviewUrl = null;
  let processedPreviewUrl = null;
  let focusPoint = { x: 0.5, y: 0.5 };

  function clearDetailCanvas(canvas) {
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#eef3fb";
    context.fillRect(0, 0, canvas.width, canvas.height);
  }

  function clearDetailInspectors() {
    clearDetailCanvas(originalDetail);
    clearDetailCanvas(processedDetail);
    detailFocusLabel.textContent = "Focus: center";
    originalFocus.hidden = true;
    processedFocus.hidden = true;
  }

  function drawDetailCrop(image, canvas, sceneSizeInSourcePixels) {
    const context = canvas.getContext("2d");

    if (!context) {
      return;
    }

    const cropSize = Math.min(sceneSizeInSourcePixels, image.naturalWidth, image.naturalHeight);
    const centerX = focusPoint.x * image.naturalWidth;
    const centerY = focusPoint.y * image.naturalHeight;
    const cropX = clamp(centerX - cropSize / 2, 0, image.naturalWidth - cropSize);
    const cropY = clamp(centerY - cropSize / 2, 0, image.naturalHeight - cropSize);

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(image, cropX, cropY, cropSize, cropSize, 0, 0, canvas.width, canvas.height);
  }

  function positionFocusIndicator(frame, image, indicator) {
    if (!image.getAttribute("src")) {
      indicator.hidden = true;
      return;
    }

    const rendered = getRenderedImageBox(image);
    indicator.hidden = false;
    indicator.style.left = `${rendered.offsetX + rendered.width * focusPoint.x}px`;
    indicator.style.top = `${rendered.offsetY + rendered.height * focusPoint.y}px`;
  }

  function renderDetailInspectors() {
    positionFocusIndicator(originalFrame, originalPreview, originalFocus);
    positionFocusIndicator(processedFrame, processedPreview, processedFocus);

    if (!currentImage || !currentResultImage) {
      clearDetailInspectors();
      return;
    }

    const sceneSize = 96;
    drawDetailCrop(currentImage, originalDetail, sceneSize);
    drawDetailCrop(currentResultImage, processedDetail, sceneSize * MODEL_CONFIG.upscaling.tiling.scale);
    detailFocusLabel.textContent = `Focus: ${Math.round(focusPoint.x * 100)}% x, ${Math.round(focusPoint.y * 100)}% y`;
  }

  function updateFocusPoint(event, imageElement) {
    if (!imageElement.getAttribute("src")) {
      return;
    }

    const rect = imageElement.getBoundingClientRect();
    const rendered = getRenderedImageBox(imageElement);
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    const normalizedX = clamp((localX - rendered.offsetX) / rendered.width, 0, 1);
    const normalizedY = clamp((localY - rendered.offsetY) / rendered.height, 0, 1);

    focusPoint = {
      x: normalizedX,
      y: normalizedY,
    };
    renderDetailInspectors();
  }

  function resetResult() {
    if (processedPreviewUrl) {
      URL.revokeObjectURL(processedPreviewUrl);
      processedPreviewUrl = null;
    }

    processedPreview.removeAttribute("src");
    processedResolution.textContent = "Not available";
    downloadButton.disabled = true;
    currentResultBlob = null;
    currentResultImage = null;
    renderDetailInspectors();
  }

  function setBusyState(isBusy, label) {
    fileInput.disabled = isBusy;
    upscaleButton.disabled = isBusy || !currentImage;
    downloadButton.disabled = isBusy || !currentResultBlob;
    status.textContent = label;
  }

  function showError(message) {
    errorBox.hidden = false;
    errorBox.textContent = message;
  }

  function clearError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  function syncStatusMessage() {
    if (!currentImage) {
      status.textContent = "Select an image to begin.";
      return;
    }

    const accelerator = getActiveAccelerator();
    acceleratorLabel.textContent = accelerator === "webgpu" ? "WebGPU" : "Wasm / CPU";
    status.textContent = `Image loaded. Run 4x upscale when ready${accelerator === "webgpu" ? " on GPU." : " on CPU."}`;
  }

  fileInput.addEventListener("change", () => {
    const [file] = fileInput.files || [];
    clearError();
    resetResult();

    if (!file) {
      currentImage = null;
      if (originalPreviewUrl) {
        URL.revokeObjectURL(originalPreviewUrl);
        originalPreviewUrl = null;
      }
      originalPreview.removeAttribute("src");
      originalResolution.textContent = "Not available";
      upscaleButton.disabled = true;
      status.textContent = "Select an image to begin.";
      clearDetailInspectors();
      return;
    }

    const fileUrl = URL.createObjectURL(file);
    const image = new Image();

    image.onload = () => {
      if (originalPreviewUrl) {
        URL.revokeObjectURL(originalPreviewUrl);
      }

      originalPreviewUrl = fileUrl;
      currentImage = image;
      focusPoint = { x: 0.5, y: 0.5 };
      originalPreview.src = fileUrl;
      originalResolution.textContent = formatResolution(image.naturalWidth, image.naturalHeight);
      upscaleButton.disabled = false;
      syncStatusMessage();
      renderDetailInspectors();
    };

    image.onerror = () => {
      URL.revokeObjectURL(fileUrl);
      currentImage = null;
      showError("The selected file could not be opened as an image.");
      status.textContent = "Select a different image.";
    };

    image.src = fileUrl;
  });

  async function processImage(operationName, runner, modelConfig) {
    if (!currentImage) {
      showError("Upload an image before running AI processing.");
      return;
    }

    clearError();
    setBusyState(true, `Running ${operationName}...`);

    try {
      await tf.ready();
      const result = await runner(currentImage, modelConfig, {
        kind: "path",
        path: modelConfig.defaultModelPath,
      }, {
        onProgress(progress) {
          const accelerator = getActiveAccelerator();
          acceleratorLabel.textContent = accelerator === "webgpu" ? "WebGPU" : "Wasm / CPU";
          status.textContent = `Running ${operationName} on ${accelerator === "webgpu" ? "GPU" : "CPU"}... ${progress.completedTiles}/${progress.totalTiles} tiles`;
        },
      });
      const blob = await canvasToBlob(result.canvas);
      const previewUrl = URL.createObjectURL(blob);
      const resultImage = new Image();
      await new Promise((resolve, reject) => {
        resultImage.onload = resolve;
        resultImage.onerror = () => reject(new Error("The upscaled preview could not be opened as an image."));
        resultImage.src = previewUrl;
      });

      if (processedPreviewUrl) {
        URL.revokeObjectURL(processedPreviewUrl);
      }

      processedPreviewUrl = previewUrl;
      currentResultImage = resultImage;
      processedPreview.src = previewUrl;
      processedResolution.textContent = formatResolution(result.width, result.height);
      currentResultBlob = blob;
      currentOperationName = operationName;
      acceleratorLabel.textContent = getActiveAccelerator() === "webgpu" ? "WebGPU" : "Wasm / CPU";
      status.textContent = `${operationName} complete on ${getActiveAccelerator() === "webgpu" ? "GPU" : "CPU"}.`;
      downloadButton.disabled = false;
      renderDetailInspectors();
    } catch (error) {
      acceleratorLabel.textContent = getActiveAccelerator() === "webgpu" ? "WebGPU" : "Wasm / CPU";
      showError(getFriendlyError(error));
      status.textContent = `${operationName} failed.`;
    } finally {
      setBusyState(false, status.textContent);
    }
  }

  upscaleButton.addEventListener("click", () =>
    processImage("4x upscale", runUpscalingPipeline, MODEL_CONFIG.upscaling),
  );

  originalFrame.addEventListener("click", (event) => updateFocusPoint(event, originalPreview));
  processedFrame.addEventListener("click", (event) => updateFocusPoint(event, processedPreview));
  window.addEventListener("resize", renderDetailInspectors);

  downloadButton.addEventListener("click", () => {
    if (!currentResultBlob) {
      showError("Run a processing step before downloading.");
      return;
    }

    downloadBlob(currentResultBlob, `${currentOperationName.replaceAll(" ", "-")}.png`);
  });

  clearDetailInspectors();
}
