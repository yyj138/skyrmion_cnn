const serviceStatus = document.getElementById("service-status");
const exampleList = document.getElementById("example-list");
const imageInput = document.getElementById("image-input");
const dropZone = document.getElementById("drop-zone");
const previewImage = document.getElementById("preview-image");
const imageCaption = document.getElementById("image-caption");
const imagePlaceholder = document.getElementById("image-placeholder");
const inputError = document.getElementById("input-error");
const resultError = document.getElementById("result-error");
const predictButton = document.getElementById("predict-button");
const resultState = document.getElementById("result-state");
const kindValue = document.getElementById("kind-value");
const etaValue = document.getElementById("eta-value");
const voltageValue = document.getElementById("voltage-value");
const energyValue = document.getElementById("energy-value");
const clippingNote = document.getElementById("clipping-note");
const resultGrid = document.getElementById("result-grid");
const kindSection = document.querySelector(".kind-section");
const referenceDetails = document.getElementById("reference-details");
const scopeNoteText = document.getElementById("scope-note-text");
const defaultScopeText = scopeNoteText.textContent;

let selectedFile = null;
let selectedExample = null;
let selectedExampleReference = null;
let previewUrl = null;
let modelReady = false;
let requestInProgress = false;
let selectionVersion = 0;
let selectionIntentVersion = 0;
let predictionRunId = 0;
let activePredictionController = null;

function showError(element, message) {
  element.textContent = message;
  element.hidden = !message;
}

function setServiceState(state, message) {
  serviceStatus.className = `service-status ${state}`;
  serviceStatus.lastElementChild.textContent = message;
  modelReady = state === "ready";
  predictButton.disabled = !modelReady || !selectedFile || requestInProgress;
}

function setResultState(state, message) {
  resultState.className = `result-state ${state}`;
  resultState.textContent = message;
}

function clearResult() {
  kindValue.textContent = "—";
  etaValue.textContent = "—";
  voltageValue.textContent = "—";
  energyValue.textContent = "—";
  clippingNote.hidden = true;
  referenceDetails.open = false;
  referenceDetails.hidden = true;
  showError(resultError, "");
  setResultState("", "等待预测");
}

function setSelectedExample(id) {
  selectedExample = id;
  for (const button of exampleList.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.exampleId === id));
  }
}

function beginSelection(caption) {
  const intent = ++selectionIntentVersion;
  selectionVersion += 1;
  predictionRunId += 1;
  activePredictionController?.abort();
  activePredictionController = null;
  requestInProgress = false;
  resultGrid.classList.remove("is-loading");
  kindSection.classList.remove("is-loading");
  if (previewUrl?.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  selectedFile = null;
  selectedExampleReference = null;
  previewImage.removeAttribute("src");
  imagePlaceholder.hidden = false;
  imageCaption.textContent = caption;
  setSelectedExample(null);
  scopeNoteText.textContent = defaultScopeText;
  clearResult();
  setResultState("loading", "正在读取图像");
  showError(inputError, "");
  predictButton.textContent = "重新预测";
  predictButton.disabled = true;
  return intent;
}

function useFile(file, caption, exampleId = null, knownUrl = null, reference = null) {
  selectedFile = file;
  selectedExampleReference = reference;
  previewUrl = knownUrl || URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewImage.alt = `${caption}的 300×300 POM 图像预览`;
  imageCaption.textContent = caption;
  imagePlaceholder.hidden = true;
  showError(inputError, "");
  setSelectedExample(exampleId);
  clearResult();
  scopeNoteText.textContent = exampleId
    ? defaultScopeText
    : "上传图需要通过格式和基础外观检查；这些检查无法判断它是否与训练模拟图同分布。真实显微照片尚未验证。";
  predictButton.disabled = !modelReady;
  if (modelReady) predict();
}

function rejectFile(message) {
  imageCaption.textContent = "图像未通过校验";
  clearResult();
  showError(inputError, message);
  predictButton.disabled = true;
  imageInput.value = "";
}

async function inspectFile(file) {
  if (!file) return;
  const intent = beginSelection("正在校验上传图像");
  if (!["image/png", "image/jpeg"].includes(file.type)) {
    rejectFile("请选择 PNG 或 JPEG 图像文件。");
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    rejectFile("图像文件不能超过 8 MB。");
    return;
  }
  try {
    const bitmap = await createImageBitmap(file);
    const { width, height } = bitmap;
    bitmap.close();
    if (intent !== selectionIntentVersion) return;
    if (width !== 300 || height !== 300) {
      rejectFile(`当前图像为 ${width}×${height} 像素，请选择 300×300 图像；平台不会自动缩放。`);
      return;
    }
    useFile(file, `上传文件 · ${file.name}`);
  } catch {
    if (intent === selectionIntentVersion) {
      rejectFile("无法读取这张图像，请选择有效的 PNG 或 JPEG 文件。");
    }
  }
}

async function loadExample(example) {
  const intent = beginSelection("正在载入项目示例");
  try {
    const response = await fetch(example.image_url);
    if (!response.ok) throw new Error("示例暂不可用");
    const blob = await response.blob();
    if (intent !== selectionIntentVersion) return;
    const file = new File([blob], `${example.id}-demo.png`, { type: "image/png" });
    useFile(file, `${example.label} · 项目模拟图`, example.id, example.image_url, example.reference);
  } catch {
    if (intent === selectionIntentVersion) {
      imageCaption.textContent = "示例图像暂不可用";
      clearResult();
      showError(inputError, "示例图像加载失败，请检查服务后重试。");
    }
  }
}

async function loadExamples() {
  const startingIntent = selectionIntentVersion;
  try {
    const response = await fetch("/api/examples");
    if (!response.ok) throw new Error("示例列表不可用");
    const examples = await response.json();
    exampleList.replaceChildren();
    for (const example of examples) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example-button";
      button.dataset.exampleId = example.id;
      button.setAttribute("aria-pressed", "false");
      const thumbnail = document.createElement("img");
      thumbnail.src = example.image_url;
      thumbnail.alt = "";
      const text = document.createElement("span");
      const label = document.createElement("strong");
      label.textContent = example.label;
      const description = document.createElement("small");
      description.textContent = "项目模拟 POM 图";
      text.append(label, description);
      button.append(thumbnail, text);
      button.addEventListener("click", () => loadExample(example));
      exampleList.append(button);
    }
    if (examples.length && selectionIntentVersion === startingIntent) await loadExample(examples[0]);
  } catch {
    if (selectionIntentVersion === startingIntent) {
      previewImage.removeAttribute("src");
      imageCaption.textContent = "请选择 300×300 POM 图像";
      imagePlaceholder.hidden = false;
      showError(inputError, "项目示例暂不可用，仍可上传自己的模拟 POM 图像。");
    }
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("模型未就绪");
    setServiceState("ready", "模型已就绪");
    if (selectedFile) predict();
  } catch {
    setServiceState("error", "模型未就绪");
  }
}

function renderEnergy(value) {
  const [coefficient, exponent] = value.toExponential(3).split("e");
  const sign = Number(exponent) < 0 ? "−" : "+";
  const absolute = Math.abs(Number(exponent));
  energyValue.replaceChildren();
  energyValue.append(`${coefficient} × 10`);
  const superscript = document.createElement("sup");
  superscript.textContent = `${sign}${absolute}`;
  energyValue.append(superscript);
  const unit = document.createElement("small");
  unit.textContent = "J";
  energyValue.append(unit);
  energyValue.title = `${value} J`;
}

function compactEnergy(value) {
  const [coefficient, exponent] = value.toExponential(3).split("e");
  const superscripts = { "-": "⁻", "+": "⁺", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
  return `${coefficient} × 10${[...String(Number(exponent))].map((character) => superscripts[character]).join("")} J`;
}

function renderReferenceComparison(result) {
  if (!selectedExampleReference) return;
  const reference = selectedExampleReference;
  const kindName = (kind) => kind === "single" ? "单 toron" : "双 toron";
  document.getElementById("compare-kind-predicted").textContent = kindName(result.kind);
  document.getElementById("compare-kind-reference").textContent = kindName(reference.kind);
  document.getElementById("compare-eta-predicted").textContent = String(result.eta);
  document.getElementById("compare-eta-reference").textContent = String(reference.eta);
  document.getElementById("compare-voltage-predicted").textContent = `${result.voltage_V.toFixed(3)} V`;
  document.getElementById("compare-voltage-reference").textContent = `${reference.voltage_V.toFixed(3)} V`;
  document.getElementById("compare-energy-predicted").textContent = compactEnergy(result.free_energy_J);
  document.getElementById("compare-energy-reference").textContent = compactEnergy(reference.free_energy_J);
  referenceDetails.hidden = false;
}

async function predict() {
  if (!selectedFile || !modelReady || requestInProgress) return;
  const currentSelection = selectionVersion;
  const currentRun = ++predictionRunId;
  const controller = new AbortController();
  activePredictionController = controller;
  requestInProgress = true;
  predictButton.disabled = true;
  predictButton.textContent = "正在预测…";
  setResultState("loading", "正在预测");
  showError(resultError, "");
  clippingNote.hidden = true;
  resultGrid.classList.add("is-loading");
  kindSection.classList.add("is-loading");
  try {
    const form = new FormData();
    form.append("image", selectedFile, selectedFile.name);
    const response = await fetch("/api/predict", { method: "POST", body: form, signal: controller.signal });
    const result = await response.json().catch(() => ({}));
    if (currentSelection !== selectionVersion || currentRun !== predictionRunId) return;
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "预测失败，请稍后重试。");
    kindValue.textContent = result.kind === "single" ? "单 toron" : "双 toron";
    etaValue.textContent = String(result.eta);
    etaValue.title = `${result.eta} 个仿真晶格单位`;
    voltageValue.replaceChildren(`${result.voltage_V.toFixed(3)}`);
    const voltageUnit = document.createElement("small");
    voltageUnit.textContent = "V";
    voltageValue.append(voltageUnit);
    voltageValue.title = `${result.voltage_V} V`;
    renderEnergy(result.free_energy_J);
    clippingNote.hidden = !result.voltage_clipped_to_training_range;
    if (!selectedExample) {
      scopeNoteText.textContent = "上传图通过了格式和基础外观检查；平台仍无法判断它是否与训练模拟图同分布。真实显微照片或其他图像的数值不应作为有效测量。";
    }
    renderReferenceComparison(result);
    setResultState("ready", "预测完成");
  } catch (error) {
    if (currentSelection !== selectionVersion || currentRun !== predictionRunId || error.name === "AbortError") return;
    clearResult();
    setResultState("error", "预测失败");
    showError(resultError, error.message || "预测失败，请检查图像后重试。");
    if (error.message?.includes("模型尚未就绪")) setServiceState("error", "模型未就绪");
  } finally {
    if (currentRun !== predictionRunId) return;
    resultGrid.classList.remove("is-loading");
    kindSection.classList.remove("is-loading");
    activePredictionController = null;
    requestInProgress = false;
    predictButton.textContent = "重新预测";
    predictButton.disabled = !selectedFile || !modelReady;
  }
}

imageInput.addEventListener("change", () => inspectFile(imageInput.files[0]));
predictButton.addEventListener("click", predict);
for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
}
dropZone.addEventListener("drop", (event) => inspectFile(event.dataTransfer.files[0]));

checkHealth();
loadExamples();
