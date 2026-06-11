const folderInput = document.getElementById("folder-input");
const fileInput = document.getElementById("file-input");
const searchInput = document.getElementById("search-input");
const selectVisibleButton = document.getElementById("select-visible-button");
const clearSelectionButton = document.getElementById("clear-selection-button");
const imageLibrary = document.getElementById("image-library");
const selectedImages = document.getElementById("selected-images");
const sourceSummary = document.getElementById("source-summary");
const selectionSummary = document.getElementById("selection-summary");
const columnsInput = document.getElementById("columns-input");
const spacingInput = document.getElementById("spacing-input");
const paddingInput = document.getElementById("padding-input");
const imageWidthInput = document.getElementById("image-width-input");
const spacingOutput = document.getElementById("spacing-output");
const paddingOutput = document.getElementById("padding-output");
const canvasSummary = document.getElementById("canvas-summary");
const canvas = document.getElementById("output-canvas");
const canvasPlaceholder = document.getElementById("canvas-placeholder");
const copyButton = document.getElementById("copy-button");
const downloadButton = document.getElementById("download-button");
const statusMessage = document.getElementById("status-message");
const context = canvas.getContext("2d");

let images = [];
let selectedIds = [];
let draggedId = null;
let renderRevision = 0;

function isSupportedImage(file) {
  if (file.type.startsWith("image/")) {
    return true;
  }
  return /\.(png|jpe?g|gif|webp)$/i.test(file.name);
}

function imagePath(file) {
  return file.webkitRelativePath || file.name;
}

function makeImageRecord(file, index) {
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.src = url;
  return {
    id: `${file.name}-${file.lastModified}-${file.size}-${index}`,
    file,
    path: imagePath(file),
    url,
    image,
    ready: image.decode().catch(() => new Promise((resolve) => {
      if (image.complete) {
        resolve();
        return;
      }
      image.addEventListener("load", resolve, { once: true });
      image.addEventListener("error", resolve, { once: true });
    })),
  };
}

function releaseImages() {
  for (const record of images) {
    URL.revokeObjectURL(record.url);
  }
}

async function loadFiles(fileList) {
  const files = Array.from(fileList)
    .filter(isSupportedImage)
    .sort((left, right) => imagePath(left).localeCompare(imagePath(right), undefined, {
      numeric: true,
      sensitivity: "base",
    }));

  releaseImages();
  images = files.map(makeImageRecord);
  selectedIds = [];
  statusMessage.textContent = "";
  sourceSummary.textContent = files.length
    ? `${files.length} image${files.length === 1 ? "" : "s"} loaded`
    : "No supported images found";

  renderLibrary();
  renderSelected();
  await Promise.allSettled(images.map((record) => record.ready));
  renderComposite();
}

function selectedIndex(id) {
  return selectedIds.indexOf(id);
}

function toggleSelection(id) {
  const index = selectedIndex(id);
  if (index >= 0) {
    selectedIds.splice(index, 1);
  } else {
    selectedIds.push(id);
  }
  statusMessage.textContent = "";
  renderLibrary();
  renderSelected();
  renderComposite();
}

function visibleImages() {
  const term = searchInput.value.trim().toLowerCase();
  return images.filter((record) => !term || record.path.toLowerCase().includes(term));
}

function renderLibrary() {
  const visible = visibleImages();
  imageLibrary.replaceChildren();
  imageLibrary.classList.toggle("empty-state", visible.length === 0);

  if (visible.length === 0) {
    imageLibrary.textContent = images.length
      ? "No images match this filter."
      : "Select a folder containing PNG, JPEG, GIF, or WebP images.";
    return;
  }

  for (const record of visible) {
    const order = selectedIndex(record.id);
    const card = document.createElement("button");
    card.type = "button";
    card.className = `image-card${order >= 0 ? " selected" : ""}`;
    card.title = record.path;
    card.addEventListener("click", () => toggleSelection(record.id));

    const thumbnail = document.createElement("img");
    thumbnail.src = record.url;
    thumbnail.alt = record.file.name;

    const copy = document.createElement("div");
    copy.className = "image-card-copy";
    const name = document.createElement("div");
    name.className = "image-name";
    name.textContent = record.file.name;
    const path = document.createElement("div");
    path.className = "image-path";
    path.textContent = record.path;
    copy.append(name, path);
    card.append(thumbnail, copy);

    if (order >= 0) {
      const badge = document.createElement("span");
      badge.className = "selection-number";
      badge.textContent = String(order + 1);
      card.append(badge);
    }
    imageLibrary.append(card);
  }
}

function recordById(id) {
  return images.find((record) => record.id === id);
}

function renderSelected() {
  selectedImages.replaceChildren();
  selectionSummary.textContent = `${selectedIds.length} selected`;
  selectedImages.classList.toggle("empty-state", selectedIds.length === 0);

  if (selectedIds.length === 0) {
    selectedImages.textContent = "Click images above to add them to the grid.";
    return;
  }

  for (const id of selectedIds) {
    const record = recordById(id);
    if (!record) {
      continue;
    }

    const item = document.createElement("div");
    item.className = "selected-item";
    item.draggable = true;
    item.dataset.id = id;
    item.title = `Drag to reorder: ${record.path}`;

    item.addEventListener("dragstart", () => {
      draggedId = id;
      item.classList.add("dragging");
    });
    item.addEventListener("dragend", () => {
      draggedId = null;
      item.classList.remove("dragging");
    });
    item.addEventListener("dragover", (event) => {
      event.preventDefault();
    });
    item.addEventListener("drop", (event) => {
      event.preventDefault();
      if (!draggedId || draggedId === id) {
        return;
      }
      const draggedIndex = selectedIndex(draggedId);
      const targetIndex = selectedIndex(id);
      selectedIds.splice(draggedIndex, 1);
      selectedIds.splice(targetIndex, 0, draggedId);
      renderLibrary();
      renderSelected();
      renderComposite();
    });

    const thumbnail = document.createElement("img");
    thumbnail.src = record.url;
    thumbnail.alt = "";
    const label = document.createElement("span");
    label.textContent = record.file.name;
    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.className = "remove-button";
    removeButton.textContent = "x";
    removeButton.title = `Remove ${record.file.name}`;
    removeButton.addEventListener("click", (event) => {
      event.stopPropagation();
      toggleSelection(id);
    });

    item.append(thumbnail, label, removeButton);
    selectedImages.append(item);
  }
}

function positiveInteger(value, fallback, maximum) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(1, Math.min(maximum, parsed));
}

async function renderComposite() {
  const currentRevision = ++renderRevision;
  const selectedRecords = selectedIds.map(recordById).filter(Boolean);
  spacingOutput.textContent = `${spacingInput.value} px`;
  paddingOutput.textContent = `${paddingInput.value} px`;

  if (selectedRecords.length === 0) {
    canvas.style.display = "none";
    canvasPlaceholder.style.display = "block";
    canvasSummary.textContent = "Waiting for images";
    copyButton.disabled = true;
    downloadButton.disabled = true;
    return;
  }

  await Promise.allSettled(selectedRecords.map((record) => record.ready));
  if (currentRevision !== renderRevision) {
    return;
  }

  const usableRecords = selectedRecords.filter(
    (record) => record.image.naturalWidth > 0 && record.image.naturalHeight > 0,
  );
  if (usableRecords.length === 0) {
    statusMessage.textContent = "The selected images could not be decoded.";
    return;
  }

  const requestedColumns = positiveInteger(columnsInput.value, 2, 8);
  const columns = Math.min(requestedColumns, usableRecords.length);
  const spacing = Number(spacingInput.value);
  const padding = Number(paddingInput.value);
  const imageWidth = Number(imageWidthInput.value);
  const rows = Math.ceil(usableRecords.length / columns);
  const rowHeights = [];

  for (let row = 0; row < rows; row += 1) {
    const rowRecords = usableRecords.slice(row * columns, (row + 1) * columns);
    rowHeights.push(Math.max(...rowRecords.map(
      (record) => imageWidth * record.image.naturalHeight / record.image.naturalWidth,
    )));
  }

  canvas.width = Math.round(2 * padding + columns * imageWidth + (columns - 1) * spacing);
  canvas.height = Math.round(
    2 * padding + rowHeights.reduce((sum, height) => sum + height, 0) + (rows - 1) * spacing,
  );
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);

  let y = padding;
  usableRecords.forEach((record, index) => {
    const row = Math.floor(index / columns);
    const column = index % columns;
    const drawHeight = imageWidth * record.image.naturalHeight / record.image.naturalWidth;
    const x = padding + column * (imageWidth + spacing);
    const centeredY = y + (rowHeights[row] - drawHeight) / 2;
    context.drawImage(record.image, x, centeredY, imageWidth, drawHeight);
    if (column === columns - 1 || index === usableRecords.length - 1) {
      y += rowHeights[row] + spacing;
    }
  });

  canvas.style.display = "block";
  canvasPlaceholder.style.display = "none";
  canvasSummary.textContent = `${canvas.width} x ${canvas.height} px`;
  copyButton.disabled = false;
  downloadButton.disabled = false;
}

function canvasBlob() {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("Could not create PNG"));
      }
    }, "image/png");
  });
}

async function copyCanvas() {
  try {
    const blob = await canvasBlob();
    if (!navigator.clipboard || typeof ClipboardItem === "undefined") {
      throw new Error("Clipboard image copying is not supported in this browser");
    }
    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    statusMessage.textContent = "PNG copied. Paste it directly into your slide.";
  } catch (error) {
    statusMessage.textContent = `${error.message}. Use Download PNG instead.`;
  }
}

async function downloadCanvas() {
  const blob = await canvasBlob();
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "plot-grid.png";
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  statusMessage.textContent = "Downloaded plot-grid.png.";
}

folderInput.addEventListener("change", () => loadFiles(folderInput.files));
fileInput.addEventListener("change", () => loadFiles(fileInput.files));
searchInput.addEventListener("input", renderLibrary);
selectVisibleButton.addEventListener("click", () => {
  for (const record of visibleImages()) {
    if (!selectedIds.includes(record.id)) {
      selectedIds.push(record.id);
    }
  }
  renderLibrary();
  renderSelected();
  renderComposite();
});
clearSelectionButton.addEventListener("click", () => {
  selectedIds = [];
  renderLibrary();
  renderSelected();
  renderComposite();
});

for (const input of [columnsInput, spacingInput, paddingInput, imageWidthInput]) {
  input.addEventListener("input", renderComposite);
  input.addEventListener("change", renderComposite);
}

copyButton.addEventListener("click", copyCanvas);
downloadButton.addEventListener("click", downloadCanvas);
window.addEventListener("beforeunload", releaseImages);

renderLibrary();
renderSelected();
renderComposite();

if (new URLSearchParams(window.location.search).get("demo") === "1") {
  const demoColors = ["#d96c4f", "#4f7db8", "#6b9b72", "#b7863f"];
  const demoFiles = demoColors.map((color, index) => {
    const width = index % 2 === 0 ? 800 : 680;
    const height = index < 2 ? 520 : 680;
    const svg = `
      <svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">
        <rect width="100%" height="100%" fill="white"/>
        <rect x="32" y="32" width="${width - 64}" height="${height - 64}" rx="24" fill="${color}"/>
        <text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle"
              font-family="sans-serif" font-size="72" font-weight="700" fill="white">
          Plot ${index + 1}
        </text>
      </svg>
    `;
    return new File([svg], `demo_plot_${index + 1}.svg`, { type: "image/svg+xml" });
  });
  loadFiles(demoFiles).then(() => {
    selectedIds = images.map((record) => record.id);
    renderLibrary();
    renderSelected();
    renderComposite();
  });
}
