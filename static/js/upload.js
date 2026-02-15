/*
 * upload.js — Image upload, drop zone & aspect ratio
 * =====================================================
 * Handles file drag-and-drop, upload to /api/upload,
 * and auto aspect ratio detection for uploaded images.
 */

"use strict";

const Upload = (() => {
  const { $ } = App;

  function setup() {
    const dropZone = $("#drop-zone");
    const fileInput = $("#file-input");
    if (!dropZone || !fileInput) return;

    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
    });
  }

  async function uploadFile(file) {
    const fd = new FormData();
    fd.append("image", file);
    try {
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      const d = await r.json();
      if (d.path) {
        $("#input-image-path").value = d.path;
        const previewImg = $("#preview-img");
        const dropText = $("#drop-text");
        previewImg.src = `/input/${d.filename}`;
        previewImg.style.display = "block";
        dropText.style.display = "none";
        App.toast("Image uploaded", "success", 2000);
        autoAdjustAspectRatio(`/input/${d.filename}`);
      } else {
        App.toast("Upload failed: " + (d.error || "unknown"), "error");
      }
    } catch (e) {
      App.toast("Upload error: " + e.message, "error");
    }
  }

  function autoAdjustAspectRatio(imageUrl) {
    const img = new Image();
    img.onload = () => {
      const iw = img.naturalWidth;
      const ih = img.naturalHeight;
      if (!iw || !ih) return;

      const aspect = iw / ih;
      const MIN_W = 256, MAX_W = 1920;
      const MIN_H = 256, MAX_H = 1080;
      const STEP = 16;

      let bestW = 832, bestH = 480, bestErr = Infinity;

      for (let w = MIN_W; w <= MAX_W; w += STEP) {
        let h = Math.round(w / aspect / STEP) * STEP;
        if (h < MIN_H) h = MIN_H;
        if (h > MAX_H) h = MAX_H;
        const err = Math.abs((w / h) - aspect);
        if (err < bestErr) {
          bestErr = err;
          bestW = w;
          bestH = h;
        }
      }

      Controls.setParam("width", bestW);
      Controls.setParam("height", bestH);
      Controls.onParamChange("width");
      Controls.onParamChange("height");
      Controls.updateResDimLabel();
      App.toast(`Aspect ratio detected (${iw}×${ih}) → ${bestW}×${bestH}`, "info", 3000);
    };
    img.src = imageUrl;
  }

  return { setup };
})();
