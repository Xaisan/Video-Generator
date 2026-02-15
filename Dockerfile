# ─── AMD ROCm PyTorch base ──────────────────────────────────────────
FROM rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1
# HSA_OVERRIDE_GFX_VERSION is set at runtime by amd_tune.py from config.yaml
# (supports gfx1100/RDNA3, gfx1030/RDNA2, etc. without rebuilding the image)
ENV HIP_VISIBLE_DEVICES=0
# PyTorch 2.9+ uses PYTORCH_ALLOC_CONF (PYTORCH_HIP_ALLOC_CONF is deprecated)
ENV PYTORCH_ALLOC_CONF=expandable_segments:True
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
ENV TOKENIZERS_PARALLELISM=false

# ─── System deps ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

# ─── Python deps ────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Copy project code ─────────────────────────────────────────────
COPY generate.py config.yaml amd_tune.py upscale.py rife_model.py ./
COPY session_manager.py pipeline_engine.py app.py ./
COPY pipeline/ pipeline/
COPY templates/ templates/
COPY static/ static/

# ─── Models will be bind-mounted at runtime ─────────────────────────
VOLUME ["/app/models", "/app/input", "/app/output", "/app/sessions"]

EXPOSE 7860
ENTRYPOINT ["python", "app.py"]
