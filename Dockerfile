# ─── AMD ROCm PyTorch base ──────────────────────────────────────────
FROM rocm/pytorch:rocm6.3.4_ubuntu22.04_py3.10_pytorch_release_2.6.0
# Use gfx1100 for RX 7900 XTX (RDNA3)
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV HIP_VISIBLE_DEVICES=0
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
COPY generate.py config.yaml amd_tune.py upscale.py ./
COPY session_manager.py pipeline_engine.py app.py ./
COPY templates/ templates/
COPY static/ static/

# ─── Models will be bind-mounted at runtime ─────────────────────────
VOLUME ["/app/models", "/app/input", "/app/output", "/app/sessions"]

EXPOSE 7860
ENTRYPOINT ["python", "app.py"]
