/**
 * collect.js — Modo de recolección de datos de entrenamiento.
 *
 * Novedades v3:
 *   - BUG FIX: _renderSignGrid usa sampleCounts (actualizado al guardar)
 *             en lugar de allSigns (datos viejos del servidor)
 *   - Guías especiales para G, H, J, S, Z
 *   - Trail de movimiento en tiempo real durante grabación de J/Z
 *   - Contador de repeticiones para señas de movimiento
 *   - Guía de orientación horizontal para G/H
 *   - Total de muestras actualizado inmediatamente tras guardar
 */

const FRAMES_PER_SIGN   = 100;
const MIN_SIGNS_TO_TRAIN = 5;

// ── Configuración de señas especiales ─────────────────────────────────────────
// Señas que necesitan guías visuales adicionales porque son difíciles de grabar

const SIGN_CONFIG = {
  G: {
    icon: "↔️",
    mode: "orientation",
    color: "0, 245, 200",
    title: "Orientación horizontal",
    hints: [
      "Gira la muñeca 90° — la mano de lado",
      "El índice apunta hacia la derecha (→) o izquierda (←)",
      "El pulgar también queda horizontal",
    ],
    warning: "⚠️ La mano NO apunta hacia arriba — va de lado.",
  },
  H: {
    icon: "↔️",
    mode: "orientation",
    color: "100, 180, 255",
    title: "Dos dedos horizontales",
    hints: [
      "Índice y medio extendidos, los demás cerrados",
      "La mano de lado — dedos apuntan horizontalmente",
      "Palma mirando hacia abajo o hacia ti",
    ],
    warning: "⚠️ Como G pero con dos dedos. La orientación es clave.",
  },
  J: {
    icon: "✏️",
    mode: "motion",
    color: "255, 120, 80",
    tipIdx: 20,
    title: "Seña de movimiento",
    hints: [
      "Solo el meñique extendido (como la letra I)",
      "Mueve el meñique hacia ABAJO ↓",
      "Al final, gira en gancho ↙ (como una J)",
    ],
    warning: "🔄 Repite el movimiento 4-5 veces durante la grabación.",
    repLabel: "meñique",
  },
  S: {
    icon: "👊",
    mode: "static_special",
    color: "200, 100, 255",
    title: "Pulgar encima del puño",
    hints: [
      "Cierra el puño firmemente (todos los dedos doblados)",
      "El pulgar cruza POR ENCIMA de los dedos (no al lado)",
      "La diferencia con A: en S el pulgar toca el frente de los dedos",
    ],
    warning: "⚠️ S ≠ A: en A el pulgar va al LADO; en S va ENCIMA del puño.",
  },
  Z: {
    icon: "✏️",
    mode: "motion",
    color: "124, 58, 237",
    tipIdx: 8,
    title: "Seña de movimiento",
    hints: [
      "Solo el índice extendido",
      "Traza la letra Z: → luego diagonal ↙ luego →",
      "El movimiento debe ser amplio y claro",
    ],
    warning: "🔄 Repite la Z completa 4-5 veces durante la grabación.",
    repLabel: "índice",
  },
};

// Cantidad de frames en trail visual
const TRAIL_MAX = 35;

// ─────────────────────────────────────────────────────────────────────────────

class CollectMode {
  constructor() {
    this.mp = null;
    this.currentCountry = "lsc";
    this.currentSignId  = null;
    this.isRecording    = false;
    this.capturedFrames = [];
    this.sampleCounts   = {};   // signId → count (siempre actualizado)
    this.allSigns       = [];
    this.cameraActive   = false;

    this._currentFps    = 0;
    this._fpsEl         = null;

    // Motion trail para J/Z
    this._motionTrail   = [];
    this._repCount      = 0;
    this._motionPeak    = 0;
    this._lastTipPos    = null;
    this._prevVel       = 0;
    this._inMotion      = false;

    this.videoEl  = document.getElementById("collect-video");
    this.canvasEl = document.getElementById("collect-canvas");

    this._bindUI();
  }

  _bindUI() {
    document.addEventListener("handDetected", (e) => {
      if (!this.cameraActive) return;
      const { landmarks, worldLandmarks } = e.detail;
      this._drawFrame(landmarks);

      if (this.isRecording) {
        this._captureFrame(landmarks, worldLandmarks);
      }
    });

    document.addEventListener("noHand", () => {
      if (this.canvasEl) Overlay.clearCanvas(this.canvasEl);
      this._drawFpsOverlay();
    });

    document.addEventListener("cameraReady", () => {
      this.cameraActive = true;
      const dot  = document.getElementById("collect-cam-dot");
      const text = document.getElementById("collect-cam-text");
      if (dot)  dot.classList.add("active");
      if (text) text.textContent = "Detectando";
    });

    document.addEventListener("fpsUpdate", (e) => {
      this._currentFps = e.detail?.fps ?? 0;
      this._updateFpsEl();
    });

    document.getElementById("btn-collect-record")
      ?.addEventListener("click", () => this._startRecording());

    document.getElementById("collect-country-select")
      ?.addEventListener("change", (e) => {
        this.currentCountry = e.target.value;
        this._loadStatusForCountry();
      });

    document.getElementById("btn-retrain")
      ?.addEventListener("click", () => this._triggerRetrain());

    document.getElementById("btn-clear-sign")
      ?.addEventListener("click", () => this._clearCurrentSign());
  }

  // ── Inicialización ──────────────────────────────────────────────────────────

  async init(country = "lsc") {
    this.currentCountry = country;
    await this._loadStatusForCountry();
    this._createFpsEl();
    await this._startCamera();
  }

  _createFpsEl() {
    const container = this.canvasEl?.parentElement;
    if (!container) return;
    if (document.getElementById("collect-fps")) return;

    const el = document.createElement("div");
    el.id = "collect-fps";
    el.className = "collect-fps";
    el.textContent = "— FPS";
    container.style.position = "relative";
    container.appendChild(el);
    this._fpsEl = el;
  }

  _updateFpsEl() {
    if (!this._fpsEl) return;
    const fps = this._currentFps;
    this._fpsEl.textContent = `${fps} FPS`;
    this._fpsEl.className = `collect-fps ${fps >= 25 ? "fps-good" : fps >= 15 ? "fps-ok" : "fps-bad"}`;
  }

  async _loadStatusForCountry() {
    try {
      const resp = await fetch(`/api/train/status/${this.currentCountry}`);
      const status = await resp.json();
      this.allSigns = status.samples_per_sign ?? [];
      // Reconstruir sampleCounts desde datos frescos del servidor
      this.sampleCounts = {};
      this.allSigns.forEach(s => { this.sampleCounts[s.sign_id] = s.count; });
      this._renderSignGrid();
      this._updateTrainButton(status);
      this._updateReadyCount(status);
    } catch (e) {
      console.error("[Collect] Error cargando estado:", e);
    }
  }

  async _startCamera() {
    if (!this.videoEl) return;
    if (!this.mp) {
      this.mp = new MediaPipeHandler({ numHands: 1, captureMode: true });
      this._showStatus("Cargando modelo de IA (~8 MB)…", "info");
      const ok = await this.mp.init();
      if (!ok) {
        this._showStatus("❌ Error cargando MediaPipe. Verifica tu conexión.", "error");
        return;
      }
    }
    this._syncCanvas();
    this._showStatus("Permite el acceso a la cámara cuando el browser te lo pida.", "info");
    await this.mp.startCamera(this.videoEl);
    this._showStatus("", "");
  }

  _syncCanvas() {
    if (!this.videoEl || !this.canvasEl) return;
    const ro = new ResizeObserver(() => {
      this.canvasEl.width  = this.videoEl.clientWidth  || 480;
      this.canvasEl.height = this.videoEl.clientHeight || 360;
    });
    ro.observe(this.videoEl);
    this.canvasEl.width  = this.videoEl.clientWidth  || 480;
    this.canvasEl.height = this.videoEl.clientHeight || 360;
  }

  // ── Selección de seña ────────────────────────────────────────────────────

  selectSign(signId) {
    this.currentSignId = signId;

    document.querySelectorAll(".collect-sign-item").forEach(el => {
      el.classList.toggle("selected", el.dataset.signId === signId);
    });

    const count = this.sampleCounts[signId] ?? 0;
    const pct   = Math.min(100, Math.round((count / FRAMES_PER_SIGN) * 100));
    const cfg   = SIGN_CONFIG[signId];
    const infoEl = document.getElementById("collect-sign-info");
    if (infoEl) {
      let specialHtml = "";

      if (cfg) {
        const hintsHtml = cfg.hints.map(h =>
          `<li style="text-align:left; margin-bottom:0.3rem; font-size:0.78rem; color:var(--text-muted)">${h}</li>`
        ).join("");

        const modeLabel = cfg.mode === "motion"
          ? `<span class="sign-badge sign-badge-motion">✏️ Seña de movimiento</span>`
          : cfg.mode === "orientation"
          ? `<span class="sign-badge sign-badge-orientation">↔️ Orientación especial</span>`
          : `<span class="sign-badge sign-badge-special">⚡ Posición especial</span>`;

        specialHtml = `
          <div class="sign-guide-box" style="margin:0.6rem 0; background:rgba(${cfg.color},0.08);
            border:1px solid rgba(${cfg.color},0.25); border-radius:8px; padding:0.75rem; text-align:left">
            <div style="margin-bottom:0.4rem">${modeLabel}</div>
            <strong style="font-size:0.82rem; color:rgba(${cfg.color},1)">${cfg.title}</strong>
            <ul style="margin:0.4rem 0 0.3rem 0; padding-left:1.1rem">${hintsHtml}</ul>
            ${cfg.warning ? `<div style="font-size:0.75rem; margin-top:0.4rem; opacity:0.9">${cfg.warning}</div>` : ""}
          </div>`;
      }

      const imgPath = window.getSignImagePath ? window.getSignImagePath(signId) : null;
      infoEl.innerHTML = `
        ${imgPath
          ? `<div class="collect-selected-photo-wrap"><img src="${imgPath}" alt="Seña ${signId}" class="collect-selected-photo-img" /></div>`
          : `<div class="collect-target-sign">${signId}</div>`
        }
        <div class="collect-sample-count">
          <strong style="font-size:1.1rem; color:var(--blue-dark); font-family:'Martian Mono',monospace;">${signId}</strong>
          &nbsp;·&nbsp;
          <span class="${count >= FRAMES_PER_SIGN ? "text-cyan" : "text-muted"}">${count}</span>
          <span class="text-muted"> / ${FRAMES_PER_SIGN}</span>
        </div>
        <div class="collect-mini-bar">
          <div class="collect-mini-fill" style="width:${pct}%"></div>
        </div>
        <p class="text-muted" style="font-size:0.75rem; margin-top:0.4rem; font-family:'Martian Mono',monospace; line-height:1.4;">
          ${count >= FRAMES_PER_SIGN
            ? "✅ Suficientes muestras. Puedes grabar más."
            : `Faltan ${FRAMES_PER_SIGN - count} frames.`}
        </p>
        ${specialHtml}
      `;
    }

    const btn = document.getElementById("btn-collect-record");
    const clearBtn = document.getElementById("btn-clear-sign");
    if (btn) {
      btn.disabled = false;
      // Etiqueta especial para señas de movimiento
      btn.textContent = cfg?.mode === "motion"
        ? `⏺ Grabar movimiento (${FRAMES_PER_SIGN} frames)`
        : `⏺ Grabar (${FRAMES_PER_SIGN} frames)`;
    }
    if (clearBtn) clearBtn.disabled = count === 0;
  }

  // ── Grabación ────────────────────────────────────────────────────────────

  _startRecording() {
    if (!this.currentSignId) {
      this._showStatus("Selecciona una seña primero.", "warn");
      return;
    }
    if (!this.mp?.isReady()) {
      this._showStatus("La cámara no está lista.", "warn");
      return;
    }

    this.isRecording    = true;
    this.capturedFrames = [];

    // Reset motion tracking
    this._motionTrail  = [];
    this._repCount     = 0;
    this._motionPeak   = 0;
    this._lastTipPos   = null;
    this._prevVel      = 0;
    this._inMotion     = false;

    const cfg = SIGN_CONFIG[this.currentSignId];
    const btn = document.getElementById("btn-collect-record");
    if (btn) {
      btn.textContent = `⏺ Grabando 0 / ${FRAMES_PER_SIGN}`;
      btn.disabled    = true;
      btn.classList.add("recording");
    }

    const msg = cfg?.mode === "motion"
      ? `⏺ Grabando "${this.currentSignId}" — ${cfg.warning}`
      : cfg?.mode === "orientation"
      ? `⏺ Grabando "${this.currentSignId}" — ${cfg.warning}`
      : `⏺ Grabando "${this.currentSignId}" — mantén la mano estable`;

    this._showStatus(msg, "info");
  }

  _captureFrame(landmarks, worldLandmarks = null) {
    if (!this.isRecording) return;

    const frame = { landmarks };
    if (worldLandmarks) frame.world_landmarks = worldLandmarks;
    this.capturedFrames.push(frame);

    const n   = this.capturedFrames.length;
    const btn = document.getElementById("btn-collect-record");
    if (btn) btn.textContent = `⏺ Grabando ${n} / ${FRAMES_PER_SIGN}`;

    // ── Tracking de movimiento para señas J y Z ───────────────────────────
    const cfg = SIGN_CONFIG[this.currentSignId];
    if (cfg?.mode === "motion" && cfg.tipIdx !== undefined) {
      const tip = landmarks[cfg.tipIdx];
      if (tip) {
        // Añadir al trail
        this._motionTrail.push({ x: tip.x, y: tip.y });
        if (this._motionTrail.length > TRAIL_MAX) {
          this._motionTrail.shift();
        }

        // Detectar repetición: velocidad sube y luego baja (1 ciclo = 1 rep)
        if (this._lastTipPos) {
          const dx = tip.x - this._lastTipPos.x;
          const dy = tip.y - this._lastTipPos.y;
          const vel = Math.sqrt(dx * dx + dy * dy);

          if (vel > 0.012) {
            this._inMotion = true;
            this._motionPeak = Math.max(this._motionPeak, vel);
          } else if (vel < 0.004 && this._inMotion && this._motionPeak > 0.025) {
            // Movimiento terminó: completó una rep
            this._inMotion    = false;
            this._motionPeak  = 0;
            this._repCount++;
          }
        }
        this._lastTipPos = { x: tip.x, y: tip.y };
      }
    }

    this._drawRecordingProgress(n, FRAMES_PER_SIGN);

    if (n >= FRAMES_PER_SIGN) {
      this.isRecording = false;
      this._saveSamples();
    }
  }

  // ── Guardado ─────────────────────────────────────────────────────────────

  async _saveSamples() {
    if (!this.capturedFrames.length) return;

    const btn = document.getElementById("btn-collect-record");
    if (btn) {
      btn.textContent = "💾 Guardando…";
      btn.classList.remove("recording");
    }
    this._showStatus("Guardando muestras en el servidor…", "info");

    try {
      const resp = await fetch("/api/train/collect", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          sign_id:  this.currentSignId,
          country:  this.currentCountry,
          samples:  this.capturedFrames,
        }),
      });
      const data = await resp.json();

      if (resp.ok) {
        // ── BUG FIX: actualizar sampleCounts Y re-renderizar con nuevos valores ──
        this.sampleCounts[this.currentSignId] = data.total_for_sign;

        // También actualizar el objeto en allSigns para consistencia
        const idx = this.allSigns.findIndex(s => s.sign_id === this.currentSignId);
        if (idx >= 0) {
          this.allSigns[idx] = {
            ...this.allSigns[idx],
            count:  data.total_for_sign,
            enough: data.total_for_sign >= FRAMES_PER_SIGN,
          };
        }

        this._showStatus(
          `✅ ${data.saved} frames guardados. Total "${this.currentSignId}": ${data.total_for_sign} / ${FRAMES_PER_SIGN}`,
          "success",
        );

        // Actualizar grid y panel de seña seleccionada
        this._renderSignGrid();
        this.selectSign(this.currentSignId);
        await this._updateTrainButtonFromAPI();
      } else {
        this._showStatus(`❌ Error guardando: ${data.detail}`, "error");
      }
    } catch (e) {
      this._showStatus(`❌ Error de red: ${e.message}`, "error");
    } finally {
      this.capturedFrames = [];
      this._motionTrail   = [];
      if (btn) {
        const cfg = SIGN_CONFIG[this.currentSignId];
        btn.textContent = cfg?.mode === "motion"
          ? `⏺ Grabar movimiento (${FRAMES_PER_SIGN} frames)`
          : `⏺ Grabar (${FRAMES_PER_SIGN} frames)`;
        btn.disabled = false;
      }
      if (this.canvasEl) Overlay.clearCanvas(this.canvasEl);
    }
  }

  async _clearCurrentSign() {
    if (!this.currentSignId) return;
    try {
      await fetch(`/api/train/collected/${this.currentCountry}/${this.currentSignId}`, {
        method: "DELETE",
      });
      this.sampleCounts[this.currentSignId] = 0;
      const idx = this.allSigns.findIndex(s => s.sign_id === this.currentSignId);
      if (idx >= 0) {
        this.allSigns[idx] = { ...this.allSigns[idx], count: 0, enough: false };
      }
      this._renderSignGrid();
      this.selectSign(this.currentSignId);
      this._showStatus(`🗑 Datos de "${this.currentSignId}" borrados.`, "info");
      await this._updateTrainButtonFromAPI();
    } catch (e) {
      this._showStatus("Error borrando datos.", "error");
    }
  }

  // ── Reentrenamiento ──────────────────────────────────────────────────────

  async _triggerRetrain() {
    const btn = document.getElementById("btn-retrain");
    if (btn) {
      btn.disabled    = true;
      btn.textContent = "⏳ Entrenando…";
    }
    this._showStatus("🔄 Entrenando modelo… esto toma ~30-60 segundos.", "info");

    try {
      const resp = await fetch(`/api/train/retrain/${this.currentCountry}`, { method: "POST" });
      const data = await resp.json();
      this._showStatus(data.message, "info");

      if (data.success) {
        this._pollRetrainStatus();
      }
    } catch (e) {
      this._showStatus(`❌ Error iniciando entrenamiento: ${e.message}`, "error");
      if (btn) {
        btn.disabled    = false;
        btn.textContent = "🚀 Reentrenar Modelo";
      }
    }
  }

  _pollRetrainStatus() {
    const interval = setInterval(async () => {
      try {
        const resp = await fetch(`/api/train/retrain-status/${this.currentCountry}`);
        const data = await resp.json();
        const { status } = data;
        const btn = document.getElementById("btn-retrain");

        if (status === "done") {
          clearInterval(interval);
          this._showStatus("🎉 ¡Modelo reentrenado! La práctica ya usa el nuevo modelo.", "success");
          if (btn) { btn.textContent = "✅ Reentrenado"; btn.disabled = false; }
          document.dispatchEvent(new CustomEvent("modelUpdated", {
            detail: { country: this.currentCountry },
          }));
        } else if (status?.startsWith("error:")) {
          clearInterval(interval);
          this._showStatus(`❌ Error: ${status.replace("error:", "")}`, "error");
          if (btn) { btn.textContent = "🚀 Reintentar"; btn.disabled = false; }
        } else if (status === "running") {
          if (btn) btn.textContent = "⏳ Entrenando…";
        }
      } catch (e) {
        clearInterval(interval);
      }
    }, 3000);
  }

  // ── Render ───────────────────────────────────────────────────────────────

  _renderSignGrid() {
    const grid = document.getElementById("collect-signs-grid");
    if (!grid) return;

    let totalSamples = 0;
    grid.innerHTML = "";

    this.allSigns.forEach(({ sign_id }) => {
      // ── BUG FIX: usar sampleCounts (actualizado al guardar), no allSigns ──
      const count  = this.sampleCounts[sign_id] ?? 0;
      const enough = count >= FRAMES_PER_SIGN;
      const cfg    = SIGN_CONFIG[sign_id];
      totalSamples += count;

      const item = document.createElement("div");
      item.className  = "collect-sign-item";
      item.dataset.signId = sign_id;
      if (sign_id === this.currentSignId) item.classList.add("selected");

      // Añadir clase especial para señas con guía
      if (cfg) item.classList.add(`sign-mode-${cfg.mode}`);

      const pct = Math.min(100, Math.round((count / FRAMES_PER_SIGN) * 100));

      // Indicador visual del tipo de seña
      const modeIcon = cfg?.mode === "motion" ? "✏️"
                     : cfg?.mode === "orientation" ? "↔"
                     : cfg?.mode === "static_special" ? "⚡" : "";

      const imgPath = window.getSignImagePath ? window.getSignImagePath(sign_id) : null;
      item.innerHTML = `
        <div class="collect-sign-photo-wrap">
          ${imgPath
            ? `<img src="${imgPath}" alt="${sign_id}" class="collect-sign-photo-img" />`
            : `<span class="collect-sign-letter-fallback">${sign_id}</span>`
          }
          <span class="collect-sign-overlay-letter">${sign_id}${modeIcon ? `<span style="font-size:0.5em;opacity:0.7">${modeIcon}</span>` : ""}</span>
        </div>
        <div class="collect-sign-bar">
          <div class="collect-sign-fill ${enough ? "full" : ""}" style="width:${pct}%"></div>
        </div>
        <div class="collect-sign-num ${enough ? "text-cyan" : "text-muted"}">${count}</div>
      `;
      item.addEventListener("click", () => this.selectSign(sign_id));
      grid.appendChild(item);
    });

    // Actualizar total
    const readyCount = document.getElementById("signs-ready-count");
    if (readyCount) {
      const ready = this.allSigns.filter(s => (this.sampleCounts[s.sign_id] ?? 0) >= FRAMES_PER_SIGN).length;
      readyCount.textContent = `${ready}/${this.allSigns.length} listas · ${totalSamples} total`;
    }
  }

  _updateTrainButton(status) {
    const btn  = document.getElementById("btn-retrain");
    const info = document.getElementById("train-ready-info");
    if (!btn) return;

    const ready     = status.ready_to_train;
    const signsDone = status.signs_ready;
    const total     = status.total_signs;

    btn.disabled = !ready;
    if (info) {
      info.textContent = ready
        ? `${signsDone}/${total} señas listas — puedes reentrenar`
        : `${signsDone}/${total} señas con ${status.min_samples_needed}+ muestras`;
    }
  }

  _updateReadyCount(status) {
    const el = document.getElementById("signs-ready-count");
    if (!el) return;
    const total = Object.values(this.sampleCounts).reduce((a, b) => a + b, 0);
    el.textContent = `${status.signs_ready}/${status.total_signs} listas · ${total} total`;
  }

  async _updateTrainButtonFromAPI() {
    try {
      const resp = await fetch(`/api/train/status/${this.currentCountry}`);
      const status = await resp.json();
      this._updateTrainButton(status);
      this._updateReadyCount(status);
    } catch (e) {}
  }

  // ── Dibujo ───────────────────────────────────────────────────────────────

  _drawFrame(landmarks) {
    if (!this.canvasEl) return;
    Overlay.clearCanvas(this.canvasEl);
    Overlay.drawUserHand(this.canvasEl, landmarks, 0.8);

    const cfg = SIGN_CONFIG[this.currentSignId];

    // Dibujar guía de orientación para G/H (flecha horizontal)
    if (cfg?.mode === "orientation") {
      Overlay.drawOrientationGuide(this.canvasEl, cfg.color);
    }

    // Dibujar trail de movimiento para J/Z durante grabación
    if (this.isRecording && cfg?.mode === "motion" && this._motionTrail.length >= 2) {
      Overlay.drawMotionTrail(this.canvasEl, this._motionTrail, cfg.color);

      // Badge de repeticiones completadas
      if (this._repCount > 0) {
        Overlay.drawRepBadge(this.canvasEl, this._repCount, cfg.color);
      }
    }

    this._drawFpsOverlay();
  }

  _drawFpsOverlay() {
    if (!this.canvasEl || !this._currentFps) return;
    const ctx = this.canvasEl.getContext("2d");
    const W   = this.canvasEl.width;
    ctx.save();
    ctx.font      = "bold 12px monospace";
    ctx.textAlign = "right";
    ctx.fillStyle = this._currentFps >= 25 ? "rgba(0,245,200,0.8)"
                  : this._currentFps >= 15 ? "rgba(255,200,50,0.8)"
                  : "rgba(255,80,80,0.8)";
    ctx.fillText(`${this._currentFps} FPS`, W - 6, 18);
    ctx.restore();
  }

  _drawRecordingProgress(current, total) {
    if (!this.canvasEl) return;
    const ctx  = this.canvasEl.getContext("2d");
    const W    = this.canvasEl.width;
    const H    = this.canvasEl.height;
    const pct  = current / total;
    const barH = 6;
    const barY = H - barH - 8;

    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.4)";
    ctx.fillRect(8, barY, W - 16, barH);
    ctx.fillStyle = pct > 0.7 ? "rgba(0,245,200,0.9)" : "rgba(124,58,237,0.9)";
    ctx.fillRect(8, barY, (W - 16) * pct, barH);

    ctx.fillStyle = "#e0e0ff";
    ctx.font      = "bold 14px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`${current} / ${total}`, W - 10, barY - 6);

    // FPS
    ctx.font      = "bold 12px monospace";
    ctx.fillStyle = this._currentFps >= 25 ? "rgba(0,245,200,0.8)"
                  : this._currentFps >= 15 ? "rgba(255,200,50,0.8)"
                  : "rgba(255,80,80,0.8)";
    ctx.fillText(`${this._currentFps} FPS`, W - 6, 18);
    ctx.restore();
  }

  _showStatus(msg, type) {
    const el = document.getElementById("collect-status");
    if (!el) return;
    el.textContent   = msg;
    el.className     = `collect-status collect-status-${type}`;
    el.style.display = msg ? "block" : "none";
  }

  stop() {
    this.isRecording    = false;
    this.capturedFrames = [];
    this._motionTrail   = [];
    if (this.mp) this.mp.stopCamera();
    this.cameraActive = false;
  }
}

window.CollectMode = CollectMode;
