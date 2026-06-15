/**
 * practice.js — Lógica del modo práctica.
 *
 * Novedades:
 *   - Feedback de postura en tiempo real (qué dedos están mal)
 *   - Toast de "Modelo actualizado" cuando el usuario reentrena desde ⚙️
 *   - Hot-reload instantáneo: escucha el evento modelUpdated sin reiniciar
 */

class PracticeMode {
  constructor() {
    this.mp = null;
    this.wsPredictor = null;
    this.currentCountry = "lsc";
    this.lessonSigns = [];
    this.currentIdx = 0;
    this.score = 0;
    this.lastLandmarks = null;
    this.predictThrottle = 0;
    this.PREDICT_INTERVAL_MS = 50;
    this.motionTrail = [];
    this.motionTrailTip = null;
    this.isActive = false;
    this.successTimeout = null;
    this.flashAnim = null;
    this.matchFlashAlpha = 0;

    // ── Sistema de dos fases para señas dinámicas ─────────────────────────
    this.DYNAMIC_SIGNS = new Set(["J", "Z", "S", "Ñ", "G", "H"]);
    this.motionPhase = "static";         // "static" | "position" | "motion"
    this.motionPhaseProgress = 1.0;      // 0.0 → 1.0 (progreso de hold)

    // ── Modo CNN ─────────────────────────────────────────────────────────
    this.cnnMode = false;
    this.cnnReady = false;
    this._cnnCanvas = null;

    this.videoEl  = document.getElementById("video");
    this.canvasEl = document.getElementById("overlay-canvas");

    this._bindUI();
    this._initCnnCanvas();
    this._checkCnnStatus();
  }

  // ── CNN helpers ─────────────────────────────────────────────────────────────

  _initCnnCanvas() {
    this._cnnCanvas = document.createElement("canvas");
    this._cnnCanvas.width  = 100;
    this._cnnCanvas.height = 100;
  }

  async _checkCnnStatus() {
    try {
      const r = await fetch("/api/cnn/status");
      const d = await r.json();
      this.cnnReady = d.loaded === true;
      const btn = document.getElementById("btn-cnn-toggle");
      if (btn) {
        btn.disabled = !this.cnnReady;
        btn.title = this.cnnReady
          ? `CNN TF activo (${d.num_classes} clases LESHO)`
          : (d.error ?? "CNN no disponible");
      }
      const statusEl = document.getElementById("cnn-status-badge");
      if (statusEl) {
        statusEl.textContent = this.cnnReady ? "CNN ✓" : "CNN ✗";
        statusEl.className   = "cnn-badge " + (this.cnnReady ? "cnn-ok" : "cnn-off");
      }
    } catch (_) {}
  }

  _captureHandCrop(landmarks) {
    if (!this._cnnCanvas || !this.videoEl) return null;
    const vid = this.videoEl;
    const W   = vid.videoWidth  || vid.clientWidth  || 640;
    const H   = vid.videoHeight || vid.clientHeight || 480;

    if (!W || !H) return null;

    const xs = landmarks.map(l => l.x);
    const ys = landmarks.map(l => l.y);
    let minX = Math.min(...xs) * W;
    let maxX = Math.max(...xs) * W;
    let minY = Math.min(...ys) * H;
    let maxY = Math.max(...ys) * H;

    const padX = (maxX - minX) * 0.35;
    const padY = (maxY - minY) * 0.35;
    minX = Math.max(0, minX - padX);
    maxX = Math.min(W, maxX + padX);
    minY = Math.max(0, minY - padY);
    maxY = Math.min(H, maxY + padY);

    const cropW = maxX - minX || 1;
    const cropH = maxY - minY || 1;

    const ctx = this._cnnCanvas.getContext("2d");
    ctx.drawImage(vid, minX, minY, cropW, cropH, 0, 0, 100, 100);

    const imageData = ctx.getImageData(0, 0, 100, 100);
    const d = imageData.data;
    for (let i = 0; i < d.length; i += 4) {
      const gray = 0.299 * d[i] + 0.587 * d[i+1] + 0.114 * d[i+2];
      d[i] = d[i+1] = d[i+2] = gray;
    }
    ctx.putImageData(imageData, 0, 0);
    return this._cnnCanvas.toDataURL("image/png").replace(/^data:image\/png;base64,/, "");
  }

  async _predictCNN(landmarks) {
    const imgB64 = this._captureHandCrop(landmarks);
    if (!imgB64) return null;
    try {
      const resp = await fetch("/api/predict-cnn", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ image_b64: imgB64, fmt: "png" }),
      });
      if (!resp.ok) return null;
      const data = await resp.json();
      if (!data.model_ready) return null;

      this._updateCnnResult(data);

      const target = this.currentTarget();
      const isCorrect = target && data.letter === target.id;
      return {
        sign_id:    data.letter,
        sign_name:  data.letter,
        confidence: data.confidence,
        score:      data.confidence * 100,
        is_correct: isCorrect,
        fingers_up: [false, false, false, false, false],
      };
    } catch (_) {
      return null;
    }
  }

  _updateCnnResult(data) {
    const el = document.getElementById("cnn-result");
    if (!el) return;
    const pct = Math.round(data.confidence * 100);
    const top3 = (data.top3 || []).map(t =>
      `<span class="cnn-top-item">${t.letter} ${Math.round(t.confidence*100)}%</span>`
    ).join(" ");
    el.innerHTML = `
      <div class="cnn-letter">${data.letter}</div>
      <div class="cnn-conf">${pct}% · <small>${top3}</small></div>
    `;
    el.classList.remove("hidden");
  }

  toggleCnnMode() {
    if (!this.cnnReady) return;
    this.cnnMode = !this.cnnMode;
    const btn = document.getElementById("btn-cnn-toggle");
    if (btn) {
      btn.classList.toggle("active", this.cnnMode);
      btn.textContent = this.cnnMode ? "🤖 CNN activo" : "🤖 Usar CNN";
    }
    const cnnPanel = document.getElementById("cnn-result-panel");
    if (cnnPanel) cnnPanel.classList.toggle("hidden", !this.cnnMode);
    console.log("[Practice] CNN mode:", this.cnnMode ? "ON" : "OFF");
  }

  _bindUI() {
    // ── Evento unificado de dos manos ────────────────────────────────────────
    document.addEventListener("handsUpdate", (e) => {
      if (!this.isActive) return;
      const { right, left, total_hands, pose } = e.detail;
      const dominant = right ?? left;
      if (!dominant) return;

      this.lastLandmarks           = dominant.landmarks;
      this.lastLandmarksRight      = right?.landmarks ?? null;
      this.lastLandmarksLeft       = left?.landmarks  ?? null;
      this.lastWorldLandmarksRight = right?.worldLandmarks ?? null;
      this.lastWorldLandmarksLeft  = left?.worldLandmarks  ?? null;
      this.lastPoseLandmarks       = pose ?? null;
      this.lastHandsCount          = total_hands;

      this._drawFrame(dominant.landmarks, left?.landmarks ?? null, dominant.worldLandmarks ?? null);
      this._throttledPredict(
        dominant.landmarks,
        right?.landmarks ?? null, left?.landmarks ?? null,
        right?.worldLandmarks ?? null, left?.worldLandmarks ?? null,
        pose ?? null,
      );
    });

    document.addEventListener("noHand", () => {
      if (!this.isActive) return;
      Overlay.clearCanvas(this.canvasEl);
      Overlay.drawScanLine(this.canvasEl);
      this._setConfidence(0);
      this._clearPostureFeedback();
    });

    document.addEventListener("cameraReady", () => {
      document.getElementById("cam-status-dot").classList.add("active");
      document.getElementById("cam-status-text").textContent = "Detectando";
      document.querySelector(".camera-prompt")?.remove();
    });

    document.addEventListener("cameraError", (e) => {
      this._showCameraPrompt("Error de Cámara", e.detail || "No se pudo acceder a la cámara", false);
    });

    document.addEventListener("mediapipeError", (e) => {
      this._showCameraPrompt(
        "Error de MediaPipe",
        "No se pudo cargar el modelo de detección. Verifica tu conexión a internet.",
        false
      );
    });

    // ── Hot-reload: el modelo fue reentrenado → notificar sin reiniciar ──────
    document.addEventListener("modelUpdated", (e) => {
      const country = e.detail?.country ?? "?";
      this._showModelToast(
        `🎉 Modelo ${country.toUpperCase()} actualizado`,
        "¡Tu entrenamiento ya está activo! La práctica usa el nuevo modelo.",
      );
      // Resetear suavizador para que no use predicciones del modelo viejo
      if (this.wsPredictor?.isConnected()) {
        this.wsPredictor.disconnect();
        setTimeout(() => {
          this.wsPredictor = new WsPredictor(this.currentCountry);
          this.wsPredictor.connect();
          this.wsPredictor.onResult = (result) => this._handlePrediction(result);
        }, 600);
      }
    });

    document.getElementById("btn-start-practice")?.addEventListener("click", () => this.start());
    document.getElementById("btn-skip-sign")?.addEventListener("click", () => this.nextSign());
    document.getElementById("btn-restart-lesson")?.addEventListener("click", () => this.restart());
    document.getElementById("btn-cnn-toggle")?.addEventListener("click", () => this.toggleCnnMode());
  }

  currentTarget() {
    return this.lessonSigns[this.currentIdx] ?? null;
  }

  async start(country = null, signs = null) {
    if (country) this.currentCountry = country;

    if (signs) {
      this.lessonSigns = signs;
    } else {
      const data = await ApiClient.getSigns(this.currentCountry);
      if (!data?.signs?.length) {
        console.error("[Práctica] No se cargaron señas");
        return;
      }
      this.lessonSigns = data.signs;
    }

    this.currentIdx = 0;
    this.score = 0;
    this.isActive = true;

    this._syncCanvas();
    this._updateTargetUI();
    this._updateProgressUI();

    if (!this.mp) {
      this.mp = new MediaPipeHandler();
      this._showCameraPrompt(
        "Cargando modelo de IA…",
        "Descargando MediaPipe (~8MB, solo la primera vez). Espera un momento.",
        false
      );
      const ok = await this.mp.init();
      if (!ok) {
        this._showCameraPrompt(
          "Error de MediaPipe",
          "No se pudo cargar el modelo de detección. Verifica tu conexión a internet.",
          false
        );
        return;
      }
    }

    this._showCameraPrompt(
      "Permitir Cámara",
      "SignLingo necesita tu cámara para ver tus señas con las manos.",
      true
    );
    await this.mp.startCamera(this.videoEl);

    this.wsPredictor = new WsPredictor(this.currentCountry);
    this.wsPredictor.connect();
    this.wsPredictor.onResult = (result) => this._handlePrediction(result);
  }

  stop() {
    this.isActive = false;
    if (this.mp) this.mp.stopCamera();
    if (this.wsPredictor) this.wsPredictor.disconnect();
    Overlay.clearCanvas(this.canvasEl);
    document.getElementById("cam-status-dot")?.classList.remove("active");
    document.getElementById("cam-status-text").textContent = "Apagada";
  }

  restart() {
    this.currentIdx = 0;
    this.score = 0;
    this._updateTargetUI();
    this._updateProgressUI();
    document.getElementById("practice-complete")?.classList.add("hidden");
  }

  nextSign() {
    if (this.currentIdx < this.lessonSigns.length - 1) {
      this.currentIdx++;
      this._clearPostureFeedback();
      this._clearPhaseIndicator();
      this.motionPhase = "static";
      this.motionPhaseProgress = 1.0;
      this.motionTrail = [];
      this._updateTargetUI();
      this._updateProgressUI();
    } else {
      this._showComplete();
    }
  }

  _syncCanvas() {
    const video  = this.videoEl;
    const canvas = this.canvasEl;
    const ro = new ResizeObserver(() => {
      const W = video.clientWidth  || 640;
      const H = video.clientHeight || 480;
      canvas.width  = W;
      canvas.height = H;
      const tc = document.getElementById("three-hand-canvas");
      if (tc) { tc.width = W; tc.height = H; }
    });
    ro.observe(video);
    canvas.width  = video.clientWidth  || 640;
    canvas.height = video.clientHeight || 480;
    const tc = document.getElementById("three-hand-canvas");
    if (tc) { tc.width = canvas.width; tc.height = canvas.height; }
  }

  _showCameraPrompt(title, message, showBtn) {
    const container = this.videoEl.parentElement;
    const existing = container.querySelector(".camera-prompt");
    if (existing) existing.remove();

    const prompt = document.createElement("div");
    prompt.className = "camera-prompt";
    prompt.innerHTML = `
      <div class="camera-prompt-icon">📷</div>
      <h3>${title}</h3>
      <p>${message}</p>
      ${showBtn ? `<button class="btn btn-primary" id="allow-camera-btn">Iniciar Cámara</button>` : ""}
    `;
    container.appendChild(prompt);

    prompt.querySelector("#allow-camera-btn")?.addEventListener("click", async () => {
      prompt.remove();
    });
  }

  async _throttledPredict(
    landmarks,
    landmarksRight = null, landmarksLeft = null,
    worldLandmarksRight = null, worldLandmarksLeft = null,
    poseLandmarks = null,
  ) {
    const now = Date.now();
    if (now - this.predictThrottle < this.PREDICT_INTERVAL_MS) return;
    this.predictThrottle = now;

    const target = this.currentTarget();
    if (!target) return;

    let result = null;

    if (this.cnnMode && this.cnnReady) {
      result = await this._predictCNN(landmarks);
    } else if (this.wsPredictor?.isConnected()) {
      this.wsPredictor.send(
        landmarks, target.id,
        landmarksRight, landmarksLeft,
        worldLandmarksRight, worldLandmarksLeft,
        poseLandmarks,
      );
    } else {
      result = await ApiClient.predict(
        landmarks, this.currentCountry, target.id,
        landmarksRight, landmarksLeft,
        worldLandmarksRight, worldLandmarksLeft,
      );
    }

    if (result) this._handlePrediction(result);
  }

  _handlePrediction(result) {
    if (!result || !this.isActive) return;

    const confidence = result.confidence ?? 0;
    const score      = result.score ?? 0;
    let   isCorrect  = result.is_correct ?? false;
    const signId     = result.sign_id ?? "?";
    const signName   = result.sign_name ?? signId;

    // ── Sistema de dos fases ──────────────────────────────────────────────────
    const target = this.currentTarget();
    const isDynamic = target && this.DYNAMIC_SIGNS.has(target.id);

    if (isDynamic && result.motion_phase) {
      this.motionPhase         = result.motion_phase;
      this.motionPhaseProgress = result.motion_phase_progress ?? 0;
      this._updatePhaseIndicator(result.motion_phase, result.motion_phase_progress ?? 0, target.id);
    } else {
      this.motionPhase = "static";
      this._clearPhaseIndicator();
    }

    // ── Finger-state primary detection ───────────────────────────────────────
    // Solo usar finger-state para señas estáticas o cuando ya está en fase motion
    const canCheckFingers = !isDynamic || this.motionPhase === "motion";
    if (canCheckFingers
        && !isCorrect
        && target?.finger_states?.length === 5
        && result.fingers_up?.length === 5) {
      const exactMatch = result.fingers_up.every(
        (v, i) => v === (target.finger_states[i] ?? false)
      );
      if (exactMatch) isCorrect = true;
    }

    // ── Body-zone validation (señas de palabras con poses) ────────────────────
    // Si la seña requiere posición corporal, verificar zona antes de marcar correcto
    if (isCorrect && target?.body_zone && target.body_zone !== "anywhere") {
      const detectedZone = result.body_zone;
      if (detectedZone && detectedZone !== "unknown") {
        if (detectedZone !== target.body_zone) {
          isCorrect = false;
          // Mostrar hint de posición corporal
          const zoneNames = { face: "frente/sien", chin: "barbilla", chest: "pecho", belly: "abdomen" };
          this._showBodyZoneHint(zoneNames[target.body_zone] ?? target.body_zone);
        }
      }
    }

    if (result.motion_trail && result.motion_trail.length > 1) {
      this.motionTrail    = result.motion_trail;
      this.motionTrailTip = result.motion_sign === "Z" ? "index" : "pinky";
    } else if (!result.motion_sign && this.motionPhase !== "motion") {
      this.motionTrail = [];
    }

    // Store latest states for Three.js finger-state coloring
    this.lastConfidence    = confidence;
    this.lastFingersUp     = result.fingers_up ?? null;
    this.lastTargetFingers = target?.finger_states ?? null;

    this._setConfidence(confidence, score);
    this._setPrediction(signId, signName, isCorrect);
    this._setFingers(result.fingers_up ?? []);

    // ── Feedback de postura ───────────────────────────────────────────────────
    if (isDynamic && this.motionPhase === "position") {
      // En fase posición: mostrar qué dedos corregir para fijar la postura base
      if (target && result.fingers_up?.length === 5) {
        this._showPostureFeedback(result.fingers_up, target.finger_states ?? []);
      }
    } else if (target && !isCorrect && result.fingers_up?.length === 5) {
      this._showPostureFeedback(result.fingers_up, target.finger_states ?? []);
    } else if (isCorrect) {
      this._clearPostureFeedback();
    }

    if (result.motion_sign) {
      this._showMotionBadge(result.motion_sign);
    }

    if (isCorrect) {
      this._onCorrect(score > 0 ? score : 50);
    }
  }

  // ── Phase indicator ───────────────────────────────────────────────────────

  _updatePhaseIndicator(phase, progress, signId) {
    let indicator = document.getElementById("motion-phase-indicator");
    if (!indicator) {
      indicator = document.createElement("div");
      indicator.id = "motion-phase-indicator";
      const container = this.canvasEl?.parentElement ?? document.body;
      container.appendChild(indicator);
    }

    indicator.style.display = "flex";

    if (phase === "position") {
      const pct = Math.round(progress * 100);
      indicator.className = "motion-phase-indicator phase-position";
      indicator.innerHTML = `
        <div class="phase-icon">✋</div>
        <div class="phase-content">
          <div class="phase-label">FIJA LA POSICIÓN</div>
          <div class="phase-bar-wrap">
            <div class="phase-bar-fill" style="width:${pct}%"></div>
          </div>
        </div>
        <div class="phase-pct">${pct}%</div>
      `;
    } else if (phase === "motion") {
      const MOTION_TIPS = {
        J: "Traza una J con el meñique ↓",
        Z: "Dibuja una Z con el índice →",
        S: "Sacude el puño de lado a lado ↔",
        Ñ: "Mueve la mano en forma de tilde ~",
        G: "Señala horizontalmente →",
        H: "Dedos horizontales →",
      };
      indicator.className = "motion-phase-indicator phase-motion";
      indicator.innerHTML = `
        <div class="phase-icon">✨</div>
        <div class="phase-content">
          <div class="phase-label">AHORA MUEVE</div>
          <div class="phase-tip">${MOTION_TIPS[signId] ?? "Realiza el movimiento"}</div>
        </div>
      `;
    }
  }

  _clearPhaseIndicator() {
    const el = document.getElementById("motion-phase-indicator");
    if (el) el.style.display = "none";
  }

  // ── Feedback de postura ────────────────────────────────────────────────────

  /**
   * Compara el estado actual de los dedos con el objetivo.
   * Muestra pistas de corrección: "Levanta el Índice", "Baja el Pulgar", etc.
   */
  _showPostureFeedback(fingersUp, targetStates) {
    if (!fingersUp?.length || !targetStates?.length) return;

    const NAMES   = ["Pulgar", "Índice", "Medio", "Anular", "Meñique"];
    const EMOJIS  = ["👍", "☝️", "🖕", "💍", "🤙"];

    const hints = [];
    for (let i = 0; i < 5; i++) {
      const userUp   = fingersUp[i]   ?? false;
      const targetUp = targetStates[i] ?? false;
      if (userUp !== targetUp) {
        hints.push({
          text:  targetUp ? `Levanta el ${NAMES[i]}` : `Baja el ${NAMES[i]}`,
          emoji: EMOJIS[i],
          up:    targetUp,
        });
      }
    }

    const newKey = hints.map(h => h.text).join("|");

    // Solo actualizar si el contenido cambió Y ha sido estable 280ms (anti-flicker)
    if (newKey === this._lastHintsKey) return;

    clearTimeout(this._hintsDebounce);
    this._hintsDebounce = setTimeout(() => {
      this._lastHintsKey = newKey;
      const container = document.getElementById("posture-hints");
      if (!container) return;

      if (!hints.length) {
        container.innerHTML = "";
        return;
      }

      container.innerHTML = hints.slice(0, 3).map(h => `
        <span class="posture-hint posture-hint-${h.up ? "up" : "down"}">
          ${h.emoji} ${h.text}
        </span>
      `).join("");
    }, 280);
  }

  _clearPostureFeedback() {
    clearTimeout(this._hintsDebounce);
    this._lastHintsKey = "";
    const el = document.getElementById("posture-hints");
    if (el) el.innerHTML = "";
  }

  _showBodyZoneHint(zoneName) {
    const container = document.getElementById("posture-hints");
    if (!container) return;
    container.innerHTML = `
      <span class="posture-hint posture-hint-down">
        📍 Acerca la mano a la <strong>${zoneName}</strong>
      </span>
    `;
    clearTimeout(this._bodyZoneHintTimer);
    this._bodyZoneHintTimer = setTimeout(() => {
      if (container) container.innerHTML = "";
    }, 1800);
  }

  // ── Toast de modelo actualizado ──────────────────────────────────────────

  _showModelToast(title, body) {
    const existing = document.getElementById("model-toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.id = "model-toast";
    toast.className = "model-toast";
    toast.innerHTML = `
      <div class="model-toast-title">${title}</div>
      <div class="model-toast-body">${body}</div>
    `;
    document.body.appendChild(toast);

    // Entrada animada
    requestAnimationFrame(() => toast.classList.add("model-toast-show"));

    // Auto-dismiss en 5s
    setTimeout(() => {
      toast.classList.remove("model-toast-show");
      toast.addEventListener("transitionend", () => toast.remove(), { once: true });
    }, 5000);
  }

  // ── Motion badge ─────────────────────────────────────────────────────────

  _showMotionBadge(signId) {
    let badge = document.getElementById("motion-badge");
    if (!badge) {
      badge = document.createElement("div");
      badge.id = "motion-badge";
      badge.style.cssText = `
        position:absolute; top:12px; left:50%; transform:translateX(-50%);
        background: rgba(0,245,200,0.18); border: 1px solid rgba(0,245,200,0.5);
        color: #00f5c8; font-size:13px; font-weight:700; padding:4px 14px;
        border-radius:20px; pointer-events:none; z-index:10;
        text-shadow: 0 0 8px rgba(0,245,200,0.8);
        box-shadow: 0 0 12px rgba(0,245,200,0.3);
      `;
      const container = this.canvasEl?.parentElement ?? document.body;
      container.style.position = "relative";
      container.appendChild(badge);
    }
    badge.textContent = `✋ Movimiento: ${signId}`;
    badge.style.opacity = "1";
    clearTimeout(this._motionBadgeTimer);
    this._motionBadgeTimer = setTimeout(() => { badge.style.opacity = "0"; }, 1200);
  }

  // ── Events ───────────────────────────────────────────────────────────────

  _onCorrect(score) {
    if (this.successTimeout) return;

    this.score += Math.round(score);
    document.getElementById("hud-score").textContent = this.score;

    document.getElementById("success-burst")?.classList.add("show");
    this.matchFlashAlpha = 1.0;

    const target = this.currentTarget();

    ApiClient.saveProgress(
      target?.id,
      this.currentCountry,
      true,
      Date.now() - this.predictThrottle,
    );

    // ── Auto-guardar muestra de entrenamiento al acertar ──────────────────────
    this._autoSaveSample(target);

    this.successTimeout = setTimeout(() => {
      document.getElementById("success-burst")?.classList.remove("show");
      this.successTimeout = null;
      this.nextSign();
    }, 1400);
  }

  _autoSaveSample(target) {
    if (!target || !this.lastLandmarksRight) return;
    const sample = { landmarks: this.lastLandmarksRight };
    if (this.lastWorldLandmarksRight)
      sample.world_landmarks = this.lastWorldLandmarksRight;
    if (this.lastLandmarksLeft)
      sample.landmarks_left = this.lastLandmarksLeft;
    if (this.lastWorldLandmarksLeft)
      sample.world_landmarks_left = this.lastWorldLandmarksLeft;
    ApiClient.collectSamples(target.id, this.currentCountry, [sample])
      .then(res => {
        if (res?.saved) console.log(`[Auto-save] ✅ '${target.id}' guardado. Total: ${res.total_for_sign}`);
      })
      .catch(() => {});
  }

  _drawFrame(landmarks, landmarksLeft = null, worldLandmarks = null) {
    Overlay.clearCanvas(this.canvasEl);

    if (this.matchFlashAlpha > 0) {
      Overlay.drawMatchFlash(this.canvasEl, this.matchFlashAlpha);
      this.matchFlashAlpha = Math.max(0, this.matchFlashAlpha - 0.08);
    }

    Overlay.drawScanLine(this.canvasEl);

    if (this.motionTrail?.length > 1 && Overlay.drawMotionTrail) {
      const trailColor = this.motionTrailTip === "index" ? "255, 180, 50" : "0, 245, 200";
      Overlay.drawMotionTrail(this.canvasEl, this.motionTrail, trailColor);
    }

    // ── Hold-progress stability ring (2D HUD layer) ───────────────────────
    if (this.holdFrames > 0 && landmarks?.[0] && Overlay.drawStabilityRing) {
      const wrist = landmarks[0];
      const W     = this.canvasEl.width;
      const H     = this.canvasEl.height;
      const prog  = Math.min(1, this.holdFrames / (this.HOLD_FRAMES_REQUIRED || 12));
      Overlay.drawStabilityRing(
        this.canvasEl,
        (1 - wrist.x) * W,
        wrist.y * H,
        prog,
      );
    }

    // ── Three.js 3D skeleton (renderer principal) ─────────────────────────
    const threeCanvas = document.getElementById("three-hand-canvas");
    if (threeCanvas && window.ThreeHandRenderer) {
      ThreeHandRenderer.syncSize(threeCanvas, threeCanvas.width, threeCanvas.height);

      // ── Mano fantasma: seña objetivo en esquina inferior derecha ─────────
      const targetFingers = this.lastTargetFingers ?? this.currentTarget()?.finger_states;
      const estaCorrecta  = !!this.successTimeout;
      if (targetFingers?.length === 5 && !estaCorrecta && Overlay.buildGhostLandmarks) {
        const W = threeCanvas.width  || 640;
        const H = threeCanvas.height || 480;

        // Generar posiciones en espacio de píxeles (canvas completo)
        const ghostPx = Overlay.buildGhostLandmarks(targetFingers, W, H);

        // Escalar al 40 % y desplazar a esquina inferior derecha
        const ESCALA   = 0.38;
        const cx_dst   = W * 0.83;
        const cy_dst   = H * 0.70;
        const cx_src   = W * 0.50;
        const cy_src   = H * 0.62;
        const ghostLms = ghostPx.map(p => ({
          x: cx_dst + (p.x - cx_src) * ESCALA,
          y: cy_dst + (p.y - cy_src) * ESCALA,
        }));

        ThreeHandRenderer.drawGhost(threeCanvas, ghostLms);

        // Etiqueta "▷ OBJETIVO" sobre la mano fantasma en canvas 2D
        const hud = this.canvasEl.getContext("2d");
        hud.save();
        const labelY = cy_dst - H * 0.14;
        hud.fillStyle = "rgba(120, 195, 255, 0.70)";
        hud.font      = "bold 10px 'Courier New', monospace";
        hud.textAlign = "center";
        hud.shadowColor = "rgba(80,160,255,0.8)";
        hud.shadowBlur  = 6;
        hud.fillText("▷ OBJETIVO", cx_dst, labelY);
        hud.restore();
      }

      ThreeHandRenderer.draw(threeCanvas, landmarks, worldLandmarks, {
        confidence:    this.lastConfidence    ?? 1,
        fingersUp:     this.lastFingersUp     ?? null,
        targetFingers: this.lastTargetFingers ?? null,
      });
      if (landmarksLeft) {
        const worldLeft = this.lastWorldLandmarksLeft ?? null;
        ThreeHandRenderer.drawSecondary(threeCanvas, landmarksLeft, worldLeft, 0.72);
      }
    } else {
      Overlay.drawUserHand(this.canvasEl, landmarks);
      if (landmarksLeft && Overlay.drawUserHandSecondary) {
        Overlay.drawUserHandSecondary(this.canvasEl, landmarksLeft);
      }
    }
  }

  _setConfidence(confidence, score = 0) {
    const pct  = Math.round(confidence * 100);
    const fill = document.getElementById("confidence-fill");
    const val  = document.getElementById("confidence-value");
    if (fill) fill.style.width = `${pct}%`;
    if (val)  val.textContent  = `${pct}%`;

    // Anillo SVG de confianza
    const arc = document.getElementById("conf-ring-arc");
    if (arc) {
      const circumference = 251.2;
      const offset = circumference - (circumference * Math.min(pct, 100) / 100);
      arc.style.strokeDashoffset = offset;
      arc.style.stroke = pct >= 35 ? "#22c55e" : pct >= 20 ? "#f59e0b" : "#225688";
    }

    const circle = document.getElementById("precision-circle");
    if (circle) circle.textContent = `${pct}%`;
  }

  _setPrediction(signId, signName, isMatch) {
    const el = document.getElementById("prediction-value");
    if (!el) return;
    el.textContent = signId === "UNKNOWN" ? "—" : signId;
    el.classList.toggle("match", isMatch);
  }

  /**
   * Muestra estado de dedos con indicadores de correcto/incorrecto
   * comparando con la seña objetivo actual.
   */
  _setFingers(fingerStates) {
    const names  = ["👍", "☝️", "🖕", "💍", "🤙"];
    const labels = ["Pulgar", "Índice", "Medio", "Anular", "Meñique"];
    const row = document.getElementById("fingers-row");
    if (!row) return;

    const target = this.currentTarget();
    const targetStates = target?.finger_states ?? null;

    row.innerHTML = "";
    fingerStates.forEach((up, i) => {
      const item = document.createElement("div");
      item.className = "finger-item";

      let statusClass = "";
      if (targetStates && targetStates.length > i) {
        statusClass = (up === targetStates[i]) ? "finger-correct" : "finger-wrong";
      }

      item.innerHTML = `
        <div class="finger-icon ${up ? "up" : ""} ${statusClass}">${names[i]}</div>
        <span>${labels[i]}</span>
      `;
      row.appendChild(item);
    });
  }

  _updateTargetUI() {
    const target = this.currentTarget();
    if (!target) return;

    document.getElementById("target-sign-id").textContent    = target.id;
    document.getElementById("target-sign-name").textContent  = target.name;
    document.getElementById("lesson-counter").textContent =
      `${this.currentIdx + 1} / ${this.lessonSigns.length}`;

    const instrEl = document.getElementById("practice-sign-instructions");
    if (instrEl) {
      const tips = target.tips ?? [];
      if (tips.length) {
        instrEl.innerHTML = tips.map(t => `<span class="instr-tip">• ${t}</span>`).join("");
      } else if (target.description) {
        instrEl.innerHTML = `<span class="instr-tip">• ${target.description}</span>`;
      } else {
        instrEl.innerHTML = `<span class="instr-tip">• Realiza la seña con la mano derecha frente a la cámara.</span>`;
      }
    }

    document.getElementById("practice-complete")?.classList.add("hidden");

    // Mostrar foto real de la seña si está disponible
    const photoWrap = document.getElementById("practice-sign-photo-wrap");
    if (photoWrap && window.getSignImagePath) {
      const imgPath = window.getSignImagePath(target.id);
      // Eliminar imagen anterior si existe
      photoWrap.querySelectorAll("img.sign-photo-img").forEach(el => el.remove());
      if (imgPath) {
        const img = document.createElement("img");
        img.src = imgPath;
        img.alt = "Seña " + target.id;
        img.className = "sign-photo-img";
        // Insertar antes del span de emoji
        const emojiSpan = photoWrap.querySelector("#target-sign-emoji");
        if (emojiSpan) {
          photoWrap.insertBefore(img, emojiSpan);
        } else {
          photoWrap.appendChild(img);
        }
        // Ocultar el texto de letra grande cuando hay foto
        const idBig = photoWrap.querySelector(".sign-id-big, #target-sign-id");
        if (idBig) idBig.style.opacity = "0";
      } else {
        const idBig = photoWrap.querySelector(".sign-id-big, #target-sign-id");
        if (idBig) idBig.style.opacity = "1";
      }
    }
  }

  _updateProgressUI() {
    const total = this.lessonSigns.length;
    const done  = this.currentIdx;
    const pct   = total > 0 ? (done / total) * 100 : 0;

    const fill = document.getElementById("lesson-bar-fill");
    if (fill) fill.style.width = `${pct}%`;

    const dotsContainer = document.getElementById("lesson-dots");
    if (!dotsContainer) return;
    dotsContainer.innerHTML = "";
    for (let i = 0; i < total; i++) {
      const dot = document.createElement("div");
      dot.className = `lesson-dot ${i < done ? "done" : i === done ? "current" : ""}`;
      dotsContainer.appendChild(dot);
    }
  }

  _showComplete() {
    const el = document.getElementById("practice-complete");
    if (el) {
      el.classList.remove("hidden");
      el.querySelector("#final-score").textContent = this.score;
    }
    ApiClient.saveProgress("SESSION", this.currentCountry, true);
  }
}

window.PracticeMode = PracticeMode;
