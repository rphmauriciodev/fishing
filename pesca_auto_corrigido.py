"""
Pesca automática — ETAPAS 2 e 3.
  Minigame 1 (barra): detecta a faixa cinza, PRESSIONA Enter, acompanha o indicador
                      verde e SOLTA Enter de forma antecipada.
  Minigame 2 (puxar): quando aparece o disco com o arco verde, faz toques de Enter
                      (hold/rest fixos) até o check, e grava telemetria do anel interno.

Fluxo:
  IDLE      -> procura a faixa cinza (não precisa do verde). Quando ela fica estável
               por alguns frames, congela a posição, calcula o alvo e pressiona Enter.
  HOLDING   -> acompanha o topo do verde, ajusta uma reta (posição x tempo) e prevê
               QUANDO ele chegará ao alvo. Solta Enter em  t_chegada - latência.
  RELEASED  -> mede onde o verde realmente parou, calcula o erro em pixels e
               (opcional) corrige a latência automaticamente (calibracao.json).
  WAIT_REEL -> espera o disco do minigame 2 aparecer (leva alguns segundos).
  REELING   -> toques de Enter em outra thread; termina no check, quando o disco some ou no limite de tempo.
  COOLDOWN  -> espera o minigame sumir antes de armar de novo.

Modo de teste (não aperta nada):  python pesca_auto.py --disco

Teclas: F6 = sair | F7 = armar/desarmar (desarmado só observa; solta o Enter se estiver segurando)
Dependências: pip install opencv-python mss numpy keyboard
Se o jogo ignorar as teclas, rode o terminal como Administrador.
"""

import json
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# CONFIGURAÇÃO — captura e cores (herdadas do detector validado)
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from pesca_config import load_and_sync_config, GuardarDetector

CONFIG = load_and_sync_config()
REGION = CONFIG["screen"]["bar_region"]
REF_PATH = os.path.join(BASE_DIR, "referencia.png")
CAL_FILE = os.path.join(BASE_DIR, "calibracao.json")

MARGIN_X = 3
ROW_FRAC = 0.5
GREEN_LO = (35, 80, 80)
GREEN_HI = (90, 255, 255)
GRAY_S_MAX = 45
GRAY_V_LO = 70
GRAY_V_HI = 215
GRAY_V_TOL = 45

# Referência de cor medida no recorte da faixa cinza.
# O valor informado era RGB (55, 57, 63); OpenCV usa BGR.
# Tolerância evita exigir que todos os pixels tenham exatamente a mesma cor.
TARGET_BGR = (63, 57, 55)
TARGET_TOL = 18
TARGET_ROW_FRAC = 0.50

MIN_GREEN_RUN = 3
MIN_GRAY_RUN = 2
WIN_ABOVE = 14
WIN_BELOW = 6

# ----------------------------------------------------------------------------
# CONFIGURAÇÃO — controle do Enter
# ----------------------------------------------------------------------------
KEY = "enter"
ARMED_AT_START = True        # F7 alterna
AUTO_PRESS = True            # False = você aperta Enter; o script só solta na hora certa

TARGET_OFFSET_PX = 3         # alvo = base_da_faixa - OFFSET. Um frame do jogo = ~4-5px, então mire 2 a 5 linhas dentro da faixa
LATENCY_MS = 50.0            # chute inicial (captura + entrada); a autocalibração refina
AUTO_CALIBRATE = True
CAL_GAIN = 0.5               # fração do erro corrigida por tentativa (0..1)
LATENCY_MIN_MS, LATENCY_MAX_MS = 0.0, 250.0
MAX_CAL_ERROR_PX = 30        # erros maiores que isso são ignorados na calibração

# Detecção da faixa antes de pressionar
BAND_STABLE_FRAMES = 5       # frames seguidos com a faixa estável
BAND_JITTER = 1              # variação máxima (px) entre esses frames
BAND_MERGE_GAP = 0           # 0 = mede a faixa como no detector validado (2 inflava a faixa em 2-3 linhas)
GREEN_START_Y = 190          # se o verde já está acima disso, o Enter já está sendo segurado

# Verificação de que a barra do minigame está na tela (evita "faixa" em paredes cinzas do jogo)
USE_BG_CHECK = True
BG_V_MAX = 70                # pixels mais escuros que isso na referência = fundo da barra
BG_TOL = 25                  # tolerância (por canal) para considerar "igual ao fundo"
BG_FRAC = 0.40               # fração mínima de pixels de fundo para aceitar a barra

# Previsão
FIT_POINTS = 6               # nº de mudanças de posição usadas na reta
MIN_FIT_POINTS = 4
USE_SPEED_PRIOR = True       # usa a mediana das velocidades anteriores (reduz ruído); precisa de 3 tentativas
PRIOR_TOLERANCE = 0.25       # se a velocidade medida diferir mais que isso da mediana, usa a medida
MIN_SPEED = 60.0             # px/s mínimo para aceitar a previsão
COMMIT_WINDOW = 0.025        # s: se a hora de soltar cai nesta janela, agenda e solta com espera precisa

# Segurança e ciclo
MAX_HOLD = 3.0               # s: solta o Enter à força
START_TIMEOUT = 1.0          # s: se o verde não aparecer/subir após pressionar, aborta
SETTLE_S = 0.30              # s após soltar para medir onde o verde parou
MIN_SETTLE_S = 0.05          # piso; o limite real é max(isto, latência atual)
COOLDOWN_MIN = 1.0
COOLDOWN_MAX = 4.0
GONE_S = 0.3                 # s sem faixa para considerar que o minigame acabou

# Janela
SHOW_PREVIEW = True
PREVIEW_INTERVAL = 0.10
SCALE = 3
SAVE_SHOTS = True            # salva imagem anotada de cada tentativa em ./capturas
SHOTS_DIR = os.path.join(BASE_DIR, "capturas")
SHOT_SCALE = 6
VERBOSE = False              # True: imprime cada mudança de posição durante a subida

# ----------------------------------------------------------------------------
# CONFIGURAÇÃO — ETAPA 3: minigame de puxar o peixe (disco + arco verde)
# ----------------------------------------------------------------------------
REEL_ENABLED = True
REEL_TAP_HOLD = 0.108        # s com a tecla pressionada em cada toque (seu script)
REEL_TAP_REST = 0.145        # s solta entre toques (seu script)
INPUT_BACKEND = "keyboard"   # "keyboard" (já validado no minigame 1) ou "pydirectinput" (o que seu script de toques usa)
WAIT_REEL_TIMEOUT = 25.0     # s esperando o disco aparecer depois do minigame 1
REEL_MAX_S = 120.0           # s: limite de segurança do minigame 2
REEL_GONE_S = 1.0            # s sem disco => minigame 2 terminou (sucesso ou falha)
CHECK_FRAMES = 3             # frames seguidos vendo o check para dar como concluído
SAVE_TELEMETRY = True        # grava CSV do anel interno em ./telemetria (base do caminho B)
TELEMETRY_DIR = os.path.join(BASE_DIR, "telemetria")

# Geometria sincronizada com a resolução da tela
DISC_CX = CONFIG["screen"]["disc_cx"]
DISC_CY = CONFIG["screen"]["disc_cy"]
DISC_R = float(CONFIG["screen"]["disc_r"])
DISC_REGION = CONFIG["screen"]["disc_region"]
GUARDAR_REGION = CONFIG["screen"]["guardar_region"]

DISC_STABLE_FRAMES = 6       # frames seguidos com o disco para aceitar
DISC_DARK_V = 75             # V máximo da borda escura do disco
DISC_RIM_DARK_FRAC = 0.60    # fração da coroa que precisa ser escura
RING_V_MIN, RING_S_MIN, RING_S_MAX = 75, 10, 220   # anel interno (adaptado para dia/noite e sombras)
RING_H_MAX, RING_H_MIN2 = 35, 160
RING_MIN_PX = 25
ARC_H = (35, 90)             # arco verde externo
ARC_S_MIN, ARC_V_MIN = 40, 80
ARC_MIN_DEG = 15
CHECK_MIN_PX = 80            # pixels verdes no centro = check de sucesso (WhatsApp Image 22.10.27)


def _union(a, b):
    left, top = min(a["left"], b["left"]), min(a["top"], b["top"])
    right = max(a["left"] + a["width"], b["left"] + b["width"])
    bottom = max(a["top"] + a["height"], b["top"] + b["height"])
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


# Uma única captura cobre a barra, o disco e a região de guardar peixe.
CAPTURE = _union(_union(REGION, DISC_REGION), GUARDAR_REGION)
BAR_OFF = (REGION["left"] - CAPTURE["left"], REGION["top"] - CAPTURE["top"])
DISC_OFF = (DISC_REGION["left"] - CAPTURE["left"], DISC_REGION["top"] - CAPTURE["top"])
GUARDAR_OFF = (GUARDAR_REGION["left"] - CAPTURE["left"], GUARDAR_REGION["top"] - CAPTURE["top"])


def split_capture(cap):
    bx, by = BAR_OFF
    dx, dy = DISC_OFF
    gx, gy = GUARDAR_OFF
    bar = np.ascontiguousarray(cap[by:by + REGION["height"], bx:bx + REGION["width"]])
    disc = np.ascontiguousarray(cap[dy:dy + DISC_REGION["height"], dx:dx + DISC_REGION["width"]])
    guardar = np.ascontiguousarray(cap[gy:gy + GUARDAR_REGION["height"], gx:gx + GUARDAR_REGION["width"]])
    return bar, disc, guardar


# ----------------------------------------------------------------------------
# SEGMENTAÇÃO E GEOMETRIA (reaproveitadas)
# ----------------------------------------------------------------------------
def make_masks(bgr, gray_lo, gray_hi):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    _, s, v = cv2.split(hsv)
    gray = ((s <= GRAY_S_MAX) & (v >= gray_lo) & (v <= gray_hi)).astype(np.uint8) * 255
    gray[green > 0] = 0
    return green, gray


def row_flags(mask):
    inner = mask[:, MARGIN_X:mask.shape[1] - MARGIN_X]
    return (inner > 0).mean(axis=1) >= ROW_FRAC


def runs(flags):
    out, start = [], None
    for y, f in enumerate(flags):
        if f and start is None:
            start = y
        elif not f and start is not None:
            out.append((start, y - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def analyze(bgr, gray_lo, gray_hi, band_lock):
    """Usada na referência: topo do verde e faixa cinza logo acima dele."""
    green, gray = make_masks(bgr, gray_lo, gray_hi)
    info = {"tip": None, "band": None, "gap": None, "locked": False,
            "green_mask": green, "gray_mask": gray}
    g_runs = [r for r in runs(row_flags(green)) if r[1] - r[0] + 1 >= MIN_GREEN_RUN]
    if not g_runs:
        return info
    g = max(g_runs, key=lambda r: r[1] - r[0])
    tip = g[0]
    info["tip"] = tip
    b_runs = [r for r in runs(row_flags(gray))
              if r[1] - r[0] + 1 >= MIN_GRAY_RUN and r[1] < tip]
    if b_runs:
        band = max(b_runs, key=lambda r: r[1])
        info["band"] = band
        info["gap"] = tip - band[1] - 1
    elif band_lock is not None:
        info["band"] = band_lock
        info["gap"] = tip - band_lock[1] - 1
        info["locked"] = True
    return info


def class_map(green, gray):
    cls = np.zeros(green.shape, np.uint8)
    cls[gray > 0] = 2
    cls[green > 0] = 1
    return cls


def crop_window(img, tip):
    padded = cv2.copyMakeBorder(img, WIN_ABOVE, WIN_BELOW + 1, 0, 0, cv2.BORDER_REPLICATE)
    return padded[tip:tip + WIN_ABOVE + WIN_BELOW + 1, MARGIN_X:-MARGIN_X]


def window_scores(cur_bgr, cur_cls, tip, ref_bgr_w, ref_cls_w):
    cw = crop_window(cur_cls, tip)
    rel = (cw > 0) | (ref_cls_w > 0)
    if not rel.any():
        return 0.0, 0.0
    s_cls = float((cw == ref_cls_w)[rel].mean())
    diff = np.abs(crop_window(cur_bgr, tip).astype(np.int16) - ref_bgr_w.astype(np.int16)).mean(axis=2)
    s_col = float(1.0 - diff[rel].mean() / 255.0)
    return s_cls, s_col


# ----------------------------------------------------------------------------
# NOVO: detecção da faixa SEM depender do verde
# ----------------------------------------------------------------------------
def find_tip(green):
    g_runs = [r for r in runs(row_flags(green)) if r[1] - r[0] + 1 >= MIN_GREEN_RUN]
    if not g_runs:
        return None
    return max(g_runs, key=lambda r: r[1] - r[0])[0]


def find_band(bgr, gray, h_ref):
    """Detecta a faixa cinza por cor + continuidade horizontal + espessura.

    Retorna (topo, base) ou None. Não fixa a coordenada vertical do alvo.
    """
    # A imagem do OpenCV está em BGR. Aceita pixels próximos da cor de referência.
    pixels = bgr.astype(np.int16)
    alvo = np.array(TARGET_BGR, dtype=np.int16)
    color_match = np.all(np.abs(pixels - alvo) <= TARGET_TOL, axis=2)

    # A cor deve ocupar uma fração mínima da largura interna da barra.
    inner_color = color_match[:, MARGIN_X:bgr.shape[1] - MARGIN_X]
    color_rows = inner_color.mean(axis=1) >= TARGET_ROW_FRAC

    # Exige também que a linha seja classificada como cinza pela máscara existente.
    flags = color_rows & row_flags(gray)

    merged = []
    for r in runs(flags):
        if merged and r[0] - merged[-1][1] - 1 <= BAND_MERGE_GAP:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)

    h_min, h_max = max(2, h_ref - 4), h_ref + 10
    cands = [r for r in merged if h_min <= r[1] - r[0] + 1 <= h_max]
    if not cands:
        return None

    # Escolhe a candidata com espessura mais próxima da faixa de referência.
    # Em empate, prioriza a linha com maior cobertura da cor-alvo.
    def rank(r):
        height_error = abs((r[1] - r[0] + 1) - h_ref)
        coverage = float(color_rows[r[0]:r[1] + 1].mean())
        return (height_error, -coverage)

    return min(cands, key=rank)


# ----------------------------------------------------------------------------
# REFERÊNCIA E CALIBRAÇÃO
# ----------------------------------------------------------------------------
def load_reference(path=REF_PATH):
    ref = cv2.imread(path, cv2.IMREAD_COLOR)
    if ref is None:
        print(f"[INFO] '{path}' não encontrada. Operando exclusivamente em modo Minigame Circular (Etapa 3).")
        return None
    if ref.shape[1] != REGION["width"]:
        print(f"[AVISO] Largura da referência ({ref.shape[1]}) != região ({REGION['width']}). Ajustando.")
        ref = cv2.resize(ref, (REGION["width"], ref.shape[0]), interpolation=cv2.INTER_NEAREST)

    lo, hi = GRAY_V_LO, GRAY_V_HI
    info = analyze(ref, lo, hi, None)
    if info["band"] is not None:
        t, b = info["band"]
        hsv = cv2.cvtColor(ref[t:b + 1, MARGIN_X:-MARGIN_X], cv2.COLOR_BGR2HSV)
        vmed = int(np.median(hsv[:, :, 2]))
        lo, hi = max(30, vmed - GRAY_V_TOL), min(255, vmed + GRAY_V_TOL)
        info = analyze(ref, lo, hi, None)
        print(f"[REF] Cinza calibrado: V mediano={vmed} -> faixa V [{lo}, {hi}]")
    if info["tip"] is None or info["band"] is None:
        sys.exit("[ERRO] Referência sem verde/faixa detectáveis. Rode o detector da etapa 1 e ajuste as cores.")

    t, b = info["band"]
    h_ref = b - t + 1
    print(f"[REF] topo do verde Y={info['tip']} | faixa Y={t}-{b} (altura {h_ref}) | gap={info['gap']}")

    inner = ref[:, MARGIN_X:-MARGIN_X]
    hsv_in = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    g_in = info["green_mask"][:, MARGIN_X:-MARGIN_X] > 0
    gr_in = info["gray_mask"][:, MARGIN_X:-MARGIN_X] > 0
    dark = (hsv_in[:, :, 2] < BG_V_MAX) & ~g_in & ~gr_in
    bg = None
    if dark.sum() >= 50:
        bg = np.round(np.median(inner[dark], axis=0)).astype(np.int16)
        print(f"[REF] Cor do fundo da barra (BGR): {bg.tolist()}")
    else:
        print("[AVISO] Não consegui estimar o fundo da barra; verificação de fundo desativada.")

    cls = class_map(info["green_mask"], info["gray_mask"])
    return {
        "gray_lo": lo, "gray_hi": hi, "h": h_ref, "bg": bg,
        "bgr_w": crop_window(ref, info["tip"]),
        "cls_w": crop_window(cls, info["tip"]),
    }


def load_latency():
    try:
        with open(CAL_FILE, "r", encoding="utf-8") as f:
            v = float(json.load(f)["latency_ms"])
        print(f"[CAL] Latência carregada de {CAL_FILE}: {v:.1f} ms")
        return v
    except Exception:
        return LATENCY_MS


def save_latency(v):
    try:
        with open(CAL_FILE, "w", encoding="utf-8") as f:
            json.dump({"latency_ms": round(v, 2)}, f)
    except Exception as e:
        print(f"[AVISO] Não consegui salvar {CAL_FILE}: {e}")


# ----------------------------------------------------------------------------
# MINIGAME 2: leitura do disco (geometria fixa) e ciclo de toques
# ----------------------------------------------------------------------------
_DISC_MAPS = {}


def _disc_maps(h, w):
    key = (h, w)
    if key not in _DISC_MAPS:
        cx = DISC_CX - DISC_REGION["left"]
        cy = DISC_CY - DISC_REGION["top"]
        yy, xx = np.mgrid[:h, :w]
        d = np.hypot(xx - cx, yy - cy).astype(np.float32)
        ang = ((np.degrees(np.arctan2(yy - cy, xx - cx)) + 360.0) % 360.0).astype(np.float32)
        rim = (d >= DISC_R - 18) & (d <= DISC_R - 4)        # coroa escura dentro do disco (o anel fino ocupa pouco dela)
        inner = (d > 14) & (d <= DISC_R - 1)                # onde fica o anel interno
        arc = (d > DISC_R + 2) & (d < DISC_R + 14)          # faixa do arco verde
        core = d < DISC_R * 0.6                             # centro (check)
        _DISC_MAPS[key] = (d, ang, rim, inner, arc, core)
    return _DISC_MAPS[key]


def analyze_disc(img):
    """Lê o disco do minigame 2. ring_r (px) é o raio do anel interno = tensão."""
    d, ang, rim_m, inner_m, arc_m, core_m = _disc_maps(img.shape[0], img.shape[1])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    rim_dark = float((V[rim_m] < DISC_DARK_V).mean())
    ring = (inner_m & (V >= RING_V_MIN) & (S >= RING_S_MIN) & (S <= RING_S_MAX)
            & ((H <= RING_H_MAX) | (H >= RING_H_MIN2)))
    n_ring = int(ring.sum())
    ring_r = float(np.median(d[ring])) if n_ring >= RING_MIN_PX else None
    green = (H >= ARC_H[0]) & (H <= ARC_H[1]) & (S >= ARC_S_MIN) & (V >= ARC_V_MIN)
    arc = arc_m & green
    arc_deg = 0
    if arc.any():
        arc_deg = len(np.unique((ang[arc] // 5).astype(np.int32) % 72)) * 5
    check_n = int((core_m & green).sum())
    present = rim_dark >= DISC_RIM_DARK_FRAC and (n_ring >= RING_MIN_PX or arc_deg >= ARC_MIN_DEG or check_n >= CHECK_MIN_PX)
    return {"present": present, "rim_dark": rim_dark, "ring_r": ring_r, "n_ring": n_ring,
            "arc_deg": arc_deg, "check_n": check_n}


class Tapper(threading.Thread):
    """Ciclo de toques com controle dinâmico de cadência para estabilizar o anel."""

    def __init__(self, press, release, hold, rest):
        super().__init__(daemon=True)
        self.press, self.release, self.hold, self.rest = press, release, hold, rest
        self.active = threading.Event()
        self.quit = threading.Event()
        self.emergency_brake = threading.Event()
        self.down = False
        self.taps = 0
        self._lock = threading.Lock()

    def update_cadence(self, hold, rest, brake=False):
        with self._lock:
            self.hold = hold
            self.rest = rest
            if brake:
                self.emergency_brake.set()
            else:
                self.emergency_brake.clear()

    def start_taps(self):
        self.taps = 0
        self.emergency_brake.clear()
        self.active.set()

    def stop_taps(self):
        self.active.clear()
        self.emergency_brake.clear()
        self.release()
        self.down = False

    def _wait(self, dur):
        end = time.perf_counter() + dur
        while time.perf_counter() < end and self.active.is_set() and not self.quit.is_set():
            time.sleep(0.002)

    def run(self):
        while not self.quit.is_set():
            if not self.active.is_set() or self.emergency_brake.is_set():
                if self.down:
                    self.release()
                    self.down = False
                time.sleep(0.003)
                continue
            with self._lock:
                h, r = self.hold, self.rest
            try:
                self.press()
                self.down = True
                self.taps += 1
                self._wait(h)
            finally:
                self.release()
                self.down = False
            if self.active.is_set() and not self.quit.is_set() and not self.emergency_brake.is_set():
                self._wait(r)


def make_input(backend):
    if backend == "pydirectinput":
        import pydirectinput
        pydirectinput.PAUSE = 0
        def press_e():
            pydirectinput.keyDown("e")
            time.sleep(0.100)
            pydirectinput.keyUp("e")
        return (lambda: pydirectinput.keyDown(KEY)), (lambda: pydirectinput.keyUp(KEY)), press_e
    import keyboard
    def press_e():
        keyboard.press("e")
        time.sleep(0.100)
        keyboard.release("e")
    return (lambda: keyboard.press(KEY)), (lambda: keyboard.release(KEY)), press_e


# ----------------------------------------------------------------------------
# CONTROLADOR (independente de mss/keyboard para poder ser testado)
# ----------------------------------------------------------------------------
class Bot:
    def __init__(self, ref, clock, press, release, wait_until, log=print, tapper=None, press_e=None, guardar_detector=None):
        self.ref, self.clock = ref, clock
        self.tapper = tapper
        self.press_e = press_e
        self.guardar_detector = guardar_detector
        self.reel_on = REEL_ENABLED and tapper is not None
        self.disc_info = None
        self.disc_hist = deque(maxlen=DISC_STABLE_FRAMES)
        self.reel_rows = []
        self.press, self.release_key, self.wait_until, self.log = press, release, wait_until, log
        self.L = load_latency()
        self.armed = ARMED_AT_START
        self.tip = None
        self.band = None
        self.y_target = None
        self.v = 0.0
        self.diag = ""
        self.prev_tcap = None
        self.frame_release = self.tip_release = None
        self.v_hist = []
        self.v_fit = None
        self.last_tip_frame = self.last_post_frame = None
        self.t_wait_guardar = 0.0
        self.state = "IDLE"
        self._to_idle()

    # ---- utilidades -------------------------------------------------------
    def _to_idle(self):
        self.state = "IDLE"
        self.band_hist = deque(maxlen=BAND_STABLE_FRAMES)
        self.events = deque(maxlen=FIT_POINTS)
        self.last_y = None
        self.post = []
        self.gone_since = None
        self.band = None
        self.y_target = None
        self.disc_hist.clear()

    def _release_now(self):
        try:
            self.release_key()
        except Exception as e:
            self.log(f"[AVISO] Falha ao soltar tecla: {e}")

    def bar_check(self, frame):
        if self.ref is None:
            return False, 0.0
        bg = self.ref["bg"]
        if not USE_BG_CHECK or bg is None:
            return True, 1.0
        inner = frame[:, MARGIN_X:-MARGIN_X].astype(np.int16)
        frac = float((np.abs(inner - bg).max(axis=2) <= BG_TOL).mean())
        return frac >= BG_FRAC, frac

    def set_armed(self, value):
        self.armed = value
        if not value and self.state == "HOLDING":
            self._release_now()
            self.log("[DESARMADO] Enter solto.")
            self._to_idle()
        elif not value and self.state in ("WAIT_REEL", "REELING", "WAIT_GUARDAR", "RECASTING"):
            if self.tapper is not None:
                self.tapper.stop_taps()
            self._release_now()
            self.log("[DESARMADO] Toques interrompidos.")
            self._to_idle()

    # ---- passo principal --------------------------------------------------
    def step(self, frame, t_cap, disc=None, guardar_frame=None):
        st = self.state
        self.disc_info = None

        if st == "WAIT_GUARDAR":
            self.tip = None
            self._wait_guardar(guardar_frame, t_cap)
            self.prev_tcap = t_cap
            return

        if st == "RECASTING":
            self.tip = None
            self._recasting(t_cap)
            self.prev_tcap = t_cap
            return

        if self.reel_on and disc is not None and st in ("IDLE", "WAIT_REEL", "REELING"):
            self.disc_info = analyze_disc(disc)
        if st in ("WAIT_REEL", "REELING"):
            self.tip = None
            if st == "WAIT_REEL":
                self._wait_reel(t_cap)
            else:
                self._reeling(t_cap)
            self.prev_tcap = t_cap
            return
        green, gray = make_masks(frame, self.ref["gray_lo"], self.ref["gray_hi"])
        self.tip = find_tip(green)
        if st == "IDLE":
            if self._idle_disc(t_cap):
                self.prev_tcap = t_cap
                return
            self._idle(frame, gray, t_cap)
        elif st == "HOLDING":
            self._holding(frame, t_cap)
        elif st == "RELEASED":
            self._released(frame, t_cap)
        elif st == "COOLDOWN":
            self._cooldown(frame, gray, t_cap)
        self.prev_tcap = t_cap

    def _wait_guardar(self, guardar_frame, t_cap):
        if self.guardar_detector is None or guardar_frame is None:
            if self.press_e:
                self.press_e()
            self._enter_cooldown()
            return

        found, score, loc = self.guardar_detector.detect(guardar_frame)
        elapsed = t_cap - self.t_wait_guardar
        self.diag = f"guardar={'SIM' if found else 'aguardando'} score={score:.2f} t={elapsed:.1f}s"
        if found:
            self.log(f"[GUARDAR] Prompt '[E] Guardar Peixe' detectado na tela (score={score:.2f})! Pressionando E...")
            if self.press_e:
                self.press_e()
            self._enter_cooldown()
        elif elapsed >= 3.5:
            self.log("[GUARDAR] Tempo limite aguardando prompt: pressionando 'E' por seguranca...")
            if self.press_e:
                self.press_e()
            self._enter_cooldown()

    # ---- IDLE -------------------------------------------------------------
    def _idle(self, frame, gray, t_cap):
        ok, frac = self.bar_check(frame)
        cand = find_band(frame, gray, self.ref["h"]) if ok else None
        self.band_hist.append(cand)
        n_ok = sum(c is not None for c in self.band_hist)
        self.diag = f"barra={ok} fundo={frac:.2f} banda={cand} estavel={n_ok}/{BAND_STABLE_FRAMES}"
        h = self.band_hist
        if len(h) < BAND_STABLE_FRAMES or any(c is None for c in h):
            return
        tops, bots = [c[0] for c in h], [c[1] for c in h]
        if max(tops) - min(tops) > BAND_JITTER or max(bots) - min(bots) > BAND_JITTER:
            return
        if not self.armed:
            return

        self.band = (int(np.median(tops)), int(np.median(bots)))
        self.y_target = self.band[1] - TARGET_OFFSET_PX
        self.events.clear()
        self.last_y = self.tip  # verde já visível e parado = apenas referência, não entra na reta
        external = self.tip is not None and self.tip < GREEN_START_Y
        if external:
            self.log(f"[INICIO] Verde já subindo (topo={self.tip}). Faixa={self.band} alvo={self.y_target}")
        elif AUTO_PRESS:
            self.press()
            self.log(f"[PRESSIONOU] faixa={self.band} alvo={self.y_target} latencia={self.L:.1f}ms")
        else:
            self.log(f"[AGUARDANDO] faixa={self.band} alvo={self.y_target} (aperte Enter você mesmo)")
        self.t_press = self.clock()
        self.start_timeout = START_TIMEOUT if AUTO_PRESS else 15.0
        self.state = "HOLDING"

    # ---- HOLDING ----------------------------------------------------------
    def _track(self, t_cap):
        tip = self.tip
        if tip is not None and (self.last_y is None or tip != self.last_y):
            if self.prev_tcap is not None:
                t_ev = 0.5 * (self.prev_tcap + t_cap)  # a mudança ocorreu entre as duas capturas
            else:
                t_ev = t_cap
            if self.last_y is not None and tip > self.last_y + 3:
                self.events.clear()  # salto para trás = ruído/nova tentativa
            self.events.append((t_ev, tip))
            self.last_y = tip
            if VERBOSE:
                self.log(f"   t={t_ev - self.t_press:6.3f}s topo={tip}")

    def _predict(self):
        """Retorna (t_chegada, v_usada, v_medida) ou None."""
        ev = list(self.events)
        n = len(ev)
        v_prior = float(np.median(self.v_hist)) if (USE_SPEED_PRIOR and len(self.v_hist) >= 3) else None
        fit = None
        if n >= MIN_FIT_POINTS and ev[-1][0] - ev[0][0] >= 0.03:
            t_ref = ev[-1][0]
            t = np.array([e[0] - t_ref for e in ev])
            y = np.array([e[1] for e in ev], dtype=float)
            slope, a = np.polyfit(t, y, 1)  # y = a + slope * t
            if -slope >= MIN_SPEED:
                fit = (t_ref + (self.y_target - a) / slope, -slope)
        if v_prior and n >= 2 and v_prior >= MIN_SPEED:
            if fit is None or abs(fit[1] - v_prior) <= PRIOR_TOLERANCE * v_prior:
                # velocidade fixa = mediana anterior; só a posição é ajustada com todos os pontos
                t_hit = float(np.mean([t_i + (y_i - self.y_target) / v_prior for t_i, y_i in ev]))
                return t_hit, v_prior, (fit[1] if fit else None)
        if fit is not None:
            return fit[0], fit[1], fit[1]
        return None

    def _holding(self, frame, t_cap):
        self._track(t_cap)
        if self.tip is not None:
            self.last_tip_frame = frame
        held = self.clock() - self.t_press
        if held > MAX_HOLD or (not self.events and held > self.start_timeout):
            self.log(f"[ABORTAR] sem progresso (segurou {held:.2f}s, eventos={len(self.events)}). Soltando.")
            self._release_now()
            self._enter_cooldown()
            return
        pred = self._predict()
        if pred is None:
            return
        t_hit, v, v_fit = pred
        t_rel = t_hit - self.L / 1000.0
        if t_rel - self.clock() <= COMMIT_WINDOW:
            self.wait_until(t_rel)
            self.t_release = self.clock()
            self.frame_release, self.tip_release = frame, self.tip
            self._release_now()
            self.v = v
            self.v_fit = v_fit
            self.post = []
            self.last_post_frame = None
            self.state = "RELEASED"
            self.log(f"[SOLTOU] v={v:.0f}px/s{' (mediana)' if v_fit is None or abs(v - v_fit) > 1e-6 else ''} | previsto chegar ao alvo {self.y_target} em "
                     f"{(t_hit - self.t_press):.3f}s | soltou em {(self.t_release - self.t_press):.3f}s "
                     f"| topo visto={self.last_y}")

    # ---- RELEASED ---------------------------------------------------------
    def _min_settle(self):
        # a partir daqui a tela já reflete o efeito da soltura
        return max(MIN_SETTLE_S, self.L / 1000.0)

    def _released(self, frame, t_cap):
        if self.tip is not None:
            self.post.append((t_cap, self.tip))
            self.last_post_frame = frame
        t_end = self.t_release + SETTLE_S
        lost_late = (self.tip is None and self.post and self.post[-1][0] >= self.t_release + self._min_settle())
        if t_cap >= t_end or lost_late:
            self._finalize()

    def _annotate(self, frame, title, tip):
        k = SHOT_SCALE
        big = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
        w = big.shape[1]

        def tick(y, color):
            yy = int(y * k)
            cv2.line(big, (0, yy), (3 * k - 2, yy), color, 1)
            cv2.line(big, (w - 3 * k + 2, yy), (w, yy), color, 1)

        if self.band:
            tick(self.band[0], (0, 255, 255))        # topo da faixa (amarelo)
            tick(self.band[1] + 1, (0, 255, 255))    # base da faixa (amarelo)
        if self.y_target is not None:
            tick(self.y_target, (255, 0, 255))       # alvo (magenta)
        if tip is not None:
            tick(tip, (0, 255, 0))                   # topo do verde (verde)
        banner = np.zeros((34, w, 3), np.uint8)
        cv2.putText(banner, title, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([banner, big])

    def _save_shot(self, y_final, info):
        if not SAVE_SHOTS:
            return
        try:
            os.makedirs(SHOTS_DIR, exist_ok=True)
            n = len([f for f in os.listdir(SHOTS_DIR) if f.startswith("tent_")]) + 1
            parts = []
            if getattr(self, "frame_release", None) is not None:
                parts.append(self._annotate(self.frame_release, "ANTES (soltou)", self.tip_release))
            last = self.last_post_frame if self.last_post_frame is not None else getattr(self, "last_tip_frame", None)
            if last is not None:
                parts.append(self._annotate(last, f"FINAL {info}", y_final))
            if not parts:
                return
            sep = np.full((parts[0].shape[0], 8, 3), 90, np.uint8)
            img = parts[0] if len(parts) == 1 else np.hstack([parts[0], sep, parts[1]])
            path = os.path.join(SHOTS_DIR, f"tent_{n:03d}.png")
            cv2.imwrite(path, img)
            self.log(f"[FOTO] {path}  (amarelo=faixa, magenta=alvo, verde=topo do verde)")
        except Exception as e:
            self.log(f"[AVISO] Não consegui salvar a foto: {e}")

    def _finalize(self):
        y_final = None
        if self.post and self.post[-1][0] >= self.t_release + self._min_settle():
            y_final = self.post[-1][1]
        if y_final is not None:
            e = self.y_target - y_final  # >0: passou do alvo; <0: parou antes
            side = "passou" if e > 0 else ("ficou antes" if e < 0 else "exato")
            score = float("nan")
            try:
                g, gr = make_masks(self.last_post_frame, self.ref["gray_lo"], self.ref["gray_hi"])
                score, _ = window_scores(self.last_post_frame, class_map(g, gr), y_final,
                                         self.ref["bgr_w"], self.ref["cls_w"])
            except Exception:
                pass
            top, bot = self.band
            if y_final > bot:
                onde = f"FORA: {y_final - bot}px ABAIXO da base da faixa"
            elif y_final < top:
                onde = f"FORA: {top - y_final}px ACIMA do topo da faixa"
            else:
                onde = f"DENTRO: {bot - y_final}px acima da base da faixa"
            msg = (f"[RESULTADO] alvo={self.y_target} final={y_final} erro={e:+d}px ({side}) | "
                   f"{onde} {self.band} | v={self.v:.0f}px/s | score_cls={score:.3f}")
            v_ok = True
            vm = self.v_fit
            if vm is not None and len(self.v_hist) >= 3:
                med = float(np.median(self.v_hist))
                v_ok = abs(vm - med) <= 0.25 * med
                if not v_ok:
                    msg += f" | (v medida={vm:.0f} destoa da mediana {med:.0f}: sem calibração)"
            if vm is not None and v_ok:
                self.v_hist = (self.v_hist + [vm])[-8:]
            if AUTO_CALIBRATE and v_ok and abs(e) <= MAX_CAL_ERROR_PX and self.v >= MIN_SPEED:
                dL = 1000.0 * e / self.v
                new = min(LATENCY_MAX_MS, max(LATENCY_MIN_MS, self.L + CAL_GAIN * dL))
                msg += f" | latencia {self.L:.1f} -> {new:.1f} ms"
                self.L = new
                save_latency(new)
            self.log(msg)
            self._save_shot(y_final, f"erro={e:+d}px")
        else:
            self.log("[RESULTADO] sem leitura final válida (o minigame sumiu antes da tela refletir a soltura). Sem calibração.")
            self._save_shot(None, "sem leitura")
        self._after_first()

    # ---- MINIGAME 2 -------------------------------------------------------
    def _disc_ready(self):
        """Disco visível e AINDA sem o check (evita reiniciar com o check na tela)."""
        i = self.disc_info
        return bool(i and i["present"] and i["check_n"] < CHECK_MIN_PX)

    def _disc_stable(self):
        self.disc_hist.append(self._disc_ready())
        return len(self.disc_hist) == self.disc_hist.maxlen and all(self.disc_hist)

    def _idle_disc(self, t_cap):
        """Se o disco aparecer com o script ocioso (ex.: iniciou no meio da pesca), já assume o minigame 2."""
        if not self.reel_on:
            return False
        if self._disc_stable() and self.armed:
            self._start_reel(t_cap)
            return True
        return False

    def _after_first(self):
        if self.reel_on and self.armed:
            self.state = "WAIT_REEL"
            self.t_wait = self.clock()
            self.disc_hist.clear()
            self.log(f"[AGUARDANDO PUXAR] esperando o disco do minigame 2 (até {WAIT_REEL_TIMEOUT:.0f}s)...")
        else:
            self._enter_cooldown()

    def _wait_reel(self, t_cap):
        i = self.disc_info
        self.diag = (f"disco={'sim' if i and i['present'] else 'nao'} arco={i['arc_deg'] if i else '-'} "
                     f"anel={('%.1f' % i['ring_r']) if i and i['ring_r'] else '-'}")
        if self._disc_stable():
            self._start_reel(t_cap)
        elif self.clock() - self.t_wait > WAIT_REEL_TIMEOUT:
            self.log("[AGUARDANDO PUXAR] tempo esgotado sem ver o disco. Voltando a procurar a barra.")
            self._enter_cooldown()

    def _start_reel(self, t_cap):
        self.state = "REELING"
        self.t_reel = t_cap
        self.reel_rows = []
        self.gone_since = None
        self.check_cnt = 0
        self.tapper.start_taps()
        self.log(f"[PUXANDO] toques: {REEL_TAP_HOLD * 1000:.0f}ms pressionado / {REEL_TAP_REST * 1000:.0f}ms solto")

    def _reeling(self, t_cap):
        i = self.disc_info
        present = bool(i and i["present"])
        t_rel = t_cap - self.t_reel
        self.diag = (f"t={t_rel:5.1f}s arco={i['arc_deg'] if i else '-'} "
                     f"anel={('%.1f' % i['ring_r']) if i and i['ring_r'] else '-'} toques={self.tapper.taps}")
        if present:
            self.reel_rows.append((t_rel, int(self.tapper.down), self.tapper.taps, i["arc_deg"],
                                   i["ring_r"], i["check_n"], i["rim_dark"]))
            self.gone_since = None

            # Controle dinâmico adaptativo de cadência para manter o anel estável:
            if i["ring_r"] is not None:
                target_r = DISC_R * 0.50
                err = i["ring_r"] - target_r
                delta_rest = 0.0055 * err
                new_rest = float(np.clip(REEL_TAP_REST + delta_rest, 0.050, 0.320))
                new_hold = float(np.clip(REEL_TAP_HOLD - 0.0010 * err, 0.080, 0.125))
                brake = i["ring_r"] >= (DISC_R * 0.88)
                self.tapper.update_cadence(new_hold, new_rest, brake)
        elif self.gone_since is None:
            self.gone_since = t_cap
        if present and i["check_n"] >= CHECK_MIN_PX:
            self.check_cnt += 1
        else:
            self.check_cnt = 0
        if self.check_cnt >= CHECK_FRAMES:
            self._end_reel("SUCESSO (check)")
        elif self.gone_since is not None and t_cap - self.gone_since >= REEL_GONE_S:
            self._end_reel("disco sumiu (falha ou fim)")
        elif t_rel > REEL_MAX_S:
            self._end_reel("TEMPO LIMITE")

    def _end_reel(self, reason):
        taps = self.tapper.taps
        self.tapper.stop_taps()
        self._release_now()
        rows = self.reel_rows
        dur = rows[-1][0] if rows else 0.0
        rr = np.array([r[4] for r in rows if r[4] is not None], dtype=float)
        stats = ""
        if len(rr):
            stats = (f" | anel: mediana={np.median(rr):.1f} p95={np.percentile(rr, 95):.1f} max={rr.max():.1f}px "
                     f"({rr.max() / DISC_R * 100:.0f}% do raio do disco)")
        arc = max((r[3] for r in rows), default=0)
        self.log(f"[FIM PUXAR] {reason} | {dur:.1f}s | {taps} toques | arco maximo={arc}graus{stats}")
        if SAVE_TELEMETRY and rows:
            try:
                os.makedirs(TELEMETRY_DIR, exist_ok=True)
                n = len([f for f in os.listdir(TELEMETRY_DIR) if f.startswith("reel_")]) + 1
                path = os.path.join(TELEMETRY_DIR, f"reel_{n:03d}.csv")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("t_s,tecla,toques,arco_graus,raio_anel,check_px,borda_escura\n")
                    for r in rows:
                        f.write(f"{r[0]:.4f},{r[1]},{r[2]},{r[3]},{'' if r[4] is None else round(r[4], 2)},{r[5]},{r[6]:.2f}\n")
                self.log(f"[TELEMETRIA] {path}")
            except Exception as e:
                self.log(f"[AVISO] Não consegui salvar a telemetria: {e}")
        if "SUCESSO" in reason:
            self.state = "WAIT_GUARDAR"
            self.t_wait_guardar = self.clock()
            self.log("[FIM PUXAR] Sucesso! Aguardando o prompt '[E] Guardar Peixe' surgir na tela...")
        else:
            self._enter_cooldown()

    # ---- COOLDOWN ---------------------------------------------------------
    def _enter_cooldown(self):
        self.state = "COOLDOWN"
        self.t_cd = self.clock()
        self.gone_since = None

    def _cooldown(self, frame, gray, t_cap):
        since = t_cap - self.t_cd
        if since < COOLDOWN_MIN:
            return
        ok, _ = self.bar_check(frame)
        cand = find_band(frame, gray, self.ref["h"]) if ok else None
        if cand is None:
            if self.gone_since is None:
                self.gone_since = t_cap
            elif t_cap - self.gone_since >= GONE_S:
                self._finish_cooldown(t_cap)
                return
        else:
            self.gone_since = None
        if since >= COOLDOWN_MAX:
            self._finish_cooldown(t_cap)

    def _finish_cooldown(self, t_cap):
        if CONFIG["keys"].get("auto_recast", False) and self.press_e is not None:
            self.state = "RECASTING"
            self.t_recast = t_cap
            self.log("[AUTO-RECAST] Aguardando fim da animação para arremessar a vara novamente...")
        else:
            self._to_idle()

    def _recasting(self, t_cap):
        delay = CONFIG["keys"].get("recast_delay", 2.0)
        elapsed = t_cap - self.t_recast
        self.diag = f"recast em {max(0.0, delay - elapsed):.1f}s"
        if elapsed >= delay:
            self.log("[AUTO-RECAST] Pressionando 'E' para arremessar a vara na água...")
            if self.press_e:
                self.press_e()
            self._to_idle()


# ----------------------------------------------------------------------------
# JANELA
# ----------------------------------------------------------------------------
def render(bot, frame, fps):
    h, w = frame.shape[:2]
    big = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    if bot.band:
        cv2.rectangle(big, (0, bot.band[0] * SCALE), (w * SCALE - 1, (bot.band[1] + 1) * SCALE), (0, 255, 255), 1)
    if bot.y_target is not None:
        cv2.line(big, (0, bot.y_target * SCALE), (w * SCALE, bot.y_target * SCALE), (255, 0, 255), 1)
    if bot.tip is not None:
        cv2.line(big, (0, bot.tip * SCALE), (w * SCALE, bot.tip * SCALE), (0, 255, 0), 1)
    panel = np.zeros((h * SCALE, 340, 3), np.uint8)
    lines = [f"estado : {bot.state}", f"armado : {'SIM' if bot.armed else 'NAO (F7)'}",
             f"topo Y : {bot.tip}", f"faixa  : {bot.band}", f"alvo Y : {bot.y_target}",
             f"latenc.: {bot.L:.1f} ms", f"fps    : {fps:.0f}", "", "F6 sair | F7 armar/desarmar"]
    if bot.state == "IDLE":
        lines += ["", bot.diag[:44], bot.diag[44:88]]
    for i, t in enumerate(lines):
        cv2.putText(panel, t, (8, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return np.hstack([big, panel])


def render_disc(disc, info, state, extra, fps, armed=True):
    k = 4
    big = cv2.resize(disc, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    cx = int((DISC_CX - DISC_REGION["left"]) * k)
    cy = int((DISC_CY - DISC_REGION["top"]) * k)
    cv2.circle(big, (cx, cy), int(DISC_R * k), (255, 255, 255), 1)             # borda esperada do disco
    if info and info["ring_r"]:
        cv2.circle(big, (cx, cy), int(info["ring_r"] * k), (255, 0, 255), 1)   # anel medido (tensão)
    panel = np.zeros((big.shape[0], 340, 3), np.uint8)
    lines = [f"estado : {state}", f"armado : {'SIM' if armed else 'NAO (F7)'}"]
    if info:
        tens = (info['ring_r'] / DISC_R * 100) if info["ring_r"] else 0
        lines += [f"disco  : {'SIM' if info['present'] else 'nao'} (borda={info['rim_dark']:.2f})",
                  f"arco   : {info['arc_deg']} graus",
                  f"anel   : {('%.1f' % info['ring_r']) if info['ring_r'] else '-'} px ({tens:.0f}% do raio)",
                  f"check  : {info['check_n']} px"]
    lines += [f"fps    : {fps:.0f}", ""] + extra + ["", "F6 sair | F7 armar/desarmar"]
    for i, t in enumerate(lines):
        cv2.putText(panel, t, (8, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return np.hstack([big, panel])


def spin_wait(t):
    while True:
        rem = t - time.perf_counter()
        if rem <= 0:
            return
        if rem > 0.020:
            time.sleep(rem - 0.015)


# ----------------------------------------------------------------------------
# LOOP PRINCIPAL
# ----------------------------------------------------------------------------
def run_disc_only():
    """Modo de teste: só lê o disco e mostra os números. Não aperta nenhuma tecla."""
    import keyboard
    import mss

    stop = threading.Event()
    keyboard.add_hotkey("f6", stop.set)
    cv2.namedWindow("Disco")
    cv2.moveWindow("Disco", 1100, 50)
    print("[DISCO] Modo de teste (sem teclas). Abra o minigame de puxar o peixe. F6 sai.")
    last_print, fps, n_fps, t_fps = 0.0, 0.0, 0, time.perf_counter()
    with mss.mss() as sct:
        while not stop.is_set():
            cap = np.array(sct.grab(CAPTURE))[:, :, :3]
            _, disc, _ = split_capture(cap)
            t = time.perf_counter()
            info = analyze_disc(disc)
            n_fps += 1
            if t - t_fps >= 1.0:
                fps, n_fps, t_fps = n_fps / (t - t_fps), 0, t
            if t - last_print >= 0.5:
                print(f"disco={'SIM' if info['present'] else 'nao'} borda={info['rim_dark']:.2f} arco={info['arc_deg']:3d}graus "
                      f"anel={('%.1f' % info['ring_r']) if info['ring_r'] else '  - '}px check={info['check_n']:4d}px fps={fps:.0f}")
                last_print = t
            cv2.imshow("Disco", render_disc(disc, info, "TESTE", [], fps))
            cv2.waitKey(1)
    cv2.destroyAllWindows()


def main():
    import keyboard
    import mss

    if "--disco" in sys.argv:
        run_disc_only()
        return

    try:  # melhora a resolução do relógio no Windows
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

    ref = load_reference()
    press, release, press_e = make_input(INPUT_BACKEND)
    tapper = Tapper(press, release, REEL_TAP_HOLD, REEL_TAP_REST)
    tapper.start()
    guardar_detector = GuardarDetector(CONFIG)
    bot = Bot(ref, time.perf_counter, press=press, release=release, wait_until=spin_wait, tapper=tapper, press_e=press_e, guardar_detector=guardar_detector)
    if ref is None:
        bot.state = "WAIT_REEL"
        print("[MODO] 'referencia.png' ausente: operando diretamente no Minigame Circular.")

    stop = threading.Event()
    keyboard.add_hotkey("f6", stop.set)
    keyboard.add_hotkey("f7", lambda: (bot.set_armed(not bot.armed),
                                       print(f"[F7] {'ARMADO' if bot.armed else 'DESARMADO'}")))

    if SHOW_PREVIEW:
        cv2.namedWindow("Pesca")
        cv2.moveWindow("Pesca", 1300, 50)
    print(f"[OK] Rodando. Armado={bot.armed}. Entrada={INPUT_BACKEND}. Clique no JOGO para dar foco "
          f"(a janela do detector não deve ficar ativa). F6 sai, F7 arma/desarma.")

    last_prev = last_diag = 0.0
    fps, n_fps, t_fps = 0.0, 0, time.perf_counter()
    try:
        with mss.mss() as sct:
            while not stop.is_set():
                cap = np.array(sct.grab(CAPTURE))[:, :, :3]
                t_cap = time.perf_counter()
                frame, disc, guardar = split_capture(cap)
                bot.step(frame, t_cap, disc, guardar)

                n_fps += 1
                if t_cap - t_fps >= 1.0:
                    fps, n_fps, t_fps = n_fps / (t_cap - t_fps), 0, t_cap

                if bot.state in ("IDLE", "WAIT_REEL", "WAIT_GUARDAR") and t_cap - last_diag >= 1.0:
                    print(f"[{bot.state}{'' if bot.armed else ' - DESARMADO'}] {bot.diag} | fps={fps:.0f}")
                    last_diag = t_cap

                # durante a subida do minigame 1 NÃO desenha nada: o loop fica o mais rápido possível
                interval = 0.25 if bot.state == "REELING" else PREVIEW_INTERVAL
                if SHOW_PREVIEW and bot.state != "HOLDING" and t_cap - last_prev >= interval:
                    if bot.state in ("WAIT_REEL", "REELING", "WAIT_GUARDAR"):
                        img = render_disc(disc, bot.disc_info, bot.state, [bot.diag[:44]], fps, bot.armed)
                    else:
                        img = render(bot, frame, fps)
                    cv2.imshow("Pesca", img)
                    cv2.waitKey(1)
                    last_prev = t_cap
    finally:
        tapper.stop_taps()
        tapper.quit.set()
        try:
            release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[FIM] Encerrado; Enter liberado.")


if __name__ == "__main__":
    main()