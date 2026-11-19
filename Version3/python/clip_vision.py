"""Filtros y utilidades alrededor de la percepcion original (perception.py NO se modifica).

  - Mascara del COFRE del auto: los grabados/logos blancos y dorados del cofre se confundian con lineas de carril.
  - Mascara 'classic' mejorada (top-hat + filtro por forma) y limpieza de manchas para cualquier modo.
  - Suavizado temporal del carril (EMA + descarte de saltos imposibles).
  - Anti-parpadeo del estado mostrado y 'lado hacia el que esta cargado' el auto.

Convenciones del proyecto: e_y > 0  =>  el auto esta a la IZQUIERDA del centro del carril.
"""
import math

import cv2
import numpy as np

from perception import lane_from_mask

# Contorno del cofre del Fantastico en el cuadro de 1280x720 (fracciones x/W, y/H), medido sobre una captura de c01
# (ZED montada fija: es igual en todos los clips). Incluye el pequeno sensor del centro del cofre.
HOOD_POLY_FRAC = [
    (0.086, 1.000), (0.148, 0.972), (0.195, 0.924), (0.234, 0.885),
    (0.312, 0.840), (0.391, 0.824), (0.469, 0.815), (0.488, 0.793),
    (0.516, 0.793), (0.531, 0.815), (0.609, 0.824), (0.688, 0.838),
    (0.750, 0.863), (0.812, 0.910), (0.859, 0.965), (0.898, 1.000),
]
_hood_cache = {}


def hood_mask(h, w, poly_frac=None):
    """Imagen uint8 (h, w): 255 dentro del cofre, 0 fuera."""
    poly = HOOD_POLY_FRAC if poly_frac is None else poly_frac
    key = (h, w, tuple(poly))
    if key not in _hood_cache:
        pts = np.array([[int(round(x * w)), int(round(y * h))] for x, y in poly], np.int32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts], 255)
        _hood_cache[key] = m
    return _hood_cache[key]


def filter_boxes(boxes, hood):
    """Quita detecciones de vehiculos cuyo punto inferior-central cae en el cofre (falsos positivos del propio auto)."""
    if hood is None:
        return boxes
    H, W = hood.shape
    out = []
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        cx, cy = int(np.clip((x1 + x2) / 2, 0, W - 1)), int(np.clip(y2 - 2, 0, H - 1))
        if hood[cy, cx] == 0:
            out.append(b)
    return out


# ------------------------------------------------------------------ mascara ----
def classic_lane_mask(bgr, cam):
    """Fallback SIN red neuronal. Mejoras sobre el umbral simple de brillo:
      - blanco = brillante Y que destaca de su entorno (top-hat): marca delgada sobre asfalto, no cielo ni reflejos anchos;
      - amarillo en HSV con saturacion alta; nada por encima del horizonte;
      - ROI ancho (las lineas laterales de una avenida se ven muy inclinadas, no se recortan).
    La eliminacion del cofre y de manchas se hace despues, en clean_mask().  NO detecta vehiculos (objetos = [])."""
    H, W = bgr.shape[:2]
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)[:, :, 1]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))      # mayor que el ancho de una marca (~5-25 px)
    top = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k)
    white = ((top > 28) & (L > 150)).astype(np.uint8) * 255
    yellow = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), (15, 90, 110), (35, 255, 255))
    m = cv2.bitwise_or(white, yellow)
    horizon = int(cam.cy + 25)
    roi = np.zeros_like(m)
    cv2.fillPoly(roi, [np.array([[(int(-0.6 * W), H), (int(0.30 * W), horizon),
                                  (int(0.70 * W), horizon), (int(1.6 * W), H)]], np.int32)], 255)
    m = cv2.bitwise_and(m, roi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return (m > 0).astype(np.uint8), []


def remove_wide_runs(m, cam, max_width_m=0.6):
    """Por fila, quita los tramos mas anchos de lo que puede medir una marca vial a esa distancia (0.6 m en el piso):
    vehiculos, reflejos y manchas grandes. Asi un auto que toca una linea no la 'funde' con el."""
    out = m.copy()
    H = m.shape[0]
    for v in range(int(cam.cy + 25), H):
        row = m[v]
        if not row.any():
            continue
        maxrun = max(6, int(max_width_m * (v - cam.cy) / cam.height_m))
        d = np.diff(np.concatenate(([0], row, [0])).astype(np.int8))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        for s0, e0 in zip(starts[(ends - starts) > maxrun], ends[(ends - starts) > maxrun]):
            out[v, s0:e0] = 0
    return out


def clean_mask(mask, cam, hood=None, min_area=25, shape_filter=False, road=None, ring_min=0.25):
    """Limpieza comun a los dos modos: sin cofre, sin nada sobre el horizonte, sin manchas diminutas y
    (shape_filter) sin componentes alargados y casi horizontales (barras de paso peatonal, adornos del cofre) y
    (road) sin marcas que NO estan rodeadas de asfalto (maleza, ladera, manchas del acotamiento)."""
    m = (mask > 0).astype(np.uint8)
    m[:int(cam.cy + 25)] = 0
    if hood is not None:
        m[hood > 0] = 0
    m = remove_wide_runs(m, cam)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_area:
            continue
        if shape_filter:
            mo = cv2.moments((lab[y:y + h, x:x + w] == i).astype(np.uint8), binaryImage=True)
            mu20, mu02, mu11 = mo["mu20"], mo["mu02"], mo["mu11"]
            ang = 0.5 * math.degrees(math.atan2(2 * mu11, mu20 - mu02))       # 0 = horizontal
            d = math.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
            l1, l2 = (mu20 + mu02 + d) / 2, max((mu20 + mu02 - d) / 2, 1e-9)
            elong = math.sqrt(l1 / l2)
            if elong > 2.0 and abs(ang) < 10.0:      # barra casi horizontal (paso peatonal, adorno)
                continue
            if a < 120 and elong < 1.6:              # mancha chica y redonda (pasto, reflejos): una raya es alargada
                continue
        if road is not None and a >= 30 and _ring_road_ratio(lab, i, st[i], m, hood, road) < ring_min:
            continue
        keep[i] = True
    return keep[lab].astype(np.uint8)


_K_OUT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
_K_IN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def _ring_road_ratio(lab, i, stat, m, hood, road):
    """Fraccion de asfalto en el anillo (4-12 px) alrededor del componente i. -1 si no hay anillo suficiente."""
    x, y, w, h = stat[:4]
    H, W = m.shape
    x0, y0, x1, y1 = max(0, x - 14), max(0, y - 14), min(W, x + w + 14), min(H, y + h + 14)
    comp = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
    ring = (cv2.dilate(comp, _K_OUT) > 0) & (cv2.dilate(comp, _K_IN) == 0) & (m[y0:y1, x0:x1] == 0)
    if hood is not None:
        ring &= hood[y0:y1, x0:x1] == 0
    if ring.sum() < 20:
        return 1.0                                  # sin anillo valido: no se descarta
    return float(road[y0:y1, x0:x1][ring].mean())


def road_model(frame, cam):
    """Mascara booleana de 'asfalto', aprendida del propio cuadro: se toma como muestra el piso 7-14 m delante del auto
    (mediana, robusta a una raya o un auto) y se acepta lo de croma parecido y luminosidad cercana.
    Se calcula a media resolucion (es una mascara gruesa) y se devuelve a tamano completo."""
    H, W = frame.shape[:2]
    small = cv2.resize(frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.int16)
    pts = []
    for x, ys in ((7.0, (-0.7, 0.7)), (14.0, (0.7, -0.7))):
        for y in ys:
            pts.append((int((cam.cx - y * cam.f / x) / 2), int((cam.cy + cam.f * cam.height_m / x) / 2)))
    patch = np.zeros(small.shape[:2], np.uint8)
    cv2.fillPoly(patch, [np.array(pts, np.int32)], 255)
    px = lab[patch > 0]
    if len(px) < 50:
        return None
    med = np.median(px, axis=0)
    mad = np.median(np.abs(px - med), axis=0) * 1.4826
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    road = (np.abs(a - med[1]) < max(7, 3 * mad[1])) & (np.abs(b - med[2]) < max(7, 3 * mad[2])) & (np.abs(L - med[0]) < 45)
    road = cv2.medianBlur(road.astype(np.uint8) * 255, 3)
    return cv2.resize(road, (W, H), interpolation=cv2.INTER_NEAREST) > 0


# ------------------------------------------------- fronteras del carril propio ----
def _ground(us, vs, cam):
    x = cam.f * cam.height_m / np.maximum(vs - cam.cy, 1e-6)
    y = -(us - cam.cx) * x / cam.f
    return x, y


def self_boundaries(mask, cam, x_eval=8.0, y_max=4.5, max_rms=0.25, max_slope=0.35):
    """Elige, SOLO con la geometria de este cuadro, la frontera izquierda y la derecha del carril propio:
    ajusta una recta en el piso (y = c + m x) a cada componente grande y toma la mas cercana a cada lado.
    Devuelve (izq, der), cada una (c, m) o None."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    left, right = None, None
    vs_all, us_all = np.nonzero(lab)
    ids = lab[vs_all, us_all]
    order = np.argsort(ids, kind="stable")
    vs_all, us_all, ids = vs_all[order], us_all[order], ids[order]
    cuts = np.searchsorted(ids, np.arange(1, n + 1))
    for i in range(1, n):
        if st[i][4] < 80 or st[i][3] < 15:
            continue
        vs, us = vs_all[cuts[i - 1]:cuts[i]], us_all[cuts[i - 1]:cuts[i]]
        x, y = _ground(us.astype(float), vs.astype(float), cam)
        if x.max() - x.min() < 1.5:
            continue
        m, c = np.polyfit(x, y, 1)
        if abs(m) > max_slope or float(np.sqrt(np.mean((y - (m * x + c)) ** 2))) > max_rms:
            continue
        ye = c + m * x_eval
        if 0.2 < ye < y_max and (left is None or ye < left[0] + left[1] * x_eval):
            left = (float(c), float(m))
        elif -y_max < ye < -0.2 and (right is None or ye > right[0] + right[1] * x_eval):
            right = (float(c), float(m))
    return left, right


def gate_mask(mask, cam, lines, tol=0.6):
    """Deja solo los pixeles cuya posicion en el piso esta a menos de `tol` m de alguna de las fronteras."""
    out = np.zeros_like(mask, np.uint8)
    vs, us = np.nonzero(mask)
    if len(vs) == 0 or not any(lines):
        return out
    x, y = _ground(us.astype(float), vs.astype(float), cam)
    ok = np.zeros(len(vs), bool)
    for ln in lines:
        if ln is not None:
            ok |= np.abs(y - (ln[0] + ln[1] * x)) < tol
    out[vs[ok], us[ok]] = 1
    return out


def lines_from_lane(lane):
    """Fronteras esperadas (c, m) del carril dado (e_y, e_psi, ancho)."""
    m = -math.tan(lane["heading_error_rad"])
    c_c = -lane["lateral_error_m"]
    hw = lane["lane_width_m"] / 2.0
    return (c_c + hw, m), (c_c - hw, m)


class LaneTracker:
    """Envuelve lane_from_mask con: limpieza (cofre, formas, asfalto) -> fronteras del carril propio ->
    solo lo que esta sobre ellas -> carril -> suavizado.  Orden de decision:
      1. con el carril del cuadro anterior como guia (tol 0.8 m);
      2. si falla, con las fronteras elegidas solo con este cuadro (sin depender del pasado: no se 'engancha' mal);
      3. si falla, carril INVALIDO (no se inventa nada).  gate=False => lane_from_mask directo sobre la mascara limpia."""

    def __init__(self, hood=None, smooth=0.5, gate=True, road_check=True, shape_filter=True, min_area=25,
                 tol_prev=0.8, tol_self=0.6, width_range=(2.3, 4.8)):
        self.width_range = width_range      # ancho de carril creible (m): evita tomar el guardarrail por la otra linea
        self.hood, self.gate, self.road_check = hood, gate, road_check
        self.shape_filter, self.min_area, self.tol_prev, self.tol_self = shape_filter, min_area, tol_prev, tol_self
        self.smoother = LaneSmoother(smooth)
        self.prev = None            # ultimo carril valido (sin suavizar)
        self.how = "-"              # prev | self | none | off

    def update(self, frame, mask, cam):
        road = road_model(frame, cam) if self.road_check else None
        m0 = clean_mask(mask, cam, self.hood, self.min_area, self.shape_filter, road)
        if not self.gate:
            lane = lane_from_mask(m0, cam)
            self.how, used = "off", m0
        else:
            lane, used, self.how = None, m0, "none"
            if self.prev is not None:
                m1 = gate_mask(m0, cam, lines_from_lane(self.prev), self.tol_prev)
                l1 = lane_from_mask(m1, cam)
                if l1["valid"]:
                    lane, used, self.how = l1, m1, "prev"
            if lane is None:
                left, right = self_boundaries(m0, cam)
                m2 = gate_mask(m0, cam, (left, right), self.tol_self)
                l2 = lane_from_mask(m2, cam)
                used = m2
                lane, self.how = l2, ("self" if l2["valid"] else "none")
        if lane["valid"] and not (self.width_range[0] <= lane["lane_width_m"] <= self.width_range[1]):
            lane = dict(lane, valid=False)      # ancho imposible para un carril: se descarta, no se inventa
            self.how = "ancho"
        self.prev = dict(lane) if lane["valid"] else None
        return used, self.smoother.update(lane)


# --------------------------------------------------------------- suavizado ----
class LaneSmoother:
    """EMA sobre e_y, e_psi, ancho y curvatura + descarte de saltos imposibles entre cuadros.
    alpha = peso de la medicion nueva (0 = sin suavizado). Al perder el carril se reinicia (no inventa nada)."""

    KEYS = ("lateral_error_m", "heading_error_rad", "lane_width_m", "curvature_1pm")

    def __init__(self, alpha=0.5, max_jump_m=1.0, max_rejects=3):
        self.alpha, self.max_jump, self.max_rejects = alpha, max_jump_m, max_rejects
        self.prev, self.rej = None, 0

    def update(self, lane):
        if self.alpha <= 0 or not lane["valid"]:
            self.prev, self.rej = None, 0
            return lane
        if self.prev is None:
            self.prev = dict(lane)
            return lane
        if abs(lane["lateral_error_m"] - self.prev["lateral_error_m"]) > self.max_jump and self.rej < self.max_rejects:
            self.rej += 1                  # salto imposible a 10 Hz: se ignora esta medicion
            return dict(self.prev)
        self.rej = 0
        out = dict(lane)
        for k in self.KEYS:
            out[k] = self.alpha * lane[k] + (1 - self.alpha) * self.prev[k]
        self.prev = out
        return out


# ------------------------------------------------------------ presentacion ----
class StateDebouncer:
    """El estado MOSTRADO solo cambia cuando el nuevo estado se mantiene `min_frames` cuadros seguidos."""

    def __init__(self, min_frames=3):
        self.min_frames, self.shown, self.cand, self.n = max(1, min_frames), None, None, 0

    def update(self, raw):
        """Devuelve True si el estado mostrado cambio en este cuadro."""
        if self.shown is None:
            self.shown = raw
            return False
        if raw == self.shown:
            self.cand, self.n = None, 0
            return False
        self.n = self.n + 1 if raw == self.cand else 1
        self.cand = raw
        if self.n >= self.min_frames:
            self.shown, self.cand, self.n = raw, None, 0
            return True
        return False


def lean_side(e_y, prev=None, deadband=0.10):
    """Lado del carril hacia el que esta cargado el auto, con histeresis para que no parpadee en el centro."""
    if prev in ("IZQUIERDA", "DERECHA"):
        if abs(e_y) < deadband * 0.6:
            return "CENTRADO"
        return "IZQUIERDA" if e_y > 0 else "DERECHA"
    if e_y > deadband:
        return "IZQUIERDA"
    if e_y < -deadband:
        return "DERECHA"
    return "CENTRADO"


def lean_text(side, e_y):
    if side is None:
        return "carril no detectado"
    if side == "CENTRADO":
        return f"centrado ({abs(e_y):.2f} m)"
    return f"cargado a la {side} {abs(e_y):.2f} m"


def steer_text(steer_rad):
    """Convencion del proyecto: steer > 0 = a la izquierda."""
    d = math.degrees(steer_rad)
    if abs(d) < 0.5:
        return "sin correccion"
    return f"corrige a la {'IZQUIERDA' if d > 0 else 'DERECHA'} {abs(d):.1f}°"
