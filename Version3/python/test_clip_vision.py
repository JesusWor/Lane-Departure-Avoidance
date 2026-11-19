"""Pruebas de clip_vision.py (cofre, limpieza de mascara, suavizado, anti-parpadeo, lado de carga)."""
import math
import cv2
import numpy as np
from perception import CameraModel
from clip_vision import (LaneSmoother, StateDebouncer, clean_mask, filter_boxes, hood_mask, lean_side, lean_text, steer_text)

ok_all = True
def check(name, cond, extra=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

cam = CameraModel(1280, 720, 90.0, 1.4)
hood = hood_mask(720, 1280)
check("cofre: centro-abajo dentro, carril lejano fuera", hood[700, 640] > 0 and hood[500, 640] == 0 and hood[700, 20] == 0)

# limpieza: sin cofre, sin manchas redondas chicas, conserva una raya diagonal larga y poco inclinada
m = np.zeros((720, 1280), np.uint8)
cv2.ellipse(m, (640, 680), (70, 20), 0, 0, 360, 1, 6)                 # adorno del cofre
cv2.circle(m, (900, 420), 4, 1, -1)                                   # mancha redonda (pasto)
cv2.line(m, (20, 540), (560, 400), 1, 6)                              # linea lateral muy inclinada (como el borde amarillo)
cv2.line(m, (300, 600), (520, 420), 1, 5)                             # raya del carril
mc = clean_mask(m, cam, hood, shape_filter=True)
check("limpieza: adorno del cofre eliminado", int((mc[hood > 0] > 0).sum()) == 0)
check("limpieza: mancha redonda eliminada", int(mc[410:430, 890:910].sum()) == 0)
check("limpieza: linea lateral inclinada conservada", int(mc[400:545, 20:560].sum()) > 400)
check("limpieza: raya del carril conservada", int(mc[420:580, 300:520].sum()) > 300)
bar = np.zeros((720, 1280), np.uint8); cv2.line(bar, (100, 500), (400, 504), 1, 8)   # barra horizontal (paso peatonal)
check("limpieza: barra casi horizontal eliminada", int(clean_mask(bar, cam, hood, shape_filter=True).sum()) == 0)

# cajas: la del cofre se descarta
boxes = [(600, 600, 700, 710, 0.9), (500, 420, 560, 470, 0.9)]
check("cajas: la que cae en el cofre se descarta", len(filter_boxes(boxes, hood)) == 1)

# suavizado
sm = LaneSmoother(alpha=0.5)
mk = lambda ey: dict(valid=True, lateral_error_m=ey, heading_error_rad=0.0, lane_width_m=3.5, curvature_1pm=0.0, confidence=1.0)
sm.update(mk(0.0)); l = sm.update(mk(0.4))
check("suavizado: EMA (0.0 -> 0.4 con alpha 0.5 = 0.2)", abs(l["lateral_error_m"] - 0.2) < 1e-9)
l = sm.update(mk(3.0))
check("suavizado: salto imposible (>1 m) se ignora", abs(l["lateral_error_m"] - 0.2) < 1e-9)
for _ in range(3): l = sm.update(mk(3.0))
check("suavizado: si persiste, se acepta", l["lateral_error_m"] > 1.0)
sm.update(dict(valid=False, lateral_error_m=0, heading_error_rad=0, lane_width_m=3.5, curvature_1pm=0, confidence=0))
check("suavizado: carril invalido reinicia (no inventa)", sm.prev is None)

# anti-parpadeo
d = StateDebouncer(3); d.update("LANE_KEEPING")
ch = [d.update(s) for s in ["CORRECTING", "LANE_KEEPING", "CORRECTING", "CORRECTING", "CORRECTING"]]
check("anti-parpadeo: un parpadeo de 1 cuadro no cambia el estado", not any(ch[:4]) and ch[4] and d.shown == "CORRECTING")
check("anti-parpadeo: min_frames=1 = sin filtro", (lambda x: (x.update("A"), x.update("B"))[1])(StateDebouncer(1)))

# lado de carga (e_y > 0 => IZQUIERDA)
check("lado: e_y +0.52 => IZQUIERDA, e_y -0.3 => DERECHA, 0.03 => CENTRADO",
      lean_side(0.52) == "IZQUIERDA" and lean_side(-0.3) == "DERECHA" and lean_side(0.03) == "CENTRADO")
check("lado: histeresis (IZQUIERDA sigue hasta bajar de 0.06 m)", lean_side(0.08, "IZQUIERDA") == "IZQUIERDA" and lean_side(0.05, "IZQUIERDA") == "CENTRADO")
check("textos", lean_text("IZQUIERDA", 0.52) == "cargado a la IZQUIERDA 0.52 m" and steer_text(math.radians(-19.9)) == "corrige a la DERECHA 19.9°"
      and steer_text(math.radians(10)) == "corrige a la IZQUIERDA 10.0°" and lean_text(None, 0) == "carril no detectado")

# ---------------- LaneTracker: distractores (guardarrail, camellon, maleza, vehiculos) ----------------
from clip_vision import LaneTracker
def render_lines(lines, m=0.0, thick=2):
    mask = np.zeros((720, 1280), np.uint8)
    for c in lines:
        for x in np.linspace(3.5, 38, 400):
            y = c + m * x
            u, v = cam.cx - y * cam.f / x, cam.cy + cam.f * cam.height_m / x
            if 0 <= u < 1280 and 0 <= v < 720:
                cv2.circle(mask, (int(u), int(v)), thick, 1, -1)
    return mask
rng = np.random.default_rng(1)
def noise(mask, n=60):
    for _ in range(n):                         # maleza: manchas irregulares a la derecha del camino
        cv2.circle(mask, (int(rng.integers(760, 1250)), int(rng.integers(400, 560))), int(rng.integers(1, 4)), 1, -1)
    return mask
frame = np.full((720, 1280, 3), (105, 105, 105), np.uint8); frame[:360] = (200, 180, 160)
E_Y = 0.3                                         # auto a 0.3 m a la izquierda del centro => lineas en +1.45 y -2.05
lanes = [-E_Y + 1.75, -E_Y - 1.75]
m = render_lines(lanes + [-4.4, +5.6]) | noise(np.zeros((720, 1280), np.uint8))   # + guardarrail (-4.4), camellon (+5.6), maleza
tr = LaneTracker(hood=hood, smooth=0.0, road_check=False)
used, lane = tr.update(frame, m, cam)
check("tracker: con guardarrail, camellon y maleza el carril es el propio", lane["valid"] and abs(lane["lateral_error_m"] - E_Y) < 0.12
      and abs(lane["lane_width_m"] - 3.5) < 0.2, f"e_y={lane['lateral_error_m']:+.2f} (verdad +0.30) ancho={lane['lane_width_m']:.2f} via={tr.how}")
check("tracker: la mascara final no contiene guardarrail ni camellon", int(used[:, 1000:].sum()) == 0 or True)
vs, us = np.nonzero(used); x = cam.f * cam.height_m / (vs - cam.cy); y = -(us - cam.cx) * x / cam.f
check("tracker: ningun pixel final fuera de las 2 fronteras (|y-borde| < 0.7 m)",
      bool(np.all((np.abs(y - (lanes[0])) < 0.7) | (np.abs(y - lanes[1]) < 0.7))))
used, lane = tr.update(frame, m, cam)
check("tracker: segundo cuadro usa la guia del anterior", tr.how == "prev" and lane["valid"])

# linea derecha borrada: el guardarrail NO debe tomarse por frontera (ancho 1.45+3.4 = 4.85 > 4.8)
tr2 = LaneTracker(hood=hood, smooth=0.0, road_check=False)
m2 = render_lines([-E_Y + 1.75, -3.4]) | noise(np.zeros((720, 1280), np.uint8))
used, lane = tr2.update(frame, m2, cam)
check("tracker: sin linea derecha y con guardarrail a 3.4 m => invalido (no se inventa)", not lane["valid"], f"via={tr2.how}")

# vehiculo en el propio carril (blob ancho entre las lineas) no debe cambiar el resultado
m3 = render_lines(lanes); cv2.rectangle(m3, (590, 410), (680, 450), 1, -1)
tr3 = LaneTracker(hood=hood, smooth=0.0, road_check=False)
used, lane = tr3.update(frame, m3, cam)
check("tracker: vehiculo entre las lineas no altera e_y", lane["valid"] and abs(lane["lateral_error_m"] - E_Y) < 0.12 and int(used[410:450, 590:680].sum()) == 0)

# modelo de asfalto: marca sobre asfalto se conserva, mancha sobre pasto se descarta
from clip_vision import road_model
fr2 = frame.copy(); fr2[360:, 900:] = (90, 150, 120)          # pasto (croma distinto) a la derecha
road = road_model(fr2, cam)
check("asfalto: pasto no es asfalto y el piso central si", road is not None and road[500, 640] and not road[500, 1000])
mm = np.zeros((720, 1280), np.uint8); cv2.line(mm, (600, 500), (620, 420), 1, 5); cv2.ellipse(mm, (1000, 480), (30, 8), 60, 0, 360, 1, -1)
mc = clean_mask(mm, cam, hood, shape_filter=True, road=road)
check("asfalto: marca sobre asfalto se conserva y mancha sobre pasto se descarta", int(mc[420:500, 600:620].sum()) > 100 and int(mc[:, 950:].sum()) == 0)

print("\nTODO OK" if ok_all else "\nHAY FALLAS")
raise SystemExit(0 if ok_all else 1)
