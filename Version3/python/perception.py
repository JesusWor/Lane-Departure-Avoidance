"""Percepcion por camara: mascara de carril -> (e_y, e_psi, ancho); cajas -> objetos en metros.

Convencion de salida (la del automata): x adelante, y IZQUIERDA +, angulos CCW +.
Supuestos: camara montada al centro, sin cabeceo, piso plano (igual que LDA-32 del notebook).
"""
import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

INFER_W, INFER_H = 1280, 720   # YOLOPv2 demo.py trabaja a 1280x720


@dataclass
class CameraModel:
    width: int = INFER_W
    height: int = INFER_H
    fov_deg: float = 90.0        # FOV horizontal
    height_m: float = 1.4        # altura de la camara sobre el piso

    @property
    def f(self):  # focal en px
        return self.width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))

    @property
    def cx(self):
        return self.width / 2.0

    @property
    def cy(self):
        return self.height / 2.0

    def ground_distance(self, v):
        """Distancia hacia adelante de un punto del piso que se ve en la fila v."""
        return self.f * self.height_m / (v - self.cy) if v > self.cy + 1 else None


# ---------------------------------------------------------------- carril ----
def _nearest_runs(cols: np.ndarray, cx: float) -> Tuple[Optional[float], Optional[float]]:
    """Centro del grupo contiguo de pixeles mas cercano a cx a cada lado."""
    if cols.size == 0:
        return None, None
    cols = np.sort(cols)
    groups = np.split(cols, np.where(np.diff(cols) > 3)[0] + 1)
    left = [g for g in groups if g.mean() < cx]
    right = [g for g in groups if g.mean() >= cx]
    l = max(left, key=lambda g: g.mean()).mean() if left else None
    r = min(right, key=lambda g: g.mean()).mean() if right else None
    return l, r


def lane_from_mask(mask: np.ndarray, cam: CameraModel, eval_dist_m: float = 8.0) -> dict:
    """mask: HxW (0/1) con pixeles de linea de carril, ya en la resolucion de `cam`."""
    invalid = dict(valid=False, lateral_error_m=0.0, heading_error_rad=0.0, curvature_1pm=0.0,
                   lane_width_m=3.5, confidence=0.0)
    H, W = mask.shape
    v_min = int(cam.cy + 25)
    v_max = H - 40                      # evita el capot del propio auto
    L, R = [], []
    for v in range(v_min, v_max, 8):
        l, r = _nearest_runs(np.flatnonzero(mask[v]), cam.cx)
        if l is not None: L.append((v, l))
        if r is not None: R.append((v, r))
    if len(L) < 5 or len(R) < 5:
        return invalid
    L, R = np.array(L), np.array(R)
    aL, bL = np.polyfit(L[:, 0], L[:, 1], 1)        # u = a*v + b
    aR, bR = np.polyfit(R[:, 0], R[:, 1], 1)

    v_eval = cam.cy + cam.f * cam.height_m / eval_dist_m
    v_eval = float(np.clip(v_eval, v_min, v_max))
    uL, uR = aL * v_eval + bL, aR * v_eval + bR
    d = cam.ground_distance(v_eval)
    width = (uR - uL) * d / cam.f
    if not (2.2 < width < 5.5):                      # geometria implausible -> no confiar
        return invalid
    e_y = ((uL + uR) / 2.0 - cam.cx) * d / cam.f     # centro del carril a la derecha => ego a la izquierda => +

    # Heading: punto de fuga = interseccion de ambas lineas
    if abs(aL - aR) > 1e-3:
        v_vp = (bR - bL) / (aL - aR)
        u_vp = aL * v_vp + bL
        e_psi = math.atan((u_vp - cam.cx) / cam.f)   # carril se va a la derecha => ego apunta a la izquierda => +
    else:
        e_psi = 0.0
    e_y -= d * math.tan(e_psi)   # e_y se midio a `d` m adelante: llevarlo a la posicion del ego
    conf = float(min(1.0, (len(L) + len(R)) / 40.0))
    return dict(valid=True, lateral_error_m=float(e_y), heading_error_rad=float(e_psi),
                curvature_1pm=0.0, lane_width_m=float(width), confidence=conf)


# -------------------------------------------------------------- objetos ----
class SimpleTracker:
    """Asociacion por vecino mas cercano + velocidad relativa por diferencia filtrada.
    Un track solo es `valid` tras `min_hits` observaciones: antes no hay velocidad confiable
    y el automata debe responder FALSE (regla conservadora del documento)."""

    def __init__(self, gate_m=4.0, alpha=0.4, min_hits=4, max_miss=5):
        self.tracks = {}
        self.next_id = 1
        self.gate, self.alpha, self.min_hits, self.max_miss = gate_m, alpha, min_hits, max_miss

    def update(self, dets: List[dict], t: float) -> List[dict]:
        """dets: dict(x_m, y_m, conf). Devuelve objetos para el mensaje."""
        used = set()
        for tid, tr in self.tracks.items():
            best, bd = None, self.gate
            for i, d in enumerate(dets):
                if i in used: continue
                dist = math.hypot(d["x_m"] - tr["x"], d["y_m"] - tr["y"])
                if dist < bd: best, bd = i, dist
            if best is None:
                tr["miss"] += 1
                continue
            used.add(best)
            d, dt = dets[best], max(1e-3, t - tr["t"])
            vx = (d["x_m"] - tr["x"]) / dt
            tr["vx"] = vx if tr["hits"] == 1 else (1 - self.alpha) * tr["vx"] + self.alpha * vx
            tr.update(x=d["x_m"], y=d["y_m"], t=t, conf=d["conf"], miss=0)
            tr["hits"] += 1
        for i, d in enumerate(dets):
            if i not in used:
                self.tracks[self.next_id] = dict(x=d["x_m"], y=d["y_m"], t=t, vx=0.0, hits=1, miss=0, conf=d["conf"])
                self.next_id += 1
        self.tracks = {k: v for k, v in self.tracks.items() if v["miss"] <= self.max_miss}
        return [dict(id=k, x_m=v["x"], y_m=v["y"], vx_rel_mps=v["vx"], length_m=4.5,
                     valid=v["hits"] >= self.min_hits, conf=v["conf"], source="camera")
                for k, v in self.tracks.items() if v["miss"] == 0]


def boxes_to_positions(boxes, cam: CameraModel) -> List[dict]:
    """boxes: [(x1,y1,x2,y2,conf)] en pixeles de `cam`. Distancia por base del bbox (LDA-32)."""
    out = []
    for x1, y1, x2, y2, conf in boxes:
        d = cam.ground_distance(y2)
        if d is None or d > 120: continue
        u = (x1 + x2) / 2.0
        out.append(dict(x_m=d, y_m=-(u - cam.cx) * d / cam.f, conf=float(conf)))  # y izquierda +
    return out


# --------------------------------------------------------------- YOLOPv2 ----
class YolopV2:
    """Envoltura de YOLOPv2 (repo CAIC-AD/YOLOPv2, pesos yolopv2.pt) siguiendo su demo.py.
    NOTA: escrito contra la estructura de demo.py; si tu copia del repo difiere
    (nombres en utils/utils.py), ajusta solo los imports de abajo."""

    def __init__(self, yolop_dir: str, weights: str, device: str = "cuda", conf=0.3, iou=0.45):
        import torch
        sys.path.insert(0, yolop_dir)
        from utils.utils import (split_for_trace_model, non_max_suppression, scale_coords,
                                 lane_line_mask, letterbox)
        self.torch = torch
        self.split, self.nms, self.scale, self.ll_mask, self.letterbox = (
            split_for_trace_model, non_max_suppression, scale_coords, lane_line_mask, letterbox)
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.half = self.device.type != "cpu"
        self.model = torch.jit.load(weights, map_location=self.device)
        self.model = self.model.to(self.device)
        if self.half: self.model.half()
        self.model.eval()
        self.conf, self.iou = conf, iou
        self(np.zeros((INFER_H, INFER_W, 3), np.uint8))   # warm-up

    def __call__(self, frame_bgr: np.ndarray):
        """Devuelve (lane_mask HxW 0/1 a 1280x720, boxes [(x1,y1,x2,y2,conf)] en 1280x720)."""
        import cv2
        torch = self.torch
        im0 = cv2.resize(frame_bgr, (INFER_W, INFER_H)) if frame_bgr.shape[:2] != (INFER_H, INFER_W) else frame_bgr
        img = self.letterbox(im0, 640, stride=32)[0]
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        img = img.unsqueeze(0)
        with torch.no_grad():
            [pred, anchor_grid], seg, ll = self.model(img)
        pred = self.split(pred, anchor_grid)
        det = self.nms(pred, self.conf, self.iou, classes=None, agnostic=False)[0]
        boxes = []
        if det is not None and len(det):
            det[:, :4] = self.scale(img.shape[2:], det[:, :4], im0.shape).round()
            boxes = [(*[float(v) for v in d[:4]], float(d[4])) for d in det]
        mask = self.ll_mask(ll).astype(np.uint8)
        return mask, boxes
