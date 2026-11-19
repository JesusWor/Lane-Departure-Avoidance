"""Main: reproduce los clips MCAP (Mexican Streets Camera Set) por la tuberia original.

    clip MCAP -> imagen izq. ZED -> [sin cofre + limpieza] -> percepcion (perception.py) -> Link ZeroMQ
                                                                       -> automata C++ -> decision
                                          consola (eventos al cambiar de estado) + ventana/video + CSV + resumen

Es una simulacion en LAZO ABIERTO: el video ya esta grabado, asi que lo que el automata ordena
(steer_rad) NO mueve al auto; se muestra que habria ordenado con lo que vio la camara.

Consola: solo se imprime cuando el ESTADO MOSTRADO cambia (con cuanto duro el anterior y hacia que lado esta
cargado el auto), cuando cambia el resultado de un cambio de carril pedido, y un resumen al final del clip.
Con --print-every N tambien una linea cada N cuadros.

Ejemplos (desde la carpeta python/):
python run_clips.py ../../../mexican_streets_camera_set_20261110/mexican_streets_camera_set_20261110/c01_good --mode classic --show
python run_clips.py ../../../mexican_streets_camera_set_20261110/mexican_streets_camera_set_20261110/c09_poor --mode classic --show --yolop-dir YOLOPv2 --request LEFT --request-at 20
python run_clips.py ../../../mexican_streets_camera_set_20261110/mexican_streets_camera_set_20261110 --all --mode classic --stride 2 --save-video out --csv out

python run_clips.py ../../../mexican_streets_camera_set_20261112/c01_good --mode classic --show
python run_clips.py ../../../mexican_streets_camera_set_20261112/c09_poor --mode classic --show --yolop-dir YOLOPv2 --request LEFT --request-at 20
python run_clips.py ../../../mexican_streets_camera_set_20261112 --all --mode classic --stride 2 --save-video out --csv out
"""
import argparse
import configparser
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from clip_vision import (LaneTracker, StateDebouncer, classic_lane_mask, filter_boxes, hood_mask,
                         lean_side, lean_text, steer_text)
from link import Link, make_msg
from perception import CameraModel, SimpleTracker, boxes_to_positions, lane_from_mask

IMG_TOPIC = "/zed_x_front/left/image_color/compressed"
SPEED_TOPIC = "/spdg/vehicle_speed"
EKF_TOPIC = "/sbg/ekf_nav"
INFER_W, INFER_H = 1280, 720

C = dict(reset="\033[0m", green="\033[92m", yellow="\033[93m", red="\033[91m", dim="\033[2m", cyan="\033[96m")
STATE_COLOR = dict(LANE_KEEPING="green", CORRECTING="yellow", DEGRADED="red")
STATE_BGR = dict(LANE_KEEPING=(80, 220, 80), CORRECTING=(0, 220, 255), DEGRADED=(60, 60, 255))
SRC_TXT = dict(left_only="1 linea (izq) + virtual", right_only="1 linea (der) + virtual",
               held="HOLD ultimo bueno", none="sin carril")


def col(txt, name, on=True):
    return f"{C[name]}{txt}{C['reset']}" if on else txt


# ---------------------------------------------------------------- camara ----
class CalibratedCamera(CameraModel):
    """CameraModel con la focal y el punto principal reales del ZED (ya escalados al cuadro de 1280x720)."""

    def __init__(self, f, cx, cy, height_m):
        super().__init__(INFER_W, INFER_H, 90.0, height_m)
        self._f, self._cx, self._cy = f, cx, cy

    @property
    def f(self):
        return self._f

    @property
    def cx(self):
        return self._cx

    @property
    def cy(self):
        return self._cy


def load_intrinsics(conf_path, section="LEFT_CAM_FHD1200"):
    """Devuelve (fx, cx, cy) en pixeles del cuadro 1920x1200, o None si no se pudo leer."""
    try:
        cp = configparser.ConfigParser()
        cp.read(conf_path)
        s = cp[section]
        return float(s["fx"]), float(s["cx"]), float(s["cy"])
    except Exception as e:
        print(f"[aviso] no pude leer intrinsecos ({conf_path}: {e}); uso FOV 90 por defecto")
        return None


def build_preprocess(intr, cam_height, src_w=1920, src_h=1200):
    """Escala 1920x1200 -> 1280x800 (misma proporcion) y recorta 720 filas alrededor del horizonte,
    para no deformar la imagen antes de YOLOPv2 (que trabaja a 1280x720)."""
    s = INFER_W / src_w
    h_scaled = int(round(src_h * s))
    if intr:
        fx, cx, cy = intr[0] * s, intr[1] * s, intr[2] * s
    else:
        fx, cx, cy = INFER_W / (2 * math.tan(math.radians(45))), INFER_W / 2, h_scaled / 2
    top = int(np.clip(round(cy - INFER_H / 2), 0, max(0, h_scaled - INFER_H)))
    cam = CalibratedCamera(fx, cx, cy - top, cam_height)

    def prep(bgr):
        if bgr.shape[1] != INFER_W:
            bgr = cv2.resize(bgr, (INFER_W, int(round(bgr.shape[0] * INFER_W / bgr.shape[1]))), interpolation=cv2.INTER_AREA)
        return bgr[top:top + INFER_H]
    return prep, cam


# ------------------------------------------------------------ percepcion ----
def _resolve(path, base):
    """Ruta tal cual si existe desde donde corres; si no, relativa a Version3/ (carpeta padre de python/)."""
    p = Path(path)
    if p.exists():
        return p.resolve()
    q = (base / path).resolve()
    return q if q.exists() else p


def make_perceiver(a):
    if a.mode == "yolop":
        from perception import YolopV2
        version3 = Path(__file__).resolve().parent.parent
        ydir = _resolve(a.yolop_dir or "YOLOPv2", version3)
        weights = _resolve(a.weights, ydir) if a.weights else ydir / "data" / "weights" / "yolopv2.pt"
        if not (ydir / "utils" / "utils.py").exists():
            sys.exit(f"No encuentro YOLOPv2/utils/utils.py en: {ydir}\n"
                     f"Pasa --yolop-dir con la ruta a la carpeta YOLOPv2 (la que tiene utils/ y data/).")
        if not weights.exists():
            sys.exit(f"No encuentro los pesos: {weights}\nPasa --weights con la ruta a yolopv2.pt.")
        print(f"YOLOPv2: dir={ydir}\n         pesos={weights}")
        yolo = YolopV2(str(ydir), str(weights), a.device)
        return lambda frame, cam: yolo(frame)
    return classic_lane_mask


# ---------------------------------------------------------------- MCAP ------
def open_reader(mcap_path):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    f = open(mcap_path, "rb")
    return f, make_reader(f, decoder_factories=[DecoderFactory()])


def preload_speed(mcap_path, t_base_ns):
    """Velocidad del auto (m/s) vs tiempo: CAN (/spdg/vehicle_speed, km/h); si es escasa, GNSS (/sbg/ekf_nav)."""
    ts, vs, gts, gvs = [], [], [], []
    f, rd = open_reader(mcap_path)
    try:
        for _, ch, msg, dec in rd.iter_decoded_messages(topics=[SPEED_TOPIC, EKF_TOPIC]):
            t = (msg.log_time - t_base_ns) * 1e-9
            try:
                if ch.topic == SPEED_TOPIC:
                    ts.append(t); vs.append(float(dec.data) / 3.6)
                else:
                    vel = getattr(dec, "velocity", None)
                    if vel is not None and hasattr(vel, "x"):
                        gts.append(t); gvs.append(math.hypot(vel.x, vel.y))
                    elif hasattr(dec, "velocity_n"):
                        gts.append(t); gvs.append(math.hypot(dec.velocity_n, dec.velocity_e))
            except Exception:
                continue
    finally:
        f.close()
    dur = max((ts[-1] - ts[0]) if len(ts) > 1 else 0.0, 1e-6)
    can_hz = len(ts) / dur
    if can_hz >= 5 or not gts:
        return np.array(ts), np.array(vs), "CAN"
    return np.array(gts), np.array(gvs), "GNSS"


def first_log_time(mcap_path):
    f, rd = open_reader(mcap_path)
    try:
        return rd.get_summary().statistics.message_start_time
    finally:
        f.close()


def speed_at(ts, vs, t):
    if len(ts) == 0:
        return None
    i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 1))
    return float(vs[i]) if abs(ts[i] - t) < 2.0 else None


# -------------------------------------------------------------- automata ----
def find_automaton(explicit):
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    for p in [here.parent / "automaton" / "build" / n for n in ("lda_automaton.exe", "lda_automaton")] + \
             [here.parent / "automaton" / "build" / "Release" / "lda_automaton.exe"]:
        if p.exists():
            return p
    return None


def spawn_automaton(a):
    exe = find_automaton(a.automaton)
    if exe is None:
        sys.exit("No encontre lda_automaton. Compilalo (cmake -S automaton -B automaton/build && cmake --build automaton/build) "
                 "o pasa --automaton RUTA, o arrancalo tu aparte y usa --no-spawn.")
    cmd = [str(exe), "-v", "--hz", str(a.hz)]
    for flag, val in (("--kl", a.kl), ("--kh", a.kh), ("--ff", a.ff), ("--min-dist", a.min_dist), ("--min-ttc", a.min_ttc)):
        if val is not None:
            cmd += [flag, str(val)]
    if a.assist_only:
        cmd.append("--assist-only")
    if a.lane_fallback == "on":      # solo la fusion opcional necesita el autómata nuevo; por defecto sirve el original
        cmd += ["--lane-width", str(a.lane_width_prior), "--lane-hold", str(a.lane_hold)]
    # El autómata tiene watchdog de 0.3 s; con inferencia lenta damos margen (lockstep publica justo antes de esperar).
    cmd += ["--max-age-ms", str(a.max_age_ms)]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)

    def pump():
        for line in proc.stderr:
            if not a.quiet:
                print(col("    " + line.rstrip(), "dim", not a.no_color))
    threading.Thread(target=pump, daemon=True).start()
    time.sleep(0.5)
    if proc.poll() is not None:
        time.sleep(0.2)
        sys.exit(f"\nEl automata se cerro al arrancar (codigo {proc.returncode}): {exe}\n"
                 "Si arriba dice 'arg desconocido', ese lda_automaton.exe es una version vieja: recompilalo "
                 "(cmake -S automaton -B automaton/build ... && cmake --build automaton/build) y confirma con test_logic.exe.")
    return proc


# -------------------------------------------------------------- dibujo ------
def draw_overlay(img, mask, boxes, hud, name, t, hood=None, polys=()):
    """El HUD (estado y numeros) viene 'congelado' en `hud`; la mascara y las cajas si cambian en cada cuadro."""
    out = img.copy()
    if hood is not None:                                   # contorno del cofre ignorado (--show-hood)
        cnts, _ = cv2.findContours(hood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 0, 255), 2)
    if mask is not None and mask.any():
        m = cv2.dilate((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        out[m] = (0.35 * out[m] + 0.65 * np.array((0, 255, 0))).astype(np.uint8)
    for (x1, y1, x2, y2, c) in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 160, 0), 2)
    for side, virtual, pts in polys:                       # --draw-lanes: fronteras del carril usado
        p = [(int(u), int(v)) for u, v in pts if 0 <= u < INFER_W and 0 <= v < INFER_H]
        for i in range(len(p) - 1):
            if virtual and i % 2:
                continue
            cv2.line(out, p[i], p[i + 1], (255, 0, 255) if virtual else (0, 230, 255), 3, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (INFER_W, 118), (0, 0, 0), -1)
    if hud:
        bgr = STATE_BGR.get(hud["state"], (200, 200, 200))
        cv2.putText(out, hud["state"], (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, bgr, 2, cv2.LINE_AA)
        cv2.putText(out, hud["line2"], (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, hud["line3"][:120], (12, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cxb = INFER_W // 2                                 # barra de direccion (izquierda = barra a la izquierda)
        cv2.line(out, (cxb, INFER_H - 30), (cxb, INFER_H - 10), (255, 255, 255), 2)
        frac = float(np.clip(hud["steer_rad"] / math.radians(30), -1, 1))
        cv2.rectangle(out, (cxb, INFER_H - 28), (cxb - int(frac * 250), INFER_H - 12), bgr, -1)
    cv2.putText(out, f"{name} t={t:.1f}s", (INFER_W - 300, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ------------------------------------------------------------ un clip -------
def run_clip(clip_dir, a, link, perceive, label=None):
    clip_dir = Path(clip_dir)
    mcaps = sorted(clip_dir.glob("*.mcap"))
    if not mcaps:
        print(f"[{clip_dir.name}] sin .mcap, lo salto")
        return None
    mcap_path = mcaps[0]
    meta = json.loads((clip_dir / "clip.json").read_text()) if (clip_dir / "clip.json").exists() else {}
    labels = (label or {}).get("labels") or meta.get("labels") or {}
    conf = next((clip_dir / "intrinsics").glob("*.conf"), None)
    intr = load_intrinsics(conf) if conf else None
    prep, cam = build_preprocess(intr, a.cam_height)
    tracker = SimpleTracker()
    hood = None if a.no_hood_mask else hood_mask(INFER_H, INFER_W)
    ltrack = LaneTracker(hood, smooth=a.smooth, gate=not a.no_gate, road_check=not a.no_road_check,
                         shape_filter=(a.mode == "classic" and not a.no_clean), min_area=1 if a.no_clean else 25)
    deb = StateDebouncer(a.state_min_frames)
    use = not a.no_color
    if a.lane_fallback == "on":
        from lane_obs import boundary_polylines, lane_observations    # opcional (fusion de una linea en el C++)
    elif a.draw_lanes:
        from lane_obs import boundary_polylines

    print("\n" + "=" * 100)
    print(f"CLIP {clip_dir.name}  | {labels.get('city', '?')} | {labels.get('road_type', '?')} | {labels.get('light', '?')} "
          f"| marcas={labels.get('markings', '?')} (ref {labels.get('marking_score_both_sides', '?')})")
    print(f"cam: f={cam.f:.1f}px cx={cam.cx:.0f} cy={cam.cy:.0f} altura={a.cam_height} m | modo={a.mode} | stride={a.stride}")
    print(f"filtros: cofre={'NO' if a.no_hood_mask else 'si'} | limpieza={'NO' if a.no_clean else 'si'} | "
          f"asfalto={'NO' if a.no_road_check else 'si'} | gating={'NO' if a.no_gate else 'si'} | "
          f"suavizado={a.smooth} | anti-parpadeo={a.state_min_frames} cuadros | respaldo 1 linea={a.lane_fallback}")
    print("=" * 100)

    t_base = first_log_time(mcap_path)
    sp_t, sp_v, sp_src = preload_speed(mcap_path, t_base)
    if a.mode == "classic":
        print(col("[aviso] modo classic: no detecta vehiculos, asi que 'carril objetivo vacio' => TRUE no es confiable. "
                  "Usa --mode yolop para evaluar el cambio de carril.", "yellow", use))
    print(f"velocidad: fuente {sp_src} ({len(sp_t)} muestras)")

    writer, rows = None, []
    stats = dict(frames=0, lane_valid=0, lc_asked=0, lc_true=0, timeouts=0)
    time_in = {"LANE_KEEPING": 0.0, "CORRECTING": 0.0, "DEGRADED": 0.0}
    lean_n, abs_ey = {}, []
    lat_ms, inf_ms = [], []
    raw_prev, raw_trans, shown_trans = None, 0, 0
    state_since, last_t, lean, lc_key = None, None, None, None
    hud, last_hud_t = None, -1e9
    f, rd = open_reader(mcap_path)
    n_seen = 0
    sent = 0
    try:
        for _, ch, msg, dec in rd.iter_decoded_messages(topics=[IMG_TOPIC]):
            t = (msg.log_time - t_base) * 1e-9
            if t < a.start:
                continue
            if a.end and t > a.end:
                break
            n_seen += 1
            if (n_seen - 1) % a.stride:
                continue
            if a.max_frames and stats["frames"] >= a.max_frames:
                break
            raw = cv2.imdecode(np.frombuffer(bytes(dec.data), np.uint8), cv2.IMREAD_COLOR)
            if raw is None:
                continue
            frame = prep(raw)

            t0 = time.perf_counter()
            mask, boxes = perceive(frame, cam)
            boxes = filter_boxes(boxes, hood)
            # limpieza (cofre, formas, asfalto) -> solo lo que esta sobre las 2 fronteras del carril propio -> carril -> suavizado
            mask, lane_py = ltrack.update(frame, mask, cam)               # dos lineas (comportamiento original)
            obs = lane_observations(mask, cam) if a.lane_fallback == "on" else None
            objs = tracker.update(boxes_to_positions(boxes, cam), t)
            t_inf = (time.perf_counter() - t0) * 1e3

            v = speed_at(sp_t, sp_v, t)
            req = a.request if (a.request != "NONE" and t >= a.request_at) else "NONE"
            fid = sent                      # id unico por cuadro enviado (tambien si hubo timeout)
            sent += 1
            link.publish(make_msg(fid, lane_py, objs, req, v, lane_obs=obs, t_s=t))
            t1 = time.perf_counter()
            cmd = link.wait_cmd(fid, a.timeout)
            t_lat = (time.perf_counter() - t1) * 1e3
            if cmd is None:
                stats["timeouts"] += 1
                print(col(f"t={t:6.1f}s  TIMEOUT: el automata no respondio (¿esta corriendo?)", "red", use))
                continue

            lane = cmd.get("lane_used") or lane_py     # geometria de carril que REALMENTE uso el automata
            src = cmd.get("lane_source", "input")
            state_raw = cmd["state"]
            dt = (t - last_t) if last_t is not None else 0.0
            last_t = t
            stats["frames"] += 1
            stats["lane_valid"] += bool(lane["valid"])
            if cmd["lane_change_asked"]:
                stats["lc_asked"] += 1
                stats["lc_true"] += bool(cmd["lane_change_ok"])
            lat_ms.append(t_lat); inf_ms.append(t_inf)

            # ---- estado MOSTRADO (con anti-parpadeo) y lado hacia el que esta cargado el auto ----
            if raw_prev is not None and state_raw != raw_prev:
                raw_trans += 1
            raw_prev = state_raw
            prev_shown = deb.shown
            changed = deb.update(state_raw)
            shown = deb.shown
            time_in[shown] = time_in.get(shown, 0.0) + dt
            e_y = lane["lateral_error_m"]
            lean = lean_side(e_y, lean, a.lean_deadband) if lane["valid"] else None
            lean_n[lean or "SIN CARRIL"] = lean_n.get(lean or "SIN CARRIL", 0) + 1
            if lane["valid"]:
                abs_ey.append(abs(e_y))
            ltxt = lean_text(lean, e_y)
            stxt = steer_text(cmd["steer_rad"])
            vtxt = f"{v * 3.6:.0f} km/h" if v is not None else "? km/h"
            detail = (f"{ltxt} | {stxt} | e_psi {math.degrees(lane['heading_error_rad']):+.1f}° | "
                      f"ancho {lane['lane_width_m']:.2f} m | {vtxt}") if lane["valid"] else f"{ltxt} | {vtxt}"

            # ---- consola: SOLO cuando cambia el estado mostrado (o el resultado de un cambio de carril) ----
            if prev_shown is None:
                state_since = t
                print(f"[{t:6.1f} s] estado inicial: " + col(shown, STATE_COLOR[shown], use))
                print(f"           {detail}")
            elif changed:
                dwell = t - state_since
                state_since = t
                shown_trans += 1
                print(f"[{t:6.1f} s] " + col(f"{prev_shown} -> {shown}", STATE_COLOR[shown], use) + f"   (estuvo {dwell:.1f} s en {prev_shown})")
                print(f"           {detail}")
                if cmd["reason"]:
                    print(f"           motivo: {cmd['reason']}")
            if cmd["lane_change_asked"]:
                key = (req, bool(cmd["lane_change_ok"]))
                if key != lc_key:
                    lc_key = key
                    ok = bool(cmd["lane_change_ok"])
                    print(f"[{t:6.1f} s] CAMBIO DE CARRIL {req}: " + col("TRUE" if ok else "FALSE", "green" if ok else "red", use)
                          + (f"   ({cmd['reason']})" if cmd["reason"] else ""))
            else:
                lc_key = None
            if a.print_every and stats["frames"] % a.print_every == 0:
                print(f"t={t:6.1f}s #{fid:<4d} {col(f'{shown:<12}', STATE_COLOR[shown], use)} {detail}  obj={len(objs)} "
                      f"perc={t_inf:.0f}ms aut={t_lat:.1f}ms")
            if a.json:
                print(col("    " + json.dumps(cmd, ensure_ascii=False), "dim", use))

            rows.append(dict(clip=clip_dir.name, frame=fid, t_s=round(t, 2), speed_mps=v if v is not None else "",
                             lane_valid=int(lane["valid"]), lane_source=src, gate=ltrack.how, e_y_m=round(e_y, 4), lean=lean or "",
                             e_psi_deg=round(math.degrees(lane["heading_error_rad"]), 3),
                             lane_width_m=round(lane["lane_width_m"], 3), n_objects=len(objs),
                             request=req, state_raw=state_raw, state_shown=shown,
                             steer_deg=round(math.degrees(cmd["steer_rad"]), 3),
                             departing=int(cmd["departing"]), lc_asked=int(cmd["lane_change_asked"]),
                             lc_ok=int(cmd["lane_change_ok"]), reason=cmd["reason"],
                             infer_ms=round(t_inf, 1), automaton_ms=round(t_lat, 2)))

            # ---- ventana / video ----
            if a.show or a.save_video:
                if hud is None or changed or a.hud_hz <= 0 or (t - last_hud_t) >= 1.0 / a.hud_hz:
                    line3 = f"{stxt}   e_psi {math.degrees(lane['heading_error_rad']):+.1f} deg   ancho {lane['lane_width_m']:.2f} m" \
                        if lane["valid"] else "sin carril"
                    if cmd["lane_change_asked"]:
                        line3 = f"CAMBIO {req}: {'OK' if cmd['lane_change_ok'] else 'NO'}   " + line3
                    hud = dict(state=shown, line2=ltxt, line3=line3, steer_rad=cmd["steer_rad"])
                    last_hud_t = t
                polys = boundary_polylines(cam, src, lane) if a.draw_lanes else ()
                vis = draw_overlay(frame, mask, boxes, hud, clip_dir.name, t, hood if a.show_hood else None, polys)
                if a.save_video and writer is None:
                    Path(a.save_video).mkdir(parents=True, exist_ok=True)
                    fps = max(1.0, 10.0 / a.stride)
                    writer = cv2.VideoWriter(str(Path(a.save_video) / f"{clip_dir.name}.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (INFER_W, INFER_H))
                if writer is not None:
                    writer.write(vis)
                if a.show:
                    cv2.imshow("LDA clips", vis)
                    key = cv2.waitKey(1 if not a.realtime else int(100 * a.stride)) & 0xFF
                    if key in (27, ord("q")):
                        raise KeyboardInterrupt
    finally:
        f.close()
        if writer is not None:
            writer.release()

    if a.csv and rows:
        Path(a.csv).mkdir(parents=True, exist_ok=True)
        with open(Path(a.csv) / f"{clip_dir.name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    n = max(1, stats["frames"])
    tot = max(1e-9, sum(time_in.values()))
    summ = dict(clip=clip_dir.name, markings=labels.get("markings", "?"), ref_score=labels.get("marking_score_both_sides"),
                frames=stats["frames"], lane_valid_pct=100.0 * stats["lane_valid"] / n,
                keeping_pct=100.0 * time_in["LANE_KEEPING"] / tot, correcting_pct=100.0 * time_in["CORRECTING"] / tot,
                degraded_pct=100.0 * time_in["DEGRADED"] / tot, lc_asked=stats["lc_asked"], lc_true=stats["lc_true"],
                trans=shown_trans, timeouts=stats["timeouts"],
                infer_ms=float(np.mean(inf_ms)) if inf_ms else 0.0, aut_ms_max=float(np.max(lat_ms)) if lat_ms else 0.0)
    print("-" * 100)
    print(f"RESUMEN {clip_dir.name}: {stats['frames']} cuadros | carril valido {summ['lane_valid_pct']:.0f}% | timeouts={stats['timeouts']}")
    print("  tiempo por estado: " + " | ".join(f"{k} {time_in[k]:.1f} s ({100.0 * time_in[k] / tot:.0f}%)" for k in time_in))
    print(f"  transiciones de estado: {shown_trans} mostradas (crudas del automata: {raw_trans}; filtro anti-parpadeo de {a.state_min_frames} cuadros)")
    nl = max(1, sum(lean_n.values()))
    print("  lado de carga: " + " | ".join(f"{k} {100.0 * v / nl:.0f}%" for k, v in sorted(lean_n.items()))
          + (f"   |e_y| medio {np.mean(abs_ey):.2f} m" if abs_ey else ""))
    if stats["lc_asked"]:
        print(f"  cambio de carril: TRUE en {stats['lc_true']} de {stats['lc_asked']} cuadros con solicitud")
    print(f"  percepcion media {summ['infer_ms']:.0f} ms | automata max {summ['aut_ms_max']:.1f} ms")
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="carpeta de un clip (c01_good) o, con --all, la carpeta que las contiene")
    ap.add_argument("--all", action="store_true", help="correr todos los cNN_* de la carpeta (usa manifest.json si existe)")
    ap.add_argument("--mode", choices=["classic", "yolop"], default="classic",
                    help="classic = umbral blanco/amarillo sin red (solo carril); yolop = YOLOPv2 (carril + vehiculos)")
    ap.add_argument("--yolop-dir", default=None, help="carpeta YOLOPv2 (por defecto Version3/YOLOPv2)")
    ap.add_argument("--weights", default=None, help="yolopv2.pt (por defecto <yolop-dir>/data/weights/yolopv2.pt)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cam-height", type=float, default=1.4, help="altura de la camara sobre el piso (m). AJUSTALA a tu montaje")
    # filtros de percepcion
    ap.add_argument("--no-hood-mask", action="store_true", help="NO ignorar el cofre del auto")
    ap.add_argument("--show-hood", action="store_true", help="dibujar el contorno del cofre ignorado (para ajustarlo)")
    ap.add_argument("--no-clean", action="store_true", help="sin limpieza de manchas/formas en la mascara")
    ap.add_argument("--no-gate", action="store_true",
                    help="NO limitar la mascara a las 2 fronteras del carril propio (guardarrail, camellon, vehiculos y maleza pasan)")
    ap.add_argument("--no-road-check", action="store_true", help="NO exigir que las marcas esten rodeadas de asfalto")
    ap.add_argument("--smooth", type=float, default=0.5, help="peso de la medicion nueva en el suavizado del carril (0 = sin suavizar)")
    # presentacion
    ap.add_argument("--state-min-frames", type=int, default=3,
                    help="un estado nuevo se muestra solo tras N cuadros seguidos (anti-parpadeo; 1 = sin filtro)")
    ap.add_argument("--hud-hz", type=float, default=2.0, help="refresco de los numeros del HUD (0 = cada cuadro)")
    ap.add_argument("--lean-deadband", type=float, default=0.10, help="m: dentro de este margen se dice 'centrado'")
    ap.add_argument("--print-every", type=int, default=0, help="ademas de los eventos, una linea cada N cuadros (0 = no)")
    ap.add_argument("--draw-lanes", action="store_true", help="dibujar las fronteras del carril usado (requiere cam-height calibrada)")
    # fusion opcional de una linea (decide el automata C++ nuevo; apagada = comportamiento original)
    ap.add_argument("--lane-fallback", choices=["on", "off"], default="off",
                    help="on = si falta una linea, el automata (C++ nuevo) crea una frontera virtual; off = original")
    ap.add_argument("--lane-width-prior", type=float, default=3.5, help="(fallback on) ancho de carril previo, m")
    ap.add_argument("--lane-hold", type=float, default=0.6, help="(fallback on) s que se retiene el ultimo carril bueno")
    ap.add_argument("--start", type=float, default=0.0, help="s desde el inicio del clip")
    ap.add_argument("--end", type=float, default=0.0, help="s (0 = hasta el final)")
    ap.add_argument("--stride", type=int, default=1, help="procesar 1 de cada N cuadros (10 Hz nativos)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--request", choices=["NONE", "LEFT", "RIGHT"], default="NONE", help="pedir cambio de carril")
    ap.add_argument("--request-at", type=float, default=0.0, help="s del clip desde los que se pide el cambio")
    ap.add_argument("--json", action="store_true", help="imprimir tambien el JSON crudo de cada decision")
    ap.add_argument("--quiet", action="store_true", help="no mostrar el log -v del automata")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--show", action="store_true", help="ventana OpenCV con overlay (q/Esc para salir)")
    ap.add_argument("--realtime", action="store_true", help="con --show, reproducir a ~10 Hz reales")
    ap.add_argument("--save-video", metavar="DIR", help="guardar mp4 con overlay por clip")
    ap.add_argument("--csv", metavar="DIR", help="guardar CSV por clip")
    ap.add_argument("--timeout", type=float, default=2.0, help="s de espera por decision")
    # automata
    ap.add_argument("--automaton", help="ruta a lda_automaton(.exe); por defecto automaton/build/")
    ap.add_argument("--no-spawn", action="store_true", help="el automata ya esta corriendo en otra terminal")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--max-age-ms", type=float, default=300.0)
    ap.add_argument("--kl", type=float); ap.add_argument("--kh", type=float); ap.add_argument("--ff", type=float)
    ap.add_argument("--min-dist", type=float); ap.add_argument("--min-ttc", type=float)
    ap.add_argument("--assist-only", action="store_true")
    a = ap.parse_args()
    if os.name == "nt":
        os.system("")   # activa los colores ANSI en la consola de Windows

    root = Path(a.path)
    if not root.is_dir():
        sys.exit(f"No existe la carpeta: {root.resolve()}\n"
                 f"(se resuelve desde donde corres el comando: {Path.cwd()}). Ajusta el ../ de la ruta.")
    labels_by_id, clips = {}, [root]
    if a.all:
        man = root / "manifest.json"
        if man.exists():
            labels_by_id = {m["id"]: m for m in json.loads(man.read_text(encoding="utf-8"))}
        clips = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("c") and p.name[1:3].isdigit())
    else:
        man = root.parent / "manifest.json"
        if man.exists():
            labels_by_id = {m["id"]: m for m in json.loads(man.read_text(encoding="utf-8"))}

    perceive = make_perceiver(a)
    proc = None if a.no_spawn else spawn_automaton(a)
    link = Link()
    summaries = []
    try:
        for c in clips:
            s = run_clip(c, a, link, perceive, labels_by_id.get(c.name))
            if s:
                summaries.append(s)
    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
        if a.show:
            cv2.destroyAllWindows()
        if proc:
            proc.terminate()

    if len(summaries) > 1:
        print("\n" + "=" * 100 + "\nTABLA COMPARATIVA (ref = marking_score_both_sides del manifest)\n" + "=" * 100)
        print(f"{'clip':<10}{'marcas':<8}{'ref':>6}{'carril%':>9}{'KEEP%':>8}{'CORR%':>8}{'DEGR%':>8}{'trans':>7}{'LC T/P':>9}{'perc ms':>9}")
        for s in summaries:
            ref = f"{s['ref_score']:.2f}" if isinstance(s["ref_score"], (int, float)) else "-"
            print(f"{s['clip']:<10}{s['markings']:<8}{ref:>6}{s['lane_valid_pct']:>9.0f}{s['keeping_pct']:>8.0f}"
                  f"{s['correcting_pct']:>8.0f}{s['degraded_pct']:>8.0f}{s['trans']:>7}"
                  f"{str(s['lc_true']) + '/' + str(s['lc_asked']):>9}{s['infer_ms']:>9.0f}")


if __name__ == "__main__":
    sys.exit(main())
