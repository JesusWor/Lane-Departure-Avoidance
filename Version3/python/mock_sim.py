"""Prueba de lazo cerrado SIN CARLA: modelo de bicicleta cinematico + automata C++ por ZeroMQ.

Uso:  (terminal 1) ./build/lda_automaton -v
      (terminal 2) python mock_sim.py [--scenario free|blocked|rear|stale]
Devuelve codigo 0 si el comportamiento es el esperado.
"""
import argparse
import math
import sys
import time

from link import Link, make_msg

WHEELBASE, LANE_W, DT = 2.9, 3.5, 0.05


def run(scenario, duration=14.0):
    link = Link()
    # Estado del ego (carril recto, y izquierda +). Arranca desviado a proposito.
    x, y, yaw, v = 0.0, 0.6, 0.02, 15.0
    # Objetos en el carril izquierdo (y=+3.5), posicion ABSOLUTA en x.
    objs = {
        "free":    [],
        "blocked": [dict(id=1, x=30.0, y=LANE_W, v=8.0)],      # delante, mas lento: TTC/distancia malos
        "rear":    [dict(id=1, x=-15.0, y=LANE_W, v=22.0)],    # detras, mas rapido
        "stale":   [],
    }[scenario]
    expected_ok = scenario == "free"

    frame, t = 0, 0.0
    saw_correcting = saw_degraded = False
    lc_ok_last = None
    max_late = 0.0
    while t < duration:
        # Percepcion perfecta (marco del ego, y izquierda +)
        obs = []
        for o in objs:
            dx, dy = o["x"] - x, o["y"] - y
            obs.append(dict(id=o["id"], x_m=dx * math.cos(yaw) + dy * math.sin(yaw),
                            y_m=-dx * math.sin(yaw) + dy * math.cos(yaw),
                            vx_rel_mps=o["v"] - v, length_m=4.5, valid=True, conf=1.0, source="mock"))
        lane = dict(valid=True, lateral_error_m=y, heading_error_rad=yaw, curvature_1pm=0.0,
                    lane_width_m=LANE_W, confidence=1.0)
        silent = (scenario == "stale" and 9.0 <= t < 10.0)   # la percepcion "se cae" 1 s
        if not silent:
            link.publish(make_msg(frame, lane, obs, request="LEFT" if t > 1.0 else "NONE", ego_speed=v))
            t0 = time.monotonic()
            cmd = link.wait_cmd(frame, 0.2)
            max_late = max(max_late, time.monotonic() - t0)
            if cmd is None:
                print("TIMEOUT: el automata no responde. ¿Esta corriendo?")
                return 2
        else:
            time.sleep(0.4)  # >max_age (0.3 s) -> el automata debe pasar a DEGRADED por watchdog
            cmd = link.poll_cmd()
        if cmd:
            saw_correcting |= cmd["state"] == "CORRECTING"
            saw_degraded |= cmd["state"] == "DEGRADED" and silent
            if cmd["lane_change_asked"] and lc_ok_last is None:
                lc_ok_last = cmd["lane_change_ok"]   # primera respuesta a la solicitud (t~1 s)
            delta = cmd["steer_rad"]
        else:
            delta = 0.0
        # Avance cinematico
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        yaw += v / WHEELBASE * math.tan(delta) * DT
        for o in objs:
            o["x"] += o["v"] * DT
        if int(t / DT) % 20 == 0:
            print(f"t={t:5.1f}s  e_y={y:+.3f}m  yaw={math.degrees(yaw):+.2f}deg  delta={math.degrees(delta):+.2f}deg"
                  f"  state={cmd['state'] if cmd else '-'}  lc_ok={lc_ok_last}")
        frame += 1
        t += DT if not silent else 0.4

    ok = True
    def check(name, cond):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + name)
        ok &= bool(cond)
    check(f"converge al centro (|e_y|={abs(y):.3f} < 0.05)", abs(y) < 0.05)
    check("detecto y corrigio la desviacion inicial (CORRECTING visto)", saw_correcting)
    check(f"1a respuesta a lane change = {expected_ok} (obtenido {lc_ok_last})", lc_ok_last == expected_ok)
    if scenario == "stale":
        check("percepcion caida -> DEGRADED", saw_degraded)
    print(f"latencia maxima percepcion->decision: {max_late*1e3:.2f} ms")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="free", choices=["free", "blocked", "rear", "stale"])
    sys.exit(run(ap.parse_args().scenario))
