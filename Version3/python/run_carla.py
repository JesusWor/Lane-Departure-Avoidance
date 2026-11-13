import argparse
import math
import queue
import sys
import time

import numpy as np

import carla
from link import Link, make_msg
from perception import CameraModel, SimpleTracker, YolopV2, boxes_to_positions, lane_from_mask


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


# ----------------------------------------------------- percepcion GT --------
def gt_lane(world_map, ego_tf):
    loc = ego_tf.location
    wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    rel = loc - wp.transform.location
    e_y = -dot(rel, wp.transform.get_right_vector())                     # CARLA y es DERECHA: invertir
    e_psi = -math.radians(wrap180(ego_tf.rotation.yaw - wp.transform.rotation.yaw))  # yaw CARLA es horario
    nxt = wp.next(5.0)
    curv = 0.0
    if nxt:
        dyaw = -math.radians(wrap180(nxt[0].transform.rotation.yaw - wp.transform.rotation.yaw))
        curv = dyaw / max(0.1, wp.transform.location.distance(nxt[0].transform.location))
    return wp, dict(valid=True, lateral_error_m=e_y, heading_error_rad=e_psi, curvature_1pm=curv,
                    lane_width_m=wp.lane_width, confidence=1.0)


def gt_objects(world, ego, max_range=100.0):
    tf = ego.get_transform()
    fwd, right = tf.get_forward_vector(), tf.get_right_vector()
    ev = ego.get_velocity()
    out = []
    for a in world.get_actors().filter("vehicle.*"):
        if a.id == ego.id:
            continue
        rel = a.get_location() - tf.location
        x, y = dot(rel, fwd), -dot(rel, right)
        if math.hypot(x, y) > max_range:
            continue
        v = a.get_velocity()
        out.append(dict(id=a.id, x_m=x, y_m=y, vx_rel_mps=dot(v, fwd) - dot(ev, fwd),
                        length_m=2 * a.bounding_box.extent.x, valid=True, conf=1.0, source="carla_gt"))
    return out


# -------------------------------------------------------------- escena ------
def pick_spawn(world_map, direction):
    for sp in world_map.get_spawn_points():
        wp = world_map.get_waypoint(sp.location, lane_type=carla.LaneType.Driving)
        adj = wp.get_left_lane() if direction != "RIGHT" else wp.get_right_lane()
        if adj and adj.lane_type == carla.LaneType.Driving and adj.lane_id * wp.lane_id > 0 \
                and len(wp.next(150.0)) > 0:
            return sp, wp
    raise RuntimeError("No encontre un spawn con carril adyacente. Prueba --town Town04.")


def spawn_npc(world, tm, bp_lib, wp, speed_diff, rng):
    if wp is None:
        return None
    bp = rng.choice(bp_lib.filter("vehicle.audi.*") or bp_lib.filter("vehicle.*"))
    tf = wp.transform
    tf.location.z += 0.5
    npc = world.try_spawn_actor(bp, tf)
    if npc:
        npc.set_autopilot(True, tm.get_port())
        tm.vehicle_percentage_speed_difference(npc, speed_diff)   # + = mas lento que el limite
        tm.auto_lane_change(npc, False)
        tm.ignore_lights_percentage(npc, 100)
    return npc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town04")
    ap.add_argument("--mode", choices=["gt", "yolop"], default="gt")
    ap.add_argument("--yolop-dir", default="YOLOPv2")
    ap.add_argument("--weights", default="YOLOPv2/data/weights/yolopv2.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target-speed", type=float, default=15.0, help="m/s")
    ap.add_argument("--request", choices=["NONE", "LEFT", "RIGHT"], default="LEFT")
    ap.add_argument("--request-at", type=float, default=8.0, help="s de simulacion")
    ap.add_argument("--npc-ahead", type=float, default=45.0, help="m delante en el carril adyacente (0=no)")
    ap.add_argument("--npc-behind", type=float, default=0.0, help="m detras en el carril adyacente (0=no)")
    ap.add_argument("--npc-ahead-diff", type=float, default=50.0, help="%% mas lento que el limite")
    ap.add_argument("--npc-behind-diff", type=float, default=-30.0, help="negativo = mas rapido")
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--no-lockstep", action="store_true", help="no esperar la decision (modo tiempo real)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    import random
    rng = random.Random(a.seed)
    client = carla.Client(a.host, a.port)
    client.set_timeout(20.0)
    world = client.get_world()
    if not world.get_map().name.endswith(a.town):
        world = client.load_world(a.town)
    world_map = world.get_map()
    orig_settings = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, a.dt
    world.apply_settings(s)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(a.seed)

    actors = []
    link = None
    try:
        bp_lib = world.get_blueprint_library()
        direction = a.request if a.request != "NONE" else "LEFT"
        sp, wp0 = pick_spawn(world_map, direction)
        sp.location.z += 0.5
        ego = world.spawn_actor(bp_lib.find("vehicle.tesla.model3"), sp)
        actors.append(ego)
        max_steer = math.radians(ego.get_physics_control().wheels[0].max_steer_angle)

        adj = wp0.get_left_lane() if direction != "RIGHT" else wp0.get_right_lane()
        if a.npc_ahead > 0 and adj.next(a.npc_ahead):
            actors.append(spawn_npc(world, tm, bp_lib, adj.next(a.npc_ahead)[0], a.npc_ahead_diff, rng))
        if a.npc_behind > 0 and adj.previous(a.npc_behind):
            actors.append(spawn_npc(world, tm, bp_lib, adj.previous(a.npc_behind)[0], a.npc_behind_diff, rng))
        actors = [x for x in actors if x]

        cam_model = CameraModel(1280, 720, 90.0, 1.4)
        img_q, yolo, tracker = None, None, SimpleTracker()
        if a.mode == "yolop":
            cbp = bp_lib.find("sensor.camera.rgb")
            cbp.set_attribute("image_size_x", "1280")
            cbp.set_attribute("image_size_y", "720")
            cbp.set_attribute("fov", "90")
            cam = world.spawn_actor(cbp, carla.Transform(carla.Location(x=0.5, z=cam_model.height_m)), attach_to=ego)
            actors.append(cam)
            img_q = queue.Queue()
            cam.listen(img_q.put)
            yolo = YolopV2(a.yolop_dir, a.weights, a.device)

        link = Link()
        spectator = world.get_spectator()
        print("Escena lista. Asegurate de que lda_automaton esta corriendo (sin el, el ego frena).")

        frame, t, last_print = 0, 0.0, -1.0
        lockstep = not a.no_lockstep
        while t < a.duration:
            world.tick()
            t += a.dt
            tf = ego.get_transform()
            vel = ego.get_velocity()
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            spectator.set_transform(carla.Transform(tf.transform(carla.Location(x=-9, z=4)),
                                                    carla.Rotation(pitch=-15, yaw=tf.rotation.yaw)))

            t_inf = 0.0
            if a.mode == "gt":
                _, lane = gt_lane(world_map, tf)
                objs = gt_objects(world, ego)
            else:
                img = img_q.get(timeout=5.0)
                frame_bgr = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3]
                t0 = time.perf_counter()
                mask, boxes = yolo(frame_bgr)
                lane = lane_from_mask(mask, cam_model)
                objs = tracker.update(boxes_to_positions(boxes, cam_model), t)
                t_inf = time.perf_counter() - t0

            req = a.request if t >= a.request_at else "NONE"
            link.publish(make_msg(frame, lane, objs, req, speed))
            cmd = link.wait_cmd(frame, 0.5) if lockstep else link.poll_cmd()

            ok_cmd = bool(cmd) and cmd.get("fresh") and cmd.get("state") != "DEGRADED"
            steer = float(np.clip(-cmd["steer_rad"] / max_steer, -1, 1)) if ok_cmd else 0.0  # CARLA: + = derecha
            err = a.target_speed - speed
            ctrl = carla.VehicleControl(
                steer=steer,
                throttle=float(np.clip(0.4 * err, 0.0, 0.75)) if ok_cmd else 0.0,
                brake=float(np.clip(-0.3 * err, 0.0, 1.0)) if ok_cmd else 0.3)
            ego.apply_control(ctrl)

            if cmd:
                txt = f"{cmd['state']}  e_y={lane['lateral_error_m']:+.2f}m"
                if cmd["lane_change_asked"]:
                    txt += f"\nCAMBIO {req}: {'OK (TRUE)' if cmd['lane_change_ok'] else 'NO (FALSE)'}\n{cmd['reason']}"
                color = carla.Color(0, 255, 0) if (cmd["lane_change_ok"] or not cmd["lane_change_asked"]) \
                    else carla.Color(255, 60, 60)
                world.debug.draw_string(tf.location + carla.Location(z=3.2), txt, False, color, a.dt * 1.5)
            if t - last_print >= 1.0:
                last_print = t
                print(f"t={t:5.1f}s v={speed:4.1f}m/s e_y={lane['lateral_error_m']:+.2f} "
                      f"e_psi={math.degrees(lane['heading_error_rad']):+.1f}deg obj={len(objs)} "
                      f"{cmd['state'] if cmd else 'SIN-CMD'} "
                      f"lc={cmd['lane_change_ok'] if cmd and cmd['lane_change_asked'] else '-'}"
                      + (f" infer={t_inf*1e3:.0f}ms" if t_inf else ""))
            frame += 1
    except KeyboardInterrupt:
        pass
    finally:
        print("Limpiando...")
        try:
            tm.set_synchronous_mode(False)
            world.apply_settings(orig_settings)
        finally:
            for x in actors:
                try:
                    if hasattr(x, "stop"):
                        x.stop()
                    x.destroy()
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())
