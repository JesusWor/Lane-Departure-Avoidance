"""
  Python --PUB 5555--> automata (SUB)      percepcion
  Python <--SUB 5556-- automata (PUB)      decision / comando
"""
import json
import time
import zmq


class Link:
    def __init__(self, pub_ep="tcp://127.0.0.1:5555", sub_ep="tcp://127.0.0.1:5556"):
        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.CONFLATE, 1)
        self.pub.bind(pub_ep)
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.connect(sub_ep)
        self.last_cmd = None
        time.sleep(1.0)  # "slow joiner" de PUB/SUB

    def publish(self, msg: dict):
        self.pub.send_string(json.dumps(msg))

    def poll_cmd(self):
        """No bloqueante: devuelve el ultimo comando recibido (o el anterior)."""
        try:
            self.last_cmd = json.loads(self.sub.recv_string(zmq.NOBLOCK))
        except zmq.Again:
            pass
        return self.last_cmd

    def wait_cmd(self, frame_id, timeout_s=0.2):
        """Lockstep: espera la decision correspondiente a `frame_id` (o mas nueva)."""
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            if self.sub.poll(10):
                self.last_cmd = json.loads(self.sub.recv_string())
                if self.last_cmd.get("frame_id", -1) >= frame_id:
                    return self.last_cmd
        return None  # timeout: el automata no respondio


def make_msg(frame_id, lane, objects, request="NONE", ego_speed=None):
    """lane: dict(valid, lateral_error_m, heading_error_rad, curvature_1pm, lane_width_m, confidence)
    objects: lista de dict(id, x_m, y_m, vx_rel_mps, length_m, valid, conf, source)
    Convencion: x adelante, y IZQUIERDA +, angulos CCW +.
    """
    return {
        "ts_ns": time.time_ns(),
        "frame_id": frame_id,
        "request": request,
        "ego": {"speed": ego_speed or 0.0, "speed_valid": ego_speed is not None},
        "lane": lane,
        "objects": objects,
    }
