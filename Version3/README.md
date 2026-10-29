# LDA / LCV — automata en C++ + percepción en Python (CARLA)

```
CARLA ─► run_carla.py ─ percepción (gt | YOLOPv2) ──PUB :5555──►  lda_automaton (C++)
  ▲                                                                   │ FSM + Lane Keeping + TTC
  └──────────── steer (delta) ◄────────────SUB :5556──────────────────┘
```

* **C++ (`automaton/`)**: reglas deterministas (port de `LaneKeepingController`, `LaneGeometry`, `LaneChangeValidator`) + FSM
  `LANE_KEEPING / CORRECTING / DEGRADED`. Event-driven (responde al llegar cada mensaje) y con watchdog a 50 Hz.
* **Python (`python/`)**: percepción y puente CARLA. La red neuronal nunca bloquea al automata; si el dato tiene
  más de `--max-age-ms` (300 por defecto) el automata pasa a `DEGRADED`: steer 0 y cambio de carril = FALSE.
* **Transporte**: ZeroMQ PUB/SUB con `CONFLATE` (solo el último mensaje). Medido aquí: ~0.1–0.4 ms de percepción→decisión.

## Compilar y probar (sin CARLA)

```bash
sudo apt install cmake libzmq3-dev cppzmq-dev nlohmann-json3-dev
cd automaton && cmake -B build && cmake --build build -j
./build/test_logic                       # 25 pruebas de la logica (seccion 26 del documento)

# lazo cerrado con un modelo cinematico falso:
cd ../python && pip install -r requirements.txt
../automaton/build/lda_automaton -v --ff 0 &
python mock_sim.py --scenario free       # free | blocked | rear | stale
```

## Correr en CARLA (0.9.13+)

```bash
./CarlaUE4.sh                                           # terminal 1
./automaton/build/lda_automaton -v                      # terminal 2
python python/run_carla.py --mode gt --request LEFT --request-at 8        # terminal 3
```
Opciones útiles: `--npc-ahead 45 --npc-ahead-diff 50` (coche lento delante en el carril izquierdo),
`--npc-behind 20 --npc-behind-diff -30` (coche rápido que se acerca por atrás), `--request RIGHT`, `--no-lockstep`.
Sobre el ego se dibuja el estado y el TRUE/FALSE del cambio con su razón.

**Orden recomendado:** primero `--mode gt` (percepción perfecta) para validar automata + protocolo + signos;
después `--mode yolop` (necesita el repo YOLOPv2 con `data/weights/yolopv2.pt`):
```bash
python python/run_carla.py --mode yolop --yolop-dir ./YOLOPv2 --weights ./YOLOPv2/data/weights/yolopv2.pt
```
En CARLA el modo es *lockstep*: el puente espera la decisión del `frame_id` actual antes de avanzar el tick
(determinista aunque YOLOP tarde). `--no-lockstep` usa el último comando disponible, como en un vehículo real.

## Protocolo (JSON)

Convención **x adelante, y IZQUIERDA +, ángulos antihorarios +, δ>0 = girar a la izquierda** (la del documento).
El puente convierte desde CARLA (y derecha, yaw horario).

Percepción → automata (`:5555`):
```json
{"ts_ns": 1730000000123456789, "frame_id": 4521, "request": "NONE|LEFT|RIGHT",
 "ego": {"speed": 15.0, "speed_valid": true},
 "lane": {"valid": true, "lateral_error_m": 0.12, "heading_error_rad": -0.01, "curvature_1pm": 0.0,
          "lane_width_m": 3.5, "confidence": 0.93},
 "objects": [{"id": 1, "x_m": 25.0, "y_m": 3.5, "vx_rel_mps": -2.0, "length_m": 4.5,
              "valid": true, "conf": 0.9, "source": "camera"}]}
```
Automata → puente (`:5556`): `{ts_ns, frame_id (eco), state, steer_rad, departing, fresh, lane_change_asked, lane_change_ok, reason}`.

`request` simula a la capa superior que pide el cambio. Un objeto `valid:false` en el carril objetivo ⇒ FALSE.

## Qué está verificado y qué no

| Parte | Estado |
|---|---|
| Lógica C++ (25 pruebas), FSM, watchdog | **Probado** aquí |
| Lazo cerrado C++↔Python por ZeroMQ (4 escenarios, mock cinemático) | **Probado** aquí |
| `lane_from_mask` (signos, ancho, e_y, e_psi) con máscara sintética de verdad conocida | **Probado** aquí (error < 0.03 m en casos normales; subestima heading a ≥0.1 rad) |
| `run_carla.py` | Solo sintaxis; **no se pudo correr contra un servidor CARLA aquí** |
| `YolopV2` (`perception.py`) | Escrito siguiendo `demo.py`; **no ejecutado** (sin pesos/GPU). Si `utils/utils.py` de tu copia difiere, ajusta solo los imports |

Si algo falla en CARLA, lo primero a mirar es el **signo de la dirección**: invierte el signo en la línea `steer = ...`
de `run_carla.py` (o `--kl/--kh`) solo después de confirmar con `--mode gt` que `e_y` y `e_psi` tienen el signo esperado.

## Límites conocidos (honestos)

* Solo **valida** el cambio de carril (como el documento); no ejecuta la maniobra.
* `--mode yolop`: monocular, piso plano, cámara sin cabeceo; velocidad relativa por diferencia de distancia (ruidosa)
  y un track es `valid` solo tras 4 observaciones. Sin radar/LiDAR el cambio de carril será más conservador que en `gt`.
* Sin feedforward (`--ff 0`) el controlador del documento tiene error en estado estacionario en curvas;
  el automata lo activa por defecto con `curvature_1pm` (en `yolop` la curvatura es 0).
* Con ganancias del documento (0.8/1.2) y saturación a 30° la corrección inicial es brusca a 15 m/s: ajusta `--kl/--kh`.
* Umbrales (10 m, 3 s, 0.30 m) son de simulación, no certificados.
