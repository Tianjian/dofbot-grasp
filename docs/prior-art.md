# Prior art: what the two earlier repos teach

This project is the third iteration of a DOFBOT SE workflow. Two earlier public
repos were studied line by line before writing a single line of the controller:

- [`Tianjian/arm_cam_control`](https://github.com/Tianjian/arm_cam_control) — the arm's
  URDF, a three.js visualizer, the vendor `Arm_Lib.py` serial driver, and a face
  tracking controller that already aimed the wrist camera.
- [`Tianjian/face_tracker`](https://github.com/Tianjian/face_tracker) — a USB camera
  service (YOLOv8-face + FastAPI) that the arm controller polled over HTTP.

The analysis below is kept because it is the justification for several design
decisions in this repository (serial frame layout, the S2/S3/S4 double-reversal
hazard, why perception uses no deep learning, why the read path was replaced).

---

# Prior projects analysis — what to reuse for a wrist-camera 30 mm cube grasp

**Sources analysed (read-only clones):**
- [`Tianjian/arm_cam_control`](https://github.com/Tianjian/arm_cam_control) — Yahboom DOFBOT SE arm: URDF, web visualizer, `Arm_Lib.py` serial driver, face-tracking controller.
- [`Tianjian/face_tracker`](https://github.com/Tianjian/face_tracker) — USB camera + YOLOv8-face + FastAPI stream/API service.

All `file:line` refs are relative to the root of the cited repository unless stated. The arm repo has a
single squashed commit (`3942126 "commit modified files"`), so provenance questions ("where did the
+44° come from?") **cannot** be answered from history.

**Evidence markers used below:** **[V]** = verified by reading the cited lines. **[C]** = computed by
me from the cited source (method stated inline). **[U]** = unverified — no hardware, no measurement,
or a doc↔code conflict; must be settled on the bench. No robot was attached during this analysis, so
every claim about physical behaviour (torque, backlash, actual angles) is **[U]** by construction.

---

## 1. Hardware ground truth

### 1.1 Identity and transport

| Fact | Value | Evidence |
|---|---|---|
| DOF | 5 revolute joints + 1 gripper servo (6 servos, IDs 1–6) | `DOFBOT_SE_SPEC.md:5,11-18` **[V]** |
| Serial baud / timeout | 115200, `timeout=0.2 s` | `Arm_Lib.py:37`, `ARM_LIB_ANALYSIS.md:7` **[V]** |
| Default port | `/dev/myserial` | `Arm_Lib.py:12` (`__init__(com="/dev/myserial")`), `DOFBOT_SE_SPEC.md:87` **[V]** |
| "Device address" 0x15 | `self.addr = 0x15` — **set and never used again in the file** (only occurrence: `Arm_Lib.py:14`) | **[V]** — treat 0x15 as vestigial (an smbus-era leftover); it does not appear in any frame |
| TX frame header | `0xFF`, second byte `0xFC` (`self.__DEVICE_ID`) | `Arm_Lib.py:15-16`, every `cmd = [0xFF,0xFC,…]` **[V]** |
| **RX** frame header | second byte **`0xFB`** = `DEVICE_ID - 1` | `Arm_Lib.py:66` `if head2 == self.__DEVICE_ID - 1:` **[V]** |
| Servo ranges | S1,S2,S3,S4,S6 = 0–180°; **S5 = 0–270°** | `Arm_Lib.py:110` (S5 map), `:148` (`s5 > 270` guard), `ARM_LIB_ANALYSIS.md:15-18`, `DOFBOT_SE_SPEC.md:43-48` **[V]** |
| Reversed IDs | **S2, S3, S4** use `180 - angle` | `Arm_Lib.py:90-91` (single write), `:156,161,166` (write6), `:275,280,285` (write6_array), `:401-402` (read), `ARM_LIB_ANALYSIS.md:31`, `DOFBOT_SE_SPEC.md:121` **[V]** |

Other operational facts: the driver also exposes PWM servo mode, torque on/off, torque-off
mid-position trim (`Arm_serial_servo_write_offset_switch`), RGB and buzzer (`Arm_Lib.py:227,321,333,556`).

### 1.2 The exact wire format (as revealed by `Arm_Lib.py`)

Every frame is `FF FC LEN TYPE DATA… CHECKSUM`, where `LEN` counts from the `LEN` byte through the
checksum inclusive (`LEN = 1 + 1 + len(data) + 1`; confirmed by the three frame types below) **[V]**.

**Single servo** (`Arm_Lib.py:100, 118, 136, 203`) — 9 bytes:
```
FF FC 07 (0x10+id) posH posL timeH timeL checksum
                └──── PWM target, big-endian ────┘ └── ms, big-endian ──┘
```
**All six servos, type 0x1D** (`Arm_Lib.py:185`) — 19 bytes:
```
FF FC 11 1D v1H v1L v2H v2L v3H v3L v4H v4L v5H v5L v6H v6L timeH timeL checksum
```
**All six servos, type 0x1E** (`Arm_Lib.py:306`) — same payload as 0x1D, different function code.
**Read request** (`Arm_Lib.py:363,370`): `FF FC 03 (0x30+id) checksum` (6 bytes).
**Reply** (`Arm_Lib.py:60-82`): `FF FB LEN TYPE data… checksum`; for `TYPE == 0x0A`
(`FUNC_UART_SERVO`, `Arm_Lib.py:19,43-46`) `data = [servoH, servoL, id]`.

**PWM mapping** (`Arm_Lib.py:92, 110, 152-177`) **[V]**:
```
id ≠ 5:  pos = int(900 + 2200 * u / 180)     u ∈ [0,180]   → pos ∈ [900, 3100]
id = 5:  pos = int(380 + 3320 * u / 270)     u ∈ [0,270]   → pos ∈ [380, 3700]
```
**Checksum** — write side is `sum(cmd, 5) & 0xFF` (`Arm_Lib.py:101` and everywhere). This is Python
`sum(iterable, start)`, i.e. `5 + Σbytes`. Since `0xFF + 0xFC + 5 = 512 ≡ 0 (mod 256)`, it is exactly
`(LEN + TYPE + data) & 0xFF` — matching the doc's "从Length开始累加 % 256" (`ARM_LIB_ANALYSIS.md:146`)
and the RX check at `Arm_Lib.py:70,79` **[C]**. The `5` is not a magic constant to reason about; it
is `__COMPLEMENT = 257 - 0xFC` (`Arm_Lib.py:17`), chosen to cancel the two header bytes. **If you ever
change the header byte from 0xFC, that cancellation breaks and every checksum silently becomes wrong.**

Byte layouts above are the single most reusable artefact in the whole prior corpus: a `SerialBus`
can be written straight from them, and they are cheap to verify independently.

### 1.3 Read-back: what the code actually does (and why we should not trust it yet)

`Arm_serial_servo_read(id)` (`Arm_Lib.py:356-403`) **[V]**:
1. sends the read request **twice** (`:363`, `:370`) with `sleep(0.001)` between;
2. waits for a reply whose `data[2]` equals `id+0x30` (`:381`), retrying `__receive_data()` up to 3 more times (`:387`);
3. `d = servoH*256 + servoL`; **`d == 0` is treated as failure** (`:390-391`);
4. converts PWM→degrees: `u = int(270*(d-380)/3320)` for id 5, else `u = int(180*(d-900)/2200)` (`:393-400`);
5. **then** `if id in (2,3,4): u = 180 - u` (`:401-402`).

Consequences:
- The read path applies the **same** 180° reversal the write path applies, so **read and write are
  symmetric in API space**: `read(write(u)) == u` (up to the ~1° `int()` truncation — e.g. u=30 writes
  PWM 2733 and reads back as 31). The reversal is internal, in both directions. See §2.3.
- `self.id` is **never reset between calls**, so a missing/late reply can be satisfied by the *previous*
  reply's payload → a silently stale angle (no exception, no `None`). **[V]**
- `u` is `int()`-truncated on a ~12.2 PWM-counts-per-degree scale (2200/180) → **1° read resolution**
  and up to 1° truncation bias, i.e. **≈4.2 mm of tool motion at 0.24 m reach** **[C]**.
- `__receive_data` reads one byte at a time (`Arm_Lib.py:61-78`) with the 0.2 s port timeout; a missing
  reply costs up to ~1 s per servo (5 timeout windows) → **a 6-servo read sweep worst-cases at ~6 s**
  **[C]**. This is why both consumers avoid the read path in their loops (§3.2, §4.2).
- Robustness bug: `data_len` is only assigned inside `if head2 == 0xFB` (`Arm_Lib.py:66-72`) but used
  unconditionally at `:72`; any stray byte pair raises `NameError`, swallowed by the callers'
  bare `except` → `None`.
- **Protocol ambiguity [U]:** `__parse_data` only accepts reply `TYPE == 0x0A`, yet the request is
  issued with function code `0x30+id` and success is tested against `0x30+id`. It is not decidable from
  the source whether the board echoes `0x30+id` or `0x0A` in the TYPE field. **Resolve by sniffing the
  line once** (`picocom`/`ser2net` + hex log) — until then treat read-back as *unverified*.

`Arm_ping_servo` (`Arm_Lib.py:526-551`) is **broken**: `times` is only incremented *after* the `while`
loop (`:539-548`), so `while value == 0 and times < 5:` never terminates and never reads a response
**[V]**. Do not call it.

### 1.4 Doc↔code conflicts to resolve before trusting the docs

| Claim | Source | Reality |
|---|---|---|
| "默认串口 /dev/myserial" | `DOFBOT_SE_SPEC.md:87`, `Arm_Lib.py:12` | Both runtime programs hard-code **`/dev/ttyUSB0`** (`threejs_server.py:47`, `web_visualizer.py:48`, `face_tracking_controller.py:282`, `FACE_TRACKING_README.md:171,239`) |
| `get_version()` missing `self`, uses undefined `ser`, `Node→None`, `time>=5` | `ARM_LIB_ANALYSIS.md:205-211` | **Stale.** Current file has `def get_version(self)` using `self.ser` (`Arm_Lib.py:477-493`), `None` not `Node` (`:391`), `times >= 5` (`:548`). The bug list describes an older revision **[V]** |
| `effort = 30 N·m`, `velocity = 1.0 rad/s` | `dofbot.urdf:42` etc., `DOFBOT_SE_SPEC.md:35-37` | Plausible as URDF boilerplate, **implausible as datasheet values** for a desktop arm's small bus servos. No torque/current/payload figure exists anywhere in either repo → **[U]**, must not be used in any grasp-force reasoning |

---

## 2. URDF chain, dimensions, workspace, and the servo↔joint mapping

### 2.1 Kinematic chain (`dofbot_info/urdf/dofbot.urdf`) **[V]**

| Joint | Parent→child | origin xyz (m) | origin rpy (rad) | axis | limits (rad) | line |
|---|---|---|---|---|---|---|
| joint1 | base_link→link1 | `0 0 0.06605` | `-0.010805 0 0` | `0 0 1` | −1.5708 … 1.5708 | 37-43 |
| joint2 | link1→link2 | `0 -0.00031873 0.04145` | `0 1.5708 0` | `0 0 1` | −1.5708 … 1.5708 | 61-67 |
| joint3 | link2→link3 | `-0.08285 0 0` | `0 0 0` | `0 0 1` | −1.5708 … 1.5708 | 85-91 |
| joint4 | link3→link4 | `-0.08285 0 0` | `0 0 0.0083081` | `0 0 1` | −1.5708 … 1.5708 | 109-115 |
| joint5 | link4→link5 | `-0.07385 -0.001 0` | `0 -1.57 0` | `0 0 1` | −1.5708 … 3.1416 | 135-141 |

Derived structure **[C]** (from the rpy values):
- **joint1 = yaw** about the base Z axis (link lengths rotate with it).
- joint2's pitch (+90° about Y) turns its axis into the parent's **X**, so **joint2, joint3, joint4 are
  three parallel hinge axes = a planar 3R arm in a vertical plane**. That plane is then yawed by joint1.
- joint5's −90° pitch maps its axis onto the arm's forward direction ⇒ **joint5 is a roll about the tool
  axis** (the gripper/camera spin axis), *not* a wrist pitch.
- Effective DOF for pointing a link5-mounted camera: yaw (j1) + 2-DOF planar pitch (j2,j3,j4, **1
  redundant DOF**) + roll (j5). The redundancy is free "keep the elbow up / stay off the table" authority.

Link lengths **[V]**: 66.05 mm vertical (base→j1), 41.45 mm vertical (j1→j2), then three horizontal
offsets along the arm: 82.85, 82.85, 73.85 mm (+1 mm lateral at j5).

Gaps that matter for us:
- **No tool/TCP link, no gripper joint, no camera frame.** link5 is the last link and its `<visual>` origin
  is identity (`dofbot.urdf:117-133`); joint5's origin *is* the wrist centre. The gripper tip, the
  camera mount pose, and the pinch-point offset must be **hand-measured or CAD-derived [U]**.
- **No `<inertial>` blocks anywhere** → the URDF cannot drive dynamics, gravity torque, or a physically
  meaningful simulation without adding masses. Rigid kinematics only.
- Collision uses the visual STL meshes (so mesh-accurate table checks are possible once a table is added).

### 2.2 Workspace **[C]** — FK over the URDF chain, sampled 200 000 random joint vectors

| Quantity | Value |
|---|---|
| Wrist-centre (joint5 origin) max horizontal radius from base axis | **0.2396 m** (= 0.08285+0.08285+0.07385 exactly) |
| Wrist-centre z range | **−0.0490 … +0.3470 m** |
| Wrist-centre max 3-D distance from base origin | **0.3471 m** |
| Zero pose (all q = 0) | wrist at `(0, 0.0011, 0.347)` — **arm straight up**, not a grasp-ready pose |

Two consequences: (a) `DOFBOT_SE_SPEC.md:66-70`'s "水平 ±0.25 m / 垂直 0~0.25 m / 最大伸展 0.32 m" is a
loose envelope that does **not** match the URDF-derived 0.3471 m; do not design to the spec numbers.
(b) **joint limits do not protect the table**: the wrist can be driven 4.9 cm *below* the base origin
plane, and nothing in `Arm_Lib.py` or the URDF stops that. A table-clearance guard is new work.

### 2.3 The servo-degree ⇄ joint-radian mapping, both directions

Let `u` be the **API angle** — the number `Arm_Lib` takes in `Arm_serial_servo_write*` and returns from
`Arm_serial_servo_read`. Then, for **both** directions **[V]** (`ARM_LIB_ANALYSIS.md:198-199`,
`ANGLE_MAPPING.md:20-27,66,142-148`):

```
q1 = radians(u1 -  90)     q2 = radians(u2 - 90)     q3 = radians(u3 - 90)
q4 = radians(u4 -  90)     q5 = radians(u5 - 135)     [u6 = gripper, no joint]
```

| Servo | API range `u` | internal flip on **write** | internal flip on **read** | joint from `u` | joint limit |
|---|---|---|---|---|---|
| 1 | 0–180 | — | — | `rad(u1-90)` | ±1.5708 |
| 2 | 0–180 | `180 - u2` (`Arm_Lib.py:90-91`) | `d = 180 - d` (`:401-402`) | `rad(u2-90)` | ±1.5708 |
| 3 | 0–180 | `180 - u3` (`:161`) | same | `rad(u3-90)` | ±1.5708 |
| 4 | 0–180 | `180 - u4` (`:166`) | same | `rad(u4-90)` | ±1.5708 |
| 5 | 0–270 | — | — | `rad(u5-135)` | −1.5708 … 3.1416 |
| 6 | 0–180 | — | — | gripper | — |

**The reversal story, stated precisely (this is the trap):**

1. The PWM/encoder frame of S2/S3/S4 is mirrored relative to the arm's logical frame. `Arm_Lib` hides it
   by flipping **on write** (`180-u`) **and flipping back on read** (`180-d`). The read path's own
   `180-d` is *not* a second, separate correction — it is the undo of the write flip.
2. Therefore **`u` is symmetric**: `read(write(u)) == u` (±1° truncation), and the joint mapping `rad(u-90)` is applied
   identically in both directions. **No caller may apply any sign flip on top of `Arm_Lib`.** Any extra
   `180-u` / `90-u` is a double reversal.
3. This is exactly what `ANGLE_MAPPING.md:89-107` ("双重反向" / "错误1") warns about — and exactly what
   the repo itself gets wrong in `threejs_server.py:367-380`, where `send_to_arm` computes
   `servo = 180 - (degrees(joint) + 90)` for joints 2/3/4 ("需要再反向一次"). Algebra: that path is
   `u = 90 - deg(q)`, then `Arm_Lib` flips to `90 + deg(q)`, so the *physical* joint ends at **`-q`** —
   a silent sign error on three joints **[C]**. It is invisible in the UI because the 3D view renders
   the *commanded* cache, never the measured pose (§3.2).
4. `ANGLE_MAPPING.md`'s wording ("Servo 2,3,4 在读取时已经进行了反向处理") is easy to misread as "read
   returns a raw servo angle that needs un-reversing". It does not.
5. S5's formula is written as `servo[4] / 270 * 270 - 135` — the `/270*270` is a no-op typo
   (`ANGLE_MAPPING.md:26,147`); the intent was a 0–270 → −90…180 rescale, which the offset alone already does.

**The +44° joint5 bias [U].** Three code paths add it — `web_visualizer.py:119`, `threejs_server.py:101`,
`face_tracking_controller.py:114` (`servo/270*270 - 135 + 44`) — while `ANGLE_MAPPING.md` (the doc of
record) has no bias. No comment, commit, or calibration file justifies 44°. Its effect is not cosmetic:
with the bias, `u5 = 135` (mid-travel) means joint5 = +44° not 0°, and the usable S5 range shifts to
−91°…+224° — **outside the URDF's −90°…+180° limit at both ends** (the top ~44° of servo travel is
beyond the joint limit) **[C]**. Until someone re-zeros S5 against a physical reference, use the
doc-of-record mapping and treat the 44 as an unexplained fudge.

---

## 3. `arm_cam_control` — what it actually contains

### 3.1 Inventory

| Path | What it is | Verdict for us |
|---|---|---|
| `Arm_Lib.py` (752 L) | The serial driver. Only source of the wire protocol; has the bugs in §1.3 | **vendor + patch** |
| `dofbot_info/urdf/dofbot.urdf`, `meshes/*.STL` | Kinematic truth + 6 meshes (0.9–4.5 MB each) | **reuse as-is** |
| `tools/urdf_parser.py`, `urdf2json.py` | Clean URDF→dict parser, joint-chain extraction (`tools/urdf_parser.py:13,42,149,197`) | **reuse** |
| `tools/stl2gltf.py`, `export_web_assets.py`, `mesh_optimizer.py` | STL→glTF/GLB export, mesh simplification, asset manifest | reuse only if we want a web view |
| `static/models/dofbot.glb` (5.99 MB), `dofbot.json` | Pretrained browser assets. `dofbot.json` embeds **stale absolute paths** from the original authoring machine → regenerate if used | optional |
| `visualize_dofbot_fast.py` | PyVista renderer **with the only FK implementation** (`_compute_forward_kinematics`, `:231-259`; joint-position math is correct: applies the accumulated parent transform to each joint origin) | mine it for FK, don't import the renderer |
| `threejs_server.py` (496 L) | Flask app: `/api/pose` GET/POST, `/api/joint/<n>`, `/api/tracking/*`, `/api/metadata`, `/proxy/face_stream`; background thread updating `current_angles` | **do not inherit** (mapping bugs) |
| `web_visualizer.py` (465 L) | PyVista server-rendered fallback to the same Flask API | superseded |
| `templates/index.html` (27 KB), `templates/threejs_visualizer.html`, `static/js/robot_visualizer.js` (26 KB), `static/js/main.js` | three.js client: URDF→scene, joint sliders, orbit controls | optional debug tool |
| `face_tracking_controller.py` (299 L) | The P-controller (§3.2) | adapt the *shape*, replace the numbers |
| `test_servo_read.py`, `test_joint_positions.py`, `test_joint_rotation.py`, `test_flask.py` | Bench scripts; `test_servo_read.py` has useful read/ping/monitor modes | reuse as diagnostic scaffolding |
| **Absent** | IK, hand-eye calibration, camera intrinsics, gripper open/close calibration, table collision, grasp state machine, force/current sensing | **all new work** |

### 3.2 The control loop in `face_tracking_controller.py` **[V]**

- Image centre **hard-coded `(320, 240)`** (`:36`) — assumes 640×480, and nothing verifies the camera
  actually delivers 640×480 (`face_tracker/src/camera.py:44-46` ignores the return of `cap.set`).
- Gains: `pan_gain = 0.1`, `tilt_gain = 0.1` deg/px; `max_error = 50` px deadband; `min_angle_step = 0.5°`;
  `max_angle_step = 5.0°` (`:39-43`); `control_interval = 1` **second** (`:51`); polls at 10 ms (`:250`).
- Law: `delta = -error * gain`, clipped to ±5°, then zeroed if `< 0.5°` (`:133-144`) — a pure P controller
  with **no I, no D, no Jacobian, no depth**, acting in raw pixels.
- **The gain is dead code in practice:** `control_step` returns early whenever `|err| < 50` px (`:214`), so
  every executed step has `|err| > 50` ⇒ `0.1*err > 5` ⇒ the clip always saturates at 5°. The loop is a
  **bang-bang ±5° at 1 Hz (≈5°/s)** controller; the `min_angle_step` branch can never be reached **[C]**.
- Moves **only servos 1 and 2** (`:174-175`), one `Arm_serial_servo_write` each (so two frames, each with
  its own `sleep(0.012)`, `Arm_Lib.py:87`), each with a **500 ms** interpolation — the joints do not
  arrive together and the camera sweeps during the move.
- Angle source of truth is a **commanded-value cache** (`self.current_servo_angles`), updated before the
  write and never verified (`:165-176`, comment at `:176` "无需再读取串口"); `sync_current_angles()`
  reads all 6 servos once at start and **substitutes `90.0` for any servo that fails** (`:186-192`).
- `control_interval` and gains in the README (0.5/0.5, 0.1 s — `FACE_TRACKING_README.md:104,109`) **do not
  match the code** (0.1/0.1, 1.0 s) **[V]**.

### 3.3 What the visualizer is (and isn't) good for

It is a decent **kinematic sanity toy**: sliders → `send_to_arm` → serial, and a 3D pose built from
URDF-derived glTF. It is *not* evidence of correct hardware behaviour, because (i) it renders the
commanded cache rather than measured angles (`threejs_server.py:137-142` prefers the cache and only falls
back to serial reads), and (ii) its write path double-reverses joints 2/3/4 (§2.3). A 3D robot view also
adds little to a wrist-camera grasp task compared to a **2D overlay on the camera image** (bbox, centre
crosshair, bearing/depth readout), which is what I would build instead.

---

## 4. `face_tracker` — architecture and what transfers

### 4.1 Architecture **[V]**

```
USB cam thread (cv2.VideoCapture, 640x480@30 target, sleep 1/fps)   camera.py:37-46,66
   └─ asyncio task process_frame(): ret,frame = camera.get_frame(); faces = detector.detect(frame)   server.py:33-46
        └─ YOLO('models/yolov8n-face.pt').to('cuda'|'cpu')  (ultralytics ≥8.3, torch ≥2.4)   detector.py:11-20,57
   └─ FastAPI: GET /api/faces, GET /stream (MJPEG), GET /viewer, CORS *   server.py:116-130
```
API surface actually consumed by the arm controller: **only `GET /api/faces`** →
`{count, faces:[{id,bbox[x1,y1,x2,y2],confidence,width,height,area}]}`
(`face_tracking_controller.py:64-71`; response shape from `server.py:58-63` + `detector.py:119-137`).
Note `/api/v1/faces` in `src/api.py:46` is **dead code** — `register_api_routes` is never called by
`create_app` **[V]**.

### 4.2 Structural problems for a closed-loop grasp

| Problem | Evidence | Effect on us |
|---|---|---|
| **Blocking inference inside the event loop** — `process_frame` calls `self.detector.detect(frame)` synchronously from an `async def` | `server.py:33-46` | every `/api/faces` request queues behind a full inference; no back-pressure, no frame dropping |
| **No timestamps anywhere** — `current_faces` carries no frame id/time, `current_frame` has none | `server.py:42-44`, `detector.py:67-71` | a controller cannot tell a 30 ms-old observation from a 500 ms-old one; fatal for a servo loop |
| **No exposure/gain/resolution verification** — `cap.set()` returns are discarded; default V4L2 auto-exposure and auto-white-balance left on | `camera.py:44-46` | brightness/WB drift changes any colour/threshold statistic between frames |
| **Two disagreeing face lists** — `count` from `detect()`'s return, `faces` from `detector.current_faces` | `server.py:58-63` vs `detector.py:74-77` | transient `count != len(faces)` |
| **HTTP hop per control step** — `requests.get(..., timeout=1.0)` once per 1 s step | `face_tracking_controller.py:64` | adds an HTTP RTT to a 1 s loop and can block for 1 s on a stall |
| **GPU/YOLO hard dependency** — `device: "cuda"` default, torch+ultralytics in requirements | `detector.py:11,20`, `requirements.txt:1,8` | exactly what we must not carry |
| Reprojection of the same JPEG repeatedly in `/stream` (encode at 10 ms cadence regardless of new frames) | `server.py:48-56` | wasted CPU; also a second consumer of the same frame |

The repo's own performance claim is a *plan-time estimate*, not a measurement:
"推理速度：~10-30ms (GPU)" (`PROJECT_PLAN.md:97`). **No latency was ever measured in either repo [U].**

### 4.3 Reuse vs replace for a 30 mm cube on local compute

- **Replace the detector.** YOLOv8-face on CUDA is the wrong tool: wrong object, GPU dependency, ~6 MB
  weights, and a generic bbox quality too coarse for a 30 mm target. For a monochrome-ish wooden cube the
  credible local options are (all new code): HSV/Lab segmentation + contour + `minAreaRect`, or
  ArUco/AprilTag if we are allowed to put a marker on the cube, or a tiny ONNX/keypoint model only if
  colour segmentation proves insufficient. Nothing in either repo helps beyond the bbox-shaped output.
- **Reuse the bbox contract, not the transport.** `{center_px, bbox, width, height, area, confidence}`
  is the right observation tuple; return it **in-process** (a function call), not over HTTP. Keep the
  FastAPI `/stream` + `/viewer` only as an optional debugging window.
- **Reuse the camera-capture skeleton** (`camera.py:37-74`: thread + lock + latest-frame) but add:
  a monotonic timestamp per frame, locked exposure/WB, verified resolution, and an actual-FPS counter.
- **The 30 mm edge is a free range sensor.** With calibrated focal length `f` (px), distance ≈
  `f * 0.030 / max(bbox_w, bbox_h)` — this is the metric depth channel the face controller never had, and
  the reason a cube grasp can be done with a single view. It requires camera intrinsics, which **do not
  exist in either repo [U]** — first task on the bench.

---

## 5. Real-grasp lessons and risks

Ranked by how likely they are to sink the first hardware run. Every physical magnitude here is **[U]** —
neither repo contains a single measurement (no torque, no backlash, no repeatability, no latency log).

1. **Camera-mount geometry is unknown, and it is the whole task.** The only evidence that a camera sits on
   link5 is a sentence: "安装在Link5末端，朝向前方" (`FACE_TRACKING_README.md:34`). The URDF has no camera
   link or frame. Before any control law: measure the camera pose relative to the wrist centre and the
   gripper pinch point, and calibrate intrinsics. *Note (our own repo, not the priors):*
   The project's earlier interface note said the wrist cam "looks along the gripper approach
   axis (+X of link5)",
   while `sim/dofbot_se.xml` mounts it at `pos="0.030 0 0.040"`, `xyaxes="1 0 0 0 -1 0"`
   — i.e. offset sideways and **looking down +Z of link5**. Those are not the same axis; pin it down before
   the mount is printed.
2. **Camera flex / mount compliance.** A wrist camera's own lever arm turns any mount flex into a bearing
   error that no calibration can remove (it is non-repeatable). Prior art has zero structural analysis;
   the honest expectation is that a 3D-printed side mount on a moving wrist will shift sub-degree to
   several degrees under acceleration. Mitigation available for free: **image-space servoing** (close the
   loop on the observed cube, not on the commanded pose) tolerates mount error; blind move-then-grasp does not.
3. **Servo read-back is not a measurement.** `Arm_Lib` returns the servo controller's *target*, at 1°
   resolution, on a path with a stale-value failure mode and a ~1 s/servo worst case (§1.3). Both prior
   programs therefore run open-loop on a commanded cache. Design assumption: **the arm is open-loop; the
   camera is the only sensor**. Any "verify the joint actually moved" step must be visual, or must
   re-zero against a physical stop.
4. **Backlash, sag and holding torque shift the pinch point between approach and close.** With
   `effort=30 N·m` as URDF boilerplate and no payload spec, we cannot bound the sag of an extended arm
   carrying a camera. The practical rules: keep the approach short and low (`z` near the table), prefer
   poses where joints 2–4 are not fully extended (max lever = 0.2396 m), and **close the gripper first,
   then lift slowly** so the object is the reference rather than the joint encoder.
5. **Gripper is uncharacterised.** S6 is 0–180° with no URDF joint and **no open/close calibration in
   either repo** (`grep` finds S6 only being passed through: `threejs_server.py:409`). For a 30 mm cube,
   "closed" and "open" angles, the jaw travel↔mm relation, and the stall point all need bench measurement;
   there is no force sensing, so grip force can only be tuned by angle.
6. **Serial timing is not the bottleneck — our sleeps and timeouts are.** Computed from the frames in §1.2
   at 115200 8N1 (10 bits/byte): single-servo frame 9 B ≈ **0.78 ms**, `write6` frame 19 B ≈ **1.65 ms**,
   read request 6 B ≈ **0.52 ms** **[C]**. But `Arm_serial_servo_write` sleeps **12 ms before every single
   frame** (`Arm_Lib.py:87`), `write6` does not (`:147`), and the read path can cost ~1 s. Consequences:
   (a) command coordinated motion with **one `write6` frame**, never six singles; (b) treat `time` as the
   only motion-completion signal (there is no feedback) and wait `time + margin` before reading the next
   camera frame — otherwise the visual servo measures a moving camera.
7. **Control granularity vs a 30 mm object.** At 0.2396 m reach, **1° of joint motion ≈ 4.2 mm** of tool
   travel and the priors' **5°/step ≈ 21 mm** — i.e. *one* control step of the inherited loop can overshoot
   the entire cube (70 % of its width) **[C]**. Carry over the *shape* of the P controller (error → gain →
   clamp → deadband), but re-derive everything in normalised/angular terms with a step limit of a few
   millimetres, and add a settle requirement between steps.
8. **Safety and limits.** (a) joint limits do **not** keep the arm off the table — wrist z reaches −0.049 m
   (§2.2); (b) `write6`'s guard checks only the upper bound (`Arm_Lib.py:148`): a negative angle silently
   produces PWM < 900 (e.g. u = −5 → pos 838) outside the documented 900–3100 range, and out-of-range read
   values are converted to `None` then masked by the callers' `90.0`/`0.0` defaults
   (`face_tracking_controller.py:190`, `threejs_server.py:118,121`) — a fake pose that looks plausible. Clip in
   joint space **before** the driver, and keep a hard software envelope (yaw within joint1's ±90°,
   table clearance, min reach) that is written down independently of the driver.
9. **Lighting and low-contrast wooden cube.** Nothing in either repo addresses non-face detection, so this
   is unmitigated new risk: a wooden cube on a wooden/white desktop gives low colour contrast, specular
   sheen under a desk lamp, and a small target at unknown working distance. Mitigations that cost nothing:
   lock exposure/WB, add a diffused top light, and use `minAreaRect` centroid/angle (which is robust to
   partial mask loss) rather than a detection model; consider a matte marker or a coloured cube if the
   colour route stalls the schedule.
10. **`+44°` and the double reversal are live bugs, not academic.** Both are invisible in the 3D UI because
    nothing closes the loop (§2.3, §3.3). If a wrist-camera loop is layered on top of the inherited mapping,
    it will learn the *wrong* Jacobian sign and hunt. **First hardware acceptance test: command one joint
    ±10°, and check the physical direction by eye — for all five joints — before anything else.**

---

## 6. Reuse table

| Component | Source | Verdict | Reason |
|---|---|---|---|
| `Arm_Lib.py` write frames (0x10+id / 0x1D / 0x1E, PWM maps, checksum) | arm_cam_control | **adapt (vendor + wrap)** | Only authority for the protocol and it is correct for writing; but it prints-and-continues on error, sleeps 12 ms per frame, and silently ignores out-of-range input — wrap with raising + joint-space clipping |
| `Arm_Lib.py` read path (`Arm_serial_servo_read`, `__receive_data`) | arm_cam_control | **replace** | Stale-value mode, ambiguous reply TYPE, ~1 s worst case, `NameError` on stray bytes; keep as an optional diagnostic only |
| `Arm_ping_servo` | arm_cam_control | **discard** | Infinite loop, never reads a reply (`Arm_Lib.py:539-548`) |
| Protocol / hardware docs (`ARM_LIB_ANALYSIS.md`, `ANGLE_MAPPING.md`, `DOFBOT_SE_SPEC.md`) | arm_cam_control | **reuse as reference** | Accurate on frames/PWM/ranges; but `ANGLE_MAPPING` is the mapping authority (no +44), and `ARM_LIB_ANALYSIS`'s bug list is stale (§1.4) |
| `dofbot.urdf` + `meshes/*.STL` | arm_cam_control | **reuse as-is** | Single kinematic truth; already mirrored 1:1 by our MuJoCo model (`sim/dofbot_se.xml` reproduces every origin/rpy/limit) |
| `tools/urdf_parser.py` | arm_cam_control | **reuse** | Clean chain/limit extraction; the nib to build FK/IK on |
| FK (`visualize_dofbot_fast.py:231-259`) | arm_cam_control | **adapt** | Correct joint-position math but welded to PyVista + STL loading; port ~30 lines to numpy instead of importing it |
| IK / Jacobian | — | **build** | **Does not exist anywhere in either repo** (verified by grep for `ik|inverse|jacobian`) |
| `threejs_server.py`, `web_visualizer.py`, `static/js/*`, `dofbot.glb` | arm_cam_control | **replace / optional** | Double-reversal in `send_to_arm` (`:369-380`), renders commanded cache not measured pose, 6 MB asset + Flask + serial-thread complexity; a 2D overlay on the camera image is more useful |
| `tools/stl2gltf.py`, `export_web_assets.py`, `mesh_optimizer.py` | arm_cam_control | **optional reuse** | Fine tools, only needed if a browser view is wanted; regenerate `dofbot.json` (stale absolute paths) |
| P-controller *shape* (error → gain → clamp → deadband) | arm_cam_control | **adapt** | Right skeleton; but gains 0.1 deg/px, 50 px deadband, ±5°/step, 1 Hz, and the "J1+J2 only" choice are not transferable |
| `send_to_arm` joint→servo inversion, `+44°` bias, "no read-back" cache policy | arm_cam_control | **discard** | Double reversal (§2.3), unjustified 44° fudge, and a cache that hides real pose errors |
| `test_servo_read.py` (basic/ping/any/monitor) | arm_cam_control | **adapt** | Useful bench harness for the read-path sniffing test (§1.3) once `ping` is dropped |
| Camera capture thread (locked latest frame) | face_tracker | **adapt** | Right pattern; must add monotonic timestamps, locked exposure/WB, verified resolution/FPS |
| `detect_cube` (new) — HSV/Lab + contour + `minAreaRect`, or ArUco | — | **build** | YOLOv8-face is the wrong class, wrong dependency (torch+CUDA), and too coarse for a 30 mm target |
| Observation contract `{bbox, center, w, h, area, conf}` | face_tracker | **reuse (in-process)** | Correct tuple; extend with timestamp, bearing, and apparent-size→distance; drop the HTTP hop |
| FastAPI `/stream` + `/viewer`, `src/api.py` | face_tracker | **optional / discard** | Debug window only; `/api/v1/*` is dead code (`register_api_routes` never called) |
| `torch`, `ultralytics`, CUDA stack | face_tracker | **discard** | Explicit requirement: local compute, no GPU/YOLO dependency |

---

## 7. Must-measure before the first real grasp (all **[U]** today)

1. Camera intrinsics (`f`, principal point, distortion) and the camera pose w.r.t. wrist centre + gripper
   pinch point — including which axis it actually looks along (§5.1).
2. Gripper: open/closed angles, jaw travel in mm, angle that just holds a 30 mm cube, and whether it
   crushes it.
3. S5 zero: is the 44° bias real? Command `u5 = 135` and check the physical wrist angle against a reference.
4. Joint sign/direction check for all five joints (§5.10) and the read-path reply TYPE (§1.3) — one sniff.
5. Backlash and repeatability per joint (command → return to the same pose, measure the visual offset),
   plus sag with the camera mounted at full extension.
6. Frame-to-command latency end to end (camera timestamp → servo write), which neither repo ever measured.
