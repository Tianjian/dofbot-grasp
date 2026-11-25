#!/usr/bin/env python3
"""Record the DOFBOT SE wrist-camera grasp from MuJoCo, for visual review.

Two clips in one video: a successful grasp, then the same motion with the
fingers held open (the fault the harness is supposed to catch). Each frame shows
a third-person view plus the live wrist-camera image, with the phase, simulated
time and the metrics the contract evaluates.

Usage (from the repository root):
    MUJOCO_GL=egl python sim/record_grasp.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_DIR))
import run_grasp_sim as sim  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
MP4 = OUT_DIR / "grasp_demo.mp4"
POSTER = OUT_DIR / "grasp_demo_poster.png"

W, H = 1280, 720
FPS = 30
BG = (11, 15, 20)
PANEL = (17, 24, 32)
BORDER = (48, 60, 74)
TEXT = (214, 221, 229)
DIM = (139, 148, 158)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
YELLOW = (210, 153, 34)
BLUE = (88, 166, 255)

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(p: str, s: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(p, s)


F_TITLE = font(SANS_B, 34)
F_H1 = font(SANS_B, 22)
F_LBL = font(MONO_B, 14)
F_TXT = font(MONO, 15)
F_SMALL = font(MONO, 13)

PHASES = ["home", "search", "align", "approach", "grasp", "lift", "verify"]


def load_model() -> mujoco.MjModel:
    """Raise the offscreen framebuffer so we can render 960x540 + 640x480."""
    xml = sim.XML.read_text()
    # the model already declares a framebuffer size; raise it instead of adding a
    # second <global> element (MuJoCo allows only one)
    xml = re.sub(r'offwidth="\d+" offheight="\d+"', 'offwidth="1280" offheight="720"', xml, count=1)
    # render-only: a dark floor makes the light arm read clearly against the
    # table. The committed model keeps its checker material.
    xml = xml.replace(
        '<geom name="floor" type="plane" size="0 0 0.01" pos="0 0 -0.02" material="table_mat"/>',
        '<geom name="floor" type="plane" size="0 0 0.01" pos="0 0 -0.02" rgba="0.09 0.10 0.12 1"/>',
    )
    tmp = Path("/tmp/dofbot_record.xml")
    tmp.write_text(xml)
    return mujoco.MjModel.from_xml_path(str(tmp))


def contacts(data) -> int:
    n = 0
    for c in data.contact[: data.ncon]:
        names = {data.model.body(data.model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)}
        if "cube" in names and any("finger" in x for x in names):
            n += 1
    return n


def render_frame(model, data, cam, main_r, wrist_r) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    draw.text((30, 24), "DOFBOT SE  ·  wrist-camera grasp  ·  MuJoCo", font=F_TITLE, fill=TEXT)
    draw.text(
        (30, 66),
        "30 mm cube · 5-DOF + gripper · camera on link5 looking down the tool axis",
        font=F_SMALL,
        fill=DIM,
    )

    main_r.update_scene(data, camera=cam)
    main = Image.fromarray(main_r.render())
    main = main.resize((880, 495), Image.LANCZOS)
    draw.rounded_rectangle((28, 98, 28 + 884, 98 + 499), radius=10, fill=PANEL, outline=BORDER)
    img.paste(main, (30, 100))

    wrist_r.update_scene(data, camera="wrist_cam")
    wrist = Image.fromarray(wrist_r.render()).resize((316, 237), Image.LANCZOS)
    draw.rounded_rectangle((928, 98, 928 + 320, 98 + 241), radius=10, fill=PANEL, outline=BORDER)
    img.paste(wrist, (930, 100))
    draw.text((930, 344), "wrist_cam (live)", font=F_LBL, fill=DIM)

    return img


def run_clip(model, fault: bool, out_dir: Path) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cube_id = model.body("cube").id
    cube0 = data.xpos[cube_id].copy()
    z = np.array([0.0, 0.0, -1.0])
    seed = np.array([-0.058, 0.909, 0.987, 1.091, 1.127])

    home = np.zeros(5)
    search = sim.ik_tcp(model, data, cube0 + [0, 0, 0.12], z, seed)
    align = sim.ik_tcp(model, data, cube0 + [0, 0, 0.08], z, search)
    approach = sim.ik_tcp(model, data, cube0 + [0, 0, 0.004], z, align)
    grasp_q = sim.ik_tcp(model, data, cube0 + [0, 0, 0.004], z, approach)
    lift_q = sim.ik_tcp(model, data, cube0 + [0, 0, 0.12], z, grasp_q)

    open_g, closed_g = 0.010, (0.010 if fault else -0.002)
    plan = [
        ("home", home, open_g, 500),
        ("search", search, open_g, 900),
        ("align", align, open_g, 700),
        ("approach", approach, open_g, 700),
        ("grasp", grasp_q, closed_g, 700),
        ("lift", lift_q, closed_g, 800),
        ("verify", lift_q, closed_g, 400),
    ]

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (-0.03, -0.12, 0.11)
    cam.distance = 0.46
    cam.azimuth = 138.0
    cam.elevation = -20.0

    main_r = mujoco.Renderer(model, 540, 960)
    wrist_r = mujoco.Renderer(model, 480, 640)
    frames: list[tuple[Image.Image, str, float, float, int]] = []
    every = 6
    step_i = 0
    for name, q, grip, steps in plan:
        start = np.array([data.ctrl[i] for i in range(5)])
        for s in range(steps):
            a = s / max(steps - 1, 1)
            for i in range(5):
                data.ctrl[i] = start[i] + a * (q[i] - start[i])
            data.ctrl[5], data.ctrl[6] = grip, -grip
            mujoco.mj_step(model, data)
            step_i += 1
            if step_i % every == 0:
                img = render_frame(model, data, cam, main_r, wrist_r)
                cube_h = float(data.xpos[cube_id][2] * 1000.0)
                frames.append((img, name, data.time, cube_h, contacts(data)))

    lifted = float((data.xpos[cube_id][2] - cube0[2]) * 1000.0)
    return {"frames": frames, "lifted_mm": lifted, "cube0": cube0.tolist()}


def decorate(frame, phase, sim_t, cube_mm, ncon, badge, idx, total) -> Image.Image:
    img = frame.copy()
    draw = ImageDraw.Draw(img)

    label, colour = badge
    tw = F_LBL.getlength(label)
    draw.rounded_rectangle((W - 30 - tw - 26, 26, W - 30, 54), radius=14, fill=colour)
    draw.text((W - 30 - tw - 13, 33), label, font=F_LBL, fill=(8, 11, 15))

    # metrics panel under the wrist image
    y = 380
    draw.text((930, y), "phase", font=F_LBL, fill=DIM)
    draw.text((1010, y), f"{phase}", font=F_TXT, fill=TEXT)
    y += 30
    draw.text((930, y), "sim t", font=F_LBL, fill=DIM)
    draw.text((1010, y), f"{sim_t:5.2f} s", font=F_TXT, fill=TEXT)
    y += 30
    draw.text((930, y), "cube z", font=F_LBL, fill=DIM)
    draw.text((1010, y), f"{cube_mm:6.1f} mm", font=F_TXT, fill=TEXT)
    y += 30
    draw.text((930, y), "contacts", font=F_LBL, fill=DIM)
    draw.text((1010, y), f"{ncon}", font=F_TXT, fill=GREEN if ncon else RED)

    draw.rectangle((0, H - 6, W, H), fill=(30, 38, 48))
    draw.rectangle((0, H - 6, int(W * idx / max(total - 1, 1)), H), fill=BLUE)
    draw.text(
        (30, H - 30),
        "dofbot_grasp · MuJoCo · rendered from the same actions the harness runs",
        font=F_SMALL,
        fill=DIM,
    )
    return img


def card(title: str, lines: list[tuple[str, str]]) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((80, 200), title, font=font(SANS_B, 46), fill=TEXT)
    draw.rectangle((80, 280, 220, 286), fill=GREEN)
    for i, (line, colour) in enumerate(lines):
        draw.text((80, 340 + i * 38), line, font=F_TXT, fill=colour)
    return img


def main() -> int:
    model = load_model()
    good = run_clip(model, fault=False, out_dir=OUT_DIR)
    fault = run_clip(model, fault=True, out_dir=OUT_DIR)
    print(f"good: lifted {good['lifted_mm']:.1f} mm, {len(good['frames'])} frames")
    print(f"fault: lifted {fault['lifted_mm']:.1f} mm, {len(fault['frames'])} frames")

    frames: list[Image.Image] = []
    frames += [
        card(
            "DOFBOT SE: wrist-camera grasp",
            [
                ("MuJoCo model derived from the real URDF chain", TEXT),
                ("camera on link5, looking down the tool axis", TEXT),
                ("the same IK + action sequence the harness runs", DIM),
            ],
        )
    ] * (FPS * 3)

    total = len(good["frames"]) + len(fault["frames"])
    idx = 0
    for img, phase, t, cube_mm, ncon in good["frames"]:
        frames.append(decorate(img, phase, t, cube_mm, ncon, ("GRASP", "green"), idx, total))
        idx += 1
    for img, phase, t, cube_mm, ncon in fault["frames"]:
        frames.append(
            decorate(img, phase, t, cube_mm, ncon, ("FAULT: fingers open", "red"), idx, total)
        )
        idx += 1

    frames += [
        card(
            "Result",
            [
                (f"grasp   : lifted {good['lifted_mm']:.1f} mm  -> PASS", GREEN),
                (f"fault   : lifted {fault['lifted_mm']:.1f} mm  -> FAIL", RED),
                ("harness gate: exit 0 / exit 1", BLUE),
                ("contract: cube_height >= 20 mm, contact_count >= 1", DIM),
            ],
        )
    ] * (FPS * 4)

    POSTER.parent.mkdir(parents=True, exist_ok=True)
    frames[len(frames) // 3].save(POSTER)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(MP4),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    assert proc.stdin is not None
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"wrote {MP4} ({MP4.stat().st_size / 1e6:.1f} MB, {len(frames) / FPS:.1f}s)")
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
