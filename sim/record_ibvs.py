"""Record the IBVS grasp run frame by frame: wrist view beside the observer view.

The pipeline is driven through ``ibvs_sim.run_once`` (so the recording cannot drift from
the code being measured). Frames are taken by wrapping ``mujoco.mj_step``: capturing only
at phase boundaries gave ten frames, far too few to see what the arm actually does.
"""

from __future__ import annotations

import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, "/home/toby/Desktop/dofbot-grasp/sim")

import ibvs_sim as S  # noqa: E402

OUT = "/home/toby/Desktop/dofbot-grasp/sim/ibvs_sim.mp4"
SEED = (0.020, 0.140)
W, H = 480, 360
EVERY = 200         # sim steps per captured frame (dt = 2 ms -> 0.4 s)
FPS = 10.0


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W * 2, H))
    model = mujoco.MjModel.from_xml_path("/home/toby/Desktop/dofbot-grasp/sim/dofbot_se.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=S.IMG_H, width=S.IMG_W)
    state = {"n": 0, "phase": "start"}

    def grab():
        views = []
        for cam in ("wrist_cam", "overview"):
            renderer.update_scene(data, camera=cam)
            img = cv2.resize(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR), (W, H))
            views.append(img)
        t = data.time
        a = label(views[0], f"wrist  {state['phase']}")
        b = label(views[1], f"observer  t={t:.1f}s")
        writer.write(np.hstack([a, b]))

    real_step = mujoco.mj_step

    def step_with_capture(m, d, nstep=1):
        real_step(m, d, nstep)
        state["n"] += 1
        if state["n"] % EVERY == 0:
            grab()

    # run_once builds its own model/data/renderer; hand it ours
    S.mujoco.MjModel.from_xml_path = lambda _p: model
    S.mujoco.MjData = lambda _m: data
    S.mujoco.Renderer = lambda _m, height=S.IMG_H, width=S.IMG_W: renderer
    S.mujoco.mj_step = step_with_capture

    def shot(tag):
        state["phase"] = tag
        grab()

    try:
        res = S.run_once(SEED, verbose=True, shot=shot)
    finally:
        S.mujoco.mj_step = real_step
    writer.release()
    print("result:", res.get("ok"), res.get("lift_mm"), res.get("reason", ""))
    print(f"wrote {OUT}  ({state['n']} sim steps)")


if __name__ == "__main__":
    main()
