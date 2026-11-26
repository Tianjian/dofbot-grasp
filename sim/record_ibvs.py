"""Record the IBVS grasp run: wrist view (the feedback) beside the observer view.

The frames come from ``ibvs_sim.run_once`` itself via its ``shot`` callback, so the
recording shows exactly the pipeline that is measured -- an earlier version duplicated
the phases and drifted from it.
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


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (W * 2, H))
    # one shared model/data pair, driven through the pipeline
    model = mujoco.MjModel.from_xml_path("/home/toby/Desktop/dofbot-grasp/sim/dofbot_se.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=S.IMG_H, width=S.IMG_W)

    def shot(tag):
        for cam, half in (("wrist_cam", "wrist"), ("overview", "observer")):
            renderer.update_scene(data, camera=cam)
            img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (W, H))
            img = label(img, f"{half}: {tag}")
            if cam == "wrist_cam":
                wrist = img
            else:
                observer = img
        for _ in range(2):
            writer.write(np.hstack([wrist, observer]))

    import types
    mod = types.SimpleNamespace()
    mod.model, mod.data, mod.renderer = model, data, renderer
    # monkey-patch the module-level objects run_once builds internally
    orig = S.mujoco.MjModel.from_xml_path

    def patched(_path):
        return model

    S.mujoco.MjModel.from_xml_path = patched
    S.mujoco.MjData = lambda _m: data
    S.mujoco.Renderer = lambda _m, height=240, width=320: renderer
    try:
        res = S.run_once(SEED, verbose=True, shot=shot)
    finally:
        S.mujoco.MjModel.from_xml_path = orig
    writer.release()
    print("result:", res.get("ok"), res.get("lift_mm"), res.get("reason", ""))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
