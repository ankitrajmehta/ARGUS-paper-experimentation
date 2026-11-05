import sys, time
from ai2thor.controller import Controller
from ai2thor.platform import Linux64
t = time.time()
c = Controller(platform=Linux64, scene="FloorPlan1", width=640, height=640, fieldOfView=90)
print("start s", round(time.time() - t, 1), "renderer", sys.argv[1])
t = time.time()
for _ in range(10):
    e = c.step(action="RotateRight")
print("step s", round((time.time() - t) / 10, 3), "ok", e.metadata["lastActionSuccess"], e.frame.shape)
from PIL import Image
Image.fromarray(e.frame).save(f"/mnt/e/Argus/thor/smoke_{sys.argv[1]}.png")
c.stop()
