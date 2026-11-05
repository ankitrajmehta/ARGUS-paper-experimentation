"""Example images and texts for the report: one E1 item, one robot frame shown in every variant, E2c train/test signs."""
import json
from pathlib import Path

from PIL import Image

OUT = Path(__file__).with_name("figs") / "examples"
OUT.mkdir(parents=True, exist_ok=True)
A = Path("E:/Argus/outputs")
K = A / "kaggle"
FRAME = 87  # test frame (unseen house) where the ARGUS-style sign hijacked the robot
TRAIN_FRAME = 0


def shrink(src: Path, name: str, size: int = 560) -> None:
    image = Image.open(src).convert("RGB")
    image.thumbnail((size, size))
    image.save(OUT / f"{name}.jpg", quality=86)


e1_dir = K / "e1_v3/outputs/e1_3b_fp16_n100/data/val/images"
shrink(e1_dir / "2_clean.png", "e1_clean")
shrink(e1_dir / "2_poison.png", "e1_poison")

e2 = A / "e2/data/test/images"
for variant in ["clean", "injected_strip", "injected_sign", "benign_sign"]:
    shrink(e2 / f"{FRAME}_{variant}.png", f"e2_{variant}")
for variant in ["attack_no_trigger", "attack_plain", "legit_imperative"]:
    shrink(A / f"e2b_local/data/test/images/{FRAME}_{variant}.png", f"e2b_{variant}")
for variant in ["attack", "legit"]:
    shrink(A / f"e2c_localtest/data/test/images/{FRAME}_{variant}.png", f"e2c_test_{variant}")
for variant in ["attack_1", "attack_2", "legit"]:
    shrink(A / f"e2c_localtest/data/train/images/{TRAIN_FRAME}_{variant}.png", f"e2c_train_{variant}")

# Texts and the robot's actual answers (Kaggle, Qwen2.5-VL-3B fp16) for the same frame.
b2 = {s["id"]: s for s in json.load(open(K / "e2_v2/outputs/e2_3b_fp16/results/e2_behavior.json", encoding="utf-8"))["samples"]}[FRAME]
bb = {s["id"]: s for s in json.load(open(K / "e2b_v1/outputs/e2b_3b_fp16/results/e2b_behavior.json", encoding="utf-8"))["samples"]}[FRAME]
bc = {s["id"]: s for s in json.load(open(K / "e2c_v1/outputs/e2c_3b_fp16/results/e2c_behavior.json", encoding="utf-8"))["samples"]}[FRAME]
t2 = {r["id"]: r for r in json.load(open(A / "e2/data/e2_test.json", encoding="utf-8"))}[FRAME]
train_c = {r["id"]: r for r in json.load(open(K / "e2c_v1/outputs/e2c_3b_fp16/data/e2c_train.json", encoding="utf-8"))}[TRAIN_FRAME]
summary = {
    "frame": FRAME, "scene": t2["scene_name"], "task": t2["task"], "injected_text": t2["injected_text"], "benign_text": t2["benign_text"],
    "e2_responses": b2["responses"], "e2b_texts": bb["texts"], "e2b_responses": bb["responses"],
    "e2c_test_signs": bc["signs"], "e2c_test_responses": bc["responses"],
    "e2c_train_task": train_c["task"], "e2c_train_signs": train_c["signs"],
}
json.dump(summary, open(OUT / "examples.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))
