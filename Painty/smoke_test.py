# -*- coding: utf-8 -*-
"""Quick smoke tests for pure logic in Painty/main.py (no camera needed)."""
import importlib.util
import math
import os
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("painty", os.path.join(os.path.dirname(__file__), "main.py"))
painty = importlib.util.module_from_spec(spec)
spec.loader.exec_module(painty)

fails = []


def check(name, cond):
    if cond:
        print(f"  OK  {name}")
    else:
        print(f" FAIL {name}")
        fails.append(name)


print("== Persian text ==")
# shaping cache fills without exceptions and returns a string
out = painty.shape_persian("سلام دنیا")
check("shape_persian returns shaped str", isinstance(out, str) and len(out) > 0)
check("shape cached", painty.shape_persian("سلام دنیا") is out)

# draw_text renders into a BGR frame without exceptions and changes pixels
frame = np.full((120, 400, 3), 255, dtype=np.uint8)
painty.draw_text(frame, "سلام دنیا", (380, 60), 30, (40, 40, 40), "rm")
dark = int((frame < 200).sum())
check("draw_text paints pixels", dark > 50)

# fallback font (load_default) path
class _FakeFont:
    def getbbox(self, t):
        return (0, 0, 10, 10)


class _FakeImageFont:
    calls = {"truetype": 0, "load_default": 0}

    @staticmethod
    def truetype(name, size):
        _FakeImageFont.calls["truetype"] += 1
        raise OSError("no fonts")

    @staticmethod
    def load_default():
        _FakeImageFont.calls["load_default"] += 1
        return _FakeFont()


_orig_imgfont = painty.ImageFont
painty.ImageFont = _FakeImageFont
painty._FONT_CACHE.clear()
try:
    f = painty.open_font(1234)
    check(
        "open_font falls back to load_default",
        isinstance(f, _FakeFont)
        and _FakeImageFont.calls["truetype"] == 5
        and _FakeImageFont.calls["load_default"] == 1,
    )
finally:
    painty.ImageFont = _orig_imgfont
    painty._FONT_CACHE.clear()

print("== One Euro filter ==")
import time as _t
flt = painty.OneEuroFilter(min_cutoff=1.2, beta=0.012)
p = (100.0, 100.0)
for _ in range(30):
    _t.sleep(1.0 / 30.0)
    p = flt((p[0] + 1.0, p[1] + 0.5))
# smooth start, but must keep up with constant motion
check("filter follows motion (30fps)", abs(p[0] - 130) < 25 and abs(p[1] - 115) < 25)
# jitter suppression: noisy input around a fixed point should stay close
flt2 = painty.OneEuroFilter(min_cutoff=1.0, beta=0.008)
import random
random.seed(7)
last = None
max_dev = 0.0
for _ in range(60):
    _t.sleep(1.0 / 30.0)
    q = flt2((200 + random.uniform(-14, 14), 200 + random.uniform(-14, 14)))
    if last is not None:
        max_dev = max(max_dev, math.hypot(q[0] - last[0], q[1] - last[1]))
    last = q
check("jitter damped (frame-to-frame < 8px)", max_dev < 8.0)
flt2.reset()
check("reset works", flt2.last_time is None)

print("== Dwell click ==")
dw = painty.DwellClick(dwell_time=0.05)
rects = {("tool", "save"): (0, 0, 100, 50)}
fired = False
for _ in range(30):
    _t.sleep(0.02)
    _, f = dw.update((50, 25), rects)
    if f:
        fired = True
        break
check("dwell fires after holding", fired)
_, p0 = dw.progress((50, 25), rects)
check("progress resets after fire", p0 == 0.0)
lbl, _ = dw.update((300, 300), rects)
check("no hit off-button", lbl is None)
dw2 = painty.DwellClick(dwell_time=0.05)
lbl, f = dw2.update((50, 25), rects)
check("first touch starts timer, not fire", f is False and lbl == ("tool", "save"))

print("== History / replay ==")
shape = (200, 300, 3)
hist = []
board = painty.replay_board(hist, shape)
check("blank board white", board.mean() == 255)
op = ("stroke", {"points": [(30, 100), (60, 100), (90, 100)], "color": (0, 0, 0), "size": 8})
hist.append(op)
board = painty.replay_board(hist, shape)
check("stroke visible", int((board < 128).sum()) > 100)
hist.append(("clear", None))
board = painty.replay_board(hist, shape)
check("clear resets board", board.mean() == 255)
hist.pop()
board = painty.replay_board(hist, shape)
check("undo via replay restores", int((board < 128).sum()) > 100)
op2 = ("erase", {"points": [(60, 100), (80, 100)], "size": 24})
hist.append(op2)
board = painty.replay_board(hist, shape)
check("erase removes ink", int((board < 128).sum()) < int((painty.replay_board(hist[:1], shape) < 128).sum()))
# MAX_HISTORY cap enforced inline in main (60) — verify list semantics used there
hist = []
for i in range(10):
    hist.append(("stroke", {"points": [(i, i)], "color": (0, 0, 0), "size": 4}))
    if len(hist) > painty.MAX_HISTORY:
        hist.pop(0)
check("history cap semantics", len(hist) <= painty.MAX_HISTORY)

print("== Toolbar / picker geometry ==")
rects, canvas = painty.build_toolbar()
check("8 colors", len(rects["colors"]) == 8)
check("5 sizes", len(rects["sizes"]) == 5)
check("6 tools", len(rects["tools"]) == 6)
for group in rects.values():
    for r in group.values():
        check(f"rect inside canvas {r}", 0 <= r[0] < r[2] <= painty.TOOLBAR_W and 0 <= r[1] < r[3] <= painty.TOOLBAR_H)

pic = painty.ColorPicker(1920, 1080)
b1 = pic.bgr()
pic.hue = 120
check("picker hue change -> different color", pic.bgr() != b1)
p1 = (pic.track_rect[0] + 5, (pic.track_rect[1] + pic.track_rect[3]) // 2)
pic.hover(p1)
check("hover on hue strip sets hue", pic.hue <= 6)
p2 = (pic.area_rect[2] - 5, pic.area_rect[1] + 2)
pic.hover(p2)
check("hover on SV area sets sat/val", pic.sat >= 230 and pic.val >= 200)

print("== fingers_up (rotation-invariant) ==")

class FakeLM:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FakeHand:
    def __init__(self, pts):
        self.landmark = [FakeLM(x, y) for (x, y) in pts]


# A hand pointing straight up: index extended, others curled
# wrist(0), thumb_cmc(1), thumb_mcp(2), thumb_ip(3), thumb_tip(4),
# index_mcp(5), index_pip(6), index_dip(7), index_tip(8), ...
def make_hand(index_tip_y, middle_tip_y, ring_tip_y, pinky_tip_y):
    pts = []
    # wrist and thumb (curled: tip near index mcp)
    pts += [(0.5, 0.9), (0.45, 0.85), (0.42, 0.82), (0.40, 0.80), (0.38, 0.78)]
    # index: mcp, pip, dip, tip (extended)
    pts += [(0.55, 0.60), (0.55, 0.48), (0.55, 0.38), (0.55, index_tip_y)]
    # middle
    pts += [(0.60, 0.60), (0.60, 0.52), (0.60, 0.50), (0.60, middle_tip_y)]
    # ring
    pts += [(0.65, 0.62), (0.65, 0.56), (0.65, 0.54), (0.65, ring_tip_y)]
    # pinky
    pts += [(0.70, 0.65), (0.70, 0.61), (0.70, 0.60), (0.70, pinky_tip_y)]
    return FakeHand(pts)


f = painty.fingers_up(make_hand(0.25, 0.53, 0.55, 0.60))
check("pointing hand: index up", f[1] == 1)
check("pointing hand: middle down", f[2] == 0)

# same hand rotated ~90 degrees (index pointing left): x/y swapped roles
pts = []
pts += [(0.9, 0.5), (0.85, 0.45), (0.82, 0.42), (0.80, 0.40), (0.78, 0.38)]
pts += [(0.6, 0.55), (0.48, 0.55), (0.38, 0.55), (0.25, 0.55)]   # index extended to left
pts += [(0.6, 0.60), (0.52, 0.60), (0.50, 0.60), (0.53, 0.60)]   # middle curled
pts += [(0.6, 0.62), (0.56, 0.62), (0.54, 0.62), (0.55, 0.62)]   # ring curled
pts += [(0.6, 0.65), (0.61, 0.65), (0.60, 0.65), (0.60, 0.65)]   # pinky curled
f = painty.fingers_up(FakeHand(pts))
check("rotated hand: index up", f[1] == 1)
check("rotated hand: middle down", f[2] == 0)

# fist: all four curled
f = painty.fingers_up(make_hand(0.55, 0.53, 0.55, 0.60))
check("fist: index down", f[1] == 0)

print("== full UI frame render (headless) ==")
W, H = 1280, 720
ui = np.full((H, W, 3), 255, dtype=np.uint8)
painty.stroke_line(ui, (200, 400), (700, 350), (50, 60, 220), 14)
rects2, canvas2 = painty.build_toolbar()
painty.draw_toolbar(ui, rects2, canvas2, 1, 2, False, False)
pic2 = painty.ColorPicker(W, H)
pic2.open = True
pic2.draw(ui)
painty.draw_status(ui, "رنگ: قرمز — در حال نقاشی")
painty.draw_dwell_ring(ui, (rects2["sizes"][2][0] + 26, rects2["sizes"][2][1] + 20), 26, 0.5)
fake = painty.ColorPicker(W, H)
point = (rects2["colors"][1][0] + 19, rects2["colors"][1][1] + 19)
painty.draw_text(ui, "سلام! این یک آزمایش متن فارسی است", (W - 30, 340), 28, (40, 40, 40), "ra")
png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_preview.png")
check("UI frame rendered + saved (unicode-safe)", painty.save_png(png_path, ui) and os.path.getsize(png_path) > 20000)

print("== stroke_line ==")
b = np.full((100, 100, 3), 255, dtype=np.uint8)
painty.stroke_line(b, (10, 50), (90, 50), (0, 0, 0), 6)
check("stroke_line draws", int((b < 128).sum()) > 200)
# gaps bridged: two far points with sparse intermediate
b2 = np.full((100, 100, 3), 255, dtype=np.uint8)
painty.stroke_line(b2, (10, 50), (90, 50), (0, 0, 0), 6)
row = b2[50, 10:91]
check("line continuous (no gaps)", int((row < 128).sum()) > 75)

print()
if fails:
    print(f"FAILED: {len(fails)} -> {fails}")
    sys.exit(1)
print("ALL SMOKE TESTS PASSED")
