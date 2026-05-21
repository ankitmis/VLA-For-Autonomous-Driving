"""
Togglable UI-TARS agent for the diagnostic notebook.

The shipped `Session2/_act_ui/infer.py::UITarsAgent` bakes in one specific
prompt + parsing + preprocessing choice. This class exposes each of those as a
constructor knob so the notebook can do ablation runs without editing source.

Usage from the notebook:
    from harness import TogglableUITars
    a = TogglableUITars(prompt_variant="B_official_box_markers",
                        coord_scale=1000.0,
                        prev_actions_mode="chat_history",
                        image_size=None,
                        max_side=896)
    out = a.predict_step_raw(pil_screenshot, instruction, prev_actions=[...])
    # out is {"raw": "...", "parsed": {...}, "sec": float}
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from prompts import PROMPTS


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
START_BOX_RE = re.compile(
    r"start_box='(?:<\|box_start\|>)?\((\d+)\s*,\s*(\d+)\)(?:<\|box_end\|>)?'")
END_BOX_RE = re.compile(
    r"end_box='(?:<\|box_start\|>)?\((\d+)\s*,\s*(\d+)\)(?:<\|box_end\|>)?'")
ACTION_RE = re.compile(r"Action:\s*(\w+)\(")
DIRECTION_RE = re.compile(r"direction='(up|down|left|right)'")
TYPE_CONTENT_RE = re.compile(r"(?:type|input_text)\(content='([^']*)'")
APP_NAME_RE = re.compile(r"app_name='([^']*)'")
HOTKEY_RE = re.compile(r"hotkey\(key='([^']*)'")


def parse_action(text: str, coord_scale: float = 1000.0) -> dict:
    """Parse raw UI-TARS output into canonical action dict.
    coord_scale: divisor to convert model's coord output → [0,1].
                 1000.0 for 0-1000 normalized; pass img_w/img_h for raw pixel.
    """
    am = ACTION_RE.search(text)
    if not am:
        return {"action_type": "wait", "params": {}, "parse_ok": False,
                "reason": "no Action: line"}

    verb = am.group(1)

    def _xy():
        m = START_BOX_RE.search(text)
        if not m: return None, None
        return float(m.group(1)) / coord_scale, float(m.group(2)) / coord_scale

    if verb == "click":
        x, y = _xy()
        if x is None:
            return {"action_type": "click", "params": {"x": 0.5, "y": 0.5},
                    "parse_ok": False, "reason": "coord parse failed"}
        return {"action_type": "click",
                "params": {"x": round(x, 4), "y": round(y, 4)},
                "parse_ok": True}
    if verb == "long_press":
        x, y = _xy()
        x = 0.5 if x is None else x; y = 0.5 if y is None else y
        return {"action_type": "long_press",
                "params": {"x": round(x, 4), "y": round(y, 4)},
                "parse_ok": True}
    if verb in ("type", "input_text"):
        m = TYPE_CONTENT_RE.search(text)
        return {"action_type": "input_text",
                "params": {"text": m.group(1) if m else ""},
                "parse_ok": True}
    if verb == "scroll":
        d = DIRECTION_RE.search(text)
        return {"action_type": "scroll",
                "params": {"direction": d.group(1) if d else "down"},
                "parse_ok": True}
    if verb in ("press_back", "navigate_back"):
        return {"action_type": "navigate_back", "params": {}, "parse_ok": True}
    if verb in ("press_home", "navigate_home"):
        return {"action_type": "navigate_home", "params": {}, "parse_ok": True}
    if verb == "open_app":
        m = APP_NAME_RE.search(text)
        return {"action_type": "open_app",
                "params": {"app_name": m.group(1) if m else ""},
                "parse_ok": True}
    if verb == "hotkey":
        m = HOTKEY_RE.search(text)
        key = m.group(1) if m else ""
        if key in ("back", "ENTER", "enter"): mapped = "navigate_back"
        elif key in ("home", "HOME"):         mapped = "navigate_home"
        else:                                  mapped = "wait"
        return {"action_type": mapped,
                "params": {"hotkey": key}, "parse_ok": True}
    if verb == "wait":
        return {"action_type": "wait", "params": {}, "parse_ok": True}
    if verb in ("finished", "call_user", "done", "completed"):
        return {"action_type": "status",
                "params": {"goal_status": "successful"},
                "parse_ok": True}
    return {"action_type": "wait", "params": {},
            "parse_ok": False, "reason": f"unsupported verb: {verb}"}


# ---------------------------------------------------------------------------
# togglable agent
# ---------------------------------------------------------------------------
@dataclass
class TogglableUITars:
    prompt_variant: str = "A_current"
    coord_scale: float = 1000.0          # 1000 for [0,1000]; img_w for raw px
    prev_actions_mode: str = "string"    # "string" | "chat_history" | "none"
    image_size: dict | None = None       # forwarded to AutoProcessor(size=...)
    max_side: int | None = 896           # pre-resize cap; None = no pre-resize
    model_id: str = "bytedance-research/UI-TARS-2B-SFT"
    device: str = "auto"                 # "auto" → mps if available else cpu
    max_new_tokens: int = 128

    model: object = None  # lazily loaded
    proc:  object = None

    def load(self):
        if self.model is not None: return
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        if self.image_size is None:
            self.proc = AutoProcessor.from_pretrained(self.model_id)
        else:
            self.proc = AutoProcessor.from_pretrained(self.model_id,
                                                     size=self.image_size)
        if self.device == "auto":
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            dev = self.device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map=dev).eval()

    def _resize(self, pil):
        if self.max_side is None: return pil
        w, h = pil.size
        if max(w, h) <= self.max_side: return pil
        if w >= h:
            return pil.resize((self.max_side, int(h * self.max_side / w)),
                              Image.BILINEAR)
        return pil.resize((int(w * self.max_side / h), self.max_side),
                          Image.BILINEAR)

    def _build_messages(self, pil, instruction, prev_actions):
        sys_prompt = PROMPTS[self.prompt_variant]

        if self.prev_actions_mode == "none" or not prev_actions:
            user_text = f"Task: {instruction}"
        elif self.prev_actions_mode == "string":
            # join previous actions into a single string
            if isinstance(prev_actions, list):
                pa = " | ".join(str(a) for a in prev_actions)
            else:
                pa = str(prev_actions)
            user_text = f"Task: {instruction}\nPrevious actions: {pa}"
        else:  # chat_history — one user-turn per past step + the screenshot
            user_text = f"Task: {instruction}"

        if self.prev_actions_mode == "chat_history" and isinstance(prev_actions, list):
            msgs = [{"role": "system",
                     "content": [{"type": "text", "text": sys_prompt}]}]
            for pa in prev_actions:
                msgs += [{"role": "user",
                          "content": [{"type": "text", "text": f"Task: {instruction}"}]},
                         {"role": "assistant",
                          "content": [{"type": "text", "text": str(pa)}]}]
            msgs.append({"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": user_text}]})
            return msgs

        return [
            {"role": "system",
             "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": user_text}]},
        ]

    def predict_step_raw(self, screenshot: Image.Image, instruction: str,
                         prev_actions=None) -> dict:
        """Returns {"raw": <str>, "parsed": <dict>, "sec": <float>,
                    "img_size": (w,h)}"""
        self.load()
        pil = self._resize(screenshot.convert("RGB"))
        msgs = self._build_messages(pil, instruction, prev_actions)
        prompt = self.proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[prompt], images=[pil], padding=True,
                           return_tensors="pt").to(self.model.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        sec = time.time() - t0
        raw = self.proc.batch_decode(
            out[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True)[0]

        # coord_scale of "imgpx" means: divide by image dims rather than fixed 1000
        if self.coord_scale == "imgpx":
            # rebuild parser with image-pixel scale per axis: we need separate x/y
            # — for simplicity, use the larger dim as the divisor on both
            scale = max(pil.size)
        else:
            scale = float(self.coord_scale)
        parsed = parse_action(raw, coord_scale=scale)

        return {"raw": raw, "parsed": parsed, "sec": sec,
                "img_size": pil.size}
