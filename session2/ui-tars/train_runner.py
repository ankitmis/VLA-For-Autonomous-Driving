"""
CLI runner for train.ipynb — paper-style UI-TARS-2B-SFT evaluation on
AndroidControl test. Same logic as the notebook cells, but executable in
background.

Usage:
    python3 train_runner.py --n 20            # quick sanity check
    python3 train_runner.py --n 100           # bigger stratified subset
    python3 train_runner.py --n -1            # full test set (~75-150 min on MPS)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_act_ui"))

from harness import TogglableUITars
from prepare_data import iter_steps_with_images, get_split_metadata

OUT = HERE / "reports"
OUT.mkdir(exist_ok=True)


def stratified_sample(test_meta, n: int):
    if n < 0 or n >= len(test_meta):
        return list(test_meta)
    by_type = defaultdict(list)
    for s in test_meta:
        by_type[s["type_name"]].append(s)
    per_type = max(1, n // len(by_type))
    sample = []
    for t, items in by_type.items():
        sample.extend(random.sample(items, min(per_type, len(items))))
    return sample[:n]


def gt_to_action_str(step):
    t = step["type_name"]
    if t in ("click", "long_press"):
        x = int(round(step["xy"][0] * 1000))
        y = int(round(step["xy"][1] * 1000))
        return f"{t}(start_box='<|box_start|>({x},{y})<|box_end|>')"
    if t == "scroll":
        return (f"scroll(start_box='<|box_start|>(500,500)<|box_end|>', "
                f"direction='{step.get('scroll_dir_name','down')}')")
    if t == "input_text":
        return f"type(content='{(step.get('text_input') or '')[:40]}')"
    if t == "open_app":
        return f"open_app(app_name='{step.get('app_name','')}')"
    if t == "navigate_back": return "press_back()"
    if t == "navigate_home": return "press_home()"
    if t == "wait":          return "wait()"
    if t == "status":        return "finished()"
    return f"{t}()"


def score(pred, gt_step):
    t_gt = gt_step["type_name"]
    t_pr = pred["action_type"]
    type_ok = (t_pr == t_gt)
    out = {"type_ok": type_ok, "click_ok": None, "scroll_ok": None,
           "step_ok": type_ok}
    if t_gt in ("click", "long_press"):
        gx, gy = gt_step["xy"]
        px = pred["params"].get("x"); py = pred["params"].get("y")
        out["click_ok"] = (type_ok and px is not None and py is not None
                           and max(abs(px - gx), abs(py - gy)) <= 0.14)
        out["step_ok"] = out["click_ok"]
    elif t_gt == "scroll":
        gd = gt_step.get("scroll_dir_name")
        pd_ = pred["params"].get("direction")
        out["scroll_ok"] = (type_ok and gd == pd_)
        out["step_ok"] = out["scroll_ok"]
    return out


def full_eval(n: int = -1, *, seed: int = 42,
              prompt_variant: str = "D_generic_ui_tars",
              prev_actions_mode: str = "chat_history",
              max_side=None) -> dict:
    random.seed(seed)
    print(f"=== loading AndroidControl test metadata ===")
    meta = get_split_metadata()
    test_meta = meta["test"]
    print(f"  {len(test_meta)} test steps total")
    sample = stratified_sample(test_meta, n)
    print(f"  stratified sample: N={len(sample)}  "
          f"types={dict(Counter(s['type_name'] for s in sample))}")

    sample_eids = set(s["episode_id"] for s in sample)
    print(f"=== loading images for {len(sample_eids)} episodes ===")
    ep_steps = defaultdict(list)
    for s in iter_steps_with_images("test", ep_filter=sample_eids):
        ep_steps[s["episode_id"]].append(s)
    for eid in ep_steps:
        ep_steps[eid].sort(key=lambda s: s["step_id"])
    print(f"  loaded {sum(len(v) for v in ep_steps.values())} steps with images")

    print(f"=== loading UI-TARS-2B-SFT (~30 s) ===")
    agent = TogglableUITars(
        prompt_variant=prompt_variant,
        coord_scale=1000.0,
        prev_actions_mode=prev_actions_mode,
        max_side=max_side,
    )
    agent.load()
    print(f"  loaded. device={agent.model.device}")

    print(f"=== running eval ===")
    results = []
    progress = OUT / "paper_repro_progress.json"
    t0 = time.time()
    for i, target in enumerate(sample):
        eid = target["episode_id"]; sid = target["step_id"]
        all_steps = ep_steps[eid]
        target_step = next(s for s in all_steps if s["step_id"] == sid)
        prev = [gt_to_action_str(s) for s in all_steps if s["step_id"] < sid]
        img = Image.fromarray(target_step["img"]).convert("RGB")
        out = agent.predict_step_raw(
            img, target_step["instruction"], prev_actions=prev)
        sc = score(out["parsed"], target_step)
        results.append({
            "episode_id": eid, "step_id": sid,
            "gt_type":   target_step["type_name"],
            "pred_type": out["parsed"]["action_type"],
            **sc, "sec": out["sec"],
        })
        if (i + 1) % 5 == 0 or (i + 1) == len(sample):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(sample) - i - 1)
            n_ok = sum(r["type_ok"] for r in results)
            print(f"  {i+1:4d}/{len(sample)}  "
                  f"type_acc_so_far={n_ok/(i+1):.3f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")
            progress.write_text(json.dumps(results, indent=2, default=str))

    elapsed = time.time() - t0
    n_total = len(results)
    n_type = sum(r["type_ok"] for r in results)
    n_step = sum(r["step_ok"] for r in results)
    click_rows = [r for r in results if r["gt_type"] in ("click", "long_press")]
    n_click_ok = sum(1 for r in click_rows if r["click_ok"])
    scroll_rows = [r for r in results if r["gt_type"] == "scroll"]
    n_scroll_ok = sum(1 for r in scroll_rows if r["scroll_ok"])

    summary = {
        "n":             n_total,
        "type_acc":      n_type / n_total,
        "step_success":  n_step / n_total,
        "click_acc_14":  n_click_ok / len(click_rows) if click_rows else None,
        "scroll_acc":    n_scroll_ok / len(scroll_rows) if scroll_rows else None,
        "n_click":       len(click_rows),
        "n_scroll":      len(scroll_rows),
        "elapsed_s":     elapsed,
        "sec_per_step":  elapsed / n_total if n_total else 0,
        "config": {
            "prompt_variant": prompt_variant,
            "prev_actions_mode": prev_actions_mode,
            "max_side": max_side,
            "coord_scale": 1000.0,
        },
    }
    (OUT / "paper_repro_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    (OUT / "paper_repro_results.json").write_text(
        json.dumps(results, indent=2, default=str))

    print(f"\n=== UI-TARS-2B-SFT on AndroidControl test (N={n_total}) ===")
    print(f"  type_acc        : {summary['type_acc']:.3f}")
    print(f"  step_success    : {summary['step_success']:.3f}")
    if summary["click_acc_14"] is not None:
        print(f"  click@14% (where GT is click): {summary['click_acc_14']:.3f}"
              f"  (n={summary['n_click']})")
    if summary["scroll_acc"] is not None:
        print(f"  scroll dir match              : {summary['scroll_acc']:.3f}"
              f"  (n={summary['n_scroll']})")
    print(f"  avg seconds / step            : {summary['sec_per_step']:.2f}")
    print(f"  wrote {OUT}/paper_repro_summary.json")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20,
                    help="stratified sample size (-1 = full 916-step set)")
    ap.add_argument("--prompt", default="D_generic_ui_tars",
                    help="one of: A_current, B_official_box_markers, C_minimal, D_generic_ui_tars")
    ap.add_argument("--prev",   default="chat_history",
                    help="string | chat_history | none")
    ap.add_argument("--max_side", default="none",
                    help="int or 'none'")
    args = ap.parse_args()
    ms = None if args.max_side == "none" else int(args.max_side)
    full_eval(args.n, prompt_variant=args.prompt,
              prev_actions_mode=args.prev, max_side=ms)
