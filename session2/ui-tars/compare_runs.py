"""
Run a single AndroidControl test step through every combination of UI-TARS
configuration knobs, print raw output + parsed action + match vs GT.

This is the terminal-friendly version of diagnostic.ipynb.

Usage:
    python3 compare_runs.py                       # default step
    python3 compare_runs.py --episode 19349 --step 2
    python3 compare_runs.py --prompt-only         # ablate only prompts, keep other knobs default

Outputs:
    reports/ui_tars_runs.json   — every (variant, raw, parsed, time) tuple
    reports/ui_tars_table.md    — markdown comparison table

The goal: find the configuration that produces a click roughly at the GT
coordinates for at least one of the click-bearing test steps.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_act_ui"))
from harness import TogglableUITars  # noqa: E402
from prepare_data import iter_steps_with_images  # noqa: E402

OUT = HERE / "reports"
OUT.mkdir(exist_ok=True)


def pick_step(eid: int, step_idx: int) -> dict:
    """Load one specific (episode_id, step_id) from the AndroidControl test split."""
    steps = list(iter_steps_with_images("test", ep_filter={eid}))
    steps.sort(key=lambda s: s["step_id"])
    if step_idx >= len(steps):
        raise IndexError(f"episode {eid} has only {len(steps)} steps")
    return steps[step_idx]


def gt_summary(step: dict) -> str:
    t = step["type_name"]
    if t in ("click", "long_press"):
        x, y = step["xy"]
        return f"{t}(x={x:.3f}, y={y:.3f})"
    if t == "scroll":
        return f"scroll(direction={step.get('scroll_dir_name','?')})"
    if t == "input_text":
        return f"input_text(text='{(step.get('text_input') or '')[:30]}')"
    if t == "open_app":
        return f"open_app(app_name='{step.get('app_name','?')}')"
    return f"{t}()"


def click_err(pred: dict, gt_xy: tuple[float, float]) -> float | None:
    if pred["action_type"] not in ("click", "long_press"): return None
    px = pred["params"].get("x"); py = pred["params"].get("y")
    if px is None or py is None: return None
    return max(abs(px - gt_xy[0]), abs(py - gt_xy[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=19349)
    ap.add_argument("--step",    type=int, default=2,
                    help="step index inside episode (0-based)")
    ap.add_argument("--prompt-only", action="store_true",
                    help="only sweep the 4 prompt variants, keep other knobs default")
    args = ap.parse_args()

    step = pick_step(args.episode, args.step)
    img = Image.fromarray(step["img"]).convert("RGB")
    instruction = step["instruction"]

    print(f"=== test step ===")
    print(f"  episode  : {args.episode}")
    print(f"  step idx : {args.step}")
    print(f"  goal     : {instruction[:100]}")
    print(f"  GT action: {gt_summary(step)}")
    print(f"  image    : {img.size}")
    print()

    # define the knob grid
    if args.prompt_only:
        knobs = [{"prompt_variant": p,
                  "coord_scale": 1000.0,
                  "prev_actions_mode": "string",
                  "image_size": None,
                  "max_side": 896}
                 for p in ("A_current", "B_official_box_markers",
                           "C_minimal", "D_generic_ui_tars")]
    else:
        # smaller ablation grid: 4 prompts × 2 max_side × 2 prev_actions_mode = 16 runs
        knobs = []
        for p in ("A_current", "B_official_box_markers",
                  "C_minimal", "D_generic_ui_tars"):
            for ms in (896, None):
                for pa in ("string", "none"):
                    knobs.append({"prompt_variant": p,
                                  "coord_scale": 1000.0,
                                  "prev_actions_mode": pa,
                                  "image_size": None,
                                  "max_side": ms})

    print(f"running {len(knobs)} variants ... model loads once on first call.\n")

    # Share the model across variants by reusing the same agent instance.
    agent = TogglableUITars(**knobs[0])
    agent.load()
    runs = []
    for i, k in enumerate(knobs):
        # mutate config; model+proc already loaded
        agent.prompt_variant   = k["prompt_variant"]
        agent.coord_scale      = k["coord_scale"]
        agent.prev_actions_mode= k["prev_actions_mode"]
        agent.max_side         = k["max_side"]
        # (image_size + model_id changes would require a reload — skip in ablation)
        result = agent.predict_step_raw(img, instruction, prev_actions=[])
        type_ok = result["parsed"]["action_type"] == step["type_name"]
        c_err = click_err(result["parsed"],
                          tuple(step["xy"])) if step["type_name"] in ("click", "long_press") else None
        runs.append({
            **k,
            "raw":          result["raw"],
            "parsed":       result["parsed"],
            "sec":          result["sec"],
            "img_size":     list(result["img_size"]),
            "type_match":   type_ok,
            "click_err":    c_err,
        })
        flag = "OK" if type_ok and (c_err is None or c_err <= 0.14) else (".." if type_ok else "X ")
        print(f"  [{flag}] {k['prompt_variant']:<24s} max_side={str(k['max_side']):<5s} "
              f"prev={k['prev_actions_mode']:<8s} "
              f"→ {result['parsed']['action_type']:<14s} "
              f"err={c_err if c_err is None else f'{c_err:.3f}'}")

    # save raw runs JSON
    (OUT / "ui_tars_runs.json").write_text(json.dumps({
        "step": {"episode": args.episode, "step_idx": args.step,
                 "instruction": instruction, "gt_type": step["type_name"],
                 "gt_xy": list(step["xy"]) if step["type_name"] in ("click", "long_press") else None,
                 "img_size": list(img.size)},
        "runs": runs,
    }, indent=2, default=str))

    # markdown table
    lines = [f"# UI-TARS-2B-SFT ablation on ep {args.episode} step {args.step}",
             "",
             f"**Goal:** {instruction[:100]}",
             f"**GT action:** `{gt_summary(step)}`",
             "",
             "| prompt | max_side | prev_actions | pred | type? | click err | sec |",
             "|---|---|---|---|---|---|---|"]
    for r in runs:
        ce = "—" if r["click_err"] is None else f"{r['click_err']:.3f}"
        pred_str = r["parsed"]["action_type"]
        if r["parsed"]["action_type"] in ("click", "long_press") and "x" in r["parsed"]["params"]:
            pred_str += f"({r['parsed']['params']['x']:.2f}, {r['parsed']['params']['y']:.2f})"
        lines.append(f"| {r['prompt_variant']} | {r['max_side']} | "
                     f"{r['prev_actions_mode']} | `{pred_str}` | "
                     f"{'✓' if r['type_match'] else '✗'} | {ce} | {r['sec']:.2f} |")
    lines += [
        "",
        "## Raw model output per variant",
        "",
    ]
    for r in runs:
        lines += [f"### {r['prompt_variant']} · max_side={r['max_side']} · "
                  f"prev_actions={r['prev_actions_mode']}",
                  "```",
                  r["raw"].rstrip(),
                  "```",
                  ""]
    (OUT / "ui_tars_table.md").write_text("\n".join(lines))

    print(f"\nwrote {OUT}/ui_tars_runs.json  +  {OUT}/ui_tars_table.md")


if __name__ == "__main__":
    main()
