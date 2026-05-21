# UI-TARS-2B-SFT diagnostic harness

Goal: figure out why the off-the-shelf UI-TARS-2B-SFT model gets only ~27% action-type
accuracy on our AndroidControl harness, when the [UI-TARS paper](https://arxiv.org/abs/2501.12326)
reports ~67% step accuracy on the same split.

The gap is *us*, not the model. This folder is a harness for finding which
lever in our inference pipeline matters.

## Files

| file | what it does |
|---|---|
| `prompts.py`     | Four candidate system-prompt + action-grammar variants we try. |
| `harness.py`     | `TogglableUITars` agent — single class with knobs for prompt template, coord scale, image preprocessing, prev-actions serialization. |
| `compare_runs.py` | CLI. Runs each variant on a single fixed AndroidControl test step and writes `reports/ui_tars_runs.json` + `reports/ui_tars_table.md`. |
| `diagnostic.ipynb` | Interactive notebook version of `compare_runs.py`. Loads the model once, lets you tweak knobs cell-by-cell. |
| `reports/`        | Ablation outputs. Inspect `ui_tars_table.md` for the side-by-side table. |

## Initial findings (single step: episode 19349 step 2, `click(0.546, 0.077)`)

After running `python3 compare_runs.py --prompt-only`:

| variant | action emitted | type? | click err |
|---|---|---|---|
| `A_current` (no box markers) | `open_app(M&S)` | ✗ | — |
| `B_official_box_markers` | `click(0.89, 0.96)` | ✓ | 0.887 |
| `C_minimal` | parser-trip → `wait` | ✗ | — |
| **`D_generic_ui_tars`** | `click(0.23, 0.08)` | ✓ | **0.316** |

**The big takeaways:**

1. **Box markers matter — a lot.** Variants without `<|box_start|>...<|box_end|>`
   (A, C) cause the model to either degrade (`A → open_app` hallucination) or
   collapse entirely (`C → repeated "assistant" tokens`). Those markers are
   evidently part of UI-TARS-SFT's chat-template-level training, not just
   convention.

2. **The Chinese-`Thought` instruction matters.** Variants B and D both produced
   Chinese reasoning (`左键单击页面顶部中央的搜索栏...`) — that matches what
   UI-TARS was trained on. Variant C (which omitted that instruction) collapsed.

3. **The generic desktop-grammar variant (D) beats the AndroidControl-style
   grammar (B).** This is counterintuitive but real: D's `click + scroll + type +
   hotkey + drag` action space — even though half those verbs make no sense
   on Android — got the model to emit a structurally valid click pointing at
   the right *region* of the screen. B emitted `(892, 964)` (bottom-right of
   the phone) while D emitted `(230, 79)` (top of the screen, near where the
   search bar actually lives).

4. **The model's *thought* on variant D was correct, but its emitted
   coordinates aren't.** It thought: "左键单击页面顶部中央的搜索栏" ("left-click
   the search bar in the top center of the page"). Its Y coordinate (79/1000 =
   0.079) is **off by 0.002** from the GT (0.077) — essentially perfect. Its X
   coordinate (230/1000 = 0.23) is off by 0.316 — it placed the click on the
   left side instead of center. **The model perceives the right element but
   misaligns the X.**

5. **The Y-coord precision strongly suggests our 0-1000 coordinate scale is
   correct.** If the scale were wrong, we'd see both axes off. The X-only error
   points at one of:
   - Aspect-ratio handling inside the Qwen2-VL processor when the input is
     extremely tall (288×640 we used after resize, original 1080×2400)
   - Internal letterboxing / padding shifting the X mapping
   - Model genuinely confused about *which* search bar — `prev_actions=[]` means
     it doesn't know an app is already open

## Next levers to pull (not yet ablated)

The notebook scaffolds these. Each is one cell:

- **Image size**: drop `max_side`, let Qwen2-VL processor handle resize.
  AndroidControl screenshots are 1080×2400 — feeding the full image may give
  the model finer X resolution.
- **prev_actions=chat_history**: serialize past actions as alternating user/
  assistant turns instead of a single concatenated string. UI-TARS was probably
  trained this way; current `string` mode could be OOD.
- **Larger context** by sending prev screenshots too (not yet implemented in
  the harness; would need a new knob).

## How to reproduce

```bash
cd Session2/ui-tars
python3 compare_runs.py                # full 16-config grid (~5 min)
python3 compare_runs.py --prompt-only  # just the 4 prompts (~1 min)
jupyter notebook diagnostic.ipynb      # interactive mode
```

The notebook is the easier way to iterate — model loads once (~30 s on MPS),
then each cell takes ~5–10 s.

## How to fold the winning config back into the main deck

Once a config in the notebook reliably scores ≥ 50% type-acc on a multi-step
episode:

1. Edit `Session2/_act_ui/infer.py::UITarsAgent`:
   - Replace `UITARS_SYSTEM` with `prompts.PROMPTS["<winning variant>"]`
   - Update `predict_step()` to use the winning `prev_actions_mode`
   - Adjust `MAX_SIDE` if a non-default value won
2. Rerun `Session2/_act_ui/demo_episodes.py --agents uitars --n_episodes 2`
3. Re-enable UI-TARS in `Session2/build_one_deck.py` by adding it back to the
   `AGENTS` list and re-adding the `slide_7_uitars` + `slide_8_eval` calls in
   `build()`.
4. Re-run `python3 build_one_deck.py`.

## What I already verified before this notebook

- Model is producing real varied output (not a parser bug). On ep 19349 it
  chose `open_app` on step 0 then `navigate_back × 6`. That collapse points at
  prompt / prev-actions misalignment.
- 1000-scale coordinate parse is correct.
- `Qwen2VLForConditionalGeneration` + `AutoProcessor` from
  `bytedance-research/UI-TARS-2B-SFT` loads cleanly on MPS.

## Open question worth pursuing

The Y-coord precision (0.079 vs 0.077) versus the X-coord error (0.23 vs 0.546)
on the *same* step is the most concrete clue. If you find time after the
presentation, run `compare_runs.py --prompt-only` on a slide whose GT click is
on the **right** half of the screen. If the model still under-shoots X
consistently, the bug is in the Qwen2-VL aspect-ratio handling, not the prompt.
If X works fine when GT is on the right, the issue is task-specific (model
confused about the screen state, fixable with proper `prev_actions`).
