# UI-TARS-2B-SFT ablation on ep 19349 step 2

**Goal:** I want to purchase a striped shirt for myself for the upcoming office party. Look for a striped form
**GT action:** `click(x=0.546, y=0.077)`

| prompt | max_side | prev_actions | pred | type? | click err | sec |
|---|---|---|---|---|---|---|
| A_current | 896 | string | `open_app` | ✗ | — | 5.81 |
| B_official_box_markers | 896 | string | `click(0.89, 0.96)` | ✓ | 0.887 | 8.82 |
| C_minimal | 896 | string | `wait` | ✗ | — | 14.53 |
| D_generic_ui_tars | 896 | string | `click(0.23, 0.08)` | ✓ | 0.316 | 5.83 |

## Raw model output per variant

### A_current · max_side=896 · prev_actions=string
```
Thought: Open the M&S app.
Action: open_app(app_name='M&S')
```

### B_official_box_markers · max_side=896 · prev_actions=string
```
Thought: 在手机底部的快捷栏中，我轻点了底部右侧的“More”按钮，以打开更多选项菜单。该按钮位于快捷栏的最右侧，图标为三个水平排列的白色圆点。
Action: click(start_box='(892,964)')
```

### C_minimal · max_side=896 · prev_actions=string
```
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
assistant
```

### D_generic_ui_tars · max_side=896 · prev_actions=string
```
Thought: 左键单击页面顶部中央的搜索栏，激活搜索栏。
Action: click(start_box='(230,79)')
```
