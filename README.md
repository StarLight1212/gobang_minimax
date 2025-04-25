# 五子棋AI对战项目（Python实现）

本项目是一个使用 Python 实现的五子棋（Gomoku）人机对战程序，采用了 Minimax 搜索算法并配有 Tkinter 图形用户界面（GUI）。该项目改写自原有的 JavaScript 实现(https://github.com/lihongxun945/gobang/tree/master)，保留了大部分核心逻辑，并在结构和语法上进行了 Python 化处理。

---

## ⚙️ 配置与常量

- `shapes`：定义棋型，如 `FIVE`, `FOUR`, `THREE_THREE` 等。
- `config`：包含得分权重配置，用于评估不同棋型的价值。
- `SCORES`：将棋型映射到分数。

---

## 🧠 评估类（Evaluate）

用于棋局状态管理与评估，包括以下功能：

### 核心结构

- 使用二维列表（list of lists）表示棋盘，边界加1便于处理。
- `shape_cache`：缓存每个空位在不同方向上形成的棋型，提高效率。
- `black_scores` 与 `white_scores`：分别表示当前黑白双方对各空位的潜在得分。

### 核心方法

#### move / undo
- 下棋与撤销功能，同时更新历史记录与分数缓存。

#### check_win
- 检查最近一步是否形成五连，存储获胜连线。

#### is_game_over / get_winner
- 判断是否结束游戏，返回赢家。

#### _update_scores_around
- 对最近一步周围的空位进行重新评估。

#### _update_single_point_score
- 模拟某个空位落子，获取所有方向的棋型，计算其潜在得分并更新缓存。

#### evaluate_board
- 返回当前玩家与对手得分差值，若形成五连则直接返回极大/极小值。

#### get_valuable_moves
- 返回按进攻与防守优先级排列的高价值落子点，包括：
  - 立即取胜
  - 阻止对方五连
  - 四连、三三等强攻棋型

#### get_empty_points
- 获取所有空位（无权重排序）

#### board_hash
- 将当前棋盘状态转换为 hashable 元组，用于缓存。

#### get_shape_fast / count_shape
- 快速判断某一方向是否形成特定棋型。翻译自 JS 逻辑，保持核心思路，但部分条件有所简化以提升可读性。

---

## 🗂️ 缓存类（Cache）

- 使用 `dict` 存储缓存，配合 `collections.deque` 实现先进先出（FIFO）缓存淘汰策略。
- 用于加速 Minimax 过程中对重复局面的评分。

---

## 🤖 Minimax 搜索（带 Alpha-Beta 剪枝）

- 实现为 Negamax 变体，更加简洁。
- 加入 Alpha-Beta 剪枝以提升性能。
- 使用缓存避免重复评估。
- 终止条件包括：胜负已分、达到搜索深度限制。
- 调用 `get_valuable_moves` 实现启发式排序，提高搜索效率。

---

## 🔍 find_best_move 函数

- 顶层函数，调用 Minimax 进行搜索。
- 若为空棋盘，则默认走中心点。
- 可记录搜索用时、节点数等调试信息。

---

## 🖼️ 图形界面（GomokuGUI，基于 Tkinter）

### UI 组成

- 主窗口、棋盘画布、状态标签、重置按钮等。

### 核心功能

#### draw_board / draw_pieces
- 绘制棋盘网格与落子。
- 高亮最近一步与胜利连线。

#### handle_click
- 响应玩家点击落子。
- 调用 AI 落子逻辑。

#### trigger_ai_move / perform_ai_move
- AI 思考过程。
- 使用 `after()` 避免 GUI 卡顿。

#### check_game_over
- 判断游戏结束，弹出提示。

#### reset_game
- 重置棋局与界面。

---

## ▶️ 运行方式

1. 保存为 `gomoku_ai_gui.py`。
2. 确保已安装 Python 3（默认包含 Tkinter）。
3. 在终端运行：

```bash
python gomoku_ai_gui.py
```
