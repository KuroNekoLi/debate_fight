## 辯論模擬與評估引擎

這是一個以 Python 實作、採 Clean Architecture 設計的「奧瑞岡賽制多代理辯論模擬系統」。系統會讀取既有論證單位，讓正反雙方 AI 代理人自行選擇論點，完整跑完奧瑞岡辯論流程，最後由多裁判系統評分並輸出統計報告。

### 系統功能

- 從 JSON 載入 Argument Units（論證單位）。
- 模擬完整奧瑞岡賽制流程，包含六位辯士：
  - 正一 / 正二 / 正三
  - 反一 / 反二 / 反三
- 每場辯論開始前，強制雙方各自選擇：
  - 3 個主軸論點
  - 2 個防守論點
- 整場辯論只能使用已選出的 5 個論點，避免 AI 自由發揮亂講。
- 每段發言都要求符合 Argument Chain：
  - Claim（主張）
  - Reason（理由）
  - Example（生活例子）
  - Impact（影響）
- 使用三裁判系統：
  - 邏輯裁判
  - 說服裁判
  - 結構裁判
- 支援多場模擬，例如 100 場。
- 最終輸出：
  - 正反勝率
  - 各論點勝率
  - 各論點被擊敗率
  - 正方 / 反方 Top 3 論點
  - 每場評審講評
  - Key Clash（關鍵勝負衝突）
  - 最佳論點組合

---

### 專案結構

- `src/debate_sim/config.py`：環境變數與執行設定
- `src/debate_sim/domain/`：核心資料模型與奧瑞岡流程
- `src/debate_sim/application/engine.py`：辯論模擬主流程
- `src/debate_sim/infrastructure/llm.py`：LLM 抽象層，支援 mock / OpenAI / Gemini
- `src/debate_sim/infrastructure/repository.py`：論證單位讀取
- `src/debate_sim/prompts/templates.py`：角色 prompt、選論點 prompt、裁判 prompt
- `data/arguments.json`：論證單位輸入範例
- `run.py`：執行入口
- `results/simulation_report.json`：模擬後產生的統計報告

---

### 輸入格式

`data/arguments.json` 中每個論證單位格式如下：

```json
{
  "id": "P_U01",
  "side": "pro",
  "claim": "最低工資最先傷到的不是已經有工作的人，而是還沒進場、最需要第一份工作的人。",
  "warrant": "當新人一開始就必須用固定底薪聘用，雇主會更偏好熟手。",
  "impact": "就業入口變窄，最需要第一份工作的弱勢反而被制度排除。",
  "attack_points": ["研究可能顯示就業效果有限", "可用青年補助或分級工資緩解"],
  "strength_score": 8.9
}
```

欄位說明：

- `id`：論證單位 ID，例如 `P_U01`、`C_U40`
- `side`：立場，`pro` 代表正方，`con` 代表反方
- `claim`：主張
- `warrant`：主張成立的理由或因果機制
- `impact`：此論點重要的原因
- `attack_points`：此論點可能被攻擊的地方
- `strength_score`：論點強度分數，供選論點時參考

---

### 快速開始

1. 安裝套件：

```bash
pip install -r requirements.txt
```

如果你的環境只有 `python3` / `pip3`，可以改用：

```bash
pip3 install -r requirements.txt
```

2. 複製環境設定檔：

```bash
cp .env.example .env
```

3. 執行 100 場模擬：

```bash
python3 run.py --topic "我國應廢除最低工資制度" --num-runs 100
```

4. 查看輸出：

- 終端機會顯示摘要
- 完整結果會寫入 `results/simulation_report.json`

---

### LLM Provider 設定

在 `.env` 中設定：

```bash
DEBATE_LLM_PROVIDER=mock
DEBATE_MODEL=gpt-4o-mini
DEBATE_MAX_TOKENS=1200
DEBATE_REQUEST_TIMEOUT=60
DEBATE_MAX_RETRIES=5
DEBATE_RETRY_INITIAL_DELAY=1
DEBATE_RETRY_MAX_DELAY=30
DEBATE_MAX_WORKERS=20
OPENAI_API_KEY=
GEMINI_API_KEY=
```

可用 provider：

- `mock`：預設模式，不需要 API key，適合測試完整流程與統計功能。
- `openai`：使用 OpenAI API，需要設定 `OPENAI_API_KEY`。
- `gemini`：使用 Gemini API，需要設定 `GEMINI_API_KEY`。

OpenAI 範例：

```bash
DEBATE_LLM_PROVIDER=openai
DEBATE_MODEL=gpt-4o-mini
OPENAI_API_KEY=你的_API_KEY
```

Gemini 範例：

```bash
DEBATE_LLM_PROVIDER=gemini
DEBATE_MODEL=gemini-1.5-flash
GEMINI_API_KEY=你的_API_KEY
```

`mock` 模式是固定種子的模擬結果，適合確認系統是否能跑通。若要產生更自然的辯論稿與裁判講評，請改用 OpenAI 或 Gemini。

大批次 OpenAI 模擬可以調高並行數，但建議先從 10 到 20 開始：

```bash
DEBATE_MAX_WORKERS=20
```

數字越高，完成速度越快，但也越容易碰到 API rate limit，成本也會更快累積。若要跑 100 場以上，建議不要直接把 workers 設成 100，除非你的 API token rate limit 足夠高。

OpenAI provider 內建 rate limit / timeout 自動重試：

```bash
DEBATE_MAX_RETRIES=5
DEBATE_RETRY_INITIAL_DELAY=1
DEBATE_RETRY_MAX_DELAY=30
```

重試採指數退避加 jitter；若單次請求持續失敗，該場仍會被記錄到 `failed_runs`，不會中止整批模擬。

預設選點採混合模式。以 100 場為例：

- 30 場 `free`：AI 完全自由選 3 個主軸與 2 個防守。
- 50 場 `coverage`：每場強制正反各 1 個論點進入組合，其餘由 AI 補齊，用來避免冷門論點永遠不出場。
- 20 場 `stress_combo`：指定 2 到 3 個主軸壓測組合化學反應，其餘由 AI 補齊。

每場的 `selection_plan` 會寫入 `results/simulation_report.json`。

併行執行時，單場 API timeout、rate limit 或 JSON 解析失敗不會中止整批模擬；系統會先依重試設定自動退避重試，若仍失敗，才把該場記錄到 `failed_runs`，其他場次會繼續完成。部分結果會持續寫入 `results/simulation_report.partial.json`。

如果模型服務偶爾卡住，可以調整單次請求等待秒數：

```bash
DEBATE_REQUEST_TIMEOUT=60
```

---

### 核心 API

主流程方法：

```python
DebateSimulationEngine.run_simulation(topic, num_runs=100)
```

回傳的 report 會包含：

- `wins`：正反勝場數
- `completed_win_rate`：只用成功完成場次計算的正反勝率；若有失敗場，解讀時應優先看這個欄位。
- `completed_runs`：成功完成評分的場次
- `failed_runs_count`：失敗場次數
- `failed_runs`：失敗場次、錯誤類型與錯誤訊息
- `win_rate`：正反勝率
- `argument_win_rate`：各論點勝率
- `argument_beaten_rate`：各論點被擊敗率
- `top3_pro_arguments`：正方 Top 3 論點
- `top3_con_arguments`：反方 Top 3 論點
- `top_key_clashes`：最常出現的關鍵衝突
- `best_argument_combinations`：最佳論點組合
- `selection_mode_summary`：自由、覆蓋探索、組合壓測三種模式的各自勝率
- `detailed_stats.mode_argument_stats`：各模式下每個論點的使用、勝敗、勝率與敗率。
- `detailed_stats.mode_combo_stats`：各模式下每組 5 論點組合的使用、勝敗與勝率。
- `detailed_stats.required_argument_summary`：coverage / stress_combo 指定論點的指定次數、進場率與勝率。
- `detailed_stats.required_argument_results`：每次指定論點是否成功進場、進主軸或防守、該場是否勝出。
- `detailed_stats.judge_vote_patterns`：三裁判票型分布，例如 `pro2-con1`。
- `detailed_stats.judge_type_votes`：邏輯裁判、說服裁判、結構裁判各自投票分布。
- `detailed_stats.run_index`：逐場索引，包含模式、勝方、票型、正反論點組合、Key Clash。
- `detailed_stats.failed_by_type`：失敗場次的錯誤類型統計。
- `judge_comments`：每場裁判講評
- `debate_runs`：每場辯論完整資料

---

### 單場辯論資料

每個 `debate_runs[]` 會包含：

- `pro_selection`：正方選出的 3 個主軸論點與 2 個防守論點
- `con_selection`：反方選出的 3 個主軸論點與 2 個防守論點
- `selection_plan`：該場使用的選點模式與指定進場論點
- `transcript`：完整奧瑞岡流程逐輪發言
- `judge_scores`：三位裁判的評分與講評
- `final_winner`：該場勝方
- `key_clash`：該場關鍵勝負衝突
- `turning_point`：裁判認定的轉折點
- `best_argument`：該場最佳論點
- `worst_argument`：該場最弱或最容易被打穿的論點

Markdown 報告 `results/simulation_report.md` 會輸出完整分析表格，包括選點模式摘要、失敗場次摘要、裁判票型、覆蓋探索論點表、各模式論點表、各模式組合表、Top 論點、Key Clash、逐場索引，以及最佳申論 / 質詢答辯 / 結辯全文摘錄。若要做後續統計或圖表，建議以 JSON 為準；若要人工閱讀與討論，建議看 Markdown。

---

### 奧瑞岡流程

系統目前採用以下順序：

1. 正一申論 → 反二質詢
2. 反一申論 → 正三質詢
3. 正二申論 → 反三質詢
4. 反二申論 → 正一質詢
5. 正三申論 → 反一質詢
6. 反三申論 → 正二質詢
7. 正方結辯
8. 反方結辯

每位辯士都有獨立角色定位：

- 正一：立論與定義
- 正二：攻防轉換
- 正三：收束與比較
- 反一：破題
- 反二：質詢與拆解
- 反三：總攻與壓制

---

### 設計原則

- 辯論必須「用論點打架」，不能讓 AI 自由發揮。
- 每場辯論都必須先選論點，並且只能使用該場選出的論點。
- 每段發言必須有 Claim、Reason、Example、Impact。
- 系統重點不是產生漂亮文字，而是找出哪些論證單位最容易贏、最容易被打爆。
- 所有結果都必須可回溯、可統計、可比較。

---

### 已內建的範例資料

目前 `data/arguments.json` 內建 54 個最低工資辯題的論證單位：

- 正方 27 個
- 反方 27 個

這些論點來自前面整理過的 Argument Units，已轉成系統需要的結構化欄位。
