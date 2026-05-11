# 新辯題套用流程 Agent Prompt

這份文件用來處理「換一個辯題」時，如何複製本專案已驗證過的工作流：先產生論證單位，再讓 API 用這些論證單位進行多場奧瑞岡辯論模擬，最後依結果回頭優化論點池與 prompt。

核心原則：

1. 先建立高品質論證單位，不要直接叫 AI 自由辯論。
2. 每個論證單位要能獨立上場，也要能和其他論點形成組合。
3. 模擬時必須限制 AI 只能使用已選論點，避免亂講。
4. 多場模擬的目的不是產生漂亮辯稿，而是找出最強論點、最弱論點、關鍵衝突與最佳組合。
5. 每一輪模擬後，要根據裁判理由修正弱勢方，而不是只看勝率。

---

## 一、給 Agent 的總任務 Prompt

你是一位資深辯論論證設計師、賽制模擬工程師與多代理 AI 系統優化者。

現在我要把一個新辯題放進既有的「奧瑞岡賽制多代理辯論模擬系統」。請你不要直接產生一場辯論，而是依序完成以下工作：

1. 針對新辯題建立正反雙方論證單位池。
2. 將論證單位整理成可供系統讀取的 `data/arguments.json`。
3. 使用既有模擬引擎跑小規模測試。
4. 分析勝率、裁判理由、Top 論點、被擊敗率與最佳組合。
5. 回頭優化弱勢方論點與 prompt。
6. 重複測試，直到正反雙方大致均勢，或清楚知道哪一方論點池仍較弱。
7. 最後跑 100 場正式模擬，輸出完整解讀報告。

請遵守：

- 不要讓 AI 在辯論中自由新增論點。
- 所有發言必須來自已選出的 Argument Units。
- 每場辯論前，正反雙方各選 3 個主軸論點與 2 個防守論點。
- 每個論點都要能追蹤勝率、使用率與被擊敗率。
- 每輪優化都要根據模擬結果，而不是憑直覺大改。
- 修改檔案前先確認資料來源與現有結構。

---

## 二、第一階段：產生論證單位

請先針對辯題：

```text
【在此填入新辯題】
```

建立正反雙方 Argument Units。

建議規模：

- 正方 20 到 30 個
- 反方 20 到 30 個
- 若辯題很複雜，可先各 15 個，模擬後再擴充

每個 Argument Unit 必須包含：

- `id`
- `side`
- `claim`
- `warrant`
- `impact`
- `attack_points`
- `strength_score`
- `detail`

請使用以下品質標準：

- Claim 要是完整口語句，不要只寫「應支持」或「不應支持」。
- Warrant 要說清楚因果機制。
- Impact 要明確說明為什麼這點會影響勝負。
- attack_points 要列出對方最可能攻擊的地方。
- detail 要寫成可上台講的自然段落，不要只是資料堆疊。
- 每個論點都要有生活化例子。
- 正反雙方都要有：
  - 主攻論點
  - 防守論點
  - 反打對方核心的 clash 論點
  - 價值框架論點
  - 可行性或制度設計論點

論證單位 JSON 範例：

```json
{
  "id": "P_U01",
  "side": "pro",
  "claim": "正方主張……，因為……。",
  "warrant": "這個主張成立，是因為……。",
  "impact": "這會影響本場勝負，因為……。",
  "attack_points": [
    "反方可能攻擊……",
    "反方可能說……",
    "此論點最怕被質疑……"
  ],
  "strength_score": 8.5,
  "detail": "這段寫成可直接上場使用的口語論證……"
}
```

ID 命名規則：

- 正方：`P_U01`, `P_U03`, `P_U05` ...
- 反方：`C_U02`, `C_U04`, `C_U06` ...
- 不要重複 ID。
- `side` 僅使用 `pro` 或 `con`。

---

## 三、第二階段：建立 data/arguments.json

請把產生好的論證單位轉成：

```text
data/arguments.json
```

檢查項目：

- JSON 可以被 Python 正常讀取。
- 論點總數正確。
- ID 不重複。
- 正反方數量大致平衡。
- 每個單位都有 `claim / warrant / impact / attack_points / strength_score / detail`。
- `detail` 不要空白。
- 不要把 Markdown 格式直接塞進 JSON 造成解析失敗。

可用檢查指令：

```bash
python3 - <<'PY'
import json
from collections import Counter

with open("data/arguments.json", encoding="utf-8") as f:
    data = json.load(f)

ids = [x["id"] for x in data]
sides = Counter(x["side"] for x in data)

print("total:", len(data))
print("unique ids:", len(set(ids)))
print("sides:", sides)

required = ["id", "side", "claim", "warrant", "impact", "attack_points", "strength_score", "detail"]
bad = []
for item in data:
    for key in required:
        if key not in item or item[key] in ("", [], None):
            bad.append((item.get("id"), key))

print("missing fields:", bad[:20])
PY
```

---

## 四、第三階段：先跑小規模模擬

不要一開始就跑 100 場。

建議順序：

1. 先跑 3 場，確認流程沒有壞。
2. 再跑 10 場，看勝率是否極端。
3. 再跑 30 場，看模式分布與裁判理由。
4. 最後才跑 100 場。

指令：

```bash
python3 run.py --topic "【在此填入新辯題】" --num-runs 10
```

若要正式跑 100 場：

```bash
python3 run.py --topic "【在此填入新辯題】" --num-runs 100
```

`.env` 建議：

```bash
DEBATE_LLM_PROVIDER=openai
DEBATE_MODEL=gpt-4o-mini
DEBATE_MAX_WORKERS=20
DEBATE_MAX_RETRIES=5
DEBATE_RETRY_INITIAL_DELAY=1
DEBATE_RETRY_MAX_DELAY=30
OPENAI_API_KEY=你的_API_KEY
```

注意：

- `rate_limit_exceeded` 可以靠重試與降低 workers 處理。
- `insufficient_quota` 代表帳戶額度或 billing 不足，重試沒有用。
- 如果 100 場途中出現大量失敗，先停下來，不要硬跑。

---

## 五、第四階段：解讀結果

每次跑完請檢查：

- 總勝率
- `free / coverage / stress_combo` 各模式勝率
- 裁判票型
- 三類裁判投票：
  - logic
  - persuasion
  - structure
- 正反 Top10 論點
- 使用率高但勝率低的論點
- 使用率低但勝率高的論點
- 最佳組合
- Key Clash
- 每場裁判講評

解讀原則：

- 只看 100% 勝率很危險，因為使用次數可能太低。
- 使用次數高、勝率仍高的論點才是真的穩。
- 使用次數高、勝率低的論點通常是弱勢方需要重寫的地方。
- `free` 模式最接近 AI 自然策略。
- `coverage` 模式用來測冷門論點是否有潛力。
- `stress_combo` 模式用來測指定核心組合是否有化學反應。

判讀範例：

```text
如果總勝率是 60:40，通常代表一方小幅優勢，但還不算系統失衡。
如果是 75:25 以上，通常代表論點池或裁判 prompt 已明顯偏向一方。
如果 free 接近均勢，但 stress_combo 明顯偏一方，代表核心組合強度不平衡。
如果 logic 裁判偏一方，表示因果鏈或可行性框架有問題。
如果 persuasion 裁判偏一方，表示語言、例子、價值 framing 有問題。
如果 structure 裁判偏一方，表示論點排列、攻防收束或結辯比較有問題。
```

---

## 六、第五階段：迭代優化論點池

每輪優化請遵守：

1. 先找弱勢方最常輸的 Key Clash。
2. 再找弱勢方使用率高但勝率低的論點。
3. 不要一次大改所有論點。
4. 優先修改 3 到 5 個核心單位。
5. 每次修改後重新同步 `data/arguments.json`。
6. 再跑 10 或 30 場確認方向。

優化時要問：

- 弱勢方是不是只提出價值口號，沒有制度接管能力？
- 弱勢方是不是只攻擊對方，沒有回答裁判最擔心的問題？
- 弱勢方是不是缺少一句能改變裁判判準的核心 framing？
- 強勢方是不是有一個預設直覺沒有被反打？
- 某些論點是不是太依賴其他論點，獨立性不足？

重寫格式建議：

```text
請重寫【論點 ID】。

重寫目標：
- 這個論點要專門反打對方的【核心論點】。
- Claim 要改成完整句。
- Warrant 要說清楚因果機制。
- Impact 要直接連到本場勝負。
- detail 要寫成可上台講的口語段落。
- 必須加入生活化例子。
- 必須主動承認對方最強攻擊，再反轉。
- 不要套模板，要重新思考這個論點本身怎麼贏。
```

---

## 七、第六階段：調整 prompt，而不是只改論點

如果一方論點已經很完整，但模擬仍長期輸，可能不是論點池問題，而是裁判或發言 prompt 的判準問題。

可以調整：

- `src/debate_sim/prompts/templates.py`

常見需要調整的地方：

1. 選點 prompt
   如果 AI 永遠只選特定幾個論點，就要強化 `coverage` 或指定不同組合。

2. 發言 prompt
   如果辯士有論點但講不出來，要把核心 framing 寫進正反方發言規則。

3. 質詢 prompt
   如果質詢沒有形成攻防，要要求「質詢者問、答辯者答」並設計追問鏈。

4. 裁判 prompt
   如果裁判預設某方價值較高，要要求裁判正面比較雙方判準，不可只回到直覺。

重要提醒：

- 不要讓 prompt 直接指定某一方應該贏。
- 可以要求裁判比較特定判準。
- 可以要求辯士使用某種 framing。
- 但仍要讓勝負取決於場上表現。

---

## 八、第七階段：正式 100 場測試

當 10 場與 30 場都跑通後，再跑 100 場。

正式報告至少要回答：

- 正反勝率是否接近均勢？
- 哪一方仍有小幅優勢？
- 優勢來自論點池、prompt，還是裁判偏好？
- 正方 Top10 是哪些？
- 反方 Top10 是哪些？
- 哪些論點使用率高但勝率低？
- 哪些組合最常導向勝利？
- 哪些 Key Clash 最常決定勝負？
- 下一輪應該強化哪一方、哪幾個論點？

判斷標準：

- 50:50 到 55:45：高度均勢。
- 56:44 到 62:38：一方小幅優勢，可接受，但可繼續微調。
- 63:37 到 70:30：一方明顯優勢，應強化弱勢方。
- 70:30 以上：論點池或 prompt 很可能失衡。

---

## 九、可直接複製使用的完整 Agent Prompt

```text
你是一位資深辯論論證設計師、奧瑞岡賽制裁判、AI 多代理系統工程師。

我要把以下新辯題放進既有的辯論模擬系統：

【辯題】
（在此填入辯題）

請你依序完成：

一、建立論證單位池
- 正方 20 到 30 個 Argument Units。
- 反方 20 到 30 個 Argument Units。
- 每個單位包含 id, side, claim, warrant, impact, attack_points, strength_score, detail。
- 每個 claim 必須是完整句。
- 每個 warrant 必須說明因果機制。
- 每個 impact 必須連到本場勝負。
- detail 必須是可上台使用的口語論證，不要只列資料。
- 每個論點都要有生活化例子。
- 正反雙方都要有主攻、防守、反打、價值、制度可行性論點。

二、整理成 data/arguments.json
- 請直接修改專案中的 data/arguments.json。
- 不要新增多餘檔案，除非需要備份或說明文件。
- 檢查 JSON 可解析、ID 不重複、正反數量平衡。

三、跑小規模測試
- 先跑 3 場確認流程。
- 再跑 10 場看是否一方大勝。
- 若可行，再跑 30 場。
- 每次跑完都要讀 results/simulation_report.md 與 .json。

四、分析結果
- 回報總勝率。
- 回報 free / coverage / stress_combo 各模式勝率。
- 回報裁判票型與三類裁判投票。
- 回報正反 Top10 論點。
- 回報使用率高但勝率低的弱點論點。
- 回報最佳論點組合。
- 回報主要 Key Clash。

五、迭代優化
- 不要一次批量亂改。
- 根據數據挑 3 到 5 個最需要強化的弱勢方論點。
- 每個論點要重寫成可獨立上場、也能和組合產生化學反應。
- 如果問題是裁判判準或辯士發言方式，就調整 src/debate_sim/prompts/templates.py。
- 修改後同步 data/arguments.json。
- 再跑 10 或 30 場驗證。

六、正式測試
- 當 30 場結果大致合理後，跑 100 場。
- 最終輸出完整解讀。
- 如果勝率超過 70:30，請判斷是否論點池或 prompt 失衡。
- 如果勝率落在 55:45 到 62:38，請說明哪一方有小幅優勢，以及下一輪微調方向。

最重要原則：
- 辯論必須用論點打架。
- 不可讓 AI 自由發揮亂講。
- 每場都必須可分析。
- 系統目標不是產生漂亮辯稿，而是找最強論證、最弱論證和最佳論點組合。
```

---

## 十、建議的檔案命名

如果要保存每個辯題的論點池，建議：

```text
data/topics/
  minimum_wage_arguments.json
  nuclear_power_arguments.json
  death_penalty_arguments.json
```

但目前系統預設讀取：

```text
data/arguments.json
```

因此正式跑某一個辯題前，請把該辯題論點池放到 `data/arguments.json`，或在 `.env` 設定：

```bash
DEBATE_ARGUMENTS_PATH=data/topics/你的辯題_arguments.json
```

---

## 十一、實務提醒

- 每個新辯題都要先重新建立論點池，不要沿用舊辯題的 prompt framing。
- 可以沿用系統流程，但不要沿用某一方的勝負語言。
- 若新辯題是高度法律、醫療、金融或科技議題，論證單位產生時應加入資料查證步驟。
- 若新辯題本身有現行法或最新政策，請先查最新資料，再寫論點。
- 若模擬結果一方大勝，不要急著相信結果；先檢查論點池是否一方比較具體、另一方太抽象。
- 最後要保留人工判斷：模擬是用來找弱點，不是取代辯手與教練的判斷。
