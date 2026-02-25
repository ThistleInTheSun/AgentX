# 新类别模板

复制本目录（`_template`）为新的类别目录后，按下面步骤即可新增一个信息类别，降低重复开发量。

## 1. 复制并重命名

- 复制整个 `_template` 文件夹，重命名为你的类别 id（仅小写、数字、下划线），例如：
  - `cocoon_break` — 信息茧房破圈
  - `tech_news` — 科技新闻
  - `weekly_digest` — 周报

## 2. 实现逻辑

- 打开 `logic.py`：
  - 把 `MyCategory` / `my_category` 换成你的类名和 id。
  - 在 `fetch()` 里实现拉取逻辑（可复用 `digest_info.sources` 里的 Source，或自己写请求）。
  - 若需要「最近 24h/7d」过滤，可从 `kwargs.get("window_hours")` 读取（由配置里 `schedule.window` 传入）。

## 3. 写 prompt

- 编辑同目录下的 `prompt.txt`，写该类别的展示说明或筛选要求（会出现在摘要块里）。

## 4. 注册类别

- 在 `digest_info/categories/__init__.py` 里增加一行：
  - `from . import 你的目录名`  
  （例如 `from . import cocoon_break`）

## 5. 本类别的 config.yaml

在本类别目录下把 `config.example.yaml` 复制为 `config.yaml`，填写 **params** 和 **schedule**（只对本类别生效），例如：

```yaml
params:
  feed_url: "https://example.com/feed.rss"
  limit: 20

schedule:
  time: "09:15"
  weekday: 4    # 仅周五（0=周一 … 6=周日）
  window: "7d"
```

- **schedule.time**：推送时刻。**schedule.weekday** / **schedule.days**：星期几。**schedule.window**：可选，会以 `window_hours` 传入 `fetch(**params)`。

## 6. 在主 config 里启用

在项目根目录的 `digest_info/config.yaml` 的 `categories` 列表里加上本类别的 id，例如：`categories: [hackernews, cocoon_break]`。

完成以上步骤后，运行 `python -m digest_info.run` 时会在对应推送时间自动执行该类别并推送。
