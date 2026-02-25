# 信息摘要服务 (digest_info)

从指定网络源搜索/拉取信息，总结后推送到微信（企业微信机器人）。  
**每个信息类别都有单独的逻辑、prompt 和推送时间**，例如：每天 9:15 推送最近 24 小时科技新闻，每周五推送「信息茧房破圈」类内容。新增类别时可用 **模板** 快速复制，降低开发量。

## 目录结构

```
digest_info/
├── core/
│   ├── source.py             # 搜索源
│   ├── summarizer.py         # 总结器（支持按类别 prompt）
│   ├── notifier.py           # 推送渠道
│   ├── category.py           # 信息类别抽象
│   └── schedule.py           # 每类别推送时间判断
├── categories/
│   ├── _template/            # 【模板】复制此目录可快速新增类别
│   │   ├── logic.py
│   │   ├── prompt.txt
│   │   └── README.md
│   ├── hackernews/
│   └── rss/
├── config.example.yaml
└── run.py
```

## 使用步骤

1. **安装依赖**（项目根目录）  
   `pip install -r digest_info/requirements.txt`

2. **配置**  
   - 复制 `config.example.yaml` 为 `config.yaml`  
   - 为每个类别设置 `params` 和 **schedule**（推送时间）  
   - 填写企业微信 Webhook 或环境变量 `WECHAT_WEBHOOK_URL`

3. **运行方式**  
   - 由系统定时任务在固定时刻执行（推荐）：例如每天 9:15、每周五 9:15 各执行一次 `python -m digest_info.run`，程序会根据每个类别的 `schedule` 只推送「当前该推」的类别。  
   - 或手动执行：会推送所有「当前时刻符合 schedule」的类别。

## 每类别推送时间（schedule）

在 `config.yaml` 里每个类别可单独设置：

```yaml
- id: hackernews
  params: { feed: top, limit: 10 }
  schedule:
    time: "09:15"       # 每天 9:15 推送
    window: "24h"       # 内容时间窗口（最近 24h），会以 window_hours 传入 fetch

- id: cocoon_break
  params: { feed_url: "https://...", limit: 20 }
  schedule:
    time: "09:15"
    weekday: 4          # 仅周五（0=周一 … 6=周日）
    window: "7d"        # 最近一周
```

- **time**：每日推送时刻，如 `"09:15"`。  
- **weekday**：仅该星期几推送（0=周一 … 6=周日）。  
- **days**：多日时用列表，如 `[0,1,2,3,4]` 表示周一至周五。  
- **window**：可选，`"24h"` / `"7d"`，会以 `window_hours` 传入该类别的 `fetch(**params)`，便于按时间过滤内容。  
- 配置项 **timezone**（如 `Asia/Shanghai`）用于判断「当前是否该推送」。

## 定制信息：逻辑 + prompt

- **逻辑**：`categories/<类别名>/logic.py`，实现 `fetch()`（可复用 `sources/`）。  
- **Prompt**：同目录 `prompt.txt`，会作为该类别的展示说明出现在摘要里，**直接改即可定制**。

### 用模板快速新增类别

1. 复制 `categories/_template` 整个目录，重命名为新类别 id（如 `cocoon_break`）。  
2. 在 `logic.py` 里改类名、id，实现 `fetch()`；在 `prompt.txt` 里写该类别的说明或筛选要求。  
3. 在 `categories/__init__.py` 里增加 `from . import cocoon_break`。  
4. 在 `config.yaml` 的 `categories` 里添加一项，并设置 **schedule**（推送时间）和 **params**。

详见 `categories/_template/README.md`。

## 其他扩展

- **总结方式**：在 `summarizers/` 实现 `Summarizer.summarize(..., category_id=..., prompt=..., **kwargs)`，可据此做 LLM 总结等。
- **推送渠道**：在 `notifiers/` 实现 `Notifier` 并注册即可。

## 微信说明

使用 **企业微信群机器人** Webhook 推送。个人微信可后续扩展 Server 酱、Bark 等 Notifier。
