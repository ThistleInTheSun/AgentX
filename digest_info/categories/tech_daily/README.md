# 每日科技摘要（tech_daily）

本类别在**每天上午 9:15** 推送**过去 24 小时**内的：

1. **科技圈重大新闻**：来自 TechCrunch、The Verge、Ars Technica 的 RSS  
2. **新论文**：arXiv CS 分类（cs.AI、cs.LG、cs.CL、cs.CV）当日提交的论文  

## 三步流程

- **第一步·确认信息源**：科技新闻来自上述三个站点的 RSS；论文来自 arXiv API（按 `submittedDate` 过滤 24h）。  
- **第二步·爬取**：`logic.py` 中聚合 `TechNewsFeedsSource` 与 `ArxivSource`，按 `schedule.window: "24h"` 过滤时间。  
- **第三步·总结**：由当前全局 `summarizer` 根据本目录下 `prompt.txt` 生成摘要；若主配置中设置 `summarizer: llm` 并配置 API，则使用大模型按 prompt 要求整理成「科技新闻」与「新论文」两部分并附一句话摘要。  

## 启用方式

1. 将本目录下 `config.example.yaml` 复制为 `config.yaml`（可按需改 `limit_news`、`limit_papers`、`feed_urls`、`arxiv_categories`）。  
2. 在主配置 `digest_info/config.yaml` 的 `categories` 中加入 `tech_daily`。  
3. 若要用大模型总结：在主配置中设置 `summarizer: llm` 并配置 `summarizer_params`（或环境变量 `OPENAI_API_KEY`）。  

## 可配置项（本类 config.yaml）

- `params.limit_news`：新闻条数上限（默认 15）  
- `params.limit_papers`：论文条数上限（默认 10）  
- `params.feed_urls`：可选，自定义 `[(url, 显示名), ...]` 覆盖默认三个 RSS  
- `params.arxiv_categories`：可选，如 `["cs.AI", "cs.LG", "cs.CL", "cs.CV"]`  
- `schedule.time`：推送时刻，默认 `"09:15"`  
- `schedule.window`：内容时间窗口，默认 `"24h"`  
