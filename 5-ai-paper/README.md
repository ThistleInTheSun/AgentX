# AI 论文速读（ai-paper）

每天自动从 arXiv AI 相关领域选一篇较新的论文，生成一篇面向 AI 从业者/爱好者的中文解读草稿，输出到公众号草稿箱或本地 Markdown。

- 内容源：arXiv（cs.AI / cs.CL / cs.CV / cs.LG / cs.IR / cs.RO）
- 输出：优先写入公众号草稿箱；未配置公众号凭证时落盘到 `drafts/*.md`
- 解读风格：严肃科普，突出「为什么重要」「核心创新」「关键结果」「适合谁读」

## 安装

```bash
cd 5-ai-paper
pip install -r requirements.txt   # 只依赖 requests
cp .env.example .env              # 填入你的 Key
```

## .env 字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `LLM_API_KEY` | 是 | LLM API Key（默认 DeepSeek，任何 OpenAI 兼容 API 均可） |
| `LLM_BASE_URL` | 否 | 默认 `https://api.deepseek.com` |
| `LLM_MODEL` | 否 | 默认 `deepseek-chat` |
| `WECHAT_APPID` | 否 | 公众号 AppID；不填则只输出本地 Markdown |
| `WECHAT_APPSECRET` | 否 | 公众号 AppSecret |
| `WECHAT_THUMB_MEDIA_ID` | 否 | 草稿封面图 media_id（草稿箱接口必须有封面），获取方式见下 |
| `WECHAT_AUTHOR` | 否 | 草稿显示的作者名 |
| `WECHAT_AUTO_PUBLISH` | 否 | 设为 `1`：草稿写入后自动「发布」到公众号主页（不推送粉丝；群发/定时群发仍需在后台手动操作） |

## 手动跑一次

```bash
python run.py            # 完整流水线：选论文 → 解读 → 组装 → 草稿（→ 自动发布）
python run.py --dry-run  # 只测试选论文，不调 LLM / 公众号
python run.py --push drafts/xxx.md  # 把已有本地草稿推送到草稿箱（不重新解读）
```

成功后：

- `drafts/2026-07-29-xxx.md`：完整解读草稿
- `state.json`：已处理论文记录（同一篇不会重复处理）
- `logs/日期.log`：运行日志；失败时退出码非 0

前 7 天建议只进草稿箱、人工点发布，不要自动群发。

## 启用公众号草稿箱

1. 公众号后台（设置与开发 → 基本配置）拿到 AppID / AppSecret，填入 `.env`；
   把服务器出口 IP 加入「IP 白名单」。
2. 上传一张封面图获取 media_id（草稿必须有封面）：

   ```bash
   python run.py --upload-thumb cover.jpg
   # 输出 media_id，写入 .env: WECHAT_THUMB_MEDIA_ID=xxx
   ```

3. 再跑 `python run.py`，即写入草稿箱。写入失败时会自动降级：本地 Markdown 已保存，日志有错误详情。

## 定时任务

参考 `4daily-gloss` 的 `daily_run.sh`，把路径改为 `/mnt/d/xq/AgentX/5-ai-paper` 即可。

## 目录结构

```
run.py            # 入口：跑一次完整流水线
src/
  config.py       # .env 加载与常量
  fetch.py        # 拉取 arXiv AI 论文列表并筛选
  explain.py      # LLM 解读：总结、意义、创新、结果
  assemble.py     # 组装 Markdown
  wechat_draft.py # 公众号草稿箱写入（Markdown→HTML）
  state.py        # state.json 去重记录
drafts/           # 本地草稿输出
logs/             # 运行日志
```
