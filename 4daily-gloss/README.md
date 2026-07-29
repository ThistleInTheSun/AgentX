# 每日外刊精读（daily-gloss）

每天自动生成一篇「英文原文精读」公众号草稿：中文翻译 + 10 个考研向生词 + 1 个长难句拆解 + 艾宾浩斯复习表 + CC BY 版权声明。

- 内容源：[Our World in Data](https://ourworldindata.org/)（CC BY 4.0，允许翻译转载，国内可直连）
- 输出：优先写入公众号草稿箱；未配置公众号凭证时落盘到 `drafts/*.md`
- 独立脚本，跑一次出一篇；定时靠系统计划任务

## 安装

```bash
cd 4daily-gloss
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
python run.py            # 完整流水线：选文 → 翻译 → 生词 → 组装 → 草稿（→ 自动发布）
python run.py --dry-run  # 只测试选文，不调 LLM / 公众号
python run.py --push drafts/xxx.md  # 把已有本地草稿推送到草稿箱（不重新翻译）
```

成功后：

- `drafts/2026-07-29-xxx.md`：完整草稿（翻译+5词+1句+复习表+声明）
- `state.json`：已处理文章记录（同一篇不会重复处理）
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

## 定时任务（已配置方案）

当前方案：**Windows 计划任务** 每天 20:00 调用 WSL 执行 `daily_run.sh`（WSL 未启动也会被自动唤醒）：

```bat
schtasks /create /tn daily-gloss /tr "wsl.exe -d Ubuntu-22.04 -- bash /mnt/d/xq/AgentX/4daily-gloss/daily_run.sh" /sc daily /st 20:00 /f
```

- 日志：`logs/cron.log`（脚本输出）+ `logs/日期.log`（详细日志）
- 删除任务：`schtasks /delete /tn daily-gloss /f`
- 改时间：重跑上面命令换 `/st` 即可

**推荐日常节奏**（个人订阅号无法 API 群发）：

1. 每晚 20:00 自动：生成草稿 → 写入草稿箱 → 发布上主页
2. 晚上你花 1 分钟：检查草稿 → 后台设「定时群发」到次日早 7:00
3. 微信次日自动推送给粉丝

### 备选：纯 WSL cron（需 WSL 常驻）

```bash
crontab -e
0 20 * * * cd /mnt/d/xq/AgentX/4daily-gloss && python3 run.py >> logs/cron.log 2>&1
```

## 版权说明

本项目仅使用 Our World in Data（CC BY 4.0）内容，每篇文末自动附带原文标题、作者、链接与许可声明。不抓取 Economist / BBC / Guardian 等需授权来源。

## 目录结构

```
run.py            # 入口：跑一次完整流水线
src/
  config.py       # .env 加载与常量
  fetch.py        # 拉 OWID feed + 文章正文，选文过滤
  translate.py    # LLM 翻译 + 生词/长难句抽取
  assemble.py     # 组装 Markdown（复习表、版权声明）
  wechat_draft.py # 公众号草稿箱写入（Markdown→HTML）
  state.py        # state.json 去重记录
drafts/           # 本地草稿输出
logs/             # 运行日志
```
