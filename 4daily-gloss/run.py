#!/usr/bin/env python3
"""每日外刊精读流水线：跑一次，产出一篇公众号草稿（或本地 Markdown）。

用法:
  python run.py                     # 跑一次完整流水线
  python run.py --dry-run           # 不调 LLM/公众号，只测试选文
  python run.py --upload-thumb X.jpg  # 上传公众号封面图，获取 media_id
"""
import argparse
import logging
import re
import sys
from datetime import date

from src import assemble, config, fetch, state, translate, wechat_draft


def setup_logging() -> None:
    config.LOGS_DIR.mkdir(exist_ok=True)
    logfile = config.LOGS_DIR / f"{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(logfile, encoding="utf-8")],
    )


def slugify(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower()
    return slug[:60] or "article"


def run_once(dry_run: bool = False) -> int:
    log = logging.getLogger("run")
    st = state.load()

    log.info("拉取 OWID feed ...")
    articles = fetch.fetch_feed()
    article = fetch.pick_article(articles, lambda url: state.is_processed(st, url))
    if article is None:
        log.warning("没有可处理的新文章（全部已处理或不符合条件）")
        return 2
    log.info("选中文章：%s (%s)", article.title, article.url)

    if dry_run:
        print(f"[dry-run] 选中：{article.title}\n{article.url}\n正文 {len(article.text)} 字符")
        return 0

    log.info("翻译 %d 段 ...", len(article.paragraphs))
    translations = translate.translate(article.paragraphs)

    log.info("抽取生词与长难句 ...")
    notes = translate.extract_study_notes(article.text)

    publish_day = date.today()
    md = assemble.assemble(article, translations, notes, publish_day)

    config.DRAFTS_DIR.mkdir(exist_ok=True)
    out_path = config.DRAFTS_DIR / f"{publish_day.isoformat()}-{slugify(article.title)}.md"
    out_path.write_text(md, encoding="utf-8")
    log.info("已写入本地草稿：%s", out_path)

    status = "drafted"
    if wechat_draft.credentials_ready():
        try:
            wechat_draft.create_draft(
                title=article.title,
                markdown=md,
                source_url=article.url,
                digest=f"外刊精读 | {article.title}",
            )
            status = "published_to_wechat"
        except wechat_draft.WeChatError as e:
            log.error("公众号草稿写入失败（本地 Markdown 已保存）：%s", e)
    else:
        log.info("未配置公众号凭证，跳过草稿箱写入（见 README）")

    state.mark_processed(st, article.url, article.title, status)
    print(f"完成：{out_path}（状态：{status}）")
    return 0


def push_draft(md_path: str) -> int:
    """把已生成的本地 Markdown 草稿推送到公众号草稿箱（不重新翻译）。"""
    from pathlib import Path

    md = Path(md_path).read_text(encoding="utf-8")
    first_line = md.splitlines()[0]
    title = first_line.lstrip("# ").split("|", 1)[-1].strip()
    m = re.search(r"> 链接：(\S+)", md)
    source_url = m.group(1) if m else ""
    wechat_draft.create_draft(
        title=title,
        markdown=md,
        source_url=source_url,
        digest=f"外刊精读 | {title}",
    )
    print(f"已推送到公众号草稿箱：{title}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="每日外刊精读流水线")
    parser.add_argument("--dry-run", action="store_true", help="只测试选文，不调 LLM/公众号")
    parser.add_argument("--upload-thumb", metavar="IMAGE", help="上传公众号封面图并打印 media_id")
    parser.add_argument("--push", metavar="MD_FILE", help="把已有的本地 Markdown 草稿推送到公众号草稿箱")
    args = parser.parse_args()

    config.load_env()
    setup_logging()
    log = logging.getLogger("run")

    try:
        if args.upload_thumb:
            media_id = wechat_draft.upload_thumb(args.upload_thumb)
            print(f"media_id: {media_id}\n请写入 .env: WECHAT_THUMB_MEDIA_ID={media_id}")
            return 0
        if args.push:
            return push_draft(args.push)
        return run_once(dry_run=args.dry_run)
    except Exception:
        log.exception("流水线执行失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
