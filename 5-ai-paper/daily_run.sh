#!/bin/bash
# AI 论文速读：由 Windows 计划任务经 wsl.exe 调用
cd /mnt/d/xq/AgentX/5-ai-paper
/mnt/d/APP/WSL/miniconda3/bin/python3 run.py >> logs/cron.log 2>&1
