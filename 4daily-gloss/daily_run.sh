#!/bin/bash
# 每日外刊精读：由 Windows 计划任务经 wsl.exe 调用
cd /mnt/d/xq/AgentX/4daily-gloss
/mnt/d/APP/WSL/miniconda3/bin/python3 run.py >> logs/cron.log 2>&1
