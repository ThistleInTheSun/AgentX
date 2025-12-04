#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 JSONL 文件格式的脚本
将 JSON 字符串中的换行符转义为 \n，并转换为标准 JSONL 格式（每行一个 JSON 对象）
"""

import json
import os
import sys


def fix_jsonl_file(input_file, output_file=None):
    """
    修复 JSONL 文件格式
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为 None 则覆盖原文件
    """
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        return False
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    objects = []
    current_pos = 0
    
    # 解析每个 JSON 对象
    while current_pos < len(content):
        start_pos = content.find('{"messages":', current_pos)
        if start_pos == -1:
            break
        
        # 找到匹配的结束 }
        brace_count = 0
        in_string = False
        escape_next = False
        pos = start_pos
        
        while pos < len(content):
            char = content[pos]
            
            if escape_next:
                escape_next = False
                pos += 1
                continue
            
            if char == '\\':
                escape_next = True
                pos += 1
                continue
            
            if char == '"':
                in_string = not in_string
            elif not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = pos + 1
                        obj_str = content[start_pos:end_pos]
                        
                        try:
                            # 使用非严格模式解析（允许控制字符）
                            obj = json.loads(obj_str, strict=False)
                            objects.append(obj)
                            print(f"✓ 成功解析第 {len(objects)} 个对象")
                        except json.JSONDecodeError as e:
                            print(f"✗ 解析错误: {e}")
                            print(f"  位置: {start_pos}-{end_pos}")
                            return False
                        
                        current_pos = end_pos
                        break
            
            pos += 1
        else:
            break
    
    if not objects:
        print("错误: 没有找到任何有效的 JSON 对象")
        return False
    
    # 确定输出文件路径
    if output_file is None:
        output_file = input_file
    
    # 重新写入文件（标准 JSONL 格式，换行符已自动转义）
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in objects:
            json_line = json.dumps(obj, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"\n✓ 修复完成！共处理 {len(objects)} 个对象")
    print(f"✓ 已保存到: {output_file}")
    
    # 验证修复后的文件
    print("\n验证修复后的文件...")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"✗ 第 {i} 行仍有错误: {e}")
                return False
    
    print("✓ 所有行都有效！")
    return True


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python fix_jsonl.py <输入文件> [输出文件]")
        print("示例: python fix_jsonl.py Trainingdata.jsonl")
        print("示例: python fix_jsonl.py Trainingdata.jsonl Trainingdata_fixed.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = fix_jsonl_file(input_file, output_file)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

