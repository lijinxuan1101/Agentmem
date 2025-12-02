import json
from pathlib import Path

def read_jsonl_file(file_path):
    """逐行读取 JSONL 文件并解析为 Python 列表"""
    data_list = []
    
    # 确保文件路径是 Path 对象
    file_path = Path(file_path)
    if not file_path.exists():
        # 如果文件不存在，则创建模拟文件 (用于演示)
        print(f"File not found: {file_path}. Creating simulated content...")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        simulated_content = [
            {"id": "C001_0", "topic": "C001", "context": "SESSION_1...", "question": "When did Alice report latency improvements?", "answer": "2025-02-02"},
            {"id": "C001_1", "topic": "C001", "context": "SESSION_1...", "question": "Who confirmed the patch deployment?", "answer": "Bob"},
        ]
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in simulated_content:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    data_list.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}. Error: {e}")
                    
    return data_list

# 假设您要读取转换后的 LoCoMo 文件
file_to_read = "data/external/locomo/processed/locomo10.jsonl" 

# --- 文件路径设置 ---
input_path = Path(file_to_read)
output_dir = input_path.parent 
output_filename = "modified_first_record.json" # 定义输出文件名
output_path = output_dir / output_filename 

# 调用函数获取数据
all_records = read_jsonl_file(file_to_read)

if all_records:
    # 1. 获取第一条记录 (这是字典对象)
    first_record = all_records[0]
    
    # 2. 修改记录内容
    print("--- 正在修改第一条记录 ---")
    
    # 示例修改 1: 添加一个新字段
    first_record["is_modified"] = True
    
    # 示例修改 2: 修改 question 字段
    original_question = first_record.get("question", "")
    first_record["question"] = original_question + " [MODIFIED FOR TESTING]"
    
    print(f"原问题: {original_question}")
    print(f"新问题: {first_record['question']}")
    
    # 3. 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. 储存修改后的记录到同级文件夹
    with open(output_path, 'w', encoding='utf-8') as f:
        # 使用 indent=2 进行美化打印，方便阅读
        json.dump(first_record, f, indent=2, ensure_ascii=False)
    
    print(f"\n修改后的第一条记录已成功保存到: {output_path}")

else:
    print("文件未读取到任何有效记录，无法保存。")