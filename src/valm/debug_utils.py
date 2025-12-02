from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


def log_debug_json(filename: str, record: Dict[str, Any]) -> None:
    """
    将调试记录以 JSON 列表的形式写入到 `debug/{filename}` 中：
    - 文件整体是一个 JSON 数组：[record1, record2, ...]
    - 每次调用会在数组末尾追加一条记录并整体重写文件
    - 自动在记录中加入 `ts` 字段（写入时间戳，float）
    - 若写入失败（例如权限问题），静默忽略，避免干扰主流程
    """
    try:
        os.makedirs("debug", exist_ok=True)
        record_with_ts = {"ts": time.time(), **record}
        path = os.path.join("debug", filename)
        data = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = f.read().strip()
                    if existing:
                        loaded = json.loads(existing)
                        if isinstance(loaded, list):
                            data = loaded
            except Exception:
                # 旧文件内容异常时，从空列表重新开始
                data = []
        data.append(record_with_ts)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except Exception:
        # 调试日志写入失败不应影响主逻辑
        return