#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import base64
import time
from typing import Dict, Any, List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ==================== 配置 ====================
API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 生成模型（多模态 VL）
#GENERATION_MODEL = "qwen3.6-plus"   # 或 qwen-vl-max
#GENERATION_MODEL = "qwen-vl-plus"
GENERATION_MODEL = os.environ.get("BASELINE_GENERATION_MODEL", "qwen2.5-vl-7b-instruct")
# 评估模型（纯文本，用于比较答案质量）
EVAL_MODEL = os.environ.get("BASELINE_EVAL_MODEL", "qwen-plus")           # 也可用 gpt-4o-mini 等

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
INPUT_DATASET = os.environ.get("BASELINE_INPUT_DATASET", "2025_dataset.jsonl")
IMAGE_ROOT = Path(os.environ.get("BASELINE_IMAGE_ROOT", "2025"))
OUTPUT_EVAL = os.environ.get("BASELINE_OUTPUT_EVAL", "evaluation_results.jsonl")
OUTPUT_SUMMARY = os.environ.get("BASELINE_OUTPUT_SUMMARY", "evaluation_summary.json")

MAX_WORKERS = 5

client = OpenAI(api_key=API_KEY or "missing", base_url=BASE_URL)

# ==================== 辅助函数 ====================

def encode_image_to_base64(image_path: str) -> str:
    """读取本地图片并返回 base64 data URL"""
    path = Path(image_path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        img_data = f.read()
    base64_str = base64.b64encode(img_data).decode("utf-8")
    # 根据扩展名确定 MIME 类型
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{base64_str}"

def convert_messages_images_to_base64(messages: List[Dict], post_id: str) -> List[Dict]:
    """
    将 messages 中所有 image_url 的本地相对路径转换为 base64 Data URL。
    默认图片位于 2025/<month>/<post_id>/images/ 目录下。
    """
    new_messages = []
    for msg in messages:
        if msg["role"] != "user" or not isinstance(msg["content"], list):
            new_messages.append(msg)
            continue
        new_content = []
        for item in msg["content"]:
            if item["type"] == "text":
                new_content.append(item)
            elif item["type"] == "image_url":
                original_url = item["image_url"]["url"]
                # 期望格式如 "images/xxx.png"
                img_filename = Path(original_url).name
                local_img_path = resolve_image_path(post_id, img_filename)
                if os.environ.get("BASELINE_DEBUG_IMAGES"):
                    print("image path:", local_img_path)
                base64_url = encode_image_to_base64(local_img_path)
                if base64_url:
                    new_content.append({
                        "type": "image_url",
                        "image_url": {"url": base64_url}
                    })
                else:
                    # 图片缺失，添加文本占位符
                    new_content.append({
                        "type": "text",
                        "text": f"[图片缺失: {original_url}]"
                    })
        new_messages.append({"role": msg["role"], "content": new_content})
    return new_messages

def resolve_image_path(post_id: str, img_filename: str) -> Path:
    """Resolve 2025/<month>/<post_id>/images/<filename> image layout."""
    matches = list(IMAGE_ROOT.glob(f"*/{post_id}/images/{img_filename}"))
    if matches:
        return matches[0]
    return IMAGE_ROOT / post_id / "images" / img_filename

def call_qwen_vl(messages: List[Dict]) -> str:
    """调用 qwen-vl 模型生成回复（messages 中已包含 base64 图片）"""
    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"生成调用失败: {e}")
        return "[ERROR] 生成失败"

def evaluate_answer(golden_answer: str, generated_answer: str) -> Dict[str, Any]:
    """使用大模型评估生成答案的质量"""
    eval_prompt = f"""你是一个公正的答案质量评估专家。请对比以下两个答案，评估生成答案的质量。

【标准答案（正确参考）】
{golden_answer}

【待评估答案（模型生成）】
{generated_answer}

请按以下维度打分（1-5分，5分为最高）：
1. 准确性：答案是否正确解决了问题？（1-5）
2. 完整性：是否覆盖了关键信息？（1-5）
3. 清晰度：表达是否清晰易理解？（1-5）
4. 有用性：是否提供了实用价值？（1-5）

最终输出格式必须为 JSON，例如：
{{"accuracy": 5, "completeness": 4, "clarity": 5, "usefulness": 4, "average_score": 4.5, "comment": "简要评语"}}
只输出 JSON，不要有其他文字。
"""
    try:
        response = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的答案质量评估工具。"},
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0.1,
        )
        result_text = response.choices[0].message.content.strip()
        # 提取 JSON（可能包含前后 markdown 标记）
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result = json.loads(result_text.strip())
        return result
    except Exception as e:
        print(f"评估调用失败: {e}")
        return {
            "accuracy": 0,
            "completeness": 0,
            "clarity": 0,
            "usefulness": 0,
            "average_score": 0,
            "comment": f"评估失败: {str(e)}"
        }

def process_one_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """处理单个样本：转换图片为 base64，调用 qwen-vl 生成，再评估"""
    post_id = sample["post_id"]
    messages = sample["messages"]
    # 提取 golden answer
    golden_answer = ""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            golden_answer = msg["content"]
            break
    if not golden_answer:
        return {"post_id": post_id, "error": "未找到标准答案"}

    # 构造 user 消息（只保留 user 部分）
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    if not user_messages:
        return {"post_id": post_id, "error": "没有 user 消息"}

    # 转换图片为 base64
    converted_user_messages = convert_messages_images_to_base64(user_messages, post_id)

    # 调用 qwen-vl 生成
    generated_answer = call_qwen_vl(converted_user_messages)

    # 评估
    evaluation = evaluate_answer(golden_answer, generated_answer)

    return {
        "post_id": post_id,
        "golden_answer": golden_answer,
        "generated_answer": generated_answer,
        "evaluation": evaluation
    }

def main():
    if not API_KEY:
        raise RuntimeError("请先设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY，再运行 qwen_eval.py baseline")
    # 读取数据集
    samples = []
    with open(INPUT_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    print(f"共加载 {len(samples)} 个样本")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_sample, sample): sample for sample in samples}
        for future in tqdm(as_completed(futures), total=len(futures), desc="评估进度"):
            res = future.result()
            results.append(res)
            with open(OUTPUT_EVAL, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    # 汇总统计
    valid_results = [r for r in results if r.get("evaluation") and r["evaluation"].get("average_score")]
    if not valid_results:
        print("没有有效的评估结果")
        return

    total = len(valid_results)
    avg_accuracy = sum(r["evaluation"]["accuracy"] for r in valid_results) / total
    avg_completeness = sum(r["evaluation"]["completeness"] for r in valid_results) / total
    avg_clarity = sum(r["evaluation"]["clarity"] for r in valid_results) / total
    avg_usefulness = sum(r["evaluation"]["usefulness"] for r in valid_results) / total
    avg_overall = sum(r["evaluation"]["average_score"] for r in valid_results) / total

    summary = {
        "total_samples": len(samples),
        "successful_samples": total,
        "average_scores": {
            "accuracy": round(avg_accuracy, 2),
            "completeness": round(avg_completeness, 2),
            "clarity": round(avg_clarity, 2),
            "usefulness": round(avg_usefulness, 2),
            "overall": round(avg_overall, 2)
        },
        "generation_model": GENERATION_MODEL,
        "evaluation_model": EVAL_MODEL
    }

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 评估汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n详细评估结果已保存到 {OUTPUT_EVAL}")
    print(f"汇总报告保存到 {OUTPUT_SUMMARY}")

def test():
    samples = []
    with open(INPUT_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            break
    print(f"共加载 {len(samples)} 个样本")
    result = process_one_sample(samples[0])
    print(result)
    

if __name__ == "__main__":
    main()
    #test()
