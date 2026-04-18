#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import time
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm


# ==================== 配置 ====================
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-4aaeaca4559f455da9a05a124c8e3dc5") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 生成模型（多模态 VL）
GENERATION_MODEL = os.environ.get("BASELINE_GENERATION_MODEL", "qwen3.6-plus")
# GENERATION_MODEL = "qwen-vl-plus"
# GENERATION_MODEL = os.environ.get("BASELINE_GENERATION_MODEL", "qwen2.5-vl-7b-instruct")

# 评估模型（纯文本，用于比较答案质量）
EVAL_MODEL = os.environ.get("BASELINE_EVAL_MODEL", "qwen-plus")  # 也可用 gpt-4o-mini 等

INPUT_DATASET = os.environ.get("BASELINE_INPUT_DATASET", "2025_dataset.jsonl")
IMAGE_ROOT = Path(os.environ.get("BASELINE_IMAGE_ROOT", "2025"))
OUTPUT_EVAL = os.environ.get("BASELINE_OUTPUT_EVAL", "evaluation_results.jsonl")
OUTPUT_SUMMARY = os.environ.get("BASELINE_OUTPUT_SUMMARY", "evaluation_summary.json")

MAX_WORKERS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 qwen_eval.py baseline，支持 limit/resume/output-dir。")
    parser.add_argument("--input-dataset", default=INPUT_DATASET)
    parser.add_argument("--image-root", default=str(IMAGE_ROOT))
    parser.add_argument("--output-dir", default=None, help="baseline 原始输出目录。")
    parser.add_argument("--output-eval", default=None, help="JSONL 输出路径，会覆盖 --output-dir 默认值。")
    parser.add_argument("--output-summary", default=None, help="汇总 JSON 输出路径，会覆盖 --output-dir 默认值。")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个样本。")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--resume", action="store_true", default=True, help="跳过输出 JSONL 中已完成的 post_id。")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> tuple[str, Path, Path, Path, int]:
    global IMAGE_ROOT
    IMAGE_ROOT = Path(args.image_root)
    output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    output_eval = Path(args.output_eval) if args.output_eval else output_dir / OUTPUT_EVAL
    output_summary = Path(args.output_summary) if args.output_summary else output_dir / OUTPUT_SUMMARY
    output_eval.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    return args.input_dataset, output_eval, output_summary, args.max_workers


def load_completed_post_ids(output_eval: Path) -> set[str]:
    if not output_eval.exists():
        return set()
    completed: set[str] = set()
    with output_eval.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            post_id = row.get("post_id")
            if post_id and not row.get("error"):
                completed.add(str(post_id))
    return completed


# ==================== 辅助函数 ====================


def encode_image_to_base64(image_path: str | Path) -> str | None:
    """读取本地图片并返回 base64 data URL。"""
    path = Path(image_path)
    if not path.exists():
        return None
    with path.open("rb") as f:
        img_data = f.read()
    base64_str = base64.b64encode(img_data).decode("utf-8")
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{base64_str}"


def convert_messages_images_to_base64(messages: List[Dict[str, Any]], post_id: str) -> List[Dict[str, Any]]:
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
                img_filename = Path(original_url).name
                local_img_path = resolve_image_path(post_id, img_filename)
                if os.environ.get("BASELINE_DEBUG_IMAGES"):
                    print("image path:", local_img_path)
                base64_url = encode_image_to_base64(local_img_path)
                if base64_url:
                    new_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_url},
                        }
                    )
                else:
                    new_content.append(
                        {
                            "type": "text",
                            "text": f"[图片缺失: {original_url}]",
                        }
                    )
        new_messages.append({"role": msg["role"], "content": new_content})
    return new_messages


def resolve_image_path(post_id: str, img_filename: str) -> Path:
    """Resolve 2025/<month>/<post_id>/images/<filename> image layout."""
    matches = list(IMAGE_ROOT.glob(f"*/{post_id}/images/{img_filename}"))
    if matches:
        return matches[0]
    return IMAGE_ROOT / post_id / "images" / img_filename


def make_client() -> OpenAI:
    return OpenAI(api_key=API_KEY or "missing", base_url=BASE_URL)


def call_qwen_vl(api_client: OpenAI, messages: List[Dict[str, Any]]) -> str:
    """调用 qwen-vl 模型生成回复。"""
    try:
        response = api_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"生成调用失败: {e}")
        return "[ERROR] 生成失败"


def evaluate_answer(api_client: OpenAI, golden_answer: str, generated_answer: str) -> Dict[str, Any]:
    """使用大模型评估生成答案的质量。"""
    eval_prompt = f"""你是一个公正的答案质量评估专家。请对比以下两个答案，评估生成答案的质量。

【标准答案（正确参考）】
{golden_answer}

【待评估答案（模型生成）】
{generated_answer}

请按以下维度打分（1-5 分，5 分为最高）：
1. 准确性：答案是否正确解决了问题？
2. 完整性：是否覆盖了关键信息？
3. 清晰度：表达是否清晰易理解？
4. 有用性：是否提供了实用价值？

最终输出格式必须为 JSON，例如：
{{"accuracy": 5, "completeness": 4, "clarity": 5, "usefulness": 4, "average_score": 4.5}}
只输出 JSON，不要有其他文字。"""
    try:
        response = api_client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的答案质量评估工具。"},
                {"role": "user", "content": eval_prompt},
            ],
            temperature=0.1,
        )
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result = json.loads(result_text.strip())
        return {key: result.get(key, 0) for key in ["accuracy", "completeness", "clarity", "usefulness", "average_score"]}
    except Exception as e:
        print(f"评估调用失败: {e}")
        return {
            "accuracy": 0,
            "completeness": 0,
            "clarity": 0,
            "usefulness": 0,
            "average_score": 0,
        }


def process_one_sample(sample: Dict[str, Any], api_client: OpenAI) -> Dict[str, Any]:
    """处理单个样本：转换图片为 base64，调用 qwen-vl 生成，再评估。"""
    post_id = sample["post_id"]
    messages = sample["messages"]
    golden_answer = ""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            golden_answer = msg["content"]
            break
    if not golden_answer:
        return {"post_id": post_id, "error": "未找到标准答案"}

    user_messages = [msg for msg in messages if msg["role"] == "user"]
    if not user_messages:
        return {"post_id": post_id, "error": "没有 user 消息"}

    converted_user_messages = convert_messages_images_to_base64(user_messages, post_id)
    generated_answer = call_qwen_vl(api_client, converted_user_messages)
    evaluation = evaluate_answer(api_client, golden_answer, generated_answer)

    return {
        "post_id": post_id,
        "golden_answer": golden_answer,
        "generated_answer": generated_answer,
        "evaluation": evaluation,
    }


def process_one_sample_with_worker_client(worker_state: threading.local, sample: Dict[str, Any]) -> Dict[str, Any]:
    api_client = getattr(worker_state, "client", None)
    if api_client is None:
        api_client = make_client()
        worker_state.client = api_client
    started = time.perf_counter()
    try:
        result = process_one_sample(sample, api_client)
    except Exception as exc:  # noqa: BLE001
        result = {"post_id": str(sample.get("post_id")), "error": str(exc)}
    result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    return result


def write_result(output_eval: Path, result: Dict[str, Any]) -> None:
    with output_eval.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def update_progress(progress: tqdm, result: Dict[str, Any]) -> None:
    post_id = str(result.get("post_id", ""))
    average_score = result.get("evaluation", {}).get("average_score", 0)
    elapsed = float(result.get("elapsed_seconds", 0.0) or 0.0)
    progress.set_postfix(
        {
            "id": post_id,
            "score": average_score,
            "sec": f"{elapsed:.1f}",
            "err": int(bool(result.get("error"))),
        }
    )
    tqdm.write(
        f"[qwen_eval] done post_id={post_id} "
        f"score={average_score} elapsed={elapsed:.2f}s error={result.get('error', '')}"
    )


def main() -> None:
    args = parse_args()
    input_dataset, output_eval, output_summary, max_workers = configure_from_args(args)
    if not API_KEY:
        raise RuntimeError("请先设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY，再运行 qwen_eval.py baseline")

    samples = []
    with open(input_dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit is not None:
        samples = samples[: args.limit]
    requested_count = len(samples)
    print(f"共加载 {requested_count} 个样本")

    completed_post_ids = load_completed_post_ids(output_eval) if args.resume else set()
    if completed_post_ids:
        before = len(samples)
        samples = [sample for sample in samples if str(sample.get("post_id")) not in completed_post_ids]
        print(f"Resume enabled: skipped {before - len(samples)} completed samples; remaining {len(samples)}")

    max_workers = max(1, int(max_workers))
    print(f"[qwen_eval] pending={len(samples)} max_workers={max_workers} output={output_eval}")
    start = time.perf_counter()
    if max_workers == 1:
        worker_state = threading.local()
        progress = tqdm(samples, desc="baseline 生成评估", unit="sample")
        for sample in progress:
            post_id = str(sample.get("post_id"))
            progress.set_postfix_str(f"id={post_id}")
            result = process_one_sample_with_worker_client(worker_state, sample)
            write_result(output_eval, result)
            update_progress(progress, result)
    else:
        worker_state = threading.local()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_one_sample_with_worker_client, worker_state, sample): sample
                for sample in samples
            }
            progress = tqdm(total=len(futures), desc="baseline 生成评估", unit="sample")
            for future in as_completed(futures):
                sample = futures[future]
                post_id = str(sample.get("post_id"))
                progress.set_postfix_str(f"id={post_id}")
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {"post_id": post_id, "error": str(exc), "elapsed_seconds": 0.0}
                write_result(output_eval, result)
                update_progress(progress, result)
                progress.update(1)
            progress.close()
    elapsed = time.perf_counter() - start

    all_results = []
    if output_eval.exists():
        with output_eval.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    valid_results = [r for r in all_results if r.get("evaluation") and r["evaluation"].get("average_score")]
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
        "requested_samples": requested_count,
        "new_samples": len(samples),
        "successful_samples": total,
        "average_scores": {
            "accuracy": round(avg_accuracy, 2),
            "completeness": round(avg_completeness, 2),
            "clarity": round(avg_clarity, 2),
            "usefulness": round(avg_usefulness, 2),
            "overall": round(avg_overall, 2),
        },
        "generation_model": GENERATION_MODEL,
        "evaluation_model": EVAL_MODEL,
        "input_dataset": str(input_dataset),
        "output_eval": str(output_eval),
        "resume": args.resume,
        "max_workers": max_workers,
        "elapsed_seconds": round(elapsed, 4),
    }

    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 评估汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n详细评估结果已保存到 {output_eval}")
    print(f"汇总报告已保存到 {output_summary}")


def test() -> None:
    samples = []
    with open(INPUT_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            break
    print(f"共加载 {len(samples)} 个样本")
    result = process_one_sample(samples[0], make_client())
    print(result)


if __name__ == "__main__":
    main()
    # test()
