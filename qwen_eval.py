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

from configs.config import load_config
from evaluator.evaluate import Evaluator
from evaluator.report import build_error_analysis, summarize_scores
from tools.dataset_parser import DatasetParser
from tools.sample_ids import filter_items_by_sample_ids, read_sample_ids_file
from tools.utils import append_jsonl, write_json


# ==================== 配置 ====================
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-4aaeaca4559f455da9a05a124c8e3dc5") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 生成模型（多模态 VL）
GENERATION_MODEL = os.environ.get("BASELINE_GENERATION_MODEL", "qwen3.6-plus")
# GENERATION_MODEL = "qwen-vl-plus"
# GENERATION_MODEL = os.environ.get("BASELINE_GENERATION_MODEL", "qwen2.5-vl-7b-instruct")

INPUT_DATASET = os.environ.get("BASELINE_INPUT_DATASET", "2025_dataset.jsonl")
IMAGE_ROOT = Path(os.environ.get("BASELINE_IMAGE_ROOT", "2025"))
OUTPUT_EVAL = os.environ.get("BASELINE_OUTPUT_EVAL", "evaluation_results.jsonl")
OUTPUT_SUMMARY = os.environ.get("BASELINE_OUTPUT_SUMMARY", "predictions.summary.json")

MAX_WORKERS = 5
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("BASELINE_REQUEST_TIMEOUT_SECONDS", "180"))
REQUEST_MAX_RETRIES = int(os.environ.get("BASELINE_MAX_RETRIES", "0"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 qwen_eval.py baseline，支持 limit/resume/output-dir。")
    parser.add_argument("--config", default=None, help="统一评测配置，默认使用 configs/default.yaml。")
    parser.add_argument("--input-dataset", default=INPUT_DATASET)
    parser.add_argument("--image-root", default=str(IMAGE_ROOT))
    parser.add_argument("--output-dir", default=None, help="baseline 原始输出目录。")
    parser.add_argument("--output-predictions", default=None, help="agent 格式 predictions.jsonl 输出路径。")
    parser.add_argument("--output-eval", default=None, help="统一 eval_results.jsonl 输出路径。")
    parser.add_argument("--output-summary", default=None, help="汇总 JSON 输出路径，会覆盖 --output-dir 默认值。")
    parser.add_argument("--sample-ids-file", default=None, help="每行一个 post_id/sample_id，用于固定样本集合。")
    parser.add_argument("--retry-failed-only", action="store_true", help="只重跑当前 predictions.jsonl 中失败或空答案的样本。")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个样本。")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--request-timeout", type=float, default=REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=REQUEST_MAX_RETRIES)
    parser.add_argument(
        "--mode",
        choices=["composite", "local-only", "generate-only"],
        default="composite",
        help="composite 使用与 agent 相同的统一 LLM Judge+本地指标；local-only 只跑本地指标；generate-only 只生成不评测。",
    )
    parser.add_argument("--resume", action="store_true", default=True, help="跳过输出 JSONL 中已完成的 post_id。")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> tuple[str, Path, Path, Path, Path, int]:
    global IMAGE_ROOT, REQUEST_TIMEOUT_SECONDS, REQUEST_MAX_RETRIES
    IMAGE_ROOT = Path(args.image_root)
    REQUEST_TIMEOUT_SECONDS = float(args.request_timeout)
    REQUEST_MAX_RETRIES = max(0, int(args.max_retries))
    output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    output_predictions = Path(args.output_predictions) if args.output_predictions else output_dir / "predictions.jsonl"
    output_eval = Path(args.output_eval) if args.output_eval else output_dir / "eval_results.jsonl"
    output_summary = Path(args.output_summary) if args.output_summary else output_dir / OUTPUT_SUMMARY
    legacy_eval = output_dir / OUTPUT_EVAL
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    output_eval.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    return args.input_dataset, output_predictions, output_eval, legacy_eval, output_summary, args.max_workers


def load_completed_post_ids(predictions_path: Path) -> set[str]:
    if not predictions_path.exists():
        return set()
    completed: set[str] = set()
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = row.get("sample_id") or row.get("post_id")
            if sample_id and is_successful_prediction(row):
                completed.add(str(sample_id))
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
    return OpenAI(
        api_key=API_KEY or "missing",
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=REQUEST_MAX_RETRIES,
    )


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


def process_one_sample(
    sample: Dict[str, Any],
    api_client: OpenAI,
    evaluator: Evaluator | None = None,
    standard_sample: Any = None,
    use_llm_judge: bool = True,
) -> Dict[str, Any]:
    """处理单个样本：转换图片为 base64，调用 qwen-vl 生成，并返回 agent 格式结果。"""
    post_id = sample["post_id"]
    messages = sample["messages"]
    golden_answer = ""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            golden_answer = msg["content"]
            break
    question = standard_sample.question_text if standard_sample is not None else _question_from_messages(messages)
    if not golden_answer:
        return _baseline_prediction_row(post_id, question, "", ["未找到标准答案"], 0.0)

    user_messages = [msg for msg in messages if msg["role"] == "user"]
    if not user_messages:
        return _baseline_prediction_row(post_id, question, "", ["没有 user 消息"], 0.0)

    converted_user_messages = convert_messages_images_to_base64(user_messages, post_id)
    generated_answer = call_qwen_vl(api_client, converted_user_messages)
    errors = ["baseline generation failed"] if is_generation_failure(generated_answer) else []
    result = _baseline_prediction_row(post_id, question, generated_answer, errors, 0.0)
    if errors or evaluator is None or standard_sample is None:
        return result
    eval_row = evaluator.evaluate(standard_sample, generated_answer, use_llm_judge=use_llm_judge).to_json()
    result.update(eval_row)
    result["evaluation"] = _qwen_compatible_evaluation(eval_row)
    return result


def _question_from_messages(messages: List[Dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
    return "\n".join(part.strip() for part in parts if part.strip())


def _baseline_prediction_row(sample_id: str, question: str, answer: str, errors: list[str], elapsed: float) -> Dict[str, Any]:
    return {
        "sample_id": str(sample_id),
        "question": question,
        "answer": answer,
        "tools_used": ["qwen_baseline"],
        "web_searched": False,
        "tool_trace": [],
        "reasoning_summary": "qwen_eval baseline direct multimodal generation.",
        "elapsed_seconds": round(elapsed, 4),
        "token_usage": {},
        "errors": errors,
        "plan": None,
    }


def process_one_sample_with_worker_client(
    worker_state: threading.local,
    sample: Dict[str, Any],
    config: Any | None = None,
    sample_map: Dict[str, Any] | None = None,
    mode: str = "composite",
) -> Dict[str, Any]:
    api_client = getattr(worker_state, "client", None)
    if api_client is None:
        api_client = make_client()
        worker_state.client = api_client
    evaluator = None
    standard_sample = None
    if mode != "generate-only":
        evaluator = getattr(worker_state, "evaluator", None)
        if evaluator is None:
            evaluator = Evaluator(config)
            worker_state.evaluator = evaluator
        standard_sample = (sample_map or {}).get(str(sample.get("post_id")))
    started = time.perf_counter()
    try:
        result = process_one_sample(
            sample,
            api_client,
            evaluator=evaluator,
            standard_sample=standard_sample,
            use_llm_judge=mode == "composite",
        )
    except Exception as exc:  # noqa: BLE001
        result = _baseline_prediction_row(str(sample.get("post_id")), "", "", [str(exc)], 0.0)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    return result


def _qwen_compatible_evaluation(eval_row: Dict[str, Any]) -> Dict[str, Any]:
    judge = eval_row.get("llm_judge", {}) or {}
    return {
        "accuracy": judge.get("accuracy", 0),
        "completeness": judge.get("completeness", 0),
        "clarity": judge.get("clarity", 0),
        "usefulness": judge.get("usefulness", 0),
        "average_score": judge.get("average_score", 0),
        "unified_score": judge.get("score", 0),
        "factual_consistency": judge.get("factual_consistency", 0),
    }


def write_result(output_eval: Path, result: Dict[str, Any]) -> None:
    with output_eval.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def update_progress(progress: tqdm, result: Dict[str, Any]) -> None:
    post_id = str(result.get("sample_id") or result.get("post_id") or "")
    average_score = result.get("evaluation", {}).get("average_score", 0)
    final_score = result.get("final_score")
    elapsed = float(result.get("elapsed_seconds", 0.0) or 0.0)
    score_text = f"{float(final_score):.4f}" if final_score is not None else str(average_score)
    progress.set_postfix(
        {
            "id": post_id,
            "score": score_text,
            "sec": f"{elapsed:.1f}",
            "err": len(result.get("errors", []) or []),
        }
    )
    tqdm.write(
        f"[qwen_eval] done post_id={post_id} "
        f"score={score_text} elapsed={elapsed:.2f}s errors={len(result.get('errors', []) or [])}"
    )


def main() -> None:
    args = parse_args()
    input_dataset, output_predictions, output_eval, legacy_eval, output_summary, max_workers = configure_from_args(args)
    config = load_config(args.config)
    if not API_KEY:
        raise RuntimeError("请先设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY，再运行 qwen_eval.py baseline")

    samples = []
    with open(input_dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    sample_ids = read_sample_ids_file(args.sample_ids_file)
    if sample_ids:
        before = len(samples)
        samples = filter_items_by_sample_ids(samples, sample_ids, lambda sample: str(sample.get("post_id")))
        if not samples:
            raise SystemExit(f"no samples matched --sample-ids-file: {args.sample_ids_file}")
        missing = len(sample_ids) - len(samples)
        print(f"[qwen_eval] sample_ids_file={args.sample_ids_file} matched={len(samples)}/{before} missing={missing}")
    if args.retry_failed_only:
        failed_ids = _failed_prediction_ids(output_predictions)
        before = len(samples)
        samples = _filter_retry_failed_samples(samples, failed_ids)
        print(f"[qwen_eval] retry_failed_only matched={len(samples)}/{before} failed_ids={len(failed_ids)}")
    if args.limit is not None:
        samples = samples[: args.limit]
    requested_count = len(samples)
    print(f"共加载 {requested_count} 个样本")

    if not args.resume:
        for path in [
            output_predictions,
            output_eval,
            legacy_eval,
            output_summary,
            output_eval.with_name("evaluation_summary.json"),
            output_eval.with_name("baseline_score.json"),
        ]:
            if path.exists():
                path.unlink()
    completed_post_ids = load_completed_post_ids(output_predictions) if args.resume else set()
    if completed_post_ids:
        before = len(samples)
        samples = [sample for sample in samples if str(sample.get("post_id")) not in completed_post_ids]
        print(f"Resume enabled: skipped {before - len(samples)} completed samples; remaining {len(samples)}")

    max_workers = max(1, int(max_workers))
    sample_map = {}
    if args.mode != "generate-only":
        sample_map = {sample.sample_id: sample for sample in DatasetParser(Path(input_dataset), IMAGE_ROOT).load()}
    print(
        f"[qwen_eval] pending={len(samples)} max_workers={max_workers} mode={args.mode} "
        f"predictions={output_predictions} eval={output_eval}"
    )
    start = time.perf_counter()
    if max_workers == 1:
        worker_state = threading.local()
        progress = tqdm(samples, desc="baseline 生成评估", unit="sample")
        for sample in progress:
            post_id = str(sample.get("post_id"))
            progress.set_postfix_str(f"id={post_id}")
            result = process_one_sample_with_worker_client(worker_state, sample, config, sample_map, args.mode)
            _write_outputs(output_predictions, output_eval, result)
            update_progress(progress, result)
    else:
        worker_state = threading.local()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_one_sample_with_worker_client, worker_state, sample, config, sample_map, args.mode): sample
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
                    result = _baseline_prediction_row(post_id, "", "", [str(exc)], 0.0)
                _write_outputs(output_predictions, output_eval, result)
                update_progress(progress, result)
                progress.update(1)
            progress.close()
    elapsed = time.perf_counter() - start

    predictions = _dedupe_predictions(_read_jsonl(output_predictions))
    _rewrite_jsonl(output_predictions, predictions)
    eval_rows = _dedupe_eval_rows(_read_jsonl(output_eval), {row["sample_id"] for row in predictions if is_successful_prediction(row)})
    _rewrite_jsonl(output_eval, eval_rows)
    _rewrite_jsonl(legacy_eval, eval_rows)

    if args.mode == "generate-only":
        summary = _generation_only_summary(predictions)
    else:
        eval_cfg = config.raw.get("evaluation", {})
        summary = summarize_scores(
            eval_rows,
            final_weights=eval_cfg.get("final_weights"),
        )
        summary["error_analysis"] = build_error_analysis(eval_rows)
        write_json(
            output_eval.with_name("evaluation_summary.json"),
            summarize_scores(
                eval_rows,
                final_weights=eval_cfg.get("final_weights"),
            ),
        )
    hard_failed = [row for row in predictions if is_hard_failed_prediction(row)]
    warning_samples = [row for row in predictions if row.get("errors") and not is_hard_failed_prediction(row)]
    failed_ids = [row["sample_id"] for row in hard_failed]
    _write_failed_ids(output_predictions.parent, failed_ids)
    summary.update(
        {
            "requested_samples": requested_count,
            "new_samples": len(samples),
            "completed": len([row for row in predictions if is_successful_prediction(row)]),
            "hard_failed": len(hard_failed),
            "warning_samples": len(warning_samples),
            "successful_samples": len(eval_rows) if args.mode != "generate-only" else len([row for row in predictions if is_successful_prediction(row)]),
            "generation_model": GENERATION_MODEL,
            "evaluation_model": config.judge_model if args.mode == "composite" else None,
            "evaluation_mode": args.mode,
            "input_dataset": str(input_dataset),
            "output_predictions": str(output_predictions),
            "output_eval": str(output_eval),
            "failed_ids_file": str(output_predictions.parent.with_name(f"{output_predictions.parent.name}_failed_ids.txt")),
            "resume": args.resume,
            "max_workers": max_workers,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "request_max_retries": REQUEST_MAX_RETRIES,
            "elapsed_seconds": round(elapsed, 4),
        }
    )

    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 评估汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n详细评估结果已保存到 {output_eval}")
    print(f"汇总报告已保存到 {output_summary}")


def _write_outputs(predictions_path: Path, eval_path: Path, result: Dict[str, Any]) -> None:
    append_jsonl(predictions_path, _prediction_payload(result))
    eval_payload = _eval_payload(result)
    if eval_payload is not None:
        append_jsonl(eval_path, eval_payload)


def _prediction_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": str(row.get("sample_id") or row.get("post_id") or ""),
        "question": row.get("question", ""),
        "answer": row.get("answer") or row.get("generated_answer") or "",
        "tools_used": row.get("tools_used", ["qwen_baseline"]),
        "web_searched": bool(row.get("web_searched", False)),
        "tool_trace": row.get("tool_trace", []),
        "reasoning_summary": row.get("reasoning_summary", "qwen_eval baseline direct multimodal generation."),
        "elapsed_seconds": row.get("elapsed_seconds", 0.0),
        "token_usage": row.get("token_usage", {}),
        "errors": row.get("errors", []),
        "plan": row.get("plan"),
    }


def _eval_payload(row: Dict[str, Any]) -> Dict[str, Any] | None:
    if "final_score" not in row or is_hard_failed_prediction(row):
        return None
    return {
        "sample_id": str(row.get("sample_id") or row.get("post_id") or ""),
        "semantic_similarity": row.get("semantic_similarity", {}),
        "llm_judge": row.get("llm_judge", {}),
        "scoring_points": row.get("scoring_points", {}),
        "final_score": row.get("final_score", 0.0),
        "claim_rouge_l": row.get("claim_rouge_l", {}),
        "technical_entity_match": row.get("technical_entity_match", {}),
        "fully_correct": row.get("fully_correct", False),
        "error_analysis": row.get("error_analysis", {}),
        "elapsed_seconds": row.get("elapsed_seconds", 0.0),
    }


def is_generation_failure(answer: str) -> bool:
    normalized = str(answer or "").strip()
    return not normalized or "[ERROR]" in normalized or "生成失败" in normalized


def is_successful_prediction(row: Dict[str, Any]) -> bool:
    return not is_hard_failed_prediction(row)


def is_hard_failed_prediction(row: Dict[str, Any]) -> bool:
    return bool(row.get("errors")) or is_generation_failure(str(row.get("answer") or row.get("generated_answer") or ""))


def _failed_prediction_ids(predictions_path: Path) -> list[str]:
    rows = _dedupe_predictions(_read_jsonl(predictions_path))
    return [row["sample_id"] for row in rows if is_hard_failed_prediction(row)]


def _filter_retry_failed_samples(samples: List[Dict[str, Any]], failed_ids: list[str]) -> List[Dict[str, Any]]:
    if not failed_ids:
        return []
    return filter_items_by_sample_ids(samples, failed_ids, lambda sample: str(sample.get("post_id")))


def _dedupe_predictions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: dict[str, Dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        payload = _prediction_payload(row)
        sample_id = payload["sample_id"]
        if not sample_id:
            continue
        if sample_id not in latest:
            order.append(sample_id)
            latest[sample_id] = payload
            continue
        if is_successful_prediction(payload) or is_hard_failed_prediction(latest[sample_id]):
            latest[sample_id] = payload
    return [latest[sample_id] for sample_id in order]


def _dedupe_eval_rows(rows: List[Dict[str, Any]], success_ids: set[str]) -> List[Dict[str, Any]]:
    latest: dict[str, Dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        payload = _eval_payload(row)
        if payload is None:
            continue
        sample_id = payload["sample_id"]
        if sample_id not in success_ids:
            continue
        if sample_id not in latest:
            order.append(sample_id)
        latest[sample_id] = payload
    return [latest[sample_id] for sample_id in order]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _rewrite_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp_path, path)


def _write_failed_ids(output_dir: Path, failed_ids: List[str]) -> None:
    content = "\n".join(failed_ids)
    for path in [output_dir / "failed_sample_ids.txt", output_dir.with_name(f"{output_dir.name}_failed_ids.txt")]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def _generation_only_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total": len(rows),
        "generated_samples": len([row for row in rows if is_successful_prediction(row)]),
    }


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
