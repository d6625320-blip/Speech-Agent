"""Explicit real-model evaluation; no synthetic scores, training or automatic cloud fallback."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emotion_agent.config import load_settings
from emotion_agent.evaluation import character_errors, latency_summary
from emotion_agent.pipeline.runner import Agent


def main():
    parser = argparse.ArgumentParser(description="批量 ASR CER、D0/D1/D2-S 消融、延迟与可选 TTS 回转写")
    parser.add_argument("--config")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("text_only", "speech_label", "policy"), default=["policy"])
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--tts-roundtrip", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.no_tts and args.tts_roundtrip:
        parser.error("--tts-roundtrip 不能与 --no-tts 同时使用")
    settings = load_settings(args.config)
    manifest = args.manifest.resolve()
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not records:
        parser.error("评测清单为空")
    # Validate every input before starting potentially billable cloud evaluations.
    for record in records:
        if ("audio" in record) == ("text" in record):
            parser.error("每行必须且只能提供 audio / text 中的一项")
        if "audio" in record and not (manifest.parent / record["audio"]).is_file():
            parser.error(f"音频不存在：{record['audio']}")
    output = args.output or settings.path(settings.data["output_dir"]) / f"evaluation_{int(time.time())}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, buckets, sessions = [], defaultdict(list), {}
    last_index = {str(record.get("conversation_id", f"single_{i}")): i for i, record in enumerate(records)}
    agent = Agent(settings)
    try:
        for index, record in enumerate(records):
            for mode in args.modes:
                print(f"[{index + 1}/{len(records)}] {mode}", flush=True)
                conversation = str(record.get("conversation_id", f"single_{index}"))
                key = (mode, conversation)
                kwargs = {"mode": mode, "synthesize": not args.no_tts, "session_id": sessions.get(key)}
                if "audio" in record:
                    kwargs["audio"] = (manifest.parent / record["audio"]).resolve()
                else:
                    kwargs["text"] = record["text"]
                try:
                    result = agent.run(**kwargs)
                    sessions[key] = result.session_id
                    row = {"index": index, "mode": mode, "conversation_id": conversation, **result.to_dict()}
                    if "reference" in record and "audio" in record:
                        errors, length = character_errors(record["reference"], result.transcript)
                        row["asr_errors"], row["reference_chars"] = errors, length
                    expected = record.get("expected_strategy", {}).get(mode)
                    if expected:
                        row["strategy_match"] = result.policy.strategy == expected
                    if args.tts_roundtrip and result.audio_path:
                        transcription = agent.asr.transcribe(Path(result.audio_path))
                        errors, length = character_errors(result.reply, transcription.text)
                        row["tts_errors"], row["tts_reference_chars"] = errors, length
                    buckets[mode].append(row)
                except Exception as exc:
                    row = {"index": index, "mode": mode, "conversation_id": conversation,
                           "status": "error", "error_type": type(exc).__name__}
                    buckets[mode].append(row)
                if index == last_index[conversation] and key in sessions:
                    agent.memory.clear(sessions.pop(key))
                rows.append(row)
                # Persist progress after each turn; interruption does not erase completed runs.
                output.with_suffix(".jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    finally:
        agent.close()
    summary = {}
    for mode, values in buckets.items():
        ok = [v for v in values if v["status"] != "error"]
        all_stages = {name for v in ok for name in v["timings_ms"]}
        item = {"turns": len(values), "completed": len(ok), "degraded": sum(v["status"] == "degraded" for v in ok),
                "latency": {name: latency_summary([v["timings_ms"][name] for v in ok if name in v["timings_ms"]]) for name in sorted(all_stages)}}
        for key, numerator, denominator in (("asr_cer", "asr_errors", "reference_chars"), ("tts_roundtrip_cer", "tts_errors", "tts_reference_chars")):
            length = sum(v.get(denominator, 0) for v in ok)
            item[key] = sum(v.get(numerator, 0) for v in ok) / length if length else None
        matches = [v["strategy_match"] for v in ok if "strategy_match" in v]
        item["strategy_match_rate"] = sum(matches) / len(matches) if matches else None
        summary[mode] = item
    output.write_text(json.dumps({"summary": summary, "rows": rows,
        "notes": ["包含首轮冷启动，不等于稳态延迟", "CER 采用 NFKC、小写、去空白及标点后的微平均",
                  "独立多轮会话按 conversation_id 分组；改变模式时使用独立记忆", "策略匹配率不是回复质量评分，真实质量仍需人工盲评"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    if any(v["status"] == "error" for v in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
