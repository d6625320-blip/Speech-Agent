from __future__ import annotations

import math

from ..schemas import EmotionState, LABELS, SpeechEmotion


def assemble_state(speech: SpeechEmotion, config: dict, transcript: str = "") -> EmotionState:
    if not speech.available:
        return EmotionState("unknown_emotion", speech, reason=speech.reason)
    p = speech.probabilities
    valid = (
        speech.label in LABELS and set(p) == set(LABELS)
        and all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in p.values())
        and abs(sum(p.values()) - 1) < 1e-4
        and all(v is not None and math.isfinite(v) and 0 <= v <= 1 for v in
                (speech.confidence, speech.margin, speech.normalized_entropy))
    )
    if not valid:
        return EmotionState("uncertain_emotion", speech, reason="invalid_expert_distribution")
    reliable = (
        speech.confidence >= config["confidence_threshold"]
        and speech.margin >= config["margin_threshold"]
        and speech.normalized_entropy <= config["max_normalized_entropy"]
    )
    # Only explicit first-person expressions; this is not a text emotion classifier.
    positive = any(s in transcript for s in ("我很开心", "我很高兴", "我很快乐"))
    negative = any(s in transcript for s in ("我很难过", "我很伤心", "我很害怕", "我很生气"))
    conflict = reliable and ((speech.label == "happy" and negative) or
                             (speech.label in {"sad", "angry", "fear", "disgust"} and positive))
    return EmotionState(
        "known_emotion" if reliable and not conflict else "uncertain_emotion", speech,
        reliable and not conflict, speech.label if reliable and not conflict else None,
        conflict, "explicit_self_report_conflict" if conflict else ("speech_gate_passed" if reliable else "speech_gate_failed"),
    )
