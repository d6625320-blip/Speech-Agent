from ..schemas import DialoguePolicy, EmotionState


def choose_policy(state: EmotionState, mode: str, text: str) -> DialoguePolicy:
    # Deliberately small deterministic emergency guard; not a clinical classifier.
    if any(term in text for term in ("我要自杀", "我现在要自杀", "我准备自杀", "我马上要伤害自己")):
        return DialoguePolicy(
            "immediate_safety", "优先处理明确的当前人身危险。", "语气平稳、清晰，不催促。",
            "请先暂停行动，远离可能伤害自己的物品，尽量到有人的地方，并马上联系身边可信任的人陪着你。如果危险迫在眉睫，请立即联系当地紧急服务。你现在身边有人可以陪你吗？",
        )
    if mode == "text_only":
        return DialoguePolicy("text_only", "仅根据用户文本和上下文自然回应。", "自然、清晰地说话。")
    if mode == "speech_label":
        hint = state.speech.label if state.speech.available else "unknown"
        return DialoguePolicy("speech_label", f"声学分类参考为 {hint}，可能出错，用户自述优先。", "自然、清晰地说话。")
    if not state.reliable:
        return DialoguePolicy("neutral_fallback", "情绪证据不足或与自述冲突。根据文本回应，不猜测情绪。", "自然、平和、清晰地说话。")
    mapping = {
        "angry": ("deescalate", "平和地确认用户关切，避免争辩，给出一个可执行步骤。", "语气冷静、友好，语速稍慢，不要模仿愤怒。"),
        "sad": ("support", "如内容适合，简短承接感受，给予选择，不强行积极。", "语气温和、关切，语速稍慢，保持自然。"),
        "fear": ("reassure", "先澄清眼前问题，提供具体、简单的下一步，避免无依据保证。", "语气安定、清晰，语速稍慢。"),
        "disgust": ("respect_boundary", "尊重用户的不适与边界，围绕实际问题提供选项。", "语气平稳、尊重，不夸张表达。"),
        "happy": ("share_positive", "适度回应积极表达，继续帮助完成用户目标。", "语气轻快、友好，不要过分兴奋。"),
        "surprised": ("clarify", "自然回应新信息，必要时简短澄清。", "语气自然、清晰，带少量好奇。"),
        "neutral": ("task_focused", "直接帮助用户完成任务，表达简洁。", "自然、清晰地说话。"),
    }
    return DialoguePolicy(*mapping[state.label])
