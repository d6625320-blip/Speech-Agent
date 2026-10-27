"use strict";
const $ = id => document.getElementById(id);
let sessionId = crypto.randomUUID(), busy = false, recording = null;
let activePlayback = null, streamAbort = null;
const objectURLs = new Set();
const emotionNames = {angry: "愤怒", disgust: "厌恶", fear: "恐惧", happy: "开心", neutral: "中性", sad: "悲伤", surprised: "惊讶"};
const stateNames = {known_emotion: "情绪证据通过筛选", uncertain_emotion: "情绪不确定（未通过筛选或与自述冲突）", unknown_emotion: "未获得语音情绪结果", no_audio: "未检测到清晰声音", unclear_speech: "未识别到清晰话语"};
const strategyNames = {neutral_fallback: "按文本自然回应，不猜测情绪", text_only: "仅依据文字回应", speech_label: "参考语音候选类别（D1）", share_positive: "回应积极表达", support: "温和支持", reassure: "安定与澄清", deescalate: "平和回应", respect_boundary: "尊重不适与边界", clarify: "澄清新信息", task_focused: "直接协助任务", immediate_safety: "优先处理当前安全问题"};
const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "未提供";
const emotionName = value => value ? `${emotionNames[value] || value}（${value}）` : "未采用";

function status(text, error = false) { $("status").textContent = text; $("status").className = error ? "error" : ""; }
function headers() { const token = $("token").value.trim(); return token ? {Authorization: `Bearer ${token}`} : {}; }
function setBusy(value) {
  busy = value;
  for (const id of ["send", "file", "reset", "mode", "tts", "stream-tts", "record", "microphone", "refresh-microphones"]) $(id).disabled = value;
}
function message(role, text) {
  document.querySelector(".empty")?.remove();
  const node = document.createElement("div"); node.className = `message ${role}`;
  const title = document.createElement("strong"); title.textContent = role === "user" ? "你" : "助手";
  const content = document.createElement("span"); content.className = "message-text"; content.textContent = text;
  node.append(title, content); $("conversation").append(node); return node;
}
function localAudio(node, blob, description) {
  const caption = document.createElement("span"); caption.className = "audio-caption"; caption.textContent = description;
  const player = document.createElement("audio"); player.controls = true; player.preload = "metadata";
  player.addEventListener("play", () => { activePlayback?.pause().catch(() => {}); });
  player.setAttribute("aria-label", description);
  const url = URL.createObjectURL(blob); objectURLs.add(url); player.src = url;
  node.append(caption, player); return {player, caption};
}
function gateChecks(result) {
  const speech = result.emotion.speech, gate = result.emotion_gate;
  if (!speech.available || !gate) return [];
  return [
    {name: "最高类别分数", value: speech.confidence, threshold: gate.confidence_threshold, direction: "≥"},
    {name: "前两类分数差", value: speech.margin, threshold: gate.margin_threshold, direction: "≥"},
    {name: "归一化熵", value: speech.normalized_entropy, threshold: gate.max_normalized_entropy, direction: "≤"},
  ].filter(check => Number.isFinite(check.value) && Number.isFinite(check.threshold)).map(check => ({...check,
    passed: check.direction === "≥" ? check.value >= check.threshold : check.value <= check.threshold}));
}
function emotionReason(result, checks) {
  const reason = result.emotion.reason;
  if (reason === "text_input" && Number.isFinite(result.usage?.input_audio_seconds)) {
    return result.mode === "text_only" ? "本轮为 D0 仅文本模式：音频只做语音识别，没有运行语音情感专家。" : "本轮未运行语音情感专家，请检查专家是否启用。";
  }
  if (reason === "speech_gate_failed") {
    const failed = checks.filter(check => !check.passed).map(check => `${check.name} ${percent(check.value)}，要求 ${check.direction} ${percent(check.threshold)}`);
    return failed.length ? failed.join("；") : "语音情绪未达到策略层的筛选条件；请查看完整诊断数据。";
  }
  const reasons = {explicit_self_report_conflict: "语音预测与识别文字中的明确自述冲突，优先尊重自述。", invalid_expert_distribution: "专家分数分布无效，未采用结果。", speech_gate_passed: "分数、类别差距和熵均通过筛选。", text_input: "本轮为文字输入，没有运行语音情感专家；未配置独立的文字情感分类器。", disabled: "本轮未运行语音情感专家（例如 D0 仅文本模式或专家已关闭）。", expert_unavailable: "语音情感专家运行失败，请查看服务器日志。", silent_audio: "输入音频被判为静音。", empty_transcript: "未识别到文字，本轮没有运行语音情感专家。"};
  return reasons[reason] || reason || "未提供原因";
}
function renderEmotion(result) {
  const state = result.emotion, speech = state.speech, checks = gateChecks(result), root = $("emotion-summary");
  root.replaceChildren();
  const intro = document.createElement("p");
  intro.textContent = speech.available ? "来源：你训练的语音情感专家，分析录音的声学特征；不是文字情感分类。文字仅用于少量明确自述的冲突检查。" : "本轮没有可用的语音情感预测；LLM 仍可根据文字内容回应。";
  root.append(intro);
  const list = document.createElement("dl"); list.className = "metric-list";
  const add = (key, value) => { const term = document.createElement("dt"), description = document.createElement("dd"); term.textContent = key; description.textContent = value; list.append(term, description); };
  add("情绪筛选状态", stateNames[state.status] || state.status);
  add("专家原始候选", speech.available ? emotionName(speech.label) : "未运行或不可用");
  add("最高类别分数", speech.available ? `${percent(speech.confidence)}（${speech.calibrated ? "已应用温度缩放" : "未校准，不等于预测正确率"}）` : "无");
  add("本轮参考类别", result.mode === "speech_label" && speech.available ? `${emotionName(speech.label)}（D1 直接参考候选）` : emotionName(state.label));
  add("判定原因", emotionReason(result, checks));
  add("回应策略", strategyNames[result.policy.strategy] || result.policy.strategy);
  add("输入音频时长", Number.isFinite(result.usage?.input_audio_seconds) ? `${result.usage.input_audio_seconds.toFixed(2)} 秒` : "文字输入");
  const timingNames = {audio: "音频预处理", asr: "语音识别", speech: "语音情绪", llm: "LLM", tts: "语音合成", tts_first_audio: "TTS首块（合成请求起）", total: "总计"};
  add("耗时", Object.entries(result.timings_ms || {}).map(([key, value]) => `${timingNames[key] || key} ${(value / 1000).toFixed(2)} 秒`).join(" · "));
  root.append(list);
  if (result.usage?.tts?.streaming) {
    const streaming = document.createElement("p"), tts = result.usage.tts;
    streaming.textContent = `流式 TTS：${tts.backend || "未知后端"} · 输出 ${tts.audio_seconds.toFixed(2)} 秒 · 实时因子 RTF ${tts.rtf.toFixed(2)}（合成耗时/音频时长，越小越快）`;
    root.append(streaming);
  }
  if (checks.length) {
    const gates = document.createElement("p");
    gates.textContent = "筛选条件：" + checks.map(check => `${check.passed ? "通过" : "未通过"} ${check.name} ${percent(check.value)}（要求 ${check.direction} ${percent(check.threshold)}）`).join("；");
    if (checks.some(check => !check.passed)) gates.className = "gate-failed";
    root.append(gates);
  }
  const probabilities = Object.entries(speech.probabilities || {}).sort((a, b) => b[1] - a[1]);
  if (speech.available && probabilities.length) {
    const table = document.createElement("table"); table.className = "probability-table";
    const caption = document.createElement("caption"); caption.textContent = "七类声学分类分数（从高到低）"; table.append(caption);
    const heading = document.createElement("tr");
    for (const text of ["类别", "分数", "分布"]) { const th = document.createElement("th"); th.scope = "col"; th.textContent = text; heading.append(th); }
    const head = document.createElement("thead"); head.append(heading); table.append(head);
    const body = document.createElement("tbody");
    for (const [label, probability] of probabilities) {
      const row = document.createElement("tr"), name = document.createElement("td"), score = document.createElement("td"), bar = document.createElement("td"), meter = document.createElement("progress");
      name.textContent = emotionName(label); score.textContent = percent(probability); meter.max = 1; meter.value = probability; meter.setAttribute("aria-label", `${emotionName(label)} ${percent(probability)}`);
      bar.append(meter); row.append(name, score, bar); body.append(row);
    }
    table.append(body); root.append(table);
  }
  if (Number.isFinite(result.usage?.input_audio_seconds) && result.usage.input_audio_seconds < 3) {
    const hint = document.createElement("p"); hint.className = "hint"; hint.textContent = "这段录音不足 3 秒。建议另用几段 3–10 秒的清晰完整话语对比，结合候选类别和分数判断；时长更长不保证识别正确。"; root.append(hint);
  }
  $("metrics").textContent = JSON.stringify({mode: result.mode, emotion: state, emotion_gate: result.emotion_gate,
    gate_checks: checks, policy: result.policy, timings_ms: result.timings_ms, usage: result.usage,
    warnings: result.warnings, trace_id: result.trace_id}, null, 2);
}
async function api(url, options = {}) {
  const response = await fetch(url, {...options, headers: {...headers(), ...options.headers}});
  if (!response.ok) {
    let detail = `请求失败（${response.status}）`;
    try { const body = await response.json(); if (typeof body.detail === "string") detail = body.detail; } catch {}
    throw new Error(detail);
  }
  return response;
}
async function display(result, inputNode = null, inputAudio = null) {
  if (inputNode) inputNode.querySelector(".message-text").textContent = result.transcript || "（未识别到清晰话语）";
  if (inputAudio && result.input_audio_url) {
    let loadingFallback = false;
    const fallback = async () => {
      if (loadingFallback) return; loadingFallback = true;
      try {
        const response = await api(result.input_audio_url), url = URL.createObjectURL(await response.blob()); objectURLs.add(url);
        inputAudio.player.src = url;
        inputAudio.caption.textContent = "输入录音（分析用 WAV，16 kHz 单声道；点击播放核对识别结果）";
      } catch { inputAudio.caption.textContent = "输入录音回放失败；可在服务器 outputs 对应轮次中查看 input.wav。"; }
    };
    inputAudio.player.addEventListener("error", fallback, {once: true});
    if (inputAudio.player.error) await fallback();
  }
  const node = message("assistant", result.reply);
  renderEmotion(result);
  if (result.audio_url) {
    try {
      const response = await api(result.audio_url);
      const {player} = localAudio(node, await response.blob(), "助手语音回复");
      if (![...document.querySelectorAll("audio")].some(other => other !== player && !other.paused)) player.play().catch(() => {});
    } catch { node.append(document.createTextNode("\n音频加载失败，文字回复已保留。")); }
  }
  status(result.warnings.length ? result.warnings.join("；") : (inputNode ? "回复完成；可点击输入录音的播放按钮核对原声。" : "回复完成"));
  node.scrollIntoView({behavior: "smooth", block: "nearest"});
  return node;
}

function prepareStreamingPlayback() {
  activePlayback?.stop(); activePlayback = null;
  if (!$("tts").checked || !$("stream-tts").checked) return null;
  try {
    const player = new PCMStreamPlayer(); activePlayback = player;
    // Call in the user's click/change handler before ASR/LLM awaits so browsers
    // can grant audio playback. A manual resume button remains available.
    player.resume().catch(() => {}); return player;
  } catch { status("浏览器无法启用流式播放，本轮将等待完整语音。", true); return null; }
}
async function streamSpeech(result, node, player) {
  const controller = new AbortController(); streamAbort = controller;
  const caption = document.createElement("p"); caption.className = "hint"; caption.textContent = "正在生成语音块…若未自动出声，请点击“允许播放”。";
  const resume = document.createElement("button"); resume.type = "button"; resume.className = "secondary"; resume.textContent = "允许播放";
  resume.addEventListener("click", () => {
    for (const audio of document.querySelectorAll("audio")) audio.pause();
    player.resume().catch(() => { caption.textContent = "浏览器仍未允许播放，请在生成结束后使用音频播放器。"; });
  });
  const stop = document.createElement("button"); stop.type = "button"; stop.className = "secondary"; stop.textContent = "停止语音";
  stop.addEventListener("click", () => { controller.abort(); player.stop(); stop.disabled = true; resume.disabled = true; caption.textContent = "已停止语音。"; });
  node.append(caption, resume, stop);
  for (const audio of document.querySelectorAll("audio")) audio.pause();
  let rate = null, sequence = 0, completed = false, received = false;
  try {
    status("文字回复已返回，正在接收并播放语音块…");
    const response = await api(`/api/tts/${result.trace_id}/stream`, {method: "POST", signal: controller.signal});
    await readNDJSON(response, event => {
      if (completed) throw new Error("语音完成后收到多余数据");
      if (event.type === "error") throw new Error(event.detail || "流式语音失败");
      if (event.type === "start") {
        if (rate !== null || event.sample_rate !== 24000 || event.channels !== 1 || event.encoding !== "pcm_s16le") throw new Error("语音流格式不正确");
        rate = event.sample_rate;
      } else if (event.type === "audio") {
        if (rate === null || event.sequence !== sequence++) throw new Error("语音块顺序不正确");
        player.enqueue(event.pcm, rate); received = true; caption.textContent = "正在接收语音；已收到的音频块按顺序播放。";
      } else if (event.type === "done") {
        if (!event.cached && !received) throw new Error("未收到可播放的语音");
        completed = true; result = event.result; renderEmotion(result);
      } else throw new Error("未知语音流事件");
    });
    if (!completed) throw new Error("语音流提前结束；文字回复已保留");
    player.finish();
    if (result.audio_url) {
      try {
        const response = await api(result.audio_url);
        localAudio(node, await response.blob(), "完整语音回复（可重播）");
      } catch { caption.textContent = "已收到语音块，完整音频暂时无法加载。"; }
    }
    caption.textContent = "语音生成完成。已排队的音频将继续播放，也可使用下方播放器重播。";
    status(result.warnings.length ? result.warnings.join("；") : "回复完成（流式语音）");
  } catch (error) {
    player.stop(); resume.disabled = true;
    const detail = error.name === "AbortError" ? "已停止语音，文字回复已保留。" : `${error.message}；文字回复已保留。`;
    caption.textContent = detail; status(detail, error.name !== "AbortError");
  } finally { if (!completed) stop.disabled = true; if (streamAbort === controller) streamAbort = null; }
}
$("text-form").addEventListener("submit", async event => {
  event.preventDefault(); if (busy || recording) return;
  const text = $("text").value.trim(); if (!text) return;
  const playback = prepareStreamingPlayback();
  setBusy(true); status("正在生成回复…");
  try {
    const result = await (await api("/api/text", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text, session_id: sessionId, mode: $("mode").value, synthesize: $("tts").checked && !playback})})).json();
    message("user", text); $("text").value = ""; const node = await display(result);
    if (playback) await streamSpeech(result, node, playback);
  } catch (error) { playback?.stop(); status(error.message, true); } finally { setBusy(false); }
});
async function upload(blob, preparedPlayback = null) {
  if (blob.size > 10 * 1024 * 1024) { status("音频超过 10 MiB，请缩短或转换为 16kHz 单声道 WAV。", true); return; }
  const playback = preparedPlayback || prepareStreamingPlayback();
  setBusy(true); status("正在识别语音、分析情绪并生成回复；你可以先播放输入录音。");
  const inputNode = message("user", "正在识别这段录音…");
  const inputAudio = localAudio(inputNode, blob, `输入录音${blob.name ? `：${blob.name}` : ""}（点击播放核对原声）`);
  inputAudio.player.addEventListener("error", () => {
    inputAudio.caption.textContent = "浏览器无法播放当前格式，识别完成后将尝试加载分析用 WAV 音频。";
  }, {once: true});
  try {
    const query = new URLSearchParams({session_id: sessionId, mode: $("mode").value, synthesize: String($("tts").checked && !playback)});
    const result = await (await api(`/api/turn?${query}`, {method: "POST", headers: {"Content-Type": "application/octet-stream"}, body: blob})).json();
    const node = await display(result, inputNode, inputAudio);
    if (playback) await streamSpeech(result, node, playback);
  } catch (error) { playback?.stop(); inputNode.querySelector(".message-text").textContent = "本轮处理失败；输入录音仍可回放。"; status(error.message, true); }
  finally { setBusy(false); $("file").value = ""; }
}
$("file").addEventListener("change", () => { if (!busy && !recording && $("file").files[0]) upload($("file").files[0]); });

function microphoneError(error) {
  const messages = {
    NotFoundError: "未找到可用麦克风。请检查打开此网页的电脑的 Windows 声音“输入”设备。若浏览器在远程桌面内，请启用远程音频录制/麦克风重定向后重新连接；也可先上传音频文件。",
    DevicesNotFoundError: "未找到可用麦克风。请检查 Windows 声音输入设备或远程桌面的麦克风重定向。",
    NotAllowedError: "麦克风访问被拒绝。请允许浏览器地址栏中的麦克风权限，并检查 Windows 麦克风隐私设置。",
    PermissionDeniedError: "麦克风权限被拒绝，请检查浏览器及 Windows 麦克风权限。",
    NotReadableError: "麦克风无法读取，可能被其他程序占用或设备驱动异常。请关闭占用程序后重试。",
    OverconstrainedError: "所选麦克风已不可用，或不支持录音设置。请点击“检测麦克风”后重新选择。",
    SecurityError: "浏览器禁止访问麦克风，请使用 HTTPS 或 localhost，并检查麦克风权限。",
  };
  return messages[error.name] || `录音失败：${error.message || error.name || "未知错误"}`;
}
function microphoneSupported() {
  if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext) {
    status("录音需要 HTTPS 或 localhost。请使用 SSH 转发后访问 localhost，或上传音频文件。", true); return false;
  }
  return true;
}
async function listMicrophones(requestPermission = false) {
  if (!microphoneSupported()) return;
  if (requestPermission) {
    const probe = await navigator.mediaDevices.getUserMedia({audio: true});
    probe.getTracks().forEach(track => track.stop());
  }
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === "audioinput");
  const select = $("microphone"), selected = select.value;
  select.replaceChildren();
  const defaultOption = document.createElement("option"); defaultOption.value = ""; defaultOption.textContent = "系统默认麦克风"; select.append(defaultOption);
  devices.forEach((device, index) => {
    if (!device.deviceId) return;
    const option = document.createElement("option"); option.value = device.deviceId; option.textContent = device.label || `麦克风 ${index + 1}`; select.append(option);
  });
  select.value = devices.some(device => device.deviceId === selected) ? selected : "";
  $("microphone-info").textContent = devices.length ? `检测到 ${devices.length} 个音频输入条目（系统默认项可能与实际设备重复）。录音使用当前浏览器所在电脑的输入设备。` : "浏览器未列出音频输入设备。请先允许麦克风权限；远程桌面内还需检查麦克风重定向。";
  return devices.length;
}
$("refresh-microphones").addEventListener("click", async () => {
  if (busy || recording || !microphoneSupported()) return;
  setBusy(true); status("正在申请麦克风权限并检测设备…");
  try { const count = await listMicrophones(true); status(count ? "麦克风检测完成，请选择输入设备后开始录音。" : "未列出麦克风，请检查 Windows 输入设备或远程桌面设置。", !count); }
  catch (error) { const detail = microphoneError(error); $("microphone-info").textContent = detail; status(detail, true); }
  finally { setBusy(false); }
});

async function encodeWav(chunks, sampleRate) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  if (!length) throw new Error("录音为空，请重新录音。");
  const source = new Float32Array(length); let offset = 0;
  for (const chunk of chunks) { source.set(chunk, offset); offset += chunk.length; }
  const context = new OfflineAudioContext(1, Math.ceil(length * 16000 / sampleRate), 16000);
  const buffer = context.createBuffer(1, length, sampleRate); buffer.copyToChannel(source, 0);
  const player = context.createBufferSource(); player.buffer = buffer; player.connect(context.destination); player.start();
  const values = (await context.startRendering()).getChannelData(0);
  const raw = new ArrayBuffer(44 + values.length * 2), view = new DataView(raw);
  function ascii(position, text) { for (let i = 0; i < text.length; i++) view.setUint8(position + i, text.charCodeAt(i)); }
  ascii(0, "RIFF"); view.setUint32(4, raw.byteLength - 8, true); ascii(8, "WAVE"); ascii(12, "fmt "); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, 16000, true); view.setUint32(28, 32000, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); ascii(36, "data"); view.setUint32(40, values.length * 2, true);
  for (let i = 0; i < values.length; i++) { const v = Math.max(-1, Math.min(1, values[i])); view.setInt16(44 + 2 * i, v < 0 ? v * 32768 : v * 32767, true); }
  return new Blob([raw], {type: "audio/wav"});
}
async function stopRecording() {
  const current = recording; if (!current) return;
  recording = null; setBusy(true); clearTimeout(current.timer);
  const playback = prepareStreamingPlayback();
  current.stream.getTracks().forEach(track => track.stop());
  try {
    current.node.disconnect(); current.source.disconnect(); await current.context.close();
    await upload(await encodeWav(current.chunks, current.rate), playback);
  } catch (error) { playback?.stop(); status(error.message, true); }
  finally { $("record").textContent = "开始录音"; setBusy(false); }
}
$("record").addEventListener("click", async () => {
  if (recording) { await stopRecording(); return; }
  if (busy || !microphoneSupported()) return;
  let stream, context;
  try {
    setBusy(true);
    const audio = {channelCount: {ideal: 1}, echoCancellation: false, noiseSuppression: false, autoGainControl: false};
    if ($("microphone").value) audio.deviceId = {exact: $("microphone").value};
    stream = await navigator.mediaDevices.getUserMedia({audio});
    await listMicrophones();
    context = new AudioContext();
    if (!context.audioWorklet) throw new Error("此浏览器不支持 AudioWorklet 录音，请更新 Edge/Chrome，或上传音频文件。");
    await context.audioWorklet.addModule("/static/recorder-worklet.js"); await context.resume();
    const source = context.createMediaStreamSource(stream), node = new AudioWorkletNode(context, "capture-processor"), chunks = [];
    node.port.onmessage = event => chunks.push(event.data); source.connect(node); node.connect(context.destination);
    recording = {stream, context, source, node, chunks, rate: context.sampleRate, timer: setTimeout(stopRecording, 115000)};
    $("record").disabled = false; $("record").textContent = "停止并发送"; status("录音中…点击停止后发送，最长 115 秒。发送后可回放本轮录音。");
  } catch (error) {
    stream?.getTracks().forEach(track => track.stop());
    if (context && context.state !== "closed") await context.close().catch(() => {});
    setBusy(false); const detail = microphoneError(error); $("microphone-info").textContent = detail; status(detail, true);
  }
});
$("reset").addEventListener("click", async () => {
  if (busy || recording) return;
  try { await api(`/api/session/${sessionId}`, {method: "DELETE"}); }
  catch (error) { status(error.message, true); return; }
  activePlayback?.stop(); activePlayback = null;
  for (const player of document.querySelectorAll("audio")) player.pause();
  sessionId = crypto.randomUUID(); $("conversation").replaceChildren(); $("metrics").textContent = "新对话。"; $("emotion-summary").textContent = "发送一条消息后显示。";
  for (const url of objectURLs) URL.revokeObjectURL(url); objectURLs.clear(); status("已开始新对话");
});
$("health").addEventListener("click", async () => {
  try { const result = await (await api("/api/health")).json(); $("models").textContent = Object.entries(result.models).map(([k, v]) => `${k}: ${v.model}`).join(" · "); status("主服务连接正常；模型就绪情况请在服务器运行 doctor --probe。"); }
  catch (error) { status(error.message, true); }
});
window.addEventListener("beforeunload", () => { streamAbort?.abort(); activePlayback?.stop(); recording?.stream.getTracks().forEach(track => track.stop()); for (const url of objectURLs) URL.revokeObjectURL(url); });
