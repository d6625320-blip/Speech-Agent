// Browser logic tests with a small DOM fixture; no microphone or model is used.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const web = path.resolve(__dirname, "../src/emotion_agent/web");

class Element {
  constructor(tag = "div") {
    this.tagName = tag; this.children = []; this.listeners = {}; this.attributes = {};
    this.value = ""; this.className = ""; this._text = ""; this.paused = true; this.files = [];
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this._text = ""; this.children = nodes; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(type, handler, options = {}) { (this.listeners[type] ||= []).push({handler, once: options.once}); }
  async fire(type, event = {}) {
    const listeners = [...(this.listeners[type] || [])];
    for (const item of listeners) {
      if (item.once) this.listeners[type] = this.listeners[type].filter(other => other !== item);
      await item.handler({preventDefault() {}, ...event});
    }
  }
  querySelectorAll(selector) {
    const matches = node => selector.startsWith(".") ? node.className.split(" ").includes(selector.slice(1)) : node.tagName === selector;
    return this.children.flatMap(child => [...(matches(child) ? [child] : []), ...child.querySelectorAll(selector)]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  scrollIntoView() {}
  async play() { this.paused = false; }
  pause() { this.paused = true; }
}

function page() {
  const ids = new Map([...fs.readFileSync(path.join(web, "index.html"), "utf8").matchAll(/id="([^"]+)"/g)].map(match => [match[1], new Element()]));
  ids.get("tts").checked = true; ids.get("mode").value = "policy";
  const document = {
    getElementById(id) { assert.ok(ids.has(id), `HTML is missing #${id}`); return ids.get(id); },
    createElement: tag => new Element(tag), createTextNode(text) { const node = new Element("#text"); node.textContent = text; return node; },
    querySelector() { return null; },
    querySelectorAll(selector) { return [...ids.values()].flatMap(node => node.querySelectorAll(selector)); },
  };
  let nextURL = 0, stopped = 0;
  const revoked = [], requests = [], audioContexts = [];
  const stream = {getTracks: () => [{stop() { stopped++; }}]};
  const context = vm.createContext({document, console, Blob, URLSearchParams, TextDecoder, AbortController, atob,
    crypto: {randomUUID: () => "test-session"},
    URL: {createObjectURL: () => `blob:test-${++nextURL}`, revokeObjectURL: url => revoked.push(url)},
    navigator: {mediaDevices: {async getUserMedia(options) { requests.push(options); return stream; }, async enumerateDevices() { return [{kind: "audioinput", deviceId: "usb", label: "USB 麦克风"}, {kind: "videoinput", deviceId: "camera"}]; }}},
    window: {isSecureContext: true, addEventListener() {}},
    setTimeout: () => 1, clearTimeout() {},
    fetch: async () => { throw new Error("unexpected request"); },
    AudioContext: class {
      constructor() { this.state = "running"; this.sampleRate = 48000; this.currentTime = 0; this.started = []; this.buffers = []; this.audioWorklet = {async addModule() {}}; audioContexts.push(this); }
      async resume() { this.state = "running"; } async suspend() { this.state = "suspended"; } async close() { this.state = "closed"; }
      createMediaStreamSource() { return {connect() {}, disconnect() {}}; }
      createBuffer(channels, length, rate) {
        const data = new Float32Array(length), buffer = {duration: length / rate, getChannelData: () => data};
        this.buffers.push(buffer); return buffer;
      }
      createBufferSource() {
        const context = this;
        return {connect() {}, disconnect() {}, start(time) { context.started.push(time); }, stop() {}};
      }
    },
    AudioWorkletNode: class { constructor() { this.port = {}; } connect() {} disconnect() {} },
  });
  vm.runInContext(fs.readFileSync(path.join(web, "stream-player.js"), "utf8"), context);
  vm.runInContext(fs.readFileSync(path.join(web, "app.js"), "utf8"), context);
  return {context, ids, revoked, requests, audioContexts, get stopped() { return stopped; }, run: code => vm.runInContext(code, context)};
}

function result() {
  return {mode: "policy", transcript: "我今天很开心", reply: "想聊聊发生了什么吗？", warnings: [], trace_id: "a".repeat(32),
    emotion: {status: "uncertain_emotion", label: null, reliable: false, reason: "speech_gate_failed", speech: {
      available: true, label: "happy", confidence: 0.37, margin: 0.10, normalized_entropy: 0.91, calibrated: false,
      probabilities: {angry: .08, disgust: .06, fear: .07, happy: .37, neutral: .27, sad: .08, surprised: .07}}},
    emotion_gate: {confidence_threshold: .55, margin_threshold: .12, max_normalized_entropy: .85},
    policy: {strategy: "neutral_fallback"}, usage: {input_audio_seconds: .89675},
    timings_ms: {asr: 1381.9, speech: 82.1, llm: 2043.8, tts: 45546.5, total: 49086.7}};
}

test("uncertain speech preserves the raw happy candidate, scores and failed thresholds", () => {
  const p = page(); p.context.result = result(); p.run("renderEmotion(result)");
  const text = p.ids.get("emotion-summary").textContent;
  assert.match(text, /语音情感专家/); assert.match(text, /开心（happy）/);
  assert.match(text, /37\.0%/); assert.match(text, /≥ 55\.0%/);
  assert.match(text, /语音合成 45\.55 秒/); assert.match(text, /总计 49\.09 秒/);
  assert.match(text, /不足 3 秒/); assert.match(text, /不等于预测正确率/);
  const data = JSON.parse(p.ids.get("metrics").textContent);
  assert.equal(data.emotion.label, null); assert.equal(data.emotion.speech.label, "happy");
  assert.equal(data.gate_checks.filter(item => !item.passed).length, 3);
  assert.equal(p.ids.get("emotion-summary").querySelectorAll("progress").length, 7);
});

test("text input states that no speech or separate text classifier ran", () => {
  const p = page(), data = result();
  data.emotion = {status: "unknown_emotion", label: null, reason: "text_input", speech: {available: false, probabilities: {}}};
  data.usage = {}; p.context.result = data; p.run("renderEmotion(result)");
  assert.match(p.ids.get("emotion-summary").textContent, /未配置独立的文字情感分类器/);
  assert.equal(p.ids.get("emotion-summary").querySelectorAll("progress").length, 0);
});

test("D0 audio explains that ASR ran while speech emotion was skipped", () => {
  const p = page(), data = result(); data.mode = "text_only";
  data.emotion = {status: "unknown_emotion", label: null, reason: "text_input", speech: {available: false, probabilities: {}}};
  data.policy.strategy = "text_only"; p.context.result = data; p.run("renderEmotion(result)");
  assert.match(p.ids.get("emotion-summary").textContent, /D0 仅文本模式：音频只做语音识别/);
  assert.doesNotMatch(p.ids.get("emotion-summary").textContent, /本轮为文字输入/);
});

test("D1 explicitly identifies its use of the raw candidate despite a failed gate", () => {
  const p = page(), data = result(); data.mode = "speech_label"; data.policy.strategy = "speech_label";
  p.context.result = data; p.run("renderEmotion(result)");
  assert.match(p.ids.get("emotion-summary").textContent, /开心（happy）（D1 直接参考候选）/);
});

test("input audio is playable before the request completes and remains after a failure", async () => {
  const p = page(); let rejectRequest;
  p.context.fetch = () => new Promise((resolve, reject) => { rejectRequest = reject; });
  const pending = p.run("upload(new Blob(['recording'], {type: 'audio/wav'}))");
  const conversation = p.ids.get("conversation"), player = conversation.querySelector("audio");
  assert.ok(player.controls); assert.match(player.src, /^blob:/);
  assert.match(conversation.textContent, /输入录音/); assert.ok(p.ids.get("send").disabled);
  rejectRequest(new Error("模拟断网")); await pending;
  assert.match(conversation.textContent, /输入录音仍可回放/); assert.equal(p.ids.get("send").disabled, false);
});

test("successful audio turn puts the transcript beside the original recording", async () => {
  const p = page(); p.context.fetch = async () => ({ok: true, json: async () => result()});
  await p.run("upload(new Blob(['recording'], {type: 'audio/wav'}))");
  assert.equal(p.ids.get("conversation").children.length, 2);
  const user = p.ids.get("conversation").children[0];
  assert.equal(user.querySelector(".message-text").textContent, "我今天很开心");
  assert.ok(user.querySelector("audio"));
});

test("unsupported input playback falls back to the authenticated normalized WAV", async () => {
  const p = page(), data = result(); data.input_audio_url = "/api/input-audio/" + "a".repeat(32);
  p.ids.get("token").value = "secret"; p.context.result = data;
  let options;
  p.context.fetch = async (url, request) => { assert.equal(url, data.input_audio_url); options = request; return {ok: true, blob: async () => new Blob(["wav"])}; };
  p.run("var user = message('user', '处理中'); var input = localAudio(user, new Blob(['flac']), '输入录音'); input.player.error = {code: 4}");
  await p.run("display(result, user, input)");
  assert.equal(options.headers.Authorization, "Bearer secret");
  assert.match(p.ids.get("conversation").children[0].textContent, /分析用 WAV/);
});

test("microphone discovery requests permission only on click and stops the probe stream", async () => {
  const p = page(); assert.equal(p.requests.length, 0);
  await p.ids.get("refresh-microphones").fire("click");
  assert.equal(p.requests.length, 1); assert.equal(p.requests[0].audio, true); assert.equal(p.stopped, 1);
  assert.equal(p.ids.get("microphone").children.length, 2);
  assert.match(p.ids.get("microphone").textContent, /USB 麦克风/);
});

test("recording uses the selected device and an ideal mono preference", async () => {
  const p = page(); p.ids.get("microphone").value = "usb";
  await p.ids.get("record").fire("click");
  assert.equal(p.requests[0].audio.deviceId.exact, "usb");
  assert.equal(p.requests[0].audio.channelCount.ideal, 1);
  assert.equal(p.ids.get("record").disabled, false);
  assert.equal(p.ids.get("record").textContent, "停止并发送");
});

test("missing recording device gives actionable RDP guidance and restores controls", async () => {
  const p = page(); p.context.navigator.mediaDevices.getUserMedia = async () => { throw {name: "NotFoundError", message: "Requested device not found"}; };
  await p.ids.get("record").fire("click");
  assert.match(p.ids.get("status").textContent, /未找到可用麦克风/);
  assert.match(p.ids.get("status").textContent, /麦克风重定向/);
  assert.equal(p.ids.get("send").disabled, false);
});

test("insecure HTTP refuses microphone capture before accessing a device", async () => {
  const p = page(); p.context.window.isSecureContext = false;
  await p.ids.get("record").fire("click");
  assert.equal(p.requests.length, 0); assert.match(p.ids.get("status").textContent, /HTTPS 或 localhost/);
});

test("new conversation pauses audio and releases original recording URLs", async () => {
  const p = page(); p.context.fetch = async () => ({ok: true});
  p.run("localAudio(message('user', '录音'), new Blob(['wav']), '输入录音')");
  const player = p.ids.get("conversation").querySelector("audio"); await player.play();
  await p.ids.get("reset").fire("click");
  assert.equal(player.paused, true); assert.equal(p.revoked.length, 1);
  assert.equal(p.ids.get("conversation").children.length, 0);
});

test("NDJSON reader handles split UTF-8 text and closes the reader on failure", async () => {
  const p = page(), encoder = new TextEncoder(), bytes = encoder.encode('{"type":"note","text":"中文"}\n{"type":"done"}\n');
  const stream = new ReadableStream({start(controller) {
    for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
    controller.close();
  }});
  const events = []; p.context.response = new Response(stream); p.context.callback = event => events.push(event);
  await p.run("readNDJSON(response, callback)");
  assert.equal(events[0].text, "中文"); assert.equal(events[1].type, "done");
});

test("text displays and first PCM plays while the rest of generation is still pending", async () => {
  const p = page(); p.ids.get("stream-tts").checked = true; p.ids.get("text").value = "你好";
  let controller, textPayload;
  const start = {type: "start", sample_rate: 24000, channels: 1, encoding: "pcm_s16le"};
  const audio = {type: "audio", sequence: 0, pcm: Buffer.from([0, 64, 0, 192]).toString("base64")};
  const encoder = new TextEncoder();
  p.context.fetch = async (url, options) => {
    if (url === "/api/text") { textPayload = JSON.parse(options.body); return {ok: true, json: async () => result()}; }
    assert.match(url, /\/api\/tts\/a+\/stream/);
    return new Response(new ReadableStream({start(stream) {
      controller = stream;
      stream.enqueue(encoder.encode(JSON.stringify(start) + "\n" + JSON.stringify(audio) + "\n"));
    }}));
  };
  const pending = p.ids.get("text-form").fire("submit");
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(textPayload.synthesize, false);
  assert.match(p.ids.get("conversation").textContent, /想聊聊发生了什么/);
  assert.equal(p.audioContexts[0].started.length, 1);
  assert.equal(p.audioContexts[0].buffers[0].getChannelData(0)[0], .5);
  assert.equal(p.audioContexts[0].buffers[0].getChannelData(0)[1], -.5);
  assert.equal(p.ids.get("send").disabled, true);
  controller.enqueue(encoder.encode(JSON.stringify({...audio, sequence: 1}) + "\n" + JSON.stringify({type: "done", result: result()}) + "\n"));
  controller.close(); await pending;
  assert.equal(p.audioContexts[0].started.length, 2);
  assert.ok(Math.abs(p.audioContexts[0].started[1] - p.audioContexts[0].started[0] - 2 / 24000) < 1e-8);
  assert.equal(p.ids.get("send").disabled, false);
});

test("malformed PCM sequence preserves text and does not retry a whole synthesis", async () => {
  const p = page(); p.ids.get("stream-tts").checked = true; p.ids.get("text").value = "你好";
  let calls = 0;
  p.context.fetch = async url => {
    calls++;
    if (url === "/api/text") return {ok: true, json: async () => result()};
    return new Response('{"type":"audio","sequence":1,"pcm":"AAAAAA=="}\n');
  };
  await p.ids.get("text-form").fire("submit");
  assert.equal(calls, 2); assert.match(p.ids.get("status").textContent, /语音块顺序不正确/);
  assert.match(p.ids.get("conversation").textContent, /想聊聊发生了什么/);
  assert.equal(p.audioContexts[0].state, "closed");
  assert.equal(p.ids.get("send").disabled, false);
});

test("stop button aborts the active stream and closes the playback context", async () => {
  const p = page(); p.ids.get("stream-tts").checked = true; p.ids.get("text").value = "你好";
  let aborted = false;
  p.context.fetch = async (url, options) => {
    if (url === "/api/text") return {ok: true, json: async () => result()};
    return new Response(new ReadableStream({start(controller) {
      options.signal.addEventListener("abort", () => { aborted = true; controller.error(new DOMException("cancelled", "AbortError")); });
    }}));
  };
  const pending = p.ids.get("text-form").fire("submit");
  await new Promise(resolve => setImmediate(resolve));
  const stop = p.ids.get("conversation").querySelectorAll("button").find(button => button.textContent === "停止语音");
  await stop.fire("click"); await pending;
  assert.equal(aborted, true); assert.equal(p.audioContexts[0].state, "closed");
  assert.match(p.ids.get("status").textContent, /已停止语音/);
  assert.equal(p.ids.get("send").disabled, false);
});
