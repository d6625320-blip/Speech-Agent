"use strict";

// Keep a single Web Audio timeline. A new <audio> per chunk would leave gaps.
class PCMStreamPlayer {
  constructor() {
    this.context = new AudioContext({latencyHint: "interactive"});
    this.sources = new Set(); this.nextTime = 0; this.bytes = 0; this.finished = false; this.closed = false;
  }
  resume() { return this.closed ? Promise.resolve() : this.context.resume(); }
  pause() { return this.closed ? Promise.resolve() : this.context.suspend(); }
  enqueue(encoded, sampleRate) {
    if (this.closed) return;
    const binary = atob(encoded);
    if (!binary.length || binary.length % 2 || binary.length > 32768 || sampleRate !== 24000) throw new Error("流式音频块格式错误");
    this.bytes += binary.length;
    if (this.bytes > 20 * 1024 * 1024) throw new Error("流式音频超过大小限制");
    const bytes = Uint8Array.from(binary, character => character.charCodeAt(0)), view = new DataView(bytes.buffer);
    const buffer = this.context.createBuffer(1, bytes.length / 2, sampleRate), channel = buffer.getChannelData(0);
    for (let i = 0; i < channel.length; i++) channel[i] = view.getInt16(i * 2, true) / 32768;
    const source = this.context.createBufferSource(); source.buffer = buffer; source.connect(this.context.destination);
    this.nextTime = Math.max(this.nextTime, this.context.currentTime + 0.08);
    this.sources.add(source);
    source.onended = () => { this.sources.delete(source); source.disconnect(); if (this.finished && !this.sources.size) this.stop(); };
    source.start(this.nextTime); this.nextTime += buffer.duration;
  }
  finish() { this.finished = true; if (!this.sources.size) this.stop(); }
  stop() {
    if (this.closed) return; this.closed = true;
    for (const source of this.sources) { source.onended = null; try { source.stop(); } catch {} source.disconnect(); }
    this.sources.clear(); this.context.close().catch(() => {});
  }
}

async function readNDJSON(response, onEvent) {
  if (!response.body?.getReader) throw new Error("浏览器不支持读取音频流，请取消流式播放或更新浏览器");
  const reader = response.body.getReader(), decoder = new TextDecoder("utf-8", {fatal: true});
  let pending = "";
  const consume = async final => {
    let end;
    while ((end = pending.indexOf("\n")) !== -1) {
      const line = pending.slice(0, end); pending = pending.slice(end + 1);
      if (line.length > 131072) throw new Error("流事件过大");
      if (line.trim()) await onEvent(JSON.parse(line));
    }
    if (pending.length > 131072) throw new Error("流事件过大");
    if (final && pending.trim()) { await onEvent(JSON.parse(pending)); pending = ""; }
  };
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) { pending += decoder.decode(); await consume(true); break; }
      pending += decoder.decode(value, {stream: true}); await consume(false);
    }
  } finally {
    await reader.cancel().catch(() => {}); reader.releaseLock();
  }
}
