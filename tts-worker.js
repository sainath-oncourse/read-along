// Free, on-device neural voices. Runs off the main thread so highlighting stays smooth.
// Kokoro (best quality, needs WebGPU) and Piper (lighter, works on any device).
const KOKORO = 'https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm';
const PIPER = 'https://cdn.jsdelivr.net/npm/@mintplex-labs/piper-tts-web@1.0.4/+esm';

let kokoroP = null, piperMod = null;
const piperReady = {};
let chain = Promise.resolve(); // one job at a time

function progress(engine, voice) {
  const files = {};
  return (p) => {
    if (p && p.total) files[p.file || p.url || 'f'] = [p.loaded || 0, p.total];
    let l = 0, t = 0;
    for (const [a, b] of Object.values(files)) { l += a; t += b; }
    if (t) postMessage({ type: 'progress', engine, voice, pct: Math.min(99, Math.round(l / t * 100)), mb: Math.round(t / 1e6) });
  };
}

async function kokoro(voice) {
  if (!kokoroP) {
    kokoroP = (async () => {
      if (!navigator.gpu || !(await navigator.gpu.requestAdapter())) throw new Error('nogpu');
      const { KokoroTTS } = await import(KOKORO);
      return KokoroTTS.from_pretrained('onnx-community/Kokoro-82M-v1.0-ONNX', {
        dtype: 'fp32', device: 'webgpu', progress_callback: progress('k', voice),
      });
    })();
    kokoroP.catch(() => { kokoroP = null; });
  }
  return kokoroP;
}

async function piper(voice) {
  if (!piperMod) piperMod = await import(PIPER);
  if (!piperReady[voice]) {
    piperReady[voice] = (async () => {
      const have = await piperMod.stored().catch(() => []);
      if (!have.includes(voice)) await piperMod.download(voice, progress('p', voice));
    })();
    piperReady[voice].catch(() => { delete piperReady[voice]; });
  }
  await piperReady[voice];
  return piperMod;
}

async function run({ id, type, engine, voice, text }) {
  try {
    if (type === 'warm') {
      if (engine === 'k') await kokoro(voice); else await piper(voice);
      postMessage({ id, ok: true });
      return;
    }
    let blob;
    if (engine === 'k') {
      const tts = await kokoro(voice);
      const audio = await tts.generate(text, { voice });
      blob = audio.toBlob();
    } else {
      const p = await piper(voice);
      blob = await p.predict({ text, voiceId: voice });
    }
    postMessage({ id, blob });
  } catch (e) {
    postMessage({ id, err: String((e && e.message) || e) });
  }
}

let epoch = 0; // the page bumps this when the listener skips, so stale jobs are dropped
const KMODEL = 'https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx';
async function status(id) {
  let k = false, p = [];
  try { k = !!(await (await caches.open('transformers-cache')).match(KMODEL)); } catch {}
  try { if (!piperMod) piperMod = await import(PIPER); p = await piperMod.stored(); } catch {}
  postMessage({ id, k, p });
}
onmessage = ({ data }) => {
  if (data.type === 'epoch') { epoch = data.n; return; }
  if (data.type === 'status') { status(data.id); return; }
  chain = chain.then(() => {
    if (data.type === 'gen' && data.epoch !== undefined && data.epoch < epoch) return postMessage({ id: data.id, err: 'stale' });
    return run(data);
  });
};
