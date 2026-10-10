// Video editor for the SRT → Speech project (runs as the js_on_load of a gr.HTML in app.py).
// Available here: element, trigger, props, server (the editor_* functions in app.py), upload, watch.
//
// Clips are keyed by id: "12" is line #12; cutting it adds pieces "12~1", "12~2", … (clip.line says which line).
// Each piece plays the part [trim_in, duration - trim_out] of its line's audio (seconds of that audio) at `start`,
// `speed` times as fast (pitch kept), on row `lane` of its speaker's track.
const root = element.querySelector('.vxe');
const q = (role) => root.querySelector('[data-role="' + role + '"]');
const video = q('video'), music = q('music'), scroll = q('scroll'), content = q('content'), insp = q('inspector');
const HEAD = 150;
const ROW = 46; // height of one row of a speaker track
const DUCK_DB = -15;
const MIN_PIECE = 0.05;
const SPEED_MIN = 0.5, SPEED_MAX = 2.0;
const PALETTE = ['#4f8cff', '#e8833a', '#3fb68b', '#c25bd6', '#e05d6f', '#d6b23a', '#3ab7d6', '#8a7cf0', '#7fb13f', '#d67a3a'];

const S = {
  data: null, lines: [], lineMap: {}, clips: {}, tracks: {}, edits: {},
  mix: { original: 'replace', original_gain_db: 0, vocals_gain_db: 0 },
  sel: new Set(), panel: null, pps: 80, dur: 10,
  buffers: {}, loading: {}, sources: [], playing: false, clockT: 0, clockCtx: 0, schedT: 0, schedCtx: 0,
  undo: [], redo: [], pendingUndo: null, poll: null, saveTimer: null, working: new Set(), busy: false,
  stretched: {}, // "line|speed" -> url of the line's audio at that speed (pitch kept), made on the server
};
let ctx = null;
// New subtitles: { cues: [{ id, start, end, text }], burn: on/off, size: text height as a fraction of the picture }.
S.subs = null;
S.selCue = null;
const SUB_DEFAULTS = { burn: true, size: 0.034 };

// ---------- small helpers ----------
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const r3 = (v) => Math.round(v * 1000) / 1000;
const dbg = (db) => Math.pow(10, (db || 0) / 20);
function fmt(t, short) {
  t = Math.max(0, t || 0);
  const m = Math.floor(t / 60), s = t - m * 60;
  const ss = short ? String(Math.floor(s)).padStart(2, '0') : s.toFixed(2).padStart(5, '0');
  return String(m).padStart(2, '0') + ':' + ss;
}
function ensureCtx() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  return ctx;
}
async function call(name, arg) {
  const res = await server[name](arg || {});
  if (res && res.error) throw new Error(res.error);
  return res;
}
function speakers() {
  const names = (S.data && S.data.speakers) ? S.data.speakers.slice() : [];
  S.lines.forEach((l) => { if (!names.includes(l.speaker)) names.push(l.speaker); });
  return names;
}
const colorOf = (name) => PALETTE[Math.max(speakers().indexOf(name), 0) % PALETTE.length];

// ---------- clips and pieces ----------
const lineIdOf = (key) => { const c = S.clips[key]; return c && c.line != null ? +c.line : parseInt(key, 10); };
const lineOf = (key) => S.lineMap[lineIdOf(key)];
const keys = () => Object.keys(S.clips).filter((k) => lineOf(k));
const keysOfLine = (idx) => keys().filter((k) => lineIdOf(k) === idx).sort((a, b) => (S.clips[a].trim_in || 0) - (S.clips[b].trim_in || 0));
const keysOfSpeaker = (name) => keys().filter((k) => lineOf(k).speaker === name);
const byStart = (list) => list.slice().sort((a, b) => S.clips[a].start - S.clips[b].start);
const speedOf = (key) => clamp(+(S.clips[key] && S.clips[key].speed) || 1, SPEED_MIN, SPEED_MAX);
function sourceLen(key) {
  // Seconds of the line's own audio the clip plays.
  const l = lineOf(key), c = S.clips[key];
  return Math.max((l.duration || 0) - (c.trim_in || 0) - (c.trim_out || 0), MIN_PIECE);
}
function clipLen(key) {
  const l = lineOf(key), c = S.clips[key];
  if (!l || !c) return 0;
  if (!l.url) return Math.max(l.end - l.start, 0.3);
  return Math.max(sourceLen(key) / speedOf(key), MIN_PIECE);
}
const speedLabel = (v) => (Math.round(v * 100) / 100) + '×';
const clipEnd = (key) => S.clips[key].start + clipLen(key);
const regions = () => (S.data && S.data.video && S.data.video.regions) || [];
function audible(key) {
  const l = lineOf(key), c = S.clips[key];
  if (!l || !c || !l.url || c.muted) return false;
  const tr = S.tracks[l.speaker] || {};
  if (tr.muted) return false;
  const solo = Object.values(S.tracks).some((t) => t.solo);
  return !solo || !!tr.solo;
}
function newKey(idx) {
  let n = 1;
  while (S.clips[idx + '~' + n]) n++;
  return idx + '~' + n;
}
const selKeys = () => [...S.sel].filter((k) => S.clips[k] && lineOf(k));
const selLines = () => [...new Set(selKeys().map(lineIdOf))];
const generated = (list) => list.filter((k) => lineOf(k).url);

function setStatus(html) { q('status').innerHTML = html || ''; }
function setBanner(html) { q('banner').innerHTML = html || ''; }
function busy(text) {
  S.busy = !!text;
  q('busy').classList.toggle('vxe-on', !!text);
  q('busy-text').textContent = text || '';
}
function showError(e) {
  console.error(e);
  setStatus('<span style="color:var(--vxe-bad)">⚠️ ' + esc(e && e.message ? e.message : e) + '</span>');
}

// ---------- loading the project ----------
function computeDur() {
  let d = 5;
  if (S.data && S.data.video) d = Math.max(d, S.data.video.duration || 0);
  if (video.duration && isFinite(video.duration)) d = Math.max(d, video.duration);
  S.lines.forEach((l) => { d = Math.max(d, l.end); });
  keys().forEach((k) => { d = Math.max(d, clipEnd(k)); });
  return d + 2;
}

function applyPayload(p) {
  S.data = p;
  S.lines = (p.lines || []).slice().sort((a, b) => a.start - b.start);
  S.lineMap = {};
  S.lines.forEach((l) => { S.lineMap[l.index] = l; });
  S.clips = p.clips || {};
  S.tracks = p.tracks || {};
  Object.assign(S.mix, p.mix || {});
  S.sel = new Set(selKeys());
  S.edits = {};
  S.subs = p.subtitles && Array.isArray(p.subtitles.cues) ? Object.assign({}, SUB_DEFAULTS, p.subtitles) : null;
  if (!S.subs && S.lines.some((l) => l.url) && p.clips) {
    // First time with generated lines: one subtitle per line, on its voice's time.
    S.subs = Object.assign({}, SUB_DEFAULTS, { cues: subsFromVoices() });
    clearTimeout(S.saveTimer); S.saveTimer = setTimeout(save, 600);
  }
  if (S.selCue && !cueById(S.selCue)) S.selCue = null;
  const v = p.video;
  if (v && v.url) {
    if (video.dataset.src !== v.url) {
      video.dataset.src = v.url;
      video.src = v.url;
    }
    if ((music.dataset.src || '') !== (v.music_url || '')) {
      music.dataset.src = v.music_url || '';
      if (v.music_url) music.src = v.music_url; else music.removeAttribute('src');
    }
    root.classList.add('vxe-has-video');
  } else {
    root.classList.remove('vxe-has-video');
    if (video.dataset.src) { video.removeAttribute('src'); video.dataset.src = ''; video.load(); }
  }
  const audioOnly = !!(v && v.has_video === false);
  root.classList.toggle('vxe-audio-only', audioOnly);
  q('audio-name').textContent = audioOnly ? '🎵 ' + v.name + ' (audio only — the export is the soundtrack)' : '';
  root.querySelector('[data-act="export"]').textContent = audioOnly ? '⬇ Export audio' : '⬇ Export video';
  q('mode').value = S.mix.original || 'replace';
  const subs = S.mix.subs || {};
  q('subs-mode').value = subs.mode || 'off';
  q('subs-group').style.display = v && v.has_video !== false ? '' : 'none';
  q('orig-gain').value = S.mix.original_gain_db || 0;
  showOriginalToggle();
  showCaptionsButton();
  showLevelButton();
  q('voc-gain').value = S.mix.vocals_gain_db || 0;
  if (!S.lines.length) {
    setBanner('No lines yet — upload an SRT on the <b>📝 SRT → Speech</b> tab, click <b>Analyze</b> and <b>Generate</b>; then come back here.');
  } else if (!S.lines.some((l) => l.url)) {
    setBanner('None of the lines has been generated yet — click <b>Generate all lines</b> on the <b>📝 SRT → Speech</b> tab first.');
  } else {
    setBanner('');
  }
  S.dur = computeDur();
  render();
  renderInspector();
  S.stretched = {};
  S.lines.forEach((l) => loadBuffer(l));
  keys().forEach(loadStretched);
  const job = p.job;
  if (job && job.status === 'running') {
    markWorking(job);
    startPolling();
  } else {
    S.working.clear();
  }
}

async function reload() {
  try {
    applyPayload(await call('editor_load'));
  } catch (e) {
    S.data = null; S.lines = []; S.lineMap = {}; S.clips = {};
    setBanner(esc(e.message || e));
    render(); renderInspector();
  }
}

function stretchKey(key) { return lineIdOf(key) + '|' + speedOf(key).toFixed(2); }
async function loadStretched(key) {
  // The line's audio at the clip's speed with its pitch kept; until it arrives the preview changes the pitch.
  const sk = stretchKey(key), l = lineOf(key);
  if (!l || !l.url || Math.abs(speedOf(key) - 1) < 1e-3 || S.stretched[sk] || S.loading[sk]) return;
  S.loading[sk] = true;
  try {
    const r = await call('editor_speed', { line: l.index, speed: speedOf(key) });
    const res = await fetch(r.url);
    S.buffers[r.url] = await ensureCtx().decodeAudioData(await res.arrayBuffer());
    S.stretched[sk] = r.url;
    if (S.playing) schedule(now());
  } catch (e) {
    console.warn('Could not change the speed of line', l.index, e);
  } finally {
    delete S.loading[sk];
  }
}

async function loadBuffer(line) {
  if (!line.url || S.buffers[line.url] || S.loading[line.url]) return;
  S.loading[line.url] = true;
  try {
    const res = await fetch(line.url);
    const buf = await ensureCtx().decodeAudioData(await res.arrayBuffer());
    S.buffers[line.url] = buf;
    keysOfLine(line.index).forEach(drawClipWave);
    if (S.playing) schedule(now());
    if (S.panel === 'levels') { clearTimeout(S.lvTimer); S.lvTimer = setTimeout(() => S.panel === 'levels' && renderLevelsPanel(), 300); }
  } catch (e) {
    console.warn('Could not load line', line.index, e);
  } finally {
    delete S.loading[line.url];
  }
}

// ---------- timeline ----------
const tToX = (t) => t * S.pps;
function timeAtClientX(x) {
  const rect = content.getBoundingClientRect();
  return Math.max(0, (x - rect.left - HEAD) / S.pps);
}
const rowsOf = (name) => Math.max(1, ...keysOfSpeaker(name).map((k) => (S.clips[k].lane || 0) + 1));

function overlaps() {
  // Pieces of the same speaker that sound at the same time (on any row).
  const bad = new Set();
  speakers().forEach((name) => {
    const list = byStart(keysOfSpeaker(name).filter(audible));
    let lastEnd = -1, lastKey = null;
    list.forEach((k) => {
      if (lastKey && S.clips[k].start < lastEnd - 0.02) { bad.add(k); bad.add(lastKey); }
      if (clipEnd(k) > lastEnd) { lastEnd = clipEnd(k); lastKey = k; }
    });
  });
  return bad;
}

function replacedAt(t) {
  // "Muted where the new voices replace it" (as video_editor.replaced_spans): a new voice is playing, or t is in
  // a stretch of original speech that a new voice overlaps.
  const playing = keys().filter(audible);
  if (playing.some((k) => S.clips[k].start <= t && clipEnd(k) > t)) return true;
  return regions().some((r) => r[0] <= t && r[1] > t && playing.some((k) => S.clips[k].start < r[1] && clipEnd(k) > r[0]));
}

function uncovered(region) {
  return !keys().some((k) => audible(k) && S.clips[k].start < region[1] && clipEnd(k) > region[0]);
}

function clipHtml(key, bad) {
  const l = lineOf(key), c = S.clips[key];
  const len = clipLen(key);
  const pieces = keysOfLine(l.index);
  const cls = ['vxe-clip'];
  if (S.sel.has(key)) cls.push('vxe-selected');
  if (!l.url) cls.push('vxe-missing');
  else {
    if (c.muted || (S.tracks[l.speaker] || {}).muted) cls.push('vxe-muted');
    if (!l.ok) cls.push('vxe-bad');
    if (bad.has(key)) cls.push('vxe-overlap');
  }
  if (pieces.length > 1) cls.push('vxe-piece');
  if (S.working.has(l.index)) cls.push('vxe-working');
  const part = (pieces.length > 1 ? ' ✂' + (pieces.indexOf(key) + 1) + '/' + pieces.length : '') +
    (Math.abs(speedOf(key) - 1) > 1e-3 ? ' ⏩' + speedLabel(speedOf(key)) : '');
  const title = '#' + l.index + part + ' ' + l.speaker + (l.emotion ? ' (' + l.emotion + ')' : '') + ' · ' + fmt(c.start) + ' → ' +
    fmt(c.start + len) + '\n' + l.text + '\n' + (l.status || '');
  return '<div class="' + cls.join(' ') + '" data-clip="' + esc(key) + '" title="' + esc(title) + '" style="left:' +
    tToX(c.start) + 'px;width:' + Math.max(tToX(len), 3) + 'px;top:' + (4 + (c.lane || 0) * ROW) + 'px;height:' + (ROW - 8) +
    'px;--c:' + colorOf(l.speaker) + '">' +
    (l.url ? '<canvas></canvas><div class="vxe-handle vxe-l" data-h="l"></div><div class="vxe-handle vxe-r" data-h="r"></div>' : '') +
    '<div class="vxe-clip-label">#' + l.index + part + ' ' + esc(l.text) + '</div></div>';
}

function render() {
  const W = Math.ceil(tToX(S.dur)) + 40;
  content.style.width = (HEAD + W) + 'px';
  const bad = overlaps();
  let html = '<div class="vxe-trackrow vxe-ruler"><div class="vxe-head">' + esc(S.data ? S.data.project : '') +
    '</div><div class="vxe-lane" data-lane="ruler" style="width:' + W + 'px"></div></div>';
  const v = S.data && S.data.video;
  if (v && v.has_video !== false) {
    const cues = S.subs ? S.subs.cues : [];
    const subRows = cueRows();
    let subGuides = '';
    for (let r = 1; r < subRows; r++) subGuides += '<div class="vxe-rowline" style="top:' + (r * CUE_ROW + 2) + 'px"></div>';
    html += '<div class="vxe-trackrow vxe-subs" style="height:' + (subRows * CUE_ROW + 6) + 'px"><div class="vxe-head"><div class="vxe-head-name">💬 Subtitles' +
      (subRows > 1 ? ' <span style="color:var(--vxe-dim);font-weight:400;font-size:11px">' + subRows + ' rows</span>' : '') + '</div>' +
      '<div class="vxe-head-tools"><button data-act="subs-burn" class="' + (subsOn() ? '' : 'vxe-on-m') +
      '" title="Turn the new subtitles off / on (preview and export)">M</button><button data-act="cue-add" title="Add a subtitle at the playhead (N)">+</button>' +
      '<button data-act="subs-panel" title="Subtitle settings">⚙ ' + cues.length + '</button></div></div>' +
      '<div class="vxe-lane" data-lane="subs" style="width:' + W + 'px">' + subGuides + cues.map(cueHtml).join('') + '</div></div>';
  }
  if (v) {
    const regs = regions().map((r) => '<div class="vxe-region' + (uncovered(r) ? ' vxe-uncovered' : '') + '" style="left:' +
      tToX(r[0]) + 'px;width:' + Math.max(tToX(r[1] - r[0]), 2) + 'px" title="Speech in the video ' + fmt(r[0]) + ' – ' + fmt(r[1]) + '"></div>').join('');
    const secs = sections().map((sec, i) => {
      const cls = ['vxe-osec'];
      if (S.selSec === i) cls.push('vxe-selected');
      if (sec.sound === 'mute') cls.push('vxe-muted');
      const label = sec.sound === 'mute' ? '🔇 Silent' : sec.sound === 'full' ? '🗣 Full original' : sec.sound === 'music' ? '🎵 Voices removed' : '';
      const gain = sec.gain_db ? (sec.gain_db > 0 ? '+' : '') + sec.gain_db + ' dB' : '';
      return '<div class="' + cls.join(' ') + '" data-osec="' + i + '" style="left:' + tToX(sec.start) + 'px;width:' +
        Math.max(tToX(sec.end - sec.start), 2) + 'px" title="Original sound, part ' + (i + 1) + ': ' + fmt(sec.start) + ' – ' + fmt(sec.end) +
        '"><span class="vxe-osec-label">' + esc([label, gain].filter(Boolean).join(' · ')) + '</span></div>';
    }).join('');
    html += '<div class="vxe-trackrow vxe-speech"><div class="vxe-head"><div class="vxe-head-name">🎞 Original sound</div>' +
      '<div class="vxe-head-tools"><button data-act="orig-mute" class="' + (originalOn() ? '' : 'vxe-on-m') +
      '" title="Turn the original video\'s sound off / on (O)">M</button><button data-act="speech-panel" title="Speech found in the video: detection settings">⚙ ' +
      regions().length + ' speech</button></div></div>' +
      '<div class="vxe-lane" data-lane="speech" style="width:' + W + 'px"><canvas class="vxe-wave" data-role="speech-wave"></canvas>' + regs + secs + '</div></div>';
  }
  speakers().forEach((name) => {
    const own = S.lines.filter((l) => l.speaker === name);
    if (!own.length) return;
    const tr = S.tracks[name] || {};
    const rows = rowsOf(name);
    const slots = own.map((l) => '<div class="vxe-slot' + (selLines().includes(l.index) ? ' vxe-sel' : '') + '" data-slot="' + l.index +
      '" style="left:' + tToX(l.start) + 'px;width:' + Math.max(tToX(l.end - l.start), 2) + 'px"></div>').join('');
    let guides = '';
    for (let r = 1; r < rows; r++) guides += '<div class="vxe-rowline" style="top:' + (r * ROW) + 'px"></div>';
    html += '<div class="vxe-trackrow" data-track="' + esc(name) + '" style="height:' + (rows * ROW + 2) + 'px"><div class="vxe-head"><div class="vxe-head-name">' +
      '<span class="vxe-dot" style="background:' + colorOf(name) + ';width:9px;height:9px;border-radius:50%"></span>' + esc(name) +
      '</div><div class="vxe-head-tools"><button data-act="track-mute" data-name="' + esc(name) + '" class="' + (tr.muted ? 'vxe-on-m' : '') +
      '" title="Mute this speaker (also in the export)">M</button><button data-act="track-solo" data-name="' + esc(name) + '" class="' +
      (tr.solo ? 'vxe-on-s' : '') + '" title="Hear only this speaker (preview only)">S</button><span style="color:var(--vxe-dim);font-size:11px">' +
      own.length + ' lines' + (rows > 1 ? ' · ' + rows + ' rows' : '') + '</span></div></div><div class="vxe-lane" data-lane="clips" data-name="' +
      esc(name) + '" style="width:' + W + 'px">' + guides + slots + keysOfSpeaker(name).map((k) => clipHtml(k, bad)).join('') + '</div></div>';
  });
  if (!S.lines.length) html += '<div class="vxe-empty-lines">No lines in this project yet.</div>';
  html += '<div class="vxe-playhead" data-role="playhead"></div>';
  content.innerHTML = html;
  keys().forEach(drawClipWave);
  renderVisible();
  updatePlayhead(now());
}

function renderVisible() {
  // Ruler ticks and the video waveform are drawn only for the part of the timeline in view.
  const lane = content.querySelector('[data-lane="ruler"]');
  if (!lane) return;
  const x0 = Math.max(scroll.scrollLeft - 200, 0), x1 = scroll.scrollLeft + scroll.clientWidth + 200;
  const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200];
  const step = steps.find((s) => s * S.pps >= 70) || 1200;
  const minor = step / 5;
  let html = '';
  for (let t = Math.floor(x0 / S.pps / step) * step; t * S.pps <= x1 && t <= S.dur; t += step) {
    html += '<div class="vxe-tick" style="left:' + tToX(t) + 'px">' + fmt(t, step >= 1) + '</div>';
    if (minor * S.pps >= 10) {
      for (let k = 1; k < 5; k++) html += '<div class="vxe-tick vxe-minor" style="left:' + tToX(t + k * minor) + 'px"></div>';
    }
  }
  lane.innerHTML = html;
  drawSpeechWave(x0, x1);
}

function drawSpeechWave(x0, x1) {
  const canvas = content.querySelector('[data-role="speech-wave"]');
  const v = S.data && S.data.video;
  if (!canvas || !v || !v.peaks || !v.peaks.length) return;
  const lane = canvas.parentElement;
  const width = Math.min(Math.max(x1 - x0, 1), 8000), height = lane.clientHeight || 48;
  const dpr = window.devicePixelRatio || 1;
  canvas.style.left = x0 + 'px';
  canvas.style.width = width + 'px';
  canvas.style.right = 'auto';
  canvas.width = Math.ceil(width * dpr);
  canvas.height = Math.ceil(height * dpr);
  const g = canvas.getContext('2d');
  g.scale(dpr, dpr);
  g.fillStyle = 'rgba(150,160,180,0.55)';
  const perSec = 40, mid = height / 2;
  for (let x = 0; x < width; x++) {
    const a = Math.floor(((x0 + x) / S.pps) * perSec), b = Math.max(Math.floor(((x0 + x + 1) / S.pps) * perSec), a + 1);
    let peak = 0;
    for (let i = a; i < b && i < v.peaks.length; i++) peak = Math.max(peak, v.peaks[i]);
    const h = peak * (height / 2 - 3);
    g.fillRect(x, mid - h, 1, h * 2 || 1);
  }
}

function clipEl(key) { return content.querySelector('[data-clip="' + CSS.escape(key) + '"]'); }

function drawClipWave(key) {
  const l = lineOf(key), c = S.clips[key];
  const buf = l && S.buffers[l.url];
  const el = clipEl(key);
  const canvas = el && el.querySelector('canvas');
  if (!buf || !canvas) return;
  const width = Math.min(Math.max(el.clientWidth, 1), 3000), height = Math.max(el.clientHeight, 10);
  canvas.width = width; canvas.height = height;
  const g = canvas.getContext('2d');
  const data = buf.getChannelData(0);
  const a0 = Math.floor((c.trim_in || 0) * buf.sampleRate);
  const n = Math.max(Math.floor(sourceLen(key) * buf.sampleRate), 1);
  const per = Math.max(Math.floor(n / width), 1);
  g.fillStyle = 'rgba(0,0,0,0.75)';
  for (let x = 0; x < width; x++) {
    let peak = 0;
    const s = a0 + Math.floor((x / width) * n);
    for (let i = s; i < s + per && i < data.length; i += 4) peak = Math.max(peak, Math.abs(data[i]));
    const h = Math.min(peak * 1.6, 1) * (height / 2 - 2);
    g.fillRect(x, height / 2 - h, 1, h * 2 || 1);
  }
}

function updateClipEl(key) {
  const el = clipEl(key);
  if (!el) return;
  el.style.left = tToX(S.clips[key].start) + 'px';
  el.style.width = Math.max(tToX(clipLen(key)), 3) + 'px';
  el.style.top = (4 + (S.clips[key].lane || 0) * ROW) + 'px';
}

function renderSelection() {
  if (S.sel.size) { S.selCue = null; S.selSec = null; }
  if (S.selCue) S.selSec = null;
  content.querySelectorAll('[data-osec]').forEach((el) => el.classList.toggle('vxe-selected', +el.dataset.osec === S.selSec));
  content.querySelectorAll('[data-cue]').forEach((el) => el.classList.toggle('vxe-selected', el.dataset.cue === S.selCue));
  content.querySelectorAll('[data-clip]').forEach((el) => el.classList.toggle('vxe-selected', S.sel.has(el.dataset.clip)));
  const lines = selLines();
  content.querySelectorAll('[data-slot]').forEach((el) => el.classList.toggle('vxe-sel', lines.includes(+el.dataset.slot)));
  S.panel = null;
  renderInspector();
}

function updatePlayhead(t) {
  const ph = content.querySelector('[data-role="playhead"]');
  if (ph) ph.style.left = (HEAD + tToX(t)) + 'px';
  q('time').textContent = fmt(t);
  drawPreviewSubs(t);
}

function setZoom(pps, anchorT, clientX) {
  const rect = scroll.getBoundingClientRect();
  if (anchorT == null) { anchorT = now(); clientX = rect.left + HEAD + (scroll.clientWidth - HEAD) / 2; }
  S.pps = clamp(pps, 4, 400);
  q('zoom').value = S.pps;
  render();
  scroll.scrollLeft = Math.max(0, anchorT * S.pps + HEAD - (clientX - rect.left));
  renderVisible();
}

// ---------- edits, undo, save ----------
const snapshot = () => JSON.stringify({ clips: S.clips, tracks: S.tracks, subs: S.subs, secs: S.mix.orig_sections || null });
function pushUndo(before) {
  S.undo.push(before || snapshot());
  if (S.undo.length > 200) S.undo.shift();
  S.redo = [];
}
function restore(snap) {
  const s = JSON.parse(snap);
  S.clips = s.clips; S.tracks = s.tracks; S.subs = s.subs;
  if (s.secs) S.mix.orig_sections = s.secs; else delete S.mix.orig_sections;
  if (S.selSec != null && S.selSec >= sections().length) S.selSec = null;
  S.sel = new Set(selKeys());
  if (S.selCue && !cueById(S.selCue)) S.selCue = null;
  changed();
}
function undo() { if (S.undo.length) { S.redo.push(snapshot()); restore(S.undo.pop()); } }
function redo() { if (S.redo.length) { S.undo.push(snapshot()); restore(S.redo.pop()); } }

function changed(keepInspector) {
  S.dur = computeDur();
  const left = scroll.scrollLeft, top = scroll.scrollTop;
  render();
  scroll.scrollLeft = left; scroll.scrollTop = top;
  if (!keepInspector) renderInspector();
  if (S.playing) schedule(now());
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(save, 600);
}
async function save() {
  clearTimeout(S.saveTimer);
  try { await call('editor_save', { clips: S.clips, mix: S.mix, tracks: S.tracks, subtitles: S.subs || undefined }); } catch (e) { showError(e); }
}

function bestRegion(l, taken) {
  let best = null, score = 0;
  regions().forEach((r, k) => {
    if (r[1] < l.start - 0.8 || r[0] > l.end + 0.8) return;
    const overlap = Math.max(0, Math.min(r[1], l.end) - Math.max(r[0], l.start));
    if (overlap <= 0 && taken && k in taken) return;
    const s = overlap > 0 ? overlap : 1e-3 / (1 + Math.abs(r[0] - l.start));
    if (s > score) { best = k; score = s; }
  });
  return best;
}
function moveLine(idx, start) {
  // Moves every piece of a line together, keeping their spacing.
  const list = keysOfLine(idx);
  if (!list.length) return;
  const d = start - Math.min(...list.map((k) => S.clips[k].start));
  list.forEach((k) => { S.clips[k].start = r3(Math.max(0, S.clips[k].start + d)); });
}
function stackOverlaps(names) {
  // Puts pieces that sound at the same time on separate rows, so each one can be seen and moved.
  let rows = 0;
  (names || speakers()).forEach((name) => {
    const ends = [];
    byStart(keysOfSpeaker(name)).forEach((k) => {
      let lane = ends.findIndex((e) => e <= S.clips[k].start + 0.02);
      if (lane < 0) { lane = ends.length; ends.push(0); }
      ends[lane] = clipEnd(k);
      S.clips[k].lane = lane;
    });
    rows = Math.max(rows, ends.length);
  });
  return rows;
}
function alignClips(lineIds, quiet) {
  // Same rule as video_editor.align_to_speech: a line starts where its speech starts; lines spoken without a
  // pause share one stretch, and the later ones move by the same amount as the first.
  if (!regions().length) { setStatus('No speech was found in the video — open ⚙ on the <b>Video speech</b> track to detect it again.'); return; }
  pushUndo();
  const shift = {};
  let moved = 0, missed = 0;
  S.lines.forEach((l) => {
    if (!l.url) return;
    const k = bestRegion(l, shift);
    if (k == null) { if (lineIds.includes(l.index)) missed++; return; }
    const r = regions()[k];
    const start = k in shift ? Math.max(l.start + shift[k], r[0]) : r[0];
    if (!(k in shift)) shift[k] = r[0] - l.start;
    if (!lineIds.includes(l.index)) return;
    moveLine(l.index, start);
    moved++;
  });
  const touched = [...new Set(lineIds.map((i) => S.lineMap[i] && S.lineMap[i].speaker).filter(Boolean))];
  const before = overlaps().size;
  stackOverlaps(touched);
  changed();
  if (!quiet) {
    setStatus('🎯 Lined up ' + moved + ' clip(s) with the speech in the video' + (missed ? '; ' + missed + ' had no speech near their subtitle time' : '') + '.' +
      (before ? ' Clips that now overlap were put on separate rows — drag them up/down, or use <b>⇥ Remove overlaps</b>.' : ''));
  }
}
function toSrt(lineIds) {
  pushUndo();
  lineIds.forEach((i) => { if (S.lineMap[i]) moveLine(i, S.lineMap[i].start); });
  stackOverlaps();
  changed();
}
function toggleMute(list) {
  if (!list.length) return;
  pushUndo();
  const mute = list.some((k) => !S.clips[k].muted);
  list.forEach((k) => { S.clips[k].muted = mute; });
  changed();
}
function nudge(list, dt) {
  pushUndo();
  list.forEach((k) => { S.clips[k].start = r3(Math.max(0, S.clips[k].start + dt)); });
  changed();
}
function setSpeed(list, speed) {
  list = generated(list);
  if (!list.length) return;
  pushUndo();
  list.forEach((k) => { S.clips[k].speed = Math.round(clamp(speed, SPEED_MIN, SPEED_MAX) * 100) / 100; });
  changed();
  list.forEach(loadStretched);
}
function stepSpeed(list, d) {
  list = generated(list);
  if (list.length) setSpeed(list, speedOf(list[0]) + d);
}
function fitSpeed(list) {
  // Speed each clip so it fills its line's subtitle time exactly.
  list = generated(list);
  if (!list.length) return;
  pushUndo();
  const out = [];
  list.forEach((k) => {
    const l = lineOf(k), slot = l.end - l.start;
    const pieces = keysOfLine(l.index);
    const share = pieces.length > 1 ? sourceLen(k) / pieces.reduce((a, p) => a + sourceLen(p), 0) : 1;
    const want = clamp(sourceLen(k) / Math.max(slot * share, 0.1), SPEED_MIN, SPEED_MAX);
    S.clips[k].speed = Math.round(want * 100) / 100;
    out.push('#' + l.index + ' ' + speedLabel(S.clips[k].speed));
  });
  changed();
  list.forEach(loadStretched);
  setStatus('⏱ Fitted to the subtitle time: ' + out.join(', ') + (out.length > 12 ? '…' : '') + ' (between 0.5× and 2×).');
}
function moveRows(list, d) {
  if (!list.length) return;
  pushUndo();
  list.forEach((k) => { S.clips[k].lane = clamp((S.clips[k].lane || 0) + d, 0, 7); });
  compactRows();
  changed();
}
function compactRows() {
  // Drops empty rows (a speaker's rows are always 0, 1, 2, … with something on each).
  speakers().forEach((name) => {
    const used = [...new Set(keysOfSpeaker(name).map((k) => S.clips[k].lane || 0))].sort((a, b) => a - b);
    keysOfSpeaker(name).forEach((k) => { S.clips[k].lane = used.indexOf(S.clips[k].lane || 0); });
  });
}
function removeOverlaps() {
  // Moves clips later, where needed, so no two new voices play at the same time; all go back to the first row.
  pushUndo();
  let moved = 0, end = -1;
  byStart(keys().filter(audible)).forEach((k) => {
    const c = S.clips[k];
    if (c.start < end + 0.05 && end >= 0) { c.start = r3(end + 0.05); moved++; }
    end = Math.max(end, clipEnd(k));
  });
  keys().forEach((k) => { S.clips[k].lane = 0; });
  changed();
  setStatus(moved ? '⇥ Moved ' + moved + ' clip(s) later so no two voices overlap.' : 'No clips overlap.');
}

// ---------- cut and join ----------
function cutAt(t, list) {
  // Splits each piece under the playhead in two at time t.
  list = generated(list || keys().filter(audible)).filter((k) => S.clips[k].start + MIN_PIECE < t && clipEnd(k) - MIN_PIECE > t);
  if (!list.length) { setStatus('✂ Put the playhead over a clip (and select it) to cut it there.'); return; }
  pushUndo();
  const made = [];
  list.forEach((k) => {
    const c = S.clips[k], l = lineOf(k);
    const offset = (t - c.start) * speedOf(k); // seconds of the line's audio before the cut
    const b = newKey(l.index);
    S.clips[b] = Object.assign({}, c, { line: l.index, start: r3(t), trim_in: r3((c.trim_in || 0) + offset) });
    c.line = l.index;
    c.trim_out = r3((l.duration || 0) - (c.trim_in || 0) - offset);
    made.push(b);
  });
  S.sel = new Set(made);
  changed();
  setStatus('✂ Cut ' + list.length + ' clip(s) at ' + fmt(t) + '. Move or delete a piece; <b>🔗 Join</b> connects the pieces of a line back together.');
}
function joinPieces(list) {
  // Connects the selected pieces of each line back into one clip (also restores a deleted part in between).
  const lines = [...new Set(list.map(lineIdOf))].filter((i) => keysOfLine(i).length > 1);
  if (!lines.length) { setStatus('🔗 Select a piece of a cut line (✂ in its name) to join it back.'); return; }
  pushUndo();
  const kept = [];
  lines.forEach((idx) => {
    const all = keysOfLine(idx);
    const chosen = all.filter((k) => list.includes(k));
    const join = chosen.length > 1 ? chosen : all;
    const first = join[0], last = join[join.length - 1];
    const c = S.clips[first];
    c.trim_out = S.clips[last].trim_out || 0;
    join.slice(1).forEach((k) => { delete S.clips[k]; });
    kept.push(first);
  });
  S.sel = new Set(kept);
  compactRows();
  changed();
  setStatus('🔗 Joined the pieces of ' + lines.map((i) => '#' + i).join(', ') + '.');
}
function deletePieces(list) {
  // Removes pieces of cut lines; a line's last piece is muted instead (the line stays on the timeline).
  if (!list.length) return;
  pushUndo();
  let removed = 0, muted = 0;
  list.forEach((k) => {
    if (keysOfLine(lineIdOf(k)).length > 1) { delete S.clips[k]; removed++; S.sel.delete(k); }
    else { S.clips[k].muted = !S.clips[k].muted; muted++; }
  });
  compactRows();
  changed();
  setStatus((removed ? '🗑 Deleted ' + removed + ' piece(s). ' : '') + (muted ? '🔇 Muted / unmuted ' + muted + ' clip(s) (a whole line is muted, not deleted).' : ''));
}
function collapseLines(lineIds) {
  // Before a line is regenerated its audio changes, so its pieces become one clip again (where the first one was).
  lineIds.forEach((idx) => {
    const all = byStart(keysOfLine(idx));
    if (!all.length) return;
    const keep = all[0];
    all.slice(1).forEach((k) => { delete S.clips[k]; S.sel.delete(k); });
    Object.assign(S.clips[keep], { trim_in: 0, trim_out: 0 });
  });
}

// ---------- dragging ----------
function snapStart(key, start, len, moving) {
  const tol = 8 / S.pps;
  const points = [now()];
  regions().forEach((r) => points.push(r[0], r[1]));
  const l = lineOf(key);
  if (l) points.push(l.start);
  keys().forEach((o) => { if (!moving.has(o) && lineOf(o).url) points.push(S.clips[o].start, clipEnd(o)); });
  let best = start, dist = tol;
  points.forEach((p) => {
    if (Math.abs(p - start) < dist) { dist = Math.abs(p - start); best = p; }
    if (Math.abs(p - (start + len)) < dist) { dist = Math.abs(p - (start + len)); best = p - len; }
  });
  return best;
}

function startDrag(e, el) {
  const key = el.dataset.clip;
  const handle = e.target.dataset.h;
  if (e.shiftKey && !handle) {
    if (S.sel.has(key)) S.sel.delete(key); else S.sel.add(key);
    renderSelection();
    return;
  }
  if (!S.sel.has(key)) { S.sel = new Set([key]); renderSelection(); }
  if (el.classList.contains('vxe-missing')) return;
  const before = snapshot();
  const moving = new Set(handle ? [key] : generated(selKeys()));
  const init = {};
  moving.forEach((k) => { init[k] = Object.assign({}, S.clips[k]); });
  const x0 = e.clientX, y0 = e.clientY, line = lineOf(key);
  let moved = false;
  el.setPointerCapture(e.pointerId);
  const onMove = (ev) => {
    if (!moved && Math.abs(ev.clientX - x0) < 3 && Math.abs(ev.clientY - y0) < 6) return;
    moved = true;
    const dt = (ev.clientX - x0) / S.pps;
    const snapOn = q('snap').checked && !ev.altKey;
    const c = S.clips[key], c0 = init[key];
    const full = line.duration || 0;
    if (!handle) {
      let start = c0.start + dt;
      if (snapOn) start = snapStart(key, start, clipLen(key), moving);
      let d = start - c0.start;
      const first = Math.min(...Object.values(init).map((v) => v.start));
      if (first + d < 0) d = -first;
      // Up / down: to another row of the same speaker's track.
      const dl = Math.round((ev.clientY - y0) / ROW);
      const low = Math.min(...Object.values(init).map((v) => v.lane || 0));
      const dLane = Math.max(dl, -low);
      moving.forEach((k) => {
        S.clips[k].start = r3(init[k].start + d);
        S.clips[k].lane = clamp((init[k].lane || 0) + dLane, 0, 7);
        updateClipEl(k);
      });
    } else if (handle === 'l') {
      let start = c0.start + dt;
      if (snapOn) start = snapStart(key, start, 0, moving);
      const sp = speedOf(key);
      const ti = clamp((c0.trim_in || 0) + (start - c0.start) * sp, 0, full - (c0.trim_out || 0) - 0.1);
      c.trim_in = r3(ti);
      c.start = r3(Math.max(0, c0.start + (ti - (c0.trim_in || 0)) / sp));
      updateClipEl(key);
    } else {
      const sp = speedOf(key);
      const end0 = c0.start + (full - (c0.trim_in || 0) - (c0.trim_out || 0)) / sp;
      let end = end0 + dt;
      if (snapOn) end = snapStart(key, end, 0, moving);
      c.trim_out = r3(clamp((c0.trim_out || 0) - (end - end0) * sp, 0, full - (c0.trim_in || 0) - 0.1));
      updateClipEl(key);
    }
    q('time').textContent = fmt(S.clips[key].start);
  };
  const onUp = () => {
    el.removeEventListener('pointermove', onMove);
    el.removeEventListener('pointerup', onUp);
    el.removeEventListener('pointercancel', onUp);
    if (moved) { pushUndo(before); compactRows(); changed(); }
  };
  el.addEventListener('pointermove', onMove);
  el.addEventListener('pointerup', onUp);
  el.addEventListener('pointercancel', onUp);
}

content.addEventListener('pointerdown', (e) => {
  if (e.button !== 0 || e.target.closest('.vxe-head')) return;
  root.focus({ preventScroll: true });
  const secEl = e.target.closest('[data-osec]');
  if (secEl) {
    // A part of the original sound: select it (it does not move: it has to stay in time with the picture).
    seek(timeAtClientX(e.clientX));
    S.selSec = +secEl.dataset.osec; S.sel.clear(); S.selCue = null; S.panel = null;
    renderSelection();
    return;
  }
  const cueEl = e.target.closest('[data-cue]');
  if (cueEl) { startCueDrag(e, cueEl); return; }
  const el = e.target.closest('[data-clip]');
  if (el) {
    if (e.altKey && !el.classList.contains('vxe-missing') && !e.target.dataset.h) {
      // Alt+click: cut this clip where you click.
      cutAt(timeAtClientX(e.clientX), [el.dataset.clip]);
      return;
    }
    startDrag(e, el);
    return;
  }
  const lane = e.target.closest('.vxe-lane');
  if (!lane) return;
  seek(timeAtClientX(e.clientX));
  if (lane.dataset.lane === 'ruler') {
    lane.setPointerCapture(e.pointerId);
    const onMove = (ev) => seek(timeAtClientX(ev.clientX));
    const onUp = () => { lane.removeEventListener('pointermove', onMove); lane.removeEventListener('pointerup', onUp); };
    lane.addEventListener('pointermove', onMove);
    lane.addEventListener('pointerup', onUp);
  } else if (!e.shiftKey && (S.sel.size || S.selCue || S.selSec != null)) {
    S.sel.clear();
    S.selCue = null;
    S.selSec = null;
    renderSelection();
  }
});
content.addEventListener('dblclick', (e) => {
  const cueEl = e.target.closest('[data-cue]');
  if (cueEl) {
    seek(cueById(cueEl.dataset.cue).start);
    const box = insp.querySelector('[data-cf="text"]');
    if (box) { box.focus(); box.select(); }
    return;
  }
  const lane = e.target.closest('[data-lane="subs"]');
  if (lane) { addCue(timeAtClientX(e.clientX)); return; }
  const el = e.target.closest('[data-clip]');
  if (!el || e.altKey) return;
  seek(S.clips[el.dataset.clip].start);
  play();
});
content.addEventListener('click', (e) => {
  const b = e.target.closest('[data-act]');
  if (!b) return;
  const name = b.dataset.name;
  if (b.dataset.act === 'track-mute' || b.dataset.act === 'track-solo') {
    pushUndo();
    const tr = S.tracks[name] = S.tracks[name] || {};
    if (b.dataset.act === 'track-mute') tr.muted = !tr.muted; else tr.solo = !tr.solo;
    changed();
  } else if (b.dataset.act === 'speech-panel') {
    S.panel = 'speech';
    renderInspector();
  } else if (b.dataset.act === 'orig-mute') {
    toggleOriginal();
  } else if (b.dataset.act === 'subs-burn') {
    toggleSubs();
  } else if (b.dataset.act === 'cue-add') {
    addCue(now());
  } else if (b.dataset.act === 'subs-panel') {
    S.panel = 'subs';
    renderInspector();
  }
});

let scrollRaf = 0;
scroll.addEventListener('scroll', () => {
  if (scrollRaf) return;
  scrollRaf = requestAnimationFrame(() => { scrollRaf = 0; renderVisible(); });
});
scroll.addEventListener('wheel', (e) => {
  if (!(e.ctrlKey || e.metaKey)) return;
  e.preventDefault();
  setZoom(S.pps * Math.exp(-e.deltaY * 0.002), timeAtClientX(e.clientX), e.clientX);
}, { passive: false });

// ---------- playback ----------
const hasVideo = () => !!(S.data && S.data.video && video.getAttribute('src'));
function now() {
  if (hasVideo()) return video.currentTime || 0;
  return S.playing && ctx ? S.clockT + (ctx.currentTime - S.clockCtx) : S.clockT;
}
function stopSources() {
  S.sources.forEach((s) => { try { s.stop(); } catch (e) { /* already stopped */ } });
  S.sources = [];
}
const originalOn = () => S.mix.original_on !== false;
function showOriginalToggle() {
  const on = originalOn(), b = q('orig-toggle');
  b.textContent = on ? '🔊 Original sound: on' : '🔇 Original sound: off';
  b.classList.toggle('vxe-off', !on);
  q('mode').disabled = !on;
  const m = content.querySelector('[data-act="orig-mute"]');
  if (m) m.classList.toggle('vxe-on-m', !on);
}
function showCaptionsButton() {
  const b = q('captions');
  b.classList.toggle('vxe-primary', subsOn());
  b.textContent = subsOn() ? '💬 Subtitles: on' : '💬 Subtitles: off';
}
function toggleOriginal() {
  S.mix.original_on = !originalOn();
  showOriginalToggle();
  if (originalOn()) syncMusic(now(), S.playing);
  changed(true);
  setStatus(originalOn() ? '🔊 The original video sound is on (' + q('mode').selectedOptions[0].text.toLowerCase() + ').'
    : '🔇 The original video sound is off — only the new voices play, in the preview and the export.');
}
function useMusic() {
  return !!music.getAttribute('src') && (S.mix.original === 'music' || sections().some((x) => x.sound === 'music'));
}
function syncMusic(t, playing) {
  if (!useMusic()) { music.pause(); return; }
  if (Math.abs((music.currentTime || 0) - t) > 0.05) music.currentTime = t;
  if (playing) music.play().catch(() => {}); else music.pause();
}
function schedule(t) {
  stopSources();
  const c = ensureCtx();
  S.schedT = t; S.schedCtx = c.currentTime;
  keys().forEach((k) => {
    if (!audible(k)) return;
    const sp = speedOf(k);
    const stretched = Math.abs(sp - 1) > 1e-3 ? S.buffers[S.stretched[stretchKey(k)]] : null;
    const buf = stretched || S.buffers[lineOf(k).url];
    const clip = S.clips[k], len = clipLen(k);
    if (!buf || clip.start + len <= t) return;
    const into = Math.max(0, t - clip.start);
    const src = c.createBufferSource();
    src.buffer = buf;
    const g = c.createGain();
    const flat = (clip.gain_db || 0) + (S.mix.vocals_gain_db || 0) + autoGainDb(k);
    g.gain.value = dbg(flat);
    src.connect(g); g.connect(voiceBus(c));
    const when = c.currentTime + Math.max(0, clip.start - t);
    if (levelOn() && len - into > 0.02) {
      // The evening-out curve of the line, over the part that plays (in the line's own seconds).
      const curve = eveningCurve(S.buffers[lineOf(k).url]), from = (clip.trim_in || 0) + into * sp;
      const n = Math.max(2, Math.ceil((len - into) / 0.01)), values = new Float32Array(n);
      for (let i = 0; i < n; i++) values[i] = dbg(flat + curveDbAt(curve, from + (i / (n - 1)) * (len - into) * sp));
      try { g.gain.setValueCurveAtTime(values, when, len - into); } catch (e) { /* keeps the flat gain */ }
    }
    if (stretched || Math.abs(sp - 1) < 1e-3) {
      // The stretched copy runs on the timeline's clock: the line's second s is at s / speed in it.
      const scale = stretched ? sp : 1;
      src.start(when, (clip.trim_in || 0) / scale + into, len - into);
    } else {
      src.playbackRate.value = sp; // not ready yet: faster/slower with a pitch change, for now
      src.start(when, (clip.trim_in || 0) + into * sp, (len - into) * sp);
    }
    S.sources.push(src);
  });
  syncMusic(t, true);
}
// "Level voices": every clip brought to the same loudness, as heard (as video_editor.speech_level / level_gain_db).
const LEVEL_TARGET_DB = -20, LEVEL_RANGE_DB = 24, LEVEL_TARGET_MIN = -30, LEVEL_TARGET_MAX = -10;
// How strongly a line is evened out from the inside (as video_editor.LEVEL_STRENGTHS): [ratio, most dB].
const LEVEL_STRENGTHS = { off: [1, 0], light: [2, 6], normal: [3, 9], strong: [6, 12] };
const levelOn = () => S.mix.level !== false;
const levelTarget = () => clamp(S.mix.level_target != null ? +S.mix.level_target : LEVEL_TARGET_DB, LEVEL_TARGET_MIN, LEVEL_TARGET_MAX);
const levelStrength = () => (S.mix.level_strength in LEVEL_STRENGTHS ? S.mix.level_strength : 'normal');
const levelCache = {};
function biquad(x, b, a) {
  const y = new Float64Array(x.length), b0 = b[0] / a[0], b1 = b[1] / a[0], b2 = b[2] / a[0], a1 = a[1] / a[0], a2 = a[2] / a[0];
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = x[i], o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = v; y2 = y1; y1 = o; y[i] = o;
  }
  return y;
}
function kWeighted(data, sr) {
  // The ITU-R BS.1770 "K" filter (as video_editor.k_weighted), so power follows how loud a voice sounds.
  const A = Math.pow(10, 3.99984385397 / 40);
  let w0 = 2 * Math.PI * 1681.9744509555319 / sr, cos = Math.cos(w0), alpha = Math.sin(w0) / (2 * 0.7071752369554193);
  const root = 2 * Math.sqrt(A) * alpha;
  const y = biquad(data,
    [A * ((A + 1) + (A - 1) * cos + root), -2 * A * ((A - 1) + (A + 1) * cos), A * ((A + 1) + (A - 1) * cos - root)],
    [(A + 1) - (A - 1) * cos + root, 2 * ((A - 1) - (A + 1) * cos), (A + 1) - (A - 1) * cos - root]);
  w0 = 2 * Math.PI * 38.13547087613982 / sr; cos = Math.cos(w0); alpha = Math.sin(w0) / (2 * 0.5003270373253953);
  return biquad(y, [(1 + cos) / 2, -(1 + cos), (1 + cos) / 2], [1 + alpha, -2 * cos, 1 - alpha]);
}
function speechLevel(data, sr, a, b) {
  // data is already K-weighted.
  const n = Math.max(Math.round(sr * 0.05), 1), powers = [];
  for (let s = a; s + n <= b; s += n) {
    let sum = 0;
    for (let i = s; i < s + n; i++) sum += data[i] * data[i];
    powers.push(sum / n);
  }
  if (!powers.length) return null;
  const dbs = powers.map((p) => 10 * Math.log10(p + 1e-12)), gate = Math.max(Math.max(...dbs) - 30, -60);
  const voiced = powers.filter((p, i) => dbs[i] > gate);
  return voiced.length ? 10 * Math.log10(voiced.reduce((x, y) => x + y, 0) / voiced.length) : null;
}
// Evening out a line from the inside (as video_editor.evening_curve): a gain in dB every EVEN_FRAME seconds.
const EVEN_FRAME = 0.05, EVEN_WINDOW = 0.3;
const curveCache = {};
function eveningCurve(buf) {
  buf.__id = buf.__id || ('b' + Math.random());
  const strength = levelStrength(), id = buf.__id + '|' + strength;
  if (curveCache[id]) return curveCache[id];
  const [ratio, most] = LEVEL_STRENGTHS[strength];
  const sr = buf.sampleRate, data = kWeighted(buf.getChannelData(0), sr);
  const hop = Math.max(Math.round(sr * EVEN_FRAME), 1), half = Math.max(Math.round(sr * EVEN_WINDOW / 2), 1);
  const count = Math.ceil(data.length / hop), line = speechLevel(data, sr, 0, data.length);
  const gains = new Float64Array(Math.max(count, 1));
  if (count && line != null && most > 0) {
    const squares = new Float64Array(data.length + 1);
    for (let i = 0; i < data.length; i++) squares[i + 1] = squares[i] + data[i] * data[i];
    for (let i = 0; i < count; i++) {
      const c = Math.floor((i + 0.5) * hop), a = Math.max(c - half, 0), b = Math.min(c + half, data.length);
      const db = 10 * Math.log10((squares[b] - squares[a]) / Math.max(b - a, 1) + 1e-12);
      if (db > Math.max(line - 20, -60)) gains[i] = clamp((line - db) * (1 - 1 / ratio), -most, most);
    }
  }
  const smooth = new Float64Array(gains.length);  // over 0.25 s, so the volume glides
  for (let i = 0; i < gains.length; i++) {
    let sum = 0;
    for (let d = -2; d <= 2; d++) sum += gains[clamp(i + d, 0, gains.length - 1)];
    smooth[i] = sum / 5;
  }
  curveCache[id] = smooth;
  return smooth;
}
function curveDbAt(curve, t) {
  // Linear between points (as numpy.interp), held flat before the first and after the last.
  const x = t / EVEN_FRAME - 0.5;
  if (x <= 0) return curve[0];
  if (x >= curve.length - 1) return curve[curve.length - 1];
  const i = Math.floor(x);
  return curve[i] + (curve[i + 1] - curve[i]) * (x - i);
}
function rawLevel(key) {
  // How loud the clip's part sounds after evening out, before any gain (null: silent or not loaded).
  const l = lineOf(key), c = S.clips[key], buf = l && S.buffers[l.url];
  if (!buf) return undefined;
  buf.__id = buf.__id || ('b' + Math.random());
  const id = buf.__id + '|' + (c.trim_in || 0) + '|' + (c.trim_out || 0) + '|' + levelStrength();
  if (!(id in levelCache)) {
    // Measured on the evened-out part, as the export does.
    const sr = buf.sampleRate, data = buf.getChannelData(0), curve = eveningCurve(buf);
    const a = Math.floor((c.trim_in || 0) * sr), b = Math.max(a, data.length - Math.floor((c.trim_out || 0) * sr));
    const part = new Float32Array(b - a);
    for (let i = a; i < b; i++) part[i - a] = data[i] * Math.pow(10, curveDbAt(curve, i / sr) / 20);
    levelCache[id] = speechLevel(kWeighted(part, sr), sr, 0, part.length);
  }
  return levelCache[id];
}
function autoGainDb(key) {
  if (!levelOn()) return 0;
  const level = rawLevel(key);
  return level == null ? 0 : clamp(levelTarget() - level, -LEVEL_RANGE_DB, LEVEL_RANGE_DB);
}
function voiceBus(c) {
  // All voices go through a limiter, so levelled-up clips never clip (like the export's soft limiter).
  if (!S.bus || S.bus.context !== c) {
    S.bus = c.createDynamicsCompressor();
    S.bus.threshold.value = -2; S.bus.knee.value = 4; S.bus.ratio.value = 20;
    S.bus.attack.value = 0.003; S.bus.release.value = 0.1;
    S.bus.connect(c.destination);
  }
  return S.bus;
}
function showLevelButton() {
  const b = q('level');
  b.textContent = levelOn() ? '🎚 Level voices: on' : '🎚 Level voices: off';
  b.classList.toggle('vxe-primary', levelOn());
}
function updatePlayBtn() { root.querySelector('[data-act="play"]').textContent = S.playing ? '⏸' : '▶'; }
async function play() {
  if (!S.lines.length) return;
  const c = ensureCtx();
  try { await c.resume(); } catch (e) { /* ignore */ }
  if (hasVideo()) {
    if (video.ended || video.currentTime >= (video.duration || 0) - 0.05) video.currentTime = 0;
    try { await video.play(); } catch (e) { showError(e); }
  } else {
    S.clockT = now() >= S.dur - 0.05 ? 0 : now();
    S.clockCtx = c.currentTime;
    S.playing = true;
    schedule(S.clockT);
    updatePlayBtn();
  }
}
function pause() {
  if (hasVideo()) { video.pause(); return; }
  S.clockT = now();
  S.playing = false;
  stopSources();
  music.pause();
  updatePlayBtn();
}
const toggle = () => (S.playing ? pause() : play());
function seek(t) {
  t = clamp(t, 0, S.dur);
  if (hasVideo()) video.currentTime = t;
  else { S.clockT = t; if (ctx) S.clockCtx = ctx.currentTime; if (S.playing) schedule(t); }
  if (!S.playing) syncMusic(t, false);
  updatePlayhead(t);
}
video.addEventListener('playing', () => { S.playing = true; schedule(video.currentTime); updatePlayBtn(); });
video.addEventListener('pause', () => { S.playing = false; stopSources(); music.pause(); updatePlayBtn(); });
video.addEventListener('waiting', () => { stopSources(); music.pause(); });
video.addEventListener('seeked', () => { if (S.playing && !video.paused) schedule(video.currentTime); updatePlayhead(video.currentTime); });
video.addEventListener('loadedmetadata', () => { S.dur = computeDur(); render(); });
video.addEventListener('error', () => {
  if (video.getAttribute('src')) setStatus('⚠️ This browser cannot play the video. Try Change video with an .mp4 (H.264) file.');
});
video.addEventListener('click', toggle);

function frame() {
  const t = now();
  updatePlayhead(t);
  if (S.playing) {
    if (hasVideo() && !video.paused && ctx) {
      const expected = S.schedT + (ctx.currentTime - S.schedCtx);
      if (Math.abs(expected - t) > 0.15) schedule(t);
    }
    if (useMusic() && Math.abs(music.currentTime - t) > 0.25) music.currentTime = t;
    if (!hasVideo() && t >= S.dur) { pause(); seek(S.dur); }
    const x = HEAD + tToX(t);
    if (x < scroll.scrollLeft + HEAD || x > scroll.scrollLeft + scroll.clientWidth - 30) {
      scroll.scrollLeft = Math.max(0, x - HEAD - (scroll.clientWidth - HEAD) * 0.2);
    }
  }
  const vocalsOn = keys().some((k) => audible(k) && S.clips[k].start <= t && clipEnd(k) > t);
  // The original sound: the part of it at t has its own volume and sound (as set / full original / voices removed / off).
  const sec = sectionAt(t), mode = S.mix.original;
  const base = clamp(dbg((S.mix.original_gain_db || 0) + (sec.gain_db || 0)), 0, 1);
  const sound = !originalOn() ? 'mute' : sec.sound || 'auto';
  let fromVideo = 0, fromMusic = 0;
  if (sound === 'full') fromVideo = base;
  else if (sound === 'music') fromMusic = base;
  else if (sound === 'auto' && mode === 'music') fromMusic = base;
  else if (sound === 'auto' && mode !== 'mute') {
    fromVideo = base;
    if (mode === 'duck' && vocalsOn) fromVideo *= dbg(DUCK_DB);
    if (mode === 'replace' && replacedAt(t)) fromVideo = 0;
  }
  if (!originalOn()) music.pause();
  video.muted = fromVideo <= 0;
  video.volume = fromVideo;
  music.volume = useMusic() ? fromMusic : 0;
  placeSubBox();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---------- inspector ----------
function field(label, html) { return '<div class="vxe-row"><label>' + label + '</label>' + html + '</div>'; }
const speedButtons =
  '<button class="vxe-btn" data-act="slower" title="Slower ([)">🐢 −</button>' +
  '<button class="vxe-btn" data-act="faster" title="Faster (])">🐇 +</button>' +
  '<button class="vxe-btn" data-act="speed-1" title="Normal speed">1×</button>' +
  '<button class="vxe-btn" data-act="speed-fit" title="Speed it so it fills its subtitle time exactly">Fit to subtitle</button>';
const editTools =
  '<button class="vxe-btn" data-act="cut-sel" title="Cut the selected clip(s) at the playhead (C)">✂ Cut at playhead</button>' +
  '<button class="vxe-btn" data-act="join-sel" title="Connect the pieces of a cut line back together (J)">🔗 Join pieces</button>' +
  '<button class="vxe-btn" data-act="up-sel" title="Move to the row above (Alt+↑)">⬆ Row up</button>' +
  '<button class="vxe-btn" data-act="down-sel" title="Move to the row below (Alt+↓)">⬇ Row down</button>';

function renderInspector() {
  if (!S.data) { insp.innerHTML = '<div class="vxe-muted-note">Nothing loaded.</div>'; return; }
  if (S.panel === 'speech') return renderSpeechPanel();
  if (S.panel === 'subs') return renderSubsPanel();
  if (S.panel === 'levels') return renderLevelsPanel();
  if (S.selSec != null && sections()[S.selSec]) return renderSectionPanel(S.selSec);
  if (S.selCue && cueById(S.selCue)) return renderCuePanel(cueById(S.selCue));
  const sel = selKeys();
  if (sel.length === 1) return renderClipPanel(sel[0]);
  if (sel.length > 1) {
    insp.innerHTML = '<h3>' + sel.length + ' clips selected</h3>' +
      '<div class="vxe-muted-note">Drag any of them to move them together (up/down for another row), or:</div><div class="vxe-actions">' +
      '<button class="vxe-btn" data-act="align-sel">🎯 Align to video speech</button>' +
      '<button class="vxe-btn" data-act="srt-sel">⏱ To SRT time</button>' +
      '<button class="vxe-btn" data-act="mute-sel">🔇 Mute / unmute</button>' + editTools + speedButtons +
      '<button class="vxe-btn" data-act="delete-sel" title="Delete (pieces) / mute (whole lines)">🗑 Delete pieces</button>' +
      '<button class="vxe-btn vxe-primary" data-act="regen-sel">🔁 Regenerate these lines</button></div>';
    return;
  }
  const done = S.lines.filter((l) => l.url).length;
  const warn = S.lines.filter((l) => l.url && !l.ok).length;
  const v = S.data.video;
  const unc = regions().filter(uncovered).length;
  const ov = overlaps().size;
  insp.innerHTML = '<h3>📂 ' + esc(S.data.project) + '</h3>' +
    field('Lines', done + ' of ' + S.lines.length + ' generated' + (warn ? ' · <span style="color:var(--vxe-warn)">⚠ ' + warn + ' to check</span>' : '')) +
    field('Speakers', speakers().map((n) => '<span class="vxe-badge" style="color:' + colorOf(n) + '">' + esc(n) + '</span>').join(' ')) +
    (v ? field('Video', esc(v.name) + ' · ' + fmt(v.duration)) +
      field('Speech', regions().length + ' stretch(es) found' + (unc ? ' · <span style="color:var(--vxe-bad)">' + unc + ' without a new voice</span>' : '') +
        ' <button class="vxe-btn" data-act="speech-panel">⚙ Detection</button>')
      : field('Video', '<span class="vxe-muted-note">none yet — drop one on the left</span>')) +
    field('Voice levels', (levelOn() ? '🎚 same loudness (' + levelTarget() + ' dB)' : '<span class="vxe-muted-note">off — each clip at its own loudness</span>') +
      ' <button class="vxe-btn" data-act="levels-panel">⚙ Manage</button>') +
    (ov ? field('Overlaps', '<span style="color:var(--vxe-bad)">' + ov + ' clips overlap</span> <button class="vxe-btn" data-act="stack">↕ Stack</button> <button class="vxe-btn" data-act="spread">⇥ Remove</button>') : '') +
    '<div class="vxe-muted-note">Select a clip on the timeline to move it (left/right in time, up/down to another row), trim, cut, ' +
    'join, mute, edit its text or regenerate it. Lines marked ⚠ did not pass the voice check; a dashed outline is a line\'s subtitle time.</div>';
}

function renderClipPanel(key) {
  const l = lineOf(key), c = S.clips[key], idx = l.index;
  const edit = S.edits[idx] || {};
  const k = bestRegion(l);
  const reg = k == null ? null : regions()[k];
  const len = clipLen(key);
  const pieces = keysOfLine(idx);
  const working = S.working.has(idx);
  const t = now();
  const under = c.start + MIN_PIECE < t && c.start + len - MIN_PIECE > t;
  insp.innerHTML =
    '<h3><span class="vxe-dot" style="background:' + colorOf(l.speaker) + '"></span>#' + idx +
    (pieces.length > 1 ? ' ✂ piece ' + (pieces.indexOf(key) + 1) + ' of ' + pieces.length : '') + ' · ' + esc(l.speaker) +
    ' <span class="vxe-badge ' + (l.ok ? 'vxe-ok' : 'vxe-warn') + '">' + esc(working ? '⏳ regenerating…' : l.status) + '</span></h3>' +
    (l.url ? field('Starts at', '<input type="number" step="0.01" min="0" data-f="start" value="' + c.start.toFixed(2) + '"> s · ends ' + fmt(c.start + len) +
        ' · row ' + ((c.lane || 0) + 1)) +
      field('Subtitle', fmt(l.start) + ' – ' + fmt(l.end) + (Math.abs(c.start - l.start) > 0.005 ? ' (clip ' + (c.start - l.start > 0 ? '+' : '') + (c.start - l.start).toFixed(2) + ' s)' : '')) +
      field('In video', reg ? 'speech ' + fmt(reg[0]) + ' – ' + fmt(reg[1]) + ' (' + (c.start - reg[0] > 0 ? '+' : '') + (c.start - reg[0]).toFixed(2) + ' s)' : '<span class="vxe-muted-note">no speech found near it</span>') +
      field('Length', len.toFixed(2) + ' s' + (Math.abs(speedOf(key) - 1) > 1e-3 ? ' at ' + speedLabel(speedOf(key)) : '') + ' · line ' + (l.duration || 0).toFixed(2) + ' s · subtitle ' + (l.end - l.start).toFixed(2) + ' s' +
        ((c.trim_in || c.trim_out) && pieces.length === 1 ? ' (trimmed) <button class="vxe-btn" data-act="untrim">Undo trim</button>' : '')) +
      field('Volume', '<input type="range" min="-20" max="12" step="1" data-f="gain_db" value="' + (c.gain_db || 0) + '"> <span data-role="gain-val">' + (c.gain_db || 0) + ' dB</span>' +
        (levelOn() && S.buffers[l.url] ? ' <span class="vxe-muted-note" title="Added by 🎚 Level voices so this clip is as loud as the others">· levelled ' +
          (autoGainDb(key) >= 0 ? '+' : '') + autoGainDb(key).toFixed(1) + ' dB</span>' : '') +
        ' <button class="vxe-btn vxe-small" data-act="levels-panel" title="Loudness of every voice clip">🎚</button>') +
      field('Speed', '<input type="range" min="' + SPEED_MIN + '" max="' + SPEED_MAX + '" step="0.05" data-f="speed" value="' + speedOf(key) +
        '"> <span data-role="speed-val">' + speedLabel(speedOf(key)) + '</span> ' + speedButtons) +
      field('', '<label class="vxe-check"><input type="checkbox" data-f="muted"' + (c.muted ? ' checked' : '') + '> Muted (left out of the preview and export)</label>') +
      '<div class="vxe-actions">' + editTools.replace('data-act="cut-sel"', 'data-act="cut-sel"' + (under ? '' : ' disabled')) +
      (pieces.length > 1 ? '<button class="vxe-btn" data-act="delete-sel" title="Delete this piece (Delete)">🗑 Delete piece</button>' : '') + '</div>' +
      (under ? '' : '<div class="vxe-muted-note">To cut: put the playhead on this clip (click the ruler), or Alt+click the clip where you want to cut.</div>')
      : '<div class="vxe-muted-note">This line has not been generated yet — click Regenerate to make it.</div>') +
    field('Speaker', '<select data-e="speaker">' + speakers().map((n) => '<option' + (n === (edit.speaker || l.speaker) ? ' selected' : '') + '>' + esc(n) + '</option>').join('') + '</select>') +
    field('Emotion', '<input type="text" data-e="emotion" value="' + esc(edit.emotion != null ? edit.emotion : l.emotion) + '" placeholder="happy, sad, angry, … or any word">') +
    field('Tone', '<input type="text" data-e="tone" value="' + esc(edit.tone != null ? edit.tone : (l.tone || '')) +
      '" placeholder="how it is said, e.g. louder and more forceful, speaking faster" title="Read from the original video when you added it on the SRT → Speech tab">') +
    '<textarea data-e="text">' + esc(edit.text != null ? edit.text : l.text) + '</textarea>' +
    (Object.keys(edit).length ? '<div class="vxe-muted-note">✏️ Edited — click <b>Regenerate</b> to hear it.</div>' : '') +
    '<div class="vxe-actions">' +
    (l.url ? '<button class="vxe-btn" data-act="play-clip">▶ Play</button>' +
      '<button class="vxe-btn" data-act="align-sel" title="S">🎯 Align to video speech</button>' +
      '<button class="vxe-btn" data-act="srt-sel">⏱ To SRT time</button>' +
      '<button class="vxe-btn" data-act="mute-sel" title="M">' + (c.muted ? '🔊 Unmute' : '🔇 Mute') + '</button>' : '') +
    '<button class="vxe-btn vxe-primary" data-act="regen-sel" title="R"' + (working ? ' disabled' : '') + '>🔁 Regenerate</button>' +
    '<button class="vxe-btn" data-act="new-voice" title="Design a new voice for this speaker and regenerate all of their lines"' + (working ? ' disabled' : '') + '>🎭 New voice for ' + esc(l.speaker) + '</button>' +
    '</div>' + (pieces.length > 1 ? '<div class="vxe-muted-note">Regenerating a cut line makes it one clip again.</div>' : '');
}

function renderSpeechPanel() {
  const v = S.data.video;
  if (!v) { S.panel = null; renderInspector(); return; }
  insp.innerHTML = '<h3>🎙 Speech in the video</h3>' +
    '<div class="vxe-muted-note">' + regions().length + ' stretch(es) of speech found in ' + esc(v.name) +
    '. They show where each line should start; <b>Align</b> moves clips onto them.</div>' +
    field('Sensitivity', '<input type="range" min="0" max="1" step="0.05" data-role="sens" value="' + (v.sensitivity != null ? v.sensitivity : 0.5) + '"> <span class="vxe-muted-note">higher finds quieter speech</span>') +
    field('', '<label class="vxe-check"><input type="checkbox" data-role="iso"' + (v.isolated ? ' checked' : '') +
      '> Isolate voices from music first (slower, much better with music)</label>') +
    '<div class="vxe-actions"><button class="vxe-btn vxe-primary" data-act="detect">Detect again</button>' +
    '<button class="vxe-btn" data-act="close-panel">Close</button></div>';
}

function currentEdits(lineIds) {
  return lineIds.filter((i) => S.edits[i]).map((i) => Object.assign({ index: i }, S.edits[i]));
}

insp.addEventListener('input', (e) => {
  const sel = selKeys();
  const f = e.target.dataset.f, ed = e.target.dataset.e;
  if (f === 'gain_db' && sel.length === 1) {
    if (!S.pendingUndo) S.pendingUndo = snapshot();
    S.clips[sel[0]].gain_db = +e.target.value;
    insp.querySelector('[data-role="gain-val"]').textContent = e.target.value + ' dB';
    if (S.playing) schedule(now());
  } else if (f === 'speed' && sel.length === 1) {
    // Live while dragging the slider (pitch follows until the stretched audio is ready on release).
    if (!S.pendingUndo) S.pendingUndo = snapshot();
    S.clips[sel[0]].speed = +e.target.value;
    insp.querySelector('[data-role="speed-val"]').textContent = speedLabel(+e.target.value);
    updateClipEl(sel[0]);
    if (S.playing) schedule(now());
  } else if (ed && sel.length === 1) {
    const l = lineOf(sel[0]);
    const edit = S.edits[l.index] = S.edits[l.index] || {};
    edit[ed] = e.target.value;
    if (edit[ed] === (ed === 'text' ? l.text : ed === 'emotion' ? l.emotion : ed === 'tone' ? (l.tone || '') : l.speaker)) delete edit[ed];
    if (!Object.keys(edit).length) delete S.edits[l.index];
  }
});
insp.addEventListener('change', async (e) => {
  const sel = selKeys();
  if (sel.length !== 1) return;
  const key = sel[0], idx = lineIdOf(key), f = e.target.dataset.f, ed = e.target.dataset.e;
  if (f === 'gain_db') {
    pushUndo(S.pendingUndo); S.pendingUndo = null; changed(true);
  } else if (f === 'speed') {
    pushUndo(S.pendingUndo); S.pendingUndo = null; changed(); loadStretched(key);
  } else if (f === 'start') {
    pushUndo(); S.clips[key].start = r3(Math.max(0, +e.target.value || 0)); changed();
  } else if (f === 'muted') {
    pushUndo(); S.clips[key].muted = e.target.checked; changed();

  } else if (ed && S.edits[idx]) {
    // Keep the edit in the project tables (it is heard after Regenerate).
    try {
      const keep = S.edits[idx];
      await save();
      applyPayload(await call('editor_edit_lines', { edits: currentEdits([idx]) }));
      S.edits[idx] = keep;
      renderInspector();
      setStatus('✏️ Line #' + idx + ' saved — click <b>Regenerate</b> to hear the change.');
    } catch (err) { showError(err); }
  }
});
function runAction(act) {
  const sel = selKeys();
  const lines = selLines();
  if (act === 'play-clip' && sel.length) { seek(S.clips[sel[0]].start); play(); }
  else if (act === 'align-sel') alignClips(lines);
  else if (act === 'srt-sel') toSrt(lines);
  else if (act === 'mute-sel') toggleMute(generated(sel));
  else if (act === 'untrim' && sel.length) { pushUndo(); S.clips[sel[0]].trim_in = 0; S.clips[sel[0]].trim_out = 0; changed(); }
  else if (act === 'cut-sel') cutAt(now(), sel.length ? sel : null);
  else if (act === 'join-sel') joinPieces(sel);
  else if (act === 'delete-sel') deletePieces(generated(sel));
  else if (act === 'slower') stepSpeed(sel, -0.05);
  else if (act === 'faster') stepSpeed(sel, 0.05);
  else if (act === 'speed-1') setSpeed(sel, 1);
  else if (act === 'speed-fit') fitSpeed(sel);
  else if (act === 'up-sel') moveRows(generated(sel), -1);
  else if (act === 'down-sel') moveRows(generated(sel), 1);
  else if (act === 'stack') { pushUndo(); const rows = stackOverlaps(); changed(); setStatus('↕ Overlapping clips are on separate rows (' + rows + ' row(s) at most).'); }
  else if (act === 'spread') removeOverlaps();
  else if (act === 'regen-sel') regenerate(lines);
  else if (act === 'new-voice' && lines.length) {
    const name = (S.edits[lines[0]] && S.edits[lines[0]].speaker) || S.lineMap[lines[0]].speaker;
    regenerate(lines, name);
  }
  else if (act === 'speech-panel') { S.panel = 'speech'; renderInspector(); }
  else if (act === 'close-panel') { S.panel = null; renderInspector(); }
  else if (act === 'levels-panel') { S.panel = 'levels'; S.sel.clear(); S.selCue = null; S.selSec = null; render(); renderInspector(); }
  else if (act === 'lv-reset') resetClipVolumes();
  else if (act === 'detect') detectAgain();
  else return false;
  return true;
}
insp.addEventListener('click', (e) => {
  const b = e.target.closest('[data-act]');
  if (b && !b.disabled) runAction(b.dataset.act);
});

// ---------- regenerate (runs in the background on the server) ----------
function markWorking(job) {
  S.working.clear();
  if (job.new_voices && job.new_voices.length) S.lines.forEach((l) => { if (job.new_voices.includes(l.speaker)) S.working.add(l.index); });
  else if (job.only && job.only.length) job.only.forEach((i) => S.working.add(i));
  else S.lines.forEach((l) => S.working.add(l.index));
  render();
  renderInspector();
}
function showJob(job) {
  const pct = Math.round(100 * (job.i || 0) / Math.max(job.n || 1, 1));
  setStatus('⏳ <span>' + esc(job.desc || 'Starting…') + '</span><div class="vxe-progress"><div style="width:' + pct + '%"></div></div>' +
    '<span class="vxe-muted-note">You can keep editing; it carries on if you close the tab.</span>');
}
async function regenerate(lineIds, newVoice) {
  if (!lineIds.length) return;
  if (newVoice && !window.confirm('Design a new voice for ' + newVoice + ' and regenerate ALL of their lines?')) return;
  try {
    const redo = newVoice ? S.lines.filter((l) => l.speaker === newVoice).map((l) => l.index) : lineIds;
    if (redo.some((i) => keysOfLine(i).length > 1)) { pushUndo(); collapseLines(redo); changed(); }
    await save();
    const r = await call('editor_regenerate', { indices: lineIds, edits: currentEdits(lineIds), new_voice: newVoice || null });
    markWorking(r.job);
    showJob(r.job);
    startPolling();
  } catch (e) { showError(e); }
}
function startPolling() {
  if (S.poll) return;
  S.poll = setInterval(async () => {
    try {
      const r = await call('editor_status');
      if (r.job && r.job.status === 'running') { showJob(r.job); return; }
      clearInterval(S.poll); S.poll = null;
      const job = r.job;
      if (r.lines) applyPayload(r);
      S.working.clear();
      render(); renderInspector();
      if (job && job.status === 'failed') setStatus('<span style="color:var(--vxe-bad)">⚠️ Regeneration stopped: ' + esc(job.error) + '</span>');
      else setStatus('✅ Done — the new audio is on the timeline (positions, rows and volumes were kept).' +
        (S.lines.some((l) => l.url && !l.ok) ? ' Lines still marked ⚠ can be regenerated again.' : ''));
    } catch (e) { clearInterval(S.poll); S.poll = null; showError(e); }
  }, 1000);
}

// ---------- video ----------
async function useVideoFile(file) {
  if (!file || S.busy) return;
  if (!S.data) { showError('Generate the lines on the SRT → Speech tab first.'); return; }
  const isolate = q('isolate-upload').checked;
  try {
    busy('Uploading ' + file.name + '…');
    const up = await upload(file);
    busy('Reading the video and finding the speech in it…' + (isolate ? ' Separating voices from music can take a few minutes.' : ''));
    pause();
    applyPayload(await call('editor_video', { path: up.path, isolate: isolate }));
    setStatus('🎬 ' + esc(file.name) + ' loaded · ' + regions().length + ' stretch(es) of speech found. Clips sit on their subtitle times; ' +
      'use <b>🎯 Align all to video speech</b> to move them onto the speech in the video.');
  } catch (e) { showError(e); } finally { busy(''); root.focus({ preventScroll: true }); }
}
async function detectAgain() {
  const sens = +insp.querySelector('[data-role="sens"]').value;
  const iso = insp.querySelector('[data-role="iso"]').checked;
  try {
    busy(iso && !S.data.video.isolated ? 'Separating voices from music and finding the speech… (a few minutes)' : 'Finding the speech in the video…');
    applyPayload(await call('editor_detect', { sensitivity: sens, isolate: iso }));
    S.panel = 'speech';
    renderInspector();
    setStatus('🎙 ' + regions().length + ' stretch(es) of speech found.');
  } catch (e) { showError(e); } finally { busy(''); }
}
q('file').addEventListener('change', (e) => { useVideoFile(e.target.files[0]); e.target.value = ''; });
const viewer = q('viewer'), drop = q('drop');
viewer.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('vxe-over'); });
viewer.addEventListener('dragleave', () => drop.classList.remove('vxe-over'));
viewer.addEventListener('drop', (e) => {
  e.preventDefault();
  drop.classList.remove('vxe-over');
  const file = [...(e.dataTransfer.files || [])].find((f) => f.type.startsWith('video/') || f.type.startsWith('audio/') || /\.(mkv|avi|mov|mp4|webm|m4v|wav|mp3|m4a|aac|flac|ogg|opus)$/i.test(f.name));
  if (file) useVideoFile(file);
});

// ---------- export ----------
async function doExport() {
  if (!S.data || !S.data.video) { showError('Add a video first.'); return; }
  pause();
  try {
    const cleaning = ((S.mix.subs && S.mix.subs.mode !== 'off') || (subsOn() && S.subs.cues.length)) && S.data.video.has_video !== false;
    busy('Exporting: mixing the new voices onto the video…' + (S.mix.original === 'music' && !S.data.video.isolated ? ' Separating the music first can take a few minutes.' : '') +
      (cleaning ? ' Removing the subtitles from the picture re-encodes it at its original size (about half the video\'s length).' : ''));
    clearTimeout(S.saveTimer);
    const overlays = await subtitleImages();
    const r = await call('editor_export', { clips: S.clips, mix: S.mix, tracks: S.tracks, subtitles: S.subs || undefined, overlays: overlays });
    const a = (url, name, label) => '<a href="' + esc(url) + '" download="' + esc(name) + '">' + label + '</a>';
    setStatus('✅ Exported: ' + (r.video_url ? a(r.video_url, r.video_name, '⬇ ' + esc(r.video_name)) + ' · ' : '') +
      a(r.soundtrack_url, r.soundtrack_name, '⬇ soundtrack (.wav)') + ' · ' + a(r.vocals_url, r.vocals_name, '⬇ new voices only (.wav)'));
  } catch (e) { showError(e); } finally { busy(''); }
}

// ---------- subtitles burned into the picture ----------
function videoRect() {
  // Where the picture is drawn inside the <video> element (object-fit: contain), relative to the viewer.
  const vw = video.videoWidth, vh = video.videoHeight;
  const el = video.getBoundingClientRect(), host = q('viewer').getBoundingClientRect();
  if (!vw || !vh || !el.width) return null;
  const scale = Math.min(el.width / vw, el.height / vh);
  const w = vw * scale, h = vh * scale;
  return { x: el.left - host.left + (el.width - w) / 2, y: el.top - host.top + (el.height - h) / 2, w, h };
}
function placeSubBox() {
  const box = q('subbox'), subs = S.mix.subs || {};
  const v = S.data && S.data.video;
  const r = v && v.has_video !== false && hasVideo() ? videoRect() : null;
  const on = !!(r && subs.box && (subs.mode !== 'off' || S.editSubs));
  box.classList.toggle('vxe-on', on);
  box.classList.toggle('vxe-edit', on && !!S.editSubs);
  if (!on) return;
  const [x, y, w, h] = subs.box;
  box.style.left = (r.x + x * r.w) + 'px'; box.style.top = (r.y + y * r.h) + 'px';
  box.style.width = (w * r.w) + 'px'; box.style.height = (h * r.h) + 'px';
  // Kept: only the outline while editing; filled in / blurred: the preview blurs it (the export fills it in).
  const blur = subs.mode === 'off' ? '0px' : subs.mode === 'blur' ? '9px' : '14px';
  box.style.backdropFilter = box.style.webkitBackdropFilter = 'blur(' + blur + ')';
}
q('subbox').addEventListener('pointerdown', (e) => {
  if (!S.editSubs) return;
  e.preventDefault(); e.stopPropagation();
  const r = videoRect(), subs = S.mix.subs;
  if (!r || !subs || !subs.box) return;
  const corner = e.target.dataset.corner;
  const [x0, y0, w0, h0] = subs.box, px = e.clientX, py = e.clientY;
  const el = q('subbox');
  el.setPointerCapture(e.pointerId);
  const move = (ev) => {
    const dx = (ev.clientX - px) / r.w, dy = (ev.clientY - py) / r.h;
    let [x, y, w, h] = [x0, y0, w0, h0];
    if (!corner) { x = x0 + dx; y = y0 + dy; }
    else {
      if (corner.includes('w')) { x = Math.min(x0 + dx, x0 + w0 - 0.03); w = w0 - (x - x0); }
      if (corner.includes('e')) w = Math.max(w0 + dx, 0.03);
      if (corner.includes('n')) { y = Math.min(y0 + dy, y0 + h0 - 0.01); h = h0 - (y - y0); }
      if (corner.includes('s')) h = Math.max(h0 + dy, 0.01);
    }
    w = Math.min(w, 1); h = Math.min(h, 1);
    subs.box = [clamp(x, 0, 1 - w), clamp(y, 0, 1 - h), w, h].map((v) => Math.round(v * 10000) / 10000);
    placeSubBox();
  };
  const up = () => { el.removeEventListener('pointermove', move); el.removeEventListener('pointerup', up); changed(true); };
  el.addEventListener('pointermove', move);
  el.addEventListener('pointerup', up);
});
q('subs-mode').addEventListener('change', (e) => {
  S.mix.subs = Object.assign({ box: [0.1, 0.75, 0.8, 0.07] }, S.mix.subs || {}, { mode: e.target.value });
  changed(true);
  setStatus(e.target.value === 'off' ? 'The subtitles in the picture are kept.' :
    'The subtitles in the picture are ' + (e.target.value === 'fill' ? 'removed (filled in from around them)' : 'blurred') +
    ' in the export — the video is re-encoded at its original size (about half its length to export). ' +
    'The preview shows the area blurred; <b>✏️ Area</b> moves it.');
});
async function findSubtitles() {
  try {
    busy('Looking for subtitles in the picture…');
    const r = await call('editor_find_subtitles');
    applyPayload(r);
    S.editSubs = true;
    setStatus(r.found ? '🔍 Found the subtitles — the area is shown on the video; drag it or its corners to adjust, then ✏️ Area to finish.'
      : '🔍 No subtitles found in the picture. If there are some, use ✏️ Area and drag the box over them.');
  } catch (e) { showError(e); } finally { busy(''); }
}

// ---------- new subtitles ----------
const SUB_FONT = '"Noto Sans Khmer", "Khmer Sangam MN", "Khmer MN", "Noto Sans", sans-serif';
const cueById = (id) => (S.subs ? S.subs.cues.find((c) => c.id === id) : null);
const subsOn = () => !!(S.subs && S.subs.burn !== false);
function subsFromVoices() {
  // One subtitle per line that is heard, from the start of its first piece to the end of its last.
  const byLine = {};
  keys().forEach((k) => {
    if (!audible(k)) return;
    const i = lineIdOf(k), a = S.clips[k].start, b = clipEnd(k);
    const g = byLine[i] = byLine[i] || { start: a, end: b };
    g.start = Math.min(g.start, a); g.end = Math.max(g.end, b);
  });
  const cues = Object.keys(byLine).map((i) => ({ start: r3(byLine[i].start), end: r3(byLine[i].end), text: S.lineMap[i].text, lane: 0 }))
    .sort((a, b) => a.start - b.start).map((c, n) => Object.assign({ id: 'c' + (n + 1) }, c));
  cues.forEach((c, n) => { while (cues.slice(0, n).some((o) => o.lane === c.lane && cuesOverlap(o, c))) c.lane++; });
  return cues;
}
function newCueId() {
  let n = (S.subs ? S.subs.cues.length : 0) + 1;
  while (cueById('c' + n)) n++;
  return 'c' + n;
}
function ensureSubs() {
  if (!S.subs) S.subs = Object.assign({}, SUB_DEFAULTS, { cues: [] });
  return S.subs;
}
function sortCues() { S.subs.cues.sort((a, b) => a.start - b.start); }
const CUE_ROW = 30; // height of one row of the subtitle track
const cueRows = () => Math.max(1, ...(S.subs ? S.subs.cues : []).map((c) => (c.lane || 0) + 1));
const cuesOverlap = (a, b) => a.start < b.end - 0.01 && b.start < a.end - 0.01;
function settleCues(keep, moving) {
  // Subtitles that overlap on the same row move to a free row: ``keep`` stays where it was put, ``moving`` (one just
  // dragged, added or split off) is the one that moves aside; empty rows are dropped. Row 1 is drawn at the usual
  // place on the video, higher rows above it.
  const rank = (c) => (c === keep ? -1 : c === moving ? 1 : 0);
  const cues = S.subs.cues.slice().sort((a, b) => rank(a) - rank(b) || a.start - b.start);
  const placed = [];
  cues.forEach((c) => {
    let lane = c.lane || 0;
    const busy = (l) => placed.some((o) => (o.lane || 0) === l && cuesOverlap(o, c));
    if (busy(lane)) { let l = 0; while (busy(l)) l++; lane = l; }
    c.lane = lane;
    placed.push(c);
  });
  const used = [...new Set(S.subs.cues.map((c) => c.lane || 0))].sort((a, b) => a - b);
  S.subs.cues.forEach((c) => { c.lane = used.indexOf(c.lane || 0); });
  sortCues();
}
function moveCueRow(id, d) {
  const c = cueById(id);
  if (!c) return;
  pushUndo();
  c.lane = Math.max(0, (c.lane || 0) + d);
  // Moving onto a row where it overlaps something: the other subtitle takes this one's old row instead.
  S.subs.cues.forEach((o) => { if (o !== c && (o.lane || 0) === c.lane && cuesOverlap(o, c)) o.lane = Math.max(0, c.lane - d); });
  settleCues(c);
  changed();
}
function cueHtml(c) {
  return '<div class="vxe-cue' + (c.id === S.selCue ? ' vxe-selected' : '') + (c.text.trim() ? '' : ' vxe-empty') + '" data-cue="' + esc(c.id) +
    '" title="' + esc(fmt(c.start) + ' → ' + fmt(c.end) + '\n' + c.text) + '" style="left:' + tToX(c.start) + 'px;width:' +
    Math.max(tToX(c.end - c.start), 4) + 'px;top:' + (4 + (c.lane || 0) * CUE_ROW) + 'px;height:' + (CUE_ROW - 6) +
    'px"><div class="vxe-handle vxe-l" data-h="l"></div><div class="vxe-handle vxe-r" data-h="r"></div>' +
    '<div class="vxe-clip-label">' + esc(c.text || '(empty)') + '</div></div>';
}
function selectCue(id) {
  S.selCue = id; S.sel.clear(); S.panel = null;
  renderSelection();
}
function addCue(t) {
  const subs = ensureSubs();
  pushUndo();
  // The text of the line heard there, if any; up to 2 s, stopping before the next subtitle.
  const heard = byStart(keys().filter((k) => audible(k) && S.clips[k].start <= t + 0.01 && clipEnd(k) > t));
  const next = subs.cues.filter((c) => c.start > t + 0.05).map((c) => c.start);
  const end = Math.min(t + 2, next.length ? Math.min(...next) - 0.05 : t + 2);
  const cue = { id: newCueId(), start: r3(t), end: r3(Math.max(end, t + 0.4)), text: heard.length ? lineOf(heard[0]).text : '' };
  subs.cues.push(cue); settleCues(null, cue);
  S.selCue = cue.id; S.sel.clear(); S.panel = null;
  changed();
  const box = insp.querySelector('[data-cf="text"]');
  if (box) box.focus();
  setStatus('💬 Subtitle added at ' + fmt(t) + ' — type its text on the right; drag it or its edges on the track to change its time.');
}
function deleteCue(id) {
  if (!cueById(id)) return;
  pushUndo();
  S.subs.cues = S.subs.cues.filter((c) => c.id !== id);
  S.selCue = null;
  changed();
}
function splitCueAt(id, t) {
  const c = cueById(id);
  if (!c || t <= c.start + 0.2 || t >= c.end - 0.2) { setStatus('✂ Put the playhead inside the subtitle (not at its very edge) to split it there.'); return; }
  pushUndo();
  const [a, b] = splitWords(c.text, (t - c.start) / (c.end - c.start));
  const second = { id: newCueId(), start: r3(t), end: c.end, text: b };
  c.end = r3(t); c.text = a;
  S.subs.cues.push(second); settleCues(c, second);
  S.selCue = second.id;
  changed();
}
function splitWords(text, frac) {
  // At the word boundary nearest to frac of the text (Intl.Segmenter knows Khmer words).
  const target = text.length * clamp(frac, 0, 1);
  let cuts = [];
  try { for (const p of new Intl.Segmenter(undefined, { granularity: 'word' }).segment(text)) if (p.index > 0) cuts.push(p.index); } catch (e) { /* none */ }
  if (!cuts.length) return [text, ''];
  const at = cuts.reduce((best, x) => (Math.abs(x - target) < Math.abs(best - target) ? x : best), cuts[0]);
  return [text.slice(0, at).trim(), text.slice(at).trim()];
}
function toggleSubs() {
  ensureSubs();
  S.subs.burn = !subsOn();
  if (S.subs.burn && !S.subs.cues.length && S.lines.some((l) => l.url)) S.subs.cues = subsFromVoices();
  showCaptionsButton();
  changed(true);
  setStatus(subsOn() ? '💬 New subtitles on — shown on the video ' + (S.mix.subs && S.mix.subs.mode !== 'off' ? 'where the original ones were removed' : 'near the bottom') + ', and burned into the export.'
    : '💬 New subtitles off — not shown and not in the export (they are kept on the track).');
}
function rebuildSubs() {
  if (S.subs && S.subs.cues.length && !window.confirm('Make the subtitles again from the voices? Your subtitle edits are replaced (Undo brings them back).')) return;
  pushUndo();
  ensureSubs().cues = subsFromVoices();
  S.selCue = null;
  changed();
  setStatus('↻ ' + S.subs.cues.length + ' subtitles made from the voices.');
}

function startCueDrag(e, el) {
  const id = el.dataset.cue, c = cueById(id), handle = e.target.dataset.h;
  if (S.selCue !== id) selectCue(id);
  const before = snapshot(), c0 = Object.assign({}, c), x0 = e.clientX, y0 = e.clientY;
  let moved = false;
  el.setPointerCapture(e.pointerId);
  const snaps = [now()];
  keys().forEach((k) => { if (audible(k)) snaps.push(S.clips[k].start, clipEnd(k)); });
  S.subs.cues.forEach((o) => { if (o.id !== id) snaps.push(o.start, o.end); });
  const snapTo = (t) => {
    let best = t, dist = 8 / S.pps;
    snaps.forEach((p) => { if (Math.abs(p - t) < dist) { dist = Math.abs(p - t); best = p; } });
    return best;
  };
  const onMove = (ev) => {
    if (!moved && Math.abs(ev.clientX - x0) < 3 && Math.abs(ev.clientY - y0) < 6) return;
    moved = true;
    const dt = (ev.clientX - x0) / S.pps, snapOn = q('snap').checked && !ev.altKey;
    const len = c0.end - c0.start;
    if (!handle) {
      let start = Math.max(0, c0.start + dt);
      if (snapOn) { const s1 = snapTo(start), s2 = snapTo(start + len) - len; start = Math.abs(s1 - start) <= Math.abs(s2 - start) ? s1 : s2; }
      c.start = r3(Math.max(0, start)); c.end = r3(c.start + len);
      // Up / down: to another row (a new one below the last too).
      c.lane = clamp((c0.lane || 0) + Math.round((ev.clientY - y0) / CUE_ROW), 0, 7);
      el.style.top = (4 + c.lane * CUE_ROW) + 'px';
    } else if (handle === 'l') {
      let start = c0.start + dt;
      if (snapOn) start = snapTo(start);
      c.start = r3(clamp(start, 0, c.end - 0.2));
    } else {
      let end = c0.end + dt;
      if (snapOn) end = snapTo(end);
      c.end = r3(Math.max(end, c.start + 0.2));
    }
    el.style.left = tToX(c.start) + 'px';
    el.style.width = Math.max(tToX(c.end - c.start), 4) + 'px';
    q('time').textContent = fmt(handle === 'r' ? c.end : c.start);
    drawPreviewSubs(now(), true);
  };
  const onUp = () => {
    el.removeEventListener('pointermove', onMove);
    el.removeEventListener('pointerup', onUp);
    el.removeEventListener('pointercancel', onUp);
    if (!moved) return;
    pushUndo(before);
    if ((c.lane || 0) !== (c0.lane || 0)) {
      // Dragged to another row: what it lands on there takes its old row (a swap), it stays where it was put.
      S.subs.cues.forEach((o) => { if (o !== c && (o.lane || 0) === c.lane && cuesOverlap(o, c)) o.lane = c0.lane || 0; });
      settleCues(c);
    } else settleCues(null, c);
    changed();
  };
  el.addEventListener('pointermove', onMove);
  el.addEventListener('pointerup', onUp);
  el.addEventListener('pointercancel', onUp);
}

function renderCuePanel(c) {
  insp.innerHTML = '<h3>💬 Subtitle</h3>' +
    field('Starts at', '<input type="number" step="0.01" min="0" data-cf="start" value="' + c.start.toFixed(2) + '"> s') +
    field('Ends at', '<input type="number" step="0.01" min="0" data-cf="end" value="' + c.end.toFixed(2) + '"> s · ' + (c.end - c.start).toFixed(2) + ' s long') +
    '<textarea data-cf="text" placeholder="Subtitle text (Enter for a new line)">' + esc(c.text) + '</textarea>' +
    '<div class="vxe-actions"><button class="vxe-btn" data-act="cue-play">▶ Play</button>' +
    '<button class="vxe-btn" data-act="cue-split" title="Split it in two at the playhead">✂ Split at playhead</button>' +
    '<button class="vxe-btn" data-act="cue-up" title="Row up (Alt+↑) — drawn lower on the video">⬆ Row up</button>' +
    '<button class="vxe-btn" data-act="cue-down" title="Row down (Alt+↓) — drawn higher on the video">⬇ Row down</button>' +
    '<button class="vxe-btn" data-act="cue-add-here">+ New subtitle at playhead</button>' +
    '<button class="vxe-btn" data-act="cue-del" title="Delete">🗑 Delete</button></div>' +
    '<div class="vxe-muted-note">Drag it on the 💬 Subtitles track to move it, its edges to change when it starts or ends. ' +
    'It is drawn on the video ' + (S.mix.subs && S.mix.subs.mode !== 'off' ? 'where the original subtitles were removed' : 'near the bottom') +
    '. Subtitles that overlap in time go on separate rows (drag up/down): row ' + ((c.lane || 0) + 1) + ' of ' + cueRows() +
    '; a higher row is drawn above the lower ones.</div>';
}
function renderSubsPanel() {
  const subs = ensureSubs();
  insp.innerHTML = '<h3>💬 Subtitles</h3>' +
    field('', '<label class="vxe-check"><input type="checkbox" data-sf="burn"' + (subsOn() ? ' checked' : '') + '> Show them on the video (preview and export)</label>') +
    field('Text size', '<input type="range" min="0.02" max="0.06" step="0.002" data-sf="size" value="' + subs.size + '"> <span data-role="size-val">' + Math.round(subs.size * 1000) / 10 + '% of the height</span>') +
    field('Where', S.mix.subs && S.mix.subs.mode !== 'off' ? 'Over the removed original subtitles (move it with ✏️ Area)' : 'Near the bottom (remove the original subtitles to put them in their place)') +
    field('Count', subs.cues.length + ' subtitle(s)') +
    '<div class="vxe-actions"><button class="vxe-btn" data-act="subs-rebuild" title="One subtitle per voice line, on its time">↻ Make again from the voices</button>' +
    '<button class="vxe-btn" data-act="close-panel">Close</button></div>' +
    '<div class="vxe-muted-note">Add one: double-click the track, the + button, or N. Edit one: click it. Remove one: select it and press Delete.</div>';
}
insp.addEventListener('input', (e) => {
  const cf = e.target.dataset.cf, sf = e.target.dataset.sf;
  if (cf === 'text' && S.selCue) {
    if (!S.pendingUndo) S.pendingUndo = snapshot();
    cueById(S.selCue).text = e.target.value;
    const label = content.querySelector('[data-cue="' + CSS.escape(S.selCue) + '"] .vxe-clip-label');
    if (label) label.textContent = e.target.value || '(empty)';
    drawPreviewSubs(now(), true);
  } else if (sf === 'size') {
    if (!S.pendingUndo) S.pendingUndo = snapshot();
    S.subs.size = +e.target.value;
    insp.querySelector('[data-role="size-val"]').textContent = Math.round(S.subs.size * 1000) / 10 + '% of the height';
    drawPreviewSubs(now(), true);
  }
});
insp.addEventListener('change', (e) => {
  const cf = e.target.dataset.cf, sf = e.target.dataset.sf;
  if (cf && S.selCue) {
    const c = cueById(S.selCue);
    if (cf === 'text') { pushUndo(S.pendingUndo); S.pendingUndo = null; changed(true); return; }
    pushUndo();
    if (cf === 'start') c.start = r3(clamp(+e.target.value || 0, 0, c.end - 0.2));
    if (cf === 'end') c.end = r3(Math.max(+e.target.value || 0, c.start + 0.2));
    settleCues(null, c); changed();
  } else if (sf === 'size') { pushUndo(S.pendingUndo); S.pendingUndo = null; changed(true); }
  else if (sf === 'burn') toggleSubs();
});
insp.addEventListener('click', (e) => {
  const b = e.target.closest('[data-act]');
  if (!b || (!b.dataset.act.startsWith('cue-') && !b.dataset.act.startsWith('subs-'))) return;
  const act = b.dataset.act;
  if (act === 'cue-play' && S.selCue) { seek(cueById(S.selCue).start); play(); }
  else if (act === 'cue-split' && S.selCue) splitCueAt(S.selCue, now());
  else if (act === 'cue-add-here') addCue(now());
  else if (act === 'cue-del' && S.selCue) deleteCue(S.selCue);
  else if (act === 'cue-up' && S.selCue) moveCueRow(S.selCue, -1);
  else if (act === 'cue-down' && S.selCue) moveCueRow(S.selCue, 1);
  else if (act === 'subs-rebuild') rebuildSubs();
});

// Drawing: the same layout for the preview (canvas over the picture) and the export (one image per subtitle).
const measure = document.createElement('canvas').getContext('2d');
function subLayout(text, W, H) {
  const px = Math.max(Math.round(H * (S.subs ? S.subs.size : SUB_DEFAULTS.size)), 8);
  const font = '600 ' + px + 'px ' + SUB_FONT;
  measure.font = font;
  const maxW = W * 0.9, lines = [];
  String(text || '').split('\n').forEach((para) => {
    let words = [];
    try { words = [...new Intl.Segmenter(undefined, { granularity: 'word' }).segment(para)].map((p) => p.segment); } catch (e) { words = para.split(/(\s+)/); }
    let line = '';
    words.forEach((w) => {
      if (line && measure.measureText(line + w).width > maxW) { lines.push(line.trim()); line = w.trimStart(); } else line += w;
    });
    if (line.trim()) lines.push(line.trim());
  });
  const lineH = Math.round(px * 1.45);
  const box = S.mix.subs && S.mix.subs.mode !== 'off' && S.mix.subs.box ? S.mix.subs.box : null;
  const cy = box ? (box[1] + box[3] / 2) * H : H * 0.86;
  return { font, px, lines, lineH, cy, W, H };
}
function drawLayout(g, L, shift) {
  g.font = L.font; g.textAlign = 'center'; g.textBaseline = 'middle'; g.lineJoin = 'round';
  const top = L.cy - (L.lines.length * L.lineH) / 2 - (shift || 0);
  L.lines.forEach((line, i) => {
    const y = top + i * L.lineH + L.lineH / 2;
    g.lineWidth = Math.max(L.px * 0.18, 2); g.strokeStyle = 'rgba(0,0,0,0.92)'; g.strokeText(line, L.W / 2, y);
    g.fillStyle = '#ffffff'; g.fillText(line, L.W / 2, y);
  });
}
function cueShift(c, L) {
  // How far above the usual place a subtitle is drawn: the lowest row showing with it sits at the usual place,
  // each higher row right on top of the one below (so a subtitle on a higher row never covers a lower one).
  const below = S.subs.cues.filter((o) => o !== c && o.text.trim() && (o.lane || 0) < (c.lane || 0) && cuesOverlap(o, c))
    .sort((a, b) => (a.lane || 0) - (b.lane || 0));
  if (!below.length) return 0;
  const height = (o) => { const ol = o === c ? L : subLayout(o.text, L.W, L.H); return ol.lines.length * ol.lineH; };
  const gap = L.px * 0.3;
  return height(below[0]) / 2 + below.slice(1).reduce((sum, o) => sum + height(o), 0) + gap * below.length + height(c) / 2;
}
const subCanvas = q('subcanvas');
let subDrawn = '';
function drawPreviewSubs(t, force) {
  const r = hasVideo() && subsOn() ? videoRect() : null;
  const cues = r ? S.subs.cues.filter((c) => c.text.trim() && c.start <= t && c.end > t) : [];
  const key = r && cues.length ? [r.x, r.y, r.w, r.h, S.subs.size, JSON.stringify(S.mix.subs), cues.map((c) => c.id + c.text).join('|')].join(';') : '';
  if (key === subDrawn && !force) return;
  subDrawn = key;
  if (!key) { subCanvas.style.display = 'none'; return; }
  const W = video.videoWidth, Hh = video.videoHeight, dpr = window.devicePixelRatio || 1;
  Object.assign(subCanvas.style, { display: 'block', left: r.x + 'px', top: r.y + 'px', width: r.w + 'px', height: r.h + 'px' });
  subCanvas.width = Math.round(r.w * dpr); subCanvas.height = Math.round(r.h * dpr);
  const g = subCanvas.getContext('2d');
  g.setTransform(subCanvas.width / W, 0, 0, subCanvas.height / Hh, 0, 0);
  cues.forEach((c) => { const L = subLayout(c.text, W, Hh); drawLayout(g, L, cueShift(c, L)); });
}
async function subtitleImages() {
  // Each subtitle as a PNG band at the video's own size, with when and where the export puts it.
  if (!subsOn() || !hasVideo()) return [];
  const W = video.videoWidth, H = video.videoHeight;
  if (!W || !H) return [];
  try { await document.fonts.ready; } catch (e) { /* ignore */ }
  const out = [];
  S.subs.cues.filter((c) => c.text.trim() && c.end > c.start).forEach((c) => {
    const L = subLayout(c.text, W, H), shift = cueShift(c, L);
    const bandH = Math.ceil(L.lines.length * L.lineH + L.px);
    const top = clamp(Math.round(L.cy - shift - bandH / 2), 0, Math.max(H - bandH, 0));
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = bandH;
    const g = cv.getContext('2d');
    g.translate(0, -top);
    drawLayout(g, L, shift);
    out.push({ start: c.start, end: c.end, x: 0, y: top, png: cv.toDataURL('image/png').split(',')[1] });
  });
  return out;
}

// ---------- the original sound as parts (cut, volume, sound per part) ----------
const SECTION_SOUNDS = [['auto', 'As set in the toolbar'], ['full', 'Full original (its voices too)'],
  ['music', 'Voices removed (music & effects)'], ['mute', 'Silent']];
S.selSec = null;
function videoLength() { return (S.data && S.data.video && S.data.video.duration) || (isFinite(video.duration) ? video.duration : 0) || S.dur; }
function sections() {
  const list = S.mix.orig_sections;
  if (list && list.length) return list;
  return [{ start: 0, end: r3(videoLength()), gain_db: 0, sound: 'auto' }];
}
function ownSections() {
  // The parts as their own list (made the first time one is changed).
  if (!S.mix.orig_sections || !S.mix.orig_sections.length) S.mix.orig_sections = sections().map((x) => Object.assign({}, x));
  const list = S.mix.orig_sections, end = r3(videoLength());
  if (list.length && list[list.length - 1].end < end) list[list.length - 1].end = end;
  return list;
}
function sectionAt(t) { return sections().find((x) => x.start <= t && t < x.end) || { gain_db: 0, sound: 'auto' }; }
function cutSection(t) {
  const list = ownSections(), i = list.findIndex((x) => x.start + 0.05 < t && t < x.end - 0.05);
  if (i < 0) { setStatus('✂ Put the playhead inside the part of the original sound to cut it there.'); return; }
  pushUndo();
  const right = Object.assign({}, list[i], { start: r3(t) });
  list[i].end = r3(t);
  list.splice(i + 1, 0, right);
  S.selSec = i + 1;
  changed();
  setStatus('✂ Cut the original sound at ' + fmt(t) + ' — each part has its own volume and sound (click a part to change it).');
}
function joinSection(i) {
  const list = ownSections();
  if (i >= list.length - 1) { setStatus('🔗 This is the last part — select the part before the one to join.'); return; }
  pushUndo();
  list[i].end = list[i + 1].end;
  list.splice(i + 1, 1);
  if (list.length === 1 && !list[0].gain_db && list[0].sound === 'auto') delete S.mix.orig_sections;
  S.selSec = i;
  changed();
}
function muteSection(i) {
  const list = ownSections(), x = list[i];
  pushUndo();
  if (x.sound === 'mute') { x.sound = x.before || 'auto'; delete x.before; } else { x.before = x.sound; x.sound = 'mute'; }
  changed();
}
function renderSectionPanel(i) {
  const list = sections(), x = list[i];
  insp.innerHTML = '<h3>🎞 Original sound — part ' + (i + 1) + ' of ' + list.length + '</h3>' +
    field('Time', fmt(x.start) + ' – ' + fmt(x.end) + ' (' + (x.end - x.start).toFixed(2) + ' s)') +
    field('Sound', '<select data-of="sound">' + SECTION_SOUNDS.map(([v, label]) => '<option value="' + v + '"' + ((x.sound || 'auto') === v ? ' selected' : '') + '>' + label + '</option>').join('') + '</select>') +
    field('Volume', '<input type="range" min="-30" max="12" step="1" data-of="gain_db" value="' + (x.gain_db || 0) + '"> <span data-role="sec-gain">' + (x.gain_db || 0) + ' dB</span>') +
    '<div class="vxe-actions"><button class="vxe-btn" data-act="sec-play">▶ Play</button>' +
    '<button class="vxe-btn" data-act="sec-cut" title="C">✂ Cut at playhead</button>' +
    '<button class="vxe-btn" data-act="sec-join" title="J"' + (i >= list.length - 1 ? ' disabled' : '') + '>🔗 Join with the next part</button>' +
    '<button class="vxe-btn" data-act="sec-mute" title="M">' + (x.sound === 'mute' ? '🔊 Unmute' : '🔇 Silence this part') + '</button>' +
    '<button class="vxe-btn" data-act="sec-reset">↺ Back to normal</button></div>' +
    '<div class="vxe-muted-note">Cut the original sound into parts to treat them differently: e.g. keep a real scream from the ' +
    'original (Full original), take the voices out of one scene (Voices removed), or make a part quieter. Parts do not move — ' +
    'they stay in time with the picture. The toolbar\'s Original audio and Orig dB still apply to every part.</div>';
}
insp.addEventListener('input', (e) => {
  if (e.target.dataset.of !== 'gain_db' || S.selSec == null) return;
  if (!S.pendingUndo) S.pendingUndo = snapshot();
  ownSections()[S.selSec].gain_db = +e.target.value;
  insp.querySelector('[data-role="sec-gain"]').textContent = e.target.value + ' dB';
});
insp.addEventListener('change', (e) => {
  const of = e.target.dataset.of;
  if (!of || S.selSec == null) return;
  if (of === 'gain_db') { pushUndo(S.pendingUndo); S.pendingUndo = null; changed(); return; }
  pushUndo();
  ownSections()[S.selSec].sound = e.target.value;
  changed();
  if (e.target.value === 'music' && !music.getAttribute('src')) separateVoices();
});
insp.addEventListener('click', (e) => {
  const b = e.target.closest('[data-act]');
  if (!b || !b.dataset.act.startsWith('sec-') || S.selSec == null) return;
  const act = b.dataset.act, x = sections()[S.selSec];
  if (act === 'sec-play') { seek(x.start); play(); }
  else if (act === 'sec-cut') cutSection(now());
  else if (act === 'sec-join') joinSection(S.selSec);
  else if (act === 'sec-mute') muteSection(S.selSec);
  else if (act === 'sec-reset') { pushUndo(); Object.assign(ownSections()[S.selSec], { gain_db: 0, sound: 'auto' }); changed(); }
});

// ---------- 🎚 Voice levels: one place to keep every voice clip equally loud ----------
const STRENGTH_LABELS = [['off', 'Off — keep each line as it was spoken'], ['light', 'Light'], ['normal', 'Normal'],
  ['strong', 'Strong — every word about as loud as the next']];
function clipLoudness(key) {
  // How loud the clip plays (dB, as heard), with levelling and its own Volume; null if silent, undefined if not loaded.
  const level = rawLevel(key);
  if (level == null) return level;
  return level + autoGainDb(key) + (S.clips[key].gain_db || 0);
}
function renderLevelsPanel() {
  const on = levelOn(), target = levelTarget(), list = keys().filter((k) => lineOf(k) && lineOf(k).url && !S.clips[k].muted)
    .sort((a, b) => S.clips[a].start - S.clips[b].start);
  const rows = list.map((k) => ({ k, l: lineOf(k), db: clipLoudness(k), own: S.clips[k].gain_db || 0 }));
  const known = rows.filter((r) => r.db != null);
  const lo = known.length ? Math.min(...known.map((r) => r.db)) : 0, hi = known.length ? Math.max(...known.map((r) => r.db)) : 0;
  const mid = on ? target : (known.length ? known.map((r) => r.db).sort((a, b) => a - b)[Math.floor(known.length / 2)] : 0);
  const tweaked = rows.filter((r) => r.own).length;
  const off = rows.filter((r) => r.db != null && Math.abs(r.db - mid) > 1.5);
  const bar = (db) => {
    const x = clamp(50 + (db - mid) * 5, 2, 98);
    return '<span class="vxe-lv-bar"><span class="vxe-lv-mid"></span><span class="vxe-lv-dot" style="left:' + x + '%"></span></span>';
  };
  insp.innerHTML = '<h3>🎚 Voice levels</h3>' +
    '<div class="vxe-muted-note">Keeps every voice clip just as loud as the others — measured the way loudness is heard, ' +
    'so a deep voice and a bright shout come out even. It follows your edits (trim, cut, speed, regenerate) and applies to the preview and the export.</div>' +
    field('', '<label class="vxe-check"><input type="checkbox" data-lv="on"' + (on ? ' checked' : '') + '> Level voices (same loudness for every clip)</label>') +
    field('Loudness', '<input type="range" min="' + LEVEL_TARGET_MIN + '" max="' + LEVEL_TARGET_MAX + '" step="1" data-lv="target" value="' + target + '"' + (on ? '' : ' disabled') +
      '> <span data-role="lv-target">' + target + ' dB</span> <span class="vxe-muted-note">how loud every voice is</span>') +
    field('Inside a line', '<select data-lv="strength"' + (on ? '' : ' disabled') + '>' + STRENGTH_LABELS.map(([v, label]) =>
      '<option value="' + v + '"' + (levelStrength() === v ? ' selected' : '') + '>' + label + '</option>').join('') + '</select>' +
      ' <span class="vxe-muted-note">evens out loud and quiet words within each line</span>') +
    field('Now', known.length ? (hi - lo <= 1.5 ? '<span style="color:#6fdc8c">✔ all ' + known.length + ' clips within ' + (hi - lo).toFixed(1) + ' dB of each other</span>'
      : '<span style="color:var(--vxe-warn)">' + known.length + ' clips span ' + (hi - lo).toFixed(1) + ' dB · ' + off.length + ' stand out</span>')
      : '<span class="vxe-muted-note">loading the voices…</span>') +
    field('Clip volumes', tweaked ? tweaked + ' clip(s) have their own Volume change on top <button class="vxe-btn" data-act="lv-reset">↺ Reset all to 0 dB</button>'
      : '<span class="vxe-muted-note">no clip has its own Volume change</span>') +
    '<div class="vxe-lv-list">' + rows.map((r) => '<div class="vxe-lv-row' + (r.db != null && Math.abs(r.db - mid) > 1.5 ? ' vxe-lv-out' : '') +
      '" data-lvkey="' + esc(r.k) + '" title="Click to select this clip">' +
      '<span class="vxe-dot" style="background:' + colorOf(r.l.speaker) + '"></span><span class="vxe-lv-name">#' + r.l.index + ' ' + esc(r.l.speaker) + '</span>' +
      (r.db == null ? '<span class="vxe-muted-note">' + (r.db === null ? 'silent' : '…') + '</span>'
        : bar(r.db) + '<span class="vxe-lv-db">' + (r.db - mid >= 0 ? '+' : '') + (r.db - mid).toFixed(1) + ' dB</span>') +
      (r.own ? '<span class="vxe-badge" title="This clip\'s own Volume">' + (r.own > 0 ? '+' : '') + r.own + ' dB</span>' : '') + '</div>').join('') + '</div>' +
    '<div class="vxe-muted-note">Each row shows how much louder (+) or quieter (−) a clip plays than ' + (on ? 'the chosen loudness' : 'the middle clip') +
    '. Click a row to select the clip and change its own Volume.</div>' +
    '<div class="vxe-actions"><button class="vxe-btn" data-act="close-panel">Close</button></div>';
}
function setLevel(change) {
  Object.assign(S.mix, change);
  showLevelButton();
  changed(true);
  if (S.panel === 'levels') renderLevelsPanel();
}
function resetClipVolumes() {
  const list = keys().filter((k) => S.clips[k].gain_db);
  if (!list.length) return;
  pushUndo();
  list.forEach((k) => { S.clips[k].gain_db = 0; });
  changed();
  setStatus('🎚 ' + list.length + ' clip(s) set back to 0 dB — every voice now plays at the same loudness.');
}
insp.addEventListener('input', (e) => {
  if (e.target.dataset.lv !== 'target') return;
  const v = insp.querySelector('[data-role="lv-target"]');
  if (v) v.textContent = e.target.value + ' dB';
});
insp.addEventListener('change', (e) => {
  const lv = e.target.dataset.lv;
  if (lv === 'on') setLevel({ level: e.target.checked });
  else if (lv === 'target') setLevel({ level_target: +e.target.value });
  else if (lv === 'strength') setLevel({ level_strength: e.target.value });
});
insp.addEventListener('click', (e) => {
  const row = e.target.closest('[data-lvkey]');
  if (!row) return;
  const k = row.dataset.lvkey, c = S.clips[k];
  if (!c) return;
  S.panel = null; S.selCue = null; S.selSec = null; S.sel = new Set([k]);
  seek(c.start);
  scroll.scrollLeft = Math.max(0, c.start * S.pps + HEAD - scroll.clientWidth / 3);
  render(); renderInspector();
});

// ---------- full screen ----------
const isFull = () => document.fullscreenElement === root || root.classList.contains('vxe-max');
function toggleFullscreen() {
  if (document.fullscreenElement === root) { document.exitFullscreen().catch(() => {}); return; }
  if (root.classList.contains('vxe-max')) { root.classList.remove('vxe-max'); afterResize(); return; }
  // Browsers that refuse real full screen (e.g. inside a frame) get the editor over the whole window instead.
  const fallback = () => { root.classList.add('vxe-max'); afterResize(); };
  if (root.requestFullscreen) root.requestFullscreen().catch(fallback); else fallback();
}
function afterResize() {
  root.querySelector('[data-act="fullscreen"]').textContent = isFull() ? '✕ Exit full screen' : '⛶ Full screen';
  requestAnimationFrame(() => { renderVisible(); keys().forEach(drawClipWave); });
  root.focus({ preventScroll: true });
}
document.addEventListener('fullscreenchange', afterResize);
try {
  const h = localStorage.getItem('vxe-top-h');
  if (h) root.style.setProperty('--vxe-top-h', h);
} catch (e) { /* storage unavailable */ }
q('split').addEventListener('pointerdown', (e) => {
  const bar = e.currentTarget, top = root.querySelector('.vxe-top');
  const y0 = e.clientY, h0 = top.getBoundingClientRect().height;
  bar.setPointerCapture(e.pointerId);
  const onMove = (ev) => {
    const h = clamp(h0 + ev.clientY - y0, 120, window.innerHeight - 220);
    root.style.setProperty('--vxe-top-h', h + 'px');
  };
  const onUp = () => {
    bar.removeEventListener('pointermove', onMove);
    bar.removeEventListener('pointerup', onUp);
    try { localStorage.setItem('vxe-top-h', root.style.getPropertyValue('--vxe-top-h')); } catch (err) { /* ignore */ }
    afterResize();
  };
  bar.addEventListener('pointermove', onMove);
  bar.addEventListener('pointerup', onUp);
});

// ---------- toolbar and keys ----------
function stepClip(dir) {
  const t = now();
  const list = byStart(keys());
  const next = dir > 0 ? list.find((k) => S.clips[k].start > t + 0.01) : list.slice().reverse().find((k) => S.clips[k].start < t - 0.05);
  if (!next) return;
  S.sel = new Set([next]);
  renderSelection();
  seek(S.clips[next].start);
}
root.querySelector('.vxe-toolbar').addEventListener('click', (e) => {
  const b = e.target.closest('[data-act]');
  if (!b) return;
  const act = b.dataset.act;
  if (runAction(act)) return;
  if (act === 'play') toggle();
  else if (act === 'start') seek(0);
  else if (act === 'prev') stepClip(-1);
  else if (act === 'next') stepClip(1);
  else if (act === 'undo') undo();
  else if (act === 'redo') redo();
  else if (act === 'fit') setZoom((scroll.clientWidth - HEAD - 20) / S.dur, 0, scroll.getBoundingClientRect().left + HEAD);
  else if (act === 'align-all') alignClips(S.lines.map((l) => l.index));
  else if (act === 'reset-all') toSrt(S.lines.map((l) => l.index));
  else if (act === 'export') doExport();
  else if (act === 'change-video') q('file').click();
  else if (act === 'reload') reload();
  else if (act === 'fullscreen') toggleFullscreen();
  else if (act === 'subs-area') {
    S.editSubs = !S.editSubs;
    if (S.editSubs && (!S.mix.subs || S.mix.subs.mode === 'off')) setStatus('The subtitles are kept — choose <b>Removed</b> or <b>Blurred</b> to hide this area.');
    else if (S.editSubs) setStatus('✏️ Drag the yellow box over the subtitles in the video; drag a corner to resize. Click ✏️ Area again when done.');
    else setStatus('');
  }
  else if (act === 'subs-find') findSubtitles();
  else if (act === 'orig-toggle') toggleOriginal();
  else if (act === 'captions') toggleSubs();
  else if (act === 'level') {
    S.mix.level = !levelOn();
    showLevelButton();
    changed();
    setStatus(levelOn() ? '🎚 Every voice clip is kept at the same loudness (in the preview and the export) — a clip\'s Volume is a change on top.'
      : '🎚 Voice levelling off — each clip plays at its own loudness.');
  }
});
q('zoom').addEventListener('input', (e) => setZoom(+e.target.value));
async function separateVoices() {
  // Splits the original soundtrack into voices and music & effects, so "Voices removed" can be heard right away.
  const v = S.data && S.data.video;
  if (!v || music.getAttribute('src')) return;
  pause();
  try {
    busy('Removing the voices from the original sound (keeping the music & sound effects)… this takes a few minutes the first time.');
    await save();
    applyPayload(await call('editor_detect', { sensitivity: v.sensitivity != null ? v.sensitivity : 0.5, isolate: true }));
    setStatus('🎵 The original voices are removed — you now hear only its music & sound effects under the new voices (also in the export).');
  } catch (e) { showError(e); } finally { busy(''); }
}
q('mode').addEventListener('change', (e) => {
  S.mix.original = e.target.value;
  syncMusic(now(), S.playing);
  changed(true);
  if (S.mix.original === 'music' && !music.getAttribute('src')) separateVoices();
});
q('orig-gain').addEventListener('change', (e) => { S.mix.original_gain_db = +e.target.value; changed(true); });
q('voc-gain').addEventListener('change', (e) => { S.mix.vocals_gain_db = +e.target.value; changed(true); });

document.addEventListener('keydown', (e) => {
  // The editor tab is not shown (offsetParent can't tell: it is empty in full screen too).
  if (!root.isConnected || !root.getClientRects().length) return;
  const t = e.target;
  const typing = t && (t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || (t.tagName === 'INPUT' && !['range', 'checkbox', 'file'].includes(t.type)));
  if (typing || !(root.contains(t) || t === document.body)) return;
  const sel = generated(selKeys());
  const mod = e.metaKey || e.ctrlKey;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  let handled = true;
  if (e.key === ' ') toggle();
  else if (mod && key === 'z') (e.shiftKey ? redo() : undo());
  else if (mod && key === 'y') redo();
  else if (mod && key === 'a') { S.sel = new Set(keys()); renderSelection(); }
  else if (mod) handled = false;
  else if (e.altKey && (key === 'ArrowUp' || key === 'ArrowDown') && S.selCue) moveCueRow(S.selCue, key === 'ArrowUp' ? -1 : 1);
  else if (e.altKey && (key === 'ArrowUp' || key === 'ArrowDown') && sel.length) moveRows(sel, key === 'ArrowUp' ? -1 : 1);
  else if ((key === 'ArrowLeft' || key === 'ArrowRight') && sel.length) nudge(sel, (key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 0.5 : 0.05));
  else if (key === 'ArrowLeft' || key === 'ArrowRight') seek(now() + (key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 5 : 1));
  else if (key === 'ArrowUp') stepClip(-1);
  else if (key === 'ArrowDown') stepClip(1);
  else if ((key === 'c' || key === 'b') && S.selSec != null) cutSection(now());
  else if (key === 'c' || key === 'b') cutAt(now(), sel.length ? sel : null);
  else if (key === 'j' && S.selSec != null) joinSection(S.selSec);
  else if (key === 'j') joinPieces(selKeys());
  else if ((key === 'm' || key === 'Delete' || key === 'Backspace') && S.selSec != null) muteSection(S.selSec);
  else if ((key === '[' || key === ']') && sel.length) stepSpeed(sel, key === '[' ? -0.05 : 0.05);
  else if ((key === 'Delete' || key === 'Backspace') && S.selCue) deleteCue(S.selCue);
  else if ((key === 'Delete' || key === 'Backspace') && sel.length) deletePieces(sel);
  else if (key === 'n' && S.data && S.data.video) addCue(now());
  else if (key === 'm' && sel.length) toggleMute(sel);
  else if (key === 's' && sel.length) alignClips(selLines());
  else if (key === 'r' && S.sel.size) regenerate(selLines());
  else if (key === 'f') toggleFullscreen();
  else if (key === 'o' && S.data && S.data.video) toggleOriginal();
  else if (key === 'Escape' && root.classList.contains('vxe-max')) toggleFullscreen();
  else if (key === 'Home') seek(0);
  else if (key === 'End') seek(S.dur);
  else if (key === 'Escape') { S.sel.clear(); S.selCue = null; S.selSec = null; renderSelection(); }
  else handled = false;
  if (handled) e.preventDefault();
});

// Opened from the SRT → Speech tab (or the tab was selected): load the project again.
watch('value', () => reload());
reload();
