/* main.js — shared helpers: formatting, toasts, engine chips, counters */

function fmtTime(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function fmtBytes(b) {
  if (!b) return '—';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return b.toFixed(b < 10 && i > 0 ? 1 : 0) + ' ' + u[i];
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : String(s);
  return d.innerHTML;
}

const SEV_COLORS = { NORMAL: '#22c55e', LOW: '#eab308', MEDIUM: '#f97316', HIGH: '#ef4444' };

/* ---- toast notifications ---- */
function toast(msg, type = '', timeout = 4200) {
  const w = document.getElementById('toastWrap');
  if (!w) return;
  const t = document.createElement('div');
  t.className = 'toast ' + type;
  t.innerHTML = '<span>' + esc(msg) + '</span>';
  w.appendChild(t);
  setTimeout(() => {
    t.style.transition = 'opacity .3s, transform .3s';
    t.style.opacity = '0';
    t.style.transform = 'translateX(14px)';
    setTimeout(() => t.remove(), 300);
  }, timeout);
}

/* ---- engine status chips in the navbar ---- */
function shortHeadPose(n) {
  if (/FaceMesh|FaceLandmarker|MediaPipe/.test(n)) return 'MediaPipe';
  if (/Haar/.test(n)) return 'Haar';
  return n;
}
function chipClass(name) {
  if (/unavailable/i.test(name)) return 'off';
  if (/fallback/i.test(name)) return 'fallback';
  return 'ok';
}
async function loadEngineChips() {
  const el = document.getElementById('engineChips');
  try {
    const r = await fetch('/api/system');
    const d = await r.json();
    const e = d.engines || {};
    if (el) {
      const chips = [];
      if (e.object_detection) chips.push([/YOLO/.test(e.object_detection) ? 'YOLOv8' : 'HOG', chipClass(e.object_detection)]);
      if (e.head_pose) chips.push([shortHeadPose(e.head_pose), chipClass(e.head_pose)]);
      if (e.tracking) chips.push([/DeepSORT/.test(e.tracking) ? 'DeepSORT' : 'IoU', chipClass(e.tracking)]);
      const deviceLabel = (d.device || 'cpu').toUpperCase();
      chips.push([deviceLabel, deviceLabel !== 'CPU' ? 'ok' : '']);
      el.innerHTML = chips.map(([t, c]) => `<span class="chip ${c}"><i></i>${esc(t)}</span>`).join('');
      el.title = `Object: ${e.object_detection} · Head pose: ${e.head_pose} · Tracking: ${e.tracking} · Device: ${(d.device || '').toUpperCase()}`;
    }
    renderReadinessPanel(d);
  } catch (_) { /* ignore */ }
}

function renderReadinessPanel(d) {
  const state = document.getElementById('systemState');
  const engines = document.getElementById('systemEngines');
  const warnings = document.getElementById('systemWarnings');
  if (!state || !engines) return;

  if (d.full_detection_ready) {
    state.textContent = 'Full';
    state.className = 'badge status-pill status-completed';
  } else if (d.ready) {
    state.textContent = 'Degraded';
    state.className = 'badge status-pill status-processing';
  } else {
    state.textContent = 'Offline';
    state.className = 'badge status-pill status-failed';
  }

  const e = d.engines || {};
  engines.innerHTML = [
    ['Object engine', e.object_detection || 'unavailable'],
    ['Item detection', e.prohibited_item_detection || 'unavailable'],
    ['Head pose', e.head_pose || 'unavailable'],
    ['Tracking', e.tracking || 'unavailable'],
    ['Device', (d.device || 'cpu').toUpperCase()],
  ].map(([k, v]) => `<div class="kv"><span class="k">${esc(k)}</span><span class="mono">${esc(v)}</span></div>`).join('');

  if (warnings) {
    const list = d.warnings || [];
    if (list.length) {
      warnings.innerHTML = list.map(esc).join('<br>');
      warnings.classList.remove('hidden');
    } else {
      warnings.classList.add('hidden');
    }
  }
}

/* ---- animated number counters ([data-count]) ---- */
function animateCounters() {
  document.querySelectorAll('[data-count]').forEach((el) => {
    const target = parseFloat(el.dataset.count) || 0;
    if (target <= 0) { el.textContent = '0'; return; }
    const dur = 900, t0 = performance.now();
    function step(now) {
      const p = Math.min(1, (now - t0) / dur);
      el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(step);
      else el.textContent = target;
    }
    requestAnimationFrame(step);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  loadEngineChips();
  animateCounters();
});
