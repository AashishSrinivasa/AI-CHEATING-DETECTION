/* live.js — live camera monitoring controller (circular gauges + toasts) */
(function () {
  const $ = (id) => document.getElementById(id);
  const startBtn = $('startBtn'), stopBtn = $('stopBtn');
  const feed = $('feed'), placeholder = $('placeholder'), recBadge = $('recBadge');
  const dot = $('statusDot'), statusText = $('statusText');
  const fpsEl = $('fps'), countEl = $('count'), alertCountEl = $('alertCount');
  const studentsEl = $('students'), alertsEl = $('alerts');
  const sourceEl = $('source'), flash = $('flash'), sensitivityGroup = $('sensitivityGroup');
  let poll = null, lastAlertCount = 0, enginesShown = false;

  function showErr(m) { flash.textContent = m; flash.classList.remove('hidden'); }
  function hideErr() { flash.classList.add('hidden'); }

  function sensitivity() {
    const checked = document.querySelector('input[name="sensitivity"]:checked');
    return checked ? checked.value : 'strict';
  }

  function gauge(risk, color) {
    const r = 24, c = 2 * Math.PI * r, off = c * (1 - Math.max(0, Math.min(1, risk)));
    return `<div class="gauge"><svg width="58" height="58" viewBox="0 0 58 58">
      <circle cx="29" cy="29" r="${r}" fill="none" stroke="rgba(255,255,255,.1)" stroke-width="6"/>
      <circle cx="29" cy="29" r="${r}" fill="none" stroke="${color}" stroke-width="6" stroke-linecap="round"
        stroke-dasharray="${c.toFixed(1)}" stroke-dashoffset="${off.toFixed(1)}"/>
      </svg><div class="val" style="color:${color}">${risk.toFixed(2)}</div></div>`;
  }

  async function start() {
    hideErr();
    startBtn.disabled = true;
    try {
      const r = await fetch('/api/live/start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: sourceEl.value, sensitivity: sensitivity() }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || 'Failed to start camera.');
      feed.src = '/video_feed?' + Date.now();
      feed.classList.remove('hidden');
      placeholder.classList.add('hidden');
      recBadge.classList.add('on');
      dot.classList.add('on');
      statusText.textContent = 'Monitoring';
      stopBtn.disabled = false;
      enginesShown = false;
      toast('Monitoring started', 'ok');
      poll = setInterval(refresh, 700);
    } catch (e) {
      showErr(e.message);
      toast(e.message, 'err', 6000);
      startBtn.disabled = false;
    }
  }

  async function stop(silent) {
    stopBtn.disabled = true;
    try { await fetch('/api/live/stop', { method: 'POST' }); } catch (_) {}
    if (poll) { clearInterval(poll); poll = null; }
    feed.classList.add('hidden'); feed.src = '';
    placeholder.classList.remove('hidden');
    recBadge.classList.remove('on');
    dot.classList.remove('on');
    statusText.textContent = 'Idle';
    startBtn.disabled = false;
    if (!silent) toast('Monitoring stopped');
  }

  async function refresh() {
    try {
      const r = await fetch('/api/live/stats');
      const d = await r.json();
      if (d.error) { showErr(d.error); toast(d.error, 'err', 6000); await stop(true); return; }
      if (!enginesShown && d.engines && d.engines.head_pose) {
        enginesShown = true;
        toast(`Engines · ${d.engines.object_detection} · ${shortHeadPose(d.engines.head_pose)} · ` +
              `${/DeepSORT/.test(d.engines.tracking) ? 'DeepSORT' : 'IoU'} · ${d.engines.sensitivity || 'Strict Exam'} ` +
              `(${(d.engines.device || '').toUpperCase()})`, 'ok', 5000);
      }
      fpsEl.textContent = d.fps || 0;
      countEl.textContent = (d.students || []).length;
      alertCountEl.textContent = d.total_alerts || 0;
      if ((d.total_alerts || 0) > lastAlertCount && lastAlertCount > 0) {
        const a = (d.alerts || [])[0];
        if (a) toast(`S${a.student_id}: ${a.message}`, 'warn', 4000);
      }
      lastAlertCount = d.total_alerts || 0;
      renderStudents(d.students || []);
      renderAlerts(d.alerts || []);
    } catch (_) { /* transient */ }
  }

  function renderStudents(list) {
    if (!list.length) {
      studentsEl.innerHTML = '<div class="empty" style="padding:20px;">No students detected yet.</div>';
      return;
    }
    list.sort((a, b) => b.risk - a.risk);
    studentsEl.innerHTML = list.map((s) => `
      <div class="student-row">
        ${gauge(s.risk, s.hex)}
        <div>
          <div><strong>Student ${s.id}</strong> <span class="badge sev sev-${s.severity}">${s.severity}</span></div>
          <div class="student-meta">
            ${studentSignals(s)}
          </div>
        </div>
        <div style="min-width:96px;">
          <div class="riskbar"><span style="width:${Math.round(s.risk * 100)}%;background:${s.hex}"></span></div>
        </div>
      </div>`).join('');
  }

  function objectText(s) {
    const conf = s.object_conf == null ? '' : ' ' + Math.round(s.object_conf * 100) + '%';
    const source = s.object_source === 'yolo_world' ? 'World' : 'YOLO';
    return `Malpractice ${esc(s.object)}${conf} (${source})`;
  }

  function studentSignals(s) {
    const items = [];
    if (s.object) items.push(objectText(s));
    if (s.pose && Math.abs(s.pose.yaw_adj || 0) >= 20) items.push('Sideways look');
    if (s.pose && (s.pose.pitch_adj || 0) >= 18) items.push('Downward look');
    if ((s.velocity || 0) > 12) items.push('Movement');
    return items.join(' · ') || 'No active signal';
  }

  function renderAlerts(list) {
    if (!list.length) {
      alertsEl.innerHTML = '<div class="empty" style="padding:20px;">Alerts will appear here.</div>';
      return;
    }
    alertsEl.innerHTML = list.map((a) => `
      <div class="alert-item sev-${a.severity}">
        <span class="t">${a.time_str || fmtTime(a.timestamp)}</span>
        <span><strong>S${a.student_id}</strong> · ${esc(a.message)}</span>
      </div>`).join('');
  }

  startBtn.addEventListener('click', start);
  stopBtn.addEventListener('click', () => stop(false));
  if (sensitivityGroup) {
    sensitivityGroup.querySelectorAll('label').forEach((label) => {
      label.addEventListener('click', () => {
        sensitivityGroup.querySelectorAll('label').forEach((l) => l.classList.remove('active'));
        label.classList.add('active');
      });
    });
  }
  window.addEventListener('beforeunload', () => {
    if (poll && navigator.sendBeacon) navigator.sendBeacon('/api/live/stop');
  });
})();
