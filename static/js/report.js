/* report.js — renders the JSON incident report + interactive timeline */
(function () {
  const jobId = document.getElementById('jobId').value;
  const player = document.getElementById('player');

  const TYPE_LABELS = {
    looking_sideways: 'Sideways gaze',
    looking_down: 'Downward gaze',
    object: 'Prohibited object',
    movement: 'Excessive movement',
    proximity: 'Proximity',
    left_zone: 'Left seating zone',
  };

  fetch('/api/report/' + jobId)
    .then((r) => r.json())
    .then(render)
    .catch(() => {});

  function seek(t) {
    if (!player) return;
    player.currentTime = t;
    if (player.play) player.play().catch(() => {});
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function render(rep) {
    const v = rep.video || {};
    const dur = v.duration_sec || 0;

    document.getElementById('sStudents').textContent = rep.total_students ?? '—';
    document.getElementById('sAlerts').textContent = rep.total_alerts ?? '—';
    document.getElementById('sDuration').textContent = fmtTime(dur);
    document.getElementById('sProc').textContent = rep.processing_time_sec ? rep.processing_time_sec + 's' : '—';
    document.getElementById('tlEnd').textContent = fmtTime(dur);
    document.getElementById('mRes').textContent = (v.width && v.height) ? v.width + '×' + v.height : '—';
    document.getElementById('mFps').textContent = v.fps ?? '—';
    document.getElementById('mFrames').textContent = v.total_frames ?? '—';
    document.getElementById('mDevice').textContent = (rep.device || '—').toUpperCase();
    const engines = rep.engines || {};
    const mObjects = document.getElementById('mObjects');
    const mSensitivity = document.getElementById('mSensitivity');
    if (mObjects) mObjects.textContent = engines.object_detection || '—';
    if (mSensitivity) mSensitivity.textContent = engines.sensitivity || '—';

    // --- interactive timeline ---
    const tl = document.getElementById('timeline');
    const alerts = rep.alerts || [];
    tl.innerHTML = '';
    if (!alerts.length) {
      tl.innerHTML = '<div class="empty" style="padding:20px;">No incidents flagged.</div>';
    } else {
      alerts.forEach((a) => {
        const x = dur > 0 ? Math.min(99.4, (a.timestamp / dur) * 100) : 0;
        const tick = document.createElement('div');
        tick.className = 'tick';
        tick.style.left = x + '%';
        tick.style.background = SEV_COLORS[a.severity] || '#9aa6c4';
        tick.title = (a.time_str || fmtTime(a.timestamp)) + ' · Student ' + a.student_id + ' · ' + a.message;
        tick.onclick = () => seek(a.timestamp);
        tl.appendChild(tick);
      });
    }

    // --- severity distribution (donut) ---
    renderDonut(rep.severity_distribution || {});

    // --- incident type distribution ---
    const td = rep.alert_type_distribution || {};
    const tdTotal = Object.values(td).reduce((a, b) => a + b, 0) || 1;
    document.getElementById('typeDist').innerHTML =
      Object.entries(td).map(([k, val]) =>
        bar(esc(TYPE_LABELS[k] || k), val, tdTotal, '#5b8cff')
      ).join('') || emptyRow();

    // --- per-student breakdown ---
    const stBody = document.querySelector('#studentTable tbody');
    stBody.innerHTML = (rep.students || []).map((s) => {
      const beh = Object.entries(s.events || {})
        .filter(([k]) => k !== 'alerts')
        .map(([k, val]) => labelEvent(k) + ' ×' + val).join(', ') || '—';
      return `<tr>
        <td><strong>Student ${s.student_id}</strong></td>
        <td><span class="badge sev sev-${s.final_severity}">${s.final_severity}</span></td>
        <td class="num mono" style="color:${SEV_COLORS[s.final_severity]}">${(s.max_risk || 0).toFixed(2)}</td>
        <td class="num">${s.total_alerts || 0}</td>
        <td class="student-meta">${esc(beh)}</td>
      </tr>`;
    }).join('') || '<tr><td colspan="5" class="empty">No students detected.</td></tr>';

    // --- incident log ---
    const alBody = document.querySelector('#alertTable tbody');
    alBody.innerHTML = alerts.map((a) => `
      <tr style="cursor:pointer" data-t="${a.timestamp}">
        <td class="mono">${a.time_str || fmtTime(a.timestamp)}</td>
        <td class="num mono">${a.frame}</td>
        <td>Student ${a.student_id}</td>
        <td>${esc(a.message)}</td>
        <td><span class="badge sev sev-${a.severity}">${a.severity}</span></td>
      </tr>`).join('') || '<tr><td colspan="5" class="empty">No incidents recorded.</td></tr>';
    alBody.querySelectorAll('tr[data-t]').forEach((tr) =>
      tr.onclick = () => seek(parseFloat(tr.dataset.t)));
  }

  function renderDonut(dist) {
    const el = document.getElementById('sevDonut');
    const leg = document.getElementById('sevDonutLegend');
    if (!el) return;
    const entries = Object.entries(dist);
    const total = entries.reduce((a, [, v]) => a + v, 0);
    const r = 52, c = 2 * Math.PI * r, cx = 65, cy = 65;
    let offset = 0, segs = '';
    if (total === 0) {
      segs = `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="16"/>`;
    } else {
      entries.forEach(([k, v]) => {
        if (v <= 0) return;
        const len = c * (v / total);
        segs += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${SEV_COLORS[k] || '#888'}"
          stroke-width="16" stroke-dasharray="${len.toFixed(2)} ${(c - len).toFixed(2)}"
          stroke-dashoffset="${(-offset).toFixed(2)}" transform="rotate(-90 ${cx} ${cy})"/>`;
        offset += len;
      });
    }
    el.innerHTML = `<svg width="130" height="130" viewBox="0 0 130 130">${segs}</svg>
      <div class="center"><b>${total}</b><span>incidents</span></div>`;
    leg.innerHTML = entries.map(([k, v]) =>
      `<div class="row"><i style="background:${SEV_COLORS[k] || '#888'}"></i>${k}<span class="n">${v}</span></div>`
    ).join('');
  }

  function bar(label, val, total, color) {
    const pct = Math.round(val / total * 100);
    return `<div class="row"><span>${label}</span>
      <div class="bar"><span style="width:${pct}%;background:${color}"></span></div>
      <span class="num mono">${val}</span></div>`;
  }
  function emptyRow() { return '<div class="empty" style="padding:14px;">No data</div>'; }
  function labelEvent(k) {
    const m = {
      looking_sideways: 'Sideways', looking_down: 'Downward',
      excessive_movement: 'Movement', proximity: 'Proximity',
      left_zone: 'Left zone',
    };
    if (k.startsWith('object:')) return 'Object: ' + k.slice(7);
    return m[k] || k;
  }
})();
