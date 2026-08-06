/* upload.js — drag & drop upload with progress + job tracking */
(function () {
  const $ = (id) => document.getElementById(id);
  const dz = $('dropzone'), fi = $('fileInput'), info = $('fileInfo');
  const nameEl = $('fileName'), sizeEl = $('fileSize');
  const submitBtn = $('submitBtn'), resetBtn = $('resetBtn'), form = $('uploadForm');
  const pWrap = $('progressWrap'), pBar = $('progressBar'), pMsg = $('progressMsg');
  const flash = $('flash'), sensitivityGroup = $('sensitivityGroup');
  const ALLOWED = ['mp4', 'avi', 'mov', 'mkv'];
  const MAX_MB = parseInt(dz.dataset.max || '500', 10);
  let file = null, jobId = null, poll = null;

  function showErr(m) { flash.textContent = m; flash.classList.remove('hidden'); }
  function hideErr() { flash.classList.add('hidden'); }

  function pickFile(f) {
    if (!f) return;
    const ext = f.name.split('.').pop().toLowerCase();
    if (!ALLOWED.includes(ext)) { showErr('Unsupported format. Use ' + ALLOWED.join(', ').toUpperCase() + '.'); return; }
    if (f.size > MAX_MB * 1024 * 1024) { showErr('File exceeds the ' + MAX_MB + ' MB limit.'); return; }
    file = f;
    nameEl.textContent = f.name;
    sizeEl.textContent = fmtBytes(f.size);
    info.classList.remove('hidden');
    submitBtn.disabled = false;
    hideErr();
  }

  dz.addEventListener('click', () => fi.click());
  fi.addEventListener('change', (e) => pickFile(e.target.files[0]));
  ['dragover', 'dragenter'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', (e) => { if (e.dataTransfer.files[0]) pickFile(e.dataTransfer.files[0]); });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!file) return;
    submitBtn.disabled = true;
    resetBtn.classList.add('hidden');
    hideErr();
    pWrap.classList.remove('hidden');
    pBar.style.width = '0%';
    pMsg.textContent = 'Uploading…';

    const fd = new FormData();
    fd.append('video', file);
    const selectedSensitivity = document.querySelector('input[name="sensitivity"]:checked');
    fd.append('sensitivity', selectedSensitivity ? selectedSensitivity.value : 'strict');
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload');
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable) {
        const p = Math.round(ev.loaded / ev.total * 100);
        pBar.style.width = p + '%';
        pMsg.textContent = 'Uploading… ' + p + '%';
      }
    };
    xhr.onload = () => {
      let d = {};
      try { d = JSON.parse(xhr.responseText); } catch (_) {}
      if (xhr.status >= 200 && xhr.status < 300 && d.job_id) {
        jobId = d.job_id;
        pBar.style.width = '100%';
        pMsg.textContent = 'Uploaded. Starting analysis…';
        trackJob();
      } else {
        showErr(d.error || ('Upload failed (HTTP ' + xhr.status + ').'));
        submitBtn.disabled = false;
        resetBtn.classList.remove('hidden');
      }
    };
    xhr.onerror = () => { showErr('Network error during upload.'); submitBtn.disabled = false; };
    xhr.send(fd);
  });

  function trackJob() {
    poll = setInterval(async () => {
      try {
        const r = await fetch('/api/status/' + jobId);
        const d = await r.json();
        const p = d.progress || 0;
        pBar.style.width = p + '%';
        pMsg.textContent = d.message || ('Processing… ' + p + '%');
        if (d.status === 'completed') {
          clearInterval(poll);
          pMsg.innerHTML = 'Analysis complete. <a style="color:var(--brand)" href="/report/' + jobId + '">Open report</a>';
          resetBtn.classList.remove('hidden');
        } else if (d.status === 'failed') {
          clearInterval(poll);
          showErr(d.message || 'Processing failed.');
          resetBtn.classList.remove('hidden');
        }
      } catch (_) { /* transient */ }
    }, 1200);
  }

  resetBtn.addEventListener('click', () => {
    if (poll) clearInterval(poll);
    file = null; jobId = null; fi.value = '';
    info.classList.add('hidden');
    pWrap.classList.add('hidden');
    submitBtn.disabled = true;
    resetBtn.classList.add('hidden');
    hideErr();
  });

  if (sensitivityGroup) {
    sensitivityGroup.querySelectorAll('label').forEach((label) => {
      label.addEventListener('click', () => {
        sensitivityGroup.querySelectorAll('label').forEach((l) => l.classList.remove('active'));
        label.classList.add('active');
      });
    });
  }
})();
