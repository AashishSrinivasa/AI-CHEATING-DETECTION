/* history.js — live progress updates + delete handling */
(function () {
  const table = document.getElementById('historyTable');

  function bindDelete() {
    document.querySelectorAll('.del-btn').forEach((b) => {
      b.onclick = async () => {
        if (!confirm('Delete this session and all of its files?')) return;
        try {
          await fetch('/api/job/' + b.dataset.job, { method: 'DELETE' });
          const tr = b.closest('tr');
          if (tr) tr.remove();
          if (!document.querySelector('tbody tr')) location.reload();
        } catch (_) {}
      };
    });
  }
  bindDelete();

  async function refresh() {
    const rows = [...document.querySelectorAll('tr[data-status]')]
      .filter((r) => ['processing', 'queued'].includes(r.dataset.status));
    if (!rows.length) return;
    for (const tr of rows) {
      try {
        const r = await fetch('/api/status/' + tr.dataset.job);
        const d = await r.json();
        if (d.status === 'completed' || d.status === 'failed') {
          location.reload();
          return;
        }
        tr.dataset.status = d.status;
        const cell = tr.querySelector('.status-cell');
        if (cell) {
          cell.innerHTML =
            '<span class="badge status-pill status-' + d.status + '">' + d.status + '</span>' +
            '<div class="progress" style="margin:8px 0 0;height:6px;"><span style="width:' +
            (d.progress || 0) + '%"></span></div>';
        }
      } catch (_) {}
    }
  }

  if (table) setInterval(refresh, 1500);
})();
