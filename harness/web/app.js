const statusEl = document.querySelector('#system-status');
const emptyEl = document.querySelector('#empty');
const findingsEl = document.querySelector('#findings');
const processingEl = document.querySelector('#processing');
const form = document.querySelector('#upload-form');
const input = document.querySelector('#image-input');
const dropzone = document.querySelector('#dropzone');

const labels = ['crack', 'spalling', 'honeycombing_rock_pocket', 'exposed_rebar', 'rust_staining', 'efflorescence_leaching', 'no_visible_target_defect', 'unknown_review'];
const readable = value => value.replaceAll('_', ' ');

async function loadState() {
  const response = await fetch('/api/state');
  const state = await response.json();
  statusEl.textContent = state.status === 'ready' ? 'Harness ready' : state.status;
  render(state.findings || []);
}

function render(findings) {
  emptyEl.classList.toggle('hidden', findings.length > 0);
  findingsEl.innerHTML = findings.map(findingCard).join('');
  findingsEl.querySelectorAll('[data-review]').forEach(button => button.addEventListener('click', reviewFinding));
  findingsEl.querySelectorAll('[data-relabel]').forEach(form => form.addEventListener('submit', relabelFinding));
}

function findingCard(finding) {
  const prediction = finding.prediction;
  const review = finding.review;
  const geometry = prediction.geometry.box_xywh.map(value => Number(value).toFixed(1)).join(' × ');
  const options = labels.map(label => `<option value="${label}" ${review.reviewer_label === label ? 'selected' : ''}>${readable(label)}</option>`).join('');
  return `<article class="finding">
    <img class="preview" src="/api/findings/${finding.finding_id}/image" alt="Uploaded inspection image">
    <div class="finding-head"><div><h3>${readable(review.reviewer_label || prediction.defect_family)}</h3><p class="meta">${finding.image.source_image_id}<br>Box ${geometry} px</p></div><span class="confidence">${Math.round(prediction.confidence * 100)}% proxy</span></div>
    <div class="review-panel"><span class="review-state">${readable(review.state)}</span>
      <div class="review-controls"><button class="button" data-review="approve" data-id="${finding.finding_id}">Approve</button><button class="button" data-review="reject" data-id="${finding.finding_id}">Reject</button></div>
      <form class="review-form" data-relabel data-id="${finding.finding_id}"><select name="label" aria-label="Relabel finding">${options}</select><input name="notes" type="text" placeholder="Review note" aria-label="Review note"><button class="button" type="submit">Relabel</button></form>
    </div>
  </article>`;
}

async function reviewFinding(event) {
  const response = await fetch(`/api/findings/${event.currentTarget.dataset.id}`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({action: event.currentTarget.dataset.review}) });
  if (!response.ok) return alert((await response.json()).error || 'Review failed');
  await loadState();
}

async function relabelFinding(event) {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  const response = await fetch(`/api/findings/${event.currentTarget.dataset.id}`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({action:'relabel', label:data.get('label'), notes:data.get('notes')}) });
  if (!response.ok) return alert((await response.json()).error || 'Relabel failed');
  await loadState();
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (!input.files[0]) return;
  processingEl.classList.remove('hidden');
  const data = new FormData(form);
  const response = await fetch('/api/upload', {method:'POST', body:data});
  processingEl.classList.add('hidden');
  if (!response.ok) return alert((await response.json()).error || 'Upload failed');
  form.reset();
  await loadState();
});

['dragenter', 'dragover'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.add('drag'); }));
['dragleave', 'drop'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove('drag'); }));
dropzone.addEventListener('drop', event => { input.files = event.dataTransfer.files; });
loadState().catch(() => { statusEl.textContent = 'Harness offline'; });
