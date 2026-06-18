"use strict";

// ══════════════════════════════════════════════════════════════
// FILE HANDLING
// ══════════════════════════════════════════════════════════════
const ALLOWED = /^image\//;
function uid() { return Math.random().toString(36).slice(2,9); }
function fmtSize(b) {
  if (b<1024) return `${b} B`;
  if (b<1048576) return `${(b/1024).toFixed(1)} KB`;
  return `${(b/1048576).toFixed(1)} MB`;
}

function addFiles(list) {
  let added = 0;
  for (const file of list) {
    if (state.gtMode && state.files.length + added >= 1) { toast("GT mode requires one image and one GT depth file", "warning"); break; }
    if (!ALLOWED.test(file.type)) { toast(`Skipped "${file.name}" — image file required`,"warning"); continue; }
    if (file.size > 20*1024*1024) { toast(`Skipped "${file.name}" — exceeds 20 MB`,"warning"); continue; }
    if (settings.warnLargeFiles && file.size > 16*1024*1024 && !confirm(`Process large file "${file.name}" (${fmtSize(file.size)})?`)) continue;
    if (state.files.some(f=>f.file.name===file.name&&f.file.size===file.size)) continue;
    const entry = {id:uid(),file,thumb:null,status:"pending",result:null};
    state.files.push(entry);
    renderFileItem(entry);
    const rd = new FileReader();
    rd.onload = e => {
      entry.thumb = e.target.result;
      const img = $(`#fthumb-${entry.id}`);
      if (img) img.src = e.target.result;
    };
    rd.readAsDataURL(file);
    added++;
  }
  if (added) { syncQueueControls(); toast(`${added} file${added>1?"s":""} added`); }
}

function renderFileItem(entry) {
  const li = document.createElement("li");
  li.className="file-item"; li.id=`fitem-${entry.id}`;
  li.innerHTML = `
    <img class="file-thumb" id="fthumb-${entry.id}" src="" alt="" />
    <div class="file-meta">
      <div class="file-name" title="${esc(entry.file.name)}">${esc(entry.file.name)}</div>
      <div class="file-size">${fmtSize(entry.file.size)}</div>
    </div>
    <span class="file-status pending" id="fst-${entry.id}">Pending</span>
    <button class="file-remove" data-id="${entry.id}" aria-label="Remove ${esc(entry.file.name)}"></button>`;
  el.fileQueue.appendChild(li);
  $(".file-remove",li).addEventListener("click",()=>removeFile(entry.id));
}

function removeFile(id) {
  if (state.files.find(f=>f.id===id)?.status==="running") return;
  state.files = state.files.filter(f=>f.id!==id);
  $(`#fitem-${id}`)?.remove();
  syncQueueControls();
}

function syncQueueControls() {
  const has = state.files.length>0;
  if (el.clearBtn) el.clearBtn.disabled = !has || Boolean(state.abort);
  const ready = engineReady();
  const gtBlocked = state.gtMode && (state.files.length !== 1 || !state.gtFile);
  if (el.runBtn) {
    el.runBtn.disabled = !has || Boolean(state.abort) || state.initializingBackend || !ready || gtBlocked;
    el.runBtn.title = state.initializingBackend ? "Starting depth engine" : (!backendOnline ? `Depth engine unavailable at ${API || DEFAULT_API_BASE_URL}` : (!inferenceReady ? "Inference runtime is not ready · check Diagnostics or /ready" : (gtBlocked ? "GT mode requires one image and one GT depth file" : "")));
  }
}

function setFileSt(id,cls,txt) {
  const s=$(`#fst-${id}`); if (!s) return;
  s.className=`file-status ${cls}`; s.textContent=txt;
}

el.fileInput?.addEventListener("change",()=>{ addFiles(el.fileInput.files); el.fileInput.value=""; });
el.dropZone?.addEventListener("dragover",e=>{ e.preventDefault(); el.dropZone.classList.add("drag-over"); });
el.dropZone?.addEventListener("dragleave",e=>{ if (!el.dropZone.contains(e.relatedTarget)) el.dropZone.classList.remove("drag-over"); });
el.dropZone?.addEventListener("drop",e=>{ e.preventDefault(); el.dropZone.classList.remove("drag-over"); addFiles(e.dataTransfer.files); });
el.dropZone?.addEventListener("click",e=>{ if (e.target.closest("#fileInput")) return; el.fileInput.click(); });
el.dropZone?.addEventListener("keydown",e=>{ if (e.key==="Enter"||e.key===" "){e.preventDefault();el.fileInput.click();} });
el.clearBtn?.addEventListener("click",()=>{ state.files=[]; el.fileQueue.innerHTML=""; syncQueueControls(); });

// ══════════════════════════════════════════════════════════════
