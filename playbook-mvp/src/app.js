'use strict';
// Only the current session's navigation, latest results, and open document live here.
const $=s=>document.querySelector(s), $$=s=>Array.from(document.querySelectorAll(s));
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const paths={book:'M4 4h6a3 3 0 0 1 3 3v14a4 4 0 0 0-4-3H4z M20 4h-4a3 3 0 0 0-3 3v14a4 4 0 0 1 4-3h3z',menu:'M4 6h16M4 12h16M4 18h16',search:'M21 21l-5-5 M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16',chevron:'M6 9l6 6 6-6',right:'M9 5l7 7-7 7',folder:'M3 7V4h6l2 3h10v13H3z',file:'M14 2H5v20h14V7z M14 2v6h5 M8 12h8M8 16h6',shield:'M12 2l8 3v6c0 5-4 8-8 11-4-3-8-6-8-11V5z', 'shield-check':'M12 2l8 3v6c0 5-4 8-8 11-4-3-8-6-8-11V5z M8 11l3 3 5-5',lock:'M6 10h12v10H6z M8 10V6a4 4 0 0 1 8 0v4',globe:'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20 M2 12h20 M12 2c6 6 6 14 0 20-6-6-6-14 0-20',users:'M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8 M2 21v-3c0-3 3-5 7-5s7 2 7 5v3 M17 4a4 4 0 0 1 0 7 M18 14c3 1 4 2 4 5v2',user:'M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10 M3 22v-2c0-4 4-6 9-6s9 2 9 6v2',info:'M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20 M12 11v6 M12 7h.01',close:'M6 6l12 12M18 6L6 18',check:'M5 12l5 5L20 7',download:'M12 3v12M7 10l5 5 5-5 M4 16v5h16v-5',code:'M8 7l-5 5 5 5M16 7l5 5-5 5M14 3l-4 18',eye:'M2 12s4-8 10-8 10 8 10 8-4 8-10 8S2 12 2 12z M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6',clock:'M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20 M12 6v6l4 2',copy:'M8 8h12v13H8z M16 8V3H3v13h5',external:'M14 3h7v7M21 3L11 13 M21 14v7H3V3h7',fold:'M4 5h16M4 19h16M8 10l4-4 4 4M8 14l4 4 4-4',arrow:'M5 12h14M14 7l5 5-5 5',hash:'M4 9h16M3 15h16M10 3L8 21M16 3l-2 18',home:'M3 10l9-8 9 8 M5 9v12h14V9 M9 21v-8h6v8',layers:'M12 2l10 6-10 6L2 8z M2 12l10 6 10-6 M2 16l10 6 10-6',up:'M6 14l6-6 6 6'};
function icon(name,cls=''){return `<svg class="icon ${cls}" viewBox="0 0 24 24" aria-hidden="true"><path d="${paths[name]||paths.file}"></path></svg>`}
function mountIcons(root=document){root.querySelectorAll('[data-icon]').forEach(n=>n.innerHTML=icon(n.dataset.icon))}
const BIZ = {0: 'NCE', 1: 'OSPA'};
const STALE = Symbol('discarded response');
const state = {epoch: 0, revision: null, principal: null, nodes: [], doc: null,
  lang: 'zh', mode: 'render', scope: 'all', filter: '', expanded: new Set(),
  selected: null, viewSeq: 0, checkSeq: 0, lastQuery: null, changing: false, checking: false, polling: false};
const pending = new Map(), downloadURLs = new Set();
const channel = 'BroadcastChannel' in window ? new BroadcastChannel('playbook-session') : null;
let toastTimer, profilesReady = false;
function cancel(lane) { pending.get(lane)?.controller.abort(); pending.delete(lane); }
function current(ticket) { return (ticket.lane === 'profiles' || ticket.epoch === state.epoch) && pending.get(ticket.lane) === ticket && !ticket.controller.signal.aborted; }
async function request(path, {lane, method = 'GET', body, auth = true, raw = false} = {}) {
  cancel(lane);
  const ticket = {lane, epoch: state.epoch, revision: state.revision, controller: new AbortController()};
  pending.set(lane, ticket);
  try {
    const response = await fetch(path, {method, credentials: 'same-origin', cache: 'no-store', signal: ticket.controller.signal,
      headers: body === undefined ? {} : {'Content-Type': 'application/json'}, body: body === undefined ? undefined : JSON.stringify(body)});
    const text = await response.text();
    if (!current(ticket)) throw STALE;
    const data = raw && response.ok ? text : text ? JSON.parse(text) : null;
    if (!response.ok) {
      if (response.status === 401 && auth) clearPrivate('会话已失效，请重新选择用户或核对会话。');
      const error = new Error(data?.error?.message || '请求失败，请重试。');
      error.status = response.status;
      throw error;
    }
    const revision = response.headers.get('X-Session-Revision');
    if (auth && (!revision || revision !== ticket.revision || !raw && data?.session_revision !== revision)) {
      clearPrivate('会话已经变更，请重新核对会话。');
      throw STALE;
    }
    return {data, revision, status: response.status};
  } catch (error) {
    if (error === STALE || error.name === 'AbortError' || !current(ticket)) throw STALE;
    throw error instanceof SyntaxError ? new Error('服务响应无效，请重试。') : error;
  }
}
function empty(title, message, retry = '') {
  return `<div class="empty-state"><div class="empty-icon">${icon('book')}</div><h1>${esc(title)}</h1><p>${esc(message)}</p>${retry ? `<button class="primary-button" data-retry="${esc(retry)}">重试</button>` : ''}</div>`;
}
function closeDialog() { if ($('#dialog').open) $('#dialog').close(); $('#dialogBody').replaceChildren(); }
function closeAsk() { if ($('#askPanel').open) $('#askPanel').close(); }
function openAsk() { closeDialog(); resetMobile(); if (!$('#askPanel').open) $('#askPanel').showModal(); $('#questionInput').focus(); }
function dialog(title, html) { closeAsk(); closeDialog(); $('#dialogTitle').textContent = title; $('#dialogBody').innerHTML = html; $('#dialog').showModal(); }
function toast(message) { clearTimeout(toastTimer); $('#toast').textContent = message; $('#toast').classList.add('show'); toastTimer = setTimeout(() => $('#toast').classList.remove('show'), 3000); }
function resetMobile() { $('#sidebar').classList.remove('open'); $('#sidebar').inert = matchMedia('(max-width:640px)').matches; $('#mobileMenu').setAttribute('aria-expanded', 'false'); }
function controls() {
  const ready = !!state.principal && !state.changing && !state.checking;
  for (const id of ['questionInput', 'askScope', 'askLanguage', 'askSubmit', 'searchInput', 'logoutButton']) $('#' + id).disabled = !ready;
  $('#userSelect').disabled = state.changing || state.checking || $('#userSelect').options.length < 2;
}
function clearPrivate(message = '请在右上角选择用户，再查阅授权资料。') {
  ++state.epoch; ++state.viewSeq; ++state.checkSeq;
  state.checking = state.polling = false; document.body.classList.remove('session-checking');
  // Profile labels are public boot data; identity changes must not strand initialization.
  for (const lane of pending.keys()) if (lane !== 'profiles') cancel(lane);
  for (const url of downloadURLs) URL.revokeObjectURL(url);
  downloadURLs.clear(); closeAsk(); closeDialog(); resetMobile(); clearTimeout(toastTimer);
  Object.assign(state, {revision: null, principal: null, nodes: [], doc: null, selected: null, lastQuery: null, filter: '', scope: 'all', mode: 'render'});
  state.expanded.clear();
  $('#toast').textContent = ''; $('#toast').classList.remove('show');
  $('#menuTree').replaceChildren(); $('#breadcrumb').replaceChildren(); $('#searchInput').value = '';
  $('#askForm').reset(); $('#askResults').replaceChildren(); $('#askResults').setAttribute('aria-busy', 'false');
  $('#askStatus').textContent = '选择用户后开始检索。'; $('#askSubmit').innerHTML = icon('search') + '检索文档';
  $('#view').innerHTML = empty('知识文档工作台', message, 'session');
  for (const id of ['visibleCount', 'accessibleTotal', 'publicCount', 'controlledCount']) $('#' + id).textContent = '—';
  $('#identityRole').textContent = '选择本地实验身份'; $('#identityNote').textContent = message;
  $('#avatar').textContent = '—'; $('#userSelect').value = ''; $('#sessionStatus').textContent = message;
  controls();
}
function acceptSession(result) {
  if (!result.revision || result.data?.session_revision !== result.revision || !result.data?.principal) throw new Error('会话响应无效。');
  state.revision = result.revision; state.principal = result.data.principal;
  const p = state.principal;
  $('#userSelect').value = p.profile_id;
  $('#identityRole').textContent = `${p.role_names[0]} · ${p.effective_business_categories.map(x => BIZ[x]).join(' + ')}`;
  $('#avatar').textContent = p.role_names[0] === '21V' ? '21' : p.role_names[0][0];
  $('#identityNote').textContent = `${p.user_name} · 当前会话授权范围`;
  $('#sessionStatus').textContent = '已连接 · 按当前身份读取';
  $('#askStatus').textContent = '输入问题，查找相关文档。'; controls();
}
async function synchronize(quiet = false) {
  if (!profilesReady || state.changing || state.polling) return;
  const checkSeq = ++state.checkSeq;
  state.polling = true; state.checking = quiet !== true;
  if (state.checking) { controls(); document.body.classList.add('session-checking'); }
  try {
    const result = await request('/api/me', {lane: 'session', auth: false});
    if (result.revision !== state.revision) {
      clearPrivate(); acceptSession(result);
      const epoch = state.epoch; await loadMenu(); if (epoch === state.epoch) await route();
    }
  } catch (error) {
    if (error !== STALE) clearPrivate(error.status === 401 ? undefined : '无法核对会话，请检查本地服务后重试。');
  } finally {
    if (checkSeq === state.checkSeq) { state.checking = state.polling = false; document.body.classList.remove('session-checking'); controls(); }
  }
}
async function switchUser(profile) {
  if (!profile || state.changing) return;
  const keepRoute = !state.principal;
  state.changing = true; clearPrivate('正在切换身份…');
  if (!keepRoute) history.replaceState(null, '', location.pathname);
  channel?.postMessage('changing');
  try {
    let result;
    try { result = await request('/api/dev/session', {lane: 'session', method: 'POST', body: {profile_id: profile}, auth: false}); }
    catch (error) {
      if (error === STALE || error.status !== 401) throw error;
      // Clear an expired HttpOnly cookie through the server before a fresh login.
      await request('/api/session', {lane: 'session', method: 'DELETE', auth: false});
      result = await request('/api/dev/session', {lane: 'session', method: 'POST', body: {profile_id: profile}, auth: false});
    }
    acceptSession(result);
    const epoch = state.epoch; await loadMenu(); if (epoch === state.epoch) await route();
  } catch (error) {
    if (error !== STALE) clearPrivate(error.message || '会话建立失败，请重试。');
  } finally { state.changing = false; controls(); channel?.postMessage('changed'); }
}
async function logout() {
  state.changing = true; clearPrivate('已清空本页，正在注销…');
  history.replaceState(null, '', location.pathname); channel?.postMessage('changing');
  try { await request('/api/session', {lane: 'session', method: 'DELETE', auth: false}); clearPrivate('已注销，请选择用户开始新会话。'); }
  catch (error) { if (error !== STALE) clearPrivate('注销未完成，请重试注销或刷新核对会话。'); }
  finally { state.changing = false; controls(); channel?.postMessage('changed'); }
}
channel?.addEventListener('message', event => {
  if (!['changing', 'changed'].includes(event.data)) return;
  clearPrivate('另一标签页正在变更会话…');
  if (event.data === 'changed' && !state.changing) synchronize();
});
async function loadMenu() {
  try {
    const {data} = await request('/api/menu', {lane: 'menu'});
    state.nodes = data.nodes;
    const keys = new Set(), publicKeys = new Set();
    for (const node of state.nodes) for (const key of node.document_keys) { keys.add(key); if (node.is_public) publicKeys.add(key); }
    $('#accessibleTotal').textContent = keys.size; $('#visibleCount').textContent = keys.size;
    $('#publicCount').textContent = publicKeys.size; $('#controlledCount').textContent = keys.size - publicKeys.size;
    for (const node of state.nodes) if (!node.parent_id) state.expanded.add(node.menu_id);
    renderTree();
  } catch (error) {
    if (error !== STALE) $('#menuTree').innerHTML = `<div class="nav-error">${esc(error.message)}<button data-retry="menu">重试目录</button></div>`;
  }
}
function parents(mid) {
  const path = [], seen = new Set();
  while (mid && !seen.has(mid)) { const node = state.nodes.find(n => n.menu_id === mid); if (!node) break; path.unshift(node); seen.add(mid); mid = node.parent_id; }
  return path;
}
function label(node) { return state.lang === 'en' ? node.label_en || node.label : node.label; }
function renderTree() {
  const kept = new Set(), filter = state.filter.toLocaleLowerCase();
  for (const node of state.nodes) {
    if ((state.scope === 'all' || node.business_category === +state.scope) && (!filter || `${node.label} ${node.label_en}`.toLocaleLowerCase().includes(filter)))
      for (const ancestor of parents(node.menu_id)) kept.add(ancestor.menu_id);
  }
  function tree(parent, depth = 0, business = null) {
    return state.nodes.filter(n => n.parent_id === parent && kept.has(n.menu_id) && (parent !== 0 || n.business_category === business)).map(node => {
      const children = state.nodes.some(n => n.parent_id === node.menu_id && kept.has(n.menu_id));
      const open = state.expanded.has(node.menu_id) || !!filter;
      const key = node.document_keys[0], selected = key && key === state.selected;
      return `<div class="tree-node"><div class="tree-row ${selected ? 'active' : ''}" style="padding-left:${10 + depth * 13}px">${children ? `<button class="tree-toggle ${open ? 'open' : ''}" data-toggle="${node.menu_id}" aria-label="${open ? '折叠' : '展开'} ${esc(label(node))}" aria-expanded="${open}">${icon('chevron')}</button>` : '<span class="tree-spacer"></span>'}<button class="tree-label" ${key ? `data-doc="${esc(key)}"` : `data-menu="${node.menu_id}"`} ${selected ? 'aria-current="page"' : ''}>${icon(key ? node.is_public ? 'globe' : 'file' : 'folder')}<span>${esc(label(node))}</span>${children ? `<small>${node.visible_document_count}</small>` : ''}</button></div>${children && open ? tree(node.menu_id, depth + 1) : ''}</div>`;
    }).join('');
  }
  $('#menuTree').innerHTML = [0, 1, null].map(biz => {
    const content = tree(0, 0, biz);
    return content ? `<div class="domain-header"><span class="domain-bullet ${biz === 1 ? 'ospa' : ''}"></span>${BIZ[biz] || '上级目录'}</div>${content}` : '';
  }).join('') || '<p class="nav-error">没有匹配的目录名称。</p>';
  $$('[data-scope]').forEach(b => b.classList.toggle('active', b.dataset.scope === state.scope));
}
function home(push = true) {
  ++state.viewSeq; for (const lane of ['document', 'raw-zh', 'raw-en', 'download']) cancel(lane);
  closeAsk(); closeDialog(); state.doc = null; state.selected = null;
  if (push) history.pushState(null, '', location.pathname); $('#breadcrumb').replaceChildren();
  $('#view').innerHTML = empty('从目录或问题开始', '选择左侧目录阅读，或在 ASK AI 中检索当前身份可见的资料。'); renderTree();
}
function openMenu(mid) {
  home(); const node = state.nodes.find(n => n.menu_id === mid); if (!node) return;
  state.expanded.add(mid); renderTree(); resetMobile();
  const nodes = state.nodes.filter(n => n.document_keys.length && parents(n.menu_id).some(p => p.menu_id === mid));
  $('#view').innerHTML = `<h1 class="doc-title">${esc(label(node))}</h1>${[2, 3].includes(node.type) ? '<p class="doc-title-en">该入口没有可读取的 Document 正文，当前仅保留目录位置。</p>' : `<p class="doc-title-en">${nodes.length} 篇当前可读文档</p><div class="section-cards">${nodes.map(n => `<button class="section-card" data-doc="${esc(n.document_keys[0])}"><strong>${esc(label(n))}</strong><small>${esc(BIZ[n.business_category] || '')}</small></button>`).join('')}</div>`}`;
}
function hash(key) { return '#doc=' + key.split(':').at(-1) + '&lang=' + state.lang; }
async function route() {
  if (!state.principal) return;
  const params = new URLSearchParams(location.hash.slice(1));
  if (['zh', 'en', 'both'].includes(params.get('lang'))) state.lang = params.get('lang');
  if (/^[1-9][0-9]{0,9}$/.test(params.get('doc') || '')) await openDocument('playbook-test:Document:' + params.get('doc'), false);
  else home(false);
}
async function openDocument(key, setHash = true) {
  if (!state.principal) return;
  ++state.viewSeq; for (const lane of ['raw-zh', 'raw-en', 'download']) cancel(lane);
  closeAsk(); closeDialog(); state.doc = null; state.selected = key; state.mode = 'render';
  $('#view').innerHTML = empty('正在读取文档…', ''); $('#breadcrumb').replaceChildren();
  $('#reader').focus({preventScroll: true});
  if (setHash) history.pushState(null, '', hash(key));
  const menu = state.nodes.find(n => n.document_keys.includes(key));
  if (menu) parents(menu.menu_id).forEach(n => state.expanded.add(n.menu_id));
  state.scope = 'all'; state.filter = ''; $('#searchInput').value = ''; renderTree(); resetMobile();
  try {
    const {data} = await request('/api/documents/' + encodeURIComponent(key), {lane: 'document'});
    state.doc = data.document; await renderDocument();
    if (state.doc?.document_key === key) $('#reader').scrollTop = 0;
    $('#menuTree [aria-current="page"]')?.scrollIntoView({block: 'nearest'});
  } catch (error) { if (error !== STALE) $('#view').innerHTML = empty(error.status === 404 ? '资源不存在或不可访问' : '读取失败', error.message, 'document'); }
}
function provenance(doc) {
  return `<dl class="source-values"><dt>来源</dt><dd>${esc(doc.source.instance_key)} · ${esc(doc.source.table)} #${doc.source.primary_key}</dd><dt>Document key</dt><dd>${esc(doc.document_key)}</dd><dt>内容版本</dt><dd>${esc(doc.content_revision)}</dd><dt>ACL 版本</dt><dd>${esc(doc.acl_revision)}</dd><dt>快照</dt><dd>${esc(doc.source.snapshot_sha256)}</dd></dl>`;
}
async function renderDocument() {
  const doc = state.doc; if (!doc) return;
  const seq = ++state.viewSeq, epoch = state.epoch;
  for (const lane of ['raw-zh', 'raw-en', 'download']) cancel(lane);
  closeDialog(); const languages = state.lang === 'both' ? ['zh', 'en'] : [state.lang];
  $('#view').innerHTML = empty('正在准备阅读视图…', '');
  try {
    const raws = {};
    if (state.mode === 'source') await Promise.all(languages.map(async lang => {
      const result = await request(`/api/documents/${encodeURIComponent(doc.document_key)}/raw?language=${lang}`, {lane: 'raw-' + lang, raw: true}); raws[lang] = result;
    }));
    if (epoch !== state.epoch || seq !== state.viewSeq) return;
    $('#breadcrumb').innerHTML = `<button data-home aria-label="首页">${icon('home')}</button>${icon('right')}<span>${BIZ[doc.business_category]}</span>${icon('right')}<span>Document #${doc.document_id}</span>`;
    const pane = lang => {
      const value = doc.languages[lang], title = lang === 'zh' ? '中文' : '英文';
      const content = state.mode === 'source' ? raws[lang].status === 204 ? '<p>源字段为 null。</p>' : `<pre class="source-code"><code>${esc(raws[lang].data)}</code></pre>` : value.raw_is_null ? '<p>源字段为 null。</p>' : value.raw_is_empty ? '<p>源字段为空字符串。</p>' : value.html;
      return `<section class="language-pane"><div class="pane-label">${title} · ${esc(value.source_field)}${state.mode === 'source' ? ' · 原文' : ''}</div><div class="article-prose">${content}</div></section>`;
    };
    $('#view').innerHTML = `<article class="doc-view"><div class="doc-kicker"><span class="chip biz">${BIZ[doc.business_category]}</span><span class="chip ${doc.is_public ? 'public' : 'controlled'}">${icon(doc.is_public ? 'globe' : 'lock')}${doc.is_public ? 'Public' : '受控'}</span><span class="chip neutral">Document #${doc.document_id}</span></div><h1 class="doc-title">${esc(state.lang === 'en' ? doc.title_en || doc.title : doc.title)}</h1><p class="doc-title-en">${esc(state.lang === 'en' ? doc.title : doc.title_en)}</p><div class="doc-meta"><span>${icon('user')}${esc(doc.metadata.Author || '未提供作者')}</span><span>${icon('clock')}更新 ${esc(doc.metadata.UpdatedTime || '未提供')}</span><span>发布状态待核验 · 本地实验</span></div><div class="doc-toolbar"><div class="language-tabs" role="group" aria-label="阅读语言">${[['zh', '中文'], ['en', 'English'], ['both', '双语']].map(([v, t]) => `<button data-lang="${v}" class="${state.lang === v ? 'active' : ''}" aria-pressed="${state.lang === v}">${t}</button>`).join('')}</div><div class="doc-actions"><button data-source>${icon('code')}${state.mode === 'render' ? '查看原文' : '阅读视图'}</button><button data-metadata>${icon('info')}元数据</button><button data-copy-url>${icon('copy')}复制链接</button><details class="download-control"><summary>${icon('download')}下载</summary><div class="download-popover"><button data-download="zh">中文原文 .md</button><button data-download="en">英文原文 .md</button><button data-download="metadata">元数据 .json</button></div></details></div></div>${doc.same_content ? '<p class="content-notice">两个源字段内容相同，按源数据展示，不自动翻译。</p>' : ''}<div class="reading-grid"><div class="document-body ${state.lang === 'both' ? 'dual-pane' : ''}">${languages.map(pane).join('')}</div><aside class="article-rail reading-rail"><p class="rail-heading">ON THIS PAGE</p><div id="documentTOC" class="toc-list"></div><details class="rail-source"><summary>来源与版本</summary>${provenance(doc)}</details></aside></div></article>`;
    $('#documentTOC').innerHTML = $$('.article-prose h1,.article-prose h2,.article-prose h3').map(h => `<button class="toc-link" data-heading="${esc(h.id)}">${esc(h.textContent)}</button>`).join('') || '<p class="toc-empty">当前视图没有章节标题。</p>';
    renderTree();
  } catch (error) { if (error !== STALE && seq === state.viewSeq) $('#view').innerHTML = empty('读取失败', error.message, 'document'); }
}
function highlightedText(text, parts) {
  if (!Array.isArray(parts) || parts.some(p => typeof p.text !== 'string' || typeof p.highlighted !== 'boolean')
      || parts.map(p => p.text).join('') !== text) return esc(text);
  return parts.map(p => p.highlighted ? `<mark>${esc(p.text)}</mark>` : esc(p.text)).join('');
}
async function search(query) {
  if (!state.principal) return;
  state.lastQuery = query;
  $('#askResults').replaceChildren(); $('#askResults').setAttribute('aria-busy', 'true');
  $('#askStatus').textContent = '正在检索…'; $('#askSubmit').innerHTML = icon('search') + '重新检索';
  try {
    const {data} = await request('/api/search', {lane: 'search', method: 'POST', body: query});
    $('#askStatus').textContent = data.items.length ? `“${query.query}” · ${data.items.length} 篇相关文档 · 按相关性得分降序` : '未找到相关文档。可以调整问题、语言或业务范围后重试。';
    $('#askResults').innerHTML = data.items.map(doc => `<article class="search-card" data-result-id="${doc.document_id}"><div class="result-tags"><span class="chip biz">${BIZ[doc.business_category]}</span><span class="chip neutral">${doc.is_public ? 'Public' : '受控'}</span><span>Document #${doc.document_id}</span>${Number.isFinite(doc.score) ? `<span class="result-score" title="BM25 相关性得分，仅供同一次检索比较">相关性得分 <strong>${doc.score.toFixed(2)}</strong></span>` : ''}</div><h3><button data-doc="${esc(doc.document_key)}">${highlightedText(doc.title, doc.title_highlights?.title)}</button></h3><p class="search-title-en">${highlightedText(doc.title_en, doc.title_highlights?.title_en)}</p>${doc.snippets.map(s => `<blockquote><small>${esc(s.source_field)}</small><p>${highlightedText(s.text, s.parts)}</p></blockquote>`).join('')}<details><summary>来源与版本</summary>${provenance(doc)}</details></article>`).join('');
    $('#askResults').setAttribute('aria-busy', 'false'); $('#askSubmit').innerHTML = icon('search') + '检索文档';
  } catch (error) {
    if (error === STALE) return;
    $('#askResults').setAttribute('aria-busy', 'false'); $('#askSubmit').innerHTML = icon('search') + '检索文档';
    $('#askStatus').innerHTML = `${esc(error.message || '检索请求失败。')} <button data-retry="search">重试检索</button>`;
  }
}
async function download(kind) {
  const doc = state.doc; if (!doc) return;
  const language = kind === 'metadata' ? state.lang === 'en' ? 'en' : 'zh' : kind;
  try {
    const result = await request(`/api/documents/${encodeURIComponent(doc.document_key)}/download?kind=${kind === 'metadata' ? 'metadata' : 'markdown'}&language=${language}`, {lane: 'download', raw: true});
    if (result.status === 204) { toast('此语言源字段为 null，没有可下载的原文。'); return; }
    const url = URL.createObjectURL(new Blob([result.data], {type: kind === 'metadata' ? 'application/json' : 'text/plain;charset=utf-8'}));
    downloadURLs.add(url);
    const a = document.createElement('a'); a.href = url; a.download = `document-${doc.document_id}-${language}.${kind === 'metadata' ? 'json' : 'md'}`;
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => { URL.revokeObjectURL(url); downloadURLs.delete(url); }, 1000);
    $$('details.download-control').forEach(d => d.open = false); toast('已下载当前授权文档。');
  } catch (error) { if (error !== STALE) toast(error.message || '下载失败，请重试。'); }
}
function about() {
  dialog('使用说明与实验边界', '<p>ASK AI 使用 Elasticsearch BM25 查找当前身份可见的文档，提供原文摘录。每次提交独立检索，不生成回答，也不保留多轮上下文。</p><p>左侧只筛选已授权目录名称；全文检索请使用 ASK AI。Public 文档对有效用户跨业务开放，受控文档仍要求菜单和业务授权。</p><p>中文和英文分别来自 Content / ContentEn，不自动翻译。表格保留源结构；SQL 未提供的图片和附件只显示引用。Status 的发布语义待核验，当前仅为本地实验开放。</p><p>右上角选择本地预设身份，切换或注销会清空本页资料；多标签页共享服务端会话。资料和授权更新后需要重新选择身份。生产登录与上游发布语义尚待确认。</p><p>点击 ASK AI 或按 ⌘ / Ctrl + K 打开搜索弹窗，Esc 关闭；点击结果进入文档。请查阅原文确认是否包含所问答案。源链接未提供可定位目标时，本页只显示引用，不推测权限。</p>');
}
$('#userSelect').addEventListener('change', e => switchUser(e.target.value));
$('#logoutButton').onclick = logout; $('#homeButton').onclick = home;
$('#aboutButton').onclick = $('#scopeInfo').onclick = about; $('#dialogClose').onclick = closeDialog;
$('#askTrigger').onclick = openAsk; $('#askClose').onclick = closeAsk;
$('#askPanel').addEventListener('click', e => { if (e.target === e.currentTarget) closeAsk(); });
$('#searchInput').addEventListener('input', e => { state.filter = e.target.value; renderTree(); });
$('#mobileMenu').onclick = () => { const open = $('#sidebar').classList.toggle('open'); $('#sidebar').inert = !open; $('#mobileMenu').setAttribute('aria-expanded', String(open)); };
$('#sidebarBackdrop').onclick = resetMobile;
$('#collapseAll').onclick = () => { if (state.expanded.size) state.expanded.clear(); else state.nodes.forEach(n => state.expanded.add(n.menu_id)); renderTree(); };
$('#askForm').addEventListener('submit', e => {
  e.preventDefault(); const query = $('#questionInput').value.trim(); if (!query) return;
  search({query, top_k: 5, language: $('#askLanguage').value, business_scope: $('#askScope').value === 'all' ? 'all' : +$('#askScope').value});
});
document.addEventListener('click', e => {
  const button = e.target.closest('button,a'); if (!button) return;
  if (button.hasAttribute('data-doc')) { openDocument(button.dataset.doc); return; }
  if (button.hasAttribute('data-menu')) { openMenu(+button.dataset.menu); return; }
  if (button.hasAttribute('data-toggle')) { const id = +button.dataset.toggle; state.expanded.has(id) ? state.expanded.delete(id) : state.expanded.add(id); renderTree(); return; }
  if (button.hasAttribute('data-scope')) { state.scope = button.dataset.scope; renderTree(); return; }
  if (button.hasAttribute('data-home')) { home(); return; }
  if (button.hasAttribute('data-lang') && state.doc) { state.lang = button.dataset.lang; history.replaceState(null, '', hash(state.doc.document_key)); renderDocument(); return; }
  if (button.hasAttribute('data-source')) { state.mode = state.mode === 'render' ? 'source' : 'render'; renderDocument(); return; }
  if (button.hasAttribute('data-download')) { download(button.dataset.download); return; }
  if (button.hasAttribute('data-metadata') && state.doc) { const {languages, ...metadata} = state.doc; dialog('文档元数据', `<pre class="json-view">${esc(JSON.stringify(metadata, null, 2))}</pre>`); return; }
  if (button.hasAttribute('data-copy-url')) { navigator.clipboard.writeText(location.href).catch(() => toast('浏览器限制剪贴板，请复制地址栏链接。')); return; }
  if (button.hasAttribute('data-heading')) { document.getElementById(button.dataset.heading)?.scrollIntoView({block: 'start'}); return; }
  if (button.hasAttribute('data-retry')) {
    const action = button.dataset.retry;
    if (action === 'session') synchronize();
    if (action === 'menu') loadMenu();
    if (action === 'document' && state.selected) openDocument(state.selected, false);
    if (action === 'search' && state.lastQuery) search(state.lastQuery);
    return;
  }
  if (button.hasAttribute('data-local-link')) { e.preventDefault(); dialog('源链接暂不可用', `<p>当前没有经过授权的目标映射或附件字节，仅保留原始引用。</p><pre class="json-view">${esc(button.dataset.localLink)}</pre>`); return; }
  if (button.hasAttribute('data-anchor')) {
    e.preventDefault(); let anchor = button.dataset.anchor; try { anchor = decodeURIComponent(anchor); } catch (_) { /* Keep malformed source reference as text. */ }
    const target = $$('.article-prose [id]').find(n => n.id === anchor || n.textContent.trim() === anchor);
    if (target) target.scrollIntoView({block: 'start'}); else toast('源锚点未能在此文档中定位。');
  }
});
document.addEventListener('keydown', e => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openAsk(); } if (e.key === 'Escape') resetMobile(); });
window.addEventListener('hashchange', route);
window.addEventListener('pagehide', () => clearPrivate());
window.addEventListener('pageshow', e => { if (e.persisted) synchronize(); });
window.addEventListener('focus', synchronize);
// ponytail: visible tabs recheck within 15 seconds; multi-host push needs a shared session service.
setInterval(() => { if (document.visibilityState === 'visible' && state.principal) synchronize(true); }, 15000);
matchMedia('(max-width:640px)').addEventListener('change', resetMobile);
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') { document.body.classList.add('session-checking'); } else synchronize(); });
async function init() {
  mountIcons(); clearPrivate();
  try {
    const {data} = await request('/api/dev/profiles', {lane: 'profiles', auth: false});
    $('#userSelect').innerHTML = '<option value="">请选择用户</option>' + data.profiles.map(p => `<option value="${esc(p.profile_id)}">${esc(p.label)} · ${esc(p.profile_id)}</option>`).join('');
    profilesReady = true;
    controls(); await synchronize(); document.documentElement.dataset.ready = 'true';
  } catch (error) { if (error !== STALE) clearPrivate('无法连接本地实验服务，请检查服务启动参数后刷新。'); }
}
init();
