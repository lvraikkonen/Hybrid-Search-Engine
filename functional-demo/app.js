/* Offline meeting prototype. No requests, API keys, or external assets. */
(() => {
  'use strict';
  const C = window.DemoCore, DATA = window.DEMO_DATA, KEY = '21v-knowledge-demo-v1';
  const $ = s => document.querySelector(s);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const paths = {
    spark: '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z"/><path d="m20 2 .7 1.3L22 4l-1.3.7L20 6l-.7-1.3L18 4l1.3-.7L20 2Z"/>',
    chat: '<path d="M20 11.5a8 8 0 0 1-8 8H5l-3 2v-10a9 9 0 0 1 18 0Z"/><path d="M7 10h8M7 14h5"/>',
    book: '<path d="M3 4h6a4 4 0 0 1 3 2 4 4 0 0 1 3-2h6v15h-6a4 4 0 0 0-3 2 4 4 0 0 0-3-2H3V4Z"/><path d="M12 6v15"/>',
    cube: '<path d="m12 2 9 5v10l-9 5-9-5V7l9-5Z"/><path d="m3 7 9 5 9-5M12 12v10M7.5 4.5l9 5"/>',
    lock: '<rect x="5" y="10" width="14" height="11" rx="3"/><path d="M8 10V7a4 4 0 0 1 8 0v3M12 15v2"/>',
    history: '<path d="M3 11a9 9 0 1 1 2.7 7M3 4v7h7M12 7v5l3 2"/>',
    feedback: '<path d="M21 11a8 8 0 0 1-8 8H7l-4 3V7a4 4 0 0 1 4-4h6a8 8 0 0 1 8 8Z"/><path d="M8 8h7M8 12h5"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    shield: '<path d="M12 2 4 5v7c0 5 8 10 8 10s8-5 8-10V5l-8-3Z"/><path d="m8 11 3 3 5-5"/>',
    users: '<path d="M16 21v-2a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v2M17 4a4 4 0 0 1 0 8M21 21v-2a4 4 0 0 0-3-3.87"/><circle cx="9.5" cy="7" r="4"/>',
    model: '<rect x="5" y="5" width="14" height="14" rx="4"/><path d="M9 1v4M15 1v4M9 19v4M15 19v4M1 9h4M1 15h4M19 9h4M19 15h4"/><rect x="9" y="9" width="6" height="6" rx="1"/>',
    sliders: '<path d="M4 4v16M12 4v16M20 4v16"/><path d="M1 8h6M9 16h6M17 10h6"/>',
    chart: '<path d="M3 3v18h18M7 16v-5M12 16V7M17 16v-3"/>',
    folder: '<path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/>',
    upload: '<path d="M12 16V3m-5 5 5-5 5 5M3 15v4a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-4"/>',
    arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>',
    up: '<path d="M12 20V4m-6 6 6-6 6 6"/>',
    down: '<path d="m6 9 6 6 6-6"/>',
    close: '<path d="m6 6 12 12M6 18 18 6"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z"/><path d="M14 2v6h6M8 13h8M8 17h6"/>',
    copy: '<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V4a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h4"/>',
    thumb: '<path d="M7 10h-4v11h4V10Zm0 0 5-8c3 0 3 3 2 7h5a2 2 0 0 1 2 2l-2 8a2 2 0 0 1-2 2H7"/>',
    flag: '<path d="M4 22V3c5-4 10 4 16 0v12c-6 4-11-4-16 0"/>',
    search: '<circle cx="10.5" cy="10.5" r="7.5"/><path d="m16 16 5 5"/>',
    info: '<circle cx="12" cy="12" r="10"/><path d="M12 11v6M12 7h.01"/>',
    menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
    pin: '<path d="m16 3 5 5-4 1-4 4v4l-2 2-3-5-5-3 2-2h4l4-4 1-4ZM8 16l-5 5"/>',
    logout: '<path d="M10 17l5-5-5-5M15 12H3M13 3h6a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-6"/>',
    guide: '<rect x="3" y="3" width="18" height="18" rx="5"/><path d="m10 8 6 4-6 4V8Z"/>',
    calendar: '<rect x="3" y="5" width="18" height="16" rx="3"/><path d="M7 2v6M17 2v6M3 11h18"/>',
    link: '<path d="m10 13 4-4M8 16l-2 2a4 4 0 0 1-6-6l5-5a4 4 0 0 1 6 0M16 8l2-2a4 4 0 0 1 6 6l-5 5a4 4 0 0 1-6 0" transform="translate(0 -1) scale(.94)"/>',
    branch: '<circle cx="6" cy="4" r="2"/><circle cx="18" cy="6" r="2"/><circle cx="6" cy="20" r="2"/><path d="M6 6v12M6 14h4a8 8 0 0 0 8-6"/>',
    download: '<path d="M12 3v13m-5-5 5 5 5-5M3 17v4h18v-4"/>'
  };
  const icon = name => `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.file}</svg>`;
  const badge = (text, color = '') => `<span class="badge ${color}">${esc(text)}</span>`;
  const btn = (action, label, symbol = '', cls = '', attrs = '') => `<button type="button" class="btn ${cls}" data-action="${action}" ${attrs}>${symbol ? icon(symbol) : ''}${esc(label)}</button>`;
  const avatar = u => `<span class="avatar ${u.color}">${esc(u.initials)}</span>`;
  const date = s => new Date(s).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false });
  const formatStatus = { published: ['当前生效', 'green'], pending: ['待审核', 'orange'], archived: ['历史版本', ''], rejected: ['已退回', 'red'] };
  const revisionBadge = v => badge(...(formatStatus[v?.status] || ['未发布', '']));
  let state;
  try { const saved = JSON.parse(localStorage.getItem(KEY)); state = saved?.schema === 1 && saved.documents && saved.users ? saved : C.seed(DATA); }
  catch (_) { state = C.seed(DATA); }
  for (const c of state.conversations) for (const m of c.messages) if (m.streaming) { m.streaming = false; m.body = (m.body || '') + '\n\n> 上次生成已中断，可以重新提问。'; }
  let ui = { route: 'chat', identityMenu: false, sidebarOpen: false, scope: 'all', filter: 'all', kbId: null, search: '', conversation: null, doc: null, streaming: false, streamToken: 0, region: '中国北部 3', diffMode: 'readable' };
  const current = () => C.user(state, state.currentUser);
  const admin = () => current().role === 'admin';
  const permittedKBs = () => state.knowledgeBases.filter(k => C.can(state, current().id, k.id));
  const pending = () => state.documents.flatMap(d => d.versions.filter(v => v.status === 'pending' && C.can(state, current().id, d.kbId) && (admin() || v.author === current().id)).map(v => ({ d, v })));
  function toast(message, error = false) {
    const div = document.createElement('div'); div.className = 'toast' + (error ? ' error' : ''); div.innerHTML = icon(error ? 'info' : 'check') + `<span>${esc(message)}</span>`;
    $('#toasts').append(div); setTimeout(() => div.remove(), 4200);
  }
  function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (_) { toast('浏览器未能保存进度，当前页面仍可继续演示。', true); } }
  function safeLink(url) { try { const u = new URL(url); return ['http:', 'https:'].includes(u.protocol) ? esc(u.href) : '#'; } catch (_) { return '#'; } }
  function inline(text) {
    text = text.replace(/<a\b[^>]*\bhref=(["'])(https?:\/\/[^"']+)\1[^>]*>([^<]*)<\/a>/gi, (_, quote, url, label) => `[${label}](${url})`);
    return esc(text.replace(/&nbsp;/g, ' ')).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) => `<a href="${safeLink(url.replace(/&amp;/g, '&'))}" target="_blank" rel="noopener noreferrer">${label}</a>`);
  }
  function markdown(text) {
    const lines = String(text).replace(/\r\n/g, '\n').split('\n'); let html = '', list = false;
    const closeList = () => { if (list) { html += '</ul>'; list = false; } };
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line || line === '&nbsp;') { closeList(); continue; }
      if (line.startsWith('|') && lines[i + 1]?.match(/^\s*\|[\s:|-]+\|\s*$/)) {
        closeList(); const cells = s => s.trim().replace(/^\||\|$/g, '').split('|').map(x => x.trim());
        html += '<div class="table-scroll"><table><thead><tr>' + cells(line).map(x => `<th>${inline(x)}</th>`).join('') + '</tr></thead><tbody>'; i++;
        while (lines[i + 1]?.trim().startsWith('|')) html += '<tr>' + cells(lines[++i]).map(x => `<td>${inline(x)}</td>`).join('') + '</tr>';
        html += '</tbody></table></div>'; continue;
      }
      if (/^[*\-] +/.test(line)) { if (!list) { html += '<ul>'; list = true; } html += `<li>${inline(line.replace(/^[*\-] +/, ''))}</li>`; continue; }
      closeList();
      if (/^#{1,6} /.test(line)) html += `<h3>${inline(line.replace(/^#{1,6} /, ''))}</h3>`;
      else if (line.startsWith('>')) html += `<blockquote>${inline(line.replace(/^>\s*/, ''))}</blockquote>`;
      else if (/^---+$/.test(line)) html += '<hr>';
      else html += `<p>${inline(line)}</p>`;
    }
    closeList(); return html;
  }
  function pageHead(title, subtitle, actions = '', eyebrow = 'WORKSPACE') {
    return `<div class="page-head"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p>${subtitle}</p></div><div class="head-actions">${actions}</div></div>`;
  }
  function empty(title, subtitle, action = '', symbol = 'book') { return `<div class="empty">${icon(symbol)}<h2>${esc(title)}</h2><p>${esc(subtitle)}</p>${action}</div>`; }
  function stat(label, value, foot, symbol = 'chart') { return `<div class="stat"><div class="stat-label">${label}${icon(symbol)}</div><div class="stat-value">${value}</div><div class="stat-foot">${foot}</div></div>`; }
  const navs = [['chat', 'Ask AI', 'spark'], ['knowledge', '知识库', 'book'], ['history', '会话记录', 'history'], ['feedback', '反馈', 'feedback']];
  const management = [['reviews', '审核与发布', 'branch'], ['permissions', '用户与权限', 'users'], ['models', '模型管理', 'model'], ['rag', 'RAG 实验室', 'sliders'], ['cost', '用量与费用', 'chart']];
  const titles = Object.fromEntries([...navs, ...management]);
  function sidebar() {
    const navButton = ([route, title, symbol]) => `<button class="nav-button ${ui.route === route ? 'active' : ''}" data-action="navigate" data-route="${route}" aria-label="${title}" title="${title}" ${ui.route === route ? 'aria-current="page"' : ''}>${icon(symbol)}<span>${title}</span>${route === 'reviews' && pending().length ? `<span class="count">${pending().length}</span>` : ''}</button>`;
    const conversations = state.conversations.filter(c => c.owner === current().id).slice().sort((a, b) => (b.messages.at(-1)?.at || b.at).localeCompare(a.messages.at(-1)?.at || a.at));
    return `<aside id="sidebar" class="sidebar" aria-label="主导航"><div class="sidebar-heading"><div class="brand"><div class="brandmark">21v</div><div class="brand-copy"><strong>知识工作台</strong><small>KNOWLEDGE WORKSPACE</small></div></div><button class="icon-button sidebar-pin" data-action="pin-sidebar">${icon('pin')}</button><button class="icon-button sidebar-close" data-action="sidebar-toggle" aria-label="收起侧栏">${icon('close')}</button></div><div class="workspace">${icon('folder')}21v-DEV<span class="dot"></span></div><div class="sidebar-scroll"><div class="nav-label">工作空间</div><nav class="nav-items">${navs.filter(([route]) => route !== 'history').map(navButton).join('')}</nav><div class="nav-label">${admin() ? '管理控制台' : '知识维护'}</div><nav class="nav-items">${(admin() ? management : [['reviews', '我的提交', 'branch']]).map(navButton).join('')}</nav></div><section class="sidebar-history" aria-label="最近会话"><div class="history-heading"><button data-action="navigate" data-route="history">会话记录 <span>${conversations.length}</span></button><button class="icon-button" data-action="new-chat" aria-label="新建会话">${icon('plus')}</button></div><div class="recent-conversations">${conversations.length ? conversations.map(c => `<button class="recent-chat ${ui.route === 'chat' && ui.conversation === c.id ? 'active' : ''}" data-action="open-chat" data-id="${c.id}" title="${esc(c.title)}" ${ui.route === 'chat' && ui.conversation === c.id ? 'aria-current="page"' : ''}>${icon('chat')}<span>${esc(c.title)}</span></button>`).join('') : '<p class="history-empty">从第一个问题开始。</p>'}</div></section><div class="sidebar-bottom"><button class="guide-card" data-action="guide" aria-label="会议演示导览" title="会议演示导览">${icon('guide')}<span><strong>会议演示导览</strong><small>从一个问题，到一次知识更新</small></span></button><button class="reset" data-action="reset" aria-label="重置演示进度" title="重置演示进度">${icon('history')}<span>重置演示进度</span></button></div></aside><button class="sidebar-backdrop" data-action="sidebar-toggle" aria-label="收起侧栏"></button>`;
  }
  function syncSidebar() {
    const side = $('.sidebar');
    side.classList.toggle('open', ui.sidebarOpen);
    side.classList.toggle('pinned', !!state.settings.sidebarPinned);
    document.body.classList.toggle('sidebar-pinned', !!state.settings.sidebarPinned);
    document.body.classList.toggle('sidebar-open', ui.sidebarOpen);
    const pinned = !!state.settings.sidebarPinned;
    const pin = $('[data-action="pin-sidebar"]');
    pin.setAttribute('aria-label', pinned ? '取消固定侧栏' : '固定展开侧栏');
    pin.setAttribute('aria-pressed', String(pinned)); pin.title = pinned ? '取消固定，恢复自动收起' : '固定展开侧栏';
    const toggle = $('.sidebar-toggle');
    const expanded = ui.sidebarOpen || (pinned && !matchMedia('(max-width:760px)').matches);
    toggle.setAttribute('aria-expanded', String(expanded));
    toggle.setAttribute('aria-label', expanded ? '收起侧栏' : '展开侧栏');
  }
  function topbar() {
    const u = current();
    const memoryEntry = `<button class="identity-option" data-action="memory">${icon('lock')}<span><strong>个人记忆</strong><small>查看与管理自己的记忆</small></span>${icon('arrow')}</button>`;
    return `<header class="topbar"><div class="breadcrumbs"><button class="icon-button sidebar-toggle" aria-label="展开侧栏" aria-controls="sidebar" aria-expanded="false" data-action="sidebar-toggle">${icon('menu')}</button><span class="crumb-parent">知识工作台</span><span class="crumb-parent">/</span><strong>${titles[ui.route] || 'Ask AI'}</strong></div><div class="top-actions"><span class="demo-pill"><span class="dot"></span>演示环境</span><button class="identity" data-action="identity" aria-label="切换演示身份，当前 ${esc(u.id)}" aria-expanded="${ui.identityMenu}">${avatar(u)}<span><strong>${esc(u.id)}</strong><small>${admin() ? '总体管理员' : esc(u.group)}</small></span>${icon('down')}</button></div></header>${ui.identityMenu ? `<div class="identity-menu">${memoryEntry}<div class="menu-label">切换演示身份 · 保留当前进度</div>${state.users.map(p => `<button class="identity-option ${p.id === u.id ? 'selected' : ''}" data-action="switch-user" data-user="${p.id}">${avatar(p)}<span><strong>${p.id}</strong><small>${p.role === 'admin' ? '总体管理员 · 管理全部项目知识库' : p.group + ' · 普通用户'}</small></span>${p.id === u.id ? icon('check') : ''}</button>`).join('')}</div>` : ''}`;
  }
  function suggestions() {
    const policy = C.can(state, current().id, 'playbook'), price = C.can(state, current().id, 'acn');
    return [
      ...(policy ? [{ type: '政策解读', icon: 'book', query: 'Azure 预留交换策略有什么变化？' }, { type: '操作清单', icon: 'check', query: '提交预留交换申请前需要哪些检查？' }] : []),
      ...(price ? [{ type: '产品价格', icon: 'cube', query: '中国北部 3 的 MySQL B1MS 每小时多少钱？' }, { type: '费用估算', icon: 'chart', query: '中国东部 3 的 MySQL D2ds v4 按 730 小时每月多少钱？' }] : []),
      ...(!price ? [{ type: '权限边界', icon: 'shield', query: '中国北部 3 的 MySQL B1MS 每小时多少钱？' }] : []),
      ...(!policy ? [{ type: '权限边界', icon: 'shield', query: 'Azure 预留交换策略有什么变化？' }] : []),
      { type: '使用提示', icon: 'lock', query: '如何上传我的个人知识？', guide: true }
    ].slice(0, 4);
  }
  function composer() {
    return `<div class="composer-wrap"><form id="chat-form" class="composer"><textarea aria-label="你的问题" id="question" placeholder="向你的知识提问，让答案有据可循…" rows="2" maxlength="1500" ${ui.streaming ? 'disabled' : ''}></textarea><div class="composer-tools"><button type="button" class="icon-button" aria-label="上传知识文档" data-action="upload">${icon('plus')}</button><select class="scope-select" id="scope" aria-label="问答知识库"><option value="all">全部已授权知识库</option>${permittedKBs().map(k => `<option value="${k.id}" ${ui.scope === k.id ? 'selected' : ''}>${k.name}</option>`).join('')}</select><span class="spacer"></span><span class="keyboard">↵</span><button class="send-button" type="${ui.streaming ? 'button' : 'submit'}" aria-label="${ui.streaming ? '停止生成' : '发送问题'}" ${ui.streaming ? 'data-action="stop"' : ''}>${ui.streaming ? '<span style="width:10px;height:10px;background:currentColor;border-radius:2px"></span>' : icon('up')}</button></div></form><div class="composer-note">${icon('shield')}仅检索已授权、已发布的内容<span>·</span>静态演示 · 本地样例回答</div></div>`;
  }
  function chatPage() {
    const conversation = state.conversations.find(c => c.id === ui.conversation && c.owner === current().id);
    return `<section class="chat-page ${conversation ? 'has-messages' : ''}">${conversation ? `<div class="chat-conversation" data-conversation="${conversation.id}" role="region" aria-label="聊天消息" tabindex="0"><div class="conversation-heading"><h2>${esc(conversation.title)}</h2>${btn('new-chat', '新建会话', 'plus', 'quiet')}</div>${conversation.messages.map((m, i) => renderMessage(m, i)).join('')}</div>` : `<div class="welcome"><div class="assistant-emblem">${icon('spark')}</div><div class="greeting">你好，${esc(current().id)} <span style="margin:0 6px;color:#c1c9d6">/</span> 让团队知识触手可及</div><h1>今天，想了解什么？</h1><p class="subtitle">从产品价格到操作手册，找到有来源、可追溯的答案。</p><div class="suggestions">${suggestions().map(s => `<button class="suggestion" data-action="${s.guide ? 'upload' : 'ask'}" ${s.guide ? '' : `data-query="${esc(s.query)}"`}><span class="suggestion-type">${icon(s.icon)}${s.type}</span><strong>${s.query}</strong></button>`).join('')}</div><div class="scope-summary">${icon('shield')}你的知识范围${permittedKBs().map(k => `<span class="scope-chip"><span class="dot"></span>${k.name}</span>`).join('')}</div></div>`}${composer()}<div class="page-footer">21v-DEV <span style="margin:0 10px">·</span> KNOWLEDGE, WITH CONTEXT.</div></section>`;
  }
  function renderMessage(m, index) {
    if (m.role === 'user') return `<div class="message user"><div class="user-bubble">${esc(m.body)}</div></div>`;
    const allowed = (m.sources || []).every(s => C.can(state, current().id, s.kbId));
    if (!allowed) return `<div class="message"><div class="info-strip">${icon('lock')}此回答引用的资料访问权限已变化，内容和引用已隐藏。</div></div>`;
    const status = m.streaming ? '正在生成' : m.kind === 'restricted' ? '权限范围检查' : m.kind === 'memory' ? '个人上下文' : m.kind === 'clarify' ? '需要补充条件' : m.kind === 'unknown' ? (m.sources?.length ? '当前版本检查' : '未找到相关内容') : '基于已发布资料';
    const feedbackActions = m.kind === 'memory' ? '' : `<button class="icon-button" aria-label="回答有帮助" data-action="helpful" data-index="${index}">${icon('thumb')}</button><button class="icon-button" aria-label="反馈回答问题" data-action="feedback-answer" data-index="${index}">${icon('flag')}</button>`;
    const details = m.streaming ? '' : `${(m.sources || []).map((s, n) => `<button class="source-card" data-action="source" data-message="${index}" data-source="${n}">${icon('file')}<span><strong>${esc(s.title)}</strong><small>${esc(s.section)} · 文档 v${s.version} · 知识库 R${s.release}</small></span>${badge(s.demo ? '演示修订' : '引用 ' + (n + 1), s.demo ? 'orange' : 'blue')}${icon('arrow')}</button>`).join('')}<div class="answer-actions"><button class="icon-button" aria-label="复制回答" data-action="copy-answer" data-index="${index}">${icon('copy')}</button>${feedbackActions}<span class="time">${date(m.at)}${m.kind === 'memory' ? '' : ' · ' + (m.sources?.length || 0) + ' 个来源'}</span></div>`;
    return `<div class="message ${m.streaming ? 'streaming' : ''}" data-message-index="${index}" aria-busy="${!!m.streaming}"><div class="assistant-heading"><span class="tiny-emblem">${icon('spark')}</span><strong>21v Ask AI</strong>${badge(status, m.kind === 'restricted' ? 'orange' : '')}</div><div class="answer-body prose"><div class="stream-content">${m.body ? markdown(m.body) : '<div class="typing-dots" aria-label="正在生成"><span></span><span></span><span></span></div>'}</div>${details}</div></div>`;
  }
  function knowledgePage() {
    const bases = permittedKBs(), docs = state.documents.filter(d => C.can(state, current().id, d.kbId));
    const selected = ui.kbId && bases.find(k => k.id === ui.kbId);
    if (selected) return kbPage(selected);
    const shown = bases.filter(k => (ui.filter === 'all' || (ui.filter === 'personal' ? !!k.owner : !k.owner)) && (k.name + k.description).toLowerCase().includes(ui.search.toLowerCase()));
    return pageHead('知识库', '让知识有归属，让每次回答有依据。', btn('upload', '上传资料', 'upload', 'primary')) + `<div class="stats">${stat('可访问知识库', bases.length, '按当前身份授权显示', 'shield')}${stat('已发布资料', docs.filter(d => d.activeId).length, '当前可用于 Ask AI', 'file')}${stat('待审修订', pending().length, '确认发布后进入检索', 'branch')}</div><div class="toolbar"><div class="segmented">${[['all', '全部知识库'], ['project', '项目知识'], ['personal', '个人知识']].map(([key, label]) => `<button data-action="kb-filter" data-filter="${key}" class="${ui.filter === key ? 'active' : ''}">${label}</button>`).join('')}</div><label class="search-box">${icon('search')}<input id="kb-search" aria-label="搜索知识库" placeholder="搜索知识库…" value="${esc(ui.search)}"></label></div><div class="knowledge-grid">${shown.map(k => {
      const kd = docs.filter(d => d.kbId === k.id), submissions = kd.flatMap(d => d.versions).filter(v => v.status === 'pending').length;
      return `<button class="knowledge-card" data-action="open-kb" data-kb="${k.id}"><div class="card-top"><span class="kb-icon ${k.color}">${icon(k.icon)}</span>${badge(k.owner ? '仅自己可见' : '项目知识库', k.owner ? 'purple' : '')}</div><h2>${k.name}</h2><p>${k.description}</p><div class="card-meta"><span>${kd.filter(d => d.activeId).length} 份已发布资料</span><span>${submissions ? submissions + ' 份待审核' : '发布版本 R' + k.release}</span></div><div class="card-bottom"><span>${k.owner ? current().id + ' · 个人空间' : k.group + ' · 已授权访问'}</span>${icon('arrow')}</div></button>`;
    }).join('')}</div>${!shown.length ? empty('没有匹配的知识库', '试试调整分类或搜索词。') : ''}<div class="onboarding-line">${icon('lock')}不同可见范围的资料，分别放入对应的知识库。</div>`;
  }
  function kbPage(k) {
    const docs = state.documents.filter(d => d.kbId === k.id);
    return pageHead(k.name, k.description, btn('navigate', '全部知识库', 'arrow', 'quiet', 'data-route="knowledge"') + (C.can(state, current().id, k.id, 'upload') ? btn('upload', '上传资料', 'upload', 'primary', `data-kb="${k.id}"`) : '')) + `<div class="two-col"><div class="stack"><div class="panel"><div class="panel-title"><h2>知识文档</h2>${badge('发布版本 R' + k.release, 'blue')}</div>${docs.length ? `<div class="table-scroll"><table><thead><tr><th>文档</th><th>生效版本</th><th>操作</th></tr></thead><tbody>${docs.map(d => `<tr><td><div class="doc-name"><span class="file-icon ${d.format}">${d.format.toUpperCase()}</span><div><strong>${esc(d.title)}</strong><small>${esc(d.name)}</small></div></div></td><td>${d.activeId ? badge('v' + C.active(d).number, 'green') : badge('未生效', 'orange')}${d.versions.some(v => v.status === 'pending') ? '<small>有修订待审核</small>' : ''}</td><td>${btn('open-doc', '查看', '', '', `data-doc="${d.id}"`)}</td></tr>`).join('')}</tbody></table></div>` : empty('留下一份属于你的知识', '上传 Markdown 或 JSON，确认发布后就可以对它提问。', btn('upload', '上传第一份资料', 'plus', '', `data-kb="${k.id}"`))}</div><div class="panel"><div class="panel-title"><h2>发布记录</h2><span class="small muted">保留每一次知识变更</span></div><div class="panel-pad">${k.releases.length ? `<div class="timeline">${k.releases.slice(0, 8).map(r => `<div class="timeline-item"><strong>R${r.number} · ${esc(r.reason)}</strong><p>${esc(r.actor)} · ${date(r.at)}</p><p>${Object.keys(r.manifest).length} 份生效文档</p></div>`).join('')}</div>` : '<p class="small muted">第一份资料发布后，将在这里留下记录。</p>'}</div></div></div><div class="stack"><div class="panel panel-pad"><span class="kb-icon ${k.color}">${icon(k.icon)}</span><div class="section-row"><h2>访问范围</h2>${badge(k.owner ? '个人' : '按组授权')}</div><p class="small muted">${k.owner ? '只有你可以查询和管理这里的内容。' : '资料只向已授权用户组开放，上传与发布分别授权。'}</p><div class="divider"></div>${state.users.filter(u => C.can(state, u.id, k.id)).map(u => `<div class="table-person" style="margin-top:13px">${avatar(u)}<div><strong class="small">${u.id}</strong><small class="muted" style="display:block;font-size:10px">${u.role === 'admin' ? '总体管理 · 审核发布' : k.owner ? '所有者' : u.group}</small></div></div>`).join('')}</div><div class="info-strip">${icon('info')}文档更新后先生成待审修订，当前已发布内容继续可用。</div></div></div>`;
  }
  function historyPage() {
    const list = state.conversations.filter(c => c.owner === current().id).slice().reverse();
    return pageHead('会话记录', '每个问题，都有当时的依据。', btn('new-chat', '新建会话', 'plus', 'primary')) + `<div class="panel">${list.length ? `<table><thead><tr><th>会话</th><th>消息</th><th>时间</th><th></th></tr></thead><tbody>${list.map(c => `<tr><td><strong>${esc(c.title)}</strong><small>${c.owner} · 仅自己可见</small></td><td>${c.messages.length}</td><td class="small">${date(c.at)}</td><td>${btn('open-chat', '继续对话', 'arrow', 'quiet', `data-id="${c.id}"`)}</td></tr>`).join('')}</tbody></table>` : empty('从第一个问题开始', '对话和引用的版本会保存在当前浏览器中，切换身份后各自独立。', btn('new-chat', '前往 Ask AI', 'spark', 'primary'), 'history')}</div>`;
  }
  function reviewsPage() {
    const items = pending();
    return pageHead(admin() ? '审核与发布' : '我的提交', admin() ? '看清变化，再让新知识生效。' : '跟踪资料从提交到生效的每一步。', '', 'KNOWLEDGE LIFECYCLE') + `<div class="stats">${stat('待审核', items.length, '等待确认的内容修订', 'branch')}${stat('并行修订', items.filter(({ d, v }) => C.conflicts(d, v).length).length, '需人工确认保留版本', 'shield')}${stat('本次发布', state.activity.filter(a => a.text.startsWith('确认发布') && C.can(state, current().id, a.kbId)).length, '变更均保留审计记录', 'check')}</div><div class="panel"><div class="panel-title"><h2>待处理修订</h2>${badge(admin() ? '21v-admin 审核' : '当前用户提交')}</div>${items.length ? `<div class="table-scroll"><table><thead><tr><th>修订内容</th><th>提交人</th><th>状态</th><th>操作</th></tr></thead><tbody>${items.map(({ d, v }) => `<tr><td><div class="doc-name"><span class="file-icon ${d.format}">${d.format.toUpperCase()}</span><div><strong>${esc(d.title)}</strong><small>v${v.number} · ${esc(v.label)}</small></div></div></td><td><strong>${esc(v.author)}</strong><small>${date(v.at)}</small></td><td>${C.conflicts(d, v).length ? badge('并行修订冲突', 'red') : v.indexStatus === 'failed' ? badge('索引准备失败', 'red') : revisionBadge(v)}</td><td>${btn('review', '查看差异', 'branch', '', `data-doc="${d.id}" data-version="${v.id}"`)}</td></tr>`).join('')}</tbody></table></div>` : empty('当前没有待审修订', '在知识库中打开一份文档，选择“创建演示修订”，即可开始完整发布流程。', btn('navigate', '前往知识库', 'book', '', 'data-route="knowledge"'), 'check')}</div><div class="section-row"><h2>最近的知识活动</h2></div><div class="panel panel-pad">${activityList()}</div>`;
  }
  function activityList() {
    const records = state.activity.filter(a => !a.kbId ? admin() || a.actor === current().id : C.can(state, current().id, a.kbId)).slice(0, 10);
    return records.length ? `<div class="timeline">${records.map(a => `<div class="timeline-item"><strong>${esc(a.text)}</strong><p>${esc(a.actor)} · ${date(a.at)}</p></div>`).join('')}</div>` : '<p class="small muted">尚未发生变更。每次提交、审核和发布都会在这里留下记录。</p>';
  }
  function permissionsPage() {
    return pageHead('用户与权限', '系统角色决定职责，知识库授权决定访问范围。', badge('单组织 · 21v-DEV', 'blue'), 'ACCESS CONTROL') + `<div class="panel"><div class="panel-title"><h2>团队成员</h2><span class="small muted">${state.users.length} 位演示成员</span></div><div class="table-scroll"><table><thead><tr><th>用户</th><th>用户组</th><th>系统角色</th><th>可访问项目知识库</th></tr></thead><tbody>${state.users.map(u => `<tr><td><div class="table-person">${avatar(u)}<strong>${u.id}</strong></div></td><td>${u.group}</td><td>${badge(u.role === 'admin' ? '总体管理员' : '普通用户', u.role === 'admin' ? 'blue' : '')}</td><td>${state.knowledgeBases.filter(k => !k.owner && C.can(state, u.id, k.id)).map(k => badge(k.name)).join(' ') || '<span class="muted">暂无授权</span>'}</td></tr>`).join('')}</tbody></table></div></div><div class="section-row"><h2>用户组授权</h2><p>同组成员共享项目权限，个人知识仍仅本人可见。</p></div><form id="permissions-form" class="panel"><div class="table-scroll"><table><thead><tr><th>用户组</th><th>知识库</th><th>查询 / 阅读</th><th>上传资料</th><th>发布权限</th></tr></thead><tbody>${Object.keys(state.grants).flatMap(group => ['playbook', 'acn'].map(kbId => `<tr><td><strong>${group}</strong></td><td>${C.kb(state, kbId).name}</td><td><input class="switch" type="checkbox" aria-label="${group} 查询 ${kbId}" name="${group}|${kbId}|read" data-grant-read="${group}|${kbId}" ${state.grants[group][kbId].read ? 'checked' : ''}></td><td><input class="switch" type="checkbox" aria-label="${group} 上传 ${kbId}" name="${group}|${kbId}|upload" ${state.grants[group][kbId].upload ? 'checked' : ''} ${!state.grants[group][kbId].read ? 'disabled' : ''}></td><td><span class="small muted">统一由 21v-admin 审核</span></td></tr>`)).join('')}</tbody></table></div><div class="dialog-footer"><span class="left-note">保存后可切换用户，直接验证新的访问范围。</span><button class="btn primary" type="submit">${icon('check')}保存授权</button></div></form><div class="info-strip">${icon('shield')}授权撤销后，后续查询不再使用该知识库；历史回答中涉及失去权限的资料也会隐藏。</div>`;
  }
  function modelsPage() {
    const cards = [
      { name: state.settings.model, role: '回答生成', subtitle: '结合授权资料生成答案，保留可核查的引用。', kind: 'Chat completion', tag: '默认问答模型', icon: 'spark' },
      { name: 'BGE · 中文向量检索', role: '文档向量', subtitle: '为 Markdown 和提取后的价格内容建立语义索引。', kind: 'Embedding', tag: '知识入库', icon: 'cube' },
      { name: 'BGE Reranker', role: '结果精排', subtitle: '对召回片段重新排序，把相关内容放在前面。', kind: 'Reranking', tag: '检索优化', icon: 'sliders' }
    ];
    return pageHead('模型管理', '为每一个知识环节，选择合适的模型。', btn('configure-model', '配置问答模型', 'sliders', 'primary'), 'MODEL REGISTRY') + `<div class="model-grid">${cards.map((m, i) => `<div class="panel"><div class="model-card"><div class="model-logo">${icon(m.icon)}</div><div class="model-info"><h3>${esc(m.name)}</h3><p>${m.subtitle}</p><div class="model-tags">${badge(m.role)}${badge(m.tag, 'blue')}</div></div>${badge(i === 0 && !state.settings.modelsEnabled ? '已停用' : '已配置', i === 0 && !state.settings.modelsEnabled ? '' : 'green')}</div><div class="model-footer"><span>${m.kind} · 演示配置</span>${btn('test-model', '模拟连接测试', 'link', 'quiet', `data-name="${esc(m.name)}"`)}</div></div>`).join('')}<div class="panel panel-pad"><h3>统一配置，独立职责</h3><p class="small muted" style="margin-top:13px;line-height:2">生成模型、Embedding 和重排序分别管理。模型调整与知识内容版本分开记录。</p><div class="divider"></div><div class="form-field inline"><label for="model-enabled">启用问答服务</label><input id="model-enabled" type="checkbox" class="switch" ${state.settings.modelsEnabled ? 'checked' : ''}></div><p class="note">本 Demo 的连接与回答均为本地模拟，不需要填写密钥。</p></div></div>`;
  }
  const labTabs = [['ingest', '入库与索引'], ['retrieve', '召回'], ['context', '精排与上下文'], ['generate', '生成'], ['evaluate', '评测'], ['release', '审核与发布']];
  const labConfig = () => C.labState(state).draft.config;
  const labBaselineLabel = scope => scope === 'joint' ? ['acn', 'playbook'].map(k => (k === 'acn' ? 'ACN' : 'Playbook') + ' R' + C.labBaseline(state, k).number).join(' / ') : (C.labBaseline(state, scope).scope === 'global' ? '全局 ' : '') + 'R' + C.labBaseline(state, scope).number;
  const labRun = () => C.labState(state).runs.find(r => r.snapshot.experimentId === C.labState(state).draft.id);
  function labField(key, label, options, hint = '') {
    const value = labConfig()[key], attrs = `id="lab-${key}" data-lab-field="${key}"`;
    const control = Array.isArray(options) ? `<select ${attrs}>${options.map(([v, t]) => `<option value="${esc(v)}" ${value === v ? 'selected' : ''}>${esc(t)}</option>`).join('')}</select>` : `<input ${attrs} value="${esc(value)}" ${options || 'type="text" maxlength="300"'}>`;
    return `<div class="form-field"><label for="lab-${key}">${label}</label>${control}${hint ? `<span class="hint">${hint}</span>` : ''}</div>`;
  }
  function labToggle(key, label, hint) {
    return `<div class="lab-toggle"><div><label for="lab-${key}">${label}</label><p>${hint}</p></div><input class="switch" id="lab-${key}" data-lab-field="${key}" type="checkbox" ${labConfig()[key] ? 'checked' : ''}></div>`;
  }
  const labNumber = (key, label, min, max, hint = '', step = 1) => labField(key, label, `type="number" min="${min}" max="${max}" step="${step}"`, hint);
  function labPanel(title, subtitle, body, extra = '') {
    return `<section class="panel lab-panel"><header><div><h3>${title}</h3><p>${subtitle}</p></div>${extra}</header><div class="lab-panel-body">${body}</div></section>`;
  }
  function labEvidence() {
    const scope = C.labState(state).draft.scope, sources = C.labSources(state, scope);
    const source = sources.find(s => s.id === ui.labSource) || sources.find(s => s.id === (scope === 'playbook' ? 'playbook' : 'mysql')) || sources[0];
    const doc = source && state.documents.find(d => d.id === source.id), v = doc && C.active(doc);
    if (!v) return { source: null, context: '当前范围没有已发布资料。', quote: '', locator: '' };
    if (doc.id === 'mysql') {
      const payload = JSON.parse(v.raw), row = C.tables(v.raw, '中国北部 3').flat().find(r => r[0] === 'B1MS' && r.length === 4);
      return { source, context: `${payload.slug} · ${payload.language} · 中国北部 3\n实例 B1MS · 现用现付 · 计算资源`, quote: row ? `实例 ${row[0]} · ${row[1]} vCore · 内存 ${row[2]}\n计算单价 ${row[3]}\n存储、备份、I/O 需分别核对。` : '该版本未找到北部 3 的 B1MS 记录，请人工检查。', locator: 'contentGroups[groupName="中国北部 3"].content → HTML 表格 / B1MS' };
    }
    const section = C.sections(v.raw)[0];
    return { source, context: 'Playbook 操作手册\n标题层级 + 完整段落 + 政策适用时间', quote: section ? C.plain(section.text).slice(0, 260) : C.plain(v.raw).slice(0, 260), locator: section?.title || '原文首段' };
  }
  function labIngest() {
    const d = C.labState(state).draft, e = labEvidence(), ready = d.index?.key === C.labIndexKey(state);
    const form = `<div class="lab-fields">${labField('jsonMode', 'Pricing JSON 解析', [['structured', '结构化字段 + HTML 表格'], ['text', '对照实验：整段文本']])}${labField('markdownMode', 'Markdown 切分', [['heading', '标题层级 + 完整步骤'], ['fixed', '对照实验：固定长度']])}${labNumber('chunk', 'Chunk 大小（字符）', 200, 2000, '演示以字符计；完整表格行与步骤优先保留。')}${labNumber('overlap', '重叠长度（字符）', 0, 500)}${labField('analyzer', '中文分词', [['smartcn', '中文分词 + 术语保护'], ['standard', '标准分词器']])}${labField('embedding', 'Embedding 模型', [['BGE-M3', 'BGE-M3 · 多语言'], ['中文 Embedding', '中文 Embedding · 对照']])}</div>${labField('keywords', '领域术语 / KEYWORD', null, '逗号分隔；产品名、SKU 与业务术语保持完整。')}${labField('enrichment', '入库理解模型', [['independent', '独立供应商 · 文档理解模型'], ['shared', '复用回答供应商 · 独立模型配置'], ['off', '关闭 LLM 辅助理解']], '模型建议业务标签与证据类型；价格、单位和日期必须回指原文。低置信度进入人工复核。')}<div class="lab-callout">原文与版本始终保留。文档变更通过 diff 审核，只重建受影响证据；冲突待人工确认。</div><div class="lab-actions">${btn('lab-index', ready ? '重新准备候选索引' : '准备候选索引', 'cube', 'primary')}${ready ? badge('候选索引已就绪', 'green') : badge('尚未准备 / 需要重建', 'orange')}</div>`;
    const preview = `${['global', 'joint'].includes(d.scope) ? `<div class="form-field"><label for="lab-source-preview">查看来源</label><select id="lab-source-preview"><option value="mysql" ${e.source?.id === 'mysql' ? 'selected' : ''}>Pricing JSON</option><option value="playbook" ${e.source?.id === 'playbook' ? 'selected' : ''}>Markdown 操作手册</option></select></div>` : ''}<ol class="lab-lineage"><li><span>01</span><div><h4>原始来源</h4><p>${esc(e.source?.name || '暂无资料')}</p><small>${e.source ? `文档 v${e.source.version} · 知识库 R${e.source.release}` : '先发布一份知识文档'}</small></div></li><li><span>02</span><div><h4>业务上下文</h4><p class="lab-pre">${esc(e.context)}</p></div></li><li><span>03</span><div><h4>可检索的证据</h4><p class="lab-pre">${esc(e.quote)}</p><code>${esc(e.locator)}</code></div></li></ol><p class="lab-note">以上为本地样例字段预览。LLM 理解、分块与 Elasticsearch / 向量索引均为流程演示。</p>${e.source ? btn('open-doc', '查看原文、diff 与版本', 'file', '', `data-doc="${e.source.id}"`) : ''}`;
    return `<div class="lab-columns">${labPanel('让知识带着上下文入库', '针对文档类型保留结构，再建立候选索引。', form)}${labPanel('从来源到证据', '每一条证据都能回到原始位置与版本。', preview, badge('样例预览'))}</div>`;
  }
  function labPreviewActor() { const scope = C.labState(state).draft.scope; return ui.labActor || (scope === 'playbook' ? 'Laoyang' : ['joint', 'global'].includes(scope) ? '21v-admin' : 'Laojiu'); }
  function labPreviewAllowed(e) { return !!e.source && C.can(state, labPreviewActor(), e.source.kbId); }
  function labRetrieval() {
    const d = C.labState(state).draft, c = d.config, e = labEvidence(), actor = labPreviewActor();
    const allowed = e.source && C.can(state, actor, e.source.kbId);
    const form = `${labField('retrieval', '召回方式', [['hybrid', '混合检索 · BM25 + 向量'], ['bm25', 'Elasticsearch · BM25 全文'], ['vector', '语义 / 向量检索']])}<div class="lab-fields">${labNumber('lexicalK', 'BM25 候选条数', 1, 100)}${labNumber('vectorK', '向量候选条数', 1, 100)}</div>${labToggle('regionFilter', '约束区域、规格与适用时间', '让相关性排序在业务条件内进行。缺少关键条件时先澄清。')}<div class="lab-callout">权限过滤固定开启：先限制组织、用户与知识范围，再召回；精排和引用输出时再次核验。</div><p class="lab-note">混合结果使用 RRF 融合。BM25 与向量分数不能直接相加；非选中检索通道的候选数不参与本次实验。</p>`;
    const trace = `<div class="form-field"><label for="lab-actor">模拟提问用户</label><select id="lab-actor">${state.users.map(u => `<option ${u.id === actor ? 'selected' : ''}>${u.id}</option>`).join('')}</select></div><div class="lab-query">${d.scope === 'playbook' ? '预留交换策略何时变化？' : '中国北部 3 的 MySQL B1MS 每小时多少钱？'}</div>${btn('lab-retrieve', '模拟召回', 'search', 'primary')}${ui.labTrace ? `<div class="lab-trace">${allowed ? `<div class="lab-result-row"><span class="lab-rank">1</span><div><strong>${esc(e.source.title)}</strong><p>${esc(e.locator)}</p></div>${badge('授权通过', 'green')}</div><div class="lab-result-row"><span class="lab-rank">2</span><div><strong>相似内容 · 不同区域 / 时间</strong><p>${c.regionFilter ? '业务条件不匹配，排除在候选之外。' : '进入候选：语义相似，但适用条件可能不符。'}</p></div>${badge(c.regionFilter ? '已过滤' : '待核对', c.regionFilter ? '' : 'orange')}</div><div class="lab-result-row"><span class="lab-rank">3</span><div><strong>计费说明 / 政策补充</strong><p>与主证据关联，供后续上下文组装。</p></div></div>` : `<div class="lab-callout">${esc(actor)} 在当前范围没有这份资料的访问权限。召回为空，后续精排与生成都不接收其内容。</div>`}</div>` : '<p class="lab-note">运行后查看预置候选示意。可切换跨组用户，展示权限过滤。</p>'}<p class="lab-note">这里展示 3 类候选的处理方式，数量与分数不是实际搜索结果。</p>`;
    return `<div class="lab-columns">${labPanel('先找对，再找全', '全文检索负责精确术语，向量检索补充语义表达。', form)}${labPanel('看一次召回如何发生', '以用户身份验证范围与候选过滤。', trace)}</div>`;
  }
  function labContext() {
    const c = labConfig(), e = labEvidence();
    return `<div class="lab-columns">${labPanel('把证据排好，再交给模型', '候选数量、精排数量与最终 Top K 分别管理。', `${labToggle('rerank', '启用 BGE Reranker 精排', '在融合候选上重新判断问题与片段的相关性。')}<div class="lab-fields">${labNumber('rerankPool', '送入精排的候选数', 1, 100)}${labNumber('topK', '最终 Top K', 1, 10)}${labNumber('contextBudget', '上下文预算（Token）', 1000, 16000)}</div>${labToggle('dedupe', '去重并补齐相邻上下文', '合并重复证据，补回标题、完整步骤、注意事项和适用条件。')}<div class="lab-callout">权限边界、区域、有效时间与版本信息随证据传递，不能在压缩上下文时丢失。</div>`)}${labPanel('最终上下文包', '模拟用户：' + labPreviewActor() + ' · 预置顺序示意', !labPreviewAllowed(e) ? '<div class="lab-callout">当前模拟用户没有证据访问权限，上下文为空。可在召回阶段切换身份。</div>' : `<div class="lab-flow"><span>融合候选</span>${icon('arrow')}<span>${c.rerank ? c.rerankPool + ' 条精排' : '跳过精排'}</span>${icon('arrow')}<strong>Top ${c.topK}</strong></div><div class="lab-evidence"><div>${badge('E1', 'blue')}${badge(c.rerank ? '原排名 3 → 1' : '原始顺序')}</div><h4>${esc(e.source?.title || '暂无证据')}</h4><p class="lab-pre">${esc(e.quote)}</p><small>${esc(e.locator)}</small></div><div class="lab-evidence"><div>${badge('E2')}</div><h4>适用条件与补充说明</h4><p>${c.dedupe ? '保留父标题和关联说明，移除重复片段。' : '未去重，重复片段可能占用上下文预算。'}</p></div><p class="lab-note">预算上限 ${c.contextBudget == null ? "待填写" : c.contextBudget.toLocaleString()} Token · 实际 Token 统计需接入模型 tokenizer。知识证据与个人 Memory 分别组装；实验样本不读取个人 Memory。</p>`)}</div>`;
  }
  function labGeneration() {
    const c = labConfig(), e = labEvidence();
    const body = `${labField('model', '回答生成模型', [['Azure OpenAI · 标准问答', 'Azure OpenAI · 标准问答'], ['兼容 API · 对照模型', '兼容 API · 对照模型']])}<div class="lab-fields">${labNumber('temperature', 'Temperature', 0, 1, '', .1)}${labNumber('maxTokens', '最大输出 Token', 200, 4000)}</div><div class="form-field"><div class="lab-label-row"><label for="lab-prompt">生成 Prompt · v${c.promptVersion}</label>${btn('lab-prompt', '采用有据回答模板', 'file', 'quiet')}</div><textarea id="lab-prompt" data-lab-field="prompt" maxlength="4000" rows="7">${esc(c.prompt)}</textarea><span class="hint">每次实验记录保存 Prompt 全文与版本。修改后需要重新评测。</span></div><details class="lab-details"><summary>与生效 Prompt 对比</summary><pre id="lab-prompt-diff" class="raw-code">${esc(C.textDiff(C.labBaseline(state, C.labState(state).draft.scope).config.prompt, c.prompt, 'generation-prompt.md'))}</pre></details>${labToggle('citations', '强制来源引用与适用条件', '答案携带文档版本、证据定位、区域与政策日期。')}${labToggle('abstain', '证据不足时澄清或拒答', '不补写未知价格，不把推测当成政策。')}`;
    const preview = `${badge('固定样例 · 非模型输出', 'orange')}<div class="lab-query">${C.labState(state).draft.scope === 'playbook' ? '预留交换策略何时变化？' : '中国北部 3 的 MySQL B1MS 每小时多少钱？'}</div>${btn('lab-generate', '预览模拟回答', 'spark', 'primary')}${ui.labAnswer ? `<div class="lab-answer"><p class="lab-pre">${esc(labPreviewAllowed(e) ? e.quote : '当前模拟用户没有此知识的访问权限，无法基于这些资料回答。')}</p>${labPreviewAllowed(e) && c.citations && e.source ? `<div class="lab-citation">${icon('file')}<span>${esc(e.source.name)} · v${e.source.version}<br>${esc(e.locator)}</span></div>` : labPreviewAllowed(e) ? '<p class="lab-warning">当前模板未强制保留引用，评测将检查这一缺口。</p>' : ''}</div>` : '<p class="lab-note">预览用于展示回答结构与引用。任意 Prompt 的实际效果，需在真实模型接入后评估。</p>'}<div class="lab-callout">个人 Memory 按用户隔离，仅提供当前用户的偏好。它不能替代知识证据、改变权限，或进入共享金标集。</div>`;
    return `<div class="lab-columns">${labPanel('答案也需要一个可追溯的版本', '入库理解模型与回答模型独立配置。', body)}${labPanel('让回答保持可核查', '模拟用户：' + labPreviewActor() + ' · 对照引用与拒答规则', preview)}</div>`;
  }
  function labMetricValue(value, unit) { return unit === 'ms' ? value + ' ms' : unit === '元' ? '¥' + value.toFixed(3) : unit === 'score' ? value.toFixed(2) : (value * 100).toFixed(0) + '%'; }
  function labMetricTable(run) {
    return `<div class="table-scroll"><table class="lab-metrics"><caption>同一知识快照与金标集上的基线 / 候选对照 · 全部数值为模拟</caption><thead><tr><th>阶段 / 指标</th><th>基线</th><th>候选</th><th>发布门槛</th><th>判定</th></tr></thead><tbody>${C.labMetrics.map(([key, phase, label, limit, op, unit]) => { const value = run.result.metrics[key], pass = op === 'min' ? value >= limit : value <= limit; return `<tr><td><strong>${label === 'Recall@K' ? 'Recall@' + run.snapshot.config.topK : label}</strong><small>${phase}</small></td><td>${labMetricValue(run.baseline.metrics[key], unit)}</td><td class="${pass ? 'lab-good' : 'lab-bad'}">${labMetricValue(value, unit)}</td><td>${op === 'min' ? '≥' : '≤'} ${labMetricValue(limit, unit)}</td><td>${badge(pass ? '通过' : '未达标', pass ? 'green' : 'red')}</td></tr>`; }).join('')}</tbody></table></div>`;
  }
  function labEvaluation() {
    const l = C.labState(state), run = labRun(), fresh = C.labFresh(state, run), cases = run && fresh ? run.result.cases : C.labSnapshot(state).cases;
    const shown = ui.labFailures && run && fresh ? cases.filter(c => c.pass !== true) : cases;
    const result = run ? `<div class="lab-verdict"><div><span class="eyebrow">RUN ${String(run.number).padStart(3, '0')} · 模拟对照</span><h3>${!fresh ? '配置已变化，需要重新评测' : C.labPassed(run) ? '候选方案已通过演示门槛' : '发现差距，先改进再发布'}</h3><p>冻结：知识版本 · 索引 · 配置 · Prompt · 金标 v${run.snapshot.datasetVersion}</p></div>${badge(!fresh ? '历史结果' : C.labPassed(run) ? '可提交审核' : '尚未达标', fresh && C.labPassed(run) ? 'green' : 'orange')}</div>${labMetricTable(run)}<p class="lab-note">${run.snapshot.scope === 'joint' ? '联合基线读取各库生效配置，以各项较差的模拟指标汇总。' : ''}这些门槛用于讨论产品验收标准。数值由预置规则生成，并非在这 ${run.snapshot.cases.length} 条用例上统计得到的真实指标。</p>` : `<div class="lab-empty"><div class="tiny-emblem">${icon('chart')}</div><h3>准备好验证这次改变</h3><p>在相同的知识快照与金标集上比较基线和候选方案，分别检查检索、生成、权限与费用。</p><div class="lab-flow"><span>冻结输入</span>${icon('arrow')}<span>对照评测</span>${icon('arrow')}<span>定位失败</span></div></div>`;
    return `${labPanel('Benchmark · 基线与候选', '调优集与正式保留测试集在产品中分开管理；本 Demo 使用同一组演示用例。', result, btn('lab-evaluate', '运行模拟评测', 'guide', 'primary'))}<div class="lab-section-heading"><div><h3>Golden dataset <span>v${l.datasetVersion}</span></h3><p>预置条目是示例金标；新增条目需人工标注并在每次运行中判定。</p></div><div class="lab-actions">${run && fresh ? btn('lab-failures', ui.labFailures ? '显示全部' : '只看失败 / 待判定', 'flag') : ''}${btn('lab-add-case', '人工添加金标', 'plus')}</div></div><div class="panel"><div class="table-scroll"><table><thead><tr><th>问题与范围</th><th>标注来源</th><th>本次判定</th><th></th></tr></thead><tbody>${shown.map(c => `<tr><td><strong>${esc(c.question)}</strong><small>${esc(C.labScopes[c.scope] || '公共边界用例')}</small></td><td>${c.manual ? '人工标注 · ' + esc(c.actor) : '预置示例'}</td><td>${badge(!run || !fresh ? '待运行' : c.pass === null ? '待人工判定' : c.pass ? '通过' : '未通过', !run || !fresh ? '' : c.pass ? 'green' : 'orange')}</td><td>${btn('lab-case', '查看', '', 'quiet', `data-id="${c.id}"`)}</td></tr>`).join('') || '<tr><td colspan="4">当前没有失败或待判定用例。</td></tr>'}</tbody></table></div></div><div class="lab-callout lab-loop"><strong>反馈 → 人工筛选与脱敏 → 金标新版本 → 回归评测 → 审核发布</strong><p>反馈候选可从“人工添加金标”中选取。用户的私人资料与 Memory 不进入共享实验集。</p></div>`;
  }
  function labReleasePage() {
    const l = C.labState(state), d = l.draft, run = labRun(), baseline = C.labBaseline(state, d.scope), fresh = C.labFresh(state, run), passed = C.labPassed(run);
    let actions = '', message = '配置与模型调整先留在候选方案中，统一确认后才发布。';
    if (d.scope === 'joint') message = '联合实验用于验证多知识库协作，保留评测报告；具体方案在各知识库或全局范围审核发布。';
    else if (run?.status === 'published') message = '这个实验已发布。可以创建下一次实验，或在下方回退当前方案。';
    else if (run?.status === 'pending' && fresh) {
      message = '请核对输入快照、指标与失败用例，填写审核意见后确认。';
      actions = `<div class="form-field"><label for="lab-review-note">审核意见</label><textarea id="lab-review-note" maxlength="500" placeholder="例如：已核对区域、单位和权限用例，同意在开发组试运行。"></textarea></div><div class="lab-actions">${btn('lab-publish', '确认发布方案', 'check', 'primary', `data-id="${run.id}"`)}${btn('lab-reject', '退回修改', '', '', `data-id="${run.id}"`)}</div>`;
    } else actions = `${btn('lab-submit', '提交人工审核', 'shield', 'primary', `data-id="${run?.id || ''}" ${fresh && passed && ['evaluated', 'rejected'].includes(run?.status) ? '' : 'disabled'}`)}<p class="lab-note">${!run ? '先准备候选索引并运行评测。' : !fresh ? '配置、知识、授权或金标已变化，必须重新评测。' : !passed ? '全部门槛与人工用例通过后，才能提交审核。' : '试运行阶段由 21v-admin 统一确认。'}</p>`;
    const releaseInfo = `<div class="lab-callout">${d.scope === 'joint' ? '联合基线：' + esc(labBaselineLabel(d.scope)) + '。联合候选使用本实验的统一参数，仅生成报告。' : '这里发布的是 RAG 配置方案。'}文档内容仍在“审核与发布”中独立确认。静态 Ask AI 继续使用本地样例逻辑。</div><div class="lab-release-current"><span>${d.scope === 'joint' ? '各库生效配置' : '当前生效 · ' + esc(C.labScopes[d.scope])}</span><h3>${d.scope === 'joint' ? esc(labBaselineLabel(d.scope)) : esc(baseline.name)}</h3><p>${d.scope === 'joint' ? '各来源方案版本保存在联合运行快照中。' : (baseline.scope !== d.scope ? '继承全局默认 · ' : '') + 'R' + baseline.number + ' · ' + date(baseline.at)}</p>${baseline.previousId && d.scope !== 'joint' ? btn('lab-rollback', '回退上一方案', 'history', '', `data-id="${baseline.previousId}" data-scope="${d.scope}"`) : ''}</div>`;
    return `<div class="lab-columns">${labPanel('发布前，保留一次人工判断', message, `${run ? `<div class="lab-snapshot"><p>实验：${esc(run.snapshot.name)} · Run ${run.number}</p><p>知识：${run.snapshot.sources.map(s => esc(s.name) + ' v' + s.version).join(' / ')}</p><p>金标：v${run.snapshot.datasetVersion} · Prompt：v${run.snapshot.config.promptVersion}</p><p>索引：<code>${esc(run.indexId)}</code></p></div>` : ''}${actions}`)}${labPanel('方案版本与回退', '发布切换配置，回退也创建新版本，历史仍然保留。', releaseInfo)}</div><div class="lab-section-heading"><div><h3>实验记录</h3><p>保留最近 10 次运行的入口，快照不会随草稿变化。</p></div></div><div class="panel"><div class="table-scroll"><table><thead><tr><th>实验 / 运行</th><th>范围</th><th>结果</th><th>流程状态</th><th></th></tr></thead><tbody>${l.runs.slice(0, 10).map(r => `<tr><td><strong>${esc(r.snapshot.name)}</strong><small>Run ${r.number} · ${date(r.at)}</small></td><td>${esc(C.labScopes[r.snapshot.scope])}</td><td>${badge(C.labPassed(r) ? '通过' : '未达标', C.labPassed(r) ? 'green' : 'orange')}</td><td>${({ evaluated: '已评测', pending: '待审核', published: '已发布', rejected: '已退回' })[r.status]}</td><td>${btn('lab-run-detail', '快照', '', 'quiet', `data-id="${r.id}"`)}</td></tr>`).join('') || '<tr><td colspan="5">还没有实验记录。</td></tr>'}</tbody></table></div></div><details class="lab-details"><summary>查看方案发布历史（${l.releases.length}）</summary>${l.releases.map(r => `<div class="lab-history-row"><strong>${esc(C.labScopes[r.scope])} · R${r.number} · ${esc(r.name)}</strong><p>${esc(r.note || '初始化基线')} · ${esc(r.actor)} · ${date(r.at)}</p></div>`).join('')}</details>`;
  }
  function ragPage() {
    const l = C.labState(state), d = l.draft, tab = l.tab || 'ingest', run = labRun();
    const stages = { ingest: labIngest, retrieve: labRetrieval, context: labContext, generate: labGeneration, evaluate: labEvaluation, release: labReleasePage };
    const indexReady = d.index?.key === C.labIndexKey(state);
    const status = run?.status === 'published' ? '已发布' : !run ? '配置中' : !C.labFresh(state, run) ? '待重新评测' : run.status === 'pending' ? '待人工审核' : C.labPassed(run) ? '评测通过' : '待调优';
    return `<div class="rag-lab">${pageHead('RAG 实验室', '从一份知识，到一个经得起验证的答案。', btn('lab-new', '新建实验', 'plus'), 'EXPERIMENT WORKSPACE')}<div class="lab-experiment"><div><div class="lab-overline">当前实验 ${badge(status, status === '评测通过' || status === '已发布' ? 'green' : 'blue')}</div><h2>${esc(d.name)}</h2><p>${esc(C.labScopes[d.scope])} <span>·</span> 基线 ${labBaselineLabel(d.scope)} <span>·</span> 金标 v${l.datasetVersion}</p></div><div class="lab-experiment-actions">${badge('本地模拟 · 非真实 benchmark', 'orange')}${btn('lab-recommend', '载入推荐调优', 'sliders', '')}<small id="lab-save-state">草稿自动保存 · ${indexReady ? '候选索引已就绪' : '待准备候选索引'}</small></div></div><div class="lab-tabs" role="tablist" aria-label="RAG 实验阶段">${labTabs.map(([key, label], i) => `<button type="button" role="tab" id="lab-tab-${key}" aria-selected="${tab === key}" aria-controls="lab-stage-${key}" tabindex="${tab === key ? 0 : -1}" data-action="lab-tab" data-tab="${key}"><span>${String(i + 1).padStart(2, '0')}</span>${label}</button>`).join('')}</div><section class="lab-stage" role="tabpanel" id="lab-stage-${tab}" aria-labelledby="lab-tab-${tab}" tabindex="0">${stages[tab]()}</section>${labTabs.filter(([key]) => key !== tab).map(([key]) => `<section hidden role="tabpanel" id="lab-stage-${key}" aria-labelledby="lab-tab-${key}"></section>`).join('')}<div class="lab-stage-footer"><span>全局默认 → 知识库独立调优 → 跨库联合验收</span>${tab !== 'release' ? btn('lab-next', '下一阶段', 'arrow', 'quiet') : btn('lab-tab', '返回入库与索引', 'arrow', 'quiet', 'data-tab="ingest"')}</div></div>`;
  }
  function labRefresh(focusId) { save(); render(); if (focusId) document.getElementById(focusId)?.focus({ preventScroll: true }); }
  function labSetTab(tab) {
    if (!labTabs.some(([key]) => key === tab)) return;
    C.labState(state).tab = tab; labRefresh('lab-tab-' + tab);
    const selected = $('#lab-tab-' + tab), tabs = $('.lab-tabs');
    tabs.scrollLeft = selected.offsetLeft - Math.max(0, (tabs.clientWidth - selected.offsetWidth) / 2);
  }
  function labNewDialog() {
    const l = C.labState(state);
    openDialog('新建实验', '从当前生效方案复制候选配置，已运行的实验快照会保留。', `<form id="lab-new-form" class="rag-lab"><div class="form-field"><label for="lab-name">实验名称</label><input id="lab-name" maxlength="80" value="${esc(l.draft.name)}" required></div><div class="form-field"><label for="lab-scope">实验范围</label><select id="lab-scope">${Object.entries(C.labScopes).map(([key, name]) => `<option value="${key}" ${key === l.draft.scope ? 'selected' : ''}>${name}</option>`).join('')}</select></div><p class="lab-note">知识库优先使用自己的发布方案，否则继承全局默认。联合实验用于验证两个项目库；个人知识与 Memory 不参与。</p><div class="form-footer"><button class="btn primary" type="submit">创建候选实验</button></div></form>`);
  }
  function labCaseDialog(caseId) {
    const run = labRun(), fresh = C.labFresh(state, run), c = (fresh ? run.result.cases : C.labState(state).cases).find(c => c.id === caseId);
    if (!c) throw Error('用例不存在。');
    const body = `<div class="rag-lab"><div class="lab-query">${esc(c.question)}</div><div class="lab-evidence"><h4>期望答案 / 行为</h4><p>${esc(c.expected)}</p><h4>证据定位</h4><p>${esc(c.source)}</p></div>${fresh ? `<div class="lab-callout"><strong>${c.pass === null ? '待人工判定' : c.pass ? '通过' : '未通过'}</strong><p>${esc(c.reason)}</p></div>` : ''}<p class="lab-note">${c.manual ? '自定义问题在本 Demo 中不生成模型答案，按钮仅演示人工复核流程；真实评测需结合实际答案与证据逐条判定。' : '这是预置示例金标与规则判定，用于会议讨论，并非已验证的生产测试集。'}</p></div>`;
    const footer = fresh && c.manual && run.status === 'evaluated' ? btn('lab-judge', '判定未通过', '', '', `data-id="${c.id}" data-pass="false"`) + btn('lab-judge', '人工判定通过', 'check', 'primary', `data-id="${c.id}" data-pass="true"`) : fresh && !c.pass ? btn('lab-fix-stage', '前往对应阶段调整', 'sliders', 'primary', `data-tab="${c.stage}"`) : btn('close-dialog', '关闭');
    openDialog('金标用例', `${c.manual ? '人工标注' : '预置示例'} · ${c.actor}`, body, footer);
  }
  function labCaseForm() {
    const feedback = state.feedback.filter(f => f.status === 'evaluation' && f.sources.length && f.sources.every(s => ['acn', 'playbook'].includes(s.kbId)));
    openDialog('人工构建金标', '人工筛选、脱敏并确认依据后，才加入共享评测集。', `<form id="lab-case-form" class="rag-lab"><div class="form-field"><label for="lab-feedback">从反馈候选开始（可选）</label><select id="lab-feedback"><option value="">手工填写${feedback.length ? '' : ' · 暂无项目知识反馈候选'}</option>${feedback.map(f => `<option value="${f.id}">${esc(f.question.slice(0, 65))}</option>`).join('')}</select></div><div class="form-field"><label for="lab-case-question">测试问题</label><input id="lab-case-question" maxlength="300" required></div><div class="form-field"><label for="lab-case-scope">知识范围</label><select id="lab-case-scope"><option value="acn">ACN 产品价格</option><option value="playbook">Playbook 操作手册</option><option value="joint">跨知识库</option></select></div><div class="form-field"><label for="lab-case-expected">期望答案要点 / 应拒答行为</label><textarea id="lab-case-expected" maxlength="2000" required></textarea></div><div class="form-field"><label for="lab-case-source">证据定位（文档、版本、章节或 JSON 路径）</label><input id="lab-case-source" maxlength="500" required></div><label class="lab-consent"><input type="checkbox" required> 我已核对依据并移除个人信息与私人知识</label><div class="form-footer"><button class="btn primary" type="submit">确认标注并创建金标版本</button></div></form>`);
  }
  function costPage() {
    const sessionCost = state.usage.reduce((n, u) => n + u.cost, 0), baseline = 112.86, used = baseline + sessionCost;
    const percent = Math.min(100, used / state.settings.budget * 100);
    return pageHead('用量与费用', '看见每次知识处理与问答的成本。', btn('export-usage', '导出用量明细', 'download'), 'USAGE & COST') + `<div class="stats">${stat('本月演示费用', '¥' + used.toFixed(2), '演示基线 ¥112.86 + 本次估算', 'chart')}${stat('本次演示问答', state.usage.filter(u => u.type === '问答生成').length, '未实际调用计费服务', 'chat')}${stat('剩余演示预算', '¥' + Math.max(0, state.settings.budget - used).toFixed(2), '月度预算 ¥' + state.settings.budget, 'shield')}</div><div class="two-col"><div class="panel panel-pad"><div class="section-row" style="margin-top:0"><h2>用量趋势</h2>${badge('过去 7 天 · 模拟数据')}</div><svg class="chart" viewBox="0 0 620 200" role="img" aria-label="七天演示费用趋势，从 9.2 元增加到 21.36 元"><defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#739ad1" stop-opacity=".18"/><stop offset="1" stop-color="#739ad1" stop-opacity="0"/></linearGradient></defs><path d="M20 30H600M20 80H600M20 130H600M20 180H600" stroke="#edf1f7" stroke-dasharray="3 5"/><path d="M20 150C65 150 72 98 115 107S165 145 210 123 266 83 307 94 360 57 405 66 453 103 503 64 565 22 600 35V185H20Z" fill="url(#area)"/><path d="M20 150C65 150 72 98 115 107S165 145 210 123 266 83 307 94 360 57 405 66 453 103 503 64 565 22 600 35" fill="none" stroke="#81a2d3" stroke-width="2.4"/><circle cx="600" cy="35" r="5" fill="white" stroke="#81a2d3" stroke-width="2.4"/></svg><div class="chart-labels"><span>周一</span><span>周二</span><span>周三</span><span>周四</span><span>周五</span><span>周六</span><span>周日</span></div></div><div class="panel panel-pad"><div class="section-row" style="margin-top:0"><h2>预算与分布</h2>${btn('budget', '调整', 'sliders', 'quiet')}</div><div class="cost-row"><span>月度预算使用</span><strong>${percent.toFixed(1)}%</strong></div><div class="progress-track"><div class="progress-fill" style="width:${percent}%"></div></div><div class="divider"></div>${[['回答生成', '¥86.40'], ['文档向量化', '¥18.62'], ['结果重排序', '¥7.84'], ['本次演示估算', '¥' + sessionCost.toFixed(4)]].map(([a, b]) => `<div class="cost-row"><span>${a}</span><strong>${b}</strong></div>`).join('')}<p class="note">预算用于展示管理流程，统计和单价均为演示值。</p></div></div><div class="section-row"><h2>本次演示用量</h2><p>按操作、用户和知识范围留痕</p></div><div class="panel"><div class="table-scroll"><table><thead><tr><th>操作</th><th>用户</th><th>Token 估算</th><th>费用估算</th><th>时间</th></tr></thead><tbody>${state.usage.length ? state.usage.slice().reverse().map(u => `<tr><td><strong>${u.type}</strong><small>${esc(u.label)}</small></td><td>${u.actor}</td><td>${u.tokens.toLocaleString()}</td><td>¥${u.cost.toFixed(4)}</td><td class="small">${date(u.at)}</td></tr>`).join('') : '<tr><td colspan="5" class="muted">进行一次问答或知识发布后，这里会出现本次操作记录。</td></tr>'}</tbody></table></div></div>`;
  }
  function feedbackPage() {
    const items = state.feedback.filter(f => admin() || f.actor === current().id);
    return pageHead(admin() ? '反馈中心' : '我的反馈', '让每一条反馈，成为知识改进的起点。', badge(items.length + ' 条反馈'), 'FEEDBACK LOOP') + `<div class="panel">${items.length ? items.slice().reverse().map(f => `<div class="feedback-item"><div class="feedback-meta"><div class="table-person">${avatar(C.user(state, f.actor))}<strong>${f.actor}</strong>${badge(f.category, f.category === '有帮助' ? 'green' : 'orange')}</div>${badge(f.status === 'resolved' ? '已处理' : f.status === 'evaluation' ? '待加入评估集' : f.status === 'gold' ? '已纳入金标 v' + f.goldVersion : '待处理', f.status === 'resolved' ? 'green' : '')}</div><p><strong>问题：</strong>${esc(f.question)}</p><p>${esc(f.comment || '这条回答对我有帮助。')}</p><div class="split-meta"><span>${date(f.at)}</span><span>${f.sources.map(s => esc(s.title) + ' v' + s.version).join(' · ') || '无引用来源'}</span></div>${admin() ? `<div style="display:flex;gap:7px;margin-top:14px">${btn('feedback-resolve', '标记已处理', 'check', 'quiet', `data-id="${f.id}"`)}${btn('feedback-evaluate', '标记为评估候选', 'sliders', 'quiet', `data-id="${f.id}"`)}</div>` : ''}</div>`).join('') : empty('反馈，让答案更进一步', '在回答下方选择“有帮助”或反馈问题，会保留问题与对应的来源版本。', btn('new-chat', '开始一次问答', 'spark', '', ''), 'feedback')}</div>`;
  }
  let renderedSidebar = '', sidebarOwner = null;
  function render() {
    if (!admin() && ['permissions', 'models', 'rag', 'cost'].includes(ui.route)) ui.route = 'chat';
    const feed = $('.chat-conversation'), previousId = feed?.dataset.conversation, previousScroll = feed?.scrollTop;
    const pages = { chat: chatPage, knowledge: knowledgePage, history: historyPage, reviews: reviewsPage, permissions: permissionsPage, models: modelsPage, rag: ragPage, cost: costPage, feedback: feedbackPage };
    const sidebarHTML = sidebar(), shellHTML = `<div class="shell ${ui.route === 'chat' ? 'chat-shell' : ''}">${topbar()}<main id="main" tabindex="-1">${(pages[ui.route] || chatPage)()}</main></div>`;
    const side = $('#sidebar');
    if (!side) $('#app').innerHTML = sidebarHTML + shellHTML;
    else {
      // Keep the sidebar mounted: replacing it restarts its width transition after streaming.
      if (sidebarHTML !== renderedSidebar) {
        const historyScroll = sidebarOwner === current().id ? $('.recent-conversations').scrollTop : 0;
        const navScroll = sidebarOwner === current().id ? $('.sidebar-scroll').scrollTop : 0;
        const template = document.createElement('template'); template.innerHTML = sidebarHTML;
        side.replaceChildren(...template.content.querySelector('#sidebar').childNodes);
        $('.recent-conversations').scrollTop = historyScroll; $('.sidebar-scroll').scrollTop = navScroll;
      }
      $('.shell').outerHTML = shellHTML;
    }
    renderedSidebar = sidebarHTML; sidebarOwner = current().id;
    if (previousId && $('.chat-conversation')?.dataset.conversation === previousId) $('.chat-conversation').scrollTop = previousScroll;
    syncSidebar();
    if (ui.scope !== 'all' && !permittedKBs().some(k => k.id === ui.scope)) ui.scope = 'all';
  }
  const dialog = $('#dialog'); let dialogReturnFocus = null;
  function openDialog(title, subtitle, body, footer = '', wide = false, tabs = '') {
    if (!dialog.open) dialogReturnFocus = document.activeElement;
    dialog.className = wide ? 'wide' : '';
    dialog.innerHTML = `<div class="dialog-head"><div><h2 id="dialog-title">${esc(title)}</h2>${subtitle ? `<p>${esc(subtitle)}</p>` : ''}</div><button class="icon-button" aria-label="关闭对话框" data-action="close-dialog">${icon('close')}</button></div>${tabs}<div class="dialog-body">${body}</div>${footer ? `<div class="dialog-footer">${footer}</div>` : ''}`;
    if (!dialog.open) dialog.showModal();
  }
  function closeDialog() { dialog.close(); ui.doc = null; if (dialogReturnFocus?.isConnected) dialogReturnFocus.focus(); }
  dialog.addEventListener('click', e => { if (e.target === dialog && !e.target.closest('.dialog-body') && (e.clientX < dialog.getBoundingClientRect().left || e.clientX > dialog.getBoundingClientRect().right || e.clientY < dialog.getBoundingClientRect().top || e.clientY > dialog.getBoundingClientRect().bottom)) closeDialog(); });
  dialog.addEventListener('cancel', () => { ui.doc = null; });
  function previewDocument(doc, revision) {
    if (doc.id === 'mysql') {
      const rows = C.tables(revision.raw, ui.region);
      return `<div class="preview-title"><h3>Azure Database for MySQL</h3><select id="preview-region" class="scope-select" aria-label="价格预览区域">${['中国北部 3', '中国东部 3', '中国东部 2', '中国北部 2'].map(r => `<option ${r === ui.region ? 'selected' : ''}>${r}</option>`).join('')}</select></div>${revision.demo ? `<div class="info-strip warning" style="margin:0 0 18px">${icon('info')}当前内容包含演示修订，修改后的数字不代表实际报价。</div>` : ''}<div class="prose">${rows.map((table, index) => `<h3>${['可突发计算', '常规用途 · D 系列', '业务关键 · E 系列', '存储', '其他 IOPS', '付费 IO', '备份存储', '扩展支持'][index] || '计费说明'}</h3><div class="table-scroll"><table><thead><tr>${table[0].map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${table.slice(1).map(row => `<tr>${row.map(c => `<td>${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`).join('')}</div><p class="note">表格提取自原始 JSON 的区域 content 字段。完整页面说明请查看“原始内容”。</p>`;
    }
    return doc.format === 'md' ? `<div class="prose">${markdown(revision.raw)}</div>` : `<pre class="raw-code">${esc(JSON.stringify(JSON.parse(revision.raw), null, 2))}</pre>`;
  }
  function diffView(doc, revision) {
    const before = doc.versions.find(v => v.id === revision.parent)?.raw || '';
    const sample = DATA[doc.id];
    let rawDiff = sample && before === sample.raw && revision.raw === sample.updated ? sample.diff : sample && before === sample.raw && revision.raw === sample.conflict ? sample.conflictDiff : C.textDiff(before, revision.raw, doc.name);
    const actualGit = sample && before === sample.raw && [sample.updated, sample.conflict].includes(revision.raw);
    let readable = '';
    if (doc.id === 'mysql' && sample && before === sample.raw && [sample.updated, sample.conflict].includes(revision.raw)) {
      const get = raw => C.tables(raw, '中国北部 3').flat().find(r => r[0] === 'B1MS')?.[3] || '无';
      readable = `<div class="change-card"><header>中国北部 3 / 可突发计算 / B1MS / 现用现付</header><div class="change-columns"><div><small>− 修订前</small><strong>${esc(get(before))}</strong><p>原生效版本中的价格</p></div><div><small>+ 修订后</small><strong>${esc(get(revision.raw))}</strong><p>待审核的演示价格</p></div></div></div>`;
    } else if (doc.id === 'playbook' && sample && ((before === sample.raw && [sample.updated, sample.conflict].includes(revision.raw)) || ([sample.updated, sample.conflict].includes(before) && revision.raw === sample.raw))) {
      const old = C.sections(before).find(s => s.title.startsWith('16.')), next = C.sections(revision.raw).find(s => s.title.startsWith('16.'));
      readable = `<div class="change-card"><header>第 16 节 / 内部执行检查清单</header><div class="change-columns"><div><small>− 修订前</small><div class="prose">${old ? markdown(old.text) : '<p>当前版本未包含内部执行检查清单。</p>'}</div></div><div><small>+ 修订后</small><div class="prose">${next ? markdown(next.text) : '<p>移除演示检查清单，恢复原始内容。</p>'}</div></div></div></div>`;
    }
    const conflictCount = C.conflicts(doc, revision).length;
    return `${revision.demo ? `<div class="info-strip warning">${icon('info')}此修订为会议演示构造，原始样例保持不变。</div>` : ''}${conflictCount && revision.status === 'pending' ? `<div class="info-strip warning">${icon('branch')}还有 ${conflictCount} 份并行待审修订。发布前需要明确保留版本，其他修订将退回。</div>` : ''}<div class="diff-top"><div class="segmented"><button data-action="diff-mode" data-mode="readable" class="${ui.diffMode === 'readable' ? 'active' : ''}">可读对比</button><button data-action="diff-mode" data-mode="raw" class="${ui.diffMode === 'raw' ? 'active' : ''}">原始差异</button></div><small>${actualGit ? 'git diff --no-index · 原文逐行差异' : '上传 / 回退内容 · 文本差异'}</small></div>${ui.diffMode === 'readable' && readable ? readable : `<div class="diff-code">${rawDiff.split('\n').map(line => `<span class="diff-line ${line.startsWith('+') && !line.startsWith('+++') ? 'add' : line.startsWith('-') && !line.startsWith('---') ? 'del' : line.startsWith('@@') ? 'hunk' : ''}">${esc(line) || ' '}</span>`).join('')}</div>`}<p class="note">修订基线：${revision.parent ? 'v' + doc.versions.find(v => v.id === revision.parent)?.number : '新文档'} → v${revision.number}。${revision.label ? esc(revision.label) + '。' : ''}审核记录绑定此具体修订。</p>`;
  }
  function documentDialog(docId, versionId, tab = 'preview') {
    const doc = state.documents.find(d => d.id === docId);
    if (!doc || !C.can(state, current().id, doc.kbId)) throw Error('当前身份无权查看此资料。');
    const revision = doc.versions.find(v => v.id === versionId) || C.active(doc) || doc.versions.at(-1);
    ui.doc = { id: docId, versionId: revision.id, tab };
    const k = C.kb(state, doc.kbId), mayPublish = C.can(state, current().id, doc.kbId, 'publish');
    let body = `<div class="preview-title"><div style="display:flex;gap:7px;align-items:center">${badge('文档 v' + revision.number, 'blue')}${revisionBadge(revision)}${badge('知识库 R' + k.release)}</div><span class="small muted">${esc(revision.author)} · ${date(revision.at)}</span></div>`;
    if (tab === 'preview') body += previewDocument(doc, revision);
    else if (tab === 'raw') body += `<pre class="raw-code">${esc(revision.raw)}</pre>${DATA[doc.id] ? `<p class="note mono">原始样例 SHA-256：${DATA[doc.id].sha256}</p>` : ''}`;
    else if (tab === 'diff') body += revision.number === 1 && DATA[doc.id] ? empty('这是原始样例版本', '创建一次演示修订后，可以在这里查看原文差异。', '', 'branch') : diffView(doc, revision);
    else body += `<div class="table-scroll"><table><thead><tr><th>版本</th><th>修订内容</th><th>状态</th><th>操作</th></tr></thead><tbody>${doc.versions.slice().reverse().map(v => `<tr><td><strong>v${v.number}</strong><small>${esc(v.author)}</small></td><td>${esc(v.label)}<small>${date(v.at)}</small></td><td>${revisionBadge(v)}</td><td style="white-space:nowrap">${btn('version-view', '查看', '', 'quiet', `data-version="${v.id}"`)}${mayPublish && v.status === 'archived' && v.raw !== C.active(doc)?.raw ? btn('rollback', '恢复此内容', 'history', 'quiet', `data-doc="${doc.id}" data-version="${v.id}"`) : ''}</td></tr>`).join('')}</tbody></table></div><div class="info-strip">${icon('history')}回退会创建新的修订和发布记录，已发生的历史不会被抹去。</div>`;
    let footer = `<span class="left-note">${esc(k.name)} · ${doc.format.toUpperCase()}</span>`;
    if (revision.status === 'pending' && mayPublish) {
      if (revision.indexStatus === 'failed') body += `<div class="info-strip warning">${icon('info')}上次索引准备失败。原生效版本未改变，可重试发布。</div>`;
      footer += btn('reject', '退回', '', '', `data-doc="${doc.id}" data-version="${revision.id}"`);
      if (C.conflicts(doc, revision).length) footer += btn('resolve', '采用此修订，退回其他', 'branch', 'primary', `data-doc="${doc.id}" data-version="${revision.id}"`);
      else footer += btn('fail-publish', '模拟索引失败', '', 'quiet', `data-doc="${doc.id}" data-version="${revision.id}"`) + btn('publish', revision.indexStatus === 'failed' ? '重试并发布' : '确认发布', 'check', 'primary', `data-doc="${doc.id}" data-version="${revision.id}"`);
    } else if (revision.status === 'pending') footer += badge('等待 21v-admin 确认', 'orange');
    else {
      if (DATA[doc.id] && C.can(state, current().id, doc.kbId, 'upload')) footer += btn('stage', '创建演示修订', 'branch', '', `data-doc="${doc.id}"`);
      if (mayPublish && doc.activeId) footer += btn('withdraw', '撤回文档', '', 'quiet', `data-doc="${doc.id}"`);
      footer += btn('download-doc', '下载原文', 'download', '', `data-doc="${doc.id}" data-version="${revision.id}"`);
    }
    if (revision.status === 'pending' && DATA[doc.id] && C.can(state, current().id, doc.kbId, 'upload') && !C.conflicts(doc, revision).length) footer = btn('stage-conflict', '创建并行修订', 'branch', 'quiet', `data-doc="${doc.id}"`) + footer;
    const tabs = `<div class="dialog-tabs">${[['preview', '内容预览'], ['diff', '差异对比'], ['history', '版本与发布'], ['raw', '原始内容']].map(([key, title]) => `<button data-action="doc-tab" data-tab="${key}" class="${key === tab ? 'active' : ''}">${title}</button>`).join('')}</div>`;
    openDialog(doc.title, doc.name, body, footer, true, tabs);
  }
  function uploadDialog(preselected) {
    const choices = permittedKBs().filter(k => C.can(state, current().id, k.id, 'upload'));
    if (!choices.length) throw Error('当前没有可上传的知识库。');
    const defaultKB = choices.find(k => k.id === preselected) || choices.find(k => k.owner) || choices[0];
    openDialog('上传知识资料', '资料将保留在当前浏览器中，确认发布后才用于问答。', `<form id="upload-form"><div class="form-field"><label for="upload-kb">目标知识库</label><select id="upload-kb">${choices.map(k => `<option value="${k.id}" ${k.id === defaultKB.id ? 'selected' : ''}>${k.name} · ${k.owner ? '仅自己可见' : k.group}</option>`).join('')}</select><span class="hint" id="upload-visibility">${defaultKB.owner ? '个人空间，仅你自己可见，由你确认发布。' : '项目资料按组授权，提交后由 21v-admin 审核。'}</span></div><div class="drop-zone" id="drop-zone">${icon('upload')}<strong>选择文件，或拖放到这里</strong><p>Markdown / JSON · 单文件不超过 250 KB</p><input type="file" id="upload-file" aria-label="选择知识文档" accept=".md,.json"></div><div class="form-field"><label for="upload-name">文件名</label><input id="upload-name" placeholder="例如：我的项目笔记.md" maxlength="180" required></div><div class="form-field"><label for="upload-content">内容预览 / 直接粘贴</label><textarea id="upload-content" placeholder="# 我的项目笔记&#10;&#10;记录你希望被检索的知识…" required></textarea><span class="hint">在同一知识库上传同名文件，会创建新修订。</span></div><div class="form-footer">${btn('close-dialog', '取消')}<button class="btn primary" type="submit">${icon('branch')}提交修订</button></div></form>`);
  }
  function setIdentityMenu(open) {
    ui.identityMenu = open; $('.identity-menu')?.remove(); $('.topbar').outerHTML = topbar(); syncSidebar();
  }
  function memoryDialog() {
    const memory = C.memoryState(state, current().id);
    const entries = memory.items.map(m => `<div class="memory-item"><div><strong>${C.memoryLabels[m.key]}</strong><p>${esc(m.value)}</p><small>${date(m.updatedAt)} 更新</small></div>${btn('forget-memory', '删除', '', 'quiet', `data-id="${esc(m.id)}" aria-label="删除${C.memoryLabels[m.key]}记忆"`)}</div>`).join('');
    const controls = `<div class="form-field inline"><div><label for="memory-enabled">使用个人记忆</label><span class="hint">在聊天中自然记住你的背景和偏好，供后续会话参考。</span></div><input class="switch" type="checkbox" id="memory-enabled" ${memory.enabled ? 'checked' : ''}></div>`;
    const status = memory.enabled ? '已开启。记忆在后台生效，不会进入项目知识库或共享反馈。' : '已暂停。停止新增和使用记忆，已保存内容仍可查看、删除。';
    openDialog('个人记忆', current().id + ' · 只在自己的会话中使用', controls + `<p class="small muted">${status}</p><div class="memory-list">${entries || empty('还没有个人记忆', '例如：我主要负责 MySQL 运维，以后回答简洁一点。', '', 'lock')}</div><p class="note">修改信息时，在聊天中重新告诉我即可。删除记忆不会删除已有会话；不会从旧会话自动重新学习已删除的内容。</p>`, (memory.items.length ? btn('clear-memory', '清空记忆', '', 'danger') : '') + btn('close-dialog', '完成', '', 'primary'));
  }
  function guideDialog() {
    const steps = [
      ['用真实资料回答问题', '保持 Laoyang 身份，提问“Azure 预留交换策略有什么变化？”，打开引用，核对章节与文档 v1。', 'guide-chat', '开始问答'],
      ['验证项目权限隔离', 'Laoyang 查询 MySQL 价格会被限制；切换为 Laojiu，即可查询中国北部 3 的 B1MS 价格。', 'guide-permission', '体验权限边界'],
      ['提交并对比一次变化', '在操作手册或 MySQL 文档中创建演示修订，查看可读对比与真实 Git 原文差异。', 'guide-revision', '创建 Playbook 修订'],
      ['审核、冲突与失败恢复', '切换为 21v-admin，打开审核与发布。可创建并行修订，选择保留版本；模拟索引失败后重试发布。', 'guide-review', '进入管理员审核'],
      ['验证新答案，再回退', '回到 Laoyang，询问内部检查清单。发布后会引用新版本；管理员可在版本记录恢复原始内容，生成新的发布。', 'guide-reask', '验证更新后的回答'],
      ['查看管理与反馈闭环', '管理员可调整组授权、模型配置、模拟 RAG 评估与预算。回答下方可提交反馈，费用页记录本次演示用量。', 'guide-admin', '查看管理能力']
    ];
    openDialog('一次知识更新的完整旅程', '建议展示 10–15 分钟 · 随时切换身份 · 一键恢复初始状态', `<div class="steps">${steps.map(([title, text, action, label]) => `<div class="step"><div><strong>${title}</strong><p>${text}</p>${btn(action, label, 'arrow', 'quiet')}</div></div>`).join('')}</div><div class="info-strip">${icon('info')}演示修订和运行指标均有明确标注。原始样例不被修改，无外部模型调用。</div>`, btn('close-dialog', '开始探索', 'spark', 'primary'));
  }
  function stopStream() {
    ui.streamToken++; ui.streaming = false;
    const c = state.conversations.find(c => c.id === ui.conversation);
    if (c?.messages.at(-1)?.streaming) { c.messages.at(-1).streaming = false; c.messages.at(-1).body += '\n\n> 本次生成已停止。'; }
    save();
  }
  function switchUser(userId) {
    if (!C.user(state, userId)) throw Error('用户不存在。');
    stopStream(); if (dialog.open) closeDialog();
    state.currentUser = userId; ui = { ...ui, route: 'chat', scope: 'all', conversation: null, kbId: null, search: '', identityMenu: false, sidebarOpen: false };
    save(); render(); toast('已切换为 ' + userId + '，知识范围已更新。');
  }
  function navigate(route) {
    if (ui.streaming) stopStream(); ui.route = route; ui.kbId = null; ui.search = ''; ui.identityMenu = false; ui.sidebarOpen = false; render(); window.scrollTo({ top: 0 });
  }
  function addUsage(type, label, actor, tokens) {
    state.usage.push({ id: C.id('usage'), type, label, actor, tokens, cost: Number((tokens * 0.000006).toFixed(4)), at: C.now() });
  }
  async function ask(query) {
    query = String(query).trim(); if (!query || ui.streaming) return;
    if (!state.settings.modelsEnabled) { toast('问答服务已在模型管理中停用，请联系管理员启用。', true); return; }
    if (query.length > 1500) { toast('演示问题请控制在 1500 字符以内。', true); return; }
    const actor = current().id, chosen = ui.scope === 'all' ? permittedKBs().map(k => k.id) : [ui.scope];
    let conversation = state.conversations.find(c => c.id === ui.conversation && c.owner === actor);
    if (!conversation) { conversation = { id: C.id('chat'), owner: actor, title: query.slice(0, 42), at: C.now(), messages: [] }; state.conversations.push(conversation); ui.conversation = conversation.id; }
    conversation.messages.push({ role: 'user', body: query, at: C.now() });
    const result = C.answer(state, actor, query, chosen);
    const message = { ...result, body: '', role: 'assistant', at: C.now(), streaming: true };
    conversation.messages.push(message); ui.route = 'chat'; ui.streaming = true; const token = ++ui.streamToken;
    const messageIndex = conversation.messages.length - 1;
    render(); save(); $('.chat-conversation').scrollTop = $('.chat-conversation').scrollHeight;
    // Local streaming simulation; only the active answer changes between chunks.
    const characters = Array.from(result.body), step = Math.max(2, Math.ceil(characters.length / 180));
    await new Promise(resolve => setTimeout(resolve, 180));
    for (let count = step; count < characters.length + step; count += step) {
      await new Promise(resolve => setTimeout(resolve, 42));
      if (ui.streamToken !== token || current().id !== actor) return;
      const feed = $('.chat-conversation'), follow = feed && feed.scrollHeight - feed.scrollTop - feed.clientHeight < 72;
      message.body = characters.slice(0, count).join('');
      const content = $(`[data-message-index="${messageIndex}"] .stream-content`);
      if (content) content.innerHTML = markdown(message.body);
      if (follow) feed.scrollTop = feed.scrollHeight;
    }
    if (ui.streamToken !== token) return;
    message.body = result.body; message.streaming = false; ui.streaming = false;
    addUsage('问答生成', result.sources.map(s => s.title).join(' / ') || (result.kind === 'memory' ? '个人上下文' : result.kind === 'restricted' ? '权限范围检查' : '问题澄清'), actor, Math.ceil((query.length + result.body.length) * 1.7));
    const feed = $('.chat-conversation'), follow = feed && feed.scrollHeight - feed.scrollTop - feed.clientHeight < 72;
    save(); render();
    if (follow) $('.chat-conversation').scrollTop = $('.chat-conversation').scrollHeight;
    if (!dialog.open) $('#question')?.focus({ preventScroll: true });
  }
  function download(content, name, type = 'text/plain;charset=utf-8') {
    const a = document.createElement('a'), url = URL.createObjectURL(new Blob([content], { type })); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function selectedAnswer(index, forFeedback = false) {
    const conversation = state.conversations.find(c => c.id === ui.conversation && c.owner === current().id);
    const m = conversation?.messages[Number(index)];
    if (!m || m.role !== 'assistant' || !m.sources.every(s => C.can(state, current().id, s.kbId))) throw Error('当前身份无法访问这条回答。');
    if (forFeedback && m.kind === 'memory') throw Error('个人上下文不提交到共享反馈。');
    return { m, question: conversation.messages[Number(index) - 1]?.body || '', conversation };
  }
  function feedbackDialog(index) {
    const { question } = selectedAnswer(index, true);
    openDialog('反馈这条回答', '反馈将连同问题和引用版本提交给演示管理员。', `<form id="feedback-form" data-index="${index}"><div class="info-strip" style="margin:0 0 20px">${icon('chat')}${esc(question)}</div><div class="form-field"><label for="feedback-category">问题类型</label><select id="feedback-category"><option>内容过时</option><option>回答不准确</option><option>引用不相关</option><option>权限或访问问题</option><option>其他建议</option></select></div><div class="form-field"><label for="feedback-comment">补充说明</label><textarea id="feedback-comment" placeholder="你期待怎样的回答，或者哪一段需要更正？" maxlength="1000" required></textarea></div><div class="form-footer">${btn('close-dialog', '取消')}<button type="submit" class="btn primary">提交反馈</button></div></form>`);
  }
  async function handleAction(target) {
    const a = target.dataset.action, d = target.dataset;
    if (a.startsWith('lab-')) {
      if (!admin()) throw Error('RAG 实验需要管理员权限。');
      const l = C.labState(state), run = labRun();
      if (a === 'lab-tab') return labSetTab(d.tab);
      if (a === 'lab-next') return labSetTab(labTabs[Math.min(5, labTabs.findIndex(([key]) => key === (l.tab || 'ingest')) + 1)][0]);
      if (a === 'lab-new') return labNewDialog();
      if (a === 'lab-index') { C.labPrepare(state, current().id); labRefresh(); toast('候选索引准备完成（模拟），可以继续召回与评测。'); return; }
      if (a === 'lab-recommend') {
        C.labUpdate(state, current().id, { ...C.labDefaults, regionFilter: true, rerank: true, citations: true, promptVersion: l.draft.config.promptVersion + 1, prompt: '仅使用已授权、已发布的证据。\n先核对产品、区域、规格、计费单位与政策适用时间。\n结论后附文档版本与证据定位；不同来源分别引用。\n缺少区域先澄清，缺少证据说明无法确认。\n个人 Memory 仅提供表达偏好，不能改写事实或权限。' });
        labRefresh(); toast('已载入推荐调优：业务过滤、精排与完整引用。请重新评测。'); return;
      }
      if (a === 'lab-prompt') {
        C.labUpdate(state, current().id, { promptVersion: l.draft.config.promptVersion + 1, prompt: '你是 21v 知识助手。仅使用当前用户有权限的已发布证据。\n回答顺序：结论 → 适用条件 → 来源。\n价格必须包含产品、区域、规格、币种和计费单位。\n政策必须包含适用日期、范围和例外。\n缺少证据时澄清或拒答，禁止猜测。\n每项结论附文档版本和证据定位。', citations: true, abstain: true });
        labRefresh('lab-prompt'); return;
      }
      if (a === 'lab-retrieve') { ui.labTrace = true; labRefresh(); return; }
      if (a === 'lab-generate') { ui.labAnswer = true; labRefresh(); return; }
      if (a === 'lab-evaluate') {
        C.labRun(state, current().id); l.tab = 'evaluate'; ui.labFailures = false; labRefresh('lab-tab-evaluate'); toast('模拟对照已完成，输入快照已保存。'); return;
      }
      if (a === 'lab-failures') { ui.labFailures = !ui.labFailures; labRefresh(); return; }
      if (a === 'lab-case') return labCaseDialog(d.id);
      if (a === 'lab-add-case') return labCaseForm();
      if (a === 'lab-fix-stage') { closeDialog(); return labSetTab(d.tab); }
      if (a === 'lab-judge') { C.labJudge(state, current().id, run?.id, d.id, d.pass === 'true'); closeDialog(); labRefresh(); toast('人工判定已记录。'); return; }
      if (a === 'lab-submit' || a === 'lab-publish' || a === 'lab-reject') {
        C.labTransition(state, current().id, d.id, a.replace('lab-', ''), $('#lab-review-note')?.value || '');
        labRefresh(); toast(a === 'lab-submit' ? '已提交，等待人工确认。' : a === 'lab-publish' ? '演示方案已发布，历史快照已保留。' : '已退回，可以继续调整。'); return;
      }
      if (a === 'lab-rollback') return openDialog('回退 RAG 方案', '恢复上一套配置，并创建新的发布记录。', '<p class="small">只切换演示配置方案，知识文档版本由知识库独立管理。</p>', btn('close-dialog', '取消') + btn('lab-confirm-rollback', '确认回退', 'history', 'primary', `data-id="${d.id}" data-scope="${d.scope}"`));
      if (a === 'lab-confirm-rollback') { C.labRollback(state, current().id, d.scope, d.id); closeDialog(); labRefresh(); toast('已回退上一方案并保留新发布记录。'); return; }
      if (a === 'lab-run-detail') {
        const record = l.runs.find(r => r.id === d.id); if (!record) throw Error('运行记录不存在。');
        return openDialog('实验快照 · Run ' + record.number, record.snapshot.name, `<div class="rag-lab">${labMetricTable(record)}<details class="lab-details"><summary>完整配置、知识版本与金标快照</summary><pre class="raw-code">${esc(JSON.stringify(record.snapshot, null, 2))}</pre></details></div>`, btn('lab-export', '导出本次报告', 'download', '', `data-id="${record.id}"`), true);
      }
      if (a === 'lab-export') { const record = l.runs.find(r => r.id === d.id); if (!record) throw Error('运行记录不存在。'); download(JSON.stringify({ notice: '静态 Demo 模拟评测报告，非真实 benchmark', ...record }, null, 2), 'rag-demo-run-' + record.number + '.json', 'application/json'); return; }
    }
    if (['permissions', 'models', 'rag', 'cost'].includes(d.route) && !admin()) throw Error('此页面需要管理员权限。');
    if (a === 'navigate') return navigate(d.route);
    if (a === 'sidebar-toggle') {
      if (state.settings.sidebarPinned && !matchMedia('(max-width:760px)').matches) { state.settings.sidebarPinned = false; ui.sidebarOpen = false; save(); }
      else ui.sidebarOpen = !ui.sidebarOpen;
      syncSidebar(); if (!ui.sidebarOpen) $('.sidebar-toggle').focus({ preventScroll: true }); return;
    }
    if (a === 'pin-sidebar') {
      state.settings.sidebarPinned = !state.settings.sidebarPinned; ui.sidebarOpen = !!state.settings.sidebarPinned;
      save(); syncSidebar(); if (!state.settings.sidebarPinned) $('.sidebar-toggle').focus({ preventScroll: true }); return;
    }
    if (a === 'identity') { setIdentityMenu(!ui.identityMenu); $('.identity').focus({ preventScroll: true }); return; }
    if (a === 'memory') { setIdentityMenu(false); $('.identity').focus({ preventScroll: true }); return memoryDialog(); }
    if (a === 'clear-memory') return openDialog('清空个人记忆', '只影响 ' + current().id + ' 的个人记忆。', '<p class="small muted">后续回答不再使用已保存的背景和偏好，已有会话消息仍然保留。</p>', btn('memory', '取消') + btn('confirm-clear-memory', '确认清空', '', 'danger'));
    if (a === 'forget-memory' || a === 'confirm-clear-memory') {
      if (ui.streaming) { stopStream(); render(); }
      if (a === 'forget-memory') C.forgetMemory(state, current().id, d.id); else C.clearMemory(state, current().id);
      save(); memoryDialog(); $('#memory-enabled').focus(); return;
    }
    if (a === 'switch-user') return switchUser(d.user);
    if (a === 'close-dialog') return closeDialog();
    if (a === 'new-chat') { stopStream(); ui.conversation = null; return navigate('chat'); }
    if (a === 'open-chat') { const c = state.conversations.find(c => c.id === d.id && c.owner === current().id); if (!c) throw Error('没有这条会话的访问权限。'); if (ui.streaming) stopStream(); ui.conversation = c.id; return navigate('chat'); }
    if (a === 'ask') return ask(d.query);
    if (a === 'stop') { stopStream(); render(); return; }
    if (a === 'kb-filter') { ui.filter = d.filter; render(); return; }
    if (a === 'open-kb') { if (!C.can(state, current().id, d.kb)) throw Error('没有访问权限。'); ui.kbId = d.kb; render(); return; }
    if (a === 'open-doc') return documentDialog(d.doc);
    if (a === 'review') return documentDialog(d.doc, d.version, 'diff');
    if (a === 'doc-tab') return documentDialog(ui.doc.id, ui.doc.versionId, d.tab);
    if (a === 'version-view') return documentDialog(ui.doc.id, d.version, 'preview');
    if (a === 'diff-mode') { ui.diffMode = d.mode; return documentDialog(ui.doc.id, ui.doc.versionId, 'diff'); }
    if (a === 'upload') return uploadDialog(d.kb || ui.kbId);
    if (a === 'guide') return guideDialog();
    if (a === 'guide-chat') { closeDialog(); switchUser('Laoyang'); return ask('Azure 预留交换策略有什么变化？'); }
    if (a === 'guide-permission') { closeDialog(); switchUser('Laoyang'); return ask('中国北部 3 的 MySQL B1MS 每小时多少钱？'); }
    if (a === 'guide-revision') { closeDialog(); switchUser('Laoyang'); navigate('knowledge'); return documentDialog('playbook'); }
    if (a === 'guide-review') { closeDialog(); switchUser('21v-admin'); return navigate('reviews'); }
    if (a === 'guide-reask') { closeDialog(); switchUser('Laoyang'); return ask('提交预留交换申请前需要哪些检查？'); }
    if (a === 'guide-admin') { closeDialog(); switchUser('21v-admin'); return navigate('permissions'); }
    if (a === 'stage' || a === 'stage-conflict') {
      const sample = DATA[d.doc]; if (!sample) throw Error('这份文档没有预置演示修订，请上传同名文件更新。');
      const revision = C.revise(state, current().id, d.doc, a === 'stage' ? sample.updated : sample.conflict, { demo: true, label: a === 'stage' ? sample.label : '并行修订 · 不同候选内容' });
      save(); render(); documentDialog(d.doc, revision.id, 'diff'); toast('已提交 v' + revision.number + '，当前已发布知识保持可用。'); return;
    }
    if (a === 'publish' || a === 'fail-publish') {
      const ok = C.publish(state, current().id, d.doc, d.version, a === 'fail-publish');
      if (ok) { const doc = state.documents.find(x => x.id === d.doc); addUsage('知识发布', doc.title, current().id, Math.ceil(C.active(doc).raw.length / 3)); }
      save(); render(); documentDialog(d.doc, d.version, ok ? 'history' : 'diff'); toast(ok ? '新版本已发布，后续问答将使用新的内容。' : '演示索引准备失败，原生效版本保持可用。', !ok); return;
    }
    if (a === 'resolve') {
      const doc = state.documents.find(x => x.id === d.doc), revision = doc.versions.find(v => v.id === d.version);
      for (const peer of C.conflicts(doc, revision)) C.reject(state, current().id, doc.id, peer.id, '人工确认保留 v' + revision.number + '，退回并行修订');
      save(); render(); documentDialog(d.doc, d.version, 'diff'); toast('并行修订已处理，请再次核对内容并确认发布。'); return;
    }
    if (a === 'reject') { C.reject(state, current().id, d.doc, d.version); save(); render(); documentDialog(d.doc, d.version, 'history'); toast('修订已退回，当前生效内容未改变。'); return; }
    if (a === 'rollback') {
      const revision = C.rollback(state, current().id, d.doc, d.version); save(); render(); documentDialog(d.doc, revision.id, 'history'); toast('内容已恢复，并创建了新的修订和发布记录。'); return;
    }
    if (a === 'withdraw') {
      const doc = state.documents.find(x => x.id === d.doc);
      if (!C.can(state, current().id, doc.kbId, 'publish')) throw Error('没有撤回权限。');
      return openDialog('撤回已发布资料', '撤回后，后续问答不再检索这份资料。', `<p class="small muted">${esc(doc.title)} 的历史版本仍会保留，可从版本记录恢复。</p>`, btn('open-doc', '返回文档', '', '', `data-doc="${doc.id}"`) + btn('confirm-withdraw', '确认撤回', '', 'danger', `data-doc="${doc.id}"`));
    }
    if (a === 'confirm-withdraw') { C.withdraw(state, current().id, d.doc); save(); render(); documentDialog(d.doc, null, 'history'); toast('文档已撤回，后续问答已退出检索。'); return; }
    if (a === 'download-doc') {
      const doc = state.documents.find(x => x.id === d.doc); if (!doc || !C.can(state, current().id, doc.kbId)) throw Error('没有下载权限。');
      const version = doc.versions.find(v => v.id === d.version); if (!version) throw Error('版本不存在。');
      download(version.raw, doc.name); return;
    }
    if (a === 'source') {
      const { m } = selectedAnswer(d.message), s = m.sources[Number(d.source)];
      return openDialog('引用依据', s.title, `<div style="display:flex;gap:8px;margin-bottom:19px">${badge('文档 v' + s.version, 'blue')}${badge('回答时知识库 R' + s.release)}${s.demo ? badge('演示修订', 'orange') : ''}</div><h3 style="margin-bottom:15px">${esc(s.section)}</h3><div class="prose">${markdown(s.quote)}</div><div class="info-strip">${icon('history')}这份引用保留了回答时的版本。即使之后发布新版本，历史依据也不会被替换。</div>`, btn('source-full', '查看引用版本全文', 'file', 'primary', `data-doc="${s.docId}" data-version="${s.revisionId}"`));
    }
    if (a === 'source-full') return documentDialog(d.doc, d.version);
    if (a === 'copy-answer') {
      const { m } = selectedAnswer(d.index);
      try { await navigator.clipboard.writeText(m.body); toast('回答已复制。'); }
      catch (_) { openDialog('复制回答', '可选中下方文字复制。', `<textarea class="raw-code" style="width:100%;height:280px" readonly>${esc(m.body)}</textarea>`); }
      return;
    }
    if (a === 'feedback-answer') return feedbackDialog(d.index);
    if (a === 'helpful') {
      const { m, question } = selectedAnswer(d.index, true);
      state.feedback.push({ id: C.id('feedback'), actor: current().id, question, category: '有帮助', comment: '', sources: C.clone(m.sources), at: C.now(), status: 'new' }); save(); toast('谢谢，已记录这条回答及其来源版本。'); return;
    }
    if (['feedback-resolve', 'feedback-evaluate', 'configure-model', 'test-model', 'budget', 'export-usage'].includes(a) && !admin()) throw Error('此操作需要管理员权限。');
    if (a === 'feedback-resolve' || a === 'feedback-evaluate') {
      const f = state.feedback.find(f => f.id === d.id); if (f) { f.status = a === 'feedback-resolve' ? 'resolved' : 'evaluation'; save(); render(); toast(a === 'feedback-resolve' ? '反馈已标记为已处理。' : '已标记为待加入评估集的候选。'); } return;
    }
    if (a === 'configure-model') return openDialog('配置问答模型', '此处保存的是演示配置，不会建立外部连接。', `<form id="model-form"><div class="form-field"><label for="model-name">模型显示名称</label><input id="model-name" value="${esc(state.settings.model)}" maxlength="60" required></div><div class="form-field"><label for="model-provider">提供方</label><select id="model-provider">${['Azure OpenAI', '本地模型', '兼容 API'].map(p => `<option ${p === (state.settings.provider || 'Azure OpenAI') ? 'selected' : ''}>${p}</option>`).join('')}</select></div><div class="info-strip">${icon('lock')}演示不收集 API Key，正式服务将在后端保存模型凭据。</div><div class="form-footer"><button class="btn primary" type="submit">保存配置</button></div></form>`);
    if (a === 'test-model') { toast(esc(d.name) + ' · 模拟连接测试通过，未发出网络请求。'); return; }
    if (a === 'budget') return openDialog('调整演示预算', '预算用于费用管理界面的功能展示。', `<form id="budget-form"><div class="form-field"><label for="budget-value">月度预算（元）</label><input id="budget-value" type="number" min="1" max="1000000" step="1" value="${state.settings.budget}" required></div><div class="form-footer"><button class="btn primary" type="submit">保存预算</button></div></form>`);
    if (a === 'export-usage') {
      const csv = [['类型', '用户', '内容', 'Token估算', '费用估算CNY', '时间'], ...state.usage.map(u => [u.type, u.actor, u.label, u.tokens, u.cost, u.at])].map(row => row.map(v => '"' + String(v).replace(/^[=+@-]/, "'").replace(/"/g, '""') + '"').join(',')).join('\r\n');
      download('\ufeff' + csv, '21v-demo-usage.csv', 'text/csv;charset=utf-8'); return;
    }
    if (a === 'reset') return openDialog('重新开始这场演示', '恢复初始身份、授权和两个原始样例版本。', '<p class="small muted">本次浏览器中的演示会话、个人记忆、上传资料、修订和反馈将清空。sample-data 中的原始文件不受影响。</p>', btn('close-dialog', '保留进度') + btn('confirm-reset', '重置演示', 'history', 'primary'));
    if (a === 'confirm-reset') { stopStream(); state = C.seed(DATA); ui.conversation = null; ui.scope = 'all'; closeDialog(); save(); navigate('chat'); toast('已恢复初始演示状态。'); return; }
  }
  document.addEventListener('pointerover', e => {
    if (!matchMedia('(max-width:760px)').matches && e.target.closest('.sidebar') && !ui.sidebarOpen) { ui.sidebarOpen = true; syncSidebar(); }
  });
  document.addEventListener('pointerout', e => {
    const side = e.target.closest('.sidebar');
    if (side && !matchMedia('(max-width:760px)').matches && !side.contains(e.relatedTarget) && !side.contains(document.activeElement)) { ui.sidebarOpen = false; syncSidebar(); }
  });
  document.addEventListener('focusin', e => {
    if (e.target.closest('.sidebar') && !ui.sidebarOpen) { ui.sidebarOpen = true; syncSidebar(); }
  });
  document.addEventListener('focusout', e => {
    const side = e.target.closest('.sidebar');
    if (side && !side.contains(e.relatedTarget) && !side.matches(':hover')) { ui.sidebarOpen = false; syncSidebar(); }
  });
  document.addEventListener('click', e => {
    const target = e.target.closest('[data-action]');
    if (target) { e.preventDefault(); Promise.resolve().then(() => handleAction(target)).catch(error => toast(error.message, true)); }
    else if (ui.identityMenu && !e.target.closest('.identity-menu')) setIdentityMenu(false);
  });
  document.addEventListener('submit', async e => {
    e.preventDefault(); const form = e.target;
    try {
      if (form.id === 'chat-form') { ui.scope = $('#scope').value; await ask($('#question').value); return; }
      if (form.id === 'upload-form') {
        const result = C.upload(state, current().id, $('#upload-kb').value, $('#upload-name').value.trim(), $('#upload-content').value);
        save(); closeDialog(); navigate('knowledge'); ui.kbId = result.doc.kbId; render(); documentDialog(result.doc.id, result.revision.id, 'diff'); toast('资料已提交为待审修订。'); return;
      }
      if (form.id === 'feedback-form') {
        const { m, question } = selectedAnswer(form.dataset.index, true);
        state.feedback.push({ id: C.id('feedback'), actor: current().id, question, category: $('#feedback-category').value, comment: $('#feedback-comment').value.trim(), sources: C.clone(m.sources), at: C.now(), status: 'new' });
        save(); closeDialog(); toast('反馈已提交，保留了问题与引用版本。'); return;
      }
      if (!admin()) throw Error('此操作需要管理员权限。');
      if (form.id === 'lab-new-form') {
        C.labStart(state, current().id, $('#lab-scope').value, $('#lab-name').value);
        Object.assign(ui, { labTrace: false, labAnswer: false, labFailures: false, labSource: null, labActor: null });
        C.labState(state).tab = 'ingest'; closeDialog(); labRefresh(); toast('候选实验已创建，尚未影响生效方案。');
      } else if (form.id === 'lab-case-form') {
        C.labAddCase(state, current().id, { scope: $('#lab-case-scope').value, question: $('#lab-case-question').value, expected: $('#lab-case-expected').value, source: $('#lab-case-source').value, feedbackId: $('#lab-feedback').value });
        closeDialog(); labRefresh(); toast('金标版本已更新，请重新运行评测。');
      } else if (form.id === 'permissions-form') {
        for (const input of form.querySelectorAll('input')) { const [group, kbId, permission] = input.name.split('|'); state.grants[group][kbId][permission] = input.checked && !input.disabled; }
        C.audit(state, current().id, '更新用户组的知识库访问授权'); save(); render(); toast('授权已保存，切换用户即可验证。');
      } else if (form.id === 'model-form') {
        state.settings.model = $('#model-name').value.trim(); state.settings.provider = $('#model-provider').value; C.audit(state, current().id, '更新问答模型演示配置'); save(); closeDialog(); render(); toast('模型配置已保存。');
      } else if (form.id === 'budget-form') {
        state.settings.budget = Number($('#budget-value').value); save(); closeDialog(); render(); toast('演示预算已更新。');
      }
    } catch (error) { toast(error.message, true); }
  });
  document.addEventListener('input', e => {
    if (e.target.dataset.labField && admin()) {
      const key = e.target.dataset.labField, value = e.target.type === 'checkbox' ? e.target.checked : e.target.type === 'number' ? (e.target.value === '' ? null : Number(e.target.value)) : e.target.value;
      const changes = { [key]: value };
      if (key === 'prompt') changes.promptVersion = Math.max(labConfig().promptVersion, (labRun()?.snapshot.config.promptVersion || C.labBaseline(state, C.labState(state).draft.scope).config.promptVersion) + 1);
      C.labUpdate(state, current().id, changes); ui.labTrace = false; ui.labAnswer = false; save();
      if (key === 'prompt') { $('label[for="lab-prompt"]').textContent = '生成 Prompt · v' + labConfig().promptVersion; $('#lab-prompt-diff').textContent = C.textDiff(C.labBaseline(state, C.labState(state).draft.scope).config.prompt, value, 'generation-prompt.md'); }
      if ($('#lab-save-state')) $('#lab-save-state').textContent = '草稿已保存 · 改动后需重新评测';
    }
    if (e.target.id === 'kb-search') { const cursor = e.target.selectionStart; ui.search = e.target.value; render(); const field = $('#kb-search'); field.focus(); field.setSelectionRange(cursor, cursor); }
  });
  async function loadFile(file) {
    if (!file) return;
    if (file.size > 250 * 1024) throw Error('演示文件请控制在 250 KB 以内。');
    const format = file.name.split('.').pop().toLowerCase(); const text = await file.text(); C.validate(text, format);
    $('#upload-name').value = file.name; $('#upload-content').value = text;
    toast('已读取文件，可检查内容并提交。');
  }
  document.addEventListener('change', async e => {
    try {
      if (e.target.id.startsWith('lab-') && !admin()) throw Error('RAG 实验需要管理员权限。');
      if (e.target.id === 'lab-source-preview') { ui.labSource = e.target.value; labRefresh('lab-source-preview'); }
      if (e.target.id === 'lab-actor') { ui.labActor = e.target.value; ui.labTrace = false; labRefresh('lab-actor'); }
      if (e.target.dataset.labField && (e.target.tagName === 'SELECT' || e.target.type === 'checkbox')) labRefresh(e.target.id);
      if (e.target.id === 'lab-feedback' && e.target.value) {
        const f = state.feedback.find(f => f.id === e.target.value && f.status === 'evaluation' && f.sources.length && f.sources.every(s => ['acn', 'playbook'].includes(s.kbId)));
        if (!f) throw Error('反馈候选不可用。');
        $('#lab-case-question').value = f.question; $('#lab-case-source').value = f.sources.map(s => s.title + ' v' + s.version + ' / ' + s.section).join('; ');
        $('#lab-case-scope').value = new Set(f.sources.map(s => s.kbId)).size > 1 ? 'joint' : f.sources[0].kbId;
      }
      if (e.target.id === 'memory-enabled') {
        const enabled = e.target.checked;
        if (ui.streaming) { stopStream(); render(); }
        C.setMemoryEnabled(state, current().id, enabled); save(); memoryDialog(); $('#memory-enabled').focus(); return;
      }
      if (e.target.id === 'scope') ui.scope = e.target.value;
      if (e.target.id === 'preview-region') { ui.region = e.target.value; documentDialog(ui.doc.id, ui.doc.versionId, 'preview'); }
      if (e.target.id === 'upload-file') await loadFile(e.target.files[0]);
      if (e.target.id === 'upload-kb') { const k = C.kb(state, e.target.value); $('#upload-visibility').textContent = k.owner ? '个人空间，仅你自己可见，由你确认发布。' : '项目资料按组授权，提交后由 21v-admin 审核。'; }
      if (e.target.dataset.grantRead) { const checkbox = document.getElementsByName(e.target.dataset.grantRead + '|upload')[0]; checkbox.disabled = !e.target.checked; if (checkbox.disabled) checkbox.checked = false; }
      if (e.target.id === 'model-enabled') { if (!admin()) throw Error('此操作需要管理员权限。'); state.settings.modelsEnabled = e.target.checked; save(); render(); toast(state.settings.modelsEnabled ? '问答服务已启用。' : '问答服务已停用。'); }
    } catch (error) { toast(error.message, true); }
  });
  document.addEventListener('keydown', e => {
    if (e.target.getAttribute('role') === 'tab' && e.target.closest('.lab-tabs') && ['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) {
      e.preventDefault(); const i = labTabs.findIndex(([key]) => key === e.target.dataset.tab);
      labSetTab(labTabs[e.key === 'Home' ? 0 : e.key === 'End' ? 5 : (i + (e.key === 'ArrowRight' ? 1 : 5)) % 6][0]); return;
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.target.id === 'question') { e.preventDefault(); $('#chat-form').requestSubmit(); }
    if (e.key === 'Escape' && !dialog.open) { ui.identityMenu = false; ui.sidebarOpen = false; render(); }
  });
  document.addEventListener('dragover', e => { const zone = e.target.closest('#drop-zone'); if (zone) { e.preventDefault(); zone.classList.add('drag'); } });
  document.addEventListener('dragleave', e => e.target.closest('#drop-zone')?.classList.remove('drag'));
  document.addEventListener('drop', async e => { const zone = e.target.closest('#drop-zone'); if (zone) { e.preventDefault(); zone.classList.remove('drag'); try { await loadFile(e.dataTransfer.files[0]); } catch (error) { toast(error.message, true); } } });
  render();
})();
