(function (root) {
  'use strict';
  const clone = value => JSON.parse(JSON.stringify(value));
  const now = () => new Date().toISOString();
  const id = prefix => prefix + '-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 7);
  function seed(data) {
    const users = [
      { id: 'Laoyang', group: '21v-Playbook', initials: 'LY', color: 'sand', role: 'user' },
      { id: 'ZZQ', group: '21v-Playbook', initials: 'ZZ', color: 'blue', role: 'user' },
      { id: 'Laojiu', group: '21v-ACN', initials: 'LJ', color: 'violet', role: 'user' },
      { id: 'Daozi', group: '21v-ACN', initials: 'DZ', color: 'green', role: 'user' },
      { id: '21v-admin', group: '21v-DEV', initials: '21', color: 'dark', role: 'admin' }
    ];
    const knowledgeBases = [
      { id: 'playbook', name: 'Playbook 操作手册', group: '21v-Playbook', icon: 'book', color: 'orange', description: '把政策变化，变成有据可循的操作。', release: 1, releases: [] },
      { id: 'acn', name: 'ACN 产品价格', group: '21v-ACN', icon: 'cube', color: 'blue', description: '按区域、规格和计费单位，找到准确依据。', release: 1, releases: [] },
      ...users.filter(u => u.role !== 'admin').map(u => ({ id: 'personal-' + u.id, name: '我的个人知识', owner: u.id, icon: 'lock', color: 'violet', description: '随手记录，只对自己可见。', release: 0, releases: [] }))
    ];
    const documents = Object.values(data).map(d => ({
      id: d.id, name: d.name, title: d.title, kbId: d.kbId, format: d.format, activeId: d.id + '-v1',
      versions: [{ id: d.id + '-v1', number: 1, raw: d.raw, label: '原始样例', author: '21v-admin', at: now(), status: 'published', parent: null, demo: false, indexStatus: 'ready' }]
    }));
    for (const kb of knowledgeBases.filter(k => !k.owner)) {
      kb.releases.push({ number: 1, at: now(), reason: '演示初始化 · 原始样例', actor: '21v-admin', manifest: Object.fromEntries(documents.filter(d => d.kbId === kb.id).map(d => [d.id, d.activeId])) });
    }
    return { schema: 1, users, knowledgeBases, documents, grants: {
      '21v-Playbook': { playbook: { read: true, upload: true }, acn: { read: false, upload: false } },
      '21v-ACN': { acn: { read: true, upload: true }, playbook: { read: false, upload: false } }
    }, conversations: [], memories: {}, feedback: [], activity: [], usage: [], currentUser: 'Laoyang',
    settings: { model: 'Azure OpenAI · 标准问答', temperature: 0.2, topK: 5, rerank: true, hybrid: true, budget: 500, modelsEnabled: true }, evaluations: [] };
  }
  function user(state, userId) { return state.users.find(u => u.id === userId); }
  function kb(state, kbId) { return state.knowledgeBases.find(k => k.id === kbId); }
  function can(state, userId, kbId, action = 'read') {
    const u = user(state, userId), k = kb(state, kbId);
    if (!u || !k) return false;
    if (k.owner) return k.owner === userId;
    if (u.role === 'admin') return true;
    if (action === 'publish' || action === 'manage') return false;
    return !!state.grants[u.group]?.[kbId]?.[action];
  }
  function requireAccess(state, userId, kbId, action) {
    if (!can(state, userId, kbId, action)) throw Error('当前身份没有执行此操作的权限。');
  }
  function active(doc) { return doc.versions.find(v => v.id === doc.activeId); }
  function audit(state, actor, text, kbId) { state.activity.unshift({ id: id('evt'), actor, text, kbId, at: now() }); }
  function validate(raw, format) {
    if (typeof raw !== 'string' || !raw.trim()) throw Error('文档内容不能为空。');
    if (raw.length > 300000) throw Error('演示版单份文档最多支持 300,000 个字符。');
    if (!['md', 'json'].includes(format)) throw Error('首期仅支持 .md 和 .json 文档。');
    if (format === 'json') { try { JSON.parse(raw); } catch (_) { throw Error('JSON 格式不正确，请检查后重新提交。'); } }
  }
  function revise(state, actor, docId, raw, options = {}) {
    const doc = state.documents.find(d => d.id === docId);
    if (!doc) throw Error('文档不存在。');
    requireAccess(state, actor, doc.kbId, 'upload'); validate(raw, doc.format);
    if (active(doc)?.raw === raw || doc.versions.some(v => v.status === 'pending' && v.raw === raw)) throw Error('内容与当前版本或待审修订相同，无需重复提交。');
    const number = Math.max(0, ...doc.versions.map(v => v.number)) + 1;
    const revision = { id: doc.id + '-v' + number, number, raw, author: actor, at: now(), status: 'pending', parent: doc.activeId, label: options.label || '文档更新', demo: !!options.demo, indexStatus: 'ready' };
    doc.versions.push(revision); audit(state, actor, '提交修订 v' + number + ' · ' + doc.title, doc.kbId);
    return revision;
  }
  function upload(state, actor, kbId, name, raw) {
    requireAccess(state, actor, kbId, 'upload');
    if (!name || /[/\\\x00-\x1f]/.test(name) || name.length > 180) throw Error('请使用有效文件名。');
    const format = name.split('.').pop().toLowerCase(); validate(raw, format);
    let doc = state.documents.find(d => d.kbId === kbId && d.name === name);
    if (!doc) {
      doc = { id: id('doc'), kbId, name, title: name.replace(/\.(md|json)$/i, ''), format, activeId: null, versions: [] };
      state.documents.push(doc);
    }
    return { doc, revision: revise(state, actor, doc.id, raw, { label: doc.activeId ? '上传更新' : '新增资料' }) };
  }
  function conflicts(doc, revision) { return doc.versions.filter(v => v.status === 'pending' && v.id !== revision.id); }
  function release(state, actor, doc, reason) {
    const k = kb(state, doc.kbId); k.release++;
    k.releases.unshift({ number: k.release, at: now(), reason, actor, manifest: Object.fromEntries(state.documents.filter(d => d.kbId === k.id && d.activeId).map(d => [d.id, d.activeId])) });
    audit(state, actor, reason + ' · ' + k.name + ' R' + k.release, k.id);
  }
  function publish(state, actor, docId, revisionId, simulateFailure = false) {
    const doc = state.documents.find(d => d.id === docId);
    if (!doc) throw Error('文档不存在。');
    requireAccess(state, actor, doc.kbId, 'publish');
    const revision = doc.versions.find(v => v.id === revisionId);
    if (!revision || revision.status !== 'pending') throw Error('该修订已处理，请刷新后查看。');
    if (revision.parent !== doc.activeId) throw Error('当前生效版本已变化。请退回此修订，基于最新版本重新提交。');
    if (conflicts(doc, revision).length) throw Error('存在并行修订，请先选择保留版本并退回其他修订。');
    if (simulateFailure) {
      revision.indexStatus = 'failed'; audit(state, actor, '索引准备失败，原生效版本保持可用', doc.kbId);
      return false;
    }
    if (active(doc)) active(doc).status = 'archived';
    revision.indexStatus = 'ready'; revision.status = 'published'; revision.reviewer = actor; revision.publishedAt = now(); doc.activeId = revision.id;
    release(state, actor, doc, '确认发布 v' + revision.number); return true;
  }
  function reject(state, actor, docId, revisionId, reason = '退回修订') {
    const doc = state.documents.find(d => d.id === docId);
    if (!doc) throw Error('文档不存在。');
    requireAccess(state, actor, doc.kbId, 'publish');
    const revision = doc.versions.find(v => v.id === revisionId);
    if (!revision || revision.status !== 'pending') throw Error('该修订已处理。');
    revision.status = 'rejected'; revision.reviewer = actor; revision.reason = reason;
    audit(state, actor, reason + ' · ' + doc.title + ' v' + revision.number, doc.kbId);
  }
  function rollback(state, actor, docId, targetId) {
    const doc = state.documents.find(d => d.id === docId);
    if (!doc) throw Error('文档不存在。');
    requireAccess(state, actor, doc.kbId, 'publish');
    const target = doc.versions.find(v => v.id === targetId && ['published', 'archived'].includes(v.status));
    if (!target) throw Error('只能回退至曾经发布的版本。');
    if (doc.versions.some(v => v.status === 'pending')) throw Error('请先处理待审修订，再进行回退。');
    const revision = revise(state, actor, docId, target.raw, { label: '回退至 v' + target.number, demo: target.demo });
    publish(state, actor, docId, revision.id); return revision;
  }
  function withdraw(state, actor, docId) {
    const doc = state.documents.find(d => d.id === docId);
    if (!doc) throw Error('文档不存在。'); requireAccess(state, actor, doc.kbId, 'publish');
    if (!active(doc)) throw Error('文档已经撤回。');
    active(doc).status = 'archived'; doc.activeId = null; release(state, actor, doc, '撤回文档 · ' + doc.title);
  }
  function plain(s) { return String(s).replace(/<[^>]*>/g, ' ').replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/\s+/g, ' ').trim(); }
  function tables(raw, region) {
    try {
      const payload = JSON.parse(raw);
      const group = payload.contentGroups?.find(g => g.groupName === region && g.isActive);
      if (!group) return [];
      return [...group.content.matchAll(/<table\b[^>]*>([\s\S]*?)<\/table>/gi)].map(t => [...t[1].matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/gi)].map(r => [...r[1].matchAll(/<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi)].map(c => plain(c[1]))));
    } catch (_) { return []; }
  }
  function sections(raw) {
    return [...raw.replace(/\r\n/g, '\n').matchAll(/^(#{1,6})\s+(.+)\n([\s\S]*?)(?=^#{1,6}\s|$(?![\s\S]))/gm)].map(m => ({ title: m[2], text: m[3].replace(/&nbsp;/g, '').trim() }));
  }
  function reference(state, doc, quote, section) {
    const v = active(doc); return { docId: doc.id, revisionId: v.id, version: v.number, kbId: doc.kbId, release: kb(state, doc.kbId).release, title: doc.title, section, quote, demo: v.demo };
  }
  const memoryLabels = { name: '称呼', background: '工作背景', region: '常用区域', style: '回答偏好' };
  const regions = ['中国北部 3', '中国东部 3', '中国东部 2', '中国北部 2'];
  const regionIn = text => regions.find(r => text.replace(/\s/g, '').includes(r.replace('中国', '').replace(/\s/g, '')));
  // The demo caller supplies actor. A real backend must derive it from authentication,
  // scope storage by tenant + user, and never accept an owner from a prompt or request body.
  function memoryBank(state, actor) {
    if (!user(state, actor)) throw Error('用户不存在。');
    state.memories ??= {};
    return state.memories[actor] ??= { enabled: true, items: [] };
  }
  function memoryState(state, actor) { return clone(memoryBank(state, actor)); }
  function setMemoryEnabled(state, actor, enabled) { memoryBank(state, actor).enabled = !!enabled; }
  function forgetMemory(state, actor, memoryId) {
    const bank = memoryBank(state, actor), index = bank.items.findIndex(m => m.id === memoryId);
    if (index < 0) throw Error('这条记忆不存在。');
    bank.items.splice(index, 1);
  }
  function clearMemory(state, actor) { memoryBank(state, actor).items = []; }
  function remember(bank, query) {
    if (/这次|本次|暂时|临时|不要记住|别记住|["“”「」『』`<>]|(^|\n)\s*>|原文|引用|转述|密码|密钥|口令|token|secret/i.test(query)) return false;
    if (!/^(?:我叫|我的(?:名字是|工作是|默认区域是|常用区域是)|我(?:目前|主要)?负责|我常用|我主要使用|我希望|我喜欢|请叫我|你可以叫我|以后(?:请)?(?:回答|回复))/.test(query.trim().replace(/^(?:请)?记住[：:,，]?\s*/, ''))) return false;
    // ponytail: four conservative first-person patterns, not semantic extraction;
    // replace with validated, consent-aware extraction when a real model is connected.
    let recognized = false;
    for (let clause of query.split(/[，。！；,;!\n]/)) {
      clause = clause.trim().replace(/^(?:请)?记住[：:,，]?\s*/, '');
      if (!clause || /[？?]|什么|谁|是否|吗|假如|如果/.test(clause)) continue;
      let key, value;
      const name = clause.match(/^(?:我叫|我的名字是|请叫我|你可以叫我)\s*([\p{L}\p{N}· _-]{1,32})$/u);
      const background = clause.match(/^(?:我(?:目前|主要)?负责|我的工作是)\s*(.{1,80})$/u);
      if (name) { key = 'name'; value = name[1].trim(); }
      else if (background) { key = 'background'; value = background[1].trim(); }
      else if (/^(?:我(?:常用|主要使用)|我的(?:默认|常用)区域是)/.test(clause) && regionIn(clause)) { key = 'region'; value = regionIn(clause); }
      else if (/^(?:以后(?:请)?(?:回答|回复)|我希望(?:你)?(?:回答|回复)|我喜欢)/.test(clause) && /简洁|简短|详细/.test(clause)) { key = 'style'; value = /详细/.test(clause) ? '详细' : '简洁'; }
      if (!key || !value) continue;
      recognized = true;
      if (!bank.enabled) continue;
      const existing = bank.items.find(m => m.key === key);
      if (existing?.value === value) continue;
      if (existing) Object.assign(existing, { value, updatedAt: now() });
      else bank.items.push({ id: id('memory'), key, value, updatedAt: now() });
    }
    return recognized;
  }
  function answer(state, actor, query, selected) {
    const bank = memoryBank(state, actor);
    const personal = body => ({ kind: 'memory', body, sources: [] });
    const memoryTopic = /记忆|记得|记住|memory|偏好|背景|负责|叫什么/i.test(query);
    if (memoryTopic && (state.users.some(u => u.id !== actor && query.toLowerCase().includes(u.id.toLowerCase())) || /别人|其他人|其他用户|所有用户|用户\s*[A-Z]|他人/i.test(query))) {
      return personal('我只能使用当前账号的个人记忆，不能查看、确认或透露其他人的记忆。');
    }
    if (/^(?:请)?(?:忘记|删除|清空)我(?:的)?(?:全部|所有)?(?:个人)?记忆[。！!]?$/i.test(query.trim())) {
      clearMemory(state, actor); return personal('已清空个人记忆，后续回答不再使用这些内容。已有会话消息仍然保留。');
    }
    const recall = /你(?:还)?记得我|记住了我|我的(?:个人)?(?:记忆|memory)|我叫什么|我的名字是什么|我(?:主要)?负责什么|我的工作是什么|我常用(?:哪个|什么)区域|我的(?:默认|常用)区域是什么|我喜欢(?:怎样|什么样)的?回答|我的回答偏好/i.test(query);
    if (recall) {
      if (!bank.enabled) return personal('个人记忆已暂停，当前回答不会使用已保存的内容。可以在右上角个人菜单中重新开启。');
      const key = /叫什么|名字/.test(query) ? 'name' : /负责|工作/.test(query) ? 'background' : /区域/.test(query) ? 'region' : /回答|偏好/.test(query) ? 'style' : null;
      const items = bank.items.filter(m => !key || m.key === key);
      return personal(items.length ? '你之前告诉我：\n\n' + items.map(m => '- ' + memoryLabels[m.key] + '：' + m.value).join('\n') : '目前还没有这方面的个人记忆。你可以在聊天中自然地告诉我你的工作背景或回答偏好。');
    }
    const learned = remember(bank, query);
    if (learned && !/[？?]|多少钱|价格|单价|怎么|如何|为什么|哪些/.test(query)) return personal(bank.enabled ? '好的，我会在后续回答中参考这些信息。' : '收到。个人记忆已暂停，这条信息不会保存到后续会话。');
    const values = Object.fromEntries(bank.enabled ? bank.items.map(m => [m.key, m.value]) : []);
    const defaultRegion = values.region && !regionIn(query) && !/中国|美国|欧洲|日本|亚洲|华北|华东|华南|东部|西部|南部|北部|region/i.test(query) && /mysql|b1ms|b2s|d2ds|iops|价格|单价|多少钱|存储/i.test(query);
    const result = knowledgeAnswer(state, actor, defaultRegion ? query + ' ' + values.region : query, selected);
    if (result.kind === 'answer' && result.sources[0]?.docId === 'mysql') {
      if (defaultRegion) result.body = '按你常用的 **' + values.region + '** 查询；如需其他区域，可以在问题中指定。\n\n' + result.body;
      if (values.style === '简洁') result.body = result.body.replace(/\n\n\| 配置项[\s\S]*?(?=\n\n)/, '');
    }
    if (result.kind === 'answer' && result.sources[0]?.docId === 'playbook' && values.style === '简洁') result.body = result.body.replace(/^根据《[^\n]+\n\n/, '');
    return result;
  }
  function knowledgeAnswer(state, actor, query, selected) {
    const visible = state.documents.filter(d => active(d) && can(state, actor, d.kbId) && (!selected || selected.includes(d.kbId)));
    const mysqlQuery = /mysql|b1ms|b2s|d2ds|iops|价格|单价|多少钱|存储/i.test(query);
    const policyQuery = /预留|交换|节省|退款|退订|清单|检查|审批|政策/.test(query);
    const target = policyQuery && !/价格|单价|多少钱|730|iops|b1ms|d2ds/i.test(query) ? 'playbook' : mysqlQuery ? 'mysql' : '';
    // ponytail: title/word matching demonstrates uploaded documents; use real retrieval for semantic questions.
    const own = visible.filter(d => !['mysql', 'playbook'].includes(d.id));
    const words = [...new Intl.Segmenter('zh', { granularity: 'word' }).segment(query.toLowerCase())].filter(w => w.isWordLike && w.segment.length > 1).map(w => w.segment);
    const matched = own.find(d => query.toLowerCase().includes(d.title.toLowerCase())) || (!target && own.find(d => words.some(w => active(d).raw.toLowerCase().includes(w))));
    if (matched) { const excerpt = active(matched).raw.slice(0, 1600); return { kind: 'answer', body: '在已发布资料中找到以下原文：\n\n' + excerpt, sources: [reference(state, matched, excerpt, '原文摘录')] }; }
    const doc = visible.find(d => d.id === target);
    if ((mysqlQuery || policyQuery) && !doc) return { kind: 'restricted', body: '当前选择的已授权知识库中，没有可用于回答这个问题的已发布资料。\n\n你可以调整知识库范围，或联系管理员申请访问权限。', sources: [] };
    if (doc?.id === 'mysql') {
      const region = regionIn(query);
      if (!region) return { kind: 'clarify', body: '你想查询哪个区域的 MySQL 价格？\n\n支持：中国北部 3、中国东部 3、中国东部 2、中国北部 2。请同时告诉我实例规格，例如 **B1MS**。', sources: [] };
      const parsed = tables(active(doc).raw, region);
      const sku = query.match(/B\d+(?:ms|s)|D\d+ds\s*v4|E\d+ds\s*v4/i)?.[0];
      let row = sku && parsed.flat().find(r => r.length === 4 && r[0].replace(/\s/g, '').toLowerCase() === sku.replace(/\s/g, '').toLowerCase());
      if (/存储/.test(query)) {
        const table = parsed.find(t => t.some(r => r[0] === 'GB/月' && r.length === 2)); row = table?.find(r => r[0] === 'GB/月');
        if (row) return { kind: 'answer', body: `根据当前发布的样例，**${region}** 的 MySQL 预配存储价格为 **${row[1]} / GB / 月**。\n\n计算资源、备份及额外 I/O 按各自项目计费，请分别核对。`, sources: [reference(state, doc, row.join(' · '), region + ' / 存储')] };
      }
      if (!row) return { kind: 'clarify', body: '请指定样例中的实例规格，例如 **B1MS、B2S 或 D2ds v4**，并保留区域名称。', sources: [] };
      let body = `根据当前发布的价格资料，**${region} · ${row[0]}** 的现用现付计算价格为 **${row[3]}**。\n\n| 配置项 | 样例中的值 |\n| --- | --- |\n| 区域 | ${region} |\n| 实例 | ${row[0]} |\n| vCore | ${row[1]} |\n| 内存 | ${row[2]} |\n| 计算单价 | ${row[3]} |\n\n存储、备份和 I/O 等费用需单独核对。`;
      if (/月|730/.test(query)) {
        const price = Number(row[3].match(/[\d.]+/)?.[0]);
        body += `\n\n按 **730 小时/月** 假设，计算资源约为 **¥${(price * 730).toFixed(2)}/月**，不包含其他计费项。`;
      }
      if (active(doc).demo) body += '\n\n> 此版本包含演示修订，修改后的价格仅用于展示发布流程，不代表实际报价。';
      return { kind: 'answer', body, sources: [reference(state, doc, row.join(' · '), region + ' / ' + row[0])] };
    }
    if (doc?.id === 'playbook') {
      let sectionNumber = /清单|检查|审批|提交/.test(query) ? 16 : /取消|退款|退订/.test(query) ? 13 : /大小|实例.*灵活/.test(query) ? 4 : /哪些服务|涉及|范围/.test(query) ? 2 : /现有|最后一次|部分交换/.test(query) ? 3 : /节省计划|换购/.test(query) ? 5 : 1;
      const section = sections(active(doc).raw).find(s => s.title.startsWith(sectionNumber + '.'));
      if (!section) return { kind: 'unknown', body: '当前发布版本尚未包含“内部执行检查清单”。\n\n你可以在知识库中提交演示修订，待管理员确认发布后，再问一次这个问题。', sources: [reference(state, doc, '当前版本包含原始政策说明，第 16 节尚未发布。', '版本说明')] };
      let body = `根据《${doc.title}》的 **${section.title.replace(/^\d+\.\s*/, '')}** 章节：\n\n${section.text}\n\n`;
      if (sectionNumber === 16) body += '> 这一节是演示构造的内部流程，不是 Azure 官方政策。';
      else body += '> 回答依据你提供的样例文档；相关政策适用时间以引用中的日期为准。';
      return { kind: 'answer', body, sources: [reference(state, doc, section.text, section.title)] };
    }
    return { kind: 'unknown', body: '这个问题还没有匹配到演示中的资料。\n\n可以试试下方的示例问题，或上传并发布一份相关的 Markdown / JSON。演示问答依据本地资料，不会编造缺失的内容。', sources: [] };
  }
  // ponytail: bounded prefix/suffix diff for uploaded text; use server-side Git for production multi-hunk diffs.
  function textDiff(before, after, name = 'document') {
    const a = before.replace(/\r\n/g, '\n').split('\n'), b = after.replace(/\r\n/g, '\n').split('\n');
    let start = 0, end = 0;
    while (start < a.length && start < b.length && a[start] === b[start]) start++;
    if (start === a.length && start === b.length) return '没有内容变化。';
    while (end < a.length - start && end < b.length - start && a[a.length - 1 - end] === b[b.length - 1 - end]) end++;
    const from = Math.max(0, start - 3), tail = Math.min(3, end);
    return [`--- a/${name}`, `+++ b/${name}`, `@@ -${from + 1},${a.length - end + tail - from} +${from + 1},${b.length - end + tail - from} @@`, ...a.slice(from, start).map(x => ' ' + x), ...a.slice(start, a.length - end).map(x => '-' + x), ...b.slice(start, b.length - end).map(x => '+' + x), ...a.slice(a.length - end, a.length - end + tail).map(x => ' ' + x)].join('\n');
  }
  const labScopes = { acn: 'ACN 产品价格', playbook: 'Playbook 操作手册', global: '全局默认方案', joint: '跨知识库联合评测' };
  const labDefaults = {
    jsonMode: 'structured', markdownMode: 'heading', chunk: 800, overlap: 120, analyzer: 'smartcn',
    keywords: 'MySQL, B1MS, 预留实例, reserved instance', enrichment: 'independent',
    retrieval: 'hybrid', lexicalK: 30, vectorK: 30, embedding: 'BGE-M3', regionFilter: false,
    rerank: false, rerankPool: 30, topK: 5, contextBudget: 6000, dedupe: true,
    model: 'Azure OpenAI · 标准问答', temperature: 0.2, maxTokens: 1000, citations: false, abstain: true,
    promptVersion: 1, prompt: '仅依据已授权的知识回答问题。\n先给出结论，再解释依据。\n证据不足时说明无法确认，不补写未知价格或政策。'
  };
  function labState(state) {
    if (!state.lab) {
      const baseline = { id: 'global-r1', number: 1, scope: 'global', name: '初始默认方案', config: clone(labDefaults), at: now(), actor: '21v-admin' };
      state.lab = { datasetVersion: 1, runs: [], releases: [baseline], active: { global: baseline.id }, draft: null, cases: [
        { id: 'G01', scope: 'acn', type: 'price', question: '中国北部 3 的 MySQL B1MS 每小时多少钱？', expected: '引用北部 3 的 B1MS 价格记录，同时保留币种、单位和计费方案。', source: 'mysql.json · 中国北部 3 / B1MS', actor: 'Laojiu' },
        { id: 'G02', scope: 'acn', type: 'clarify', question: 'MySQL B1MS 多少钱？', expected: '缺少区域时先澄清，不能自行选择某一区域报价。', source: 'mysql.json · 区域约束', actor: 'Daozi' },
        { id: 'G03', scope: 'playbook', type: 'policy', question: '预留交换策略何时变化？', expected: '保留样例中的 2027 年 2 月 1 日及政策适用范围，标明来源版本。', source: '操作手册 · 第 1 节', actor: 'Laoyang' },
        { id: 'G04', scope: 'playbook', type: 'steps', question: '现有预留还可以交换吗？', expected: '完整说明购买时间、最后一次交换机会，以及部分交换后剩余数量的规则。', source: '操作手册 · 第 3 节', actor: 'ZZQ' },
        { id: 'G05', scope: 'common', type: 'acl', question: 'Laoyang 能否读取 ACN 的价格证据？', expected: '默认组授权下不得召回、引用或输出未授权内容；授权变更后需重新审核此用例。', source: '默认组权限 · Laoyang → ACN', actor: 'Laoyang' },
        { id: 'G06', scope: 'common', type: 'unknown', question: '样例没有的产品规格，能否给出确定价格？', expected: '说明缺少证据，不编造价格。', source: '证据不足 / 拒答边界', actor: 'Laojiu' },
        { id: 'G07', scope: 'joint', type: 'joint', question: '查询 MySQL 价格时，也解释其预留交换政策。', expected: '分别引用 ACN 价格和 Playbook 政策，保留各自版本；无跨库权限的用户不得获得合并证据。', source: 'ACN + Playbook · 两份来源', actor: '21v-admin' }
      ] };
      labStart(state, '21v-admin', 'acn', 'MySQL 区域与计费证据优化');
    }
    return state.lab;
  }
  function labAdmin(state, actor) { if (user(state, actor)?.role !== 'admin') throw Error('RAG 实验需要管理员权限。'); }
  function labBaseline(state, scope) {
    const l = labState(state), releaseId = l.active[scope] || l.active.global;
    return l.releases.find(r => r.id === releaseId);
  }
  function labStart(state, actor, scope, name) {
    labAdmin(state, actor);
    if (!Object.hasOwn(labScopes, scope) || !String(name).trim() || String(name).length > 80) throw Error('请选择实验范围并填写 80 字以内的实验名称。');
    const l = labState(state), baseline = labBaseline(state, scope);
    l.draft = { id: id('exp'), name: name.trim(), scope, revision: 1, config: clone(baseline.config), index: null };
    return l.draft;
  }
  function labUpdate(state, actor, changes) {
    labAdmin(state, actor); const d = labState(state).draft;
    for (const key of Object.keys(changes)) if (!Object.hasOwn(labDefaults, key)) throw Error('未知实验参数。');
    if (Object.entries(changes).some(([key, value]) => d.config[key] !== value)) {
      Object.assign(d.config, changes); d.revision++;
    }
  }
  function labSources(state, scope) {
    return state.documents.filter(d => ['acn', 'playbook'].includes(d.kbId) && (['global', 'joint'].includes(scope) || d.kbId === scope) && active(d))
      .map(d => ({ id: d.id, title: d.title, name: d.name, kbId: d.kbId, revisionId: d.activeId, version: active(d).number, release: kb(state, d.kbId).release }));
  }
  function labSnapshot(state) {
    const l = labState(state), d = l.draft;
    return clone({ experimentId: d.id, name: d.name, scope: d.scope, revision: d.revision, config: d.config,
      sources: labSources(state, d.scope), baseline: labBaseline(state, d.scope), datasetVersion: l.datasetVersion,
      knowledgeProfiles: (d.scope === 'joint' ? ['acn', 'playbook'] : []).map(scope => ({ scope, release: labBaseline(state, scope) })),
      cases: l.cases.filter(c => ['global', 'joint'].includes(d.scope) || c.scope === d.scope || c.scope === 'common'), grants: state.grants });
  }
  function labIndexKey(state) {
    const d = labState(state).draft;
    return JSON.stringify([d.scope, labSources(state, d.scope), ...['jsonMode', 'markdownMode', 'chunk', 'overlap', 'analyzer', 'keywords', 'enrichment', 'embedding'].map(k => d.config[k])]);
  }
  function labValidate(config) {
    for (const [key, min, max] of [['chunk', 200, 2000], ['overlap', 0, 500], ['lexicalK', 1, 100], ['vectorK', 1, 100], ['rerankPool', 1, 100], ['topK', 1, 10], ['contextBudget', 1000, 16000], ['temperature', 0, 1], ['maxTokens', 200, 4000]]) {
      const value = config[key];
      if (!Number.isFinite(value) || value < min || value > max || (key !== 'temperature' && !Number.isInteger(value))) throw Error('参数超出范围：' + key);
    }
    if (config.overlap >= config.chunk) throw Error('重叠长度必须小于 Chunk 大小。');
    const candidates = (config.retrieval !== 'vector' ? config.lexicalK : 0) + (config.retrieval !== 'bm25' ? config.vectorK : 0);
    if (config.topK > candidates || (config.rerank && (config.topK > config.rerankPool || config.rerankPool > candidates))) throw Error('需满足：最终 Top K ≤ 精排候选数 ≤ 召回候选总数。');
    if (!config.prompt.trim() || config.prompt.length > 4000) throw Error('请填写 4,000 字以内的生成 Prompt。');
  }
  function labPrepare(state, actor) {
    labAdmin(state, actor); const l = labState(state); labValidate(l.draft.config);
    if (!labSources(state, l.draft.scope).length) throw Error('当前范围没有已发布资料，请先在知识库中发布样例。');
    l.draft.index = { id: id('idx'), key: labIndexKey(state), at: now() }; return l.draft.index;
  }
  // ponytail: deterministic scenario rules illustrate tuning; replace with a real evaluator before using these numbers as benchmarks.
  function labScore(config, scope, cases, state) {
    const json = scope === 'playbook' || config.jsonMode === 'structured';
    const md = scope === 'acn' || config.markdownMode === 'heading';
    const structure = json && md && config.chunk >= 400 && config.overlap < config.chunk;
    const hybrid = config.retrieval === 'hybrid' && config.lexicalK >= 10 && config.vectorK >= 10;
    const context = config.topK >= 3 && config.contextBudget >= 3000 && config.dedupe;
    const acl = !can(state, 'Laoyang', 'acn');
    const metrics = {
      recall: +(0.72 + (hybrid ? .08 : 0) + (config.regionFilter ? .12 : 0) + (structure ? .04 : 0)).toFixed(2),
      mrr: +(0.63 + (config.rerank ? .20 : 0) + (context ? .08 : 0)).toFixed(2),
      faithfulness: +(0.70 + (config.citations ? .16 : 0) + (config.abstain ? .10 : 0)).toFixed(2),
      citation: config.citations ? .98 : .66, acl: acl ? 1 : 0,
      retrievalMs: 300 + (config.retrieval !== 'vector' ? config.lexicalK * 4 : 0) + (config.retrieval !== 'bm25' ? config.vectorK * 4 : 0) + (config.rerank ? config.rerankPool * 5 : 0),
      ttftMs: Math.round(450 + config.contextBudget / 10), generationMs: Math.round(900 + config.maxTokens * .9 + config.contextBudget / 15),
      cost: +(0.004 + config.contextBudget * .000002 + config.maxTokens * .000004 + (config.rerank ? .003 : 0)).toFixed(3)
    };
    const outcomes = {
      price: [structure && hybrid && config.regionFilter && config.rerank && context && config.citations, '区域 / 规格过滤或完整引用缺失，价格与适用条件可能错配。', 'retrieve'],
      clarify: [config.abstain && config.regionFilter, '缺少区域时没有先澄清，可能错误选用某一区域。', 'generate'],
      policy: [md && config.citations && context, '政策适用时间或出处缺失。', 'ingest'],
      steps: [md && structure && config.rerank && context, '步骤与注意事项被拆散，或关键片段未进入上下文。', 'context'],
      acl: [acl, 'Laoyang 当前已获 ACN 授权，与默认拒绝访问的金标不符，请人工复核。', 'evaluate'],
      unknown: [config.abstain, '证据不足时仍尝试生成确定答案。', 'generate'],
      joint: [structure && hybrid && context && config.citations && config.regionFilter, '跨库证据覆盖或各来源的适用条件缺失。', 'retrieve']
    };
    return { metrics, cases: cases.map(c => {
      const [pass, reason, stage] = outcomes[c.type] || [false, '请人工核对新增用例。', 'evaluate'];
      return { ...clone(c), pass: c.manual ? null : pass, reason: c.manual ? '人工新增用例需要逐条判定，系统不自动打分。' : pass ? '预置场景规则通过；仍需在真实评测中核验。' : reason, stage };
    }) };
  }
  const labMetrics = [
    ['recall', '检索', 'Recall@K', .90, 'min', ''], ['mrr', '精排', 'MRR', .85, 'min', 'score'],
    ['faithfulness', '生成', '答案忠实度', .90, 'min', ''], ['citation', '生成', '引用完整率', .95, 'min', ''],
    ['acl', '权限', '权限用例通过率', 1, 'min', ''], ['retrievalMs', '检索', 'P95 检索耗时', 1200, 'max', 'ms'],
    ['ttftMs', '生成', 'P95 首字延迟', 1500, 'max', 'ms'], ['generationMs', '生成', 'P95 完整回答', 4000, 'max', 'ms'],
    ['cost', '费用', '单次问答成本', .05, 'max', '元']
  ];
  function labPassed(run) { return run && labMetrics.every(([key, , , limit, op]) => op === 'min' ? run.result.metrics[key] >= limit : run.result.metrics[key] <= limit) && run.result.cases.every(c => c.pass === true); }
  function labFresh(state, run) { return !!run && JSON.stringify(run.snapshot) === JSON.stringify(labSnapshot(state)); }
  function labRun(state, actor) {
    labAdmin(state, actor); const l = labState(state); labValidate(l.draft.config);
    if (l.draft.index?.key !== labIndexKey(state)) throw Error('请先在“入库与索引”准备当前配置的候选索引。');
    const snapshot = labSnapshot(state), run = { id: id('run'), number: l.runs.length + 1, at: now(), actor, snapshot, indexId: l.draft.index.id, status: 'evaluated',
      baseline: labScore(snapshot.baseline.config, snapshot.scope, snapshot.cases, state), result: labScore(snapshot.config, snapshot.scope, snapshot.cases, state) };
    if (snapshot.scope === 'joint') {
      const profiles = snapshot.knowledgeProfiles.map(p => ({ scope: p.scope, score: labScore(p.release.config, p.scope, snapshot.cases, state) }));
      // Joint demo baseline uses the least favorable simulated metric across the current KB profiles.
      run.baseline.metrics = Object.fromEntries(labMetrics.map(([key, , , , op]) => [key, Math[op === 'min' ? 'min' : 'max'](...profiles.map(p => p.score.metrics[key]))]));
      run.baseline.cases = snapshot.cases.map(c => {
        const results = profiles.filter(p => !['acn', 'playbook'].includes(c.scope) || p.scope === c.scope).map(p => p.score.cases.find(x => x.id === c.id));
        return { ...clone(c), pass: c.manual ? null : results.every(x => x.pass) };
      });
    }
    l.runs.unshift(run); return run;
  }
  function labAddCase(state, actor, item) {
    labAdmin(state, actor);
    if (!['acn', 'playbook', 'joint'].includes(item.scope) || !item.question?.trim() || !item.expected?.trim() || !item.source?.trim()) throw Error('请完整填写问题、期望答案要点、证据定位和知识范围。');
    if (Object.values(item).some(v => typeof v === 'string' && v.length > 2000)) throw Error('单个标注字段请控制在 2,000 字以内。');
    const feedback = item.feedbackId && state.feedback.find(f => f.id === item.feedbackId && f.status === 'evaluation' && f.sources.length && f.sources.every(s => ['acn', 'playbook'].includes(s.kbId)));
    if (item.feedbackId && !feedback) throw Error('反馈候选不可用，个人资料不能进入共享金标集。');
    const l = labState(state), c = { id: id('gold'), scope: item.scope, question: item.question.trim(), expected: item.expected.trim(), source: item.source.trim(), actor, type: 'manual', manual: true };
    l.cases.push(c); l.datasetVersion++;
    if (feedback) { c.feedbackId = feedback.id; feedback.status = 'gold'; feedback.goldCaseId = c.id; feedback.goldVersion = l.datasetVersion; }
    return c;
  }
  function labJudge(state, actor, runId, caseId, pass) {
    labAdmin(state, actor); const run = labState(state).runs.find(r => r.id === runId);
    if (!labFresh(state, run) || run.status !== 'evaluated') throw Error('只能判定当前未提交审核的评测用例。');
    const c = run.result.cases.find(c => c.id === caseId && c.manual);
    if (!c || typeof pass !== 'boolean') throw Error('人工用例不存在。');
    c.pass = pass; c.reason = actor + ' 在 ' + now() + ' 人工判定' + (pass ? '通过' : '未通过') + '（演示）';
  }
  function labTransition(state, actor, runId, action, note = '') {
    labAdmin(state, actor); const l = labState(state), run = l.runs.find(r => r.id === runId);
    if (!run || !labFresh(state, run)) throw Error('实验配置、知识、授权或金标已变化，请重新评测。');
    if (action === 'submit') {
      if (!['evaluated', 'rejected'].includes(run.status) || !labPassed(run)) throw Error('全部发布门槛与人工用例通过后，才能提交审核。');
      if (run.snapshot.scope === 'joint') throw Error('联合评测报告用于跨库验收，不单独发布配置。');
      run.status = 'pending'; run.submittedAt = now();
    } else {
      if (run.status !== 'pending') throw Error('请先提交审核。');
      if (!note.trim()) throw Error('请填写审核意见。');
      if (action === 'publish') {
        if (!labPassed(run)) throw Error('评测门槛未通过。');
        const scope = run.snapshot.scope, previous = labBaseline(state, scope);
        const release = { id: id('lab-release'), number: l.releases.filter(r => r.scope === scope).length + 1, scope, name: run.snapshot.name, config: clone(run.snapshot.config), at: now(), actor, runId, previousId: previous.id, note: note.trim() };
        l.releases.unshift(release); l.active[scope] = release.id; run.status = 'published';
      } else if (action === 'reject') run.status = 'rejected';
      else throw Error('未知审核操作。');
      run.review = { actor, at: now(), note: note.trim() };
    }
    audit(state, actor, 'RAG 实验 · ' + ({ submit: '提交审核', publish: '确认发布', reject: '退回修改' })[action] + ' · ' + run.snapshot.name);
  }
  function labRollback(state, actor, scope, releaseId) {
    labAdmin(state, actor); const l = labState(state), current = labBaseline(state, scope), target = l.releases.find(r => r.id === releaseId);
    if (!target || current.previousId !== target.id) throw Error('只能回退当前方案的上一发布版本。');
    const release = { ...clone(target), id: id('lab-release'), number: l.releases.filter(r => r.scope === scope).length + 1, scope, at: now(), actor, previousId: current.id, restoredFrom: target.id, note: '人工确认回退至 ' + target.name };
    l.releases.unshift(release); l.active[scope] = release.id; audit(state, actor, 'RAG 方案回退 · ' + labScopes[scope]); return release;
  }
  const api = { seed, clone, id, now, user, kb, can, active, audit, validate, revise, upload, conflicts, publish, reject, rollback, withdraw, plain, tables, sections, reference, answer, textDiff, memoryLabels, memoryState, setMemoryEnabled, forgetMemory, clearMemory,
    labScopes, labDefaults, labState, labBaseline, labStart, labUpdate, labSources, labSnapshot, labIndexKey, labPrepare, labMetrics, labPassed, labFresh, labRun, labAddCase, labJudge, labTransition, labRollback };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.DemoCore = api;
})(typeof window === 'undefined' ? {} : window);
