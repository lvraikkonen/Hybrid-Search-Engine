// Run: node --test tests/test_s5_client.mjs. The fake transport ignores AbortSignal
// deliberately: cancellation alone must never permit a stale response to commit.
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {test} from 'node:test';
import assert from 'node:assert/strict';
const source = readFileSync(new URL('../src/app.js', import.meta.url), 'utf8');
function client() {
  const replies = [];
  const context = {window: {}, AbortController, fetch: () => new Promise(resolve => replies.push(resolve)),
    clearPrivate: () => { api.state.epoch++; api.state.revision = null; }};
  const api = runInNewContext(source.slice(source.indexOf('const BIZ'), source.indexOf('function empty')) + '\n({state, request, cancel, STALE})', context);
  api.state.revision = 'current';
  const reply = (i, text, revision = 'current', status = 200) => replies[i]({ok: status === 200, status, headers: new Map([['X-Session-Revision', revision]]), text: async () => text});
  return {...api, reply, deliver: (i, response) => replies[i](response)};
}
test('latest request wins even when old successful response arrives last', async () => {
  const c = client();
  const first = c.request('/api/search', {lane: 'search', raw: true}).catch(e => e);
  const second = c.request('/api/search', {lane: 'search', raw: true});
  c.reply(1, 'new'); assert.equal((await second).data, 'new');
  c.reply(0, 'old'); assert.equal(await first, c.STALE);
});
test('identity epoch blocks late success and late 401 without logging out the new user', async () => {
  for (const status of [200, 401]) {
    const c = client();
    const old = c.request('/api/documents/old', {lane: 'document', raw: true}).catch(e => e);
    c.state.epoch++; c.state.revision = 'new-session';
    c.reply(0, status === 200 ? 'old private body' : '{"error":{"message":"expired"}}', 'current', status);
    assert.equal(await old, c.STALE); assert.equal(c.state.revision, 'new-session');
  }
});
test('cancelled raw/download/detail responses never commit', async () => {
  const c = client();
  for (const lane of ['raw-zh', 'download', 'document']) {
    const old = c.request('/api/documents/old', {lane, raw: true}).catch(e => e);
    c.cancel(lane); c.reply(['raw-zh', 'download', 'document'].indexOf(lane), 'private body');
    assert.equal(await old, c.STALE);
  }
});
test('mismatched server revision clears session and rejects JSON/raw/download payloads', async () => {
  for (const raw of [false, true]) {
    const c = client(); const result = c.request('/api/read', {lane: 'document', raw}).catch(e => e);
    c.reply(0, raw ? 'private body' : '{"session_revision":"current"}', 'different');
    assert.equal(await result, c.STALE); assert.equal(c.state.revision, null);
  }
});
test('identity changes during response body consumption are still rejected', async () => {
  const c = client(); let release;
  const text = new Promise(resolve => { release = resolve; });
  const result = c.request('/api/raw', {lane: 'raw-zh', raw: true}).catch(e => e);
  c.deliver(0, {ok: true, status: 200, headers: new Map([['X-Session-Revision', 'current']]), text: () => text});
  await Promise.resolve();
  c.state.epoch++; c.state.revision = 'new-session';
  release('old private body');
  assert.equal(await result, c.STALE); assert.equal(c.state.revision, 'new-session');
});
test('HTTP failure remains an error; empty success remains a valid result', async () => {
  const c = client(); const failure = c.request('/api/search', {lane: 'search'}).catch(e => e);
  c.reply(0, '{"error":{"message":"timeout"}}', 'current', 504);
  assert.equal((await failure).status, 504);
  const empty = c.request('/api/search', {lane: 'search'});
  c.reply(1, '{"session_revision":"current","items":[]}'); assert.equal((await empty).data.items.length, 0);
});
test('background session checks keep inputs enabled and still clear failed sessions', async () => {
  let resolve, reject, cleared = 0;
  const disabled = [], state = {changing: false, checking: false, polling: false, checkSeq: 0, revision: 'current'};
  const context = {state, profilesReady: true, STALE: Symbol(),
    document: {body: {classList: {add() {}, remove() {}}}},
    controls: () => disabled.push(state.checking),
    clearPrivate: () => { cleared++; state.checkSeq++; state.polling = state.checking = false; },
    request: () => new Promise((yes, no) => { resolve = yes; reject = no; })};
  const synchronize = runInNewContext(source.slice(source.indexOf('async function synchronize'), source.indexOf('async function switchUser')) + '\nsynchronize', context);
  const healthy = synchronize(true);
  assert.equal(state.polling, true); assert.equal(state.checking, false); assert.deepEqual(disabled, []);
  resolve({revision: 'current'}); await healthy;
  assert.deepEqual(disabled, [false]); assert.equal(state.polling, false);
  const failed = synchronize(true); reject(Object.assign(new Error('expired'), {status: 401})); await failed;
  assert.equal(cleared, 1); assert.equal(state.polling, false);
  const foreground = synchronize({type: 'focus'});
  assert.equal(state.checking, true); assert.equal(disabled.at(-1), true);
  resolve({revision: 'current'}); await foreground;
});

test('search renders scores in server order and escapes text inside controlled highlights', async () => {
  const nodes = new Map(['#askResults', '#askStatus', '#askSubmit'].map(key => [key,
    {innerHTML: '', setAttribute() {}, replaceChildren() { this.innerHTML = ''; }}]));
  const title = '<img src=x onerror=alert(1)> 合同', snippet = '生成 <mark>原样</mark> & 合同';
  const doc = {document_id: 25, document_key: 'playbook-test:Document:25', business_category: 0,
    score: 7.125, title, title_en: 'Contract', title_highlights: {title: [
      {text: '<img src=x onerror=alert(1)> ', highlighted: false}, {text: '合同', highlighted: true}]},
    snippets: [{source_field: 'Content', text: snippet, parts: [
      {text: '生成 <mark>原样</mark> & ', highlighted: false}, {text: '合同', highlighted: true}]}]};
  const context = {state: {principal: {}}, $: key => nodes.get(key), icon: () => '', BIZ: {0: 'NCE'}, provenance: () => '',
    request: async () => ({data: {items: [doc, {...doc, document_id: 3, score: .5}]}})};
  const functions = runInNewContext(source.slice(source.indexOf('const esc='), source.indexOf('const paths=')) +
    source.slice(source.indexOf('function highlightedText'), source.indexOf('async function download')) +
    '\n({search, highlightedText})', context);
  await functions.search({query: '合同'});
  const html = nodes.get('#askResults').innerHTML;
  assert.match(html, /相关性得分 <strong>7\.13<\/strong>/);
  assert.ok(html.indexOf('data-result-id="25"') < html.indexOf('data-result-id="3"'));
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt; <mark>合同<\/mark>/);
  assert.match(html, /生成 &lt;mark&gt;原样&lt;\/mark&gt; &amp; <mark>合同<\/mark>/);
  assert.ok(!html.includes('<img') && !html.includes('<mark>原样'));
  assert.equal(functions.highlightedText(title, [{text: 'different', highlighted: true}]), '&lt;img src=x onerror=alert(1)&gt; 合同');
  assert.equal(functions.highlightedText('no match', []), 'no match');
});
