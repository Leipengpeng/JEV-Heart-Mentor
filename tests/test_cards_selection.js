'use strict';

// Run the production browser script without a browser, network, or chat data.
// Deferred API responses exercise user clicks while an older submission is pending.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '..', 'src', 'chatlens', 'web', 'cards.js'), 'utf8');
const page = fs.readFileSync(path.join(__dirname, '..', 'src', 'chatlens', 'web', 'cards.html'), 'utf8');
const emotionLabels = ['平静', '开心', '期待', '关心', '疑惑', '焦虑', '失望', '生气', '难过'];
const messages = ['a', 'b'].map((id, index) => ({
  id: id.repeat(64), timestamp: 1700000000 + index, role: 'other',
  text: `Synthetic message ${id}`,
}));
const otherMessages = ['c', 'd'].map((id, index) => ({
  id: id.repeat(64), timestamp: 1700000100 + index, role: 'other',
  text: `Synthetic second-contact message ${id}`,
}));
const contacts = ['示例甲', '示例乙'];
const contactMessages = { [contacts[0]]: messages, [contacts[1]]: otherMessages };

class Element {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.className = '';
    this.hidden = false;
    this.open = false;
    this.value = '';
    this.scrollTop = 0;
    this._text = '';
    this.classList = {
      add: name => this.classList.toggle(name, true),
      toggle: (name, enabled) => {
        const names = new Set(this.className.split(/\s+/).filter(Boolean));
        if (enabled === undefined) enabled = !names.has(name);
        if (enabled) names.add(name); else names.delete(name);
        this.className = [...names].join(' ');
        return enabled;
      },
    };
  }
  append(child) { this.children.push(child); }
  replaceChildren(...children) { this.children = children; this._text = ''; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  get scrollHeight() { return this.children.length * 80; }
}

function descendants(root) {
  return root.children.flatMap(child => [child, ...descendants(child)]);
}
function contactOptions(document) {
  return descendants(document.getElementById('contact-select')).filter(element => element.tagName === 'option');
}
function documentMock() {
  // Use the shipped page's IDs: removed tabs/evidence controls must not be
  // silently supplied by the mock and hide an obsolete production reference.
  const ids = [...page.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  const elements = new Map(ids.map(id => [id, new Element()]));
  elements.get('error').hidden = true;
  const body = new Element('body');
  for (const element of elements.values()) body.append(element);
  return {
    body,
    getElementById(id) {
      assert.ok(elements.has(id), `Mock is missing DOM element #${id}`);
      return elements.get(id);
    },
    createElement: tag => new Element(tag),
    querySelectorAll(selector) {
      if (selector === 'nav button') return [];
      assert.ok(selector.startsWith('.'), `Unsupported selector ${selector}`);
      return descendants(body).filter(element => element.className.split(/\s+/).includes(selector.slice(1)));
    },
  };
}

const flush = () => new Promise(resolve => setImmediate(resolve));
const signature = messageId => `synthetic-context-${messageId}`;
function resultFor(messageId, marker) {
  const options = [1, 2, 3].map(i => ({ action: `Synthetic action ${i}`, score: (4 - i) / 6 }));
  return {
    context_signature: signature(messageId), selected_message: { id: messageId }, elapsed_seconds: 1,
    proposal: { analysis: marker, options, evidence: 'Synthetic evidence', uncertainties: [] },
    decision: { ranked: options, needs_review: false }, card: { next_step: marker, draft: '' },
    emotion: { available: true, fixed_labels: emotionLabels, selected_label: '期待',
      ranked: emotionLabels.map((label, i) => ({ label, score: i === 2 ? 0.451 : (9 - i) / 100 }))
        .sort((a, b) => b.score - a.score) },
    history_evidence: [],
  };
}

async function harness({ native = false, catalog = null, initialTarget, messageSets = {} } = {}) {
  const document = documentMock();
  const window = { CARD_TOKEN: 'synthetic-test-token', CARD_COMPACT: false, CARD_CONTACTS: contacts };
  const nativeCalls = [];
  if (native) window.webkit = { messageHandlers: { native: { postMessage: value => nativeCalls.push(value) } } };
  const allMessages = { ...contactMessages, ...messageSets };
  const enabled = new Map((catalog || contacts.map(name => ({ id: name, analysis_enabled: true })))
    .map(row => [row.id, !!row.analysis_enabled]));
  const backend = { selection: null, job: null, calls: [], requests: [], revision: 0,
    active_target: initialTarget === undefined ? catalog?.[0]?.id || contacts[0] : initialTarget,
    contact_revision: 'contact-1', contactCounter: 1, contact_status: '', display_name: '' };
  const contactState = () => ({ active_target: backend.active_target, contact_revision: backend.contact_revision,
    ...(catalog ? { display_name: backend.display_name || catalog.find(row => row.id === backend.active_target)?.name,
      analysis_enabled: enabled.get(backend.active_target) || false,
      requires_consent: !enabled.get(backend.active_target), contact_status: backend.contact_status } : {}),
    ...(backend.can_analyze === undefined ? {} : { can_analyze: backend.can_analyze }) });
  let polling;
  let pendingSubmission;
  const deferred = [];
  const response = (value, status = 200) => ({ ok: status < 400, status, json: async () => value });
  async function fetchMock(url, options) {
    const body = options.body ? JSON.parse(options.body) : undefined;
    backend.requests.push({ url, body });
    const held = deferred.find(item => !item.started && item.url === url && item.target === body?.target);
    if (held) {
      held.started = true;
      return new Promise(resolve => { held.release = (value, status = 200) => resolve(response(value, status)); });
    }
    switch (url) {
      case '/api/contacts': return response({ ...contactState(),
        contacts: catalog || contacts.map(name => ({ name, available: true, messages: 2 })) });
      case '/api/contact/select': {
        assert.ok(enabled.has(body.target));
        backend.active_target = body.target;
        backend.contact_revision = `contact-${++backend.contactCounter}`;
        backend.contact_status = '';
        backend.display_name = '';
        backend.selection = null;
        return response(contactState());
      }
      case '/api/contact/enable': {
        assert.equal(body.target, backend.active_target);
        assert.equal(body.contact_revision, backend.contact_revision);
        enabled.set(body.target, true);
        backend.contact_revision = `contact-${++backend.contactCounter}`;
        const message = backend.pendingMessage || allMessages[body.target]?.at(-1);
        backend.selection = { status: 'matched', revision: `revision-${++backend.revision}`, message,
          target: body.target, contact_revision: backend.contact_revision };
        return response({ ...contactState(), selection: backend.selection });
      }
      case '/api/contact/defer': return response(contactState());
      case '/api/history/sync': return response({ ok: true });
      case '/api/history/recent': return response({ messages: allMessages[body.target], before: null, has_more: false });
      case '/api/history/search': return response({ messages: [...allMessages[body.target]].reverse() });
      case '/api/status': return response({ ready: true, phase: 'Synthetic phase', selection: backend.selection, job: backend.job,
        ...contactState(), ...(backend.worker_busy === undefined ? {} : { worker_busy: backend.worker_busy }) });
      case '/api/history/select': {
        assert.equal(body.contact_revision, backend.contact_revision);
        const message = allMessages[body.target].find(message => message.id === body.message_id);
        assert.ok(message, 'Only synthetic messages may be selected');
        backend.selection = { status: 'matched', revision: `revision-${++backend.revision}`, message,
          target: body.target, contact_revision: backend.contact_revision };
        return response(backend.selection);
      }
      case '/api/history/context': return response({ messages: [], context_signature: signature(body.through_id) });
      case '/api/clear': {
        if (backend.job?.status === 'running') return response({ error: 'Synthetic analysis still running' }, 409);
        backend.job = null;
        return response({ ok: true });
      }
      case '/api/analyze/history': {
        backend.calls.push(body.through_id);
        const job = { id: `synthetic-job-${backend.calls.length}`, status: 'running' };
        backend.job = job;
        // Hold the first submission response; subsequent jobs submit immediately.
        if (backend.calls.length === 1) return new Promise(resolve => {
          pendingSubmission = {
            resolve: () => resolve(response({ id: job.id })),
            reject: () => resolve(response({ error: 'Obsolete synthetic submission failed' }, 500)),
          };
        });
        return response({ id: job.id });
      }
      default: throw new Error(`Unexpected API request ${url}`);
    }
  }
  vm.runInNewContext(source, {
    window, document, fetch: fetchMock, AbortSignal: { timeout: () => undefined },
    setInterval: callback => { assert.equal(polling, undefined); polling = callback; },
  }, { filename: 'production/cards.js' });
  await flush();
  assert.equal(typeof polling, 'function');
  assert.equal(document.querySelectorAll('.history-message').length,
    enabled.get(backend.active_target) ? allMessages[backend.active_target]?.length || 0 : 0);
  return {
    backend, document, nativeCalls,
    debug: () => window.ChatLensDebug(),
    click: messageId => {
      const button = document.querySelectorAll('.history-message').find(button => button.dataset.messageId === messageId);
      assert.ok(button); return button.onclick();
    },
    tick: async () => { await polling(); await flush(); },
    switchContact: async name => {
      document.getElementById('contact-select').value = name;
      await document.getElementById('contact-select').onchange();
      await flush();
    },
    hold: (url, target) => {
      const held = { url, target, started: false }; deferred.push(held); return held;
    },
    historyIds: () => document.querySelectorAll('.history-message').map(button => button.dataset.messageId).join(','),
    observe: (target, options = {}) => {
      backend.active_target = target;
      backend.contact_revision = `contact-${++backend.contactCounter}`;
      backend.selection = null;
      backend.contact_status = options.status || '';
      backend.display_name = options.name || '';
      backend.pendingMessage = options.message;
      if (options.message) backend.selection = { status: enabled.get(target) ? 'matched' : 'requires_consent',
        revision: `revision-${++backend.revision}`, target, message: options.message,
        contact_revision: backend.contact_revision };
    },
    releaseSubmission: failed => { assert.ok(pendingSubmission); pendingSubmission[failed ? 'reject' : 'resolve'](); },
    completeJob: (messageId, marker, overrides = {}) => {
      backend.job = { ...backend.job, status: 'done', result: { ...resultFor(messageId, marker), ...overrides } };
    },
    assertOldHidden: () => {
      assert.equal(document.querySelectorAll('.decision-card').length, 0);
      assert.equal(document.getElementById('feed').textContent.includes('OBSOLETE'), false);
      assert.equal(window.ChatLensDebug().result_selected_id, undefined);
    },
  };
}

test('a newer click survives the pending response for an older analysis submission', async () => {
  const h = await harness();
  const olderClick = h.click(messages[0].id);
  await flush();
  assert.equal(h.backend.calls.length, 1);
  assert.equal(h.debug().busy, true);
  await h.click(messages[1].id);
  h.releaseSubmission();
  await olderClick;
  await h.tick();
  h.assertOldHidden();
  h.completeJob(messages[0].id, 'OBSOLETE completed analysis');
  await h.tick();
  assert.equal(h.backend.calls.join(','), messages.map(message => message.id).join(','), 'New selection must submit automatically after the old job finishes');
  h.assertOldHidden();
  h.completeJob(messages[1].id, 'CURRENT analysis');
  await h.tick();
  assert.equal(h.debug().result_selected_id, messages[1].id);
  assert.equal(h.document.querySelectorAll('.decision-card').length, 1);
  assert.equal(h.document.getElementById('feed').textContent.includes('OBSOLETE'), false);
});

test('clearing selection hides stale partial/final results and preserves the next pending click', async () => {
  const h = await harness();
  const olderClick = h.click(messages[0].id);
  await flush();
  h.backend.selection = null; // Equivalent to the observer's clear-selection response.
  await h.tick();
  assert.equal(h.debug().selected_id, undefined);
  h.assertOldHidden();
  h.releaseSubmission();
  await olderClick;
  h.backend.job.proposal = resultFor(messages[0].id, 'OBSOLETE partial analysis').proposal;
  await h.tick();
  h.assertOldHidden();
  await h.click(messages[1].id);
  await h.tick();
  h.assertOldHidden();
  assert.equal(h.debug().selected_id, messages[1].id);
  h.completeJob(messages[0].id, 'OBSOLETE completed analysis');
  await h.tick();
  assert.equal(h.backend.calls.length, 2, 'A later selection must start after the cleared selection\'s job');
  h.assertOldHidden();
  h.completeJob(messages[1].id, 'CURRENT analysis after clear');
  await h.tick();
  assert.equal(h.debug().result_selected_id, messages[1].id);
  assert.equal(h.document.getElementById('error').hidden, true);
});

test('an obsolete submission failure does not replace the new selection with an error', async () => {
  const h = await harness();
  const olderClick = h.click(messages[0].id);
  await flush();
  await h.click(messages[1].id);
  h.releaseSubmission(true);
  await olderClick;
  h.backend.job = null;
  assert.equal(h.document.getElementById('error').hidden, true);
  await h.tick();
  assert.equal(h.backend.calls.length, 2);
  h.completeJob(messages[1].id, 'CURRENT analysis after obsolete failure');
  await h.tick();
  assert.equal(h.debug().result_selected_id, messages[1].id);
  assert.equal(h.document.getElementById('error').hidden, true);
});

test('switching contacts suppresses the former contact job and submits context/analysis for the new contact', async () => {
  const h = await harness();
  const olderClick = h.click(messages[0].id);
  await flush();
  h.releaseSubmission();
  await olderClick;
  h.backend.job.proposal = resultFor(messages[0].id, 'OBSOLETE contact partial analysis').proposal;
  await h.tick();
  assert.equal(h.document.querySelectorAll('.decision-card').length, 1, 'The fixture should first show the old contact partial');
  await h.switchContact(contacts[1]);
  assert.equal(h.debug().target, contacts[1]);
  h.assertOldHidden();
  assert.equal(h.historyIds(), otherMessages.map(message => message.id).join(','));
  await h.tick();
  h.assertOldHidden();
  await h.click(otherMessages[0].id);
  h.completeJob(messages[0].id, 'OBSOLETE contact final analysis');
  await h.tick();
  h.assertOldHidden();
  assert.equal(h.backend.calls.join(','), [messages[0].id, otherMessages[0].id].join(','));
  for (const route of ['/api/history/context', '/api/analyze/history']) {
    const requests = h.backend.requests.filter(request => request.url === route);
    assert.equal(requests.length, 2);
    assert.equal(requests[0].body.target, contacts[0]);
    assert.equal(requests[0].body.through_id, messages[0].id);
    assert.equal(requests[1].body.target, contacts[1]);
    assert.equal(requests[1].body.through_id, otherMessages[0].id);
  }
  h.completeJob(otherMessages[0].id, 'CURRENT second-contact analysis');
  await h.tick();
  assert.equal(h.debug().result_selected_id, otherMessages[0].id);
  assert.equal(h.document.getElementById('feed').textContent.includes('OBSOLETE'), false);
});

test('an old contact history page cannot overwrite the new contact history', async () => {
  const h = await harness();
  const held = h.hold('/api/history/recent', contacts[0]);
  const reload = h.document.getElementById('history-reload').onclick();
  await flush();
  assert.equal(held.started, true);
  await h.switchContact(contacts[1]);
  const expected = otherMessages.map(message => message.id).join(',');
  assert.equal(h.historyIds(), expected);
  held.release({ messages, before: null, has_more: false });
  await reload;
  assert.equal(h.historyIds(), expected);
  assert.equal(h.debug().target, contacts[1]);
});

test('an old contact search result cannot overwrite the new contact history', async () => {
  const h = await harness();
  const held = h.hold('/api/history/search', contacts[0]);
  h.document.getElementById('history-query').value = 'synthetic query';
  const search = h.document.getElementById('history-search').onclick();
  await flush();
  assert.equal(held.started, true);
  await h.switchContact(contacts[1]);
  const expected = otherMessages.map(message => message.id).join(',');
  assert.equal(h.historyIds(), expected);
  held.release({ messages: [...messages].reverse() });
  await search;
  assert.equal(h.historyIds(), expected);
});

test('an old contact search failure cannot surface an error in the new contact', async () => {
  const h = await harness();
  const held = h.hold('/api/history/search', contacts[0]);
  const search = h.document.getElementById('history-search').onclick();
  await flush();
  await h.switchContact(contacts[1]);
  held.release({ error: 'OBSOLETE contact search error' }, 500);
  await search;
  assert.equal(h.document.getElementById('error').hidden, true);
  assert.equal(h.historyIds(), otherMessages.map(message => message.id).join(','));
});

test('an old status response cannot undo a completed contact switch', async () => {
  const h = await harness();
  const held = h.hold('/api/status', undefined);
  const oldPoll = h.tick();
  await flush();
  assert.equal(held.started, true);
  await h.switchContact(contacts[1]);
  held.release({ ready: true, phase: 'Obsolete synthetic status', active_target: contacts[0],
    contact_revision: 'contact-1', selection: null, job: null });
  await oldPoll;
  assert.equal(h.debug().target, contacts[1]);
  assert.equal(h.debug().contact_revision, 'contact-2');
  assert.equal(h.historyIds(), otherMessages.map(message => message.id).join(','));
});

test('rapid contact changes reach the backend in order and the latest choice survives polling', async () => {
  const h = await harness();
  const held = h.hold('/api/contact/select', contacts[1]);
  const olderSwitch = h.switchContact(contacts[1]);
  await flush();
  assert.equal(held.started, true);
  const latestSwitch = h.switchContact(contacts[0]);
  await flush();
  const contactRequests = () => h.backend.requests.filter(request => request.url === '/api/contact/select');
  assert.equal(contactRequests().length, 1, 'The later choice must wait until the earlier POST finishes');
  await h.tick();
  assert.equal(contactRequests().length, 1, 'Polling must remain paused until the latest intent is acknowledged');

  // Apply the older POST's real backend mutation before releasing its response.
  // The queued newer POST must be sent afterward so the latest choice wins.
  h.backend.active_target = contacts[1];
  h.backend.contact_revision = `contact-${++h.backend.contactCounter}`;
  h.backend.selection = null;
  held.release({ active_target: h.backend.active_target, contact_revision: h.backend.contact_revision });
  await olderSwitch;
  await latestSwitch;
  assert.equal(contactRequests().map(request => request.body.target).join(','), contacts.slice().reverse().join(','));
  assert.equal(h.backend.active_target, contacts[0]);
  assert.equal(h.debug().target, contacts[0]);
  assert.equal(h.historyIds(), messages.map(message => message.id).join(','));
  await h.tick();
  assert.equal(h.debug().target, contacts[0], 'The next status poll must preserve the latest choice');
});

test('a failed earlier contact POST cannot block or surface an error for the queued latest choice', async () => {
  const h = await harness();
  const held = h.hold('/api/contact/select', contacts[1]);
  const olderSwitch = h.switchContact(contacts[1]);
  await flush();
  const latestSwitch = h.switchContact(contacts[0]);
  await flush();
  held.release({ error: 'OBSOLETE contact switch error' }, 500);
  await olderSwitch;
  await latestSwitch;
  assert.equal(h.backend.active_target, contacts[0]);
  assert.equal(h.debug().target, contacts[0]);
  assert.equal(h.document.getElementById('error').hidden, true);
  assert.equal(h.historyIds(), messages.map(message => message.id).join(','));
  await h.tick();
  assert.equal(h.debug().target, contacts[0]);
});

async function completeFirst(h, marker = 'CURRENT synthetic analysis', overrides = {}) {
  const click = h.click(messages[0].id);
  await flush();
  h.releaseSubmission();
  await click;
  h.completeJob(messages[0].id, marker, overrides);
  await h.tick();
}

test('a finished job with a changed signature is adopted when fresh context matches the server result', async () => {
  const h = await harness();
  const click = h.click(messages[0].id);
  await flush();
  h.releaseSubmission();
  await click;
  h.backend.job.proposal = { ...resultFor(messages[0].id, 'BEFORE partial analysis').proposal, options: [] };
  await h.tick();
  assert.equal(h.document.querySelectorAll('.action').length, 0);
  const freshSignature = 'synthetic-context-after-sync';
  h.completeJob(messages[0].id, 'CURRENT reconciled full analysis', { context_signature: freshSignature });
  const held = h.hold('/api/history/context', contacts[0]);
  const completion = h.tick();
  await flush();
  assert.equal(held.started, true, 'Completion must reread context after a source signature changes');
  assert.equal(h.debug().busy, true, 'Keep the finished job owned while context validation is pending');
  assert.equal(h.document.querySelectorAll('.decision-card').length, 0, 'The obsolete partial must not linger');
  held.release({ messages: [{ role: 'other', text: 'Synthetic freshly synchronized context' }], context_signature: freshSignature });
  await completion;
  assert.equal(h.debug().signature, freshSignature);
  assert.equal(h.debug().result_selected_id, messages[0].id);
  assert.equal(h.debug().busy, false);
  assert.equal(h.document.querySelectorAll('.action').length, 3);
  assert.equal(h.document.querySelectorAll('.emotion-chip').length, 3);
  assert.equal(h.document.getElementById('feed').textContent.includes('CURRENT reconciled'), true);
  assert.equal(h.document.getElementById('feed').textContent.includes('BEFORE'), false);
  assert.equal(h.backend.calls.length, 1, 'A current completed result must not cause another model call');
});

test('a finished job superseded by another context change is replaced by a new analysis without showing the old card', async () => {
  const h = await harness();
  const click = h.click(messages[0].id);
  await flush();
  h.releaseSubmission();
  await click;
  h.backend.job.proposal = { ...resultFor(messages[0].id, 'OBSOLETE partial').proposal, options: [] };
  await h.tick();
  h.completeJob(messages[0].id, 'OBSOLETE full result', { context_signature: 'synthetic-context-intermediate' });
  const held = h.hold('/api/history/context', contacts[0]);
  const completion = h.tick();
  await flush();
  assert.equal(held.started, true);
  const latestSignature = 'synthetic-context-newest';
  held.release({ messages: [{ role: 'other', text: 'Synthetic newest context' }], context_signature: latestSignature });
  await completion;
  assert.equal(h.debug().signature, latestSignature);
  assert.deepEqual(h.backend.calls, [messages[0].id, messages[0].id]);
  assert.equal(h.debug().busy, true);
  h.assertOldHidden();
  h.completeJob(messages[0].id, 'CURRENT newest-context result', { context_signature: latestSignature });
  await h.tick();
  assert.equal(h.debug().result_selected_id, messages[0].id);
  assert.equal(h.document.querySelectorAll('.action').length, 3);
  assert.equal(h.document.getElementById('feed').textContent.includes('CURRENT newest-context'), true);
  assert.equal(h.document.getElementById('feed').textContent.includes('OBSOLETE'), false);
});

test('a late completed-job context check cannot overwrite a newer clicked sentence', async () => {
  const h = await harness();
  const click = h.click(messages[0].id);
  await flush();
  h.releaseSubmission();
  await click;
  h.completeJob(messages[0].id, 'OBSOLETE reconciled analysis', { context_signature: 'synthetic-older-after-sync' });
  const held = h.hold('/api/history/context', contacts[0]);
  const completion = h.tick();
  await flush();
  assert.equal(held.started, true);
  await h.click(messages[1].id);
  const latest = h.debug();
  held.release({ messages: [{ role: 'other', text: 'OBSOLETE synthetic context' }], context_signature: 'synthetic-older-after-sync' });
  await completion;
  assert.equal(h.debug().selected_id, messages[1].id);
  assert.equal(h.debug().signature, latest.signature);
  assert.equal(h.debug().revision, latest.revision);
  assert.deepEqual(h.backend.calls, messages.map(message => message.id));
  h.assertOldHidden();
  h.completeJob(messages[1].id, 'CURRENT newer-click result');
  await h.tick();
  assert.equal(h.debug().result_selected_id, messages[1].id);
  assert.equal(h.document.getElementById('feed').textContent.includes('OBSOLETE'), false);
});

test('the shipped page renders only analysis, action ranking and next step without former tabs or score notes', async () => {
  const h = await harness();
  await completeFirst(h);
  const headings = h.document.querySelectorAll('.section-heading');
  assert.deepEqual(headings.map(heading => heading._text), ['分析', '动作排名', '下一步']);
  assert.equal(h.document.querySelectorAll('nav button').length, 0);
  assert.equal(h.document.querySelectorAll('.score-note').length, 0);
  assert.equal(/<nav\b|id="(?:sources|evidence|timing|layout-float|layout-hint)"/.test(page), false);
  const visibleText = h.document.getElementById('feed').textContent;
  for (const obsolete of ['相对排序分', '非准确率', '置信度', '证据', '消息旁', '精简']) {
    assert.equal(visibleText.includes(obsolete), false, `Removed explanatory text must stay absent: ${obsolete}`);
  }
});

test('emotion display contains only the three highest JEV scores while retaining nine backend labels', async () => {
  const h = await harness();
  const emotion = resultFor(messages[0].id, 'Synthetic analysis').emotion;
  // The UI must rank by score even if the backend array arrives in a different
  // order. It retains the full fixed-label result without triggering rescoring.
  emotion.ranked.find(row => row.label === '开心').score = 1.2;
  emotion.ranked.find(row => row.label === '难过').score = -0.2;
  const originalOrder = emotion.ranked.map(row => row.label);
  await completeFirst(h, 'CURRENT synthetic analysis', { emotion });
  const chips = h.document.querySelectorAll('.emotion-chip');
  assert.equal(chips.length, 3);
  assert.deepEqual(chips.map(chip => chip.children[0].textContent), ['开心', '期待', '平静']);
  for (const chip of chips) {
    const score = chip.children[1].textContent;
    assert.match(score, /^(?:[0-9]\.[0-9]|10\.0)\/10$/);
    assert.ok(Number(score.split('/')[0]) >= 0 && Number(score.split('/')[0]) <= 10);
  }
  assert.equal(chips[0].children[1].textContent, '10.0/10');
  assert.equal(chips[1].children[1].textContent, '4.5/10');
  assert.equal(chips[2].children[1].textContent, '0.9/10');
  assert.equal(chips.filter(chip => chip.className.split(/\s+/).includes('leading')).length, 1);
  assert.equal(chips[0].className.includes('leading'), true);
  assert.deepEqual(emotion.fixed_labels, emotionLabels);
  assert.equal(emotion.ranked.length, 9);
  assert.deepEqual(emotion.ranked.map(row => row.label), originalOrder, 'Rendering must not mutate the cached scorer result');
});

test('close head scores gain nonlinear visual contrast without mutating scores or ordering', async () => {
 for(const values of [[.340,.339,.330],[.9,.010,.0095]]){
  const h = await harness();
  const ranked = [{action:'直球',score:values[0]},{action:'玩笑',score:values[1]},{action:'目移',score:values[2]}];
  const original = JSON.stringify(ranked);
  const emotion = {available:true,ranked:ranked.map((row,i)=>({label:emotionLabels[i],score:row.score}))};
  await completeFirst(h,'CURRENT contrast result',{decision:{ranked},emotion});
  const actions=h.document.querySelectorAll('.action');
  const displayed=actions.map(row=>Number(row.children.find(x=>x.className==='score')._text));
  assert.ok(displayed[0]-displayed[1]>1);
  assert.ok(displayed[1]-displayed[2]>1);
  assert.ok(displayed.every(x=>x>=0&&x<=10));
  assert.deepEqual(actions.map(row=>row.children.find(x=>x.className==='label').textContent),ranked.map(row=>row.action));
  assert.deepEqual(h.document.querySelectorAll('.emotion-score').map(x=>Number(x.textContent.split('/')[0])),displayed);
  actions.forEach((row,i)=>{
    const track=row.children.find(x=>x.className==='score-track');
    assert.ok(Math.abs(parseFloat(track.children[0].style.width)-displayed[i]*10)<.6);
  });
  assert.equal(JSON.stringify(ranked),original);
 }
});

test('identical scores remain identical instead of inventing a winner advantage', async () => {
  const h=await harness();
  await completeFirst(h,'CURRENT tied scores',{decision:{ranked:[1,2,3].map(i=>({action:'合成'+i,score:.333}))}});
  const scores=h.document.querySelectorAll('.action').map(row=>row.children.find(x=>x.className==='score')._text);
  assert.deepEqual(scores,['3.3','3.3','3.3']);
});

test('action rank uses a ten-point scale and distinct recommendation states with proportional bounded bars', async () => {
  const h = await harness();
  const ranked = [
    { action: 'Synthetic best', score: 1.2 },
    { action: 'Synthetic middle', score: 0.234 },
    { action: 'Synthetic lowest', score: -0.1 },
  ];
  await completeFirst(h, 'CURRENT synthetic analysis', { decision: { ranked } });
  const actions = h.document.querySelectorAll('.action');
  assert.equal(actions.length, 3);
  const expected = [
    ['Synthetic best', 'recommend-best', '10.0/10', 100],
    ['Synthetic middle', 'recommend-middle', '2.3/10', 23.4],
    ['Synthetic lowest', 'recommend-low', '0.0/10', 0],
  ];
  actions.forEach((action, i) => {
    assert.equal(action.children[1].textContent, expected[i][0]);
    assert.equal(action.className.split(/\s+/).includes(expected[i][1]), true);
    assert.equal(action.children[2].textContent, expected[i][2]);
    assert.ok(Math.abs(Number.parseFloat(action.children[3].children[0].style.width) -
      expected[i][3]) < 0.0001);
    assert.equal(action.className.split(/\s+/).filter(name => name.startsWith('recommend-')).length, 1);
  });
});

test('partial LLM analysis does not fabricate emotion or decision scores before JEV returns', async () => {
  const h = await harness();
  const click = h.click(messages[0].id);
  await flush();
  h.releaseSubmission();
  await click;
  h.backend.job.proposal = resultFor(messages[0].id, 'CURRENT partial analysis').proposal;
  await h.tick();
  assert.equal(h.document.querySelectorAll('.decision-card').length, 1);
  assert.equal(h.document.querySelectorAll('.emotion-chip').length, 0);
  assert.equal(h.document.querySelectorAll('.score-track').length, 0);
  assert.deepEqual(h.document.querySelectorAll('.score').map(score => score.textContent), ['—', '—', '—']);
  assert.equal(h.document.querySelectorAll('.next-step')[0].textContent, '正在选择…');
});

test('completed JEV scores appear before the single MiniCPM render and survive a writing failure', async () => {
  const h = await harness();
  const click = h.click(messages[0].id);
  await flush();
  h.releaseSubmission();
  await click;
  const completed = resultFor(messages[0].id, 'Synthetic pending analysis');
  h.backend.job.proposal = { analysis: '', options: completed.decision.ranked,
    jev_scored: true, emotion: completed.emotion };
  await h.tick();
  assert.equal(h.document.querySelectorAll('.decision-card').length, 1);
  assert.equal(h.document.querySelectorAll('.emotion-chip').length, 3);
  assert.equal(h.document.querySelectorAll('.score-track').length, 3);
  assert.deepEqual(h.document.querySelectorAll('.score').map(score => score.textContent),
    ['5.0/10', '3.3/10', '1.7/10']);
  assert.equal(h.document.querySelectorAll('.analysis')[0].textContent, '正在组织分析…');
  assert.equal(h.document.querySelectorAll('.next-step')[0].textContent, '正在选择…');
  h.backend.job.status = 'error';
  h.backend.job.error = 'Synthetic one-call writing failure';
  await h.tick();
  assert.equal(h.document.querySelectorAll('.decision-card').length, 1);
  assert.equal(h.document.querySelectorAll('.emotion-chip').length, 3);
  assert.equal(h.document.querySelectorAll('.score-track').length, 3);
});

test('a terminal writing failure received without an earlier running poll still shows completed JEV scores', async () => {
  const h = await harness();
  const click = h.click(messages[0].id);
  await flush();
  h.releaseSubmission();
  await click;
  const completed = resultFor(messages[0].id, 'Synthetic failed analysis');
  h.backend.job.proposal = { analysis: '', options: completed.decision.ranked,
    jev_scored: true, emotion: completed.emotion };
  h.backend.job.status = 'error';
  h.backend.job.error = 'Synthetic one-call writing failure';
  await h.tick();
  assert.equal(h.document.querySelectorAll('.decision-card').length, 1);
  assert.equal(h.document.querySelectorAll('.emotion-chip').length, 3);
  assert.equal(h.document.querySelectorAll('.score-track').length, 3);
  assert.equal(h.document.querySelectorAll('.analysis')[0].textContent, '文案未完成');
  assert.equal(h.document.querySelectorAll('.next-step')[0].textContent, '文案未完成');
});

test('refresh keeps the selected position and completed card visible until its replacement is ready', async () => {
  const h = await harness({ native: true });
  await completeFirst(h, 'BEFORE synthetic analysis');
  await h.document.getElementById('refresh').onclick();
  assert.equal(h.debug().selected_id, messages[0].id);
  assert.equal(h.debug().result_selected_id, messages[0].id);
  assert.equal(h.document.getElementById('feed').textContent.includes('BEFORE'), true);
  assert.equal(h.document.querySelectorAll('.decision-card').length, 1);
  assert.equal(h.nativeCalls.length, 0, 'Reanalysis must not hide, refocus or relocate the native panel');
  assert.equal(h.backend.calls.join(','), [messages[0].id, messages[0].id].join(','));
  await h.tick();
  assert.equal(h.document.getElementById('feed').textContent.includes('BEFORE'), true);
  h.completeJob(messages[0].id, 'AFTER synthetic analysis');
  await h.tick();
  assert.equal(h.document.getElementById('feed').textContent.includes('AFTER'), true);
  assert.equal(h.document.getElementById('feed').textContent.includes('BEFORE'), false);
  assert.equal(h.debug().selected_id, messages[0].id);
});

test('closing the native card preserves its selected result and a later position can be analyzed independently', async () => {
  const h = await harness({ native: true });
  await completeFirst(h, 'BEFORE close synthetic analysis');
  const before = h.debug();
  h.document.getElementById('dismiss').onclick();
  assert.equal(h.nativeCalls.length, 1);
  assert.equal(h.nativeCalls[0].action, 'hide-card');
  assert.equal(h.debug().selected_id, before.selected_id);
  assert.equal(h.debug().result_selected_id, before.result_selected_id);
  assert.equal(h.backend.requests.some(request => request.url === '/api/history/clear-selection'), false);
  await h.tick();
  assert.equal(h.debug().result_selected_id, messages[0].id);
  await h.click(messages[1].id);
  h.assertOldHidden();
  h.completeJob(messages[1].id, 'AFTER close latest analysis');
  await h.tick();
  assert.equal(h.debug().selected_id, messages[1].id);
  assert.equal(h.debug().result_selected_id, messages[1].id);
  assert.equal(h.document.getElementById('feed').textContent.includes('BEFORE'), false);
  assert.equal(h.document.getElementById('feed').textContent.includes('AFTER'), true);
});

const dynamicCatalog = [
  { id: 'synthetic-contact-a', name: '同名联系人', analysis_enabled: true, available: true },
  { id: 'synthetic-contact-b', name: '同名联系人', analysis_enabled: false, available: true },
  { id: 'synthetic-contact-c', name: '待确认联系人', analysis_enabled: false, available: true },
];
const dynamicMessages = {
  [dynamicCatalog[0].id]: messages,
  [dynamicCatalog[1].id]: otherMessages,
  [dynamicCatalog[2].id]: otherMessages,
};
const analysisRoutes = new Set(['/api/history/sync', '/api/history/recent', '/api/history/search',
  '/api/history/context', '/api/analyze/history']);

test('the live contact catalog replaces bootstrap names and uses canonical IDs for same-name contacts', async () => {
  const h = await harness({ catalog: dynamicCatalog, messageSets: dynamicMessages });
  const options = contactOptions(h.document);
  assert.deepEqual(options.map(option => option.value), [dynamicCatalog[2].id, dynamicCatalog[0].id, dynamicCatalog[1].id]);
  assert.equal(options[1].textContent, options[2].textContent);
  assert.equal(h.debug().target, dynamicCatalog[0].id);
  await h.switchContact(dynamicCatalog[1].id);
  assert.equal(h.debug().target, dynamicCatalog[1].id);
  assert.equal(h.debug().analysis_enabled, false);
  assert.equal(h.debug().requires_consent, true);
  assert.equal(h.document.getElementById('consent').hidden, false);
  assert.equal(h.document.getElementById('feed').textContent, '');
  assert.equal(h.document.getElementById('selected-message').hidden, true);
  assert.equal(h.document.getElementById('history-pane').hidden, true);
  assert.equal(h.backend.requests.filter(request => analysisRoutes.has(request.url) &&
    request.body?.target === dynamicCatalog[1].id).length, 0);
});

test('observing an unapproved chat clears prior analysis and blocks every history/model entry point', async () => {
  const h = await harness({ catalog: dynamicCatalog, messageSets: dynamicMessages, native: true });
  await completeFirst(h, 'OBSOLETE approved-contact result');
  h.observe(dynamicCatalog[1].id, { message: otherMessages[0] });
  await h.tick();
  h.assertOldHidden();
  assert.equal(h.document.getElementById('consent-enable').disabled, false);
  assert.equal(h.document.getElementById('selected-quote').textContent, '');
  await h.document.getElementById('history-search').onclick();
  await h.document.getElementById('history-reload').onclick();
  await h.document.getElementById('refresh').onclick();
  h.document.getElementById('open-history').onclick();
  await h.tick();
  assert.equal(h.nativeCalls.length, 0);
  assert.equal(h.backend.requests.filter(request => analysisRoutes.has(request.url) &&
    request.body?.target === dynamicCatalog[1].id).length, 0);
  assert.equal(h.debug().selected_id, undefined);
});

test('explicit approval enables only the current canonical contact and resumes its pending clicked message', async () => {
  const h = await harness({ catalog: dynamicCatalog, initialTarget: dynamicCatalog[1].id,
    messageSets: dynamicMessages });
  assert.equal(h.backend.calls.length, 0);
  h.observe(dynamicCatalog[1].id, { message: otherMessages[0] });
  await h.tick();
  const revision = h.debug().contact_revision;
  const enabling = h.document.getElementById('consent-enable').onclick();
  await flush();
  assert.equal(h.backend.calls.length, 1);
  const enable = h.backend.requests.find(request => request.url === '/api/contact/enable');
  assert.equal(enable.body.target, dynamicCatalog[1].id);
  assert.equal(enable.body.contact_revision, revision);
  h.releaseSubmission();
  await enabling;
  assert.equal(h.debug().analysis_enabled, true);
  assert.equal(h.debug().requires_consent, false);
  assert.equal(h.document.getElementById('consent').hidden, true);
  assert.equal(h.debug().selected_id, otherMessages[0].id);
  h.completeJob(otherMessages[0].id, 'CURRENT newly approved result');
  await h.tick();
  assert.equal(h.debug().result_selected_id, otherMessages[0].id);
  await h.switchContact(dynamicCatalog[2].id);
  assert.equal(h.document.getElementById('consent').hidden, false);
  assert.equal(h.backend.requests.filter(request => request.url === '/api/contact/enable').length, 1);
  assert.equal(h.backend.requests.filter(request => analysisRoutes.has(request.url) &&
    request.body?.target === dynamicCatalog[2].id).length, 0);
});

test('an unresolved or ambiguous native chat displays its name and cannot trigger analysis', async () => {
  const h = await harness({ catalog: dynamicCatalog, messageSets: dynamicMessages });
  h.observe(null, { status: 'ambiguous', name: '同名联系人' });
  await h.tick();
  assert.equal(h.debug().target, null);
  assert.equal(h.document.getElementById('contact-select').children[0].textContent, '同名联系人');
  assert.equal(h.document.getElementById('consent').hidden, false);
  assert.equal(h.document.getElementById('consent-enable').disabled, true);
  const count = h.backend.requests.length;
  await h.document.getElementById('consent-enable').onclick();
  assert.equal(h.backend.requests.length, count);
  h.observe(dynamicCatalog[2].id, { status: 'unresolved' });
  await h.tick();
  assert.equal(h.document.getElementById('consent-enable').disabled, true);
});

test('deferring a first analysis leaves the card blank and never enables the contact', async () => {
  const h = await harness({ catalog: dynamicCatalog, initialTarget: dynamicCatalog[1].id,
    messageSets: dynamicMessages });
  await h.document.getElementById('consent-defer').onclick();
  await h.tick();
  assert.equal(h.document.getElementById('consent').hidden, true);
  assert.equal(h.document.getElementById('feed').textContent, '');
  assert.equal(h.debug().analysis_enabled, false);
  assert.equal(h.backend.requests.filter(request => analysisRoutes.has(request.url)).length, 0);
  assert.equal(h.backend.requests.filter(request => request.url === '/api/contact/enable').length, 0);
  h.observe(dynamicCatalog[1].id);
  await h.tick();
  assert.equal(h.document.getElementById('consent').hidden, false,
    'A later intentional visit should allow the user to reconsider');
});

test('a late context response for an approved chat cannot analyze a newly observed unapproved chat', async () => {
  const h = await harness({ catalog: dynamicCatalog, messageSets: dynamicMessages });
  const held = h.hold('/api/history/context', dynamicCatalog[0].id);
  const oldClick = h.click(messages[0].id);
  await flush();
  assert.equal(held.started, true);
  h.observe(dynamicCatalog[1].id);
  await h.tick();
  held.release({ messages: [], context_signature: signature(messages[0].id) });
  await oldClick;
  assert.equal(h.backend.calls.length, 0);
  assert.equal(h.debug().target, dynamicCatalog[1].id);
  assert.equal(h.document.getElementById('consent').hidden, false);
  h.assertOldHidden();
});

test('a delayed approval response cannot restore an earlier chat after native conversation changes', async () => {
  const h = await harness({ catalog: dynamicCatalog, initialTarget: dynamicCatalog[1].id,
    messageSets: dynamicMessages });
  const held = h.hold('/api/contact/enable', dynamicCatalog[1].id);
  const enabling = h.document.getElementById('consent-enable').onclick();
  await flush();
  assert.equal(held.started, true);
  h.observe(dynamicCatalog[2].id);
  await h.tick();
  held.release({ active_target: dynamicCatalog[1].id, contact_revision: 'obsolete-approval',
    display_name: dynamicCatalog[1].name, analysis_enabled: true, requires_consent: false,
    selection: { status: 'matched', revision: 'obsolete-selection', message: otherMessages[0],
      target: dynamicCatalog[1].id, contact_revision: 'obsolete-approval' } });
  await enabling;
  assert.equal(h.debug().target, dynamicCatalog[2].id);
  assert.equal(h.debug().analysis_enabled, false);
  assert.equal(h.document.getElementById('consent').hidden, false);
  assert.equal(h.document.getElementById('consent-enable').disabled, false);
  assert.equal(h.backend.calls.length, 0);
});

test('explicit backend eligibility enables a known chat even when the cached history catalog is empty', async () => {
  const catalog = dynamicCatalog.map(row => ({ ...row, available: row.id === dynamicCatalog[2].id ? false : row.available }));
  const h = await harness({ catalog, initialTarget: dynamicCatalog[2].id, messageSets: dynamicMessages });
  assert.equal(h.document.getElementById('consent-enable').disabled, true);
  h.backend.can_analyze = true;
  await h.tick();
  assert.equal(h.document.getElementById('consent-enable').disabled, false);
  assert.equal(h.document.getElementById('consent').hidden, false);
  assert.equal(h.backend.calls.length, 0);
  assert.equal(h.document.getElementById('contact-select').value, dynamicCatalog[2].id);
});

test('contact names sort A–Z across Chinese pinyin and English without changing contact IDs or selection', async () => {
  const rows = [
    ['zhao', '赵', 'zhao'], ['zoe', 'Zoe', 'zoe'], ['bob', 'Bob', 'bob'],
    ['zhang', '张', 'zhang'], ['an', '安', 'an'], ['bai', '白', 'bai'], ['alice', 'Alice', 'alice'],
  ].map(([id, name, sort_name]) => ({ id: `synthetic-${id}`, name, sort_name,
    analysis_enabled: false, available: true }));
  const h = await harness({ catalog: rows });
  const select = h.document.getElementById('contact-select');
  assert.deepEqual(select.children.map(group => group.label), ['A', 'B', 'Z']);
  assert.deepEqual(contactOptions(h.document).map(option => option.textContent), ['alice', '安', '白', 'bob', '张', '赵', 'zoe']);
  assert.deepEqual(contactOptions(h.document).map(option => option.value),
    ['alice', 'an', 'bai', 'bob', 'zhang', 'zhao', 'zoe'].map(id => `synthetic-${id}`));
  assert.equal(h.debug().target, 'synthetic-zhao');
  assert.equal(select.value, 'synthetic-zhao');
  await h.switchContact('synthetic-an');
  assert.equal(select.value, 'synthetic-an');
  assert.deepEqual(contactOptions(h.document).map(option => option.textContent), ['alice', '安', '白', 'bob', '张', '赵', 'zoe']);
});

test('Chinese contacts without a romanized sort_name still sort by pinyin', async () => {
  const rows = ['张', '白', '安'].map((name, i) => ({ id: `synthetic-pinyin-${i}`, name,
    analysis_enabled: false, available: true }));
  const h = await harness({ catalog: rows });
  assert.deepEqual(contactOptions(h.document).map(option => option.textContent), ['安', '白', '张']);
});

test('uppercase alphabet groups contain lowercase Latin labels and put miscellaneous names last', async () => {
  const rows = [
    ['digits', '123朋友', '#123'], ['symbol', '_TEAM', '#team'], ['bob', 'BOB', 'bob'],
    ['alice', 'Alice张', 'alicezhang'], ['zoe', 'ZOE', 'zoe'],
  ].map(([id, name, sort_name]) => ({ id: `synthetic-${id}`, name, sort_name,
    available: true, analysis_enabled: false }));
  const h = await harness({ catalog: rows });
  const groups = h.document.getElementById('contact-select').children;
  assert.deepEqual(groups.map(group => group.label), ['A', 'B', 'Z', '#']);
  assert.equal(groups.every(group => group.tagName === 'optgroup'), true);
  assert.deepEqual(groups[0].children.map(option => option.textContent), ['alice张']);
  assert.deepEqual(groups[1].children.map(option => option.textContent), ['bob']);
  assert.deepEqual(groups[2].children.map(option => option.textContent), ['zoe']);
  assert.deepEqual(groups[3].children.map(option => option.textContent), ['123朋友', '_team']);
  assert.equal(h.debug().target, 'synthetic-digits');
});

test('the selected opponent quote has an explicit reanalysis button that preserves its message boundary', async () => {
  const h = await harness();
  await completeFirst(h, 'CURRENT initial opponent analysis');
  const quote = h.document.getElementById('selected-quote');
  const button = h.document.getElementById('analyze-selected');
  assert.equal(h.document.getElementById('selected-message').hidden, false);
  assert.equal(quote.textContent, messages[0].text);
  assert.equal(button.disabled, false);
  await button.onclick();
  assert.equal(button.disabled, true, 'An active analysis cannot be submitted twice');
  assert.equal(h.debug().selected_id, messages[0].id);
  assert.equal(quote.textContent, messages[0].text);
  assert.deepEqual(h.backend.calls, [messages[0].id, messages[0].id]);
  h.completeJob(messages[0].id, 'CURRENT explicit opponent analysis');
  await h.tick();
  assert.equal(h.debug().result_selected_id, messages[0].id);
  assert.equal(button.disabled, false);
  assert.equal(h.document.getElementById('feed').textContent.includes('CURRENT explicit'), true);
});

test('a server self-message selection clears the quote and results without reading context or analyzing', async () => {
  const h = await harness();
  await completeFirst(h, 'OBSOLETE opponent result before self click');
  const before = h.backend.requests.filter(request => ['/api/history/context', '/api/analyze/history'].includes(request.url)).length;
  h.backend.selection = { status: 'self_message', revision: 'synthetic-self-click',
    target: contacts[0], contact_revision: h.backend.contact_revision,
    message: { id: 'e'.repeat(64), role: 'self', timestamp: 1700000200, text: 'Synthetic own message' } };
  await h.tick();
  h.assertOldHidden();
  assert.equal(h.debug().selected_id, undefined);
  assert.equal(h.debug().revision, 'synthetic-self-click');
  assert.equal(h.document.getElementById('selected-message').hidden, true);
  assert.equal(h.document.getElementById('selected-quote').textContent, '');
  assert.equal(h.document.getElementById('analyze-selected').disabled, true);
  assert.equal(h.document.getElementById('status').textContent, '请选择对方消息');
  await h.document.getElementById('analyze-selected').onclick();
  await h.document.getElementById('refresh').onclick();
  await h.tick();
  assert.equal(h.backend.requests.filter(request => ['/api/history/context', '/api/analyze/history'].includes(request.url)).length, before);
});

test('even a matched message with self or unknown role cannot pass the opponent-only frontend boundary', async () => {
  const own = { id: 'e'.repeat(64), role: 'self', timestamp: 1700000200, text: 'Synthetic own message' };
  const unknown = { id: 'f'.repeat(64), role: 'unknown', timestamp: 1700000201, text: 'Synthetic unresolved sender' };
  const h = await harness({ messageSets: { [contacts[0]]: [...messages, own, unknown] } });
  await h.click(own.id);
  assert.equal(h.backend.calls.length, 0);
  assert.equal(h.backend.requests.some(request => request.url === '/api/history/context'), false);
  assert.equal(h.document.getElementById('selected-quote').textContent, '');
  assert.equal(h.document.getElementById('status').textContent, '请选择对方消息');
  await h.click(unknown.id);
  assert.equal(h.backend.calls.length, 0);
  assert.equal(h.backend.requests.some(request => request.url === '/api/history/context'), false);
  assert.equal(h.debug().selected_id, undefined);
  assert.equal(h.document.getElementById('analyze-selected').disabled, true);
  assert.equal(h.document.getElementById('status').textContent, '请选择对方消息');
});

test('a late opponent job cannot restore recommendations after the user selects their own message', async () => {
  const h = await harness();
  const initial = h.click(messages[0].id);
  await flush();
  h.releaseSubmission();
  await initial;
  h.backend.selection = { status: 'self_message', revision: 'synthetic-own-after-pending',
    target: contacts[0], contact_revision: h.backend.contact_revision };
  await h.tick();
  h.completeJob(messages[0].id, 'OBSOLETE result after own click');
  await h.tick();
  h.assertOldHidden();
  assert.equal(h.document.getElementById('selected-message').hidden, true);
  assert.equal(h.document.getElementById('analyze-selected').disabled, true);
  assert.equal(h.document.getElementById('status').textContent, '请选择对方消息');
  assert.equal(h.backend.calls.length, 1);
});

test('six contextual JEV actions remain intact while the UI shows the best three and the global winner next step', async () => {
  const h = await harness();
  const labels = ['夸一句', '安慰一下', '先保持沉默', '认真追问', '接住这个玩笑', '提出一起行动'];
  const scores = [.02, .11, .22, .18, .05, .42];
  const ranked = labels.map((action, i) => ({ id: `synthetic-choice-${i + 1}`, action,
    detail: `Synthetic contextual next step ${i + 1}`, score: scores[i] }));
  const options = ranked.map(({ score, ...choice }) => ({ ...choice }));
  const before = ranked.map(choice => choice.id);
  const proposal = { ...resultFor(messages[0].id, 'CURRENT broader contextual choices').proposal, options };
  await completeFirst(h, 'CURRENT broader contextual choices', {
    proposal, decision: { ranked }, card: { next_step: 'OBSOLETE positional next step' },
  });
  const actions = h.document.querySelectorAll('.action');
  assert.equal(actions.length, 3);
  assert.deepEqual(actions.map(action => action.children[1].textContent), [labels[5], labels[2], labels[3]]);
  assert.deepEqual(actions.map(action => action.children[2].textContent), ['4.2/10', '2.2/10', '1.8/10']);
  assert.equal(h.document.querySelectorAll('.next-step')[0].textContent, ranked[5].detail);
  assert.equal(h.document.getElementById('feed').textContent.includes('OBSOLETE'), false);
  assert.equal(options.length, 6);
  assert.equal(ranked.length, 6);
  assert.deepEqual(ranked.map(choice => choice.id), before, 'Rendering must not prefilter or reorder the scoring result');
  assert.deepEqual(options.map(choice => choice.action), labels);
});

test('a six-option MiniCPM partial card previews three dynamic choices without fabricating JEV scores', async () => {
  const h = await harness();
  const click = h.click(messages[0].id);
  await flush();
  h.releaseSubmission();
  await click;
  const labels = ['温柔地接住情绪', '直接说明自己的想法', '轻轻开个玩笑', '承认刚才忽略了对方', '换个时间再聊', '邀请对方一起想办法'];
  const options = labels.map((action, i) => ({ id: `synthetic-partial-${i + 1}`, action,
    detail: `Synthetic partial detail ${i + 1}` }));
  h.backend.job.proposal = { ...resultFor(messages[0].id, 'CURRENT six-option partial').proposal, options };
  await h.tick();
  assert.deepEqual(h.document.querySelectorAll('.action').map(action => action.children[1].textContent), labels.slice(0, 3));
  assert.deepEqual(h.document.querySelectorAll('.score').map(score => score.textContent), ['—', '—', '—']);
  assert.equal(h.document.querySelectorAll('.score-track').length, 0);
  assert.equal(h.document.querySelectorAll('.next-step')[0].textContent, '正在选择…');
  assert.equal(options.length, 6);
  assert.deepEqual(options.map(choice => choice.action), labels);
});

test('analysis-only MiniCPM output remains visible while the complete JEV bank is still being scored', async () => {
  const h = await harness();
  const click = h.click(messages[0].id);
  await flush();
  h.releaseSubmission();
  await click;
  const proposal = { analysis: 'CURRENT opponent expression analysis without prescreened options',
    context_notes: ['Synthetic earlier conversation note'], options: [] };
  h.backend.job.proposal = proposal;
  await h.tick();
  assert.equal(h.document.querySelectorAll('.decision-card').length, 1);
  assert.equal(h.document.querySelectorAll('.analysis')[0].textContent, proposal.analysis);
  assert.equal(h.document.querySelectorAll('.actions-section').length, 0);
  assert.equal(h.document.querySelectorAll('.action').length, 0);
  assert.equal(h.document.querySelectorAll('.score-track').length, 0);
  assert.equal(h.document.querySelectorAll('.next-step')[0].textContent, '正在选择…');
  assert.equal(h.document.querySelectorAll('.emotion-chip').length, 0);
  assert.deepEqual(proposal.options, []);
});

test('a complete 32-choice JEV bank is preserved and displayed as three two-character actions with the global winner detail', async () => {
  const h = await harness();
  const labels = ['夸奖', '安慰', '沉默', '撒娇', '调侃', '共情', '认同', '鼓励',
    '关心', '追问', '澄清', '解释', '拒绝', '道歉', '感谢', '邀约',
    '示弱', '打趣', '回应', '倾听', '建议', '转移', '等待', '确认',
    '承诺', '求助', '分享', '祝福', '拥抱', '讨好', '约定', '接梗'];
  const winners = [31, 16, 2];
  const scores = new Map([[31, .43], [16, .18], [2, .10]]);
  const ranked = labels.map((action, index) => ({ id: `synthetic-bank-${index}`, action,
    score: scores.get(index) || .01,
    detail: winners.includes(index) ? `Synthetic enriched winner detail ${index}` : '' }));
  const originalIds = ranked.map(choice => choice.id);
  const proposal = { analysis: 'CURRENT full-bank opponent analysis', context_notes: [], options: [] };
  await completeFirst(h, proposal.analysis, { proposal, decision: { ranked, evaluated_count: 32 },
    card: { next_step: 'OBSOLETE prescreened choice detail' } });
  const visible = h.document.querySelectorAll('.action');
  assert.equal(visible.length, 3);
  assert.deepEqual(visible.map(action => action.children[1].textContent), winners.map(index => labels[index]));
  assert.deepEqual(visible.map(action => action.children[2].textContent), ['4.3/10', '1.8/10', '1.0/10']);
  assert.equal(visible.every(action => action.children[1].textContent.length === 2), true);
  assert.equal(h.document.querySelectorAll('.next-step')[0].textContent, ranked[31].detail);
  assert.deepEqual(h.document.querySelectorAll('.section-heading').map(heading => heading._text),
    ['分析', '动作排名', '下一步']);
  assert.equal(h.document.getElementById('feed').textContent.includes('OBSOLETE'), false);
  assert.equal(ranked.length, 32);
  assert.deepEqual(ranked.map(choice => choice.id), originalIds);
  assert.deepEqual(proposal.options, [], 'The frontend must not construct a MiniCPM prescreening list');
});

test('a late same-contact search cannot replace the latest query or surface an obsolete failure', async () => {
  const h = await harness();
  const held = h.hold('/api/history/search', contacts[0]);
  h.document.getElementById('history-query').value = 'first query';
  const first = h.document.getElementById('history-search').onclick();
  await flush();
  h.document.getElementById('history-query').value = 'latest query';
  await h.document.getElementById('history-search').onclick();
  const expected = h.historyIds();
  held.release({ messages: [messages[0]] });
  await first;
  assert.equal(h.historyIds(), expected);

  const failed = h.hold('/api/history/search', contacts[0]);
  const obsolete = h.document.getElementById('history-search').onclick();
  await flush();
  await h.document.getElementById('history-search').onclick();
  failed.release({ error: 'OBSOLETE query failure' }, 500);
  await obsolete;
  assert.equal(h.document.getElementById('error').hidden, true);
});

test('search supersedes an in-flight recent page and recent supersedes an in-flight search', async () => {
  const h = await harness();
  const page = h.hold('/api/history/recent', contacts[0]);
  const reload = h.document.getElementById('history-reload').onclick();
  await flush();
  const search = h.hold('/api/history/search', contacts[0]);
  const querying = h.document.getElementById('history-search').onclick();
  await flush();
  search.release({ messages: [] });
  await querying;
  page.release({ messages, before: null, has_more: false });
  await reload;
  assert.equal(h.historyIds(), '');
  assert.equal(h.document.getElementById('history-empty').textContent, '没有找到相关消息');

  const olderQuery = h.hold('/api/history/search', contacts[0]);
  const pending = h.document.getElementById('history-search').onclick();
  await flush();
  await h.document.getElementById('history-reload').onclick();
  olderQuery.release({ messages: [] });
  await pending;
  assert.equal(h.historyIds(), messages.map(message => message.id).join(','));
});

test('history pagination deduplicates message anchors while preserving chronological order', async () => {
  const h = await harness();
  const held = h.hold('/api/history/recent', contacts[0]);
  const older = h.document.getElementById('older').onclick();
  await flush();
  assert.equal(h.document.getElementById('older').disabled, true);
  held.release({ messages: [messages[0]], before: null, has_more: false });
  await older;
  assert.equal(h.historyIds(), messages.map(message => message.id).join(','));
  assert.equal(h.document.getElementById('older').disabled, false);
});

test('selection POSTs are serialized so a late earlier request cannot restore the old backend anchor', async () => {
  const h = await harness();
  const held = h.hold('/api/history/select', contacts[0]);
  const older = h.click(messages[0].id);
  await flush();
  const newer = h.click(messages[1].id);
  await flush();
  const requests = () => h.backend.requests.filter(request => request.url === '/api/history/select');
  assert.equal(requests().length, 1);
  const polls = h.backend.requests.filter(request => request.url === '/api/status').length;
  await h.tick();
  assert.equal(h.backend.requests.filter(request => request.url === '/api/status').length, polls);
  h.backend.selection = { status: 'matched', revision: `revision-${++h.backend.revision}`,
    message: messages[0], target: contacts[0], contact_revision: h.backend.contact_revision };
  held.release(h.backend.selection);
  await older;
  await flush();
  assert.equal(requests().length, 2);
  assert.equal(h.backend.selection.message.id, messages[1].id);
  assert.deepEqual(h.backend.calls, [messages[1].id]);
  h.releaseSubmission();
  await newer;
  await h.tick();
  assert.equal(h.debug().selected_id, messages[1].id);
});

test('a failed older selection POST cannot block or show an error for the newer selected sentence', async () => {
  const h = await harness();
  const held = h.hold('/api/history/select', contacts[0]);
  const older = h.click(messages[0].id);
  await flush();
  const newer = h.click(messages[1].id);
  await flush();
  held.release({ error: 'OBSOLETE selection failure' }, 500);
  await older;
  await flush();
  h.releaseSubmission();
  await newer;
  assert.equal(h.debug().selected_id, messages[1].id);
  assert.equal(h.document.getElementById('error').hidden, true);
});

test('an old context response cannot start analysis after a newer local click intent', async () => {
  const h = await harness();
  const held = h.hold('/api/history/context', contacts[0]);
  const older = h.click(messages[0].id);
  await flush();
  const newer = h.click(messages[1].id);
  await flush();
  held.release({ messages: [], context_signature: signature(messages[0].id) });
  await older;
  assert.deepEqual(h.backend.calls, [messages[1].id]);
  h.releaseSubmission();
  await newer;
});

test('a status snapshot requested before a click cannot clear the newly selected sentence', async () => {
  const h = await harness();
  await completeFirst(h);
  const held = h.hold('/api/status', undefined);
  const oldStatus = h.tick();
  await flush();
  await h.click(messages[1].id);
  held.release({ ready: true, active_target: contacts[0], contact_revision: h.backend.contact_revision,
    selection: null, job: null });
  await oldStatus;
  assert.equal(h.debug().selected_id, messages[1].id);
  assert.equal(h.debug().busy, true);
});

test('polling remains single-flight and clears only a recovered connection error', async () => {
  const h = await harness();
  const held = h.hold('/api/status', undefined);
  const polling = h.tick();
  await flush();
  const count = h.backend.requests.filter(request => request.url === '/api/status').length;
  await h.tick();
  assert.equal(h.backend.requests.filter(request => request.url === '/api/status').length, count);
  held.release({ error: 'Synthetic local connection failure' }, 503);
  await polling;
  assert.equal(h.document.getElementById('error').hidden, false);
  await h.tick();
  assert.equal(h.document.getElementById('error').hidden, true);

  const selected = h.click(messages[0].id);
  await flush();
  h.releaseSubmission();
  await selected;
  h.backend.job.status = 'error';
  h.backend.job.error = 'Synthetic writing failure';
  await h.tick();
  await h.tick();
  assert.equal(h.document.getElementById('error').textContent, 'Synthetic writing failure');
  assert.equal(h.document.getElementById('error').hidden, false);
});

test('a busy worker holds a new automatic analysis until the actual inference slot is free', async () => {
  const h = await harness();
  const older = h.click(messages[0].id);
  await flush();
  h.releaseSubmission();
  await older;
  await h.click(messages[1].id);
  h.completeJob(messages[0].id, 'OBSOLETE completed worker result');
  h.backend.worker_busy = true;
  await h.tick();
  assert.equal(h.backend.calls.length, 1);
  h.assertOldHidden();
  h.backend.worker_busy = false;
  await h.tick();
  assert.deepEqual(h.backend.calls, messages.map(message => message.id));
});

test('Enter submits history search without interrupting an active Chinese input composition', async () => {
  const h = await harness();
  const input = h.document.getElementById('history-query');
  let prevented = 0;
  const count = () => h.backend.requests.filter(request => request.url === '/api/history/search').length;
  const initial = count();
  input.onkeydown({ key: 'Enter', isComposing: true, preventDefault: () => prevented++ });
  await flush();
  assert.equal(count(), initial);
  input.value = '  synthetic query  ';
  input.onkeydown({ key: 'Enter', isComposing: false, preventDefault: () => prevented++ });
  await flush();
  assert.equal(prevented, 1);
  assert.equal(count(), initial + 1);
  assert.equal(h.backend.requests.at(-1).body.query, 'synthetic query');
});
