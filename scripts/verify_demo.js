'use strict';

// Optional browser check: install Playwright separately, then start chatlens demo.
// Only the authored synthetic demo is allowed; never capture live private chats.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

async function main() {
  const base = process.argv[2] || 'http://127.0.0.1:8769';
  const url = new URL(base);
  assert(['127.0.0.1', 'localhost'].includes(url.hostname));
  const browser = await chromium.launch({ headless: true, ...(process.env.CHATLENS_BROWSER_CHANNEL ? {channel: process.env.CHATLENS_BROWSER_CHANNEL} : {}) });
  const page = await browser.newPage({ viewport: { width: 1050, height: 760 }, deviceScaleFactor: 1 });
  const failures = [];
  page.on('pageerror', error => failures.push(error.message));
  page.on('requestfailed', request => failures.push(request.failure()?.errorText || 'network error'));
  try {
    await page.goto(base);
    await page.locator('.demo-banner').filter({ hasText: '合成演示' }).waitFor();
    const contactCount = await page.locator('#contact-select option').count();
    assert.equal(contactCount, 2);
    await page.locator('.history-message.other').first().click();
    await page.locator('.decision-card').waitFor();
    assert.equal(await page.locator('.card-section').count(), 3);
    assert.equal(await page.locator('.emotion-chip').count(), 3);
    assert.equal(await page.locator('.action-list .action').count(), 3);
    assert((await page.locator('#selected-message').boundingBox()).height < 110);
    const firstAnalysis = await page.locator('.analysis').textContent();
    await page.locator('.history-message.other').last().click();
    assert.notEqual(await page.locator('.analysis').textContent(), firstAnalysis);
    const screenshotDir = path.resolve(__dirname, '../docs/assets');
    fs.mkdirSync(screenshotDir, { recursive: true });
    await page.screenshot({ path: path.join(screenshotDir, 'demo-desktop.png') });
    const current = await page.locator('#contact-select').inputValue();
    const next = await page.locator('#contact-select option').evaluateAll((options, value) => options.find(o => o.value !== value).value, current);
    await page.selectOption('#contact-select', next);
    assert.equal(await page.locator('.decision-card').count(), 0);
    await page.locator('.history-message.other').last().click();
    assert.equal(await page.locator('.card-section').count(), 3);
    await page.setViewportSize({ width: 360, height: 760 });
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    await page.screenshot({ path: path.join(screenshotDir, 'demo-narrow.png'), fullPage: true });
    assert.deepEqual(failures, []);
    console.log(JSON.stringify({ passed: true, contacts: contactCount, sections: 3, top_emotions: 3, top_actions: 3,
      changed_message_changes_card: true, changed_contact_clears_card: true, narrow_overflow: false, browser_errors: 0 }));
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
