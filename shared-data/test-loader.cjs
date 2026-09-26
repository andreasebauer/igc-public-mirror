// Transport integrity tests only; not Decoder science or qualification.
const {load} = require('../microscope/shared-data.js');
const {webcrypto, createHash} = require('node:crypto');
const assert = require('node:assert/strict');
const fs = require('node:fs');

(async () => {
  const raw = Buffer.from('{"schema":"ig.reverse-microscope/1.1","value":"<sample>"}');
  const sha256 = createHash('sha256').update(raw).digest('hex');
  const descriptor = {sha256, bytes: raw.length, key: `objects/sha256/${sha256.slice(0,2)}/${sha256}`};
  const config = {schema: 'ig.shared-data.config/1', base_url: 'https://data.example.test/', microscope: descriptor};
  let calls = 0;
  const fetchGood = async (url, options) => {
    calls++; assert.equal(url, config.base_url + descriptor.key);
    assert.equal(options.credentials, 'omit'); assert.equal(options.redirect, 'error');
    return new Response(raw);
  };
  assert.equal(await load(config, fetchGood, webcrypto), raw.toString());
  assert.equal(calls, 1);
  assert.equal(await load({...config, base_url: null, microscope: null}, () => {throw Error('unexpected fetch');}), null);
  await assert.rejects(load(config, async () => new Response(Buffer.alloc(raw.length, 65)), webcrypto), /SHA-256/);
  await assert.rejects(load(config, async () => new Response(raw.subarray(1)), webcrypto), /length/);
  await assert.rejects(load(config, async () => new Response(Buffer.concat([raw, raw])), webcrypto), /exceeds/);
  await assert.rejects(load(config, async () => new Response('', {status: 404}), webcrypto), /404/);
  await assert.rejects(load({...config, base_url: 'http://example.test'}, fetchGood, webcrypto), /HTTPS/);
  await assert.rejects(load({...config, microscope: {...descriptor, key: '../private'}}, fetchGood, webcrypto), /descriptor/);
  if (process.argv[2]) {
    const dir = process.argv[2];
    const catalogue = JSON.parse(fs.readFileSync(dir + '/catalogue.json'));
    const actual = fs.readFileSync(dir + '/' + catalogue.microscope.key);
    const loaded = await load({...config, microscope: catalogue.microscope}, async () => new Response(actual), webcrypto);
    assert.equal(loaded, actual.toString('utf8'));
    assert.equal(JSON.parse(loaded).dataset.id, catalogue.microscope.dataset_id);
    console.log('Actual staged Microscope pack: byte-identical browser transport PASS');
  }
  console.log('Transport acceptance/corruption/truncation/oversize/unavailable/configuration checks PASS');
})().catch(error => { console.error(error); process.exitCode = 1; });
