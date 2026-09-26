/* Transport-only loader. Scientific validation remains in the existing viewer. */
(function (root) {
  'use strict';
  async function load(config, fetchImpl = fetch, cryptoImpl = crypto) {
    if (!config || config.schema !== 'ig.shared-data.config/1') throw new Error('Invalid data source configuration');
    if (config.base_url === null && config.microscope === null) return null;
    const entry = config.microscope;
    if (!entry || !/^[0-9a-f]{64}$/.test(entry.sha256) ||
        !Number.isSafeInteger(entry.bytes) || entry.bytes < 1 || entry.bytes > 24 * 1024 * 1024 ||
        entry.key !== `objects/sha256/${entry.sha256.slice(0, 2)}/${entry.sha256}`) {
      throw new Error('Invalid pinned dataset descriptor');
    }
    const base = new URL(config.base_url);
    if (base.protocol !== 'https:' || base.username || base.password || base.search || base.hash) {
      throw new Error('Data endpoint must be a plain HTTPS base URL');
    }
    const url = new URL(entry.key, base.href.replace(/\/?$/, '/'));
    const response = await fetchImpl(url.href, {credentials: 'omit', redirect: 'error'});
    if (!response.ok) throw new Error(`Dataset download failed (${response.status})`);
    const reader = response.body.getReader(), chunks = [];
    let size = 0;
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > entry.bytes) { await reader.cancel(); throw new Error('Dataset exceeds declared size'); }
      chunks.push(value);
    }
    if (size !== entry.bytes) throw new Error('Dataset length mismatch');
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    const hash = [...new Uint8Array(await cryptoImpl.subtle.digest('SHA-256', bytes))]
      .map(b => b.toString(16).padStart(2, '0')).join('');
    if (hash !== entry.sha256) throw new Error('Dataset SHA-256 mismatch');
    const text = new TextDecoder('utf-8', {fatal: true}).decode(bytes);
    if (JSON.parse(text).schema !== 'ig.reverse-microscope/1.1') throw new Error('Unsupported dataset schema');
    return text;
  }
  root.IGSharedData = {load};
  if (typeof module !== 'undefined' && module.exports) module.exports = {load};
})(globalThis);
