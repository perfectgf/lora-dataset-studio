import test from 'node:test';
import assert from 'node:assert/strict';
import { pexelsAttribution } from './pexelsAttribution.js';

const valid = {
  platform: 'pexels',
  source_url: 'https://www.pexels.com/photo/example-123/',
  photographer: '  Jane   Doe  ',
  photographer_url: 'https://www.pexels.com/@jane-doe/',
};

test('Pexels attribution normalizes the credit and preserves allowlisted links', () => {
  assert.deepEqual(pexelsAttribution(valid), {
    photographer: 'Jane Doe',
    sourceUrl: valid.source_url,
    photographerUrl: valid.photographer_url,
  });
});

test('Pexels attribution rejects arbitrary or credentialed links', () => {
  assert.equal(pexelsAttribution({ ...valid, source_url: 'https://evil.example/photo' }), null);
  assert.equal(pexelsAttribution({
    ...valid, photographer_url: 'https://user:pass@www.pexels.com/@jane/',
  }), null);
  assert.equal(pexelsAttribution({ ...valid, source_url: 'http://www.pexels.com/photo/1/' }), null);
});

test('Pexels attribution rejects unknown platforms and unbounded photographer names', () => {
  assert.equal(pexelsAttribution({ ...valid, platform: 'other' }), null);
  assert.equal(pexelsAttribution({ ...valid, photographer: 'x'.repeat(161) }), null);
});
