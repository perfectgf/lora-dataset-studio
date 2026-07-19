import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildLocaleRegistry,
  detectLocale,
  withCatalogAliases,
} from './localeRegistry.js';

test('locale registry discovers JSON modules and uses file metadata', () => {
  const { catalogs, locales } = buildLocaleRegistry({
    './locales/fr.json': {
      default: {
        _meta: {
          locale: 'fr',
          label: 'Français',
          documentTitle: 'Studio de datasets LoRA',
          order: 20,
        },
        common: { save: 'Enregistrer' },
      },
    },
    './locales/en.json': {
      _meta: { label: 'English', order: 10 },
      common: { save: 'Save' },
    },
  });

  assert.deepEqual(locales.map((locale) => locale.code), ['en', 'fr']);
  assert.equal(locales[1].label, 'Français');
  assert.equal(locales[1].documentTitle, 'Studio de datasets LoRA');
  assert.equal(catalogs.fr.common.save, 'Enregistrer');
});

test('locale registry keeps compatibility aliases without requiring every group', () => {
  const catalog = withCatalogAliases({
    datasets: { captions: { title: 'Captions' } },
    workspace: { training: { title: 'Training' } },
  });

  assert.equal(catalog.workspace.captions.title, 'Captions');
  assert.equal(catalog.settings.training.title, 'Training');
});

test('locale detection honors saved, exact, base and regional defaults', () => {
  const locales = [
    { code: 'en', defaultForLanguage: true },
    { code: 'zh-CN', defaultForLanguage: true },
    { code: 'zh-TW', defaultForLanguage: false },
    { code: 'fr', defaultForLanguage: true },
  ];

  assert.equal(detectLocale({
    saved: 'fr', browserLanguages: ['en-US'], locales,
  }), 'fr');
  assert.equal(detectLocale({
    browserLanguages: ['zh-TW'], locales,
  }), 'zh-TW');
  assert.equal(detectLocale({
    browserLanguages: ['zh'], locales,
  }), 'zh-CN');
  assert.equal(detectLocale({
    browserLanguages: ['fr-CA'], locales,
  }), 'fr');
  assert.equal(detectLocale({
    browserLanguages: ['de-DE'], locales, defaultLocale: 'en',
  }), 'en');
});
