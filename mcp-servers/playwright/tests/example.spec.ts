import { test, expect } from '@playwright/test';

test('fastapi health endpoint', async ({ request }) => {
  const r = await request.get('http://fastapi:8000/health');
  expect(r.status()).toBe(200);
  const body = await r.json();
  expect(body.status).toBe('ok');
});
