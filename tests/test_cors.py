from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_manifest_cors_allowed_origin():
    origin = 'https://probable-space-telegram-jq94gjprwj5fjpqp-7860.app.github.dev'
    r = client.get('/manifest.json', headers={'Origin': origin})
    assert r.status_code == 200
    # CORS middleware should add Access-Control-Allow-Origin for allowed origins
    assert r.headers.get('access-control-allow-origin') == origin

def test_manifest_cors_disallowed_origin():
    origin = 'https://evil.example.com'
    r = client.get('/manifest.json', headers={'Origin': origin})
    assert r.status_code == 200
    # disallowed origins should not be echoed
    assert r.headers.get('access-control-allow-origin') is None
