from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_auth_postback_tunnel_allowed_origin():
    origin = 'https://probable-space-telegram-jq94gjprwj5fjpqp-7860.app.github.dev'
    r = client.get('/auth/postback/tunnel', headers={'Origin': origin})
    assert r.status_code == 200
    assert r.headers.get('access-control-allow-origin') == origin

def test_pf_signin_allowed_origin():
    origin = 'https://github.dev'
    r = client.get('/pf-signin', headers={'Origin': origin})
    assert r.status_code == 200
    assert r.headers.get('access-control-allow-origin') == origin

def test_auth_postback_tunnel_disallowed_origin():
    origin = 'https://evil.example.com'
    r = client.get('/auth/postback/tunnel', headers={'Origin': origin})
    assert r.status_code == 200
    assert r.headers.get('access-control-allow-origin') is None

def test_pf_signin_disallowed_origin():
    origin = 'https://evil.example.com'
    r = client.get('/pf-signin', headers={'Origin': origin})
    assert r.status_code == 200
    assert r.headers.get('access-control-allow-origin') is None
