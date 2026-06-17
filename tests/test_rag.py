from backend.rag import RAG


def test_ingest_and_query():
    r = RAG()
    r.create_collection('test')
    ids = r.ingest_texts(['hello world', 'another document'], [{'path':'a.txt'}, {'path':'b.txt'}])
    assert len(ids) == 2
    res = r.query('hello', n_results=1)
    # Accept both chromadb-style and in-memory style results
    if isinstance(res, dict):
        # in-memory fallback
        assert 'documents' in res or 'ids' in res
    else:
        # chromadb object: ensure it has documents
        assert hasattr(res, 'documents') or hasattr(res, 'ids')
