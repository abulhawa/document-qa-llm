import utils.ingest_plans as ip


def test_ingest_plans_flow(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.docs = {}

        def index(self, index, id, body):
            self.docs[id] = body

        def update(self, index, id, body):
            if id in self.docs:
                self.docs[id].update(body["doc"])

        def search(self, index, body):
            hits = [{"_source": doc} for doc in self.docs.values()]
            return {"hits": {"hits": hits}}

        def delete_by_query(self, index, body):
            self.docs = {}

    client = FakeClient()
    monkeypatch.setattr("utils.ingest_plans.get_client", lambda: client)
    monkeypatch.setattr("utils.ingest_plans.ensure_ingest_plan_index_exists", lambda: None)

    ip.add_planned_ingestions(["a.txt", "b.txt"])
    res = ip.get_planned_ingestions()
    assert len(res) == 2

    ip.update_plan_status("a.txt", "Completed")
    res2 = ip.get_planned_ingestions()
    status_map = {d["path"]: d["status"] for d in res2}
    assert status_map["a.txt"] == "Completed"

    ip.clear_planned_ingestions()
    assert client.docs == {}
