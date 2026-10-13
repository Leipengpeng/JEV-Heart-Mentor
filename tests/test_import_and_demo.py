"""The fresh-clone path uses authored JSON, never a WeChat account."""

import json
import copy
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from chatlens.demo import DemoHandler, demo_data
from chatlens.history_store import HistoryStore
from chatlens.import_history import import_json, load_export
from chatlens.launcher import read_session


class JSONImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "input.json"

    def data(self, contacts):
        self.path.write_text(json.dumps({"contacts": contacts}))
        return self.path

    def contact(self):
        return {
            "id": "synthetic-person",
            "name": "合成好友",
            "messages": [{"server_id": "100", "timestamp": 1700000000, "role": "other", "text": "明天三点去看展。"}],
        }

    def test_import_is_repeatable_and_not_implicitly_approved(self):
        store = HistoryStore(self.root / "memory.sqlite3")
        path = self.data([self.contact()])
        self.assertEqual(import_json(path, store), {"contacts": 1, "messages_added": 1, "analysis_enabled": False})
        self.assertEqual(import_json(path, store)["messages_added"], 0)
        self.assertEqual(store.stats()["contacts"], 1)
        self.assertFalse((self.root / "analysis-approvals.json").exists())

    def test_invalid_later_contact_does_not_partially_import(self):
        store = HistoryStore(self.root / "memory.sqlite3")
        bad = self.contact()
        bad["id"] = "synthetic-invalid"
        bad["messages"][0]["role"] = "system"
        with self.assertRaises(ValueError):
            import_json(self.data([self.contact(), bad]), store)
        self.assertEqual(store.stats()["contacts"], 0)

    def test_rejects_duplicate_id_and_boolean_timestamp(self):
        item = self.contact()
        with self.assertRaises(ValueError):
            load_export(self.data([item, item]))
        item["messages"][0]["timestamp"] = True
        with self.assertRaises(ValueError):
            load_export(self.data([item]))

    def test_session_reader_rejects_corruption_and_non_loopback_inputs(self):
        self.path.write_text("not-json")
        self.assertIsNone(read_session(self.path))
        self.path.write_text(json.dumps({"port": -1, "token": "a" * 64}))
        self.assertIsNone(read_session(self.path))
        self.path.write_text(json.dumps({"port": 9000, "token": "a" * 64}))
        self.assertEqual(read_session(self.path)["port"], 9000)


class DemoHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from http.server import ThreadingHTTPServer

        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), DemoHandler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.http.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join()

    def test_demo_has_explicit_synthetic_marker_and_six_matching_cards(self):
        with urllib.request.urlopen(self.base) as response:
            body = response.read().decode()
        self.assertIn("合成演示", body)
        self.assertIn("window.CARD_PREVIEW=", body)
        data = demo_data()
        self.assertEqual(len(data["contacts"]), 2)
        for contact in data["contacts"].values():
            for index, result in contact["results"].items():
                self.assertEqual(contact["messages"][index]["role"], "other")
                self.assertTrue(result["demo"])
                self.assertEqual(len(result["decision"]["ranked"]), 3)

    def test_demo_does_not_expose_arbitrary_files_or_private_api(self):
        for path in ("/../pyproject.toml", "/api/status", "/private/memory.sqlite3"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(self.base + path)
            self.assertEqual(caught.exception.code, 404)

    def test_demo_rejects_host_rebinding(self):
        request = urllib.request.Request(self.base, headers={"Host": "attacker.invalid"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 403)


class ImportedConsentTests(unittest.TestCase):
    def test_json_history_can_be_enabled_without_native_keys_or_adapter(self):
        from chatlens import server, history_connector
        from chatlens.contact_catalog import ContactCatalog
        from http.server import ThreadingHTTPServer

        with tempfile.TemporaryDirectory() as directory:
            private = Path(directory)
            store = HistoryStore(private / "memory.sqlite3")
            import_json(Path(__file__).resolve().parents[1] / "examples/conversation.json", store)

            def unavailable():
                raise ValueError("native adapter is intentionally absent")

            catalog = ContactCatalog(private, loader=unavailable)
            catalog.refresh(store)
            state = copy.deepcopy(server.STATE)
            state.update(active_target=None, display_name="", selection=None, job=None)
            with (
                mock.patch.object(server, "STATE", state),
                mock.patch.object(server, "HISTORY", store),
                mock.patch.object(server, "CATALOG", catalog),
                mock.patch.object(server, "CONTACT_STATE", private / "contact-state.json"),
                mock.patch.object(history_connector, "PRIVATE", private),
            ):
                http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
                thread = threading.Thread(target=http.serve_forever, daemon=True)
                thread.start()

                def request(path, body):
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{http.server_port}" + path,
                        data=json.dumps(body).encode(),
                        headers={"X-ChatLens-Token": server.TOKEN},
                    )
                    with urllib.request.urlopen(req, timeout=3) as response:
                        return json.load(response)

                try:
                    contact = request("/api/contact/select", {"target": "synthetic-friend"})
                    self.assertTrue(contact["requires_consent"])
                    enabled = request(
                        "/api/contact/enable",
                        {"target": "synthetic-friend", "contact_revision": contact["contact_revision"]},
                    )
                    self.assertTrue(enabled["analysis_enabled"])
                    self.assertEqual(enabled["selection"]["message"]["role"], "other")
                    self.assertFalse((private / "runtime-candidates.txt").exists())
                finally:
                    http.shutdown()
                    http.server_close()
                    thread.join()
