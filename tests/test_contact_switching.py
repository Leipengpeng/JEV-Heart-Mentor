"""Synthetic contact consent, identity and race tests; no real private files read."""

import copy
import hashlib
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from chatlens.config import PACKAGE_DIR

PRODUCT = PACKAGE_DIR
from chatlens import server
from chatlens.history_store import HistoryStore
from chatlens.contact_catalog import ContactCatalog


def message(number, text):
    return {
        "local_id": number,
        "server_id": str(number),
        "source": "synthetic.db",
        "timestamp": 1700000000 + number,
        "sort_seq": number,
        "role": "other",
        "kind": 1,
        "text": text,
    }


class ContactSwitchingAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.http.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.private = Path(self.temp.name)
        self.store = HistoryStore(self.private / "memory.sqlite3")
        records = [
            ("synthetic-a", "好友甲", "", ""),
            ("synthetic-b", "好友乙", "", ""),
            ("synthetic-new", "新朋友", "", ""),
            ("synthetic-duplicate-a", "同名", "", ""),
            ("synthetic-duplicate-b", "同名", "", ""),
            ("group@chatroom", "群聊", "", ""),
            ("gh_public", "公众号", "", ""),
        ]
        tables = {"Msg_" + hashlib.md5(r[0].encode()).hexdigest() for r in records}
        self.catalog = ContactCatalog(
            self.private,
            ("好友甲", "好友乙"),
            loader=lambda: {
                "contacts": records,
                "history_tables": tables,
                "account_key": "synthetic-account",
                "source": "synthetic-local-db",
                "source_signature": "synthetic-snapshot",
            },
        )
        for target, name in [("synthetic-a", "好友甲"), ("synthetic-b", "好友乙")]:
            self.store.ingest(target, name, [message(1, name + "此前记录"), message(2, name + "点击记录")])
        self.catalog.refresh(self.store)
        self.ids = {t: self.store.recent(t)["messages"][-1]["id"] for t in ("synthetic-a", "synthetic-b")}
        self.state = copy.deepcopy(server.STATE)
        self.state.update(
            active_target="synthetic-a",
            display_name="好友甲",
            contact_revision="synthetic-revision-1",
            contact_observed_at=0,
            contact_status="recognized",
            pending_locator=None,
            enabling_revision=None,
            selection=None,
            job=None,
            ready=False,
            error=None,
        )
        for patch in (
            mock.patch.object(server, "HISTORY", self.store),
            mock.patch.object(server, "CATALOG", self.catalog),
            mock.patch.object(server, "CONTACT_STATE", self.private / "contact-state.json"),
            mock.patch.object(server, "STATE", self.state),
            mock.patch.object(server, "history_source_signature", return_value="synthetic-source"),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def request(self, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, data=data, headers={"X-ChatLens-Token": server.TOKEN})
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def observe(self, title):
        return self.request("/api/contact/observe", {"title": title, "source": "ax"})

    def test_discovers_all_eligible_contacts_without_importing_new_histories(self):
        with mock.patch.object(server, "sync_history") as sync:
            status, result = self.request("/api/contacts")
        self.assertEqual(status, 200)
        self.assertEqual(len(result["contacts"]), 5)
        sync.assert_not_called()
        self.assertEqual(self.store.stats()["contacts"], 2)
        self.assertEqual(sum(c["analysis_enabled"] for c in result["contacts"]), 2)
        self.assertTrue(all("id" in c and "aliases" in c for c in result["contacts"]))
        self.assertFalse((self.private / "contact-state.json").exists())

    def test_switching_to_new_friend_clears_old_card_and_requires_consent(self):
        self.state["selection"] = {"target": "synthetic-a", "status": "matched"}
        self.state["job"] = {"id": "synthetic-job", "status": "done", "result": {"private": "synthetic"}}
        status, result = self.observe("新朋友")
        self.assertEqual(status, 200)
        self.assertEqual(result["active_target"], "synthetic-new")
        self.assertTrue(result["requires_consent"])
        self.assertFalse(result["analysis_enabled"])
        self.assertIsNone(self.state["selection"])
        self.assertIsNone(self.state["job"])
        self.assertGreater(result["contact_observed_at"], 0)

    def test_repeated_observation_does_not_change_revision_or_observed_time(self):
        _, first = self.observe("新朋友")
        _, second = self.observe("新朋友")
        self.assertEqual(first["contact_revision"], second["contact_revision"])
        self.assertEqual(first["contact_observed_at"], second["contact_observed_at"])

    def test_unknown_and_ambiguous_title_clear_old_advice_without_guessing(self):
        for title in ("同名", "不存在的合成联系人", "群聊"):
            self.state.update(active_target="synthetic-a", selection={"status": "matched"}, job={"status": "done"})
            _, result = self.observe(title)
            self.assertIsNone(result["active_target"])
            self.assertEqual(result["display_name"], title)
            self.assertFalse(result["can_analyze"])
            self.assertFalse(result["requires_consent"])
            self.assertIsNone(self.state["selection"])
            self.assertIsNone(self.state["job"])

    def test_preconsent_click_is_pending_and_every_history_entry_is_gated(self):
        _, contact = self.observe("新朋友")
        body = {
            "target": "synthetic-new",
            "contact_revision": contact["contact_revision"],
            "clicked_texts": ["合成新位置"],
            "source": "ax",
        }
        with mock.patch.object(server, "sync_history") as sync, mock.patch.object(server, "resolve_click") as resolve:
            code, result = self.request("/api/history/select-position", body)
            self.assertEqual(code, 200)
            self.assertEqual(result["status"], "requires_consent")
            self.assertIsNone(self.state["selection"])
            self.assertEqual(self.state["pending_locator"]["clicked_texts"], body["clicked_texts"])
            for path in (
                "/api/history/sync",
                "/api/history/recent",
                "/api/history/search",
                "/api/history/context",
                "/api/analyze/history",
            ):
                code, _ = self.request(path, body)
                self.assertEqual(code, 403, path)
            code, _ = self.request("/api/analyze", {**body, "messages": [{"text": "合成内容"}]})
            self.assertEqual(code, 403)
            sync.assert_not_called()
            resolve.assert_not_called()
        self.assertEqual(self.store.stats()["contacts"], 2)

    def test_enable_imports_only_requested_friend_and_resolves_pending_click(self):
        _, contact = self.observe("新朋友")
        body = {"target": "synthetic-new", "contact_revision": contact["contact_revision"]}
        self.request("/api/history/select-position", {**body, "clicked_texts": ["新朋友点击记录"], "source": "ax"})

        def sync(store, target):
            self.assertEqual(target, "synthetic-new")
            store.ingest(target, "新朋友", [message(1, "新朋友此前记录"), message(2, "新朋友点击记录")])
            return {"seconds": 0.01}

        with mock.patch.object(server, "sync_history", side_effect=sync) as syncing:
            code, result = self.request("/api/contact/enable", body)
        self.assertEqual(code, 200)
        self.assertTrue(result["analysis_enabled"])
        self.assertFalse(result["requires_consent"])
        self.assertEqual(result["selection"]["status"], "matched")
        self.assertEqual(result["selection"]["target"], "synthetic-new")
        self.assertEqual(result["selection"]["message"]["sort_seq"], 2)
        self.assertEqual(syncing.call_count, 1)
        self.assertEqual(self.store.stats()["contacts"], 3)
        self.assertEqual((self.private / "analysis-approvals.json").stat().st_mode & 0o777, 0o600)

    def test_enable_without_pending_click_selects_latest_usable_message(self):
        _, contact = self.observe("新朋友")

        def sync(store, target):
            voice = message(3, "[语音]")
            voice["kind"] = 34
            store.ingest(target, "新朋友", [message(1, "此前"), message(2, "最新可用文字"), voice])
            return {"seconds": 0}

        with mock.patch.object(server, "sync_history", side_effect=sync):
            code, result = self.request(
                "/api/contact/enable", {"target": "synthetic-new", "contact_revision": contact["contact_revision"]}
            )
        self.assertEqual(code, 200)
        self.assertEqual(result["selection"]["message"]["sort_seq"], 2)

    def test_stale_confirmation_does_not_enable_or_import_old_friend(self):
        _, contact = self.observe("新朋友")
        self.observe("好友乙")
        with mock.patch.object(server, "sync_history") as sync:
            code, _ = self.request(
                "/api/contact/enable", {"target": "synthetic-new", "contact_revision": contact["contact_revision"]}
            )
        self.assertEqual(code, 409)
        self.assertFalse(self.catalog.enabled("synthetic-new"))
        sync.assert_not_called()

    def test_contact_change_during_import_does_not_publish_stale_selection(self):
        _, contact = self.observe("新朋友")

        def sync(store, target):
            store.ingest(target, "新朋友", [message(1, "合成消息")])
            with server.LOCK:
                server.set_contact_locked("synthetic-b")
            return {"seconds": 0}

        with mock.patch.object(server, "sync_history", side_effect=sync):
            code, _ = self.request(
                "/api/contact/enable", {"target": "synthetic-new", "contact_revision": contact["contact_revision"]}
            )
        self.assertEqual(code, 409)
        self.assertEqual(self.state["active_target"], "synthetic-b")
        self.assertIsNone(self.state["selection"])

    def test_alias_compatibility_returns_canonical_target_and_owner_check(self):
        _, contact = self.request("/api/contact/select", {"target": "好友乙"})
        self.assertEqual(contact["active_target"], "synthetic-b")
        code, _ = self.request(
            "/api/history/select",
            {
                "target": "好友乙",
                "message_id": self.ids["synthetic-a"],
                "contact_revision": contact["contact_revision"],
            },
        )
        self.assertEqual(code, 400)
        code, selected = self.request(
            "/api/history/select",
            {
                "target": "好友乙",
                "message_id": self.ids["synthetic-b"],
                "contact_revision": contact["contact_revision"],
            },
        )
        self.assertEqual(code, 200)
        self.assertEqual(selected["target"], "synthetic-b")

    def test_unresolved_click_within_same_contact_preserves_verified_card(self):
        self.request("/api/history/select", {"target": "好友甲", "message_id": self.ids["synthetic-a"]})
        previous = copy.deepcopy(self.state["selection"])
        with (
            mock.patch.object(server, "sync_history"),
            mock.patch.object(
                server, "resolve_click", return_value={"status": "ambiguous", "message": None, "candidates": []}
            ),
        ):
            code, result = self.request(
                "/api/history/select-position", {"target": "好友甲", "clicked_texts": ["重复内容"]}
            )
        self.assertEqual(code, 200)
        self.assertTrue(result["preserved_selection"])
        self.assertEqual(self.state["selection"], previous)

    def test_clicking_own_message_clears_old_analysis_and_preserves_anchor(self):
        own = message(3, "合成我方发言")
        own["role"] = "self"
        self.store.ingest("synthetic-a", "好友甲", [own])
        identifier = self.store.recent("synthetic-a")["messages"][-1]["id"]
        self.state["job"] = {"id": "old-job", "status": "done"}
        code, result = self.request("/api/history/select", {"target": "synthetic-a", "message_id": identifier})
        self.assertEqual(code, 200)
        self.assertEqual(result["status"], "self_message")
        self.assertEqual(result["message"]["id"], identifier)
        self.assertTrue(result["revision"])
        self.assertIsNone(self.state["job"])
        with mock.patch.object(server, "POOL") as pool:
            code, result = self.request(
                "/api/analyze/history", {"target": "synthetic-a", "through_id": identifier, "workflow": "llm_then_jev"}
            )
        self.assertEqual(code, 400)
        self.assertEqual(result["error"], "请选择对方消息")
        pool.submit.assert_not_called()

    def test_unknown_speaker_click_is_not_treated_as_opponent(self):
        resolved = {"status": "matched", "message": {"id": "synthetic-unknown", "role": "unknown"}, "candidates": []}
        self.state["job"] = {"id": "old-job", "status": "running"}
        with (
            mock.patch.object(server, "sync_history"),
            mock.patch.object(server, "resolve_click", return_value=resolved),
        ):
            code, result = self.request(
                "/api/history/select-position", {"target": "synthetic-a", "clicked_texts": ["合成待核对发言"]}
            )
        self.assertEqual(code, 200)
        self.assertEqual(result["status"], "unknown_role")
        self.assertIsNone(self.state["job"])

    def test_old_running_job_cannot_update_card_after_contact_switch(self):
        request = {"source": "manual", "target": "synthetic-a", "contact_revision": self.state["contact_revision"]}
        self.state["job"] = {"id": "old-job", "status": "running", "fingerprint": "synthetic-key"}

        class Engine:
            def analyze(_, request, progress, lookup):
                with server.LOCK:
                    server.set_contact_locked("synthetic-b")
                progress("旧任务进度")
                return {"source": "manual", "analysis": {"summary": "合成内容"}}

        with mock.patch.object(server, "ENGINE", Engine()), mock.patch.object(server, "record_check") as check:
            server.run_job("old-job", request)
        self.assertIsNone(self.state["job"])
        self.assertNotEqual(self.state.get("phase"), "旧任务进度")
        check.assert_not_called()

    def test_cached_result_after_switch_back_gets_current_revision_without_mutating_cache(self):
        self.state["ready"] = True
        count = []

        class Engine:
            def analyze(_, request, progress, lookup):
                count.append(1)
                return {
                    "source": "history",
                    "workflow": "synthetic",
                    "messages_used": 2,
                    "elapsed_seconds": 0,
                    "speaker_needs_review": False,
                    "analysis": {"summary": "合成摘要", "evidence": "合成依据", "replies": []},
                }

        class Pool:
            def submit(_, fn, *args):
                fn(*args)

        with (
            mock.patch.object(server, "ENGINE", Engine()),
            mock.patch.object(server, "POOL", Pool()),
            mock.patch.object(server, "RESULT_CACHE", server.OrderedDict()),
            mock.patch.object(server, "record_check"),
        ):
            _, contact = self.request("/api/contact/select", {"target": "synthetic-a"})
            body = {
                "target": "synthetic-a",
                "contact_revision": contact["contact_revision"],
                "through_id": self.ids["synthetic-a"],
                "workflow": "llm_then_jev",
            }
            code, _ = self.request("/api/analyze/history", body)
            self.assertEqual(code, 202)
            old_revision = self.state["job"]["result"]["contact_revision"]
            self.request("/api/contact/select", {"target": "synthetic-b"})
            _, contact = self.request("/api/contact/select", {"target": "synthetic-a"})
            code, result = self.request(
                "/api/analyze/history", {**body, "contact_revision": contact["contact_revision"]}
            )
            self.assertEqual(code, 202)
            self.assertTrue(result["cached"])
            self.assertEqual(len(count), 1)
            self.assertEqual(self.state["job"]["result"]["contact_revision"], contact["contact_revision"])
            self.assertNotEqual(old_revision, contact["contact_revision"])
            self.assertEqual(next(iter(server.RESULT_CACHE.values()))["result"]["contact_revision"], old_revision)

    def test_context_and_analysis_share_extended_source_window_and_signature(self):
        self.store.ingest("synthetic-a", "好友甲", [message(i, "合成来源" + str(i)) for i in range(1, 321)])
        identifier = self.store.recent("synthetic-a")["messages"][-1]["id"]
        body = {"target": "synthetic-a", "through_id": identifier, "workflow": "llm_then_jev"}
        code, context = self.request("/api/history/context", body)
        self.assertEqual(code, 200)
        self.assertEqual(len(context["messages"]), 300)
        self.assertEqual(context["messages"][0]["text"], "合成来源21")
        self.state["ready"] = True
        with mock.patch.object(server, "POOL") as pool:
            code, _ = self.request("/api/analyze/history", body)
        self.assertEqual(code, 202)
        submitted = pool.submit.call_args.args[2]
        self.assertEqual(submitted["source_history_limit"], 300)
        self.assertEqual(submitted["messages"], context["messages"])
        self.assertEqual(submitted["history_ids"], context["history_ids"])
        self.assertEqual(submitted["context_messages"], context["context_messages"])
        self.assertEqual(submitted["context_signature"], context["context_signature"])
        self.assertEqual(submitted["selected_message"]["id"], identifier)
        code, expanded = self.request("/api/history/context", {**body, "source_history_limit": 1000})
        self.assertEqual(code, 200)
        self.assertEqual(len(expanded["messages"]), 320)
        self.assertNotEqual(expanded["context_signature"], context["context_signature"])
        code, page = self.request("/api/history/recent", {"target": "synthetic-a"})
        self.assertEqual(code, 200)
        self.assertEqual(len(page["messages"]), 100)

    def test_extended_source_limit_rejected_before_submitting_inference(self):
        self.state["ready"] = True
        with mock.patch.object(server, "POOL") as pool:
            for endpoint in ("/api/history/context", "/api/analyze/history"):
                for limit in (True, 0, 1001, "300"):
                    code, _ = self.request(endpoint, {"target": "synthetic-a", "source_history_limit": limit})
                    self.assertEqual(code, 400)
            pool.submit.assert_not_called()

    def test_queued_obsolete_job_skips_engine_entirely(self):
        request = {"target": "synthetic-a", "contact_revision": self.state["contact_revision"]}
        self.state["job"] = {"id": "new-job", "status": "running"}
        with mock.patch.object(server, "ENGINE") as engine:
            server.run_job("obsolete-job", request)
        engine.analyze.assert_not_called()

    def test_switch_during_preconsent_gate_check_cannot_restore_old_target(self):
        _, contact = self.observe("新朋友")
        original = self.catalog.enabled

        def enabled(target):
            if target == "synthetic-new":
                with server.LOCK:
                    server.set_contact_locked("synthetic-b")
                return False
            return original(target)

        with (
            mock.patch.object(self.catalog, "enabled", side_effect=enabled),
            mock.patch.object(server, "sync_history") as sync,
        ):
            code, _ = self.request(
                "/api/history/select-position",
                {
                    "target": "synthetic-new",
                    "contact_revision": contact["contact_revision"],
                    "clicked_texts": ["合成位置"],
                },
            )
        self.assertEqual(code, 409)
        self.assertEqual(self.state["active_target"], "synthetic-b")
        self.assertIsNone(self.state["pending_locator"])
        sync.assert_not_called()

    def test_click_during_initial_import_uses_newest_pending_position(self):
        _, contact = self.observe("新朋友")
        body = {"target": "synthetic-new", "contact_revision": contact["contact_revision"]}
        self.request("/api/history/select-position", {**body, "clicked_texts": ["初始点击位置"], "source": "ax"})

        def sync(store, target):
            store.ingest(target, "新朋友", [message(1, "初始点击位置"), message(2, "更新的点击位置")])
            code, result = self.request(
                "/api/history/select-position", {**body, "clicked_texts": ["更新的点击位置"], "source": "ax"}
            )
            self.assertEqual(code, 200)
            self.assertEqual(result["status"], "pending")
            return {"seconds": 0}

        with mock.patch.object(server, "sync_history", side_effect=sync) as sync_mock:
            code, result = self.request("/api/contact/enable", body)
        self.assertEqual(code, 200)
        self.assertEqual(sync_mock.call_count, 1)
        self.assertEqual(result["selection"]["message"]["sort_seq"], 2)
        self.assertIsNone(self.state["enabling_revision"])
        self.assertIsNone(self.state["pending_locator"])

    def test_failed_initial_import_clears_enabling_gate(self):
        _, contact = self.observe("新朋友")
        with mock.patch.object(server, "sync_history", side_effect=ValueError("合成读取失败")):
            code, _ = self.request(
                "/api/contact/enable", {"target": "synthetic-new", "contact_revision": contact["contact_revision"]}
            )
        self.assertEqual(code, 400)
        self.assertIsNone(self.state["enabling_revision"])


if __name__ == "__main__":
    unittest.main()
