"""Synthetic HTTP and worker race checks; no models or WeChat files are opened."""

import copy
import http.client
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from concurrent.futures import ThreadPoolExecutor

from chatlens import server


class ServerResilienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = copy.deepcopy(server.STATE)
        self.state.update(
            ready=True,
            error=None,
            job=None,
            worker_job_id=None,
            worker_busy=False,
            active_target="synthetic-a",
            display_name="好友甲",
            contact_revision="revision-a",
            contact_status="recognized",
            selection=None,
            pending_locator=None,
            enabling_revision=None,
        )
        self.store = mock.Mock()
        self.catalog = SimpleNamespace(
            records={"synthetic-a": {"supported": True}, "synthetic-b": {"supported": True}},
            enabled=lambda _: True,
            revision="catalog",
            source="synthetic",
        )
        for patch in (
            mock.patch.object(server, "STATE", self.state),
            mock.patch.object(server, "HISTORY", self.store),
            mock.patch.object(server, "CATALOG", self.catalog),
            mock.patch.object(server, "DATA_DIR", self.root),
            mock.patch.object(server, "BUILD_DIR", self.root / "build"),
            mock.patch.object(server, "CONTACT_STATE", self.root / "contact-state.json"),
            mock.patch.object(server, "RESULT_CACHE", server.OrderedDict()),
            mock.patch.object(server, "resolve_target", side_effect=lambda value: value),
            mock.patch.object(server, "private_debug_enabled", return_value=False),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def request(self, path, body=None, raw=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=3)
        data = raw if raw is not None else json.dumps(body if body is not None else {}).encode()
        try:
            conn.request("POST", path, data, {"X-ChatLens-Token": server.TOKEN, **(headers or {})})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def result(self):
        return {
            "source": "history",
            "workflow_version": server.CARDS_VERSION,
            "messages_used": 2,
            "elapsed_seconds": 0.01,
            "speaker_needs_review": False,
            "analysis": {"summary": "合成摘要", "evidence": "合成依据", "replies": []},
        }

    def job(self):
        self.state["job"] = {"id": "job-a", "status": "running", "fingerprint": "fingerprint-a"}
        return {
            "source": "history",
            "target": "synthetic-a",
            "contact_revision": "revision-a",
            "history_ids": ["synthetic-message"],
            "history_watermark": [1, 1, "synthetic-message"],
        }

    def analyze_body(self):
        return {
            "target": "synthetic-a",
            "workflow": "llm_then_jev",
            "messages": [{"role": "other", "text": "合成测试消息"}],
        }

    def test_malformed_content_length_is_a_json_error_response(self):
        for length in ("wrong", "-1", "+2", "2.0", "１２"):
            if not length.isascii():
                # HTTP header fields are ISO-8859-1; the server independently
                # rejects non-ASCII digit strings if a transport supplies them.
                continue
            code, value = self.request("/api/clear", raw=b"{}", headers={"Content-Length": length})
            self.assertEqual(code, 400, length)
            self.assertIn("error", value)

    def test_non_ascii_wrong_secret_is_rejected_without_crashing(self):
        code, value = self.request("/api/clear", headers={"X-ChatLens-Token": "\u00e9"})
        self.assertEqual(code, 403)
        self.assertIn("error", value)

    def test_duplicate_length_and_transfer_encoding_are_rejected(self):
        for duplicate in (True, False):
            conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=3)
            try:
                conn.putrequest("POST", "/api/clear")
                conn.putheader("X-ChatLens-Token", server.TOKEN)
                conn.putheader("Content-Length", "2")
                conn.putheader("Content-Length", "2") if duplicate else conn.putheader("Transfer-Encoding", "chunked")
                conn.endheaders(b"{}")
                response = conn.getresponse()
                self.assertEqual(response.status, 400)
                self.assertIn("error", json.loads(response.read()))
            finally:
                conn.close()

    def test_bounded_and_complete_request_body_required(self):
        for length in ("0", "100001"):
            code, _ = self.request("/api/clear", raw=b"{}", headers={"Content-Length": length})
            self.assertEqual(code, 413)
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=3)
        try:
            conn.request("POST", "/api/clear", b"{", {"Content-Length": "10", "X-ChatLens-Token": server.TOKEN})
            conn.sock.shutdown(socket.SHUT_WR)
            response = conn.getresponse()
            self.assertEqual(response.status, 400)
            self.assertIn("error", json.loads(response.read()))
        finally:
            conn.close()

    def test_invalid_json_unicode_non_object_and_nonfinite_numbers_are_rejected(self):
        for raw in (b"{", b"\xff", b"[]", b'{"value":NaN}', b'{"value":Infinity}', b"[" * 1100 + b"]" * 1100):
            code, value = self.request("/api/clear", raw=raw)
            self.assertEqual(code, 400)
            self.assertIn("error", value)

    def test_unexpected_post_error_is_stable_and_does_not_echo_private_details(self):
        with mock.patch.object(server, "sync_history", side_effect=RuntimeError("private-input-sentinel")):
            code, value = self.request("/api/history/sync", {"target": "synthetic-a"})
        self.assertEqual(code, 500)
        self.assertNotIn("private-input-sentinel", json.dumps(value))
        self.assertEqual(self.request("/api/unknown")[0], 404)

    def test_unexpected_get_error_is_stable_and_does_not_echo_private_details(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=3)
        try:
            with mock.patch.object(server, "runtime_status", side_effect=OSError("private-path-sentinel")):
                conn.request("GET", "/api/history/status", headers={"X-ChatLens-Token": server.TOKEN})
                response = conn.getresponse()
                data = response.read()
            self.assertEqual(response.status, 500)
            self.assertNotIn(b"private-path-sentinel", data)
            self.assertIn("error", json.loads(data))
        finally:
            conn.close()

    def test_contact_switch_keeps_worker_busy_and_blocks_new_queue_entries(self):
        with mock.patch.object(server, "POOL") as pool:
            self.assertEqual(self.request("/api/analyze", self.analyze_body())[0], 202)
            self.assertTrue(self.state["worker_busy"])
            with server.LOCK:
                server.set_contact_locked("synthetic-b")
            self.assertIsNone(self.state["job"])
            for _ in range(3):
                self.assertEqual(self.request("/api/analyze", {**self.analyze_body(), "target": "synthetic-b"})[0], 409)
            self.assertEqual(self.request("/api/clear")[0], 409)
            self.assertEqual(pool.submit.call_count, 1)

    def test_contact_revision_is_captured_even_when_client_omits_it(self):
        with mock.patch.object(server, "POOL") as pool:
            self.assertEqual(self.request("/api/analyze", self.analyze_body())[0], 202)
        request = pool.submit.call_args.args[2]
        self.assertEqual(request["contact_revision"], "revision-a")

    def test_running_synthetic_worker_blocks_requests_after_visible_job_is_cleared(self):
        entered = threading.Event()
        release = threading.Event()

        def analyze(*args):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("synthetic test timeout")
            return self.result()

        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            mock.patch.object(server, "POOL", pool),
            mock.patch.object(server, "ENGINE", SimpleNamespace(analyze=analyze)),
            mock.patch.object(server, "record_decision") as audit,
        ):
            try:
                self.assertEqual(self.request("/api/analyze", self.analyze_body())[0], 202)
                self.assertTrue(entered.wait(2))
                with server.LOCK:
                    server.set_contact_locked("synthetic-b")
                self.assertIsNone(self.state["job"])
                self.assertTrue(self.state["worker_busy"])
                self.assertEqual(self.request("/api/analyze", {**self.analyze_body(), "target": "synthetic-b"})[0], 409)
                self.assertEqual(self.request("/api/clear")[0], 409)
            finally:
                release.set()
        self.assertFalse(self.state["worker_busy"])
        self.assertIsNone(self.state["job"])
        self.assertFalse(server.RESULT_CACHE)
        audit.assert_not_called()

    def test_late_old_contact_result_does_not_write_summary_cache_or_audit(self):
        request = self.job()

        def analyze(_, progress, lookup):
            with server.LOCK:
                server.set_contact_locked("synthetic-b")
            progress({"text": "旧任务进度", "partial_scores": {"synthetic": True}})
            return self.result()

        with (
            mock.patch.object(server, "ENGINE", SimpleNamespace(analyze=analyze)),
            mock.patch.object(server, "record_decision") as audit,
            mock.patch.object(server, "record_check") as check,
        ):
            server.run_job("job-a", request)
        self.assertIsNone(self.state["job"])
        self.assertFalse(self.state["worker_busy"])
        self.assertNotEqual(self.state["phase"], "旧任务进度")
        self.assertFalse(server.RESULT_CACHE)
        self.store.save_summary.assert_not_called()
        audit.assert_not_called()
        check.assert_not_called()
        self.assertFalse((self.root / "latest-card.json").exists())

    def test_optional_diagnostic_failures_do_not_turn_success_into_error(self):
        request = self.job()
        with (
            mock.patch.object(server, "ENGINE", SimpleNamespace(analyze=lambda *args: self.result())),
            mock.patch.object(server, "record_decision", side_effect=OSError("unwritable")),
            mock.patch.object(server, "record_check", side_effect=RuntimeError("diagnostic failure")),
        ):
            server.run_job("job-a", request)
        self.assertEqual(self.state["job"]["status"], "done")
        self.assertFalse(self.state["worker_busy"])
        self.assertIn("fingerprint-a", server.RESULT_CACHE)
        self.store.save_summary.assert_called_once()

    def test_optional_summary_failure_does_not_discard_a_completed_card(self):
        request = self.job()
        self.store.save_summary.side_effect = OSError("unwritable")
        with (
            mock.patch.object(server, "ENGINE", SimpleNamespace(analyze=lambda *args: self.result())),
            mock.patch.object(server, "record_decision"),
            mock.patch.object(server, "record_check"),
        ):
            server.run_job("job-a", request)
        self.assertEqual(self.state["job"]["status"], "done")
        self.assertFalse(self.state["worker_busy"])

    def test_optional_private_debug_write_failure_preserves_a_completed_card(self):
        request = self.job()
        (self.root / "latest-card.json").mkdir()
        with (
            mock.patch.object(server, "ENGINE", SimpleNamespace(analyze=lambda *args: self.result())),
            mock.patch.object(server, "private_debug_enabled", return_value=True),
            mock.patch.object(server, "record_decision"),
            mock.patch.object(server, "record_check"),
        ):
            server.run_job("job-a", request)
        self.assertEqual(self.state["job"]["status"], "done")
        self.assertFalse(self.state["worker_busy"])

    def test_model_failure_releases_worker_without_echoing_private_error_text(self):
        request = self.job()
        with (
            mock.patch.object(
                server,
                "ENGINE",
                SimpleNamespace(analyze=mock.Mock(side_effect=RuntimeError("private-generation-sentinel"))),
            ),
            mock.patch.object(server, "record_decision", side_effect=OSError("unwritable")),
        ):
            server.run_job("job-a", request)
        self.assertEqual(self.state["job"]["status"], "error")
        self.assertFalse(self.state["worker_busy"])
        self.assertNotIn("private-generation-sentinel", self.state["job"]["error"])

    def test_obsolete_job_cannot_release_another_workers_reservation(self):
        self.state.update(worker_job_id="new-job", worker_busy=True, job={"id": "new-job", "status": "running"})
        with mock.patch.object(server, "ENGINE") as engine:
            server.run_job("old-job", {})
        engine.analyze.assert_not_called()
        self.assertEqual(self.state["worker_job_id"], "new-job")
        self.assertTrue(self.state["worker_busy"])

    def test_submit_failure_releases_reservation_for_future_attempts(self):
        with mock.patch.object(server, "POOL") as pool:
            pool.submit.side_effect = RuntimeError("executor stopped")
            self.assertEqual(self.request("/api/analyze", self.analyze_body())[0], 500)
        self.assertFalse(self.state["worker_busy"])
        self.assertIsNone(self.state["worker_job_id"])
        self.assertEqual(self.state["job"]["status"], "error")

    def test_clear_is_serialized_with_new_inference_reservations(self):
        cache = mock.Mock()
        cache.clear.side_effect = lambda: self.assertTrue(server.LOCK.locked())
        with mock.patch.object(server, "ENGINE", SimpleNamespace(decider=SimpleNamespace(_state_kv=cache))):
            self.assertEqual(self.request("/api/clear")[0], 200)
        cache.clear.assert_called_once()

    def test_record_check_tolerates_corrupt_metadata_and_unwritable_directory(self):
        build = self.root / "build"
        build.mkdir()
        (build / "live-check-report.json").write_text("[]")
        self.assertTrue(server.record_check(synthetic=True))
        self.assertTrue(json.loads((build / "live-check-report.json").read_text())["synthetic"])
        with mock.patch.object(Path, "mkdir", side_effect=PermissionError("unwritable")):
            self.assertFalse(server.record_check(synthetic=True))


if __name__ == "__main__":
    unittest.main()
