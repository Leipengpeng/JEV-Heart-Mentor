"""Canonical identity and account-bound consent use synthetic metadata only."""

import hashlib
import tempfile
import unittest
import sys
from pathlib import Path
from chatlens.contact_catalog import ContactCatalog, contact_sort_name
from unittest import mock
from chatlens import contact_catalog


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.account = "account-a"
        self.records = [
            ("person-a", "合成好友", "", ""),
            ("person-b", "同名", "", ""),
            ("person-c", "同名", "", ""),
            ("group@chatroom", "合成群", "", ""),
            ("gh_bot", "合成公众号", "", ""),
        ]
        self.catalog = ContactCatalog(Path(self.temp.name), ("合成好友", "同名"), loader=self.snapshot)

    def snapshot(self):
        return {
            "contacts": self.records,
            "history_tables": {"Msg_" + hashlib.md5(r[0].encode()).hexdigest() for r in self.records},
            "account_key": self.account,
            "source": "synthetic",
            "source_signature": "synthetic-signature",
        }

    def test_legacy_approval_only_migrates_unique_identity_and_filters_groups(self):
        self.catalog.refresh()
        self.assertEqual(len(self.catalog.records), 3)
        self.assertTrue(self.catalog.enabled("person-a"))
        self.assertFalse(self.catalog.enabled("person-b"))
        self.assertFalse(self.catalog.enabled("person-c"))
        contact, method = self.catalog.resolve("同名")
        self.assertIsNone(contact)
        self.assertEqual(method, "ambiguous-contact-title")
        contact, _ = self.catalog.resolve("person-c")
        self.assertEqual(contact["id"], "person-c")

    def test_explicit_approval_persists_by_identity_even_when_display_changes(self):
        self.catalog.refresh()
        self.catalog.enable("person-b")
        self.records[1] = ("person-b", "新备注", "", "")
        restored = ContactCatalog(Path(self.temp.name), loader=self.snapshot)
        restored.refresh()
        self.assertTrue(restored.enabled("person-b"))
        self.assertFalse(restored.enabled("person-c"))
        self.assertEqual(restored.records["person-b"]["name"], "新备注")

    def test_approval_is_isolated_between_accounts(self):
        self.catalog.refresh()
        self.catalog.enable("person-b")
        self.account = "account-b"
        self.catalog.refresh(force=True)
        self.assertFalse(self.catalog.enabled("person-b"))
        self.account = "account-a"
        self.catalog.refresh(force=True)
        self.assertTrue(self.catalog.enabled("person-b"))

    def test_unavailable_known_contact_can_be_discovered_without_import(self):
        self.records.append(("person-no-history", "尚无历史", "", ""))
        snapshot = self.snapshot()
        snapshot["history_tables"] = set()
        self.catalog.loader = lambda: snapshot
        self.catalog.refresh()
        contact, _ = self.catalog.resolve("尚无历史")
        self.assertEqual(contact["id"], "person-no-history")
        self.assertFalse(contact["available"])
        self.assertFalse(self.catalog.enabled(contact["id"]))

    def test_same_name_history_availability_is_not_identity_proof(self):
        snapshot = self.snapshot()
        snapshot["history_tables"] = {"Msg_" + hashlib.md5("person-b".encode()).hexdigest()}
        self.catalog.loader = lambda: snapshot
        self.catalog.refresh()
        contact, method = self.catalog.resolve("同名")
        self.assertIsNone(contact)
        self.assertEqual(method, "ambiguous-contact-title")
        self.assertFalse(self.catalog.enabled("person-b"))

    def test_chinese_and_latin_names_share_one_alphabetic_order(self):
        if contact_catalog.lazy_pinyin is None:
            self.skipTest("pypinyin optional dependency unavailable")
        self.records = [
            ("person-1", "张三", "", ""),
            ("person-2", "Bob", "", ""),
            ("person-3", "陈晨", "", ""),
            ("person-4", "Anna", "", ""),
            ("person-5", "李明", "", ""),
        ]
        self.catalog.refresh()
        self.assertEqual([r["name"] for r in self.catalog.list()], ["Anna", "Bob", "陈晨", "李明", "张三"])
        self.assertEqual(self.catalog.records["person-3"]["sort_name"], "chenchen")
        self.assertEqual(self.catalog.records["person-1"]["id"], "person-1")

    def test_sort_key_normalizes_width_case_and_latin_accents(self):
        self.assertEqual(contact_sort_name("Ａnna"), "anna")
        self.assertEqual(contact_sort_name("Émile"), "emile")
        self.assertEqual(contact_sort_name(" BOB "), "bob")

    def test_sort_keys_are_cached_across_metadata_refreshes(self):
        contact_sort_name.cache_clear()
        self.catalog.refresh()
        before = contact_sort_name.cache_info()
        self.catalog.refresh(force=True)
        after = contact_sort_name.cache_info()
        self.assertEqual(before.misses, after.misses)
        self.assertGreaterEqual(after.hits - before.hits, len(self.catalog.records))

    def test_missing_optional_pinyin_keeps_catalog_usable(self):
        contact_sort_name.cache_clear()
        with mock.patch.object(contact_catalog, "lazy_pinyin", None):
            self.assertEqual(contact_sort_name("合成好友"), "")
            self.assertEqual(contact_sort_name("Alice"), "alice")
        contact_sort_name.cache_clear()


if __name__ == "__main__":
    unittest.main()
