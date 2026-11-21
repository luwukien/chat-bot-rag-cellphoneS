"""Storage corrections use indexed facts and preserve regular RAG questions."""

import asyncio
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from utils.product_capacity import ProductCapacityCatalog


class ProductCapacityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "data/prepared_products_chunks.json"
        cls.chunks = json.loads(path.read_text(encoding="utf-8"))
        cls.catalog = ProductCapacityCatalog(cls.chunks)

    def test_missing_capacity_uses_real_price_and_source(self):
        answer, sources = self.catalog.resolve_price_query("iPhone 16 Pro Max 128GB giá bao nhiêu?")
        self.assertIn("iPhone 16 Pro Max không có phiên bản 128GB trong dữ liệu hiện có", answer)
        self.assertIn("Bản 256GB có giá 30.990.000₫", answer)
        self.assertIn("giá đã thu thập và có thể thay đổi", answer)
        self.assertTrue(sources)
        self.assertEqual(sources[0]["metadata"]["product_id"], "iphone-16-pro-max")
        self.assertEqual(sources[0]["metadata"]["type"], "variants")

    def test_case_spacing_and_abbreviations(self):
        for query in ("IPHONE 16 PRO MAX 128 gb gia bao nhieu?", "ip 16 promax 128GB gia bn tien?"):
            with self.subTest(query=query):
                self.assertIn("30.990.000₫", self.catalog.resolve_price_query(query)[0])

    def test_valid_or_ambiguous_queries_keep_rag(self):
        queries = (
            "iPhone 16 Pro Max 256GB giá bao nhiêu?",
            "iPhone 16 Pro 128GB giá bao nhiêu?",
            "iPhone 16 Pro Max giá bao nhiêu?",
            "iPhone 99 Pro Max 128GB giá bao nhiêu?",
            "iPhone 16 Pro Max Ultra 128GB giá bao nhiêu?",
            "So sánh iPhone 16 Pro Max 128GB và iPhone 16 Pro",
            "iPhone 16 Pro Max 128GB giá bao nhiêu và có bảo hành không?",
            "iPhone 16 Pro Max 128GB giá và màu sắc?",
            "iPhone 16 Pro Max 128GB hay 256GB giá bao nhiêu?",
        )
        for query in queries:
            with self.subTest(query=query):
                self.assertIsNone(self.catalog.resolve_price_query(query))

    def test_tb_conversion_and_price_range(self):
        # 1024GB is the existing 1TB variant.
        self.assertIsNone(self.catalog.resolve_price_query("iPhone 16 Pro Max 1024GB giá bao nhiêu?"))
        chunks = [chunk for chunk in self.chunks if chunk["metadata"].get("product_id") == "iphone-16-pro-max-512gb"]
        answer, _ = ProductCapacityCatalog(chunks).resolve_price_query("iPhone 16 Pro Max 128GB giá bao nhiêu?")
        self.assertIn("37.990.000₫ đến 38.990.000₫ tùy màu", answer)

    def test_missing_prices_are_not_invented(self):
        chunks = [dict(chunk, text="Giá: Liên hệ") for chunk in self.chunks]
        answer, sources = ProductCapacityCatalog(chunks).resolve_price_query("iPhone 16 Pro Max 128GB giá bao nhiêu?")
        self.assertIn("256GB, 512GB, 1TB", answer)
        self.assertNotIn("₫", answer)
        self.assertEqual(sources, [])

    def test_endpoint_returns_correction_without_model_calls(self):
        import api

        with patch.object(api, "capacity_catalog", self.catalog), \
             patch.object(api, "retrieve_and_rerank", side_effect=AssertionError("unexpected retrieval")), \
             patch.object(api, "generate_answer", side_effect=AssertionError("unexpected generation")):
            response = asyncio.run(api.chat_endpoint(api.ChatRequest(message="iPhone 16 Pro Max 128GB giá bao nhiêu?")))
        self.assertIn("30.990.000₫", response.answer)
        self.assertEqual(response.sources[0].type, "variants")
        self.assertIn("256GB", response.sources[0].product_name)

    def test_endpoint_preserves_valid_query(self):
        import api

        query = "iPhone 16 Pro Max 256GB giá bao nhiêu?"
        with patch.object(api, "capacity_catalog", self.catalog), \
             patch.object(api, "retrieve_and_rerank", return_value=([], query, [query])) as retrieve, \
             patch.object(api, "generate_answer", return_value="Regular answer") as generate:
            response = asyncio.run(api.chat_endpoint(api.ChatRequest(message=query)))
        self.assertEqual(response.answer, "Regular answer")
        retrieve.assert_called_once()
        generate.assert_called_once_with(query, [])

    def test_endpoint_checks_planner_normalization(self):
        import api

        query = "iphoen 16 pro max 128gb gia bao nhieu?"
        cleaned = "iPhone 16 Pro Max 128GB giá bao nhiêu?"
        with patch.object(api, "capacity_catalog", self.catalog), \
             patch.object(api, "retrieve_and_rerank", return_value=([], cleaned, [cleaned])), \
             patch.object(api, "generate_answer", side_effect=AssertionError("unexpected generation")):
            response = asyncio.run(api.chat_endpoint(api.ChatRequest(message=query)))
        self.assertIn("30.990.000₫", response.answer)
        self.assertEqual(response.cleaned_query, cleaned)

    def test_catalog_is_built_from_indexed_documents(self):
        import api

        client = Mock()
        client.get_collection.return_value.get.return_value = {
            "ids": [chunk["chunk_id"] for chunk in self.chunks],
            "documents": [chunk["text"] for chunk in self.chunks],
            "metadatas": [chunk["metadata"] for chunk in self.chunks],
        }
        with patch.multiple(api, chroma_client=client, base_to_pids={},
                            pid_to_name={}, sorted_base_names=[],
                            capacity_catalog=ProductCapacityCatalog()):
            api.build_model_mappings()
            answer, sources = api.capacity_catalog.resolve_price_query("iPhone 16 Pro Max 128GB giá bao nhiêu?")
        self.assertIn("30.990.000₫", answer)
        self.assertEqual(sources[0]["id"], "chunk_0908")


if __name__ == "__main__":
    unittest.main()
