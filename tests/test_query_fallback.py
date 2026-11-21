import unittest
from unittest.mock import patch

import api


class QueryFallbackTest(unittest.TestCase):
    def test_empty_planner_queries_still_search_policy(self):
        query = "Chính sách đổi trả tại CellphoneS thế nào?"
        for subqueries in ([], None, ["", None]):
            with self.subTest(subqueries=subqueries), \
                    patch.object(api, "llm_process_query", return_value={"cleaned_query": query, "sub_queries": subqueries}), \
                    patch.object(api, "sorted_base_names", []), \
                    patch.object(api, "hybrid_search", return_value=[]) as search:
                _, cleaned, queries = api.retrieve_and_rerank(query)
                self.assertEqual(queries, [query])
                search.assert_called_once_with(query, "policy_collection", n_results=20, metadata_filter={"type": "policy"})


if __name__ == "__main__":
    unittest.main()
