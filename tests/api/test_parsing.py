import unittest

from oepp.api.parsing import parse_actions, parse_numbered_action_names


class ParseActionsTests(unittest.TestCase):
    def test_accepts_exact_json_actions(self) -> None:
        result = parse_actions('{"actions": [2, 0, 1]}', 3, {0, 1, 2})
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.actions, [2, 0, 1])

    def test_extracts_json_from_surrounding_text(self) -> None:
        result = parse_actions('Answer: {"actions": [0, 1, 2]} thanks', 3, {0, 1, 2})
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.actions, [0, 1, 2])

    def test_rejects_wrong_length_without_truncation(self) -> None:
        result = parse_actions('{"actions": [0, 1]}', 3, {0, 1, 2})
        self.assertEqual(result.status, "parse_failure")
        self.assertIsNone(result.actions)
        self.assertIn("expected 3", result.error)

    def test_rejects_unknown_candidate_without_substitution(self) -> None:
        result = parse_actions('{"actions": [0, 1, 9]}', 3, {0, 1, 2})
        self.assertEqual(result.status, "parse_failure")
        self.assertIsNone(result.actions)
        self.assertIn("out of range", result.error)

    def test_accepts_numbered_legacy_action_names(self) -> None:
        candidates = [
            {"id": 0, "text": "cut in half"},
            {"id": 1, "text": "clean the floor"},
            {"id": 2, "text": "wash the floor"},
        ]
        result = parse_numbered_action_names(
            "1. Cut  in half\n2. clean the floor\n3. wash the floor", 3, candidates
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.actions, [0, 1, 2])
        self.assertEqual(result.action_texts, ["Cut  in half", "clean the floor", "wash the floor"])

    def test_rejects_unknown_legacy_action_without_substitution(self) -> None:
        candidates = [
            {"id": 0, "text": "cut in half"},
            {"id": 1, "text": "clean the floor"},
            {"id": 2, "text": "wash the floor"},
        ]
        result = parse_numbered_action_names(
            "1. cut in half\n2. clean the room\n3. wash the floor", 3, candidates
        )
        self.assertEqual(result.status, "parse_failure")
        self.assertIsNone(result.actions)
        self.assertIn("not in the candidate pool", result.error)

    def test_preserves_formatted_out_of_pool_actions_for_compatible_scoring(self) -> None:
        candidates = [
            {"id": 0, "text": "cut in half"},
            {"id": 1, "text": "clean the floor"},
            {"id": 2, "text": "wash the floor"},
        ]
        result = parse_numbered_action_names(
            "1. cut in half\n2. clean the room\n3. wash the floor", 3, candidates
        )
        self.assertEqual(result.status, "parse_failure")
        self.assertIsNone(result.actions)
        self.assertEqual(result.action_texts, ["cut in half", "clean the room", "wash the floor"])

    def test_accepts_whitespace_only_candidate_aliases(self) -> None:
        candidates = [
            {"id": 0, "text": "make the detergent"},
            {"id": 1, "text": "make the  detergent"},
        ]
        result = parse_numbered_action_names("1. make the detergent", 1, candidates)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.actions, [0])


if __name__ == "__main__":
    unittest.main()
