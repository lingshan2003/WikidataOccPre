"""Guard against silent omissions, unknown-label dumping, and false resolution."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('semantic_merge', Path(__file__).resolve().parents[1] / 'Freebase/merge_bhht_semantic_review.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(**changes):
    result = dict(rank='1', raw_value='Music executive', semantic_level1='Culture', review_status='resolved', confidence='medium', semantic_rationale='负责音乐录制、发行和艺人业务。', evidence='occupation_meaning:音乐产业制作发行', candidate_l1s_json='["Culture"]', reviewer='batch')
    result.update(changes)
    return result


class MergeValidation(unittest.TestCase):
    def test_complete_valid_batch(self):
        module.validate_batch([row()], [row()], 'batch')

    def test_missing_or_duplicate_decisions_rejected(self):
        for outputs in ([], [row(), row()]):
            with self.assertRaises(AssertionError):
                module.validate_batch([row()], outputs, 'batch')

    def test_wrong_raw_profession_rejected(self):
        with self.assertRaises(AssertionError):
            module.validate_batch([row()], [row(raw_value='Official')], 'batch')

    def test_unknown_cannot_be_other_fallback(self):
        with self.assertRaises(AssertionError):
            module.validate_decision(row(review_status='needs_context', confidence='low', semantic_level1='Other'))

    def test_valid_unresolved_retains_multiple_candidates(self):
        module.validate_decision(row(review_status='needs_context', confidence='low', semantic_level1='', candidate_l1s_json='["Culture", "Leadership"]'))

    def test_resolved_cannot_hide_cross_l1_candidates(self):
        with self.assertRaises(AssertionError):
            module.validate_decision(row(candidate_l1s_json='["Culture", "Leadership"]'))

    def test_evidence_required(self):
        with self.assertRaises(AssertionError):
            module.validate_decision(row(evidence=''))


if __name__ == '__main__':
    unittest.main()
