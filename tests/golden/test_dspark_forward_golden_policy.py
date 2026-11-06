# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU tests for the DSpark full-forward numerical acceptance policy."""

import ast
from pathlib import Path
import unittest

import torch

from golden.validation import ratio_reldiff, validate_golden


def _load_policy():
    """Load the actual host comparator source without importing device libraries."""
    model = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_flash_lowlat"
    sample_tree = ast.parse((model / "sample.py").read_text())
    eps = next(n for n in sample_tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "SAMPLING_EPS" for t in n.targets))
    names = {
        "_compare_forward_rows", "compare_forward_hidden", "compare_forward_pre_hc",
        "compare_forward_logits", "compare_forward_sampled_ids", "decode_forward_compare_fn",
    }
    head_tree = ast.parse((model / "lm_head.py").read_text())
    head = next(n for n in head_tree.body if isinstance(n, ast.FunctionDef) and n.name == "compare_sampled_ids")
    forward_tree = ast.parse((model / "decode_fwd.py").read_text())
    functions = [n for n in forward_tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    namespace = {"ratio_reldiff": ratio_reldiff, "SAMPLING_EPS": ast.literal_eval(eps.value)}
    module = ast.fix_missing_locations(ast.Module(body=[head, *functions], type_ignores=[]))
    exec(compile(module, str(model / "decode_fwd.py"), "exec"), namespace)
    return namespace


policy = _load_policy()
compare_forward_hidden = policy["compare_forward_hidden"]
compare_forward_logits = policy["compare_forward_logits"]
compare_forward_pre_hc = policy["compare_forward_pre_hc"]
compare_forward_sampled_ids = policy["compare_forward_sampled_ids"]
decode_forward_compare_fn = policy["decode_forward_compare_fn"]


class ForwardGoldenPolicyTests(unittest.TestCase):
    def setUp(self):
        self.reference = torch.zeros((2, 2, 20), dtype=torch.float32)
        self.reference[:, 0, 0] = 1.0
        self.reference[:, 0, 1] = 0.999
        self.actual = self.reference.clone()
        self.inputs = {
            "logit_row_indices": torch.tensor([[0, -1], [0, -1]], dtype=torch.int32),
            "sampling_temperatures": torch.zeros((2, 2), dtype=torch.float32),
        }
        self.expected_ids = torch.zeros((2, 2, 2), dtype=torch.int32)
        self.actual_ids = self.expected_ids.clone()

    def kwargs(self):
        return {
            "actual_outputs": {"logits": self.actual},
            "expected_outputs": {"logits": self.reference},
            "inputs": self.inputs,
            "rtol": 1e-3,
            "atol": 1e-3,
        }

    def test_close_logits_may_select_a_different_reference_token(self):
        self.actual[:, 0, 0] = 0.999
        self.actual[:, 0, 1] = 1.0
        self.actual_ids[:, 0, 0] = 1
        self.assertTrue(compare_forward_logits(self.actual, self.reference, **self.kwargs())[0])
        self.assertTrue(compare_forward_sampled_ids(self.actual_ids, self.expected_ids, **self.kwargs())[0])

    def test_sampling_must_follow_actual_logits(self):
        self.actual[:, 0, 0] = 0.999
        self.actual[:, 0, 1] = 1.0
        self.assertFalse(compare_forward_sampled_ids(self.actual_ids, self.expected_ids, **self.kwargs())[0])

    def test_selected_token_cannot_hide_a_large_logit_error_in_allowed_outliers(self):
        self.actual[0, 0, 1] = 9.0
        self.actual_ids[0, 0, 0] = 1
        self.assertTrue(compare_forward_logits(self.actual, self.reference, **self.kwargs())[0])
        self.assertFalse(compare_forward_sampled_ids(self.actual_ids, self.expected_ids, **self.kwargs())[0])

    def test_nonfinite_and_inactive_logit_errors_fail(self):
        self.actual[0, 0, 0] = float("nan")
        self.assertFalse(compare_forward_logits(self.actual, self.reference, **self.kwargs())[0])
        self.actual = self.reference.clone()
        self.actual[0, 1, 0] = 0.1
        self.assertFalse(compare_forward_logits(self.actual, self.reference, **self.kwargs())[0])

    def test_hidden_error_is_checked_per_owner_and_token(self):
        expected = torch.ones((2, 2, 100), dtype=torch.float32)
        actual = expected.clone()
        actual[0, 0, :] = 1.2
        self.assertFalse(compare_forward_hidden(actual, expected, **self.kwargs())[0])
        self.assertFalse(compare_forward_pre_hc(actual, expected, **self.kwargs())[0])
        actual = expected + 0.01
        self.assertTrue(compare_forward_hidden(actual, expected, **self.kwargs())[0])
        self.assertTrue(compare_forward_pre_hc(actual, expected, **self.kwargs())[0])

    def test_harness_uses_the_policy_without_relaxing_cache_outputs(self):
        self.actual[:, 0, 0] = 0.999
        self.actual[:, 0, 1] = 1.0
        self.actual_ids[:, 0, 0] = 1
        outputs = {
            "logits": self.actual,
            "sampled_ids": self.actual_ids,
            "hidden_out": torch.ones((2, 2, 10)),
            "pre_hc_hidden_out": torch.ones((2, 2, 10)),
            "kv_cache": torch.ones((2, 2)),
        }
        golden = dict(outputs, logits=self.reference, sampled_ids=self.expected_ids)
        validate_golden(outputs, golden, rtol=1e-3, atol=1e-3,
                        compare_fn=decode_forward_compare_fn(), inputs=self.inputs)
        outputs["kv_cache"] = torch.ones((2, 2)) * 1.01
        with self.assertRaises(AssertionError):
            validate_golden(outputs, golden, rtol=1e-3, atol=1e-3,
                            compare_fn=decode_forward_compare_fn(), inputs=self.inputs)


if __name__ == "__main__":
    unittest.main()
