"""CPU checks for the registered four-arm learned-query factorial."""
import json
import unittest

import torch
from torch.nn import functional as F

from scripts.hanguard.four_arm_query_heads import ARMS, FourArmHead, parameter_count
from scripts.hanguard.multilabel_heads import MultiLabelHead


torch.set_num_threads(1)


class FourArmQueryHeadTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(391)
        self.last = torch.randn(3, 4, 12, generator=generator)
        self.tokens = torch.randn(3, 7, 12, generator=generator)
        self.mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0],
                                  [0, 1, 0, 1, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)

    def head(self, arm, num_classes=5, **kwargs):
        return FourArmHead(12, width=8, num_classes=num_classes,
                           arm=arm, dropout=0., **kwargs)

    def test_all_arms_return_two_or_five_raw_logits(self):
        for classes in (2, 5):
            for arm in ARMS:
                with self.subTest(classes=classes, arm=arm):
                    output = self.head(arm, classes)(self.last, self.tokens, self.mask)
                    self.assertEqual(output.shape, (3, classes))
                    self.assertEqual(output.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(output).all())
                    # The actual objective consumes logits without a sigmoid.
                    loss = F.cross_entropy(output, torch.arange(3) % classes)
                    self.assertTrue(torch.isfinite(loss))

    def test_paired_initialization_is_exact_for_both_factors(self):
        for a, b in (('baseline', 'fusion_only'), ('category_only', 'fusion_category')):
            with self.subTest(a=a, b=b):
                small, large = self.head(a), self.head(b)
                small_state, large_state = small.state_dict(), large.state_dict()
                for key, value in small_state.items():
                    torch.testing.assert_close(value, large_state[key], rtol=0, atol=0)
                torch.testing.assert_close(small(self.last, self.tokens, self.mask),
                                           large(self.last, self.tokens, self.mask), rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(large.branch.fusion_output.weight), 0)
                self.assertEqual(torch.count_nonzero(large.branch.fusion_output.bias), 0)

    def test_query_path_exactly_reuses_old_head_logits_and_parameters(self):
        for classes in (2, 5):
            for dropout in (0., .1):
                with self.subTest(classes=classes, dropout=dropout):
                    head = FourArmHead(12, width=8, num_classes=classes,
                                       arm='category_only', dropout=dropout, seed=73)
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(73)
                        old = MultiLabelHead(12, width=8, num_labels=classes,
                                             mode='learned_queries', dropout=dropout, query_seed=73)
                    for key, value in old.state_dict().items():
                        torch.testing.assert_close(value, head.query_head.state_dict()[key],
                                                   rtol=0, atol=0)
                    torch.manual_seed(981)
                    expected = old(self.tokens, self.mask)['logits']
                    torch.manual_seed(981)
                    actual = head(self.last, self.tokens, self.mask)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_initialization_preserves_caller_rng_and_is_order_independent(self):
        initial = torch.get_rng_state().clone()
        heads = {arm: self.head(arm, seed=19) for arm in reversed(ARMS)}
        self.assertTrue(torch.equal(initial, torch.get_rng_state()))
        for arm in ARMS:
            rebuilt = self.head(arm, seed=19)
            for key, value in heads[arm].state_dict().items():
                torch.testing.assert_close(value, rebuilt.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(self.head('category_only', seed=19).query_head.query_vectors,
                                     self.head('category_only', seed=20).query_head.query_vectors))

    def test_fixed_weights_and_actual_parameter_counts(self):
        heads = {arm: self.head(arm) for arm in ARMS}
        expected_fusion = 4 * (12 * 8 + 3 * 8) + 8 * 8 + 8
        self.assertEqual(heads['baseline'].parameter_count(), 12 * 8 + 8 + 8 * 5 + 5)
        self.assertEqual(heads['category_only'].parameter_count(), 2 * 12 * 8 + 3 * 8 * 8 + 8 * 8 + 5 * 8 + 1)
        for small, large in (('baseline', 'fusion_only'), ('category_only', 'fusion_category')):
            branch = heads[large].branch
            self.assertEqual(branch.layers, (8, 16, 24, 32))
            self.assertEqual(parameter_count(branch), expected_fusion)
            self.assertEqual(heads[large].parameter_count() - heads[small].parameter_count(), expected_fusion)
            self.assertFalse(branch.weights.requires_grad)
            torch.testing.assert_close(branch.weights, torch.full((4,), .25), rtol=0, atol=0)
            self.assertNotIn('weights', dict(branch.named_parameters()))
            self.assertNotIn('layer_logits', dict(branch.named_parameters()))
            self.assertEqual(json.loads(json.dumps(heads[large].metadata()))['fusion_weights'], [.25] * 4)

    def test_nonfinite_padding_is_ignored_and_valid_token_gradients_flow(self):
        changed = self.tokens.clone()
        changed[~self.mask] = float('nan')
        # Include both NaN and Inf sentinels in the mask's padding positions.
        changed[0, 5] = float('inf')
        for arm in ('category_only', 'fusion_category'):
            with self.subTest(arm=arm):
                head = self.head(arm)
                expected = head(self.last, self.tokens, self.mask)
                actual = head(self.last, changed, self.mask)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                raw = changed.clone().requires_grad_()
                head(self.last, raw, self.mask).square().sum().backward()
                self.assertTrue(torch.isfinite(raw.grad).all())
                self.assertEqual(float(raw.grad[~self.mask].abs().sum()), 0.)
                self.assertGreater(float(raw.grad[self.mask].abs().sum()), 0.)

    def test_padding_extension_and_other_samples_do_not_change_output(self):
        padded = torch.cat((self.tokens, torch.full((3, 13, 12), float('nan'))), dim=1)
        pad_mask = torch.cat((self.mask, torch.zeros(3, 13, dtype=torch.bool)), dim=1)
        for arm in ARMS:
            with self.subTest(arm=arm):
                head = self.head(arm).eval()
                expected = head(self.last, self.tokens, self.mask)
                torch.testing.assert_close(head(self.last, padded, pad_mask), expected)
                separate = torch.cat([head(self.last[i:i + 1], self.tokens[i:i + 1],
                                            self.mask[i:i + 1]) for i in range(3)])
                torch.testing.assert_close(separate, expected)
                order = torch.tensor([2, 0, 1])
                torch.testing.assert_close(head(self.last[order], self.tokens[order], self.mask[order]), expected[order])

    def test_single_token_keeps_category_identity(self):
        for classes in (2, 5):
            for arm in ('category_only', 'fusion_category'):
                output = self.head(arm, classes)(self.last, self.tokens[:, :1],
                                                 torch.ones(3, 1, dtype=torch.bool))
                self.assertGreater(float((output[:, 0] - output[:, 1]).abs().sum()), 1e-5)

    def test_no_implicit_token_truncation(self):
        head = self.head('category_only')
        raw = self.tokens[:, :1].repeat(1, 300, 1)
        mask = torch.ones(3, 300, dtype=torch.bool)
        expected = head(self.last, raw, mask)
        raw[:, -1] = self.tokens[:, -1] * 11
        self.assertGreater(float((head(self.last, raw, mask) - expected).abs().sum()), 1e-7)

    def test_gradients_reach_all_used_parameters_and_zero_residual_unblocks(self):
        for classes in (2, 5):
            for arm in ARMS:
                with self.subTest(classes=classes, arm=arm):
                    head = self.head(arm, classes)
                    optimizer = torch.optim.SGD(head.parameters(), lr=.2)
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        logits = head(self.last, self.tokens, self.mask)
                        F.cross_entropy(logits, torch.arange(3) % classes).backward()
                        for name, param in head.named_parameters():
                            self.assertIsNotNone(param.grad, name)
                            self.assertTrue(torch.isfinite(param.grad).all(), name)
                        optimizer.step()
                    if head.has_fusion:
                        for module in head.branch.layer_values:
                            self.assertGreater(float(module[0].weight.grad.abs().sum()), 0.)
                        self.assertGreater(float(head.branch.fusion_output.weight.abs().sum()), 0.)
                    if head.has_category:
                        self.assertGreater(float(head.query_head.query_residual.grad.abs().sum()), 0.)

    def test_bfloat16_autocast_still_returns_float32_logits(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    actual = self.head(arm)(self.last.bfloat16(), self.tokens.bfloat16(), self.mask)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.isfinite(actual).all())

    def test_rejects_invalid_shapes_and_empty_valid_token_sets(self):
        for kwargs in (dict(arm='unknown'), dict(num_classes=3), dict(hidden_size=0),
                       dict(width=0), dict(dropout=1.)):
            arguments = dict(hidden_size=12)
            arguments.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FourArmHead(**arguments)
        for arm in ARMS:
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                self.head(arm)(self.last[:, :3], self.tokens, self.mask)
        for arm in ('category_only', 'fusion_category'):
            head = self.head(arm)
            for tokens, mask in ((None, None), (self.tokens, self.mask[:, :-1]),
                                 (self.tokens, torch.zeros_like(self.mask))):
                with self.subTest(arm=arm), self.assertRaises(ValueError):
                    head(self.last, tokens, mask)


if __name__ == '__main__':
    unittest.main()
