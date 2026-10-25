"""CPU numerical checks for masked entmax and the single-layer interventions."""

import copy
import unittest

import torch
from torch.nn import functional as F

from scripts.hanguard.repaired_heads import ExperimentHead
from scripts.hanguard.sparse_token_readout_heads import (
    ARMS, SparseTokenReadoutHead, build_matched_heads, entmax15,
    initialization_audit, last_valid_indices, parameter_count,
)


def bisection_reference(scores, mask, dim=-1):
    """Independent scalar-root reference: sum(max(x/2-tau,0)^2)=1."""
    work = scores.double().masked_fill(~mask, -torch.inf)
    shifted = (work - work.amax(dim, keepdim=True)) / 2
    low = torch.full_like(work.amax(dim, keepdim=True), -1.)
    high = torch.zeros_like(low)
    for _ in range(100):
        middle = (low + high) / 2
        mass = (shifted - middle).clamp_min(0).square().sum(dim, keepdim=True)
        low = torch.where(mass > 1, middle, low)
        high = torch.where(mass > 1, high, middle)
    return (shifted - (low + high) / 2).clamp_min(0).square()


class EntmaxTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(281)

    def test_independent_bisection_reference_across_batch_and_axes(self):
        for shape, dim in (((6, 17), -1), ((3, 5, 9), 1), ((21,), 0)):
            for scale in (.001, .1, 1., 10., 1000000.):
                with self.subTest(shape=shape, dim=dim, scale=scale):
                    scores = torch.randn(shape, dtype=torch.float64) * scale
                    mask = torch.rand(shape) > .35
                    mask.select(dim, 0).fill_(True)
                    actual = entmax15(scores, mask, dim)
                    expected = bisection_reference(scores, mask, dim)
                    torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-12)
                    torch.testing.assert_close(actual.sum(dim), torch.ones_like(actual.sum(dim)), atol=3e-13, rtol=3e-12)
                    self.assertTrue((actual[~mask] == 0).all())

    def test_exact_zero_support_and_single_valid_token(self):
        scores = torch.tensor([[4., 1., 0., -10.], [float('nan'), 2., float('inf'), -5.]])
        mask = torch.tensor([[1, 1, 1, 1], [0, 1, 0, 0]], dtype=torch.bool)
        expected = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        torch.testing.assert_close(entmax15(scores, mask), expected)

    def test_padding_and_common_score_shift_do_not_change_weights(self):
        scores = torch.randn(3, 31, dtype=torch.float64)
        mask = torch.rand(3, 31) > .3
        mask[:, 9] = True
        a = entmax15(scores, mask)
        changed = scores + torch.tensor([100., -70., 15.])[:, None]
        changed[~mask] = float('nan')
        torch.testing.assert_close(a, entmax15(changed, mask), atol=1e-14, rtol=2e-13)

    def test_autograd_matches_finite_differences_with_nontrivial_support(self):
        scores = torch.tensor([[.32, -.27, .61, -3.4, .11],
                               [-.34, .09, .42, -.73, .13]], dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 1]], dtype=torch.bool)
        self.assertTrue(torch.autograd.gradcheck(lambda x: entmax15(x, mask), (scores,), eps=1e-6, atol=1e-6, rtol=1e-5))
        cotangent = torch.randn_like(scores)
        gradient, = torch.autograd.grad((entmax15(scores, mask) * cotangent).sum(), scores)
        torch.testing.assert_close(gradient.sum(1), torch.zeros(2, dtype=torch.float64), atol=1e-15, rtol=1e-10)
        self.assertTrue((gradient[~mask] == 0).all())
        self.assertGreater(float(gradient.abs().sum()), .1)

    def test_long_mixed_precision_scores_are_normalized_with_finite_gradient(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            scores = (torch.randn(3, 4096) * .02).to(dtype).requires_grad_()
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask[0, :4000] = False
            mask[1, 900:] = False
            output = entmax15(scores, mask)
            self.assertEqual(output.dtype, torch.float32)
            torch.testing.assert_close(output.sum(1), torch.ones(3), atol=2e-5, rtol=2e-5)
            (output * torch.randn_like(output)).sum().backward()
            self.assertTrue(torch.isfinite(scores.grad).all())
            self.assertEqual(float(scores.grad[~mask].abs().sum()), 0.)

    def test_invalid_inputs_fail_closed(self):
        for scores, mask in ((torch.ones(2, 3), torch.zeros(2, 3, dtype=torch.bool)),
                              (torch.tensor([[float('nan')]]), None),
                              (torch.ones(2, 3), torch.ones(2, 2)),
                              (torch.ones(2, 0), None),
                              (torch.ones(2, 3, dtype=torch.long), None)):
            with self.subTest(scores=scores, mask=mask), self.assertRaises(ValueError):
                entmax15(scores, mask)


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(51)
        self.raw = torch.randn(4, 7, 12)
        self.mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0],
                                  [0, 0, 1, 1, 1, 1, 1],
                                  [0, 1, 1, 0, 1, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        self.heads, self.audit = build_matched_heads(12, width=8, dropout=0., chunk_size=3)

    def activate_local(self, head):
        with torch.no_grad():
            head.pool_output.weight.copy_(torch.eye(8))

    def test_capacity_initialization_and_rng_preservation(self):
        self.assertEqual(self.audit['parameters'], parameter_count(self.heads[ARMS[0]]))
        self.assertEqual(len({parameter_count(head) for head in self.heads.values()}), 1)
        state = torch.get_rng_state().clone()
        build_matched_heads(12, width=8, seed=29)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        with torch.no_grad():
            self.heads[ARMS[-1]].classifier.bias.add_(1)
        with self.assertRaises(ValueError):
            initialization_audit(self.heads)

    def test_last_indices_arbitrary_padding_and_invalid_mask(self):
        torch.testing.assert_close(last_valid_indices(self.mask), torch.tensor([3, 6, 4, 0]))
        with self.assertRaises(ValueError):
            last_valid_indices(torch.zeros(2, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            self.heads[ARMS[0]](self.raw, self.mask[:, :-1])

    def test_zero_residual_matches_original_last_classifier(self):
        torch.manual_seed(42)
        old = ExperimentHead(12, mode='last', width=8, dropout=0., layers=(32,))
        for head in self.heads.values():
            torch.testing.assert_close(old.project.weight, head.project.weight, atol=0, rtol=0)
            torch.testing.assert_close(old.classifier.weight, head.classifier.weight, atol=0, rtol=0)
            actual = head(self.raw, self.mask)
            expected = old({'layers': (self.raw,), 'mask': self.mask})
            torch.testing.assert_close(actual['logits'], expected['logits'])
            torch.testing.assert_close(actual['representation'], expected['representation'])

    def test_gated_control_exactly_reproduces_existing_attention_head(self):
        head = self.heads['gated_attention_softmax']
        self.activate_local(head)
        old = ExperimentHead(12, mode='attention', width=8, dropout=0., layers=(32,), chunk_size=3)
        old.project.load_state_dict(head.project.state_dict())
        old.classifier.load_state_dict(head.classifier.state_dict())
        old.token_project.load_state_dict(head.token_project.state_dict())
        old.pool_output.load_state_dict(head.pool_output.state_dict())
        old.attention_tanh.load_state_dict(head.key.state_dict())
        old.attention_gate.load_state_dict(head.query_global.state_dict())
        with torch.no_grad():
            old.attention_context.weight.copy_(head.task_query[None])
        expected = old({'layers': (self.raw,), 'mask': self.mask})
        actual = head(self.raw, self.mask)
        torch.testing.assert_close(actual['logits'], expected['logits'])
        torch.testing.assert_close(actual['attention_weights'], expected['attention_weights'])

    def test_masked_states_and_left_right_padding_do_not_change_predictions(self):
        changed = self.raw.clone()
        changed[~self.mask] = float('nan')
        for arm, head in self.heads.items():
            with self.subTest(arm=arm):
                self.activate_local(head)
                a, b = head(self.raw, self.mask), head(changed, self.mask)
                torch.testing.assert_close(a['logits'], b['logits'])
                self.assertTrue((a['attention_weights'][~self.mask] == 0).all())
                torch.testing.assert_close(a['attention_weights'].sum(1), torch.ones(4), atol=1e-6, rtol=1e-6)
                packed = self.raw[0, self.mask[0]]
                left = torch.cat((torch.zeros(3, 12), packed))[None]
                right = torch.cat((packed, torch.zeros(3, 12)))[None]
                mleft = torch.tensor([[0, 0, 0, 1, 1, 1, 1]], dtype=torch.bool)
                mright = torch.tensor([[1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
                torch.testing.assert_close(head(left, mleft)['logits'], head(right, mright)['logits'])

    def test_chunked_projection_is_identical(self):
        for head in self.heads.values():
            self.activate_local(head)
            full = copy.deepcopy(head)
            full.chunk_size = 10000
            a, b = head(self.raw, self.mask), full(self.raw, self.mask)
            for key in ('logits', 'representation', 'attention_weights', 'query'):
                torch.testing.assert_close(a[key], b[key])

    def test_repeat_last_control_ignores_all_other_valid_tokens(self):
        head = self.heads['repeat_last']
        self.activate_local(head)
        altered = self.raw.clone()
        altered += torch.randn_like(altered) * 100
        rows = torch.arange(len(self.mask))
        last = last_valid_indices(self.mask)
        altered[rows, last] = self.raw[rows, last]
        torch.testing.assert_close(head(altered, self.mask)['logits'], head(self.raw, self.mask)['logits'])

    def test_queries_static_or_conditioned_by_actual_global_representation(self):
        changed = self.raw.clone()
        rows = torch.arange(len(self.mask))
        last = last_valid_indices(self.mask)
        changed[rows, last] = torch.randn(4, 12)
        static = self.heads['static_query_softmax']
        torch.testing.assert_close(static(self.raw, self.mask)['query'], static(changed, self.mask)['query'])
        for arm in ('global_query_softmax', 'global_query_entmax15'):
            head = self.heads[arm]
            a, b = head(self.raw, self.mask)['query'], head(changed, self.mask)['query']
            self.assertGreater(float((a - b).abs().sum()), .01)

    def test_step_two_activates_expected_routes_and_padding_gradients_stay_zero(self):
        labels = torch.tensor([0., 1., 1., 0.])
        for arm, head in self.heads.items():
            with self.subTest(arm=arm):
                optimizer = torch.optim.SGD(head.parameters(), lr=.1)
                for step in (1, 2):
                    optimizer.zero_grad(set_to_none=True)
                    raw = self.raw.clone().requires_grad_()
                    loss = F.binary_cross_entropy_with_logits(head(raw, self.mask)['logits'], labels)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    for name, parameter in head.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertEqual(float(raw.grad[~self.mask].abs().sum()), 0.)
                    if step == 1:
                        self.assertEqual(float(head.token_project[0].weight.grad.abs().sum()), 0.)
                        self.assertGreater(float(head.pool_output.weight.grad.abs().sum()), .01)
                    else:
                        self.assertGreater(float(head.token_project[0].weight.grad.abs().sum()), 1e-5)
                        if arm == 'static_query_softmax':
                            self.assertEqual(float(head.query_global.weight.grad.abs().sum()), 0.)
                        elif arm != 'repeat_last':
                            self.assertGreater(float(head.query_global.weight.grad.abs().sum()), 1e-8)
                        if arm != 'repeat_last':
                            self.assertGreater(float(head.key.weight.grad.abs().sum()), 1e-8)
                            self.assertGreater(float(head.task_query.grad.abs().sum()), 1e-8)
                    optimizer.step()


if __name__ == '__main__':
    unittest.main()
