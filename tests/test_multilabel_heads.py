"""CPU checks for independent labels, query alignment and unknown supervision."""
import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from scripts.hanguard.multilabel_heads import MultiLabelHead, masked_binary_cross_entropy


class MultiLabelReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(472)
        self.hidden = torch.randn(3, 6, 12)
        self.mask = torch.tensor([[1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
        self.descriptions = torch.randn(5, 12)

    def head(self, mode, **kwargs):
        arguments = dict(hidden_size=12, width=8, mode=mode, dropout=0.)
        if mode == 'description_queries':
            arguments['query_vectors'] = self.descriptions
        arguments.update(kwargs)
        return MultiLabelHead(**arguments)

    def test_five_independent_probabilities_allow_multiple_positives(self):
        for mode in ('last_mlp', 'learned_queries', 'description_queries'):
            with self.subTest(mode=mode):
                head = self.head(mode)
                classifier = head.classifier if mode == 'last_mlp' else head.shared_scorer[-1]
                with torch.no_grad():
                    classifier.weight.zero_()
                    classifier.bias.fill_(2.)
                result = head(self.hidden, self.mask)
                self.assertEqual(result['logits'].shape, (3, 5))
                torch.testing.assert_close(result['probabilities'], result['logits'].sigmoid())
                self.assertTrue((result['probabilities'] > .5).all())
                self.assertTrue((result['probabilities'].sum(1) > 1.).all())

    def test_padding_and_nonfinite_padding_cannot_change_predictions(self):
        for mode in ('last_mlp', 'learned_queries', 'description_queries'):
            with self.subTest(mode=mode):
                head = self.head(mode)
                changed = self.hidden.clone()
                changed[~self.mask] = float('nan')
                expected = head(self.hidden, self.mask, return_attention=True)
                actual = head(changed, self.mask, return_attention=True)
                torch.testing.assert_close(actual['logits'], expected['logits'])
                if mode != 'last_mlp':
                    self.assertEqual(actual['attention'].shape, (3, 5, 6))
                    torch.testing.assert_close(actual['attention'].sum(-1), torch.ones(3, 5))
                    self.assertEqual(float(actual['attention'].masked_select(~self.mask[:, None]).abs().sum()), 0.)

    def test_random_and_description_queries_have_identical_used_parameter_schema(self):
        torch.manual_seed(81)
        random = self.head('learned_queries', query_seed=19)
        torch.manual_seed(81)
        semantic = self.head('description_queries')
        random_params, semantic_params = dict(random.named_parameters()), dict(semantic.named_parameters())
        self.assertEqual(random_params.keys(), semantic_params.keys())
        for name in random_params:
            torch.testing.assert_close(random_params[name], semantic_params[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(random.query_vectors, semantic.query_vectors))
        self.assertFalse(semantic.query_vectors.requires_grad)
        for head in (random, semantic):
            result = head(self.hidden, self.mask)
            result['logits'].square().sum().backward()
            self.assertTrue(all(parameter.grad is not None for parameter in head.parameters()))

    def test_five_categories_share_one_scorer_and_only_queries_scale_with_count(self):
        small = MultiLabelHead(12, width=8, num_labels=3, mode='learned_queries', dropout=0.)
        large = self.head('learned_queries')
        self.assertEqual(sum(p.numel() for p in large.parameters()) - sum(p.numel() for p in small.parameters()), 2 * 8)
        scalars = [module for module in large.modules() if isinstance(module, nn.Linear) and module.out_features == 1]
        self.assertEqual(len(scalars), 1)
        permuted = copy.deepcopy(large)
        order = torch.tensor([2, 4, 0, 3, 1])
        with torch.no_grad():
            permuted.query_vectors.copy_(large.query_vectors[order])
            permuted.query_residual.copy_(large.query_residual[order])
        a = large(self.hidden, self.mask, return_attention=True)
        b = permuted(self.hidden, self.mask, return_attention=True)
        torch.testing.assert_close(a['logits'][:, order], b['logits'])
        torch.testing.assert_close(a['attention'][:, order], b['attention'])

    def test_description_vectors_remain_fixed_while_mapping_and_query_residual_learn(self):
        descriptions = self.descriptions.clone().requires_grad_()
        hidden = self.hidden.clone().requires_grad_()
        head = self.head('description_queries', query_vectors=descriptions)
        targets = torch.tensor([[0, 1, 1, 0, 1], [1, 0, 0, 1, 0], [1, 1, 0, 0, 1]])
        result = head(hidden, self.mask)
        masked_binary_cross_entropy(result['logits'], targets).backward()
        self.assertIsNone(descriptions.grad)
        self.assertGreater(float(head.query_project.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(head.query_residual.grad.abs().sum()), 0.)
        self.assertGreater(float(hidden.grad[self.mask].abs().sum()), 0.)
        self.assertEqual(float(hidden.grad[~self.mask].abs().sum()), 0.)

    def test_shared_query_key_mapping_preserves_exact_match_alignment(self):
        vectors = torch.eye(5, 8)
        head = MultiLabelHead(8, width=8, mode='description_queries', query_vectors=vectors, dropout=0.)
        with torch.no_grad():
            head.query_project.weight.copy_(torch.eye(8))
            head.query_project.bias.zero_()
        result = head(vectors[:3][None], torch.ones(1, 3, dtype=torch.bool), return_attention=True)
        self.assertEqual(result['attention'][0, :3].argmax(-1).tolist(), [0, 1, 2])

    def test_class_identity_survives_single_token_pool(self):
        head = self.head('description_queries')
        result = head(self.hidden[:, :1], torch.ones(3, 1, dtype=torch.bool), return_attention=True)
        torch.testing.assert_close(result['attention'], torch.ones(3, 5, 1))
        torch.testing.assert_close(result['representation'][:, 0], result['representation'][:, 1])
        self.assertGreater(float((result['logits'][:, 0] - result['logits'][:, 1]).abs().sum()), 1e-5)

    def test_bfloat16_features_and_autocast_still_produce_float32_logits(self):
        for mode in ('last_mlp', 'learned_queries', 'description_queries'):
            head = self.head(mode)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                result = head(self.hidden.bfloat16(), self.mask)
            self.assertEqual(result['logits'].dtype, torch.float32)
            self.assertTrue(torch.isfinite(result['logits']).all())

    def test_invalid_input_and_missing_description_fail_explicitly(self):
        with self.assertRaises(ValueError):
            MultiLabelHead(12, mode='description_queries')
        with self.assertRaises(ValueError):
            self.head('description_queries', query_vectors=torch.randn(4, 12))
        head = self.head('learned_queries')
        invalid = self.mask.clone()
        invalid[0] = False
        with self.assertRaises(ValueError):
            head(self.hidden, invalid)


class UnknownLabelLossTests(unittest.TestCase):
    def test_unknown_targets_have_exactly_zero_logit_gradient(self):
        torch.manual_seed(920)
        logits = torch.randn(2, 5, requires_grad=True)
        targets = torch.tensor([[1, 0, -1, -1, 1], [-1, 1, 0, -1, -1]])
        loss = masked_binary_cross_entropy(logits, targets)
        known = targets != -1
        expected = F.binary_cross_entropy_with_logits(logits[known], targets[known].float())
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(float(logits.grad[~known].abs().sum()), 0.)
        self.assertGreater(float(logits.grad[known].abs().sum()), 0.)

    def test_known_mask_can_suppress_but_not_make_unknown_known(self):
        logits = torch.tensor([[.2, -.1, .4, .7, -.5]], requires_grad=True)
        targets = torch.tensor([[1., 0., -1., 1., 0.]])
        mask = torch.tensor([[True, False, True, False, True]])
        actual = masked_binary_cross_entropy(logits, targets, known_mask=mask, reduction='none')
        keep = torch.tensor([[True, False, False, False, True]])
        torch.testing.assert_close(actual[keep], F.binary_cross_entropy_with_logits(logits[keep], targets[keep], reduction='none'))
        self.assertEqual(float(actual[~keep].abs().sum()), 0.)
        actual.sum().backward()
        self.assertEqual(float(logits.grad[~keep].abs().sum()), 0.)

    def test_all_unknown_batches_are_differentiable_zero(self):
        for reduction in ('none', 'mean', 'sum'):
            logits = torch.randn(4, 5, requires_grad=True)
            loss = masked_binary_cross_entropy(logits, torch.full((4, 5), -1), reduction=reduction)
            self.assertEqual(float(loss.sum()), 0.)
            loss.sum().backward()
            torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_unknown_class_query_residual_gets_no_class_loss_gradient(self):
        head = MultiLabelHead(7, width=8, mode='learned_queries', dropout=0.)
        result = head(torch.randn(3, 4, 7), torch.ones(3, 4, dtype=torch.bool))
        labels = torch.tensor([[-1, 1, 0, 1, 0], [-1, 0, 1, 1, 0], [-1, 1, 1, 0, 1]])
        masked_binary_cross_entropy(result['logits'], labels).backward()
        self.assertEqual(float(head.query_residual.grad[0].abs().sum()), 0.)
        self.assertGreater(float(head.query_residual.grad[1:].abs().sum()), 0.)

    def test_positive_weights_apply_only_to_known_category_entries(self):
        logits = torch.tensor([[1., 2., -1., .2, -.2], [-1., 1., 0., .1, -.1]])
        targets = torch.tensor([[1., 0., -1., 1., 0.], [0., 1., 1., -1., -1.]])
        weights = torch.tensor([2., 3., 4., 5., 6.])
        valid = targets != -1
        expected = F.binary_cross_entropy_with_logits(logits, targets.clamp_min(0), pos_weight=weights, reduction='none')[valid].mean()
        torch.testing.assert_close(masked_binary_cross_entropy(logits, targets, pos_weight=weights), expected)

    def test_invalid_known_label_rejected(self):
        with self.assertRaises(ValueError):
            masked_binary_cross_entropy(torch.zeros(1, 5), torch.tensor([[0, 1, 2, -1, 0]]))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
