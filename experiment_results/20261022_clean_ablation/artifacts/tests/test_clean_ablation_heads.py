"""CPU architecture checks for the new nine-arm study."""
import unittest
import torch
from scripts.hanguard.clean_ablation_heads import ARMS, MAIN_ARMS, CleanAblationHead
from scripts.hanguard.four_arm_query_heads import FourArmHead


class CleanHeadsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(5)
        self.last = torch.randn(3, 32, 6)
        self.tokens = torch.randn(3, 4, 6)
        self.mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)

    def build(self, arm, classes=2, selected_layer=32):
        return CleanAblationHead(6, width=4, num_classes=classes, arm=arm,
                                 dropout=0., seed=42, selected_layer=selected_layer)

    def test_original_four_keep_source_architecture_and_outputs(self):
        for arm in MAIN_ARMS:
            with self.subTest(arm=arm):
                new = self.build(arm)
                reference = FourArmHead(6, width=4, arm=arm, dropout=0., seed=42)
                self.assertEqual(new.parameter_count(), reference.parameter_count())
                self.assertEqual(set(new.state_dict()), set(reference.state_dict()))
                torch.testing.assert_close(new(self.last, self.tokens, self.mask),
                                           reference(self.last[:, [7, 15, 23, 31]], self.tokens, self.mask), rtol=0., atol=0.)

    def test_all_nine_shapes_and_actual_parameter_counts(self):
        for classes in (2, 5):
            for arm in ARMS:
                with self.subTest(classes=classes, arm=arm):
                    head = self.build(arm, classes)
                    output = head(self.last, self.tokens, self.mask)
                    self.assertEqual(output.shape, (3, classes))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(head.metadata()['trainable_parameters'], sum(p.numel() for p in head.parameters()))

    def test_paired_initial_logits_are_equal(self):
        for a, b in (('baseline', 'fusion_only'), ('category_only', 'fusion_category'),
                     ('fusion_only', 'fusion_repeat_last'), ('fusion_category', 'category_fusion_repeat_last')):
            torch.testing.assert_close(self.build(a)(self.last, self.tokens, self.mask),
                                       self.build(b)(self.last, self.tokens, self.mask), atol=0., rtol=0.)

    def test_repeat_control_has_same_capacity_and_uses_only_last_layer(self):
        original, repeated = self.build('fusion_only'), self.build('fusion_repeat_last')
        repeated.load_state_dict(original.state_dict())
        for model in (original, repeated):
            torch.nn.init.eye_(model.branch.fusion_output.weight)
        self.assertEqual(original.parameter_count(), repeated.parameter_count())
        altered = self.last.clone()
        altered[:, :31] += torch.randn_like(altered[:, :31])
        torch.testing.assert_close(repeated(self.last), repeated(altered), atol=0., rtol=0.)
        self.assertFalse(torch.equal(original(self.last), original(altered)))

    def test_selected_layer_reads_fresh_selected_layer(self):
        selected = self.build('probe_selected_single', selected_layer=7)
        baseline = self.build('baseline')
        last = self.last.clone()
        last[:, 31] = last[:, 6]
        torch.testing.assert_close(selected(self.last), baseline(last), atol=0., rtol=0.)

    def test_shared_query_same_parameters_and_all_queries_receive_gradient(self):
        for regular, shared in (('category_only', 'category_shared'), ('fusion_category', 'fusion_category_shared')):
            a, b = self.build(regular, 5), self.build(shared, 5)
            self.assertEqual(a.parameter_count(), b.parameter_count())
            self.assertEqual(set(a.state_dict()), set(b.state_dict()))
            loss = torch.nn.functional.cross_entropy(b(self.last, self.tokens, self.mask), torch.tensor([0, 1, 2]))
            loss.backward()
            self.assertTrue((b.query_head.query_residual.grad.abs().sum(1) > 0).all())
            self.assertTrue(all(p.grad is not None for p in b.parameters()))

    def test_invalid_shapes_and_nan_padding(self):
        head = self.build('category_shared')
        tokens = self.tokens.clone()
        tokens[~self.mask] = float('nan')
        torch.testing.assert_close(head(self.last, self.tokens, self.mask), head(self.last, tokens, self.mask))
        with self.assertRaises(ValueError):
            head(self.last[:, :4], tokens, self.mask)
        with self.assertRaises(ValueError):
            head(self.last, tokens, torch.zeros_like(self.mask))


if __name__ == '__main__':
    unittest.main()
