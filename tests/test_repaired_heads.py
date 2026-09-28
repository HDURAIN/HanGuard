"""CPU checks for shared-head gradients, masks and persistent disambiguation."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.hanguard.repaired_heads import (
    ExperimentHead, ShieldState, TapOnline, supervised_contrastive_loss,
)


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.mask = torch.tensor([[1, 1, 1, 0, 0], [0, 1, 1, 1, 1]], dtype=torch.bool)
        self.raw = tuple(torch.randn(2, 5, 12) for _ in range(4))

    def head(self, mode, **kwargs):
        return ExperimentHead(12, mode=mode, width=8, layers=(1, 2, 3, 4), dropout=0., chunk_size=3, **kwargs)

    def test_all_modes_mask_padding_and_shapes(self):
        for mode in ('last', 'fusion', 'repeat_last', 'mean', 'attention', 'fusion_attention'):
            with self.subTest(mode=mode):
                head = self.head(mode)
                with torch.no_grad():
                    for name in ('fusion_output', 'pool_output'):
                        if hasattr(head, name):
                            getattr(head, name).weight.copy_(torch.eye(8))
                raw = self.raw if mode in ('fusion', 'fusion_attention') else self.raw[-1:]
                changed = tuple(x.clone() for x in raw)
                for x in changed:
                    x[~self.mask] = torch.randn_like(x[~self.mask]) * 1000.
                result = head({'layers': raw, 'mask': self.mask})
                altered = head({'layers': changed, 'mask': self.mask})
                self.assertEqual(result['logits'].shape, (2,))
                self.assertEqual(result['representation'].shape, (2, 8))
                torch.testing.assert_close(result['logits'], altered['logits'])
                if 'attention_weights' in result:
                    self.assertEqual(float(result['attention_weights'][~self.mask].sum()), 0.)
                    torch.testing.assert_close(result['attention_weights'].sum(1), torch.ones(2))

    def test_shared_token_classifier_and_last_sentence_equal(self):
        for mode in ('last', 'fusion'):
            head = self.head(mode)
            raw = self.raw if mode == 'fusion' else self.raw[-1:]
            result = head({'layers': raw, 'mask': self.mask}, return_tokens=True)
            sentence = head({'layers': raw, 'mask': self.mask})
            torch.testing.assert_close(result['logits'], result['token_logits'][torch.arange(2), torch.tensor([2, 4])])
            torch.testing.assert_close(result['logits'], sentence['logits'])
            self.assertFalse(result['teacher_probabilities'].requires_grad)
            self.assertTrue(all(not x.requires_grad for x in result['prototype_features']))
            self.assertEqual(len([name for name, _ in head.named_modules() if name == 'classifier']), 1)

    def test_repeated_last_is_parameter_matched_control(self):
        fusion, control = self.head('fusion'), self.head('repeat_last')
        control.load_state_dict(fusion.state_dict())
        with torch.no_grad():
            fusion.fusion_output.weight.copy_(torch.eye(8))
            control.fusion_output.weight.copy_(torch.eye(8))
        self.assertEqual(sum(x.numel() for x in fusion.parameters()), sum(x.numel() for x in control.parameters()))
        a = fusion({'layers': (self.raw[-1],) * 4, 'mask': self.mask})
        b = control({'layers': self.raw[-1:], 'mask': self.mask})
        torch.testing.assert_close(a['logits'], b['logits'])

    def test_gradients_reach_all_raw_layers(self):
        head = self.head('fusion')
        with torch.no_grad():
            head.fusion_output.weight.copy_(torch.eye(8))
        raw = tuple(x.clone().requires_grad_() for x in self.raw)
        result = head({'layers': raw, 'mask': self.mask}, return_tokens=True)
        result['token_logits'][self.mask].square().mean().backward()
        for layer in raw:
            self.assertGreater(float(layer.grad.abs().sum()), 0.)
            self.assertEqual(float(layer.grad[~self.mask].abs().sum()), 0.)

    def test_chunked_projection_equals_unchunked(self):
        head = self.head('fusion_attention')
        full = copy.deepcopy(head)
        full.chunk_size = 10000
        a = head({'layers': self.raw, 'mask': self.mask}, return_tokens=True)
        b = full({'layers': self.raw, 'mask': self.mask}, return_tokens=True)
        torch.testing.assert_close(a['representation'], b['representation'])
        torch.testing.assert_close(a['token_logits'], b['token_logits'])

    def test_prototype_readout_is_trainable_and_finite(self):
        head = self.head('fusion', readout='prototype', prototypes_per_class=2)
        z = torch.randn(24, 8)
        y = torch.tensor([0] * 12 + [1] * 12)
        head.initialize_prototypes(z, y, seed=3)
        result = head({'layers': self.raw, 'mask': self.mask})
        loss = result['logits'].square().mean() + .01 * head.prototype_regularization()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(head.prototypes.grad.abs().sum()), 0.)


class ShieldTests(unittest.TestCase):
    def state(self, **kwargs):
        defaults = dict(sample_ids=['a', 'b'], lengths=[2, 3], labels=[0, 1], hidden_size=3,
                        layer_count=1, warmup_steps=1, ramp_steps=2, topk=2, chunk_size=2)
        defaults.update(kwargs)
        return ShieldState(**defaults)

    def batch(self):
        logits = torch.tensor([[-.5, .4, 0.], [.9, -.8, .2]], requires_grad=True)
        raw = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [9., 9., 9.]],
                            [[0., 0., 1.], [1., 1., 0.], [1., 0., 1.]]], requires_grad=True)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        return logits, raw, mask

    def test_fixed_is_equal_sentence_weight_and_never_changes(self):
        state = self.state(mode='fixed')
        logits, raw, mask = self.batch()
        before = state.targets.copy()
        loss, stats = state.loss(logits, (raw,), ['a', 'b'], mask, 20)
        expected = (F.binary_cross_entropy_with_logits(logits[0, :1], torch.zeros(1)) +
                    F.binary_cross_entropy_with_logits(logits[1, :2], torch.ones(2))) / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(raw.grad)
        self.assertEqual(float(logits.grad[~mask].sum()), 0.)
        self.assertEqual(float(logits.grad[0, 1]), 0.)
        self.assertEqual(float(logits.grad[1, 2]), 0.)
        state.end_macro(20)
        np.testing.assert_array_equal(before, state.targets)
        self.assertFalse(state.initialized.any())
        self.assertFalse(stats['active'])

    def test_warmup_updates_prototypes_but_keeps_initial_labels(self):
        state = self.state()
        logits, raw, mask = self.batch()
        before = state.targets.copy()
        _, stats = state.loss(logits, (raw,), ['a', 'b'], mask, 0)
        state.end_macro(0)
        self.assertFalse(stats['active'])
        np.testing.assert_array_equal(before, state.targets)
        torch.testing.assert_close(state.centers.norm(dim=-1), torch.ones(1, 2))
        self.assertTrue(state.initialized.all())
        _, stats = state.loss(logits, (raw,), ['a', 'b'], mask, 1)
        self.assertTrue(stats['active'])
        self.assertEqual(stats['updated_layer_positions'], 3)
        self.assertGreater(np.abs(before - state.targets).sum(), 0.)
        self.assertTrue(np.all(state.initial_weight[:, [0, 2, 3]] < 1.))
        np.testing.assert_array_equal(state.targets[:, [1, 4]], before[:, [1, 4]])

    def test_global_topk_and_sequential_ema_ignore_microbatch_partition(self):
        a, b = self.state(), self.state()
        logits, raw, mask = self.batch()
        a.loss(logits, (raw,), ['a', 'b'], mask, 0)
        for row, sample in enumerate(['a', 'b']):
            b.loss(logits[row:row + 1], (raw[row:row + 1],), [sample], mask[row:row + 1], 0)
        a.end_macro(0)
        b.end_macro(0)
        torch.testing.assert_close(a.centers, b.centers)
        auxiliary_mask = mask.clone()
        auxiliary_mask[0, 1] = False
        auxiliary_mask[1, 2] = False
        valid = raw.detach()[auxiliary_mask]
        p = logits.detach().sigmoid()[auxiliary_mask]
        idx = torch.stack((1-p, p)).topk(2, dim=1).indices
        selected = valid[idx]
        expected = torch.zeros(2, 3)
        for i in range(2):
            expected = F.normalize(.99 * expected + .01 * selected[:, i], dim=-1)
        torch.testing.assert_close(a.centers[0], expected)

    def test_shared_average_target_equals_average_layer_bce(self):
        state = self.state(mode='fixed', layer_count=2)
        state.targets[0] = [.1, .2, .3, .4, .5]
        state.targets[1] = [.8, .7, .6, .5, .4]
        logits, raw, mask = self.batch()
        loss, _ = state.loss(logits, (raw, raw), ['a', 'b'], mask, 0)
        expected = []
        for targets in state.targets:
            expected.append((F.binary_cross_entropy_with_logits(logits[0, :1], torch.tensor(targets[:1])) +
                             F.binary_cross_entropy_with_logits(logits[1, :2], torch.tensor(targets[2:4]))) / 2)
        torch.testing.assert_close(loss, torch.stack(expected).mean())

    def test_persistent_identity_roundtrip_and_invalid_ids(self):
        state = self.state()
        logits, raw, mask = self.batch()
        state.loss(logits, (raw,), ['a', 'b'], mask, 0)
        with self.assertRaises(RuntimeError):
            state.state_dict()
        state.end_macro(0)
        state.loss(logits, (raw,), ['a', 'b'], mask, 2)
        state.end_macro(2)
        saved = state.state_dict()
        restored = self.state()
        restored.load_state_dict(saved)
        np.testing.assert_array_equal(state.targets, restored.targets)
        self.assertEqual(state.diagnostics(), restored.diagnostics())
        with self.assertRaises(KeyError):
            restored.loss(logits, (raw,), ['validation-id', 'b'], mask, 3)
        with self.assertRaises(ValueError):
            restored.load_state_dict(dict(saved, ids=('different', 'b')))

    def test_padding_does_not_enter_prototypes(self):
        a, b = self.state(), self.state()
        logits, raw, mask = self.batch()
        altered = raw.detach().clone()
        altered[~mask] = 1e10
        a.loss(logits, (raw,), ['a', 'b'], mask, 0)
        b.loss(logits, (altered,), ['a', 'b'], mask, 0)
        a.end_macro(0)
        b.end_macro(0)
        torch.testing.assert_close(a.centers, b.centers)

    def test_single_token_has_sentence_only_and_no_nan(self):
        state = self.state(sample_ids=['one'], lengths=[1], labels=[1])
        logits = torch.tensor([[2.]], requires_grad=True)
        raw = torch.randn(1, 1, 3, requires_grad=True)
        loss, stats = state.loss(logits, (raw,), ['one'], torch.ones(1, 1, dtype=torch.bool), 2)
        self.assertEqual(float(loss), 0.)
        self.assertEqual(stats['positions'], 0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        state.end_macro(2)
        self.assertFalse(state.initialized.any())


class ContrastiveTests(unittest.TestCase):
    def test_finite_gradient_and_same_class_geometry(self):
        good = torch.tensor([[1., .1], [1., -.1], [-1., .1], [-1., -.1]], requires_grad=True)
        bad = good.detach()[torch.tensor([0, 2, 1, 3])].requires_grad_()
        labels = torch.tensor([0, 0, 1, 1])
        loss = supervised_contrastive_loss(good, labels)
        self.assertLess(float(loss), float(supervised_contrastive_loss(bad, labels)))
        loss.backward()
        self.assertTrue(torch.isfinite(good.grad).all())

    def test_no_positive_is_differentiable_zero(self):
        z = torch.randn(2, 5, requires_grad=True)
        loss = supervised_contrastive_loss(z, torch.tensor([0, 1]))
        self.assertEqual(float(loss), 0.)
        loss.backward()
        torch.testing.assert_close(z.grad, torch.zeros_like(z))


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 12)
        self.layers = nn.ModuleList([nn.Linear(12, 12) for _ in range(4)])

    def forward(self, input_ids, **kwargs):
        z = self.embedding(input_ids)
        for layer in self.layers:
            z = layer(z)
        return z


class TapTests(unittest.TestCase):
    def test_online_raw_hooks_preserve_backbone_gradients_and_close(self):
        backbone = FakeBackbone()
        model = SimpleNamespace(model=SimpleNamespace(language_model=backbone))
        head = ExperimentHead(12, mode='fusion', layers=(1, 2, 3, 4), width=8, dropout=0.)
        tap = TapOnline(model, head)
        result = tap({'input_ids': torch.tensor([[2, 3, 4], [5, 6, 0]]),
                      'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 0]])}, return_tokens=True)
        self.assertEqual(len(result['prototype_features']), 4)
        result['logits'].sum().backward()
        self.assertGreater(float(backbone.embedding.weight.grad.abs().sum()), 0.)
        self.assertEqual(tap.captured, {})
        tap.close()
        self.assertTrue(all(not layer._forward_hooks for layer in backbone.layers))

    def test_frozen_warmup_ignores_input_grad_hook_but_trains_head_and_restores_joint(self):
        class FakeLoRABlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(12, 12)
                self.lora_A = nn.Linear(12, 3, bias=False)
                self.lora_B = nn.Linear(3, 12, bias=False)

            def forward(self, value):
                return self.base(value) + self.lora_B(self.lora_A(value))

        backbone = FakeBackbone()
        backbone.layers[-1] = FakeLoRABlock()
        backbone.requires_grad_(False)
        backbone.eval()
        # Reproduce PEFT enable_input_require_grads without importing PEFT.
        input_hook = backbone.embedding.register_forward_hook(
            lambda module, inputs, output: output.requires_grad_(True))
        observed = []
        output_hook = backbone.layers[-1].register_forward_hook(
            lambda module, inputs, output: observed.append((output.requires_grad, output.grad_fn)))
        model = SimpleNamespace(model=SimpleNamespace(language_model=backbone))
        head = ExperimentHead(12, mode='fusion', layers=(1, 2, 3, 4), width=8, dropout=0.)
        tap = TapOnline(model, head)
        batch = {'input_ids': torch.tensor([[2, 3, 4], [5, 6, 0]]),
                 'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 0]])}

        warm = tap(batch)
        self.assertEqual(observed[-1], (False, None))
        self.assertFalse(tap.last_forward_diagnostics['backbone_grad_enabled'])
        self.assertFalse(any(tap.last_forward_diagnostics['captured_requires_grad']))
        self.assertTrue(warm['logits'].requires_grad)
        warm['logits'].square().sum().backward()
        self.assertGreater(float(head.classifier.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in backbone.parameters()))

        backbone.layers[-1].lora_A.requires_grad_(True)
        backbone.layers[-1].lora_B.requires_grad_(True)
        backbone.train()
        head.zero_grad(set_to_none=True)
        joint = tap(batch)
        self.assertTrue(observed[-1][0])
        self.assertTrue(tap.last_forward_diagnostics['backbone_grad_enabled'])
        self.assertTrue(all(tap.last_forward_diagnostics['captured_requires_grad']))
        self.assertIsNotNone(observed[-1][1])
        joint['logits'].square().sum().backward()
        for module in (backbone.layers[-1].lora_A, backbone.layers[-1].lora_B):
            self.assertGreater(float(module.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for name, parameter in backbone.named_parameters()
                            if 'lora_' not in name))

        # SCL's detached first pass/evaluation must not be re-enabled by the tap.
        with torch.no_grad():
            detached = tap(batch)
        self.assertEqual(observed[-1], (False, None))
        self.assertFalse(tap.last_forward_diagnostics['backbone_grad_enabled'])
        self.assertFalse(detached['representation'].requires_grad)
        self.assertFalse(detached['logits'].requires_grad)
        self.assertTrue(torch.is_grad_enabled())
        tap.close()
        input_hook.remove()
        output_hook.remove()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
