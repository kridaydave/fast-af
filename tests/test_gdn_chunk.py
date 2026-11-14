import pytest
import torch
import torch.nn.functional as F

from scripts.d1_model import D1Config, D1LM, gated_delta_chunked, gated_delta_reference


def core_inputs(length, strong=False):
    torch.manual_seed(9)
    shape = (2, length, 2, 5)
    return [F.normalize(torch.randn(shape), dim=-1), F.normalize(torch.randn(shape), dim=-1),
            torch.randn(2, length, 2, 7),
            -torch.rand(2, length, 2) * (30 if strong else 0.2),
            torch.rand(2, length, 2), torch.randn(2, 2, 5, 7)]


@pytest.mark.parametrize('length', [1, 17, 31, 32, 33, 65, 129])
@pytest.mark.parametrize('strong', [False, True])
def test_chunk_outputs_states_and_all_input_gradients(length, strong):
    torch.set_num_threads(1)
    inputs = [t.requires_grad_() for t in core_inputs(length, strong)]
    expected, expected_state = gated_delta_reference(*inputs)
    actual, actual_state = gated_delta_chunked(*inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-4, atol=2e-5)
    probes = torch.randn_like(expected), torch.randn_like(expected_state)
    reference_grads = torch.autograd.grad((expected * probes[0]).sum() + (expected_state * probes[1]).sum(), inputs, retain_graph=True)
    actual_grads = torch.autograd.grad((actual * probes[0]).sum() + (actual_state * probes[1]).sum(), inputs)
    for a, b in zip(actual_grads, reference_grads):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-5)


def test_chunk_does_not_modify_initial_state_and_supports_split_stream():
    inputs = core_inputs(71)
    original = inputs[-1].clone()
    full, final = gated_delta_chunked(*inputs)
    left, state = gated_delta_chunked(*(t[:, :35] for t in inputs[:5]), inputs[-1])
    right, split_final = gated_delta_chunked(*(t[:, 35:] for t in inputs[:5]), state)
    torch.testing.assert_close(torch.cat((left, right), dim=1), full, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(split_final, final, atol=2e-5, rtol=2e-4)
    assert torch.equal(inputs[-1], original)


def test_chunk_model_cached_logits_and_parameter_gradients_match_reference():
    torch.manual_seed(4)
    cfg = D1Config(d_model=16, n_layers=4, query_heads=2, kv_heads=1, head_dim=8,
                   ffn_hidden=32, gdn_heads=2, gdn_key_dim=4, gdn_value_dim=8, block_size=4)
    reference = D1LM(64, 12, cfg, 'torch')
    chunked = D1LM(64, 12, cfg, 'chunk')
    chunked.load_state_dict(reference.state_dict())
    tokens = torch.randint(5, 64, (2, 12))
    left, right = reference(tokens), chunked(tokens)
    torch.testing.assert_close(left, right, rtol=2e-4, atol=1e-6)
    (left.square().sum()).backward()
    (right.square().sum()).backward()
    for (name, a), (_, b) in zip(reference.named_parameters(), chunked.named_parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=2e-6, msg=name)
    with torch.no_grad():
        cache = chunked.prefill(tokens[:, :8])
        cached, _ = chunked.forward_block(tokens[:, 8:], cache)
    torch.testing.assert_close(cached, right[:, 8:], rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize('chunk', [0, -1, True])
def test_invalid_chunk_size_rejected(chunk):
    with pytest.raises(ValueError, match='positive integer'):
        gated_delta_chunked(*core_inputs(4), chunk_size=chunk)
