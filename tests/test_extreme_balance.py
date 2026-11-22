import pytest
import torch
from scripts.extreme_run import epoch_weights, balanced_ce


def test_global_not_minibatch_balance():
    batches=[[(True,1),(False,3)],[(False,6),(False,4)]]
    weights=epoch_weights(batches)
    assert weights[True]*0.25==pytest.approx(1.)
    assert weights[False]*1.75==pytest.approx(1.)


def test_unit_weights_match_masked_next_token_loss():
    logits=torch.randn(2,4,9,requires_grad=True); labels=torch.tensor([[-100,-100,3,4],[-100,5,-100,-100]])
    expected=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,9),labels[:,1:].reshape(-1))
    got=balanced_ce(logits,labels,[1.,1.]);torch.testing.assert_close(got,expected);got.backward();assert logits.grad.isfinite().all()
