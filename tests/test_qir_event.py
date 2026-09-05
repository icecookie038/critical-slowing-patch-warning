import numpy as np
import pytest
import torch
from src.qir_event.events import RasterSeries, compile_events, sample_history, split_source_blocks
from src.qir_event.model import QIREventLite, event_probability, interval_nll
from src.qir_event.validation import block_weights, SupportDiagnostic

torch.set_num_threads(2)


def series(values, valid=None):
    x = np.array(values, dtype=float)[:, None, None]
    mask = np.ones_like(x, dtype=bool) if valid is None else np.array(valid, bool)[:, None, None]
    return RasterSeries(x, mask, np.array([0., 1., 3., 6., 10., 15.])[:len(x)],
                        np.ones_like(x), "A", 1.)


def test_recovery_and_repeated_terminal_loss():
    rows, ledger = compile_events(series([1, 0, 0, 1, 0]))
    assert rows.event.tolist() == [0, 1]
    assert rows.delta_t.tolist() == [2, 3]
    assert rows.waiting_time.tolist() == [0, 2]
    assert ledger.recovered.tolist() == [True, False]
    assert ledger.exit_time.tolist() == [6, 10]


def test_right_censoring_is_not_dropped():
    rows, ledger = compile_events(series([1, 0, 0, 0]))
    assert len(rows) == 2 and rows.event.sum() == 0
    assert not ledger.recovered.any()


def test_gap_censors_without_fabricated_recovery():
    rows, ledger = compile_events(series([1, 0, 0, 1], [1, 1, 0, 1]))
    assert rows.empty
    assert ledger.exit_time.tolist() == [1.]
    assert ledger.exit_reason.tolist() == ["first_missing_endpoint"]


def test_nonbinary_observed_and_duplicate_dates_rejected():
    s = series([1, .5, 0])
    with pytest.raises(ValueError):
        compile_events(s)
    s = series([1, 0, 1]); s.times[2] = s.times[1]
    with pytest.raises(ValueError):
        compile_events(s)


def test_missing_values_not_zero_state_and_no_duplicate_pixels():
    s = series([1, 0, np.nan], [1, 1, 0])
    assert len(compile_events(s)[1]) == 1
    with pytest.raises(ValueError):
        compile_events(s, [[0, 0], [0, 0]])


def test_history_excludes_endpoint_and_future():
    s = series([1, 0, 0, 1]); rows, _ = compile_events(s)
    before = sample_history(s, rows.iloc[0], (1.,), 2)
    s.states[2:] = 999
    after = sample_history(s, rows.iloc[0], (1.,), 2)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    assert before[0].shape[0] == 2


def test_exposure_and_censor_likelihood_match_exponential():
    eta = torch.zeros(2, dtype=torch.float64)
    dt = torch.tensor([1., 3.], dtype=torch.float64)
    p = event_probability(eta, dt)
    np.testing.assert_allclose(p.numpy(), 1-np.exp(-dt.numpy()))
    assert interval_nll(eta, dt, torch.zeros(2)).item() == 2.
    # Interval partition invariance when hazard remains fixed.
    assert torch.allclose((1-p[0])**3, 1-p[1])


def test_extreme_hazards_have_finite_gradients():
    eta = torch.tensor([-100., -10., 0., 10., 100.], requires_grad=True)
    loss = interval_nll(eta, torch.ones(5), torch.tensor([1., 0., 1., 0., 1.]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(eta.grad).all()


def test_causal_model_and_mask_invariance():
    torch.manual_seed(11)
    net = QIREventLite(n_scales=2, hidden=8).eval()
    torch.nn.init.normal_(net.residual.weight)
    x = torch.rand(2, 4, 2, 8, 8)
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[..., 0, :] = False
    times = torch.tensor([[0., 1., 3., 5.]]).repeat(2, 1)
    length = torch.tensor([2, 3]); bx = torch.zeros(2, 2)
    with torch.no_grad():
        a = net(x, valid, times, length, bx)
        x[0, 2:] = 1000.; x[1, 3:] = -1000.; x[..., 0, :] = float("nan")
        b = net(x, valid, times, length, bx)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_block_splits_and_equal_total_weights():
    import pandas as pd
    rows = pd.DataFrame({"domain": ["A"]*8+["B"]*2,
                         "block_id": ["A0","A0","A1","A2","A2","A3","A4","A5","B0","B0"]})
    split = split_source_blocks(rows, "A", "B")
    tr = set(rows.iloc[split["train"]].block_id)
    ca = set(rows.iloc[split["calibration"]].block_id)
    assert not tr & ca
    assert set(rows.iloc[split["test"]].domain) == {"B"}
    weights = block_weights(["x", "x", "y"])
    assert weights[0]+weights[1] == weights[2]


def test_support_flags_extreme_extrapolation():
    x = np.linspace(0, 1, 50)[:, None]
    support = SupportDiagnostic().fit(x, x[::5])
    assert not support.accepted(np.array([[100.]]))[0]


def test_support_constant_quality_dimension_has_natural_scale():
    x = np.c_[np.linspace(0, 1, 50), np.ones(50)]
    cal = x[::5].copy(); cal[:, 1] = .5
    diagnostic = SupportDiagnostic().fit(x, cal)
    assert np.isfinite(diagnostic.threshold) and diagnostic.threshold < 10


def test_quality_baseline_cannot_read_image_content():
    torch.manual_seed(9)
    net = QIREventLite(n_scales=1, hidden=8, mode="quality").eval()
    x = torch.rand(2, 2, 1, 4, 4); valid = torch.ones_like(x, dtype=torch.bool)
    times = torch.tensor([[0., 2.], [0., 3.]])
    lengths = torch.tensor([2, 2]); bx = torch.zeros(2, 2)
    with torch.no_grad():
        a = net(x, valid, times, lengths, bx)
        b = net(1-x, valid, times, lengths, bx)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_buffered_patches_stay_in_assigned_spatial_block():
    from src.qir_event.experiment import buffered_coordinates, synthetic_series
    s = synthetic_series(3, "A")
    coords = buffered_coordinates(s, 256, 128, 5, 4)
    for r, c in coords:
        block_r, block_c = r//32, c//32
        assert (r-8)//32 == (r+8)//32 == block_r
        assert (c-8)//32 == (c+8)//32 == block_c


def test_invalid_exposure_rejected():
    with pytest.raises(ValueError):
        event_probability(torch.zeros(1), torch.zeros(1))
