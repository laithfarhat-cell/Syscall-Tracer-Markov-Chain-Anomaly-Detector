from __future__ import annotations

import json
import math
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from core.detector import CRITICAL, DRIFT, AlertWriter, WindowDetector
from core.markov_model import UNK, MarkovModel
from core.tracer import SyscallEvent, read_trace
from scripts.simulate_behavior import ANOMALY, workload, write_lab


def events(tokens, *, pid=None, source=None):
    return [SyscallEvent(token, pid, 1_750_000_000 + i, source) for i, token in enumerate(tokens)]


@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("alpha", [1e-100, 0.01, 1.0, 1e100])
def test_smoothed_matrix_is_positive_and_row_stochastic(order, alpha):
    model = MarkovModel(order=order, alpha=alpha).fit(
        [
            ["read", "write", "close", "read", "write", "exit"],
            ["openat:READ", "read", "write", "read", "write", "close"],
        ]
    )
    matrix = model.transition_matrix([*model.contexts(), (UNK,) * order])
    assert UNK in model.vocabulary
    assert (matrix > 0).all()
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(matrix[-1], 1 / len(model.vocabulary))
    assert math.isfinite(model.window_score(["new_call"] * (order + 1)))


def test_exact_counts_smoothing_and_unknown_mapping():
    model = MarkovModel(alpha=0.25).fit([["read", "write"], ["read", "write"], ["read", "close"]])
    # Four destinations: close, read, write and <UNK>; sum of counts in read's row is 3.
    assert model.probability("read", "write") == pytest.approx(2.25 / 4)
    assert model.probability("read", "close") == pytest.approx(1.25 / 4)
    unseen = model.transition("read", "new_call")
    assert unseen.uses_unknown and unseen.observed_count == 0
    assert unseen.probability == pytest.approx(0.25 / 4)
    assert model.probability("new_call", "read") == 1 / 4
    assert model.transition("write", "read").observed_count == 0  # no stitching independent captures
    col = model.transition_matrix([("read",)], ["write"])
    assert col.shape == (1, 1)
    assert col[0, 0] == pytest.approx(2.25 / 4)  # display subsets are not renormalized


def test_second_order_distinguishes_context_without_window_backoff():
    model = MarkovModel(order=2).fit([["accept4", "read", "close"], ["openat:READ", "read", "write"]])
    assert model.probability(("accept4", "read"), "close") > 0.9
    assert model.probability(("openat:READ", "read"), "close") < 0.1
    assert model.probability(("close", "read"), "close") == pytest.approx(1 / len(model.vocabulary))
    assert model.transition("read", "close").observed_count == 1  # suffix only for warm-up
    tokens = ["accept4", "read", "close", "openat:READ", "read", "write"]
    reference = -sum(math.log2(model.probability(tokens[i - 2 : i], tokens[i])) for i in range(2, len(tokens))) / 4
    assert model.window_score(tokens) == pytest.approx(reference)


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), True, "0.01"])
def test_invalid_smoothing_is_rejected(bad):
    with pytest.raises(ValueError, match="alpha"):
        MarkovModel(alpha=bad)


def test_incremental_training_is_atomic_and_invalidates_calibration(tmp_path):
    initial = ["read", "write", "close"] * 10
    extra = ["read", "write", "poll"] * 10
    model = MarkovModel(order=2).fit([initial])
    model.calibrate([events(initial)], window=8)
    detector = WindowDetector(model)
    initial_profile = model.to_dict()
    with pytest.raises(ValueError, match="invalid training token"):
        model.partial_fit([extra, ["<UNK>"]])
    assert model.to_dict() == initial_profile
    model.partial_fit([extra])
    assert model.calibration is None and "poll" in model.vocabulary
    assert model.to_dict() == MarkovModel(order=2).fit([initial, extra]).to_dict()
    with pytest.raises(ValueError, match="profile changed"):
        detector.observe(SyscallEvent("read"))
    model.calibrate([events(extra)], window=8)
    output = tmp_path / "model.json"
    model.save(output)
    recovered = MarkovModel.load(output)
    assert recovered.to_dict() == model.to_dict()
    np.testing.assert_array_equal(recovered.transition_matrix(), model.transition_matrix())
    other = tmp_path / "model-copy.json"
    recovered.save(other)
    assert output.read_bytes() == other.read_bytes()


def test_profile_validation(tmp_path):
    model = MarkovModel().fit([["read", "write"]])
    path = tmp_path / "profile.json"
    payload = model.to_dict()
    payload["rows"][0]["next"]["write"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="transition count"):
        MarkovModel.load(path)
    payload["version"] = 999
    with pytest.raises(ValueError, match="version"):
        MarkovModel.from_dict(payload)


def test_interleaved_processes_and_hosts_do_not_create_edges():
    model = MarkovModel()
    model.partial_fit_events(
        [
            SyscallEvent("read", 10, source="a"),
            SyscallEvent("openat:READ", 20, source="a"),
            SyscallEvent("accept4", 10, source="b"),
            SyscallEvent("write", 10, source="a"),
            SyscallEvent("close", 20, source="a"),
            SyscallEvent("connect", 10, source="b"),
        ]
    )
    assert model.transition_count == 3
    assert model.transition("read", "write").observed_count == 1
    assert model.transition("accept4", "connect").observed_count == 1
    assert model.transition("read", "openat:READ").observed_count == 0
    model.partial_fit_events(events(["write", "fsync"], pid=20, source="a"))
    assert model.transition("close", "write").observed_count == 0
    with pytest.raises(ValueError, match="exceeds 1 streams"):
        model.partial_fit_events([SyscallEvent("read", 1), SyscallEvent("close", 2)], max_streams=1)


@pytest.mark.parametrize("order, window", [(1, 5), (2, 3), (4, 6)])
def test_rolling_monitor_matches_direct_math_for_interleaved_streams(order, window):
    good = ["read", "write", "close", "poll"] * 11
    model = MarkovModel(order=order).fit([good])
    detector = WindowDetector(model, threshold=1e3, window=window)
    sequences = defaultdict(list)
    a = events(good, pid=42)
    b = events(good[::-1], pid=41)
    for sample in detector.scan(event for pair in zip(a, b, strict=True) for event in pair):
        reference = sequences[sample.current.event.stream]
        reference.append(sample.current.event.token)
        if len(reference) < window:
            assert sample.score is None
        else:
            assert sample.score == pytest.approx(model.window_score(reference[-window:]), abs=1e-12)
        assert sample.alert is None


def test_calibration_uses_percentile_of_complete_held_out_windows():
    model = MarkovModel(order=2).fit([["read", "write", "close"] * 50])
    x = ["read", "write", "close", "read", "openat", "write", "close"]
    y = ["read", "write", "close", "read", "write", "close"]
    window = 4
    result = model.calibrate([events(x), events(y)], window=window, percentile=99)
    baseline = [model.window_score(seq[i : i + window]) for seq in (x, y) for i in range(len(seq) - window + 1)]
    assert result.threshold == pytest.approx(np.percentile(baseline, 99))
    assert result.median == pytest.approx(np.median(baseline))
    assert result.samples == len(baseline)
    assert "openat" not in model.vocabulary  # validation doesn't become training
    with pytest.raises(ValueError, match="no complete validation windows"):
        model.calibrate([events(x[:3]), events(x[3:6])], window=window)
    with pytest.raises(ValueError, match="calibration for this window"):
        WindowDetector(model, window=window + 1)
    assert WindowDetector(model).window == window


def test_calibration_and_scoring_do_not_raise_drift_on_repeated_ties():
    trace = ["read", "write", "close"] * 4000
    model = MarkovModel().fit([trace[:30]])
    model.calibrate([events(trace[:120])], window=19)
    detector = WindowDetector(model)
    assert all(sample.alert is None for sample in detector.scan(events(trace)))


def test_critical_rule_when_unseen_probability_is_not_near_zero():
    model = MarkovModel(order=3, alpha=1).fit([["read", "write", "close"] * 5])
    detector = WindowDetector(model, window=5, threshold=100, critical_probability=0.001)
    assert detector.observe(SyscallEvent("read")).alert is None
    sample = detector.observe(SyscallEvent("mprotect:EXEC"))
    assert sample.transition.probability > 0.001
    assert sample.transition.observed_count == 0
    assert sample.score is None
    assert sample.alert.triggers == (CRITICAL,)
    assert sample.alert.as_dict()["syscall_hids"]["window"]["complete"] is False


def test_critical_rule_can_flag_an_observed_rare_edge():
    model = MarkovModel().fit([["read", "write"]] * 2000 + [["read", "execve"]])
    detector = WindowDetector(model, window=5, threshold=100, critical_probability=0.001)
    sample = list(detector.scan(events(["read", "execve"])))[-1]
    assert 0 < sample.transition.probability < 0.001
    assert sample.transition.observed_count == 1
    assert sample.alert.triggers == (CRITICAL,)
    normal = WindowDetector(model, window=5, threshold=100, critical_probability=0.00001)
    assert list(normal.scan(events(["read", "execve"])))[-1].alert is None


@pytest.mark.parametrize("pair", [("connect", "dup2"), ("dup3", "connect")])
def test_critical_pair_span_and_stream_isolation(pair):
    model = MarkovModel().fit([["read", "write"] * 15])
    opts = dict(window=5, threshold=100, pair_span=3)
    result = list(WindowDetector(model, **opts).scan(events([pair[0], "read", pair[1]])))[-1]
    assert result.alert.critical_features == ("connect+dup2/dup3",)
    separated = WindowDetector(model, **opts)
    for event in [SyscallEvent(pair[0], pid=1), SyscallEvent("read", pid=2), SyscallEvent(pair[1], pid=2)]:
        assert separated.observe(event).alert is None
    far = list(WindowDetector(model, **opts).scan(events([pair[0], "read", "write", "read", pair[1]])))[-1]
    assert far.alert is None


def test_drift_only_and_exact_full_window_evidence(tmp_path):
    model = MarkovModel(order=2).fit([["read", "write", "close"] * 4])
    detector = WindowDetector(model, window=4, threshold=0.5, critical_tokens=[])
    a = events(["read", "write", "close", "poll", "read"], pid=9, source="node")
    interleaved = [a[0], SyscallEvent("accept4", pid=8), *a[1:]]
    observation = list(detector.scan(interleaved))[-1]
    alert = observation.alert
    assert alert.triggers == (DRIFT,)
    assert [step.input_index for step in alert.events] == [3, 4, 5, 6]
    assert [step.destination_index for step in alert.transitions] == [5, 6]
    assert alert.score == pytest.approx(sum(edge.transition.surprise_bits for edge in alert.transitions) / 2)
    assert alert.as_dict()["@timestamp"].endswith("Z")
    assert alert.as_dict()["process"]["pid"] == 9
    for path in (tmp_path / "alerts.json", tmp_path / "alerts.jsonl"):
        with AlertWriter(path) as writer:
            writer.write(alert)
            if path.suffix == ".jsonl":
                assert json.loads(path.read_text(encoding="utf-8")) == alert.as_dict()
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert (saved[0] if path.suffix == ".json" else saved) == alert.as_dict()
    with AlertWriter(tmp_path / "no-alerts.json"):
        pass
    assert json.loads((tmp_path / "no-alerts.json").read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("order", [1, 2, 4])
def test_inert_lab_detects_labeled_segment_at_low_benign_alert_rate(tmp_path, order):
    lab = write_lab(tmp_path, requests=100)
    model = MarkovModel(order=order)
    for path in lab.training:
        model.partial_fit_events(read_trace(path))
    model.calibrate([read_trace(path) for path in lab.validation], window=20)
    with closing(read_trace(lab.scenario)) as trace:
        observations = list(WindowDetector(model).scan(trace))
    assert tuple(o.current.event.token for o in observations[lab.first_unusual - 1 : lab.last_unusual]) == ANOMALY
    in_segment = [o for o in observations if lab.first_unusual <= o.current.input_index <= lab.last_unusual]
    assert any(o.alert and CRITICAL in o.alert.triggers for o in in_segment)
    assert any(o.alert and DRIFT in o.alert.triggers for o in in_segment)
    baseline = list(WindowDetector(model).scan(workload(1001, 140)))
    assert not any(o.alert and CRITICAL in o.alert.triggers for o in baseline)
    assert sum(o.alert is not None for o in baseline) / sum(o.score is not None for o in baseline) < 0.06


def test_monitor_outputs_are_deterministic(tmp_path: Path):
    lab = write_lab(tmp_path, requests=50)
    model = MarkovModel().fit([[e.token for e in read_trace(path)] for path in lab.training])
    model.calibrate([read_trace(p) for p in lab.validation], window=20)

    def run():
        return [
            sample.alert.as_dict() for sample in WindowDetector(model).scan(read_trace(lab.scenario)) if sample.alert
        ]

    assert run() == run()
