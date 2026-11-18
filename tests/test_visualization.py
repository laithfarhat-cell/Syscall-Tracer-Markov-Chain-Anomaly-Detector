from core.detector import WindowDetector
from core.markov_model import MarkovModel
from core.tracer import SyscallEvent
from visualizers.graph_plotter import Diagnostics


def test_timeline_preserves_peak_and_bounds_plot_storage():
    model = MarkovModel().fit([["read", "write"] * 20])
    detector = WindowDetector(model, window=4, threshold=0.2)
    recorder = Diagnostics(sample_limit=20, alert_limit=3)
    samples = list(
        detector.scan(
            SyscallEvent(token) for token in ["read", "write"] * 200 + ["mprotect:EXEC"] + ["read", "write"] * 200
        )
    )
    for sample in samples:
        recorder.record(sample)
    assert len(recorder.points) <= recorder.sample_limit
    assert len(recorder.markers) == recorder.alert_limit
    assert recorder.stride > 1 and not recorder.clock_ok
    assert max(p.score for p in recorder.points) == max(s.score for s in samples if s.score is not None)
    assert any((("write",), "mprotect:EXEC") in marker.edges for marker in recorder.markers)
