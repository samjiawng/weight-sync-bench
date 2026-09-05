"""The null-transport control, and the artifact that records the gate's resolution.

The question behind all of this: SPEC 2e says a timing measurement of a sync that
delivered wrong weights is worthless, which is only true if a sync that delivered
NOTHING fails the correctness gate. That was assumed and is now measured, so what
these tests protect is the measurement, not a threshold.

The load-bearing one is
`test_a_correctly_delivered_stale_sync_passes_where_the_wrong_reference_fails_it`.
Deliberately stale weights and a broken transport are the same bytes on the
sampler; only the choice of reference tells them apart, and getting that backwards
produces a measurement that looks fine and is wrong.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from weight_sync_bench.model import ReferenceModel
from weight_sync_bench.phase2 import gate_resolution as gr
from weight_sync_bench.phase2.param_source import ShardedParamSource
from weight_sync_bench.tolerance import BATCH, MIN_SEPARATION, SEQ_LEN, load_threshold

THRESHOLD = load_threshold()
GATE = MIN_SEPARATION * THRESHOLD


@pytest.fixture
def single_threaded():
    """The measurement pins threads and restores them; a test running the same
    computation has to pin them too, or it compares numbers taken under a
    different reduction order. Restored so nothing else in the session inherits
    it."""
    previous = torch.get_num_threads()
    torch.set_num_threads(gr.MEASUREMENT_THREADS)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def report():
    return gr.load()


# --- the loader, in the shape load_threshold sets ---------------------------


def test_a_missing_artifact_names_the_command_that_rebuilds_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="gate_resolution"):
        gr.load(tmp_path / "absent.json")


def test_the_recorded_geometry_and_token_shape_match_the_running_ones(report):
    """FAILS rather than warns, exactly as phase 1a's geometry check does. These
    inputs are under the repo's control and they move the numbers: the deviations
    are maxima over batch * seq_len * vocab elements, so raising seq_len in a
    fixture would silently change what the gate resolves."""
    assert gr.geometry_mismatches(report) == []
    assert report["sweep"]["batch"] == BATCH
    assert report["sweep"]["seq_len"] == SEQ_LEN


def test_the_threshold_is_read_from_phase_1a_and_not_re_derived(report):
    assert report["gate"]["threshold"] == THRESHOLD
    assert report["gate"]["min_separation"] == MIN_SEPARATION
    assert report["gate"]["fires_above"] == pytest.approx(GATE)
    assert "phase1a" in report["gate"]["threshold_source"]
    assert report["gate"]["statistic"] == "max absolute logit error"


def test_the_reference_magnitude_is_marked_extrapolated(report):
    """It is arithmetic about Adam, not a measurement, and an artifact that let it
    read as measured would be claiming a training run that never happened."""
    assert report["reference_magnitude"]["extrapolated"] is True
    assert report["reference_magnitude"]["value"] == gr.REFERENCE_MAGNITUDE


def test_detectable_k_is_read_from_the_artifact_not_computed(report):
    assert gr.detectable_k("kv2", 2, gr.REFERENCE_MAGNITUDE, report) == next(
        row["smallest_detectable_k"]
        for row in report["resolution"]
        if row["config"] == "kv2"
        and row["dst_degree"] == 2
        and row["magnitude"] == gr.REFERENCE_MAGNITUDE
    )
    with pytest.raises(KeyError, match="no recorded resolution"):
        gr.detectable_k("kv2", 2, 12345.0, report)


# --- the correct-sync leg, which is what makes the null result attributable ---


def test_every_recorded_cell_had_a_passing_correct_sync(report):
    """A cell whose correct leg failed would mean a broken split, and the null
    deviation beside it would not be attributable to the missed update. Such
    cells are excluded rather than averaged, so an empty exclusion list is the
    statement that the whole grid was attributable."""
    assert report["cells"]
    assert all(row["correct_sync_passes"] for row in report["cells"])
    assert all(
        row["correct_max_deviation"] < THRESHOLD for row in report["cells"]
    )
    assert report["excluded_cells"] == []


# --- the null-transport control ----------------------------------------------


def test_a_sync_that_delivered_nothing_fails_the_invariant(single_threaded, report):
    """THE CONTROL 2e ASSUMES. The cell comes from the artifact, so a
    re-measurement that moves the boundary moves this test with it rather than
    leaving a constant that happens to still pass.

    The null transport is a local idea, not a `Transport` implementation:
    delivering nothing means the sampler still holds `params_0`, which is what
    `measure_cell` compares. Widening the protocol to take a sharded source is
    2d's design and must not be settled here by a test fixture.
    """
    magnitude, k = gr.smallest_detectable_cell("kv2", 2, report)
    cell = gr.measure_cell("kv2", 2, magnitude, k)

    assert cell["null_max_deviation"] > GATE
    assert cell["null_max_deviation"] > MIN_SEPARATION * THRESHOLD
    assert cell["gate_fires"] is True
    # The same cell's correct sync passes, so the null failure is the missed
    # update and not a broken sharded path.
    assert cell["correct_max_deviation"] < THRESHOLD
    assert cell["correct_sync_passes"] is True


def test_measuring_one_cell_from_scratch_reproduces_the_recorded_sweep(
    single_threaded, report
):
    """The sweep steps one source incrementally across k; this rebuilds a single
    cell from the seeds. Their agreeing is what makes "the loop does not leak
    state between cells" a check rather than an assumption."""
    magnitude, k = gr.smallest_detectable_cell("kv2", 2, report)
    recorded = next(
        row
        for row in report["cells"]
        if row["config"] == "kv2"
        and row["dst_degree"] == 2
        and row["magnitude"] == magnitude
        and row["skipped_steps"] == k
    )
    live = gr.measure_cell("kv2", 2, magnitude, k)

    assert live["null_max_deviation"] == pytest.approx(
        recorded["null_max_deviation"], rel=1e-9
    )
    assert live["correct_max_deviation"] == pytest.approx(
        recorded["correct_max_deviation"], rel=1e-9
    )


# --- the reference, which is the part that is easy to get backwards ----------


def test_a_correctly_delivered_stale_sync_passes_where_the_wrong_reference_fails_it(
    single_threaded, report
):
    """Phase 3 varies staleness on purpose and needs the gate quiet on it.

    A sync at step k-1 delivered exactly what it was supposed to deliver, and the
    source then moved on. Measured against the state that sync was INTENDED to
    deliver, the sampler passes -- correct, merely stale. Measured against the
    source's CURRENT state, the same correct sampler fails. Both numbers are
    computed here, because "we chose the right reference" is otherwise a claim
    about code nobody re-derives.
    """
    magnitude, k = gr.smallest_detectable_cell("kv2", 2, report)
    assert k >= 2, "this contrast needs a delivered state before the current one"

    config = gr.CONFIGS["kv2"]
    tokens = gr._tokens(config)
    model = ReferenceModel(config, seed=gr.MODEL_SEED)
    params_0 = {n: t.detach().clone() for n, t in model.full_params().items()}

    source = ShardedParamSource(
        params_0, gr._source_layouts(config)[0][1], seed=gr.SOURCE_SEED
    )
    for _ in range(k - 1):
        source.step(magnitude)
    delivered = source.intended_state()
    source.step(magnitude)
    current = source.intended_state()

    sampler = gr._sharded_logits(config, delivered, 2, tokens)

    model.load_state_dict(delivered)
    against_intended = gr._deviation(model(tokens), sampler)
    model.load_state_dict(current)
    against_current = gr._deviation(model(tokens), sampler)

    assert against_intended < THRESHOLD, (
        "a sync that delivered exactly what it was asked to deliver must pass, "
        "however stale the source has since become"
    )
    assert against_current > THRESHOLD, (
        "using the source's current state as the reference fails a correctly "
        "delivered sampler, which is the mistake this measurement avoids"
    )
    assert against_current > against_intended


# --- what the run actually found ---------------------------------------------


def test_the_artifact_states_which_of_the_two_outcomes_it_got(report):
    """Both are results. The artifact must commit to one in a field a reader
    consults, rather than leaving it to be inferred from the cells."""
    assert report["outcome"] in (
        "gate_resolves_a_single_missed_sync",
        "gate_resolves_k_or_more_consecutive_missed_syncs",
    )
    at_reference = report["reference_magnitude"]["smallest_detectable_k"]
    fires_at_one = all(row["smallest_detectable_k"] == 1 for row in at_reference)
    assert (report["outcome"] == "gate_resolves_a_single_missed_sync") == fires_at_one
    assert report["consequence_for_2e"]


def test_the_source_layout_control_shows_the_intended_state_is_layout_independent(
    report,
):
    """Recorded rather than assumed. Identical digests across a storage source at
    degree 4 -- where kv2 has no execution table at all -- a storage source at 2
    and an execution source at 2."""
    rows = report["source_layout_control"]["rows"]
    assert {row["source_layout"] for row in rows} == {
        "storage@t4",
        "storage@t2",
        "execution@t2",
    }
    for config in gr.CONFIGS:
        digests = {
            row["intended_state_digest"] for row in rows if row["config"] == config
        }
        deviations = {
            row["null_max_deviation"] for row in rows if row["config"] == config
        }
        assert len(digests) == 1, config
        assert len(deviations) == 1, config
