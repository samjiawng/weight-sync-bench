"""The composed worker extension, at every weight-broadcast transport.

A worker extension is named by exactly one qualname, which resolves to exactly
one class, which is built on exactly one of prime-rl's transport workers. So
there is one composed class per transport, and the name, the key it is bound
under, and the class it resolves to all have to agree about which transport
that is. They did not: the bind site and the probe's reporting both used the
bare name, which resolves to the filesystem composition whatever the transport.

That failure is invisible at bind time. It surfaces inside prime-rl's own
weight-update path as an arity error naming `init_broadcaster`, on the NCCL
transport that `rl` auto-selects -- so the default was exactly wrong for the
entry point the RL loop uses. These tests pin the agreement at BOTH transports,
because a single-transport assertion is what let it through.

The name-shape assertions need neither prime-rl nor vLLM. The ones that build
or bind a class need prime-rl importable and are skipped without it, the same
way the vLLM field-name test in test_phase3_flag_rule.py is.
"""

from __future__ import annotations

import pytest

from weight_sync_bench.phase3 import attach
from weight_sync_bench.phase3 import engine_probe

TRANSPORTS = sorted(engine_probe.PRIME_RL_WORKER_EXTENSIONS)


def test_more_than_one_transport_exists_to_distinguish():
    """Everything below is vacuous with a single transport, and the defect this
    file guards is only reachable because there are two."""
    assert len(TRANSPORTS) >= 2
    assert engine_probe.DEFAULT_BROADCAST_TYPE in TRANSPORTS


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_each_transport_has_its_own_qualname(transport):
    qualname = engine_probe.composed_worker_qualname(transport)
    module, _, name = qualname.rpartition(".")
    assert module == engine_probe.__name__
    assert name == f"{engine_probe.COMPOSED_WORKER_NAME}{transport.capitalize()}"


def test_the_transports_do_not_share_a_qualname():
    names = {engine_probe.composed_worker_qualname(t) for t in TRANSPORTS}
    assert len(names) == len(TRANSPORTS)


def test_no_transport_reuses_the_bare_name():
    """The bare name keeps meaning the filesystem composition for the standalone
    server path. If a transport's name collided with it, binding that transport
    would silently install the default composition -- the original defect."""
    bare = engine_probe.COMPOSED_WORKER_QUALNAME
    for transport in TRANSPORTS:
        assert engine_probe.composed_worker_qualname(transport) != bare


def test_unknown_transport_is_rejected_by_name_lookup_too():
    with pytest.raises(ValueError, match="unknown weight-broadcast transport"):
        engine_probe.composed_worker_qualname("carrier-pigeon")


# --- needs prime-rl importable ---------------------------------------------


@pytest.fixture(scope="module")
def prime_rl():
    return pytest.importorskip(
        "prime_rl.inference.vllm.server",
        reason="prime-rl is installed on the GPU box only",
    )


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_composed_class_carries_both_parents(transport, prime_rl):
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from weight_sync_bench.phase2.collective_logits import LogitsHookWorkerExtension

    composed = engine_probe.compose_worker_extension(transport)
    base = resolve_obj_by_qualname(engine_probe.PRIME_RL_WORKER_EXTENSIONS[transport])
    assert base in composed.__mro__
    assert LogitsHookWorkerExtension in composed.__mro__


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_the_qualname_resolves_to_the_class_it_names(transport, prime_rl):
    """The qualname is the only thing that travels to a spawned worker, so the
    name resolving to the right class is the whole attachment."""
    from vllm.utils.import_utils import resolve_obj_by_qualname

    qualname = engine_probe.composed_worker_qualname(transport)
    assert resolve_obj_by_qualname(qualname) is engine_probe.compose_worker_extension(
        transport
    )


def test_the_transports_resolve_to_distinct_classes(prime_rl):
    classes = [engine_probe.compose_worker_extension(t) for t in TRANSPORTS]
    assert len({id(c) for c in classes}) == len(TRANSPORTS)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_bind_writes_the_transports_own_qualname(transport, prime_rl, monkeypatch):
    """The key and the value have to agree about the transport. Binding the bare
    name under the NCCL key is the defect: it installs the filesystem
    composition on a run that will call `init_broadcaster` with NCCL's
    signature."""
    monkeypatch.setattr(
        prime_rl, "WORKER_EXTENSION_CLS", dict(prime_rl.WORKER_EXTENSION_CLS)
    )
    returned = attach.bind_worker_extension(transport)
    expected = engine_probe.composed_worker_qualname(transport)
    assert returned == expected
    assert prime_rl.WORKER_EXTENSION_CLS[transport] == expected


def test_binding_one_transport_leaves_the_other_alone(prime_rl, monkeypatch):
    monkeypatch.setattr(
        prime_rl, "WORKER_EXTENSION_CLS", dict(prime_rl.WORKER_EXTENSION_CLS)
    )
    other = [t for t in TRANSPORTS if t != TRANSPORTS[0]][0]
    before = prime_rl.WORKER_EXTENSION_CLS[other]
    attach.bind_worker_extension(TRANSPORTS[0])
    assert prime_rl.WORKER_EXTENSION_CLS[other] == before


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_check_composition_reports_the_transport_it_composed(transport, prime_rl):
    """The committed attachment artifact recorded the bare name beside an MRO
    ending in Filesystem. A name that does not match the class beside it makes
    the record unreadable, so the name is asserted against the MRO here."""
    from weight_sync_bench.phase3 import attachment_probe

    result = attachment_probe.check_composition(transport)
    assert result["broadcast_type"] == transport
    assert result["qualname"] == engine_probe.composed_worker_qualname(transport)
    assert result["resolved_by_qualname"]
    assert result["mro"][0] == result["qualname"]
    assert result["passed"]
