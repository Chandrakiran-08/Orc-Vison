"""Memory tests — bounded working memory, decaying long-term memory."""

from orcvision.brain.memory import KIND_EVENT, KIND_OUTCOME, LongTermMemory, WorkingMemory


def test_working_memory_is_bounded():
    wm = WorkingMemory(capacity=5, retention_s=1000)
    for i in range(50):
        wm.add(KIND_EVENT, f"event {i}", float(i))
    assert len(wm) == 5  # ring buffer, footprint stays flat


def test_working_memory_retention_window():
    wm = WorkingMemory(capacity=100, retention_s=10.0)
    wm.add(KIND_EVENT, "old", 0.0)
    wm.add(KIND_EVENT, "new", 20.0)
    wm.prune(now=20.0)
    assert [i.content for i in wm.recent()] == ["new"]


def test_working_memory_recent_filters_and_orders():
    wm = WorkingMemory()
    wm.add(KIND_EVENT, "e1", 1.0)
    wm.add(KIND_OUTCOME, "o1", 2.0)
    wm.add(KIND_EVENT, "e2", 3.0)
    assert [i.content for i in wm.recent(kind=KIND_EVENT)] == ["e2", "e1"]  # newest first
    assert wm.count(KIND_OUTCOME) == 1


def test_longterm_deduplicates_by_key():
    ltm = LongTermMemory()
    for _ in range(5):
        ltm.remember("obstacle@center", {"seen": True}, now=0.0)
    assert len(ltm) == 1  # reinforced, not duplicated
    assert ltm.recall("obstacle@center", 0.0).hits == 5


def test_reinforcement_raises_importance():
    ltm = LongTermMemory()
    first = ltm.remember("k", "v", now=0.0, importance=0.3).importance
    for _ in range(4):
        ltm.remember("k", "v", now=0.0, importance=0.3)
    assert ltm.recall("k", 0.0).importance > first


def test_memory_decays_and_is_forgotten():
    ltm = LongTermMemory(half_life_s=10.0, forget_below=0.05)
    ltm.remember("fleeting", "v", now=0.0, importance=0.2)
    assert ltm.recall("fleeting", 0.0) is not None
    # Many half-lives later it is below the retention floor.
    assert ltm.recall("fleeting", 500.0) is None


def test_repeated_traces_outlive_one_off_traces():
    ltm = LongTermMemory(half_life_s=60.0)
    ltm.remember("once", "v", now=0.0, importance=0.5)
    for _ in range(8):
        ltm.remember("often", "v", now=0.0, importance=0.5)
    assert ltm.recall("often", 100.0).strength(100.0, 60.0) > ltm._traces["once"].strength(
        100.0, 60.0
    )


def test_capacity_evicts_weakest():
    ltm = LongTermMemory(capacity=3, half_life_s=1e9)
    ltm.remember("weak", "v", now=0.0, importance=0.1)
    for i in range(3):
        ltm.remember(f"strong{i}", "v", now=0.0, importance=0.9)
    assert len(ltm) == 3
    assert ltm.recall("weak", 0.0) is None


def test_snapshot_restore_roundtrip():
    ltm = LongTermMemory()
    ltm.remember("k", {"successes": 2, "failures": 1}, now=0.0, kind=KIND_OUTCOME)
    restored = LongTermMemory()
    restored.restore(ltm.snapshot())
    trace = restored.recall("k", 0.0)
    assert trace is not None and trace.content["successes"] == 2
