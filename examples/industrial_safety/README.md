# Industrial cell guard — fail-safe decisions with an audit trail

```bash
python examples/industrial_safety/demo.py
```

A camera watches a machine cell; the brain decides whether it may keep
running.

| # | Situation | Decision |
|---|-----------|----------|
| 1 | Cell clear | `MOVE` (run) |
| 2 | Person inside the keep-out zone | `STOP` — a software light curtain |
| 3 | Person present but outside the zone, at 4.5 m | `MOVE` — keeps running |
| 4 | Cell empty, then the camera feed dies | `STOP` — **because perception went quiet** |
| 5 | Guard door opens (interlock) | `EMERGENCY_STOP` |
| 6 | Incident review | replay the audit log |

## The three properties that make this deployable

**Fail safe, not fail silent (step 4).** This is the one most systems get
wrong. If the detector crashes or the network stalls, a brain with no
staleness check keeps acting on a frozen snapshot of a cell that has since
had someone walk into it. Here, perception going quiet is itself a safety
event: anything but a declared safe state is vetoed once the world model is
older than `max_perception_age_s`.

**Deterministic behaviour.** The policy is `frozen`, so the cell behaves
identically at commissioning and a year later. You cannot certify a machine
whose decision policy drifts in the field — train offline, freeze, deploy.
Feedback still updates memory for analysis, but never moves a weight.

**Selectivity (step 3).** A guard that trips on everything gets bypassed by
operators within a week, which is worse than no guard at all. Keep-out zones
are normalized `(x1, y1, x2, y2)` in 0..1, so the same zone definition
survives a camera swap to a different resolution.

## The audit trail

Every decision is appended to a JSON Lines file — action, weighted reasons,
what was vetoed, what was visible, and the platform state at the time:

```
t= 1.0  STOP            saw: person@center   vetoes: 0  interlock: ok
t= 7.5  STOP            saw: nothing         vetoes: 2  interlock: ok
t= 6.0  EMERGENCY_STOP  saw: nothing         vetoes: 0  interlock: TRIPPED
```

It rotates at a size cap so an unattended machine cannot fill its disk, and
a write failure is counted and swallowed rather than stopping the line.

## Honest status

Simulated detections only; this has never guarded a real machine.

**This is not a certified safety system and must not be the only thing
between a person and a hazard.** Functional safety for machinery (IEC 62061,
ISO 13849) requires rated components, redundancy and assessment that a
camera and a Python process do not provide. Use this as a supervisory or
diagnostic layer *behind* proper interlocks, light curtains and e-stops —
never in place of them.
