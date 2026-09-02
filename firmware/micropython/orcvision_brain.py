"""Orc-Vison autonomous brain — MicroPython port.

For boards that run MicroPython but not CPython: ESP32, ESP32-S3/C3,
Raspberry Pi Pico W (RP2040), and similar. The Arduino Uno R4 WiFi does not
run MicroPython usefully at 32 KB — use the C++ library in
``firmware/OrcVisionBrain/`` there instead.

This is a deliberate subset of ``orcvision.brain`` written against what
MicroPython actually ships. Compared to the CPython original it drops:

* ``dataclasses``   -> plain classes with ``__slots__``
* ``typing`` / ``Protocol`` / ``from __future__`` -> nothing (duck typing)
* ``pathlib``       -> ``open()``
* ``collections.deque(maxlen=)`` -> a hand-rolled ring buffer, because
  MicroPython's deque is limited and its ``maxlen`` semantics differ

The decision logic, default weights and safety constraints are identical, so
it makes the same choices as the host implementation.

Usage on-device::

    from orcvision_brain import VisionBrain
    brain = VisionBrain(goal="avoid_collision")
    brain.begin_frame(t)
    brain.observe("obstacle", 0.9, cx, cy, size, depth_m=1.2, track_id=1)
    brain.end_frame()
    d = brain.decide()
    print(d.action, d.score)
    brain.feedback(False)   # tell it that did not work
    brain.learn()

Memory: a configured brain is a few kilobytes. Call ``gc.collect()`` after
construction on very tight boards.
"""

from math import exp, log, sqrt

# --- action vocabulary ------------------------------------------------------
STOP = "STOP"
AVOID = "AVOID"
MOVE = "MOVE"
TRACK = "TRACK"
WAIT = "WAIT"
SIGNAL = "SIGNAL"
ACTIONS = (STOP, AVOID, MOVE, TRACK, WAIT, SIGNAL)

ZONE_LEFT = "left"
ZONE_CENTER = "center"
ZONE_RIGHT = "right"

MOTION_UNKNOWN = "unknown"
MOTION_STATIONARY = "stationary"
MOTION_MOVING = "moving"
MOTION_APPROACHING = "approaching"
MOTION_RECEDING = "receding"

# Features, matching the host implementation's names.
F_HAZARD_PROXIMITY = "hazard_proximity"
F_HAZARD_APPROACHING = "hazard_approaching"
F_HAZARD_PRESENT = "hazard_present"
F_PATH_CLEAR = "path_clear"
F_TARGET_PRESENT = "target_present"
F_BIAS = "bias"
F_MEM_SUCCESS = "mem_success"
F_MEM_FAILURE = "mem_failure"

# Same starting weights as orcvision/brain/decision.py.
DEFAULT_WEIGHTS = {
    "AVOID|" + F_HAZARD_PROXIMITY: 1.00,
    "AVOID|" + F_HAZARD_APPROACHING: 0.90,
    "AVOID|" + F_BIAS: 0.05,
    "STOP|" + F_HAZARD_PROXIMITY: 0.90,
    "STOP|" + F_HAZARD_APPROACHING: 0.80,
    "STOP|" + F_BIAS: 0.00,
    "MOVE|" + F_PATH_CLEAR: 0.70,
    "MOVE|" + F_BIAS: 0.10,
    "TRACK|" + F_TARGET_PRESENT: 0.60,
    "WAIT|" + F_BIAS: 0.15,
    "SIGNAL|" + F_HAZARD_PRESENT: 0.30,
}
MEMORY_WEIGHTS = {F_MEM_FAILURE: -1.50, F_MEM_SUCCESS: 0.50}

# Actions forbidden outright when a hazard is closer than this proximity.
VETO_ADVANCING = (MOVE,)


def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def zone_of(cx):
    if cx < 1.0 / 3.0:
        return ZONE_LEFT
    if cx > 2.0 / 3.0:
        return ZONE_RIGHT
    return ZONE_CENTER


def default_weights():
    w = dict(DEFAULT_WEIGHTS)
    for action in ACTIONS:
        for feature, value in MEMORY_WEIGHTS.items():
            key = action + "|" + feature
            if key not in w:
                w[key] = value
        bias = action + "|" + F_BIAS
        if bias not in w:
            w[bias] = 0.0
    return w


class ObjectState:
    __slots__ = (
        "label", "confidence", "cx", "cy", "size", "depth_m", "track_id",
        "zone", "motion", "approach_rate", "first_seen", "last_seen", "misses",
    )

    def __init__(self, label, confidence, cx, cy, size, depth_m, track_id, now):
        self.label = label
        self.confidence = confidence
        self.cx = cx
        self.cy = cy
        self.size = size
        self.depth_m = depth_m
        self.track_id = track_id
        self.zone = zone_of(cx)
        self.motion = MOTION_UNKNOWN
        self.approach_rate = 0.0
        self.first_seen = now
        self.last_seen = now
        self.misses = 0

    def proximity(self, max_range_m=5.0):
        if self.depth_m is not None and max_range_m > 0:
            return _clamp01(1.0 - (self.depth_m / max_range_m))
        return _clamp01(sqrt(self.size) * 2.0)


class RingBuffer:
    """Fixed-capacity FIFO — MicroPython's deque is too limited to rely on."""

    __slots__ = ("_items", "_cap")

    def __init__(self, capacity=24):
        self._items = []
        self._cap = capacity

    def add(self, item):
        self._items.append(item)
        while len(self._items) > self._cap:
            self._items.pop(0)

    def recent(self, limit=None):
        out = list(reversed(self._items))
        return out[:limit] if limit else out

    def __len__(self):
        return len(self._items)


class LongTermMemory:
    """Keyed traces with importance, decay, dedup and capacity eviction."""

    __slots__ = ("_traces", "capacity", "half_life_s", "forget_below")

    def __init__(self, capacity=24, half_life_s=600.0, forget_below=0.05):
        self._traces = {}
        self.capacity = capacity
        self.half_life_s = half_life_s
        self.forget_below = forget_below

    def _strength(self, t, now):
        if self.half_life_s <= 0:
            return t["importance"]
        elapsed = now - t["last_reinforced"]
        if elapsed < 0:
            elapsed = 0.0
        decay = exp(-elapsed * 0.6931472 / self.half_life_s)
        repetition = 1.0 + log(1.0 + max(0, t["hits"] - 1)) * 0.25
        return t["importance"] * decay * repetition

    def remember(self, key, content, now, importance=0.5):
        t = self._traces.get(key)
        if t is None:
            t = {
                "content": content,
                "importance": importance,
                "last_reinforced": now,
                "hits": 1,
            }
            self._traces[key] = t
        else:
            t["content"] = content
            t["hits"] += 1
            t["last_reinforced"] = now
            t["importance"] = min(1.0, t["importance"] + (1.0 - t["importance"]) * 0.3)
            if t["importance"] < importance:
                t["importance"] = importance
        self.forget(now)
        return t

    def recall(self, key, now):
        t = self._traces.get(key)
        if t is None:
            return None
        if self._strength(t, now) < self.forget_below:
            del self._traces[key]
            return None
        return t

    def forget(self, now):
        for key in [k for k, t in self._traces.items()
                    if self._strength(t, now) < self.forget_below]:
            del self._traces[key]
        overflow = len(self._traces) - self.capacity
        if overflow > 0:
            ranked = sorted(self._traces.items(), key=lambda kv: self._strength(kv[1], now))
            for key, _ in ranked[:overflow]:
                del self._traces[key]

    def snapshot(self):
        return self._traces

    def restore(self, data):
        self._traces = data


class Decision:
    __slots__ = ("action", "score", "reasons", "features", "situation",
                 "vetoed", "demoted", "safety_fallback", "avoid_direction")

    def __init__(self, action, score, situation):
        self.action = action
        self.score = score
        self.situation = situation
        self.reasons = []
        self.features = {}
        self.vetoed = []
        self.demoted = []
        self.safety_fallback = False
        self.avoid_direction = ZONE_RIGHT

    def explain(self):
        out = ["Decision: " + self.action]
        if self.safety_fallback:
            out.append("  ! all actions vetoed -> safe action")
        for text, value in sorted(self.reasons, key=lambda r: -abs(r[1])):
            out.append(f"  {text} ({value:+.2f})")
        for action, penalty in self.demoted:
            out.append(f"  - {action} previously failed here ({penalty:+.2f})")
        for _action, reason in self.vetoed:
            out.append("  x " + reason)
        return "\n".join(out)


class VisionBrain:
    """The decision loop, MicroPython edition."""

    def __init__(self, goal="idle", hazard_labels=None, target_labels=None,
                 max_range_m=5.0, working_capacity=24, longterm_capacity=24,
                 learning_rate=0.1, safe_action=STOP):
        self.goal = goal
        self.hazard_labels = hazard_labels or ("person", "obstacle", "vehicle", "car")
        self.target_labels = target_labels or ()
        self.max_range_m = max_range_m
        self.learning_rate = learning_rate
        self.safe_action = safe_action

        self.weights = default_weights()
        self.objects = {}
        self.working = RingBuffer(working_capacity)
        self.longterm = LongTermMemory(longterm_capacity)

        self.move_threshold = 0.02
        self.approach_threshold = 0.05
        self.size_growth_threshold = 0.12
        self.disappear_after_misses = 5
        self.match_radius = 0.15
        self.veto_proximity = 0.6
        self.min_confidence = 0.4

        self._now = 0.0
        self._updated_at = 0.0
        self._tick = 0
        self._pending = []
        self._any_approach = False
        self.last_decision = None
        self._outcome = None
        self.last_action = None

    # --- perception ---------------------------------------------------------

    def begin_frame(self, timestamp):
        self._now = timestamp
        self._pending = []
        self._any_approach = False

    def observe(self, label, confidence, cx, cy, size, depth_m=None, track_id=None):
        self._pending.append((label, confidence, _clamp01(cx), _clamp01(cy),
                              _clamp01(size), depth_m, track_id))

    def observe_pixels(self, label, confidence, x1, y1, x2, y2, w, h,
                       depth_m=None, track_id=None):
        if not w or not h:
            return
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        size = (abs(x2 - x1) * abs(y2 - y1)) / float(w * h)
        self.observe(label, confidence, cx, cy, size, depth_m, track_id)

    def end_frame(self):
        dt = (self._now - self._updated_at) if self._tick else 0.0
        if self._tick and dt <= 0:
            dt = 1e-6
        seen = set()

        for label, conf, cx, cy, size, depth, track in self._pending:
            key = self._associate(label, cx, cy, track)
            if key is None:
                key = f"{label}#{track}" if track is not None else \
                      f"{label}~{len(self.objects)}"
                self.objects[key] = ObjectState(label, conf, cx, cy, size, depth,
                                                track, self._now)
            else:
                self._update_matched(self.objects[key], conf, cx, cy, size, depth, dt)
            seen.add(key)

        for key in list(self.objects.keys()):
            if key in seen:
                continue
            obj = self.objects[key]
            obj.misses += 1
            if obj.misses >= self.disappear_after_misses:
                del self.objects[key]

        self._updated_at = self._now
        self._tick += 1
        self.longterm.forget(self._now)

    def _associate(self, label, cx, cy, track):
        if track is not None:
            key = f"{label}#{track}"
            if key in self.objects:
                return key
            return None
        best, best_dist = None, self.match_radius
        for key, obj in self.objects.items():
            if obj.label != label:
                continue
            dist = sqrt((obj.cx - cx) ** 2 + (obj.cy - cy) ** 2)
            if dist < best_dist:
                best, best_dist = key, dist
        return best

    def _update_matched(self, obj, conf, cx, cy, size, depth, dt):
        displacement = sqrt((cx - obj.cx) ** 2 + (cy - obj.cy) ** 2)
        rate = 0.0
        if obj.depth_m is not None and depth is not None and dt > 0:
            rate = (obj.depth_m - depth) / dt
        growth = (size - obj.size) / obj.size if obj.size > 0 else 0.0

        approaching = rate > self.approach_threshold or (
            depth is None and growth > self.size_growth_threshold)
        receding = rate < -self.approach_threshold or (
            depth is None and growth < -self.size_growth_threshold)

        if approaching:
            obj.motion = MOTION_APPROACHING
            self._any_approach = True
        elif receding:
            obj.motion = MOTION_RECEDING
        elif displacement > self.move_threshold:
            obj.motion = MOTION_MOVING
        else:
            obj.motion = MOTION_STATIONARY

        obj.confidence = conf
        obj.cx, obj.cy = cx, cy
        obj.size = size
        obj.depth_m = depth
        obj.zone = zone_of(cx)
        obj.approach_rate = rate
        obj.last_seen = self._now
        obj.misses = 0

    # --- decision -----------------------------------------------------------

    def _visible(self):
        return [o for o in self.objects.values() if o.misses == 0]

    def _nearest_hazard(self):
        pool = [o for o in self._visible() if o.label in self.hazard_labels]
        if not pool:
            return None
        best = pool[0]
        for o in pool[1:]:
            if o.proximity(self.max_range_m) > best.proximity(self.max_range_m):
                best = o
        return best

    def _best_target(self):
        pool = [o for o in self._visible() if o.label in self.target_labels]
        if not pool:
            return None
        best = pool[0]
        for o in pool[1:]:
            if o.confidence > best.confidence:
                best = o
        return best

    def situation_key(self, hazard):
        if hazard is None:
            return self.goal + "|clear"
        p = hazard.proximity(self.max_range_m)
        band = "near" if p > 0.6 else ("mid" if p > 0.3 else "far")
        return f"{self.goal}|{hazard.label}|{hazard.zone}|{band}"

    def _memory_features(self, situation, action):
        t = self.longterm.recall("outcome:" + situation + ":" + action, self._now)
        if t is None:
            return 0.0, 0.0
        c = t["content"]
        total = c.get("successes", 0) + c.get("failures", 0)
        if total <= 0:
            return 0.0, 0.0
        return c.get("successes", 0) / total, c.get("failures", 0) / total

    def _features(self, hazard, situation):
        proximity = hazard.proximity(self.max_range_m) if hazard else 0.0
        approaching = 1.0 if (hazard and hazard.motion == MOTION_APPROACHING) else 0.0
        if self._any_approach:
            approaching = 1.0
        present = 1.0 if hazard else 0.0
        target = 1.0 if self._best_target() else 0.0

        relevant = {
            STOP: ((F_HAZARD_PROXIMITY, proximity), (F_HAZARD_APPROACHING, approaching)),
            AVOID: ((F_HAZARD_PROXIMITY, proximity), (F_HAZARD_APPROACHING, approaching)),
            MOVE: ((F_PATH_CLEAR, 1.0 - proximity),),
            TRACK: ((F_TARGET_PRESENT, target),),
            WAIT: (),
            SIGNAL: ((F_HAZARD_PRESENT, present),),
        }
        out = {}
        for action in ACTIONS:
            feats = {name: value for name, value in relevant.get(action, ())}
            feats[F_BIAS] = 1.0
            s, f = self._memory_features(situation, action)
            feats[F_MEM_SUCCESS] = s
            feats[F_MEM_FAILURE] = f
            out[action] = feats
        return out

    def _score(self, action, feats):
        total = 0.0
        contributions = {}
        for name, value in feats.items():
            c = self.weights.get(action + "|" + name, 0.0) * value
            contributions[name] = c
            total += c
        return total, contributions

    def _vetoes(self, action, hazard):
        if action in VETO_ADVANCING and hazard is not None:
            proximity = hazard.proximity(self.max_range_m)
            if proximity >= self.veto_proximity:
                return f"{action} forbidden: {hazard.label} too close ({proximity:.2f})"
        if action == TRACK:
            t = self._best_target()
            if t is not None and t.confidence < self.min_confidence:
                return f"{action} forbidden: low confidence {t.confidence:.2f}"
        return None

    def decide(self):
        hazard = self._nearest_hazard()
        situation = self.situation_key(hazard)
        candidates = self._features(hazard, situation)

        ranked = []
        for action in ACTIONS:
            score, contributions = self._score(action, candidates[action])
            ranked.append((action, score, contributions))
        ranked.sort(key=lambda r: r[1], reverse=True)

        vetoed = []
        chosen = None
        for action, score, contributions in ranked:
            reason = self._vetoes(action, hazard)
            if reason is None:
                chosen = (action, score, contributions)
                break
            vetoed.append((action, reason))

        fallback = chosen is None
        if fallback:
            score, contributions = self._score(self.safe_action,
                                               candidates[self.safe_action])
            chosen = (self.safe_action, score, contributions)

        action, score, contributions = chosen
        d = Decision(action, score, situation)
        d.features = candidates
        d.vetoed = vetoed
        d.safety_fallback = fallback
        d.reasons = [(name, value) for name, value in contributions.items()
                     if abs(value) > 1e-9]
        for other in ACTIONS:
            if other == action:
                continue
            fail = candidates[other].get(F_MEM_FAILURE, 0.0)
            if fail:
                penalty = self.weights.get(other + "|" + F_MEM_FAILURE, 0.0) * fail
                if penalty < 0:
                    d.demoted.append((other, penalty))
        if hazard is not None:
            d.avoid_direction = ZONE_RIGHT if hazard.zone != ZONE_RIGHT else ZONE_LEFT

        self.last_decision = d
        self.working.add(("decision", self._now, action))
        return d

    def execute(self, decision=None):
        d = decision or self.last_decision
        if d is None:
            raise ValueError("no decision to execute")
        self.last_action = d.action
        return d.action

    # --- feedback and learning ---------------------------------------------

    def feedback(self, success, decision=None, reward=None):
        d = decision or self.last_decision
        if d is None:
            raise ValueError("no decision to give feedback on")
        if reward is None:
            reward = 1.0 if success else -1.0
        key = "outcome:" + d.situation + ":" + d.action
        trace = self.longterm.recall(key, self._now)
        content = dict(trace["content"]) if trace else {"successes": 0, "failures": 0}
        content["successes" if success else "failures"] += 1
        self.longterm.remember(key, content, self._now,
                               importance=0.6 if success else 0.85)
        self.working.add(("outcome", self._now, d.action, success))
        self._outcome = (d, reward)
        return content

    def learn(self):
        if self._outcome is None:
            return False
        d, reward = self._outcome
        feats = d.features.get(d.action, {})
        for name, value in feats.items():
            key = d.action + "|" + name
            self.weights[key] = self.weights.get(key, 0.0) + self.learning_rate * reward * value
        self._outcome = None
        return True

    # --- persistence --------------------------------------------------------

    def save(self, path="brain.json"):
        import json

        with open(path, "w") as fh:
            json.dump({"weights": self.weights, "memory": self.longterm.snapshot()}, fh)

    def load(self, path="brain.json"):
        import json

        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return False
        self.weights = data.get("weights", self.weights)
        self.longterm.restore(data.get("memory", {}))
        return True
