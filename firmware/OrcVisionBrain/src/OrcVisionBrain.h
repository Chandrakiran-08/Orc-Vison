/*
 * OrcVisionBrain — the Orc-Vison autonomous decision brain, in C++ for
 * microcontrollers.
 * ---------------------------------------------------------------------------
 * A faithful port of `orcvision.brain` (Python) sized for boards with tens of
 * kilobytes of RAM, including the Arduino Uno R4 WiFi (Renesas RA4M1, 32 KB
 * SRAM). Same loop, same decisions:
 *
 *     Perception -> State -> Memory -> Decision -> Action -> Feedback
 *
 * The MCU does NOT run object detection. Detections arrive already made — over
 * MQTT, serial, or any transport — from a host running the perception half.
 * This library is the intelligence layer that decides what to do about them,
 * and it keeps running (and remembering) even if the link to the host drops.
 *
 * Embedded constraints honoured throughout:
 *   - No dynamic allocation. No malloc/new, no String, no STL containers.
 *     Every store is a fixed-capacity array sized by the macros below.
 *   - No exceptions, no RTTI, no recursion.
 *   - float (not double) maths; only expf/logf/sqrtf from <math.h>.
 *   - Deterministic, bounded work per frame — no unbounded loops.
 *   - Header-only, C++11. Compiles for AVR/ARM/ESP32/host alike.
 *
 * Tuning (define before including to override):
 *   OV_MAX_OBJECTS  tracked objects        (default 8)
 *   OV_MAX_LABELS   distinct class names   (default 12)
 *   OV_LABEL_LEN    max label length + NUL (default 14)
 *   OV_MAX_TRACES   long-term memories     (default 12)
 *
 * Verification status: the decision logic is verified against the Python
 * implementation by a golden-vector parity test (see tests/), and the library
 * is verified on real hardware — an Arduino Uno R4 WiFi runs the bundled
 * BrainSelfTest example 5/5 on-device, using 1284 bytes of SRAM and making
 * decisions byte-identical to the Python reference. Other boards (ESP32,
 * Pico W, STM32) are untested; confirm on your own before trusting them.
 *
 * License: Apache-2.0, same as the rest of Orc-Vison.
 */

#ifndef ORCVISION_BRAIN_H
#define ORCVISION_BRAIN_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef OV_MAX_OBJECTS
#define OV_MAX_OBJECTS 8
#endif
#ifndef OV_MAX_LABELS
#define OV_MAX_LABELS 12
#endif
#ifndef OV_LABEL_LEN
#define OV_LABEL_LEN 14
#endif
#ifndef OV_MAX_TRACES
#define OV_MAX_TRACES 12
#endif

// --- action vocabulary (mirrors orcvision/brain/actions.py) ----------------
enum OvAction : uint8_t {
  OV_STOP = 0,
  OV_AVOID = 1,
  OV_MOVE = 2,
  OV_TRACK = 3,
  OV_WAIT = 4,
  OV_SIGNAL = 5,
  OV_NUM_ACTIONS = 6
};

// --- decision features (mirrors the Python feature names) ------------------
enum OvFeature : uint8_t {
  OV_F_HAZARD_PROXIMITY = 0,
  OV_F_HAZARD_APPROACHING = 1,
  OV_F_HAZARD_PRESENT = 2,
  OV_F_PATH_CLEAR = 3,
  OV_F_TARGET_PRESENT = 4,
  OV_F_BIAS = 5,
  OV_F_MEM_SUCCESS = 6,
  OV_F_MEM_FAILURE = 7,
  OV_NUM_FEATURES = 8
};

// Hazard/target membership is a uint16_t bitmask indexed by label id, so a
// label table larger than 16 would leave high-numbered labels permanently
// unable to be marked as hazards — a silent safety failure rather than a
// visible error. Fail the build instead.
#if OV_MAX_LABELS > 16
#error "OV_MAX_LABELS must be <= 16 (hazard/target masks are 16-bit)"
#endif

enum OvZone : uint8_t { OV_ZONE_LEFT = 0, OV_ZONE_CENTER = 1, OV_ZONE_RIGHT = 2 };

enum OvMotion : uint8_t {
  OV_MOTION_UNKNOWN = 0,
  OV_MOTION_STATIONARY = 1,
  OV_MOTION_MOVING = 2,
  OV_MOTION_APPROACHING = 3,
  OV_MOTION_RECEDING = 4
};

static const uint8_t OV_NO_LABEL = 0xFF;
static const float OV_UNKNOWN_DEPTH = -1.0f;  // sentinel: no depth available

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
static inline float ovClamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

static inline uint8_t ovZoneOf(float cx) {
  if (cx < 1.0f / 3.0f) return OV_ZONE_LEFT;
  if (cx > 2.0f / 3.0f) return OV_ZONE_RIGHT;
  return OV_ZONE_CENTER;
}

// ---------------------------------------------------------------------------
// Interned label table — avoids storing/comparing strings per object.
// ---------------------------------------------------------------------------
class OvLabelTable {
 public:
  OvLabelTable() : count_(0) {}

  int8_t find(const char* name) const {
    for (uint8_t i = 0; i < count_; ++i) {
      if (strncmp(names_[i], name, OV_LABEL_LEN) == 0) return (int8_t)i;
    }
    return -1;
  }

  // Returns the id for `name`, adding it if new. OV_NO_LABEL when full.
  uint8_t intern(const char* name) {
    int8_t existing = find(name);
    if (existing >= 0) return (uint8_t)existing;
    if (count_ >= OV_MAX_LABELS) return OV_NO_LABEL;
    strncpy(names_[count_], name, OV_LABEL_LEN - 1);
    names_[count_][OV_LABEL_LEN - 1] = '\0';
    return count_++;
  }

  const char* name(uint8_t id) const {
    return (id < count_) ? names_[id] : "?";
  }

  uint8_t count() const { return count_; }

 private:
  char names_[OV_MAX_LABELS][OV_LABEL_LEN];
  uint8_t count_;
};

// ---------------------------------------------------------------------------
// One tracked object, as the brain understands it.
// ---------------------------------------------------------------------------
struct OvObject {
  uint8_t label_id;
  int16_t track_id;  // <0 when perception supplied none
  float cx, cy;      // normalized 0..1
  float size;        // normalized bbox area 0..1
  float depth_m;     // OV_UNKNOWN_DEPTH when unavailable
  float confidence;
  float approach_rate;  // m/s toward the sensor, >0 closing
  float first_seen, last_seen;
  uint8_t zone;
  uint8_t motion;
  uint8_t misses;
  bool used;

  float proximity(float max_range_m) const {
    if (depth_m > OV_UNKNOWN_DEPTH && max_range_m > 0.0f) {
      return ovClamp01(1.0f - (depth_m / max_range_m));
    }
    return ovClamp01(sqrtf(size) * 2.0f);
  }
};

// ---------------------------------------------------------------------------
// A long-term memory: how one action worked out in one kind of situation.
// ---------------------------------------------------------------------------
struct OvTrace {
  uint32_t key;  // situation hash combined with the action id
  uint16_t successes;
  uint16_t failures;
  uint16_t hits;
  float importance;
  float last_reinforced;
  bool used;
};

// ---------------------------------------------------------------------------
// The result of one decision, with everything needed to explain it.
// ---------------------------------------------------------------------------
struct OvDecision {
  uint8_t action;
  float score;
  uint32_t situation;
  bool safety_fallback;
  uint8_t vetoed_mask;   // bit i set => action i was forbidden
  uint8_t demoted_mask;  // bit i set => action i was pushed down by memory
  float contributions[OV_NUM_FEATURES];
  float scores[OV_NUM_ACTIONS];
  // Retained so feedback()/learn() can update exactly what was scored.
  float features[OV_NUM_ACTIONS][OV_NUM_FEATURES];
  // Parameter for AVOID: which way to steer.
  uint8_t avoid_zone;  // OV_ZONE_LEFT or OV_ZONE_RIGHT
};

// ---------------------------------------------------------------------------
// Configuration.
// ---------------------------------------------------------------------------
struct OvConfig {
  float max_range_m;
  float move_threshold;         // normalized displacement counted as motion
  float approach_threshold;     // m/s of depth closing counted as approaching
  float size_growth_threshold;  // relative bbox growth ~ approaching
  uint8_t disappear_after_misses;
  float match_radius;  // normalized distance for id-less association
  float veto_proximity;      // proximity at/above which advancing is forbidden
  float min_confidence;      // below this, target-directed actions are vetoed
  float half_life_s;         // long-term memory decay half-life
  float forget_below;        // strength floor before a trace is dropped
  float learning_rate;
  uint8_t safe_action;
  uint16_t hazard_mask;  // bit per label id
  uint16_t target_mask;  // bit per label id
  uint8_t goal_id;

  OvConfig()
      : max_range_m(5.0f),
        move_threshold(0.02f),
        approach_threshold(0.05f),
        size_growth_threshold(0.12f),
        disappear_after_misses(5),
        match_radius(0.15f),
        veto_proximity(0.6f),
        min_confidence(0.4f),
        half_life_s(600.0f),
        forget_below(0.05f),
        learning_rate(0.1f),
        safe_action(OV_STOP),
        hazard_mask(0),
        target_mask(0),
        goal_id(0) {}
};

// ---------------------------------------------------------------------------
// The brain.
// ---------------------------------------------------------------------------
class OrcVisionBrain {
 public:
  OrcVisionBrain() { begin(); }

  // --- lifecycle ----------------------------------------------------------

  void begin() {
    memset(objects_, 0, sizeof(objects_));
    memset(traces_, 0, sizeof(traces_));
    tick_ = 0;
    updated_at_ = 0.0f;
    now_ = 0.0f;
    last_action_ = OV_NUM_ACTIONS;
    any_approach_ = false;
    have_outcome_ = false;
    loadDefaultWeights();
  }

  OvConfig& config() { return cfg_; }
  OvLabelTable& labels() { return labels_; }

  // Mark a class name as a hazard (something to reason about avoiding).
  void addHazardLabel(const char* name) {
    uint8_t id = labels_.intern(name);
    if (id != OV_NO_LABEL) cfg_.hazard_mask |= (uint16_t)(1u << id);
  }

  // Mark a class name as a target (something worth tracking/following).
  void addTargetLabel(const char* name) {
    uint8_t id = labels_.intern(name);
    if (id != OV_NO_LABEL) cfg_.target_mask |= (uint16_t)(1u << id);
  }

  void setGoal(uint8_t goal_id) { cfg_.goal_id = goal_id; }

  // --- perception ---------------------------------------------------------

  // Begin a frame. Call observe*() for each detection, then endFrame().
  //
  // IMPORTANT — `timestamp` must be a SMALL monotonic value in seconds, such
  // as millis()/1000.0f. Do NOT pass a Unix epoch: float32 carries 24 bits of
  // mantissa, so near 1.79e9 representable values are ~128 seconds apart.
  // Frame-to-frame deltas round to zero, dt collapses onto the minimum-dt
  // clamp below, and every derived rate becomes meaningless — a measured
  // 1.6 m/s approach reads as 8e5 m/s, and memory decay quantises into 128 s
  // steps. Motion classification survives by luck, so this degrades quietly
  // instead of failing loudly. Keep the magnitude small.
  void beginFrame(float timestamp) {
    now_ = timestamp;
    any_approach_ = false;
    for (uint8_t i = 0; i < OV_MAX_OBJECTS; ++i) seen_[i] = false;
    pending_count_ = 0;
  }

  // Feed one detection with already-normalized geometry (0..1).
  bool observe(const char* label, float confidence, float cx, float cy, float size,
               float depth_m = OV_UNKNOWN_DEPTH, int16_t track_id = -1) {
    if (pending_count_ >= OV_MAX_OBJECTS) return false;
    uint8_t id = labels_.intern(label);
    if (id == OV_NO_LABEL) return false;
    OvPending& p = pending_[pending_count_++];
    p.label_id = id;
    p.confidence = confidence;
    p.cx = ovClamp01(cx);
    p.cy = ovClamp01(cy);
    p.size = ovClamp01(size);
    p.depth_m = depth_m;
    p.track_id = track_id;
    return true;
  }

  // Feed one detection in pixel coordinates; normalized internally.
  bool observePixels(const char* label, float confidence, float x1, float y1, float x2,
                     float y2, uint16_t frame_w, uint16_t frame_h,
                     float depth_m = OV_UNKNOWN_DEPTH, int16_t track_id = -1) {
    if (frame_w == 0 || frame_h == 0) return false;
    float cx = ((x1 + x2) * 0.5f) / (float)frame_w;
    float cy = ((y1 + y2) * 0.5f) / (float)frame_h;
    float area = (fabsf(x2 - x1) * fabsf(y2 - y1)) / ((float)frame_w * (float)frame_h);
    return observe(label, confidence, cx, cy, area, depth_m, track_id);
  }

  // Fold the frame into the world model: association, motion, ageing.
  void endFrame() {
    float dt = tick_ ? (now_ - updated_at_) : 0.0f;
    if (dt < 1e-6f && dt > -1e-6f) dt = (tick_ ? 1e-6f : 0.0f);

    for (uint8_t p = 0; p < pending_count_; ++p) {
      OvPending& obs = pending_[p];
      int8_t slot = associate(obs);
      if (slot < 0) {
        slot = freeSlot();
        if (slot < 0) continue;  // at capacity: drop, never grow
        OvObject& o = objects_[slot];
        o.used = true;
        o.label_id = obs.label_id;
        o.track_id = obs.track_id;
        o.cx = obs.cx;
        o.cy = obs.cy;
        o.size = obs.size;
        o.depth_m = obs.depth_m;
        o.confidence = obs.confidence;
        o.zone = ovZoneOf(obs.cx);
        o.motion = OV_MOTION_UNKNOWN;
        o.approach_rate = 0.0f;
        o.first_seen = now_;
        o.last_seen = now_;
        o.misses = 0;
      } else {
        updateMatched(objects_[slot], obs, dt);
      }
      seen_[slot] = true;
    }

    // Age out anything not observed this frame.
    for (uint8_t i = 0; i < OV_MAX_OBJECTS; ++i) {
      if (!objects_[i].used || seen_[i]) continue;
      if (++objects_[i].misses >= cfg_.disappear_after_misses) {
        objects_[i].used = false;
      }
    }

    updated_at_ = now_;
    ++tick_;
  }

  // --- decision -----------------------------------------------------------

  OvDecision decide() {
    OvDecision d;
    memset(&d, 0, sizeof(d));

    const OvObject* hazard = nearestHazard();
    d.situation = situationKey(hazard);
    d.avoid_zone = OV_ZONE_RIGHT;
    if (hazard) {
      d.avoid_zone = (hazard->zone == OV_ZONE_RIGHT) ? OV_ZONE_LEFT : OV_ZONE_RIGHT;
    }

    buildFeatures(hazard, d.situation, d.features);

    for (uint8_t a = 0; a < OV_NUM_ACTIONS; ++a) {
      d.scores[a] = scoreAction(a, d.features[a]);
      if (d.features[a][OV_F_MEM_FAILURE] > 0.0f) {
        float penalty = weights_[a][OV_F_MEM_FAILURE] * d.features[a][OV_F_MEM_FAILURE];
        if (penalty < 0.0f) d.demoted_mask |= (uint8_t)(1u << a);
      }
    }

    // Rank by score, then take the best action that survives every
    // constraint. A learned policy may reorder preferences; it can never
    // unlock a forbidden action.
    bool chosen = false;
    for (uint8_t rank = 0; rank < OV_NUM_ACTIONS && !chosen; ++rank) {
      int8_t best = -1;
      for (uint8_t a = 0; a < OV_NUM_ACTIONS; ++a) {
        if (considered_ & (1u << a)) continue;
        if (best < 0 || d.scores[a] > d.scores[best]) best = (int8_t)a;
      }
      if (best < 0) break;
      considered_ |= (uint8_t)(1u << best);
      if (vetoes((uint8_t)best, hazard)) {
        d.vetoed_mask |= (uint8_t)(1u << best);
      } else {
        d.action = (uint8_t)best;
        d.score = d.scores[best];
        chosen = true;
      }
    }
    considered_ = 0;

    if (!chosen) {
      // Everything forbidden: take the safe action, not the least-bad
      // forbidden one.
      d.action = cfg_.safe_action;
      d.score = d.scores[cfg_.safe_action];
      d.safety_fallback = true;
    }

    for (uint8_t f = 0; f < OV_NUM_FEATURES; ++f) {
      d.contributions[f] = weights_[d.action][f] * d.features[d.action][f];
    }

    last_ = d;
    have_decision_ = true;
    return d;
  }

  void markExecuted(const OvDecision& d) { last_action_ = d.action; }

  // --- feedback and learning ---------------------------------------------

  // Report how the last decision turned out. Updates memory immediately —
  // the next decide() already accounts for it.
  void feedback(bool success, float reward = 0.0f) {
    if (!have_decision_) return;
    if (reward == 0.0f) reward = success ? 1.0f : -1.0f;

    OvTrace* t = traceFor(last_.situation, last_.action, /*create=*/true);
    if (t) {
      if (success) {
        if (t->successes < 0xFFFF) t->successes++;
      } else {
        if (t->failures < 0xFFFF) t->failures++;
      }
      if (t->hits < 0xFFFF) t->hits++;
      t->last_reinforced = now_;
      float target = success ? 0.6f : 0.85f;
      t->importance += (1.0f - t->importance) * 0.3f;
      if (t->importance < target) t->importance = target;
      if (t->importance > 1.0f) t->importance = 1.0f;
    }
    outcome_reward_ = reward;
    outcome_action_ = last_.action;
    have_outcome_ = true;
  }

  // Fold the last outcome into the policy weights (reward-driven update).
  bool learn() {
    if (!have_outcome_) return false;
    for (uint8_t f = 0; f < OV_NUM_FEATURES; ++f) {
      weights_[outcome_action_][f] +=
          cfg_.learning_rate * outcome_reward_ * last_.features[outcome_action_][f];
    }
    have_outcome_ = false;
    return true;
  }

  // --- introspection ------------------------------------------------------

  const OvObject* objectAt(uint8_t i) const {
    return (i < OV_MAX_OBJECTS && objects_[i].used) ? &objects_[i] : nullptr;
  }

  uint8_t visibleCount() const {
    uint8_t n = 0;
    for (uint8_t i = 0; i < OV_MAX_OBJECTS; ++i) {
      if (objects_[i].used && objects_[i].misses == 0) ++n;
    }
    return n;
  }

  float weight(uint8_t action, uint8_t feature) const { return weights_[action][feature]; }
  void setWeight(uint8_t action, uint8_t feature, float v) { weights_[action][feature] = v; }

  uint8_t traceCount() const {
    uint8_t n = 0;
    for (uint8_t i = 0; i < OV_MAX_TRACES; ++i) {
      if (traces_[i].used) ++n;
    }
    return n;
  }

  static const char* actionName(uint8_t a) {
    switch (a) {
      case OV_STOP: return "STOP";
      case OV_AVOID: return "AVOID";
      case OV_MOVE: return "MOVE";
      case OV_TRACK: return "TRACK";
      case OV_WAIT: return "WAIT";
      case OV_SIGNAL: return "SIGNAL";
      default: return "?";
    }
  }

  static const char* featureName(uint8_t f) {
    switch (f) {
      case OV_F_HAZARD_PROXIMITY: return "hazard at close range";
      case OV_F_HAZARD_APPROACHING: return "hazard moving toward system";
      case OV_F_HAZARD_PRESENT: return "hazard detected";
      case OV_F_PATH_CLEAR: return "path appears clear";
      case OV_F_TARGET_PRESENT: return "tracking target visible";
      case OV_F_BIAS: return "baseline preference";
      case OV_F_MEM_SUCCESS: return "previously succeeded here";
      case OV_F_MEM_FAILURE: return "previously failed here";
      default: return "?";
    }
  }

  static const char* zoneName(uint8_t z) {
    return z == OV_ZONE_LEFT ? "left" : (z == OV_ZONE_RIGHT ? "right" : "center");
  }

  // Render a human-readable justification into a caller-provided buffer.
  // No allocation; always NUL-terminated.
  size_t explain(const OvDecision& d, char* buf, size_t cap) const {
    if (!buf || cap == 0) return 0;
    size_t n = 0;
    n += appendStr(buf, cap, n, "Decision: ");
    n += appendStr(buf, cap, n, actionName(d.action));
    if (d.action == OV_AVOID) {
      n += appendStr(buf, cap, n, " (steer ");
      n += appendStr(buf, cap, n, zoneName(d.avoid_zone));
      n += appendStr(buf, cap, n, ")");
    }
    if (d.safety_fallback) n += appendStr(buf, cap, n, "\n  ! all actions vetoed -> safe action");
    for (uint8_t f = 0; f < OV_NUM_FEATURES; ++f) {
      float c = d.contributions[f];
      if (c > -1e-6f && c < 1e-6f) continue;
      n += appendStr(buf, cap, n, "\n  ");
      n += appendStr(buf, cap, n, featureName(f));
      n += appendStr(buf, cap, n, " (");
      n += appendFloat(buf, cap, n, c);
      n += appendStr(buf, cap, n, ")");
    }
    for (uint8_t a = 0; a < OV_NUM_ACTIONS; ++a) {
      if (d.demoted_mask & (1u << a)) {
        n += appendStr(buf, cap, n, "\n  - ");
        n += appendStr(buf, cap, n, actionName(a));
        n += appendStr(buf, cap, n, " down-weighted: previously failed here");
      }
      if (d.vetoed_mask & (1u << a)) {
        n += appendStr(buf, cap, n, "\n  x ");
        n += appendStr(buf, cap, n, actionName(a));
        n += appendStr(buf, cap, n, " vetoed by safety constraint");
      }
    }
    return n;
  }

  // --- persistence (optional; caller supplies the storage) ---------------

  struct Snapshot {
    float weights[OV_NUM_ACTIONS][OV_NUM_FEATURES];
    OvTrace traces[OV_MAX_TRACES];
  };

  void save(Snapshot& out) const {
    memcpy(out.weights, weights_, sizeof(weights_));
    memcpy(out.traces, traces_, sizeof(traces_));
  }

  void restore(const Snapshot& in) {
    memcpy(weights_, in.weights, sizeof(weights_));
    memcpy(traces_, in.traces, sizeof(traces_));
  }

 private:
  struct OvPending {
    uint8_t label_id;
    int16_t track_id;
    float cx, cy, size, depth_m, confidence;
  };

  // --- defaults (must mirror DEFAULT_WEIGHTS in decision.py) -------------
  void loadDefaultWeights() {
    memset(weights_, 0, sizeof(weights_));
    weights_[OV_AVOID][OV_F_HAZARD_PROXIMITY] = 1.00f;
    weights_[OV_AVOID][OV_F_HAZARD_APPROACHING] = 0.90f;
    weights_[OV_AVOID][OV_F_BIAS] = 0.05f;
    weights_[OV_STOP][OV_F_HAZARD_PROXIMITY] = 0.90f;
    weights_[OV_STOP][OV_F_HAZARD_APPROACHING] = 0.80f;
    weights_[OV_STOP][OV_F_BIAS] = 0.00f;
    weights_[OV_MOVE][OV_F_PATH_CLEAR] = 0.70f;
    weights_[OV_MOVE][OV_F_BIAS] = 0.10f;
    weights_[OV_TRACK][OV_F_TARGET_PRESENT] = 0.60f;
    weights_[OV_WAIT][OV_F_BIAS] = 0.15f;
    weights_[OV_SIGNAL][OV_F_HAZARD_PRESENT] = 0.30f;
    for (uint8_t a = 0; a < OV_NUM_ACTIONS; ++a) {
      weights_[a][OV_F_MEM_FAILURE] = -1.50f;
      weights_[a][OV_F_MEM_SUCCESS] = 0.50f;
    }
  }

  bool isHazard(uint8_t label_id) const {
    return label_id < 16 && (cfg_.hazard_mask & (uint16_t)(1u << label_id)) != 0;
  }
  bool isTarget(uint8_t label_id) const {
    return label_id < 16 && (cfg_.target_mask & (uint16_t)(1u << label_id)) != 0;
  }

  int8_t freeSlot() const {
    for (uint8_t i = 0; i < OV_MAX_OBJECTS; ++i) {
      if (!objects_[i].used) return (int8_t)i;
    }
    return -1;
  }

  // Track-id match first; otherwise nearest same-label object in radius, so
  // the brain still keeps identity behind a detector with no tracker.
  //
  // Slots already claimed by an earlier detection in THIS frame are skipped.
  // Without that, two nearby same-label detections (two people standing
  // together, with no tracker supplying ids) both associate to the same
  // stored object and silently collapse into one — the brain would then
  // reason about one person where there are two.
  int8_t associate(const OvPending& obs) const {
    if (obs.track_id >= 0) {
      for (uint8_t i = 0; i < OV_MAX_OBJECTS; ++i) {
        if (seen_[i]) continue;
        if (objects_[i].used && objects_[i].track_id == obs.track_id &&
            objects_[i].label_id == obs.label_id) {
          return (int8_t)i;
        }
      }
      return -1;
    }
    int8_t best = -1;
    float best_dist = cfg_.match_radius;
    for (uint8_t i = 0; i < OV_MAX_OBJECTS; ++i) {
      if (seen_[i]) continue;
      if (!objects_[i].used || objects_[i].label_id != obs.label_id) continue;
      float dx = objects_[i].cx - obs.cx;
      float dy = objects_[i].cy - obs.cy;
      float dist = sqrtf(dx * dx + dy * dy);
      if (dist < best_dist) {
        best_dist = dist;
        best = (int8_t)i;
      }
    }
    return best;
  }

  void updateMatched(OvObject& o, const OvPending& obs, float dt) {
    float dx = obs.cx - o.cx;
    float dy = obs.cy - o.cy;
    float displacement = sqrtf(dx * dx + dy * dy);

    float approach_rate = 0.0f;
    if (o.depth_m > OV_UNKNOWN_DEPTH && obs.depth_m > OV_UNKNOWN_DEPTH && dt > 0.0f) {
      approach_rate = (o.depth_m - obs.depth_m) / dt;
    }
    float size_growth = (o.size > 0.0f) ? (obs.size - o.size) / o.size : 0.0f;
    bool no_depth = !(obs.depth_m > OV_UNKNOWN_DEPTH);

    bool approaching = (approach_rate > cfg_.approach_threshold) ||
                       (no_depth && size_growth > cfg_.size_growth_threshold);
    bool receding = (approach_rate < -cfg_.approach_threshold) ||
                    (no_depth && size_growth < -cfg_.size_growth_threshold);

    if (approaching) {
      o.motion = OV_MOTION_APPROACHING;
      any_approach_ = true;
    } else if (receding) {
      o.motion = OV_MOTION_RECEDING;
    } else if (displacement > cfg_.move_threshold) {
      o.motion = OV_MOTION_MOVING;
    } else {
      o.motion = OV_MOTION_STATIONARY;
    }

    o.cx = obs.cx;
    o.cy = obs.cy;
    o.size = obs.size;
    o.depth_m = obs.depth_m;
    o.confidence = obs.confidence;
    o.track_id = obs.track_id;
    o.zone = ovZoneOf(obs.cx);
    o.approach_rate = approach_rate;
    o.last_seen = now_;
    o.misses = 0;
  }

  const OvObject* nearestHazard() const {
    const OvObject* best = nullptr;
    float best_prox = -1.0f;
    for (uint8_t i = 0; i < OV_MAX_OBJECTS; ++i) {
      const OvObject& o = objects_[i];
      if (!o.used || o.misses != 0 || !isHazard(o.label_id)) continue;
      float p = o.proximity(cfg_.max_range_m);
      if (p > best_prox) {
        best_prox = p;
        best = &o;
      }
    }
    return best;
  }

  const OvObject* bestTarget() const {
    const OvObject* best = nullptr;
    for (uint8_t i = 0; i < OV_MAX_OBJECTS; ++i) {
      const OvObject& o = objects_[i];
      if (!o.used || o.misses != 0 || !isTarget(o.label_id)) continue;
      if (!best || o.confidence > best->confidence) best = &o;
    }
    return best;
  }

  // Compact equivalent of Python's "goal|label|zone|band" situation string.
  uint32_t situationKey(const OvObject* hazard) const {
    if (!hazard) return ((uint32_t)cfg_.goal_id << 24) | 0x00FFFFFFu;  // "clear"
    float p = hazard->proximity(cfg_.max_range_m);
    uint8_t band = (p > 0.6f) ? 2 : ((p > 0.3f) ? 1 : 0);
    return ((uint32_t)cfg_.goal_id << 24) | ((uint32_t)hazard->label_id << 16) |
           ((uint32_t)hazard->zone << 8) | (uint32_t)band;
  }

  OvTrace* traceFor(uint32_t situation, uint8_t action, bool create) {
    uint32_t key = situation ^ ((uint32_t)(action + 1) * 0x9E3779B1u);
    OvTrace* weakest = nullptr;
    float weakest_strength = 1e30f;
    for (uint8_t i = 0; i < OV_MAX_TRACES; ++i) {
      OvTrace& t = traces_[i];
      if (t.used && t.key == key) return &t;
      if (!t.used) {
        if (!weakest || weakest->used) {
          weakest = &t;
          weakest_strength = -1.0f;
        }
        continue;
      }
      float s = strength(t);
      if (s < cfg_.forget_below) {  // decayed away: reclaim it
        t.used = false;
        weakest = &t;
        weakest_strength = -1.0f;
        continue;
      }
      if (s < weakest_strength && (!weakest || weakest->used)) {
        weakest = &t;
        weakest_strength = s;
      }
    }
    if (!create || !weakest) return nullptr;
    weakest->used = true;
    weakest->key = key;
    weakest->successes = 0;
    weakest->failures = 0;
    weakest->hits = 0;
    weakest->importance = 0.5f;
    weakest->last_reinforced = now_;
    return weakest;
  }

  float strength(const OvTrace& t) const {
    if (cfg_.half_life_s <= 0.0f) return t.importance;
    float elapsed = now_ - t.last_reinforced;
    if (elapsed < 0.0f) elapsed = 0.0f;
    float decay = expf(-elapsed * 0.6931472f / cfg_.half_life_s);
    float repetition = 1.0f + logf(1.0f + (float)(t.hits > 0 ? t.hits - 1 : 0)) * 0.25f;
    return t.importance * decay * repetition;
  }

  void memoryFeatures(uint32_t situation, uint8_t action, float* success, float* failure) {
    *success = 0.0f;
    *failure = 0.0f;
    OvTrace* t = traceFor(situation, action, /*create=*/false);
    if (!t) return;
    float total = (float)t->successes + (float)t->failures;
    if (total <= 0.0f) return;
    *success = (float)t->successes / total;
    *failure = (float)t->failures / total;
  }

  void buildFeatures(const OvObject* hazard, uint32_t situation,
                     float out[OV_NUM_ACTIONS][OV_NUM_FEATURES]) {
    float proximity = hazard ? hazard->proximity(cfg_.max_range_m) : 0.0f;
    float approaching = 0.0f;
    if (hazard && hazard->motion == OV_MOTION_APPROACHING) approaching = 1.0f;
    if (any_approach_) approaching = 1.0f;
    float present = hazard ? 1.0f : 0.0f;
    float target = bestTarget() ? 1.0f : 0.0f;

    memset(out, 0, sizeof(float) * OV_NUM_ACTIONS * OV_NUM_FEATURES);
    for (uint8_t a = 0; a < OV_NUM_ACTIONS; ++a) {
      switch (a) {
        case OV_STOP:
        case OV_AVOID:
          out[a][OV_F_HAZARD_PROXIMITY] = proximity;
          out[a][OV_F_HAZARD_APPROACHING] = approaching;
          break;
        case OV_MOVE:
          out[a][OV_F_PATH_CLEAR] = 1.0f - proximity;
          break;
        case OV_TRACK:
          out[a][OV_F_TARGET_PRESENT] = target;
          break;
        case OV_SIGNAL:
          out[a][OV_F_HAZARD_PRESENT] = present;
          break;
        default:
          break;  // WAIT: bias only
      }
      out[a][OV_F_BIAS] = 1.0f;
      float s, f;
      memoryFeatures(situation, a, &s, &f);
      out[a][OV_F_MEM_SUCCESS] = s;
      out[a][OV_F_MEM_FAILURE] = f;
    }
  }

  float scoreAction(uint8_t action, const float* features) const {
    float total = 0.0f;
    for (uint8_t f = 0; f < OV_NUM_FEATURES; ++f) total += weights_[action][f] * features[f];
    return total;
  }

  // Deterministic safety floor — never learned, never overridden.
  bool vetoes(uint8_t action, const OvObject* hazard) const {
    if (action == OV_MOVE) {
      if (hazard && hazard->proximity(cfg_.max_range_m) >= cfg_.veto_proximity) return true;
    }
    if (action == OV_TRACK) {
      const OvObject* t = bestTarget();
      if (t && t->confidence < cfg_.min_confidence) return true;
    }
    return false;
  }

  // --- tiny formatting helpers (no snprintf float on AVR) ----------------
  static size_t appendStr(char* buf, size_t cap, size_t n, const char* s) {
    size_t w = 0;
    while (*s && n + w + 1 < cap) buf[n + w++] = *s++;
    buf[n + w < cap ? n + w : cap - 1] = '\0';
    return w;
  }

  // Format a float as +/-D[DD].DD without pulling in printf's float support.
  //
  // Casting a non-finite or out-of-range float to int is undefined
  // behaviour, and weights can genuinely run away under repeated reward
  // updates, so both cases are handled explicitly rather than trusted not
  // to happen.
  static size_t appendFloat(char* buf, size_t cap, size_t n, float v) {
    if (isnan(v)) return appendStr(buf, cap, n, "nan");
    bool neg = v < 0.0f;
    if (neg) v = -v;
    if (isinf(v)) return appendStr(buf, cap, n, neg ? "-inf" : "+inf");
    if (v > 999.0f) return appendStr(buf, cap, n, neg ? "-big" : "+big");

    char tmp[16];
    int whole = (int)v;  // safe: v is finite and <= 999 here
    int frac = (int)((v - (float)whole) * 100.0f + 0.5f);
    if (frac >= 100) {
      frac = 0;
      whole += 1;
    }
    size_t i = 0;
    tmp[i++] = neg ? '-' : '+';
    if (whole >= 100) tmp[i++] = (char)('0' + (whole / 100) % 10);
    if (whole >= 10) tmp[i++] = (char)('0' + (whole / 10) % 10);
    tmp[i++] = (char)('0' + whole % 10);
    tmp[i++] = '.';
    tmp[i++] = (char)('0' + frac / 10);
    tmp[i++] = (char)('0' + frac % 10);
    tmp[i] = '\0';
    return appendStr(buf, cap, n, tmp);
  }

  OvConfig cfg_;
  OvLabelTable labels_;
  OvObject objects_[OV_MAX_OBJECTS];
  OvTrace traces_[OV_MAX_TRACES];
  float weights_[OV_NUM_ACTIONS][OV_NUM_FEATURES];

  OvPending pending_[OV_MAX_OBJECTS];
  uint8_t pending_count_ = 0;
  bool seen_[OV_MAX_OBJECTS] = {false};

  OvDecision last_;
  bool have_decision_ = false;
  bool have_outcome_ = false;
  uint8_t outcome_action_ = 0;
  float outcome_reward_ = 0.0f;

  float now_ = 0.0f;
  float updated_at_ = 0.0f;
  uint32_t tick_ = 0;
  uint8_t last_action_ = OV_NUM_ACTIONS;
  bool any_approach_ = false;
  mutable uint8_t considered_ = 0;
};

#endif  // ORCVISION_BRAIN_H
