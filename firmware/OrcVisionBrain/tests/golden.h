// GENERATED FILE — do not edit by hand.
// Regenerate with: python firmware/OrcVisionBrain/tests/generate_golden.py
//
// Golden decision vectors captured from the Python reference brain
// (orcvision.brain). The C++ port must reproduce every action below.
#ifndef ORCVISION_GOLDEN_H
#define ORCVISION_GOLDEN_H

#define GOLDEN_FRAME_W 640
#define GOLDEN_FRAME_H 480

struct GoldenDet {
  const char* label; float conf; float x1, y1, x2, y2; float depth; int16_t track;
};
struct GoldenStep {
  int kind;  // 0 = frame, 1 = decide, 2 = feedback
  float timestamp;
  int det_count;
  const GoldenDet* dets;
  int success;              // for feedback steps
  const char* expect_action; // for decide steps
};
struct GoldenScenario {
  const char* name; int step_count; const GoldenStep* steps;
};

static const GoldenDet closing_obstacle_center_dets_0[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 4.000000f, 1}};
static const GoldenDet closing_obstacle_center_dets_1[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 3.000000f, 1}};
static const GoldenDet closing_obstacle_center_dets_2[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 2.000000f, 1}};
static const GoldenDet closing_obstacle_center_dets_3[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 1.200000f, 1}};
static const GoldenStep closing_obstacle_center_steps[] = {
  {0, 0.000000f, 1, closing_obstacle_center_dets_0, 0, nullptr},
  {0, 0.500000f, 1, closing_obstacle_center_dets_1, 0, nullptr},
  {0, 1.000000f, 1, closing_obstacle_center_dets_2, 0, nullptr},
  {0, 1.500000f, 1, closing_obstacle_center_dets_3, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "AVOID"},  // score +1.7100
};

static const GoldenDet memory_flips_choice_dets_0[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 4.000000f, 1}};
static const GoldenDet memory_flips_choice_dets_1[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 3.000000f, 1}};
static const GoldenDet memory_flips_choice_dets_2[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 2.000000f, 1}};
static const GoldenDet memory_flips_choice_dets_3[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 1.200000f, 1}};
static const GoldenDet memory_flips_choice_dets_6[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 4.000000f, 1}};
static const GoldenDet memory_flips_choice_dets_7[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 3.000000f, 1}};
static const GoldenDet memory_flips_choice_dets_8[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 2.000000f, 1}};
static const GoldenDet memory_flips_choice_dets_9[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 1.200000f, 1}};
static const GoldenStep memory_flips_choice_steps[] = {
  {0, 0.000000f, 1, memory_flips_choice_dets_0, 0, nullptr},
  {0, 0.500000f, 1, memory_flips_choice_dets_1, 0, nullptr},
  {0, 1.000000f, 1, memory_flips_choice_dets_2, 0, nullptr},
  {0, 1.500000f, 1, memory_flips_choice_dets_3, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "AVOID"},  // score +1.7100
  {2, 0.0f, 0, nullptr, 0, nullptr},
  {0, 100.000000f, 1, memory_flips_choice_dets_6, 0, nullptr},
  {0, 100.500000f, 1, memory_flips_choice_dets_7, 0, nullptr},
  {0, 101.000000f, 1, memory_flips_choice_dets_8, 0, nullptr},
  {0, 101.500000f, 1, memory_flips_choice_dets_9, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "STOP"},  // score +1.4840
};

static const GoldenStep clear_scene_moves_steps[] = {
  {0, 0.000000f, 0, nullptr, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "MOVE"},  // score +0.8000
};

static const GoldenDet hazard_on_the_left_dets_0[] = {{"obstacle", 0.900000f, 0.0f, 200.0f, 80.0f, 300.0f, 2.000000f, 1}};
static const GoldenDet hazard_on_the_left_dets_1[] = {{"obstacle", 0.900000f, 0.0f, 200.0f, 80.0f, 300.0f, 1.000000f, 1}};
static const GoldenStep hazard_on_the_left_steps[] = {
  {0, 0.000000f, 1, hazard_on_the_left_dets_0, 0, nullptr},
  {0, 0.500000f, 1, hazard_on_the_left_dets_1, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "AVOID"},  // score +1.7500
};

static const GoldenDet distant_hazard_allows_move_dets_0[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 300.0f, 220.0f, 4.900000f, 1}};
static const GoldenDet distant_hazard_allows_move_dets_1[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 300.0f, 220.0f, 4.900000f, 1}};
static const GoldenStep distant_hazard_allows_move_steps[] = {
  {0, 0.000000f, 1, distant_hazard_allows_move_dets_0, 0, nullptr},
  {0, 0.500000f, 1, distant_hazard_allows_move_dets_1, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "MOVE"},  // score +0.7860
};

static const GoldenDet person_hazard_no_depth_dets_0[] = {{"person", 0.900000f, 200.0f, 100.0f, 440.0f, 400.0f, OV_UNKNOWN_DEPTH, 1}};
static const GoldenDet person_hazard_no_depth_dets_1[] = {{"person", 0.900000f, 180.0f, 80.0f, 460.0f, 420.0f, OV_UNKNOWN_DEPTH, 1}};
static const GoldenStep person_hazard_no_depth_steps[] = {
  {0, 0.000000f, 1, person_hazard_no_depth_dets_0, 0, nullptr},
  {0, 0.500000f, 1, person_hazard_no_depth_dets_1, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "AVOID"},  // score +1.9500
};

static const GoldenDet repeated_failures_degrade_safely_dets_0[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 1.000000f, 1}};
static const GoldenDet repeated_failures_degrade_safely_dets_3[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 1.000000f, 1}};
static const GoldenDet repeated_failures_degrade_safely_dets_6[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 1.000000f, 1}};
static const GoldenDet repeated_failures_degrade_safely_dets_9[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 1.000000f, 1}};
static const GoldenStep repeated_failures_degrade_safely_steps[] = {
  {0, 0.000000f, 1, repeated_failures_degrade_safely_dets_0, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "AVOID"},  // score +0.8500
  {2, 0.0f, 0, nullptr, 0, nullptr},
  {0, 10.000000f, 1, repeated_failures_degrade_safely_dets_3, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "STOP"},  // score +0.7200
  {2, 0.0f, 0, nullptr, 0, nullptr},
  {0, 20.000000f, 1, repeated_failures_degrade_safely_dets_6, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "SIGNAL"},  // score +0.3000
  {2, 0.0f, 0, nullptr, 0, nullptr},
  {0, 30.000000f, 1, repeated_failures_degrade_safely_dets_9, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "WAIT"},  // score +0.1500
};

static const GoldenDet success_keeps_choice_dets_0[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 1.500000f, 1}};
static const GoldenDet success_keeps_choice_dets_3[] = {{"obstacle", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 1.500000f, 1}};
static const GoldenStep success_keeps_choice_steps[] = {
  {0, 0.000000f, 1, success_keeps_choice_dets_0, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "AVOID"},  // score +0.7500
  {2, 0.0f, 0, nullptr, 1, nullptr},
  {0, 10.000000f, 1, success_keeps_choice_dets_3, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "AVOID"},  // score +1.3990
};

static const GoldenDet two_people_must_not_collapse_dets_0[] = {{"person", 0.900000f, 180.0f, 200.0f, 260.0f, 300.0f, 3.000000f, -1}, {"person", 0.900000f, 200.0f, 200.0f, 280.0f, 300.0f, 3.000000f, -1}};
static const GoldenDet two_people_must_not_collapse_dets_1[] = {{"person", 0.900000f, 185.0f, 200.0f, 265.0f, 300.0f, 2.500000f, -1}, {"person", 0.900000f, 205.0f, 200.0f, 285.0f, 300.0f, 2.500000f, -1}};
static const GoldenStep two_people_must_not_collapse_steps[] = {
  {0, 0.000000f, 2, two_people_must_not_collapse_dets_0, 0, nullptr},
  {0, 0.500000f, 2, two_people_must_not_collapse_dets_1, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "AVOID"},  // score +1.4500
};

static const GoldenDet crowd_with_track_ids_dets_0[] = {{"person", 0.900000f, 40.0f, 200.0f, 120.0f, 300.0f, 3.000000f, 1}, {"person", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 2.000000f, 2}, {"person", 0.900000f, 520.0f, 200.0f, 600.0f, 300.0f, 4.000000f, 3}};
static const GoldenDet crowd_with_track_ids_dets_1[] = {{"person", 0.900000f, 40.0f, 200.0f, 120.0f, 300.0f, 2.800000f, 1}, {"person", 0.900000f, 280.0f, 200.0f, 360.0f, 300.0f, 1.200000f, 2}, {"person", 0.900000f, 520.0f, 200.0f, 600.0f, 300.0f, 3.900000f, 3}};
static const GoldenStep crowd_with_track_ids_steps[] = {
  {0, 0.000000f, 3, crowd_with_track_ids_dets_0, 0, nullptr},
  {0, 0.500000f, 3, crowd_with_track_ids_dets_1, 0, nullptr},
  {1, 0.0f, 0, nullptr, 0, "AVOID"},  // score +1.7100
};

static const GoldenScenario GOLDEN_SCENARIOS[] = {
  {"closing_obstacle_center", (int)(sizeof(closing_obstacle_center_steps) / sizeof(GoldenStep)), closing_obstacle_center_steps},
  {"memory_flips_choice", (int)(sizeof(memory_flips_choice_steps) / sizeof(GoldenStep)), memory_flips_choice_steps},
  {"clear_scene_moves", (int)(sizeof(clear_scene_moves_steps) / sizeof(GoldenStep)), clear_scene_moves_steps},
  {"hazard_on_the_left", (int)(sizeof(hazard_on_the_left_steps) / sizeof(GoldenStep)), hazard_on_the_left_steps},
  {"distant_hazard_allows_move", (int)(sizeof(distant_hazard_allows_move_steps) / sizeof(GoldenStep)), distant_hazard_allows_move_steps},
  {"person_hazard_no_depth", (int)(sizeof(person_hazard_no_depth_steps) / sizeof(GoldenStep)), person_hazard_no_depth_steps},
  {"repeated_failures_degrade_safely", (int)(sizeof(repeated_failures_degrade_safely_steps) / sizeof(GoldenStep)), repeated_failures_degrade_safely_steps},
  {"success_keeps_choice", (int)(sizeof(success_keeps_choice_steps) / sizeof(GoldenStep)), success_keeps_choice_steps},
  {"two_people_must_not_collapse", (int)(sizeof(two_people_must_not_collapse_steps) / sizeof(GoldenStep)), two_people_must_not_collapse_steps},
  {"crowd_with_track_ids", (int)(sizeof(crowd_with_track_ids_steps) / sizeof(GoldenStep)), crowd_with_track_ids_steps},
};
static const int GOLDEN_SCENARIO_COUNT = (int)(sizeof(GOLDEN_SCENARIOS) / sizeof(GoldenScenario));

#endif  // ORCVISION_GOLDEN_H
