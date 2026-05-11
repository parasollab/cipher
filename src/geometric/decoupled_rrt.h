#ifndef DECOUPLED_RRT_H
#define DECOUPLED_RRT_H

#include <yaml-cpp/yaml.h>

struct DecoupledRRTConfig {
    double time_limit     = 60.0;
    double goal_threshold = 0.5;
    int    seed           = -1;
};

// Planner that plans for each robot individually in priority order using RRT,
// coordinating via the OMPL multi-robot PP (Prioritized Planning) framework.
//
// env YAML schema (same as CoupledRRTPlanner):
//   environment:
//     min: [x, y]
//     max: [x, y]
//     obstacles:  (optional)
//       - type: box
//         size: [w, h]
//         center: [x, y]
//   robots:
//     - type: <robot_type>
//       start: [x, y, yaw]
//       goal:  [x, y, yaw]
//
// Returns a YAML::Node with:
//   solved:        bool
//   planning_time: double (seconds)
//   result:        (present only when solved)
//     - states: [[x, y, yaw], ...]   (one entry per robot)
class DecoupledRRTPlanner
{
public:
    explicit DecoupledRRTPlanner(const DecoupledRRTConfig& config = {});

    YAML::Node plan(const YAML::Node& env);

private:
    DecoupledRRTConfig config_;
};

#endif // DECOUPLED_RRT_H
