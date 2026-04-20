#ifndef COUPLED_RRT_H
#define COUPLED_RRT_H

// OMPL base headers
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/goals/GoalRegion.h>
// #include <ompl/base/spaces/CompoundStateSpace.h>

// FCL
#include <fcl/fcl.h>

// YAML
#include <yaml-cpp/yaml.h>

// Standard library
#include <memory>
#include <string>
#include <vector>

// db-CBS robot dynamics
#include "robots.h"

namespace ob = ompl::base;

struct CoupledRRTConfig {
    double time_limit = 60.0;
    double goal_threshold = 0.5;
    double goal_bias = 0.05;
    int seed = -1;  // Random seed (-1 for random)
};

class CompoundStateValidityChecker : public ob::StateValidityChecker
{
public:
    CompoundStateValidityChecker(
        const ob::SpaceInformationPtr& si,
        const std::shared_ptr<fcl::BroadPhaseCollisionManagerf>& col_mng_environment,
        const std::vector<std::shared_ptr<Robot>>& robots);

    bool isValid(const ob::State* state) const override;

private:
    bool checkRobotRobotCollision(
        const ob::State* state_i,
        const ob::State* state_j,
        const std::shared_ptr<Robot>& robot_i,
        const std::shared_ptr<Robot>& robot_j) const;

    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng_environment_;
    std::vector<std::shared_ptr<Robot>> robots_;
};


// Goal condition that is satisfied when all robots are within threshold of their
// respective goals (measured as max per-robot distance in the composite space).
class CompoundGoalCondition : public ob::GoalRegion
{
public:
    CompoundGoalCondition(
        const ob::SpaceInformationPtr& si,
        const ob::CompoundStateSpace* css,
        const std::vector<ob::State*>& goal_states,
        double threshold);

    double distanceGoal(const ob::State* st) const override;

private:
    const ob::CompoundStateSpace* css_;
    std::vector<ob::State*> goal_states_;  // non-owning; lifetime managed by planner
};

CoupledRRTConfig loadConfigFromYAML(const std::string& configFile);


// Planner that jointly plans for all robots in a compound state space using RRT.
// Instantiate with a CoupledRRTConfig and call plan() with a parsed environment.
//
// env YAML schema (same as the standalone binary):
//   environment:
//     min: [x, y]
//     max: [x, y]
//     obstacles:  (optional)
//       - type: box
//         size: [w, h]
//         center: [x, y]
//   robots:
//     - type: <robot_type>
//       start: [...]
//       goal:  [...]
//
// Returns a YAML::Node with:
//   solved:        bool
//   planning_time: double (seconds)
//   result:        (present only when solved)
//     - states: [[x, y, yaw], ...]   (one entry per robot)
class CoupledRRTPlanner
{
public:
    explicit CoupledRRTPlanner(const CoupledRRTConfig& config = {});

    // Solve the multi-robot problem described by env.
    // Returns the output node (solved, planning_time, result).
    YAML::Node plan(const YAML::Node& env);

private:
    CoupledRRTConfig config_;
};


#endif // COUPLED_RRT_H
