#ifndef COUPLED_RRT_H
#define COUPLED_RRT_H

// OMPL base headers
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/goals/GoalRegion.h>
#include <ompl/base/spaces/CompoundStateSpace.h>

// FCL
#include <fcl/fcl.h>

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
    std::vector<ob::State*> goal_states_;  // non-owning; lifetime managed by main
};

CoupledRRTConfig loadConfigFromYAML(const std::string& configFile);

#endif // COUPLED_RRT_H
