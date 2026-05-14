#pragma once

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/goals/GoalRegion.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <fcl/fcl.h>
#include <map>
#include <vector>
#include <utility>
#include <memory>
#include "robots.h"
#include "fclStateValidityChecker.hpp"

namespace ob = ompl::base;

// FCL-based state validity checker that also treats other robots' goal
// positions as static obstacles, and supports inter-robot collision
// queries needed by the multirobot PP planner.
class IndividualStateValidityChecker : public fclStateValidityChecker
{
public:
    IndividualStateValidityChecker(
        ob::SpaceInformationPtr si,
        std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng_environment,
        std::shared_ptr<Robot> robot,
        std::map<const ob::SpaceInformationPtr, std::shared_ptr<Robot>> all_robots,
        std::vector<std::pair<const ob::State*, std::shared_ptr<Robot>>> other_goals)
        : fclStateValidityChecker(si, col_mng_environment, robot, false)
        , all_robots_(std::move(all_robots))
        , other_goals_(std::move(other_goals))
    {}

    bool isValid(const ob::State* state) const override
    {
        if (!fclStateValidityChecker::isValid(state))
            return false;

        const auto& t1 = robot_->getTransform(state, 0);
        fcl::CollisionObjectf co1(robot_->getCollisionGeometry(0));
        co1.setTranslation(t1.translation());
        co1.setRotation(t1.rotation());
        co1.computeAABB();

        for (const auto& [goal_state, other_robot] : other_goals_) {
            const auto& t2 = other_robot->getTransform(goal_state, 0);
            fcl::CollisionObjectf co2(other_robot->getCollisionGeometry(0));
            co2.setTranslation(t2.translation());
            co2.setRotation(t2.rotation());
            co2.computeAABB();

            fcl::CollisionRequestf request;
            fcl::CollisionResultf  result;
            fcl::collide(&co1, &co2, request, result);
            if (result.isCollision())
                return false;
        }
        return true;
    }

    bool areStatesValid(
        const ob::State* state1,
        const std::pair<const ob::SpaceInformationPtr, const ob::State*> state2) const override
    {
        const int part = 0;

        const auto& t1 = robot_->getTransform(state1, part);
        fcl::CollisionObjectf co1(robot_->getCollisionGeometry(part));
        co1.setTranslation(t1.translation());
        co1.setRotation(t1.rotation());
        co1.computeAABB();

        auto iter = all_robots_.find(state2.first);
        if (iter == all_robots_.end())
            return false;

        auto other_robot = iter->second;
        if (!state2.second)
            return true;

        const auto& t2 = other_robot->getTransform(state2.second, part);
        fcl::CollisionObjectf co2(other_robot->getCollisionGeometry(part));
        co2.setTranslation(t2.translation());
        co2.setRotation(t2.rotation());
        co2.computeAABB();

        fcl::CollisionRequestf request;
        fcl::CollisionResultf  result;
        fcl::collide(&co1, &co2, request, result);
        return !result.isCollision();
    }

private:
    std::map<const ob::SpaceInformationPtr, std::shared_ptr<Robot>> all_robots_;
    std::vector<std::pair<const ob::State*, std::shared_ptr<Robot>>> other_goals_;
};

// Goal condition that uses the full SI distance metric.
class IndividualGoalCondition : public ob::GoalRegion
{
public:
    IndividualGoalCondition(
        const ob::SpaceInformationPtr& si,
        const ob::State* goal_state,
        double threshold)
        : ob::GoalRegion(si), goal_state_(goal_state)
    {
        threshold_ = threshold;
    }

    double distanceGoal(const ob::State* st) const override
    {
        const auto* cs = st->as<ob::CompoundState>();
        const auto* cg = goal_state_->as<ob::CompoundState>();
        double dx = cs->as<ob::RealVectorStateSpace::StateType>(0)->values[0]
                - cg->as<ob::RealVectorStateSpace::StateType>(0)->values[0];
        double dy = cs->as<ob::RealVectorStateSpace::StateType>(0)->values[1]
                - cg->as<ob::RealVectorStateSpace::StateType>(0)->values[1];
        return std::sqrt(dx*dx + dy*dy);
        // return si_->distance(st, goal_state_);
    }

private:
    const ob::State* goal_state_;
};

// Returns true if the physical footprints of the two robot states overlap.
inline bool statesOverlap(
    const ob::State* s1, std::shared_ptr<Robot> r1,
    const ob::State* s2, std::shared_ptr<Robot> r2)
{
    const auto& t1 = r1->getTransform(s1, 0);
    fcl::CollisionObjectf co1(r1->getCollisionGeometry(0));
    co1.setTranslation(t1.translation());
    co1.setRotation(t1.rotation());
    co1.computeAABB();

    const auto& t2 = r2->getTransform(s2, 0);
    fcl::CollisionObjectf co2(r2->getCollisionGeometry(0));
    co2.setTranslation(t2.translation());
    co2.setRotation(t2.rotation());
    co2.computeAABB();

    fcl::CollisionRequestf request;
    fcl::CollisionResultf  result;
    fcl::collide(&co1, &co2, request, result);
    return result.isCollision();
}
