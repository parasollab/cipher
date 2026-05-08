#ifndef KINO_COUPLED_RRT_H
#define KINO_COUPLED_RRT_H

#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/StateSpace.h>
#include <ompl/control/PathControl.h>
#include <ompl/control/StatePropagator.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <fcl/fcl.h>

#include <memory>
#include <string>
#include <vector>

#include "robots.h"
#include "fclStateValidityChecker.hpp"
#include "robotStatePropagator.hpp"

namespace ob = ompl::base;
namespace oc = ompl::control;

struct KinoCoupledRRTConfig {
    double time_limit = 60.0;
    double goal_threshold = 1.5;
    double goal_bias = 0.05;
    int seed = -1;
    double propagation_step_size = 0.1;
    int control_duration_min = 1;
    int control_duration_max = 10;
};

KinoCoupledRRTConfig loadConfigFromYAML(const std::string& configFile);

class CompoundStateValidityChecker : public ob::StateValidityChecker
{
public:
    CompoundStateValidityChecker(
        const ob::SpaceInformationPtr& si,
        const std::shared_ptr<fcl::BroadPhaseCollisionManagerf>& col_mng_environment,
        const std::vector<std::shared_ptr<Robot>>& robots)
        : ob::StateValidityChecker(si),
          col_mng_environment_(col_mng_environment),
          robots_(robots) {}

    bool isValid(const ob::State* state) const override
    {
        if (!si_->satisfiesBounds(state)) return false;

        auto compound = state->as<ob::CompoundState>();

        for (size_t i = 0; i < robots_.size(); ++i) {
            for (size_t part = 0; part < robots_[i]->numParts(); ++part) {
                const auto& transform = robots_[i]->getTransform(compound->components[i], part);

                fcl::CollisionObjectf robot_co(robots_[i]->getCollisionGeometry(part));
                robot_co.setTranslation(transform.translation());
                robot_co.setRotation(transform.rotation());
                robot_co.computeAABB();

                fcl::DefaultCollisionData<float> collision_data;
                col_mng_environment_->collide(&robot_co, &collision_data,
                    fcl::DefaultCollisionFunction<float>);

                if (collision_data.result.isCollision()) return false;
            }
        }

        for (size_t i = 0; i < robots_.size(); ++i) {
            for (size_t j = i + 1; j < robots_.size(); ++j) {
                if (checkRobotRobotCollision(
                        compound->components[i], compound->components[j],
                        robots_[i], robots_[j]))
                    return false;
            }
        }

        return true;
    }

private:
    bool checkRobotRobotCollision(
        const ob::State* state_i, const ob::State* state_j,
        const std::shared_ptr<Robot>& robot_i,
        const std::shared_ptr<Robot>& robot_j) const
    {
        for (size_t pi = 0; pi < robot_i->numParts(); ++pi) {
            for (size_t pj = 0; pj < robot_j->numParts(); ++pj) {
                const auto& ti = robot_i->getTransform(state_i, pi);
                const auto& tj = robot_j->getTransform(state_j, pj);

                fcl::CollisionObjectf co_i(robot_i->getCollisionGeometry(pi));
                co_i.setTranslation(ti.translation());
                co_i.setRotation(ti.rotation());
                co_i.computeAABB();

                fcl::CollisionObjectf co_j(robot_j->getCollisionGeometry(pj));
                co_j.setTranslation(tj.translation());
                co_j.setRotation(tj.rotation());
                co_j.computeAABB();

                fcl::CollisionRequestf request;
                fcl::CollisionResultf result;
                fcl::collide(&co_i, &co_j, request, result);
                if (result.isCollision()) return true;
            }
        }
        return false;
    }

    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng_environment_;
    std::vector<std::shared_ptr<Robot>> robots_;
};


class CompoundGoalCondition : public ob::GoalSampleableRegion
{
public:
    CompoundGoalCondition(
        const ob::SpaceInformationPtr& si,
        const std::vector<ob::State*>& goal_states,
        double threshold)
        : ob::GoalSampleableRegion(si), goal_states_(goal_states)
    {
        threshold_ = threshold;
    }

    double distanceGoal(const ob::State* st) const override
    {
        auto compound = st->as<ob::CompoundState>();
        auto css = si_->getStateSpace()->as<ob::CompoundStateSpace>();
        double max_dist = 0.0;
        for (size_t i = 0; i < goal_states_.size(); ++i) {
            double dist = css->getSubspace(i)->distance(
                compound->components[i], goal_states_[i]);
            if (dist > max_dist) max_dist = dist;
        }
        return max_dist;
    }

    void sampleGoal(ob::State* st) const override
    {
        auto* compound = st->as<ob::CompoundState>();
        auto css = si_->getStateSpace()->as<ob::CompoundStateSpace>();
        for (size_t i = 0; i < goal_states_.size(); ++i) {
            css->getSubspace(i)->copyState(compound->components[i], goal_states_[i]);
        }
    }

    unsigned int maxSampleCount() const override { return 1; }

private:
    std::vector<ob::State*> goal_states_;
};


class CompoundStatePropagator : public oc::StatePropagator
{
public:
    CompoundStatePropagator(
        const oc::SpaceInformationPtr& si,
        const std::vector<std::shared_ptr<Robot>>& robots)
        : oc::StatePropagator(si), robots_(robots) {}

    void propagate(
        const ob::State* state,
        const oc::Control* control,
        double duration,
        ob::State* result) const override
    {
        auto startTyped   = state->as<ob::CompoundState>();
        auto controlTyped = control->as<oc::CompoundControlSpace::ControlType>();
        auto resultTyped  = result->as<ob::CompoundState>();
        for (size_t i = 0; i < robots_.size(); ++i) {
            robots_[i]->propagate(
                startTyped->components[i],
                controlTyped->components[i],
                duration,
                resultTyped->components[i]);
        }
    }

    bool canPropagateBackward() const override { return false; }
    bool canSteer()             const override { return false; }

private:
    std::vector<std::shared_ptr<Robot>> robots_;
};

#endif // KINO_COUPLED_RRT_H
