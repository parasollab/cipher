#ifndef GUIDED_PLANNER_BASE_H
#define GUIDED_PLANNER_BASE_H

#include <ompl/control/PathControl.h>
#include <ompl/geometric/PathGeometric.h>
#include <utils/decomposition.h>
#include <ompl/base/State.h>
#include <fcl/broadphase/broadphase_collision_manager.h>
#include <memory>
#include <vector>
#include <string>

namespace ob = ompl::base;
namespace oc = ompl::control;
namespace og = ompl::geometric;

class Robot;

struct GuidedPlannerConfig {
    double time_per_robot = 10.0; // seconds
    double goal_threshold = 0.5; // distance threshold to consider goal reached
}

struct GeometricGuidedPlanningResult{
    bool success;
    double planning_time;
    std::shared_ptr<og::PathGeometric> path;
    size_t robot_index;
}

struct ControlGuidedPlanningResult{
    bool success;
    double planning_time;
    std::shared_ptr<oc::PathControl> path;
    size_t robot_index;
}

class GeometricGuidedPlanner {
public:
    GeometricGuidedPlanner(const GuidedPlannerConfig& config) : config_(config) {}
    virtual ~GeometricGuidedPlanner() = default;  

    virtual GeometricGuidedPlanningResult solve(
        std::shared_ptr<::Robot> robot,
        DecompositionImpl& decomposition,
        const ob::State* start,
        const ob::State* goal,
        const std::vector<int>& region_path,
        size_t robot_index) = 0;

    virtual std::string getName() const = 0;

protected:
    GuidedPlannerConfig config_;
};

class ControlGuidedPlanner {
public:
    ControlGuidedPlanner(const GuidedPlannerConfig& config) : config_(config) {}
    virtual ~ControlGuidedPlanner() = default; 

    virtual ControlGuidedPlanningResult solve(
        std::shared_ptr<::Robot> robot,
        DecompositionImpl& decomposition,
        const ob::State* start,
        const ob::State* goal,
        const std::vector<int>& region_path,
        size_t robot_index) = 0;

    virtual std::string getName() const = 0;

protected:
    GuidedPlannerConfig config_;
};

std::unique_ptr<GeometricGuidedPlanner> createGeometricGuidedPlanner(
    const std::string& planner_name,
    const GuidedPlannerConfig& config,
    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> collision_manager
);

std::unique_ptr<ControlGuidedPlanner> createControlGuidedPlanner(
    const std::string& planner_name,
    const GuidedPlannerConfig& config,
    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> collision_manager
);

#endif // GUIDED_PLANNER_BASE_H