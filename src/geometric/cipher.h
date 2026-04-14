#ifndef CIPHER_GEOMETRIC_H
#define CIPHER_GEOMETRIC_H

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/SpaceInformation.h>
#include <fcl/fcl.h>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>

#include "decomposition.h"
#include "robots.h"
// #include "coupled_rrt.h"

namespace ob = ompl::base;
namespace oc = ompl::control;

struct MAPFConfig {
    std::string method = "decoupled";  // Options: "decoupled", "astar", "cbs"
    int region_capacity = 1;           // Robots per vertex/edge (for CBS)
    double max_obstacle_volume_percent = 0.5;  // Maximum obstacle volume in a region (0.0 to 1.0)
};

struct CollisionResolutionConfig {
    // Hierarchical refinement parameters
    int max_refinement_levels = 3;                // K: refinement levels per expansion (0=skip)
    double decomposition_subdivision_factor = 2.0; // Subdivision multiplier per refinement level

    // Expansion parameters
    int max_expansion_layers = -1;                // Max expansion layers (-1 = auto-detect based on grid)
                                                  // Auto-detect: ceil(sqrt(num_regions)/2)

    // Cycle detection: after this many collisions for the same robot pair,
    // increase the minimum expansion layer by 1 (0 = disable escalation)
    int escalation_frequency = 3;

    // Composite planner fallback
    int max_composite_attempts = 1;               // Composite planner attempts (0=skip)
    bool composite_uses_full_problem = true;      // If true, composite plans ALL robots from original starts/goals

    // Re-checking behavior
    bool recheck_from_prior_segment = false;      // If true, start collision re-checking from prior segment start
};

struct CipherConfig {
    int decomposition_region_length = 1;
    std::vector<int> decomposition_resolution = {10, 10, 1};  // Grid cells in [x, y, z]
    double planning_time_limit = 60.0;
    double max_total_time = 0.0;  // Maximum total planning time in seconds (0 = no limit)
    int seed = -1;  // Random seed (-1 for random)

    // MAPF configuration
    MAPFConfig mapf_config;

    // Coupled RRT config for composite planner
    CoupledRRTConfig coupled_rrt_config;

    // Guided planner configuration
    std::string guided_planner_method = "syclop_rrt";
    mr_syclop::GuidedPlannerConfig guided_planner_config;

    // Segmentation configuration
    int segment_timesteps = 30;  // Number of timesteps per segment

    // Collision resolution configuration
    CollisionResolutionConfig collision_resolution_config;
};

#endif //CIPHER_GEOMETRIC_H