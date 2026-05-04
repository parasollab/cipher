/*********************************************************************
* Adaptive Robot Coordination (ARC)
*
* A subproblem-based hybrid approach for multi-robot motion planning
* that dynamically couples and decouples robots to resolve conflicts.
*
* Based on the paper:
* "Adaptive Robot Coordination: A Subproblem-Based Approach for
*  Hybrid Multi-Robot Motion Planning"
* by Solis et al., IEEE RA-L 2024
*********************************************************************/

// OMPL - Base
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/State.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/goals/GoalState.h>

// OMPL - Geometric
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>

// OMPL - Control (needed for propagator interface)
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/StatePropagator.h>

// OMPL - Utilities
#include <ompl/util/RandomNumbers.h>

// FCL
#include <fcl/fcl.h>

// YAML
#include <yaml-cpp/yaml.h>

// Boost program options
#include <boost/program_options.hpp>

// Project headers
#include "robots.h"
#include "fclStateValidityChecker.hpp"

// Standard library
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace po = boost::program_options;

// ============================================================================
// Configuration Structure
// ============================================================================

struct ARCConfig {
    // Overall planning parameters
    double time_limit = 60.0;
    double goal_threshold = 0.5;
    int seed = -1;  // Random seed (-1 for random)

    // Subproblem configuration
    int initial_time_window = 10;          // Initial timesteps before/after conflict
    int time_window_expansion_step = 5;    // Timesteps to expand each iteration
    int max_subproblem_expansions = 5;     // Max times to expand subproblem spatially/temporally
    double spatial_expansion_factor = 1.5; // Factor to expand environment bounds

    // Planning hierarchy timeouts (for Algorithm 2)
    double prioritized_query_timeout = 1.0;
    double decoupled_rrtconnect_timeout = 5.0;
    double composite_rrtconnect_timeout = 10.0;

    // Path discretization (for collision checking)
    int states_per_check = 10;             // Number of states to interpolate between waypoints

    // Termination criteria
    int max_conflicts_resolved = 100;      // Maximum number of conflicts to resolve
};

// ============================================================================
// Conflict Structure
// ============================================================================

struct Conflict {
    size_t robot_i;                        // Index of first robot
    size_t robot_j;                        // Index of second robot
    int timestep;                          // Timestep where conflict occurs
    ob::State* config_i;                   // Configuration of robot i at conflict (allocated)
    ob::State* config_j;                   // Configuration of robot j at conflict (allocated)

    Conflict() : robot_i(0), robot_j(0), timestep(0), config_i(nullptr), config_j(nullptr) {}

    void freeStates(const std::vector<std::shared_ptr<Robot>>& robots);
};

// ============================================================================
// Subproblem Structure
// ============================================================================

struct Subproblem {
    // Robots involved in this subproblem
    std::vector<size_t> robot_indices;

    // Local start and goal states for subproblem (allocated from robot state spaces)
    std::vector<ob::State*> local_starts;
    std::vector<ob::State*> local_goals;

    // Local environment bounds (can be smaller than global environment)
    std::vector<double> env_min;
    std::vector<double> env_max;

    // Time window parameters
    int start_timestep;                    // Timestep where subproblem starts
    int end_timestep;                      // Timestep where subproblem ends
    int time_window;                       // Current time window size (timesteps on each side of conflict)

    // Obstacles in the local region
    std::vector<fcl::CollisionObjectf*> local_obstacles;

    // Track expansion attempts
    int num_expansions;

    Subproblem() : start_timestep(0), end_timestep(0), time_window(0), num_expansions(0) {}

    void freeStates(const std::vector<std::shared_ptr<Robot>>& robots);
};

// ============================================================================
// Subproblem Solution Structure
// ============================================================================

struct SubproblemSolution {
    bool solved;                                           // Whether solution was found
    double planning_time;                                  // Time spent solving
    std::vector<std::shared_ptr<og::PathGeometric>> paths;  // Paths for each robot in subproblem
    std::string method_used;                               // Which method solved it (for debugging)

    SubproblemSolution() : solved(false), planning_time(0.0), method_used("none") {}
};

// ============================================================================
// Discretized Path Structure
// ============================================================================

struct DiscretizedPath {
    std::vector<ob::State*> states;        // States at each timestep (allocated)
    int num_timesteps;                     // Total number of timesteps

    DiscretizedPath() : num_timesteps(0) {}

    void freeStates(const std::shared_ptr<Robot>& robot);
};

// ============================================================================
// ARC Result Structure
// ============================================================================

struct ARCResult {
    bool solved;                                           // Whether exact solution was found
    double planning_time;                                  // Total time spent planning (seconds)
    std::vector<std::shared_ptr<og::PathGeometric>> paths;  // Individual robot paths

    // Statistics
    int num_conflicts_found;                               // Total conflicts detected
    int num_conflicts_resolved;                            // Conflicts successfully resolved
    int num_subproblems_created;                           // Number of subproblems created
    std::vector<int> robots_per_subproblem;                // Number of robots in each subproblem
    std::vector<std::string> methods_used;                 // Which method solved each subproblem

    ARCResult() : solved(false), planning_time(0.0),
                  num_conflicts_found(0), num_conflicts_resolved(0),
                  num_subproblems_created(0) {}
};

// ============================================================================
// Robot Specification and Planning Problem Structures
// ============================================================================

struct RobotSpec {
    std::string type;
    std::vector<double> start;
    std::vector<double> goal;
};

struct PlanningProblem {
    std::vector<double> env_min;
    std::vector<double> env_max;
    std::vector<fcl::CollisionObjectf*> obstacles;
    std::vector<RobotSpec> robots;
};

// ============================================================================
// ARCPlanner Class
// ============================================================================

class ARCPlanner {
public:
    explicit ARCPlanner(const ARCConfig& config);
    ~ARCPlanner();

    // Main planning function (Algorithm 1 from paper)
    ARCResult plan(const PlanningProblem& problem);

private:
    // ========================================================================
    // Algorithm 1: Main ARC Loop
    // ========================================================================

    void computeInitialPaths();
    Conflict* findConflict();
    Subproblem createSubproblem(const Conflict& conflict);
    SubproblemSolution solveSubproblem(Subproblem& subproblem);
    void updateSolution(const SubproblemSolution& solution, const Subproblem& subproblem);

    // ========================================================================
    // Algorithm 2: Subproblem Solving Hierarchy
    // ========================================================================

    SubproblemSolution tryPrioritizedQuery(const Subproblem& subproblem);
    SubproblemSolution tryDecoupledRRTConnect(const Subproblem& subproblem);
    SubproblemSolution tryCompositeRRTConnect(const Subproblem& subproblem);

    // ========================================================================
    // Subproblem Management
    // ========================================================================

    void expandSubproblemSpatially(Subproblem& subproblem);
    void expandSubproblemTemporally(Subproblem& subproblem);
    std::vector<size_t> getConflictingRobots(const Subproblem& subproblem);
    void addRobotsToSubproblem(Subproblem& subproblem, const std::vector<size_t>& new_robots);
    std::vector<fcl::CollisionObjectf*> getObstaclesInRegion(
        const std::vector<double>& region_min,
        const std::vector<double>& region_max) const;

    // ========================================================================
    // Conflict Detection
    // ========================================================================

    void discretizePaths();
    bool checkCollision(size_t robot_i, size_t robot_j, int timestep);
    ob::State* getStateAtTimestep(size_t robot_idx, int timestep) const;
    bool checkFCLCollision(size_t robot_i, const ob::State* state_i,
                          size_t robot_j, const ob::State* state_j) const;

    // ========================================================================
    // Path Integration
    // ========================================================================

    void splicePaths(const SubproblemSolution& solution, const Subproblem& subproblem);
    std::shared_ptr<og::PathGeometric> interpolatePath(
        size_t robot_idx,
        const ob::State* start,
        const ob::State* goal);

    // ========================================================================
    // Helper Methods
    // ========================================================================

    void setupEnvironment(const PlanningProblem& problem);
    void setupRobots(const PlanningProblem& problem);
    void cleanup();
    int getMaxTimesteps() const;

    // ========================================================================
    // Member Variables
    // ========================================================================

    ARCConfig config_;
    PlanningProblem problem_;
    ob::RealVectorBounds position_bounds_;
    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng_environment_;
    std::vector<std::shared_ptr<Robot>> robots_;
    std::vector<ob::State*> start_states_;
    std::vector<ob::State*> goal_states_;
    std::vector<std::shared_ptr<og::PathGeometric>> current_paths_;
    std::vector<DiscretizedPath> discretized_paths_;
    ARCResult result_;
    double planning_start_time_;
};

// ============================================================================
// Helper Structure Implementations
// ============================================================================

void Conflict::freeStates(const std::vector<std::shared_ptr<Robot>>& robots)
{
    if (config_i && robot_i < robots.size()) {
        robots[robot_i]->getSpaceInformation()->getStateSpace()->freeState(config_i);
        config_i = nullptr;
    }
    if (config_j && robot_j < robots.size()) {
        robots[robot_j]->getSpaceInformation()->getStateSpace()->freeState(config_j);
        config_j = nullptr;
    }
}

void Subproblem::freeStates(const std::vector<std::shared_ptr<Robot>>& robots)
{
    for (size_t i = 0; i < robot_indices.size(); ++i) {
        size_t robot_idx = robot_indices[i];
        if (robot_idx < robots.size()) {
            auto si = robots[robot_idx]->getSpaceInformation();
            if (i < local_starts.size() && local_starts[i]) {
                si->getStateSpace()->freeState(local_starts[i]);
            }
            if (i < local_goals.size() && local_goals[i]) {
                si->getStateSpace()->freeState(local_goals[i]);
            }
        }
    }
    local_starts.clear();
    local_goals.clear();
}

void DiscretizedPath::freeStates(const std::shared_ptr<Robot>& robot)
{
    auto si = robot->getSpaceInformation();
    for (auto* state : states) {
        if (state) {
            si->getStateSpace()->freeState(state);
        }
    }
    states.clear();
    num_timesteps = 0;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

ARCPlanner::ARCPlanner(const ARCConfig& config)
    : config_(config), position_bounds_(2), planning_start_time_(0.0)
{
}

ARCPlanner::~ARCPlanner()
{
    cleanup();
}

// ============================================================================
// Main Planning Function (Algorithm 1)
// ============================================================================

ARCResult ARCPlanner::plan(const PlanningProblem& problem)
{
    std::cout << "ARC: Starting Adaptive Robot Coordination planner..." << std::endl;

    // Start timing
    auto start_time = std::chrono::steady_clock::now();
    planning_start_time_ = std::chrono::duration<double>(
        start_time.time_since_epoch()).count();

    // Store problem
    problem_ = problem;

    // Initialize result
    result_ = ARCResult();
    result_.solved = false;

    try {
        // Setup environment and robots
        setupEnvironment(problem);
        setupRobots(problem);

        std::cout << "ARC: Planning for " << robots_.size() << " robots" << std::endl;

        // Step 1: Compute initial individual paths (Algorithm 1, lines 2-5)
        std::cout << "ARC: Computing initial individual paths..." << std::endl;
        computeInitialPaths();

        // Discretize paths for conflict detection
        discretizePaths();

        // Step 2-6: Iteratively find and resolve conflicts (Algorithm 1, lines 6-16)
        std::cout << "ARC: Checking for conflicts..." << std::endl;
        int iterations = 0;
        while (iterations < config_.max_conflicts_resolved) {
            // Find first conflict
            Conflict* conflict = findConflict();

            if (conflict == nullptr) {
                // No conflicts found - solution is valid!
                std::cout << "ARC: No conflicts found. Solution is valid!" << std::endl;
                result_.solved = true;
                break;
            }

            std::cout << "ARC: Found conflict between robots " << conflict->robot_i
                      << " and " << conflict->robot_j
                      << " at timestep " << conflict->timestep << std::endl;

            result_.num_conflicts_found++;

            // Create subproblem around conflict
            Subproblem subproblem = createSubproblem(*conflict);
            result_.num_subproblems_created++;
            result_.robots_per_subproblem.push_back(subproblem.robot_indices.size());

            // Clean up conflict states
            conflict->freeStates(robots_);
            delete conflict;
            conflict = nullptr;

            // Solve subproblem
            std::cout << "ARC: Solving subproblem with " << subproblem.robot_indices.size()
                      << " robots..." << std::endl;
            SubproblemSolution solution = solveSubproblem(subproblem);

            if (!solution.solved) {
                std::cout << "ARC: Failed to solve subproblem" << std::endl;
                subproblem.freeStates(robots_);
                result_.solved = false;
                break;
            }

            std::cout << "ARC: Subproblem solved using " << solution.method_used
                      << " in " << solution.planning_time << " seconds" << std::endl;

            result_.num_conflicts_resolved++;
            result_.methods_used.push_back(solution.method_used);

            // Update solution
            updateSolution(solution, subproblem);

            // Clean up subproblem
            subproblem.freeStates(robots_);

            // Re-discretize paths for next iteration
            discretizePaths();

            iterations++;
        }

        if (iterations >= config_.max_conflicts_resolved && !result_.solved) {
            std::cout << "ARC: Maximum number of conflict resolutions reached" << std::endl;
        }

        // Copy final paths to result
        if (result_.solved) {
            result_.paths = current_paths_;
        }

    } catch (const std::exception& e) {
        std::cerr << "ARC: Planning failed with exception: " << e.what() << std::endl;
        result_.solved = false;
    }

    // End timing
    auto end_time = std::chrono::steady_clock::now();
    result_.planning_time = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "ARC: Planning completed in " << result_.planning_time << " seconds" << std::endl;
    std::cout << "ARC: Conflicts found: " << result_.num_conflicts_found << std::endl;
    std::cout << "ARC: Conflicts resolved: " << result_.num_conflicts_resolved << std::endl;
    std::cout << "ARC: Subproblems created: " << result_.num_subproblems_created << std::endl;

    return result_;
}

// ============================================================================
// Algorithm 1: Main ARC Loop Implementation
// ============================================================================

void ARCPlanner::computeInitialPaths()
{
    current_paths_.clear();
    current_paths_.resize(robots_.size());

    for (size_t i = 0; i < robots_.size(); ++i) {
        std::cout << "  Planning for robot " << i << "..." << std::endl;

        auto robot = robots_[i];
        auto si = robot->getSpaceInformation();

        // Create problem definition
        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        pdef->addStartState(start_states_[i]);

        // Create goal state
        auto goal = std::make_shared<ob::GoalState>(si);
        goal->setState(goal_states_[i]);
        goal->setThreshold(config_.goal_threshold);
        pdef->setGoal(goal);

        // Create geometric planner
        auto planner = std::make_shared<og::RRTConnect>(si);
        planner->setProblemDefinition(pdef);
        planner->setup();

        // Solve
        double time_per_robot = config_.time_limit / robots_.size();
        ob::PlannerStatus solved = planner->solve(ob::timedPlannerTerminationCondition(time_per_robot));

        if (solved) {
            auto path = std::dynamic_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
            if (path) {
                path->interpolate();
                current_paths_[i] = std::make_shared<og::PathGeometric>(*path);
                std::cout << "    Robot " << i << " path found: "
                          << path->getStateCount() << " states" << std::endl;
            } else {
                throw std::runtime_error("Failed to get path for robot " + std::to_string(i));
            }
        } else {
            throw std::runtime_error("Failed to find path for robot " + std::to_string(i));
        }
    }
}

Conflict* ARCPlanner::findConflict()
{
    // Find maximum timesteps
    int max_timesteps = getMaxTimesteps();

    // Check each timestep
    for (int t = 0; t < max_timesteps; ++t) {
        // Check all pairs of robots
        for (size_t i = 0; i < robots_.size(); ++i) {
            for (size_t j = i + 1; j < robots_.size(); ++j) {
                if (checkCollision(i, j, t)) {
                    // Found a conflict - create conflict object
                    Conflict* conflict = new Conflict();
                    conflict->robot_i = i;
                    conflict->robot_j = j;
                    conflict->timestep = t;

                    // Allocate and copy states at conflict
                    auto si_i = robots_[i]->getSpaceInformation();
                    auto si_j = robots_[j]->getSpaceInformation();

                    conflict->config_i = si_i->getStateSpace()->allocState();
                    conflict->config_j = si_j->getStateSpace()->allocState();

                    si_i->copyState(conflict->config_i, getStateAtTimestep(i, t));
                    si_j->copyState(conflict->config_j, getStateAtTimestep(j, t));

                    return conflict;
                }
            }
        }
    }

    return nullptr;  // No conflict found
}

Subproblem ARCPlanner::createSubproblem(const Conflict& conflict)
{
    Subproblem subproblem;

    // Start with two conflicting robots
    subproblem.robot_indices = {conflict.robot_i, conflict.robot_j};
    subproblem.time_window = config_.initial_time_window;
    subproblem.num_expansions = 0;

    // Calculate start and end timesteps for subproblem
    subproblem.start_timestep = std::max(0, conflict.timestep - subproblem.time_window);
    subproblem.end_timestep = std::min(getMaxTimesteps(),
                                       conflict.timestep + subproblem.time_window);

    std::cout << "  Subproblem time window: [" << subproblem.start_timestep
              << ", " << subproblem.end_timestep << "]" << std::endl;

    // Extract local start and goal states
    for (size_t robot_idx : subproblem.robot_indices) {
        auto si = robots_[robot_idx]->getSpaceInformation();

        // Allocate states
        ob::State* local_start = si->getStateSpace()->allocState();
        ob::State* local_goal = si->getStateSpace()->allocState();

        // Copy from discretized paths
        si->copyState(local_start, getStateAtTimestep(robot_idx, subproblem.start_timestep));
        si->copyState(local_goal, getStateAtTimestep(robot_idx, subproblem.end_timestep));

        subproblem.local_starts.push_back(local_start);
        subproblem.local_goals.push_back(local_goal);
    }

    // Calculate local environment bounds
    // For now, use a bounding box around the start/goal states
    subproblem.env_min = problem_.env_min;
    subproblem.env_max = problem_.env_max;

    // TODO: Compute tighter bounds based on states
    // This would involve finding min/max positions from local_starts and local_goals

    // Get obstacles in region
    subproblem.local_obstacles = getObstaclesInRegion(
        subproblem.env_min, subproblem.env_max);

    std::cout << "  Subproblem has " << subproblem.local_obstacles.size()
              << " obstacles in local region" << std::endl;

    return subproblem;
}

SubproblemSolution ARCPlanner::solveSubproblem(Subproblem& subproblem)
{
    SubproblemSolution solution;

    // Try hierarchy of methods (Algorithm 2)
    // Level 1: Prioritized Query
    std::cout << "  Trying prioritized query..." << std::endl;
    solution = tryPrioritizedQuery(subproblem);
    if (solution.solved) {
        solution.method_used = "PrioritizedQuery";
        return solution;
    }

    // Level 2: Decoupled RRTConnect
    std::cout << "  Trying decoupled RRTConnect..." << std::endl;
    solution = tryDecoupledRRTConnect(subproblem);
    if (solution.solved) {
        solution.method_used = "DecoupledRRTConnect";
        return solution;
    }

    // Level 3: Composite RRTConnect (in composite space)
    std::cout << "  Trying composite RRTConnect..." << std::endl;
    solution = tryCompositeRRTConnect(subproblem);
    if (solution.solved) {
        solution.method_used = "CompositeRRTConnect";
        return solution;
    }

    // If all methods failed, try expanding subproblem
    if (subproblem.num_expansions < config_.max_subproblem_expansions) {
        std::cout << "  All methods failed. Expanding subproblem (attempt "
                  << (subproblem.num_expansions + 1) << "/"
                  << config_.max_subproblem_expansions << ")" << std::endl;

        // Try temporal expansion first
        expandSubproblemTemporally(subproblem);

        // Try again with expanded subproblem
        return solveSubproblem(subproblem);
    }

    std::cout << "  Failed to solve subproblem after all attempts" << std::endl;
    solution.solved = false;
    return solution;
}

void ARCPlanner::updateSolution(const SubproblemSolution& solution,
                                const Subproblem& subproblem)
{
    // Splice the local solution paths into the global paths
    splicePaths(solution, subproblem);
}

// ============================================================================
// Algorithm 2: Subproblem Solving Hierarchy
// ============================================================================

SubproblemSolution ARCPlanner::tryPrioritizedQuery(const Subproblem& subproblem)
{
    auto start_time = std::chrono::steady_clock::now();
    SubproblemSolution solution;
    solution.solved = false;

    // Prioritized query only works for 2-robot conflicts
    if (subproblem.robot_indices.size() != 2) {
        auto end_time = std::chrono::steady_clock::now();
        solution.planning_time = std::chrono::duration<double>(end_time - start_time).count();
        return solution;
    }

    // Helper function to create a path with waiting at the start
    auto createWaitingPath = [&](size_t robot_idx, const ob::State* start, const ob::State* goal,
                                  int wait_timesteps) -> std::shared_ptr<og::PathGeometric> {
        auto robot = robots_[robot_idx];
        auto si = robot->getSpaceInformation();
        auto path = std::make_shared<og::PathGeometric>(si);

        // Add waiting states at the start (duplicate the start state)
        for (int t = 0; t < wait_timesteps; ++t) {
            ob::State* wait_state = si->allocState();
            si->copyState(wait_state, start);
            path->append(wait_state);
        }

        // Plan from start to goal
        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        pdef->setStartAndGoalStates(start, goal, config_.goal_threshold);

        // Use fast RRTConnect for quick queries
        auto planner = std::make_shared<og::RRTConnect>(si);
        planner->setProblemDefinition(pdef);

        ob::PlannerStatus solved = planner->solve(
            ob::timedPlannerTerminationCondition(config_.prioritized_query_timeout));

        if (solved != ob::PlannerStatus::EXACT_SOLUTION) {
            return nullptr;
        }

        auto planned_path = std::dynamic_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
        planned_path->interpolate();

        // Append planned path to waiting path
        for (size_t i = 0; i < planned_path->getStateCount(); ++i) {
            ob::State* state_copy = si->allocState();
            si->copyState(state_copy, planned_path->getState(i));
            path->append(state_copy);
        }

        return path;
    };

    // Helper function to check if two paths are collision-free
    auto checkPathsCollisionFree = [&](const std::shared_ptr<og::PathGeometric>& path1,
                                        const std::shared_ptr<og::PathGeometric>& path2,
                                        size_t robot_idx1, size_t robot_idx2) -> bool {
        size_t max_timesteps = std::max(path1->getStateCount(), path2->getStateCount());

        for (size_t t = 0; t < max_timesteps; ++t) {
            // Get states (use last state if path is shorter)
            const ob::State* state1 = (t < path1->getStateCount()) ?
                path1->getState(t) : path1->getState(path1->getStateCount() - 1);
            const ob::State* state2 = (t < path2->getStateCount()) ?
                path2->getState(t) : path2->getState(path2->getStateCount() - 1);

            if (checkFCLCollision(robot_idx1, state1, robot_idx2, state2)) {
                return false;
            }
        }
        return true;
    };

    size_t robot_i = subproblem.robot_indices[0];
    size_t robot_j = subproblem.robot_indices[1];

    // Calculate time window size (how long to wait)
    int wait_duration = subproblem.time_window / 2;

    // Strategy 1: Robot i waits, robot j proceeds normally
    {
        auto path_i_wait = createWaitingPath(robot_i, subproblem.local_starts[0],
                                            subproblem.local_goals[0], wait_duration);
        auto path_j_normal = createWaitingPath(robot_j, subproblem.local_starts[1],
                                              subproblem.local_goals[1], 0);

        if (path_i_wait && path_j_normal &&
            checkPathsCollisionFree(path_i_wait, path_j_normal, robot_i, robot_j)) {
            solution.paths.resize(2);
            solution.paths[0] = path_i_wait;
            solution.paths[1] = path_j_normal;
            solution.solved = true;
            auto end_time = std::chrono::steady_clock::now();
            solution.planning_time = std::chrono::duration<double>(end_time - start_time).count();
            return solution;
        }
    }

    // Strategy 2: Robot j waits, robot i proceeds normally
    {
        auto path_i_normal = createWaitingPath(robot_i, subproblem.local_starts[0],
                                              subproblem.local_goals[0], 0);
        auto path_j_wait = createWaitingPath(robot_j, subproblem.local_starts[1],
                                            subproblem.local_goals[1], wait_duration);

        if (path_i_normal && path_j_wait &&
            checkPathsCollisionFree(path_i_normal, path_j_wait, robot_i, robot_j)) {
            solution.paths.resize(2);
            solution.paths[0] = path_i_normal;
            solution.paths[1] = path_j_wait;
            solution.solved = true;
            auto end_time = std::chrono::steady_clock::now();
            solution.planning_time = std::chrono::duration<double>(end_time - start_time).count();
            return solution;
        }
    }

    // Both strategies failed
    solution.solved = false;
    auto end_time = std::chrono::steady_clock::now();
    solution.planning_time = std::chrono::duration<double>(end_time - start_time).count();
    return solution;
}

SubproblemSolution ARCPlanner::tryDecoupledRRTConnect(const Subproblem& subproblem)
{
    auto start_time = std::chrono::steady_clock::now();
    SubproblemSolution solution;
    solution.solved = false;

    // Helper class: Validity checker that includes dynamic obstacles from other robots
    class DecoupledValidityChecker : public ob::StateValidityChecker {
    public:
        DecoupledValidityChecker(
            const ob::SpaceInformationPtr& si,
            const std::shared_ptr<fcl::BroadPhaseCollisionManagerf>& col_mng,
            const std::shared_ptr<Robot>& robot,
            const std::vector<std::shared_ptr<og::PathGeometric>>& other_paths,
            const std::vector<std::shared_ptr<Robot>>& other_robots)
            : ob::StateValidityChecker(si),
              col_mng_(col_mng),
              robot_(robot),
              other_paths_(other_paths),
              other_robots_(other_robots) {}

        bool isValid(const ob::State* state) const override {
            if (!si_->satisfiesBounds(state)) {
                return false;
            }

            // Check collision with static obstacles
            for (size_t part = 0; part < robot_->numParts(); ++part) {
                const auto& transform = robot_->getTransform(state, part);

                fcl::CollisionObjectf robot_co(robot_->getCollisionGeometry(part));
                robot_co.setTranslation(transform.translation());
                robot_co.setRotation(transform.rotation());
                robot_co.computeAABB();

                fcl::DefaultCollisionData<float> collision_data;
                col_mng_->collide(&robot_co, &collision_data,
                    fcl::DefaultCollisionFunction<float>);

                if (collision_data.result.isCollision()) {
                    return false;
                }
            }

            // Check collision with other robots' paths (at all timesteps)
            for (size_t other_idx = 0; other_idx < other_paths_.size(); ++other_idx) {
                const auto& other_path = other_paths_[other_idx];
                const auto& other_robot = other_robots_[other_idx];

                // Check against all states in the other robot's path
                for (size_t timestep = 0; timestep < other_path->getStateCount(); ++timestep) {
                    const ob::State* other_state = other_path->getState(timestep);

                    // Check collision between current robot and other robot
                    for (size_t part = 0; part < robot_->numParts(); ++part) {
                        for (size_t other_part = 0; other_part < other_robot->numParts(); ++other_part) {
                            const auto& transform = robot_->getTransform(state, part);
                            const auto& other_transform = other_robot->getTransform(other_state, other_part);

                            fcl::CollisionObjectf co1(robot_->getCollisionGeometry(part));
                            co1.setTranslation(transform.translation());
                            co1.setRotation(transform.rotation());
                            co1.computeAABB();

                            fcl::CollisionObjectf co2(other_robot->getCollisionGeometry(other_part));
                            co2.setTranslation(other_transform.translation());
                            co2.setRotation(other_transform.rotation());
                            co2.computeAABB();

                            fcl::CollisionRequestf request;
                            fcl::CollisionResultf result;
                            fcl::collide(&co1, &co2, request, result);

                            if (result.isCollision()) {
                                return false;
                            }
                        }
                    }
                }
            }

            return true;
        }

    private:
        std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng_;
        std::shared_ptr<Robot> robot_;
        std::vector<std::shared_ptr<og::PathGeometric>> other_paths_;
        std::vector<std::shared_ptr<Robot>> other_robots_;
    };

    // Plan for each robot sequentially (prioritized planning)
    solution.paths.resize(subproblem.robot_indices.size());
    std::vector<std::shared_ptr<og::PathGeometric>> planned_paths;
    std::vector<std::shared_ptr<Robot>> planned_robots;

    for (size_t i = 0; i < subproblem.robot_indices.size(); ++i) {
        size_t robot_idx = subproblem.robot_indices[i];
        auto robot = robots_[robot_idx];
        auto si = robot->getSpaceInformation();

        // Create problem definition
        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        pdef->setStartAndGoalStates(subproblem.local_starts[i], subproblem.local_goals[i],
                                   config_.goal_threshold);

        // Create validity checker that includes previously planned robots as dynamic obstacles
        auto validity_checker = std::make_shared<DecoupledValidityChecker>(
            si, col_mng_environment_, robot, planned_paths, planned_robots);
        si->setStateValidityChecker(validity_checker);
        si->setup();

        auto planner = std::make_shared<og::RRTConnect>(si);
        planner->setProblemDefinition(pdef);

        // Solve with timeout
        ob::PlannerStatus solved = planner->solve(
            ob::timedPlannerTerminationCondition(config_.decoupled_rrtconnect_timeout));

        if (solved != ob::PlannerStatus::EXACT_SOLUTION) {
            // If any robot fails to plan, the whole method fails
            solution.solved = false;
            auto end_time = std::chrono::steady_clock::now();
            solution.planning_time = std::chrono::duration<double>(end_time - start_time).count();
            return solution;
        }

        // Extract path
        auto path = std::dynamic_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
        path->interpolate();  // Ensure smooth path
        solution.paths[i] = std::make_shared<og::PathGeometric>(*path);

        // Add this path to the list of planned paths for next robot
        planned_paths.push_back(solution.paths[i]);
        planned_robots.push_back(robot);
    }

    // All robots successfully planned
    solution.solved = true;
    auto end_time = std::chrono::steady_clock::now();
    solution.planning_time = std::chrono::duration<double>(end_time - start_time).count();

    return solution;
}

SubproblemSolution ARCPlanner::tryCompositeRRTConnect(const Subproblem& subproblem)
{
    auto start_time = std::chrono::steady_clock::now();
    SubproblemSolution solution;
    solution.solved = false;

    // Helper class: Compound validity checker for subproblem robots (geometric planning)
    class SubproblemCompoundValidityChecker : public ob::StateValidityChecker {
    public:
        SubproblemCompoundValidityChecker(
            const ob::SpaceInformationPtr& si,
            const std::shared_ptr<fcl::BroadPhaseCollisionManagerf>& col_mng,
            const std::vector<std::shared_ptr<Robot>>& robots,
            const std::vector<size_t>& robot_indices)
            : ob::StateValidityChecker(si),
              col_mng_(col_mng),
              robots_(robots),
              robot_indices_(robot_indices) {}

        bool isValid(const ob::State* state) const override {
            if (!si_->satisfiesBounds(state)) {
                return false;
            }

            auto compound = state->as<ob::CompoundState>();

            // Check each robot against obstacles
            for (size_t i = 0; i < robot_indices_.size(); ++i) {
                size_t robot_idx = robot_indices_[i];
                for (size_t part = 0; part < robots_[robot_idx]->numParts(); ++part) {
                    const auto& transform = robots_[robot_idx]->getTransform(
                        compound->components[i], part);

                    fcl::CollisionObjectf robot_co(robots_[robot_idx]->getCollisionGeometry(part));
                    robot_co.setTranslation(transform.translation());
                    robot_co.setRotation(transform.rotation());
                    robot_co.computeAABB();

                    fcl::DefaultCollisionData<float> collision_data;
                    col_mng_->collide(&robot_co, &collision_data,
                        fcl::DefaultCollisionFunction<float>);

                    if (collision_data.result.isCollision()) {
                        return false;
                    }
                }
            }

            // Check robot-robot collisions within subproblem
            for (size_t i = 0; i < robot_indices_.size(); ++i) {
                for (size_t j = i + 1; j < robot_indices_.size(); ++j) {
                    size_t robot_i = robot_indices_[i];
                    size_t robot_j = robot_indices_[j];

                    if (checkRobotRobotCollision(
                        compound->components[i],
                        compound->components[j],
                        robots_[robot_i],
                        robots_[robot_j])) {
                        return false;
                    }
                }
            }

            return true;
        }

    private:
        bool checkRobotRobotCollision(
            const ob::State* state_i,
            const ob::State* state_j,
            const std::shared_ptr<Robot>& robot_i,
            const std::shared_ptr<Robot>& robot_j) const {
            for (size_t part_i = 0; part_i < robot_i->numParts(); ++part_i) {
                for (size_t part_j = 0; part_j < robot_j->numParts(); ++part_j) {
                    const auto& transform_i = robot_i->getTransform(state_i, part_i);
                    const auto& transform_j = robot_j->getTransform(state_j, part_j);

                    fcl::CollisionObjectf co_i(robot_i->getCollisionGeometry(part_i));
                    co_i.setTranslation(transform_i.translation());
                    co_i.setRotation(transform_i.rotation());
                    co_i.computeAABB();

                    fcl::CollisionObjectf co_j(robot_j->getCollisionGeometry(part_j));
                    co_j.setTranslation(transform_j.translation());
                    co_j.setRotation(transform_j.rotation());
                    co_j.computeAABB();

                    fcl::CollisionRequestf request;
                    fcl::CollisionResultf result;
                    fcl::collide(&co_i, &co_j, request, result);

                    if (result.isCollision()) {
                        return true;
                    }
                }
            }
            return false;
        }

        std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng_;
        std::vector<std::shared_ptr<Robot>> robots_;
        std::vector<size_t> robot_indices_;
    };

    try {
        // Step 1: Create compound state space
        auto compound_state_space = std::make_shared<ob::CompoundStateSpace>();
        for (size_t robot_idx : subproblem.robot_indices) {
            compound_state_space->addSubspace(
                robots_[robot_idx]->getSpaceInformation()->getStateSpace(), 1.0);
        }

        // Step 2: Create compound SpaceInformation (geometric - no control space needed)
        auto compound_si = std::make_shared<ob::SpaceInformation>(compound_state_space);

        // Step 3: Set validity checker
        auto validity_checker = std::make_shared<SubproblemCompoundValidityChecker>(
            compound_si, col_mng_environment_, robots_, subproblem.robot_indices);
        compound_si->setStateValidityChecker(validity_checker);

        // Step 4: Setup SpaceInformation
        compound_si->setup();

        // Step 5: Create problem definition
        auto pdef = std::make_shared<ob::ProblemDefinition>(compound_si);

        // Step 6: Create compound start state
        auto compound_start = compound_si->allocState();
        auto cs = compound_start->as<ob::CompoundState>();
        for (size_t i = 0; i < subproblem.robot_indices.size(); ++i) {
            size_t robot_idx = subproblem.robot_indices[i];
            auto individual_space = robots_[robot_idx]->getSpaceInformation()->getStateSpace();
            individual_space->copyState(cs->components[i], subproblem.local_starts[i]);
        }
        pdef->addStartState(compound_start);

        // Step 7: Create compound goal state
        auto compound_goal = compound_si->allocState();
        auto cg = compound_goal->as<ob::CompoundState>();
        for (size_t i = 0; i < subproblem.robot_indices.size(); ++i) {
            size_t robot_idx = subproblem.robot_indices[i];
            auto individual_space = robots_[robot_idx]->getSpaceInformation()->getStateSpace();
            individual_space->copyState(cg->components[i], subproblem.local_goals[i]);
        }
        auto goal = std::make_shared<ob::GoalState>(compound_si);
        goal->setState(compound_goal);
        goal->setThreshold(config_.goal_threshold);
        pdef->setGoal(goal);

        // Step 8: Create and run RRTConnect planner in composite space
        auto planner = std::make_shared<og::RRTConnect>(compound_si);
        planner->setProblemDefinition(pdef);
        planner->setup();

        ob::PlannerStatus solved = planner->solve(
            ob::timedPlannerTerminationCondition(config_.composite_rrtconnect_timeout));

        if (solved && pdef->getSolutionPath()) {
            // Step 9: Extract compound geometric path
            auto compound_path = std::dynamic_pointer_cast<og::PathGeometric>(
                pdef->getSolutionPath());
            compound_path->interpolate();

            // Step 10: Extract individual robot paths from compound path
            solution.paths.resize(subproblem.robot_indices.size());
            for (size_t i = 0; i < subproblem.robot_indices.size(); ++i) {
                size_t robot_idx = subproblem.robot_indices[i];
                auto robot_si = robots_[robot_idx]->getSpaceInformation();
                auto robot_path = std::make_shared<og::PathGeometric>(robot_si);

                // Extract states for this robot
                for (size_t j = 0; j < compound_path->getStateCount(); ++j) {
                    const auto* compound_state = compound_path->getState(j)->as<ob::CompoundState>();
                    auto* individual_state = robot_si->allocState();
                    robot_si->copyState(individual_state, compound_state->components[i]);
                    robot_path->append(individual_state);
                }

                solution.paths[i] = robot_path;
            }

            solution.solved = true;
        }

        // Clean up allocated states
        compound_si->freeState(compound_start);
        compound_si->freeState(compound_goal);

    } catch (const std::exception& e) {
        std::cerr << "  Composite RRTConnect failed: " << e.what() << std::endl;
        solution.solved = false;
    }

    auto end_time = std::chrono::steady_clock::now();
    solution.planning_time = std::chrono::duration<double>(end_time - start_time).count();

    return solution;
}

// ============================================================================
// Subproblem Management
// ============================================================================

void ARCPlanner::expandSubproblemSpatially(Subproblem& subproblem)
{
    // Expand environment bounds by spatial expansion factor
    for (size_t i = 0; i < subproblem.env_min.size(); ++i) {
        double center = (subproblem.env_min[i] + subproblem.env_max[i]) / 2.0;
        double half_width = (subproblem.env_max[i] - subproblem.env_min[i]) / 2.0;
        double new_half_width = half_width * config_.spatial_expansion_factor;

        subproblem.env_min[i] = std::max(problem_.env_min[i], center - new_half_width);
        subproblem.env_max[i] = std::min(problem_.env_max[i], center + new_half_width);
    }

    // Update obstacles in region
    subproblem.local_obstacles = getObstaclesInRegion(
        subproblem.env_min, subproblem.env_max);

    subproblem.num_expansions++;
}

void ARCPlanner::expandSubproblemTemporally(Subproblem& subproblem)
{
    // Free old start/goal states
    subproblem.freeStates(robots_);

    // Expand time window
    subproblem.time_window += config_.time_window_expansion_step;

    // Recalculate timesteps
    int center_timestep = (subproblem.start_timestep + subproblem.end_timestep) / 2;
    subproblem.start_timestep = std::max(0, center_timestep - subproblem.time_window);
    subproblem.end_timestep = std::min(getMaxTimesteps(),
                                       center_timestep + subproblem.time_window);

    std::cout << "    Expanded time window to [" << subproblem.start_timestep
              << ", " << subproblem.end_timestep << "]" << std::endl;

    // Reallocate local start and goal states
    for (size_t robot_idx : subproblem.robot_indices) {
        auto si = robots_[robot_idx]->getSpaceInformation();

        ob::State* local_start = si->getStateSpace()->allocState();
        ob::State* local_goal = si->getStateSpace()->allocState();

        si->copyState(local_start, getStateAtTimestep(robot_idx, subproblem.start_timestep));
        si->copyState(local_goal, getStateAtTimestep(robot_idx, subproblem.end_timestep));

        subproblem.local_starts.push_back(local_start);
        subproblem.local_goals.push_back(local_goal);
    }

    subproblem.num_expansions++;
}

std::vector<size_t> ARCPlanner::getConflictingRobots(const Subproblem& subproblem)
{
    (void)subproblem;  // Unused in Phase 1
    // TODO: Check if any other robots conflict with current subproblem solution
    return std::vector<size_t>();
}

void ARCPlanner::addRobotsToSubproblem(Subproblem& subproblem,
                                       const std::vector<size_t>& new_robots)
{
    (void)subproblem;  // Unused in Phase 1
    (void)new_robots;  // Unused in Phase 1
    // TODO: Add robots to subproblem
}

std::vector<fcl::CollisionObjectf*> ARCPlanner::getObstaclesInRegion(
    const std::vector<double>& region_min,
    const std::vector<double>& region_max) const
{
    std::vector<fcl::CollisionObjectf*> obstacles_in_region;

    // Create AABB for the region
    fcl::AABBf region_aabb(
        fcl::Vector3f(region_min[0], region_min[1], -1.0f),
        fcl::Vector3f(region_max[0], region_max[1], 1.0f)
    );

    // Check each obstacle for overlap with the region
    for (auto* obstacle : problem_.obstacles) {
        const fcl::AABBf& obstacle_aabb = obstacle->getAABB();

        if (region_aabb.overlap(obstacle_aabb)) {
            obstacles_in_region.push_back(obstacle);
        }
    }

    return obstacles_in_region;
}

// ============================================================================
// Conflict Detection
// ============================================================================

void ARCPlanner::discretizePaths()
{
    std::cout << "  Discretizing paths..." << std::endl;

    // Clear previous discretization
    for (auto& disc_path : discretized_paths_) {
        for (size_t i = 0; i < robots_.size(); ++i) {
            if (!disc_path.states.empty()) {
                disc_path.freeStates(robots_[i]);
                break;
            }
        }
    }
    discretized_paths_.clear();
    discretized_paths_.resize(robots_.size());

    // Discretize each robot's geometric path
    for (size_t i = 0; i < current_paths_.size(); ++i) {
        auto& path = current_paths_[i];
        if (!path) continue;

        auto& disc_path = discretized_paths_[i];
        auto si = robots_[i]->getSpaceInformation();

        // Interpolate path to get uniform discretization
        // Create a copy to interpolate
        og::PathGeometric interpolated_path(*path);
        interpolated_path.interpolate(config_.states_per_check);

        disc_path.num_timesteps = interpolated_path.getStateCount();

        // Copy all states from interpolated path
        for (size_t j = 0; j < interpolated_path.getStateCount(); ++j) {
            ob::State* state_copy = si->allocState();
            si->copyState(state_copy, interpolated_path.getState(j));
            disc_path.states.push_back(state_copy);
        }

        std::cout << "    Robot " << i << ": " << disc_path.num_timesteps
                  << " discretized states" << std::endl;
    }
}

bool ARCPlanner::checkCollision(size_t robot_i, size_t robot_j, int timestep)
{
    // Get states at this timestep
    ob::State* state_i = getStateAtTimestep(robot_i, timestep);
    ob::State* state_j = getStateAtTimestep(robot_j, timestep);

    if (!state_i || !state_j) {
        return false;  // Invalid robot index or empty path
    }

    // Check FCL collision
    return checkFCLCollision(robot_i, state_i, robot_j, state_j);
}

ob::State* ARCPlanner::getStateAtTimestep(size_t robot_idx, int timestep) const
{
    if (robot_idx >= discretized_paths_.size()) {
        return nullptr;
    }

    const auto& disc_path = discretized_paths_[robot_idx];

    if (timestep < 0 || disc_path.states.empty()) {
        return nullptr;
    }

    // Clamp to last state so robots remain at their goal after path ends
    if (timestep >= disc_path.num_timesteps) {
        return disc_path.states.back();
    }

    if (static_cast<size_t>(timestep) >= disc_path.states.size()) {
        return disc_path.states.back();
    }

    return disc_path.states[timestep];
}

bool ARCPlanner::checkFCLCollision(size_t robot_i, const ob::State* state_i,
                                   size_t robot_j, const ob::State* state_j) const
{
    auto r_i = robots_[robot_i];
    auto r_j = robots_[robot_j];

    // Check collision between all parts of both robots
    for (size_t part_i = 0; part_i < r_i->numParts(); ++part_i) {
        for (size_t part_j = 0; part_j < r_j->numParts(); ++part_j) {
            const auto& transform_i = r_i->getTransform(state_i, part_i);
            const auto& transform_j = r_j->getTransform(state_j, part_j);

            fcl::CollisionObjectf co_i(r_i->getCollisionGeometry(part_i));
            co_i.setTranslation(transform_i.translation());
            co_i.setRotation(transform_i.rotation());
            co_i.computeAABB();

            fcl::CollisionObjectf co_j(r_j->getCollisionGeometry(part_j));
            co_j.setTranslation(transform_j.translation());
            co_j.setRotation(transform_j.rotation());
            co_j.computeAABB();

            fcl::CollisionRequestf request;
            fcl::CollisionResultf result;
            fcl::collide(&co_i, &co_j, request, result);

            if (result.isCollision()) {
                return true;
            }
        }
    }

    return false;
}

// ============================================================================
// Path Integration
// ============================================================================

void ARCPlanner::splicePaths(const SubproblemSolution& solution,
                            const Subproblem& subproblem)
{
    const double kDuplicateStateEps = 1e-6;

    for (size_t i = 0; i < subproblem.robot_indices.size(); ++i) {
        size_t robot_idx = subproblem.robot_indices[i];

        if (i >= solution.paths.size() || !solution.paths[i]) {
            continue;
        }

        auto current_path = current_paths_[robot_idx];
        if (!current_path || current_path->getStateCount() == 0) {
            current_paths_[robot_idx] = solution.paths[i];
            continue;
        }

        auto si = robots_[robot_idx]->getSpaceInformation();
        const ob::State* local_start = (i < subproblem.local_starts.size()) ?
            subproblem.local_starts[i] : nullptr;
        const ob::State* local_goal = (i < subproblem.local_goals.size()) ?
            subproblem.local_goals[i] : nullptr;

        auto findClosestIndex = [&](const ob::State* target, size_t begin_idx) -> size_t {
            if (!target) {
                return begin_idx;
            }
            size_t best_idx = begin_idx;
            double best_dist = std::numeric_limits<double>::infinity();
            for (size_t idx = begin_idx; idx < current_path->getStateCount(); ++idx) {
                double dist = si->distance(current_path->getState(idx), target);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = idx;
                }
            }
            return best_idx;
        };

        size_t start_idx = findClosestIndex(local_start, 0);
        size_t end_idx = findClosestIndex(local_goal, start_idx);
        if (end_idx < start_idx) {
            end_idx = start_idx;
        }

        auto new_path = std::make_shared<og::PathGeometric>(si);

        auto appendUniqueCopy = [&](const ob::State* state) {
            if (!state) {
                return;
            }
            if (new_path->getStateCount() > 0) {
                const ob::State* last = new_path->getState(new_path->getStateCount() - 1);
                if (si->distance(last, state) < kDuplicateStateEps) {
                    return;
                }
            }
            ob::State* state_copy = si->allocState();
            si->copyState(state_copy, state);
            new_path->append(state_copy);
        };

        // Prefix: keep original path before the subproblem window.
        for (size_t idx = 0; idx < start_idx; ++idx) {
            appendUniqueCopy(current_path->getState(idx));
        }

        // Insert local solution path (covers the subproblem window).
        for (size_t idx = 0; idx < solution.paths[i]->getStateCount(); ++idx) {
            appendUniqueCopy(solution.paths[i]->getState(idx));
        }

        // Suffix: keep original path after the subproblem window.
        for (size_t idx = end_idx + 1; idx < current_path->getStateCount(); ++idx) {
            appendUniqueCopy(current_path->getState(idx));
        }

        current_paths_[robot_idx] = new_path;
    }
}

std::shared_ptr<og::PathGeometric> ARCPlanner::interpolatePath(
    size_t robot_idx,
    const ob::State* start,
    const ob::State* goal)
{
    auto si = robots_[robot_idx]->getSpaceInformation();
    auto path = std::make_shared<og::PathGeometric>(si);

    // Create a simple straight-line path (geometric interpolation)
    auto start_copy = si->allocState();
    auto goal_copy = si->allocState();
    si->copyState(start_copy, start);
    si->copyState(goal_copy, goal);

    path->append(start_copy);
    path->append(goal_copy);
    path->interpolate();  // OMPL will interpolate between the two states

    return path;
}

// ============================================================================
// Helper Methods
// ============================================================================

// Simple geometric propagator for robots (no dynamics, just state copying)
// This is needed because Robot::getSpaceInformation() returns control::SpaceInformation
// but we're doing geometric planning, so we provide a trivial propagator
namespace {
    namespace oc = ompl::control;

    class GeometricPropagator : public oc::StatePropagator {
    public:
        GeometricPropagator(const oc::SpaceInformationPtr& si)
            : oc::StatePropagator(si) {}

        void propagate(const ob::State* state, const oc::Control* /*control*/,
                      double /*duration*/, ob::State* result) const override {
            // For geometric planning, just copy the state (no dynamics)
            si_->getStateSpace()->copyState(result, state);
        }
    };
}

void ARCPlanner::setupEnvironment(const PlanningProblem& problem)
{
    // Setup position bounds
    position_bounds_.setLow(0, problem.env_min[0]);
    position_bounds_.setLow(1, problem.env_min[1]);
    position_bounds_.setHigh(0, problem.env_max[0]);
    position_bounds_.setHigh(1, problem.env_max[1]);

    // Setup collision manager
    col_mng_environment_ = std::make_shared<fcl::DynamicAABBTreeCollisionManagerf>();
    col_mng_environment_->registerObjects(problem.obstacles);
    col_mng_environment_->setup();
}

void ARCPlanner::setupRobots(const PlanningProblem& problem)
{
    namespace oc = ompl::control;

    robots_.clear();
    start_states_.clear();
    goal_states_.clear();

    for (size_t i = 0; i < problem.robots.size(); ++i) {
        const auto& robot_spec = problem.robots[i];

        // Create robot
        auto robot = create_robot(robot_spec.type, position_bounds_);
        auto si = robot->getSpaceInformation();

        // Setup validity checker
        si->setStateValidityChecker(
            std::make_shared<fclStateValidityChecker>(si, col_mng_environment_, robot));

        // For geometric planning, set a trivial propagator
        // (Robot SpaceInformation is control-based, but we only use geometric planners)
        auto control_si = std::dynamic_pointer_cast<oc::SpaceInformation>(si);
        if (control_si) {
            control_si->setStatePropagator(std::make_shared<GeometricPropagator>(control_si));
        }

        si->setup();

        robots_.push_back(robot);

        // Create start state
        ob::State* start = si->getStateSpace()->allocState();
        si->getStateSpace()->copyFromReals(start, robot_spec.start);
        start_states_.push_back(start);

        // Create goal state
        ob::State* goal = si->getStateSpace()->allocState();
        si->getStateSpace()->copyFromReals(goal, robot_spec.goal);
        goal_states_.push_back(goal);
    }
}

void ARCPlanner::cleanup()
{
    // Free start and goal states
    for (size_t i = 0; i < robots_.size(); ++i) {
        if (i < start_states_.size() && start_states_[i]) {
            robots_[i]->getSpaceInformation()->getStateSpace()->freeState(start_states_[i]);
        }
        if (i < goal_states_.size() && goal_states_[i]) {
            robots_[i]->getSpaceInformation()->getStateSpace()->freeState(goal_states_[i]);
        }
    }

    // Free discretized paths
    for (size_t i = 0; i < discretized_paths_.size(); ++i) {
        if (i < robots_.size()) {
            discretized_paths_[i].freeStates(robots_[i]);
        }
    }

    robots_.clear();
    start_states_.clear();
    goal_states_.clear();
    current_paths_.clear();
    discretized_paths_.clear();
    col_mng_environment_.reset();
}

int ARCPlanner::getMaxTimesteps() const
{
    int max_timesteps = 0;
    for (const auto& disc_path : discretized_paths_) {
        max_timesteps = std::max(max_timesteps, disc_path.num_timesteps);
    }
    return max_timesteps;
}

// ============================================================================
// Helper Functions for YAML Conversion
// ============================================================================

PlanningProblem loadProblemFromYAML(const std::string& inputFile,
                                   std::vector<fcl::CollisionObjectf*>& obstacle_objects)
{
    std::cout << "Loading YAML file: " << inputFile << std::endl;
    YAML::Node env = YAML::LoadFile(inputFile);
    std::cout << "YAML loaded successfully" << std::endl;

    PlanningProblem problem;

    // Parse environment bounds
    const auto& env_min = env["environment"]["min"];
    const auto& env_max = env["environment"]["max"];
    problem.env_min = {env_min[0].as<double>(), env_min[1].as<double>()};
    problem.env_max = {env_max[0].as<double>(), env_max[1].as<double>()};

    // Parse obstacles and create FCL objects
    if (env["environment"]["obstacles"]) {
        for (const auto& obs : env["environment"]["obstacles"]) {
            if (obs["type"].as<std::string>() == "box") {
                const auto& size = obs["size"];
                const auto& center = obs["center"];

                auto box = std::make_shared<fcl::Boxf>(
                    size[0].as<float>(),
                    size[1].as<float>(),
                    1.0f);

                auto* co = new fcl::CollisionObjectf(box);
                co->setTranslation(fcl::Vector3f(
                    center[0].as<float>(),
                    center[1].as<float>(),
                    0.0f));
                co->computeAABB();

                obstacle_objects.push_back(co);
                problem.obstacles.push_back(co);
            }
        }
    }

    // Parse robots
    std::cout << "Creating robots..." << std::endl;
    for (const auto& robot_node : env["robots"]) {
        RobotSpec robot;
        robot.type = robot_node["type"].as<std::string>();
        std::cout << "  Robot type: " << robot.type << std::endl;

        // Parse start state
        const auto& start_vec = robot_node["start"];
        for (size_t i = 0; i < start_vec.size(); ++i) {
            robot.start.push_back(start_vec[i].as<double>());
        }

        // Parse goal state
        const auto& goal_vec = robot_node["goal"];
        for (size_t i = 0; i < goal_vec.size(); ++i) {
            robot.goal.push_back(goal_vec[i].as<double>());
        }

        problem.robots.push_back(robot);
    }

    std::cout << "Planning for " << problem.robots.size() << " robots" << std::endl;

    return problem;
}

ARCConfig loadConfigFromYAML(const std::string& configFile)
{
    ARCConfig config;

    try {
        YAML::Node cfg = YAML::LoadFile(configFile);

        // Basic planning parameters
        if (cfg["timelimit"]) {
            config.time_limit = cfg["timelimit"].as<double>();
        }
        if (cfg["goal_threshold"]) {
            config.goal_threshold = cfg["goal_threshold"].as<double>();
        }

        // Subproblem parameters
        if (cfg["initial_time_window"]) {
            config.initial_time_window = cfg["initial_time_window"].as<int>();
        }
        if (cfg["time_window_expansion_step"]) {
            config.time_window_expansion_step = cfg["time_window_expansion_step"].as<int>();
        }
        if (cfg["max_subproblem_expansions"]) {
            config.max_subproblem_expansions = cfg["max_subproblem_expansions"].as<int>();
        }
        if (cfg["spatial_expansion_factor"]) {
            config.spatial_expansion_factor = cfg["spatial_expansion_factor"].as<double>();
        }

        // Method timeouts
        if (cfg["prioritized_query_timeout"]) {
            config.prioritized_query_timeout = cfg["prioritized_query_timeout"].as<double>();
        }
        if (cfg["decoupled_rrtconnect_timeout"]) {
            config.decoupled_rrtconnect_timeout = cfg["decoupled_rrtconnect_timeout"].as<double>();
        }
        if (cfg["composite_rrtconnect_timeout"]) {
            config.composite_rrtconnect_timeout = cfg["composite_rrtconnect_timeout"].as<double>();
        }

        // Discretization
        if (cfg["states_per_check"]) {
            config.states_per_check = cfg["states_per_check"].as<int>();
        }

        // Termination
        if (cfg["max_conflicts_resolved"]) {
            config.max_conflicts_resolved = cfg["max_conflicts_resolved"].as<int>();
        }

        // Random seed
        if (cfg["seed"]) {
            config.seed = cfg["seed"].as<int>();
        }

    } catch (const YAML::Exception& e) {
        std::cerr << "ERROR loading config file: " << e.what() << std::endl;
        throw;
    }

    return config;
}

void writeResultToYAML(const std::string& outputFile,
                      const ARCResult& result,
                      const PlanningProblem& problem)
{
    YAML::Node output;
    output["solved"] = result.solved;
    output["planning_time"] = result.planning_time;
    output["num_conflicts_found"] = result.num_conflicts_found;
    output["num_conflicts_resolved"] = result.num_conflicts_resolved;
    output["num_subproblems_created"] = result.num_subproblems_created;

    // Write statistics about subproblems
    if (!result.robots_per_subproblem.empty()) {
        YAML::Node subproblem_stats;
        for (size_t i = 0; i < result.robots_per_subproblem.size(); ++i) {
            YAML::Node sp;
            sp["robots"] = result.robots_per_subproblem[i];
            if (i < result.methods_used.size()) {
                sp["method"] = result.methods_used[i];
            }
            subproblem_stats.push_back(sp);
        }
        output["subproblem_stats"] = subproblem_stats;
    }

    if (result.solved && !result.paths.empty()) {
        std::cout << "Exact solution found! Extracting paths..." << std::endl;

        const int num_robots = problem.robots.size();
        std::cout << "Extracting paths for " << num_robots << " robots" << std::endl;

        YAML::Node result_node;
        for (int r = 0; r < num_robots; ++r) {
            YAML::Node robot_data;

            if (static_cast<size_t>(r) >= result.paths.size() || !result.paths[r]) {
                std::cout << "  Robot " << r << ": No path" << std::endl;
                result_node.push_back(robot_data);
                continue;
            }

            auto path = result.paths[r];
            std::cout << "  Robot " << r << ": " << path->getStateCount()
                     << " states" << std::endl;

            // Extract states
            YAML::Node states_node;
            for (size_t i = 0; i < path->getStateCount(); ++i) {
                const auto* state = path->getState(i);

                // Convert state to vector of doubles
                std::vector<double> state_vals;
                auto si = path->getSpaceInformation();
                si->getStateSpace()->copyToReals(state_vals, state);

                YAML::Node state_node;
                for (double val : state_vals) {
                    state_node.push_back(val);
                }
                states_node.push_back(state_node);
            }
            robot_data["states"] = states_node;

            result_node.push_back(robot_data);
        }
        output["result"] = result_node;

        std::cout << "Solution extracted successfully" << std::endl;
    } else {
        std::cout << "No solution found" << std::endl;
    }

    // Write output file
    std::ofstream fout(outputFile);
    fout << output;
    fout.close();
    std::cout << "Output written to " << outputFile << std::endl;
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv)
{
    // Parse command line arguments
    std::string inputFile;
    std::string outputFile;
    std::string configFile;
    double timelimit = 60.0;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Show help message")
        ("input,i", po::value<std::string>(&inputFile)->required(), "Input YAML file")
        ("output,o", po::value<std::string>(&outputFile)->required(), "Output YAML file")
        ("cfg,c", po::value<std::string>(&configFile), "Configuration YAML file")
        ("timelimit,t", po::value<double>(&timelimit)->default_value(60.0), "Time limit in seconds");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }

    // Vector to hold obstacle objects for cleanup
    std::vector<fcl::CollisionObjectf*> obstacle_objects;

    try {
        // Load configuration
        ARCConfig config;

        if (vm.count("cfg")) {
            config = loadConfigFromYAML(configFile);
        }

        // Command line overrides config file only if explicitly provided
        if (vm.count("timelimit")) {
            config.time_limit = timelimit;
        }

        // Set the random seed
        if (config.seed >= 0) {
            std::cout << "Setting random seed to: " << config.seed << std::endl;
            ompl::RNG::setSeed(config.seed);
        } else {
            std::cout << "Using random seed" << std::endl;
        }

        // Print configuration
        std::cout << "\n=== ARC Configuration ===" << std::endl;
        std::cout << "Time limit: " << config.time_limit << " seconds" << std::endl;
        std::cout << "Goal threshold: " << config.goal_threshold << std::endl;
        std::cout << "Initial time window: " << config.initial_time_window << " timesteps" << std::endl;
        std::cout << "Max subproblem expansions: " << config.max_subproblem_expansions << std::endl;
        std::cout << "States per check: " << config.states_per_check << std::endl;
        std::cout << "Seed: " << config.seed << std::endl;
        std::cout << "=========================\n" << std::endl;

        // Load problem from YAML
        PlanningProblem problem = loadProblemFromYAML(inputFile, obstacle_objects);

        // Create planner
        ARCPlanner planner(config);

        // Plan
        std::cout << "\n=== Starting ARC Planner ===" << std::endl;
        ARCResult result = planner.plan(problem);

        std::cout << "\n=== Planning Results ===" << std::endl;
        std::cout << "Planning completed in " << result.planning_time << " seconds" << std::endl;
        std::cout << "Planner status: " << (result.solved ? "Exact solution" : "No solution") << std::endl;
        std::cout << "Conflicts found: " << result.num_conflicts_found << std::endl;
        std::cout << "Conflicts resolved: " << result.num_conflicts_resolved << std::endl;
        std::cout << "Subproblems created: " << result.num_subproblems_created << std::endl;

        if (!result.robots_per_subproblem.empty()) {
            std::cout << "\nSubproblem details:" << std::endl;
            for (size_t i = 0; i < result.robots_per_subproblem.size(); ++i) {
                std::cout << "  Subproblem " << i << ": "
                         << result.robots_per_subproblem[i] << " robots";
                if (i < result.methods_used.size()) {
                    std::cout << ", solved by " << result.methods_used[i];
                }
                std::cout << std::endl;
            }
        }
        std::cout << "=========================\n" << std::endl;

        // Write output
        writeResultToYAML(outputFile, result, problem);

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;

        // Clean up obstacles
        for (auto* co : obstacle_objects) {
            delete co;
        }

        return 1;
    }

    // Clean up obstacles
    for (auto* co : obstacle_objects) {
        delete co;
    }

    return 0;
}
