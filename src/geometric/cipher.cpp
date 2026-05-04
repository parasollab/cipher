#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <ompl/util/Console.h>
#include <ompl/util/RandomNumbers.h>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>

#include "cipher.h"
#include "fclStateValidityChecker.hpp"
#include "robots.h"

#include "utils/decomposition.h"
#include "utils/grid_decomposition.h"
#include "guided/guided_geometric_rrt.h"
#include "mapf/cbs.h"

namespace po = boost::program_options;

// ---------------------------------------------------------------------------
// Visualization helpers
// ---------------------------------------------------------------------------
static YAML::Node makeVec3(double x, double y, double z = 0.0)
{
    YAML::Node n;
    n.push_back(x); n.push_back(y); n.push_back(z);
    return n;
}

void CipherGeometricPlanner::vizWriteFile() const
{
    YAML::Node doc;
    doc["header"] = viz_header_;
    YAML::Node evs;
    for (const auto& e : viz_events_) evs.push_back(e);
    doc["events"] = evs;
    std::ofstream fout(viz_file_);
    fout << doc;
}

void CipherGeometricPlanner::initVizHeader()
{
    if (!do_viz_) return;

    viz_header_["dimensions"] = 2;

    // Robots
    YAML::Node viz_robots;
    for (size_t r = 0; r < robots_.size(); ++r) {
        auto si = robots_[r]->getSpaceInformation();
        std::vector<double> s_reals, g_reals;
        si->getStateSpace()->copyToReals(s_reals, start_states_[r]);
        si->getStateSpace()->copyToReals(g_reals, goal_states_[r]);

        YAML::Node rn;
        rn["id"] = "r" + std::to_string(r);
        rn["dynamics"] = robot_types_[r];
        YAML::Node geom;
        geom["type"] = "sphere";
        geom["radius"] = 0.15;
        rn["geometry"] = geom;
        rn["start"] = makeVec3(s_reals[0], s_reals[1]);
        rn["goal"]  = makeVec3(g_reals[0], g_reals[1]);
        viz_robots.push_back(rn);
    }
    viz_header_["robots"] = viz_robots;

    // Grid cells
    YAML::Node viz_grid;
    YAML::Node viz_cells;
    const int num_regions = decomp_->getNumRegions();
    for (int rid = 0; rid < num_regions; ++rid) {
        const auto& rb = decomp_->getCellBounds(rid);
        YAML::Node cn;
        cn["id"] = "c" + std::to_string(rid);
        YAML::Node bounds;
        bounds["min"] = makeVec3(rb.low[0], rb.low[1]);
        bounds["max"] = makeVec3(rb.high[0], rb.high[1]);
        cn["bounds"] = bounds;
        viz_cells.push_back(cn);
    }
    viz_grid["cells"] = viz_cells;
    viz_header_["grid"] = viz_grid;

    vizWriteFile();
    std::cout << "[viz] Header written (" << num_regions << " cells) to " << viz_file_ << std::endl;
}

void CipherGeometricPlanner::vizEmitCoupledPlanning(
    const std::vector<size_t>& robot_indices,
    const std::vector<int>& cell_ids)
{
    if (!do_viz_) return;
    YAML::Node ev;
    ev["type"] = "coupled_planning";
    YAML::Node group;
    YAML::Node robots_node;
    for (size_t r : robot_indices) robots_node.push_back("r" + std::to_string(r));
    group["robots"] = robots_node;
    YAML::Node cells_node;
    for (int c : cell_ids) cells_node.push_back("c" + std::to_string(c));
    group["cells"] = cells_node;
    YAML::Node groups;
    groups.push_back(group);
    ev["groups"] = groups;
    viz_events_.push_back(ev);
    vizWriteFile();
    std::cout << "[viz] coupled_planning event: " << robot_indices.size()
              << " robots, " << cell_ids.size() << " cells" << std::endl;
}

void CipherGeometricPlanner::vizEmitGridUpdate(
    const std::vector<std::string>& removed_viz_ids,
    const std::vector<std::tuple<std::string,
                                 std::vector<double>,
                                 std::vector<double>>>& new_cells)
{
    if (!do_viz_) return;
    YAML::Node ev;
    ev["type"] = "grid_update";
    if (!removed_viz_ids.empty()) {
        YAML::Node removed;
        for (const std::string& id : removed_viz_ids) removed.push_back(id);
        ev["removed"] = removed;
    }
    YAML::Node cells_node;
    for (const auto& [id, bmin, bmax] : new_cells) {
        YAML::Node cn;
        cn["id"] = id;
        YAML::Node bounds;
        bounds["min"] = makeVec3(bmin[0], bmin[1]);
        bounds["max"] = makeVec3(bmax[0], bmax[1]);
        cn["bounds"] = bounds;
        cells_node.push_back(cn);
    }
    ev["cells"] = cells_node;
    viz_events_.push_back(ev);
    vizWriteFile();
    std::cout << "[viz] grid_update event: " << removed_viz_ids.size()
              << " removed, " << new_cells.size() << " added" << std::endl;
}

CipherGeometricPlanner::CipherGeometricPlanner(const CipherGeometricConfig& config)
    : config_(config), decomp_(nullptr), workspace_bounds_(3) {
}

CipherGeometricPlanner::~CipherGeometricPlanner() = default;

bool CipherGeometricPlanner::isTimeoutExceeded() const {
    if (config_.max_total_time <= 0.0) return false;
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - planning_start_time_).count();
    return elapsed >= config_.max_total_time;
}

void CipherGeometricPlanner::loadProblem(
        const std::vector<std::string>& robot_types,
        const std::vector<std::vector<double>>& starts,
        const std::vector<std::vector<double>>& goals,
        const std::vector<fcl::CollisionObjectf*>& obstacles,
        const std::vector<double>& env_min,
        const std::vector<double>& env_max) {
    // Clean up any previous problem
    cleanup();

    // Store problem data
    robot_types_ = robot_types;
    starts_ = starts;
    goals_ = goals;
    obstacles_ = obstacles;
    env_min_ = env_min;
    env_max_ = env_max;

    // Setup workspace bounds
    workspace_bounds_.setLow(0, env_min[0]);
    workspace_bounds_.setLow(1, env_min[1]);
    workspace_bounds_.setHigh(0, env_max[0]);
    workspace_bounds_.setHigh(1, env_max[1]);

    // Setup decomposition, collision manager, and robots
    setupCollisionManager();
    setupRobots();
    setupDecomposition();

    // Build and write visualization header now that robots + decomp are ready
    initVizHeader();

    problem_loaded_ = true;
}

CipherGeometricResult CipherGeometricPlanner::plan() {
    std::cout << "Planning with CipherGeometricPlanner..." << std::endl;

    // Check if problem is loaded
    if (!problem_loaded_) {
        throw std::runtime_error("Problem not loaded. Call loadProblem() first.");
    }
    
    CipherGeometricResult result;
    planning_start_time_ = std::chrono::steady_clock::now();

    try {
        // Phase 1: Compute high-level paths over decomposition
        std::cout << "[Phase 1] Computing high-level paths..." << std::endl;
        computeHighLevelPaths();
        std::cout << "[Phase 1] High-level paths computed" << std::endl;

        if (isTimeoutExceeded()) {
            std::cerr << "Planning timeout exceeded after computing high-level paths" << std::endl;
            result.success = false;
            result.failure_reason = "timeout_high_level_paths";
            auto end_time = std::chrono::steady_clock::now();
            result.planning_time = std::chrono::duration<double>(end_time - planning_start_time_).count();
            result.resolution_stats = resolution_stats_;
            return result;
        }

        // Phase 2: Compute guided paths for each robot
        std::cout << "[Phase 2] Computing guided paths..." << std::endl;
        computeGuidedPaths();
        std::cout << "[Phase 2] Guided paths computed" << std::endl;

        if (isTimeoutExceeded()) {
            std::cerr << "Planning timeout exceeded after computing guided paths" << std::endl;
            result.success = false;
            result.failure_reason = "timeout_guided_paths";
            auto end_time = std::chrono::steady_clock::now();
            result.planning_time = std::chrono::duration<double>(end_time - planning_start_time_).count();
            result.resolution_stats = resolution_stats_;
            return result;
        }

        // Phase 3: Segment guided paths and check for conflicts
        std::cout << "[Phase 3] Segmenting guided paths..." << std::endl;
        segmentGuidedPaths();
        std::cout << "[Phase 3] Guided paths segmented" << std::endl;

        if (isTimeoutExceeded()) {
            std::cerr << "Planning timeout exceeded after segmenting guided paths" << std::endl;
            result.success = false;
            result.failure_reason = "timeout_segmenting_guided_paths";
            auto end_time = std::chrono::steady_clock::now();
            result.planning_time = std::chrono::duration<double>(end_time - planning_start_time_).count();
            result.resolution_stats = resolution_stats_;
            return result;
        }

        std::cout << "[Phase 3.1] Checking segments for conflicts..." << std::endl;
        bool conflicts_found = checkSegmentsForConflicts();
        std::cout << "[Phase 3.1] Conflict checking complete: " << segment_conflicts_.size() << " conflicts found." << std::endl;

        // Phase 4: Resolve conflicts 
        if (conflicts_found) {
            std::cout << "[Phase 5] Resolving collisions..." << std::endl;
            bool collisions_resolved = resolveConflicts();
            if (!collisions_resolved) {
                std::cerr << "Planning failed: could not resolve all collisions" << std::endl;
                result.success = false;
                if (isTimeoutExceeded()) {
                    result.failure_reason = "timeout_collision_resolution";
                } else {
                    result.failure_reason = "strategies_exhausted";
                }
            } else {
                std::cout << "[Phase 5] All collisions resolved" << std::endl;
                result.success = true;
            }
        } else {
            std::cout << "[Phase 5] No collisions to resolve" << std::endl;
            result.success = true;
        }

    } catch (const std::exception& e) {
        std::cerr << "Planning failed with exception: " << e.what() << std::endl;
        result.success = false;
        result.failure_reason = std::string("exception: ") + e.what();
    }

    result.planning_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - planning_start_time_).count();
    result.resolution_stats = resolution_stats_;
    return result;
}

void CipherGeometricPlanner::computeHighLevelPaths() {
    std::cout << "Computing high-level paths..." << std::endl;
    CBS cbs_solver(config_.mapf_config.region_capacity, config_.mapf_config.mapf_timeout,
                    obstacles_, config_.mapf_config.max_obstacle_volume_percent);
    high_level_paths_ = cbs_solver.solve(decomp_, start_states_, goal_states_);

    if (static_cast<int>(high_level_paths_.size()) < start_states_.size()) {
        std::cerr << "CBS failed to find paths for all robots." << std::endl;
    }
    for (int r = 0; r < (int)start_states_.size(); ++r) {
        if (high_level_paths_[r].empty()) {
            std::cerr << "CBS returned empty path for robot " << r << std::endl;
        }
        std::cout << "  Robot " << r << " CBS path: "
                  << high_level_paths_[r].size() << " regions" << std::endl;
    }

    if (do_viz_) {
        YAML::Node ev;
        ev["type"] = "mapf";
        YAML::Node paths;
        for (int r = 0; r < (int)high_level_paths_.size(); ++r) {
            YAML::Node cell_ids;
            for (int rid : high_level_paths_[r])
                cell_ids.push_back("c" + std::to_string(rid));
            paths["r" + std::to_string(r)] = cell_ids;
        }
        ev["paths"] = paths;
        viz_events_.push_back(ev);
        vizWriteFile();
        std::cout << "[viz] mapf event written to " << viz_file_ << std::endl;
    }
}

void CipherGeometricPlanner::computeGuidedPaths() {
    std::cout << "Computing guided paths..." << std::endl;

    if (!problem_loaded_) {
        throw std::runtime_error("Problem not loaded. Call loadProblem() first.");
    }

    if (high_level_paths_.empty()) {
        throw std::runtime_error(
            "High-level paths not computed. Call computeHighLevelPaths() first.");
    }

    // Clear any existing guided planning results
    guided_planning_results_.clear();

    for (size_t robot_idx = 0; robot_idx < robots_.size(); ++robot_idx) {
        auto robot_si = robot_sis_[robot_idx];

        guided_planning_results_.push_back(GuidedPlanningResult());

        guided_planning_results_[robot_idx].robot_index = robot_idx;

        auto pdef = std::make_shared<ob::ProblemDefinition>(robot_si);
        pdef->addStartState(start_states_[robot_idx]);
        pdef->setGoal(std::make_shared<PositionGoalCondition>(
            robot_si, goal_states_[robot_idx], config_.goal_threshold));
        
        // Create guided planner instance
        auto planner = std::make_shared<GuidedGeometricRRT>(robot_si);
        planner->setIntermediateStates(true);
        planner->setDecomposition(decomp_);
        planner->setDecompositionPath(high_level_paths_[robot_idx]);
        planner->setProblemDefinition(pdef);
        planner->setup();

        ob::PlannerStatus status = planner->solve(
            ob::timedPlannerTerminationCondition(config_.planning_time_limit));
        
        if (status == ob::PlannerStatus::EXACT_SOLUTION ||
            status == ob::PlannerStatus::APPROXIMATE_SOLUTION) {
            auto path = pdef->getSolutionPath()->as<og::PathGeometric>();
            path->interpolate();

            guided_planning_results_[robot_idx].success = true;
            // guided_planning_results_[robot_idx].planning_time = ;
            guided_planning_results_[robot_idx].path = std::make_shared<og::PathGeometric>(*path);
            std::cout << "  Robot " << robot_idx << ": solved with "
                      << guided_planning_results_[robot_idx].path->getStateCount() << " states" << std::endl;
        }
        else {
            guided_planning_results_[robot_idx].success = false;
            std::cout << "  Robot " << robot_idx << ": FAILED" << std::endl;
        }
    }

    std::cout << "Guided planner results: " << guided_planning_results_.size() << std::endl;

    // Emit low_level_paths event per robot once paths are available
    if (do_viz_ && !guided_planning_results_.empty()) {
        for (int r = 0; r < (int)guided_planning_results_.size(); ++r) {
            auto si = robot_sis_[r];
            const auto& path = guided_planning_results_[r].path;
            YAML::Node ev;
            ev["type"] = "low_level_paths";
            YAML::Node paths;
            YAML::Node waypoints;
            const size_t n = path->getStateCount();
            for (size_t i = 0; i < n; ++i) {
                std::vector<double> reals;
                si->getStateSpace()->copyToReals(reals, path->getState(i));
                YAML::Node wp;
                YAML::Node state_node;
                for (double v : reals) state_node.push_back(v);
                while ((int)state_node.size() < 3) state_node.push_back(0.0);
                wp["state"] = state_node;
                YAML::Node ctrl; ctrl.push_back(0.0); ctrl.push_back(0.0);
                wp["control"] = ctrl;
                wp["duration"] = (i + 1 < n) ? 0.1 : 0.0;
                waypoints.push_back(wp);
            }
            paths["r" + std::to_string(r)] = waypoints;
            ev["paths"] = paths;
            viz_events_.push_back(ev);
            vizWriteFile();
            std::cout << "[viz] low_level_paths event for robot " << r
                      << " written to " << viz_file_ << std::endl;
        }
    }
}

void CipherGeometricPlanner::segmentGuidedPaths() {
    std::cout << "Segmenting guided paths..." << std::endl;

    if (!problem_loaded_) {
        throw std::runtime_error("Problem not loaded. Call loadProblem() first.");
    }

    if (guided_planning_results_.empty()) {
        throw std::runtime_error(
            "Guided paths not computed. Call computeGuidedPaths() first.");
    }

    // Clear previous segments
    path_segments_.clear();
    path_segments_.resize(robots_.size());

    for (size_t robot_idx = 0; robot_idx < guided_planning_results_.size(); ++robot_idx) {
        const auto& result = guided_planning_results_[robot_idx];

        if (!result.success || !result.path) {
            continue;
        }

        auto& path = result.path;
        
        /// TODO: How do we properly get the mapping from timesteps to robot configurations for geometric paths?
        ///       Should we convert to PathControl for this?
        size_t path_length = path->getStateCount();
        if (path_length == 0) {
            continue;
        }

        // Segment the path
        int current_timestep = 0;
        size_t segment_idx = 0;
        size_t path_idx = 0;

        while (path_idx < path_length) {
            PathSegment segment;
            segment.robot_index = robot_idx;
            segment.segment_index = segment_idx;
            segment.start_timestep = current_timestep;

            // Start state is the current state we're at
            segment.start_state = path->getState(path_idx < path_length ? path_idx : path_length);
            segment.total_duration = 0.0;

            int timesteps_in_segment = 0;

            // Accumulate cfgs until we reach segment_timesteps
            while (path_idx < path_length && timesteps_in_segment < config_.segment_timesteps) {
                segment.states.push_back(path->getState(path_idx < path_length ? path_idx : path_length));
                segment.total_duration += 1.0;
                timesteps_in_segment++;
                path_idx++;   
            }

            segment.end_timestep = current_timestep + timesteps_in_segment;
            current_timestep = segment.end_timestep;

            // Set end state (state we'll be at after this segment)
            segment.end_state = path->getState(path_idx < path_length ? path_idx : path_length-1);
            // std::cout << "Path: path_idx-> " << path_idx << " w/ path_length-> " << path_length <<std::endl;

            path_segments_[robot_idx].push_back(segment);
            segment_idx++;
        }

        std::cout << "Robot " << robot_idx << " created " << path_segments_[robot_idx].size() << " segments." << std::endl;    }

        // for (size_t robot_idx = 0; robot_idx < guided_planning_results_.size(); ++robot_idx) {
        //     std::cout << "Robot " << robot_idx << " path segments!" << std::endl;
        //     for (auto path_segment : path_segments_[robot_idx]) {
        //         std::vector<double> start_state;
        //         std::vector<double> end_state;
        //         robot_sis_[robot_idx]->getStateSpace()->copyToReals(start_state, path_segment.start_state);
        //         robot_sis_[robot_idx]->getStateSpace()->copyToReals(end_state, path_segment.end_state);
        //         std::cout << "Segment start: [";
        //         for (size_t i = 0; i < start_state.size(); ++i) std::cout << (i ? ", " : "") << start_state[i];
        //         std::cout << "], Segment end: [";
        //         for (size_t i = 0; i < end_state.size(); ++i) std::cout << (i ? ", " : "") << end_state[i];
        //         std::cout << "]" << std::endl;
        //     }
        // }
    }

bool CipherGeometricPlanner::checkSegmentsForConflicts() {
    // std::cout << "Checking segments for conflicts..." << std::endl;
    segment_conflicts_.clear();

    // Handle edge cases
    if (path_segments_.empty()) {
        return false; // No segments to check
    }

    // Check if any robot has segments
    bool any_robot_has_segments = false;
    for (const auto& robot_segments : path_segments_) {
        if (!robot_segments.empty()) {
            any_robot_has_segments = true;
            break;
        }
    }

    if (!any_robot_has_segments) {
        return false;
    }

    // Find the global maximum timestep across ALL robots and ALL segments
    // This ensures we check every timestep where any robot is still moving
    int max_timestep = 0;
    for (size_t robot_idx = 0; robot_idx < robots_.size(); ++robot_idx) {
        if (!path_segments_[robot_idx].empty()) {
            const auto& last_segment = path_segments_[robot_idx].back();
            max_timestep = std::max(max_timestep, last_segment.end_timestep);
        }
    }

    if (max_timestep == 0) {
        return false;  // No timesteps to check
    }

    // Allocate states for each robot (one per robot for current timestep)
    std::vector<ob::State*> current_states(robots_.size(), nullptr);
    for (size_t i = 0; i < robots_.size(); ++i) {
        current_states[i] = robots_[i]->getSpaceInformation()->getStateSpace()->allocState();
    }

    // Iterate over ALL absolute timesteps to ensure complete coverage
    for (int timestep = 0; timestep < max_timestep; ++timestep) {

        // Propagate each robot to this absolute timestep
        for (size_t robot_idx = 0; robot_idx < robots_.size(); ++robot_idx) {
            if (path_segments_[robot_idx].empty()) continue;

            // Find the segment that contains this timestep
            const PathSegment* seg = findSegmentAtTimestep(robot_idx, timestep);

            if (seg != nullptr) {
                // Robot has a segment at this timestep - propagate to it
                propagateToTimestep(robot_idx, seg->segment_index, timestep, current_states[robot_idx]);
            } else {
                // Robot's path has ended - use final state (goal)
                const auto& last_segment = path_segments_[robot_idx].back();
                robots_[robot_idx]->getSpaceInformation()->copyState(
                    current_states[robot_idx], last_segment.end_state);
            }
        }

        // Check robot-robot conflicts (only if 2+ robots)
        if (robots_.size() >= 2) {
            for (size_t i = 0; i < robots_.size(); ++i) {
                if (path_segments_[i].empty()) continue;

                for (size_t j = i + 1; j < robots_.size(); ++j) {
                    if (path_segments_[j].empty()) continue;

                    size_t part_i, part_j;
                    if (checkTwoRobotConflict(i, current_states[i], j, current_states[j], part_i, part_j)) {
                        // Record conflict
                        SegmentConflict conflict;
                        conflict.type = SegmentConflict::ROBOT_ROBOT;
                        conflict.robot_index_1 = i;
                        conflict.robot_index_2 = j;
                        conflict.timestep = timestep;
                        conflict.part_index_1 = part_i;
                        conflict.part_index_2 = part_j;

                        // Find which segment each robot is in at this timestep
                        const PathSegment* seg_i = findSegmentAtTimestep(i, timestep);
                        const PathSegment* seg_j = findSegmentAtTimestep(j, timestep);

                        conflict.segment_index_1 = seg_i ? seg_i->segment_index : path_segments_[i].size() - 1;
                        conflict.segment_index_2 = seg_j ? seg_j->segment_index : path_segments_[j].size() - 1;

                        segment_conflicts_.push_back(conflict);
                    }
                }
            }
        }
    }

    // Free allocated states
    for (size_t i = 0; i < current_states.size(); ++i) {
        if (current_states[i]) {
            robots_[i]->getSpaceInformation()->getStateSpace()->freeState(current_states[i]);
        }
    }

    return !segment_conflicts_.empty();
}

bool CipherGeometricPlanner::resolveConflicts() {
    std::cout << "Resolving conflicts..." << std::endl;
    // Keep resolving conflicts until none remain or a conflict exhausts all strategies
    while (!segment_conflicts_.empty()) {
        // Check for timeout before attempting to resolve each conflict
        if (isTimeoutExceeded()) {
            std::cerr << "Planning timeout exceeded during conflict resolution ("
                      << segment_conflicts_.size() << " conflicts remaining)" << std::endl;

            // Log the conflict we were about to attempt as timed out
            ConflictResolutionEntry entry;
            entry.conflict_number = resolution_stats_.total_conflicts_encountered + 1;
            entry.robot_1 = segment_conflicts_[0].robot_index_1;
            entry.robot_2 = segment_conflicts_[0].robot_index_2;
            entry.timestep = segment_conflicts_[0].timestep;
            entry.resolved = false;
            entry.outcome = "timeout";
            resolution_stats_.conflict_log.push_back(entry);
            return false;
        }

        resolution_stats_.total_conflicts_encountered++;

        SegmentConflict conflict = segment_conflicts_[0];

        // Create log entry for this conflict
        ConflictResolutionEntry entry;
        entry.conflict_number = resolution_stats_.total_conflicts_encountered;
        entry.robot_1 = conflict.robot_index_1;
        entry.robot_2 = conflict.robot_index_2;
        entry.timestep = conflict.timestep;

        std::cout << "Resolving conflict " << resolution_stats_.total_conflicts_encountered << ": Robots "
                  << conflict.robot_index_1 << " and " << conflict.robot_index_2
                  << " at timestep " << conflict.timestep << std::endl;

        bool resolved = resolveConflictWithStrategies(conflict, entry);

        // Finalize the entry
        entry.resolved = resolved;
        if (resolved) {
            entry.outcome = "resolved";
        } else if (isTimeoutExceeded()) {
            entry.outcome = "timeout";
        } else {
            entry.outcome = "strategies_exhausted";
        }
        resolution_stats_.conflict_log.push_back(entry);

        if (!resolved) {
            std::cerr << "Failed to resolve conflict " << entry.conflict_number
                      << " (outcome: " << entry.outcome << ", "
                      << entry.attempts.size() << " strategy attempts made)" << std::endl;
            return false;  // Planning failed - problem may be too hard
        }
        resolution_stats_.total_conflicts_resolved++;
    }

    std::cout << "All collisions resolved successfully after " << resolution_stats_.total_conflicts_resolved << " collision resolutions" << std::endl;
    return true;
}

// Private helpers
void CipherGeometricPlanner::setupDecomposition() {
    std::cout << "Setting up decomposition..." << std::endl;
    /// TODO: We need to remove the hardcoded 2D assumption. Need a way to get the dimension of the workspace.
    auto decomp = std::make_shared<GridDecompositionImpl>(2, workspace_bounds_, config_.decomposition_region_length);
    decomp->setStateSpace(robots_[0]->getSpaceInformation()->getStateSpace());
    decomp_ = decomp;

    // Seed the viz ID map: every initial coarse cell maps to "c{id}".
    region_viz_id_.clear();
    for (int r = 0; r < decomp_->getNumRegions(); ++r)
        region_viz_id_[r] = "c" + std::to_string(r);
}

void CipherGeometricPlanner::setupCollisionManager() {
    std::cout << "Setting up conflict manager..." << std::endl;
    collision_manager_ = std::make_shared<fcl::DynamicAABBTreeCollisionManagerf>();
    collision_manager_->registerObjects(obstacles_);
    collision_manager_->setup();
}

void CipherGeometricPlanner::setupRobots() {
    std::cout << "Setting up robots..." << std::endl;
    ob::RealVectorBounds position_bounds(env_min_.size());
    for (size_t i = 0; i < env_min_.size(); ++i) {
        position_bounds.setLow(i, env_min_[i]);
        position_bounds.setHigh(i, env_max_[i]);
    }

    for (size_t i = 0; i < robot_types_.size(); ++i) {
        // Create robot
        auto robot = create_robot(robot_types_[i], position_bounds);
        auto state_space = robot->getSpaceInformation()->getStateSpace();
        state_space->setName("Robot " + std::to_string(i));
        auto si = std::make_shared<ob::SpaceInformation>(state_space);

        si->setStateValidityChecker(
            std::make_shared<fclStateValidityChecker>(si, collision_manager_, robot));
        si->setup();

        robots_.push_back(robot);
        robot_sis_.push_back(si);

        // Create start state
        ob::State* start = si->getStateSpace()->allocState();
        si->getStateSpace()->copyFromReals(start, starts_[i]);
        start_states_.push_back(start);

        // Create goal state
        ob::State* goal = si->getStateSpace()->allocState();
        si->getStateSpace()->copyFromReals(goal, goals_[i]);
        goal_states_.push_back(goal);

        GuidedPlanningResult result;
        result.robot_index = i;
        result.planning_time = 0.0;
        result.success = false;
        guided_planning_results_.push_back(result);
    }
}

void CipherGeometricPlanner::cleanup() {
    std::cout << "Cleaning up..." << std::endl;
    // Clean up allocated states
    if (!start_states_.empty() && !robots_.empty()) {
        for (size_t i = 0; i < start_states_.size() && i < robots_.size(); ++i) {
            if (start_states_[i]) {
                robots_[i]->getSpaceInformation()->getStateSpace()->freeState(start_states_[i]);
            }
        }
        start_states_.clear();
    }

    if (!goal_states_.empty() && !robots_.empty()) {
        for (size_t i = 0; i < goal_states_.size() && i < robots_.size(); ++i) {
            if (goal_states_[i]) {
                robots_[i]->getSpaceInformation()->getStateSpace()->freeState(goal_states_[i]);
            }
        }
        goal_states_.clear();
    }

    // Clear other data (but don't delete obstacles - caller owns them)
    robots_.clear();
    high_level_paths_.clear();
    guided_planning_results_.clear();
    path_segments_.clear();
    collision_manager_.reset();
    problem_loaded_ = false;
    resolution_stats_ = ResolutionStats();  // Reset resolution statistics
    robot_pair_conflict_counts_.clear();   // Reset cycle detection counters
    decomposition_hierarchy_.clear();       // Clear decomposition hierarchy
    region_viz_id_.clear();                 // Reset viz ID map
}

// Private conflict checking helpers
std::vector<fcl::CollisionObjectf*> CipherGeometricPlanner::getObstaclesInRegion(
    const std::vector<double>& region_min,
    const std::vector<double>& region_max) const {
    std::cout << "Getting obstacles in region..." << std::endl;
    return {};
}

const PathSegment* CipherGeometricPlanner::findSegmentAtTimestep(size_t robot_idx, int timestep) const {
    // std::cout << "Finding segment at timestep..." << std::endl;
    if (robot_idx >= path_segments_.size()) {
        return nullptr;
    }

    const auto& segments = path_segments_[robot_idx];
    for (const auto& segment : segments) {
        if (segment.start_timestep <= timestep && timestep < segment.end_timestep) {
            return &segment;
        }
    }
    return nullptr;
}

void CipherGeometricPlanner::propagateToTimestep(size_t robot_idx, size_t segment_idx, int timestep, ob::State* result) const {
    // std::cout << "Propagating to timestep..." << std::endl;
    const auto& segment = path_segments_[robot_idx][segment_idx];
    auto si = robots_[robot_idx]->getSpaceInformation();

    // Validate that timestep is within the segment's range
    if (timestep < segment.start_timestep || timestep >= segment.end_timestep) {
        std::cerr << "ERROR: propagateToTimestep called with invalid timestep!" << std::endl;
        std::cerr << "  robot_idx=" << robot_idx << ", segment_idx=" << segment_idx << std::endl;
        std::cerr << "  timestep=" << timestep << std::endl;
        std::cerr << "  segment.start_timestep=" << segment.start_timestep << std::endl;
        std::cerr << "  segment.end_timestep=" << segment.end_timestep << std::endl;

        // Clamp to valid range as a fallback
        if (timestep < segment.start_timestep) {
            timestep = segment.start_timestep;
        } else {
            timestep = segment.end_timestep - 1;
        }
    }

    // Direct index lookup: segment.states[k] is the state at absolute timestep
    // (start_timestep + k), mirroring PP.cpp's invariant of array index == timestep.
    int relative_timestep = timestep - segment.start_timestep;
    if (relative_timestep < static_cast<int>(segment.states.size())) {
        si->copyState(result, segment.states[relative_timestep]);
    } else {
        si->copyState(result, segment.end_state);
    }
}

bool CipherGeometricPlanner::checkTwoRobotConflict(size_t robot_idx_1, const ob::State* state_1,
                                               size_t robot_idx_2, const ob::State* state_2,
                                               size_t& part_1, size_t& part_2) const {
    // std::cout << "Checking two-robot conflict..." << std::endl;
    auto robot_1 = robots_[robot_idx_1];
    auto robot_2 = robots_[robot_idx_2];

    for (size_t p1 = 0; p1 < robot_1->numParts(); ++p1) {
        for (size_t p2 = 0; p2 < robot_2->numParts(); ++p2) {
            const auto& transform_1 = robot_1->getTransform(state_1, p1);
            const auto& transform_2 = robot_2->getTransform(state_2, p2);

            fcl::CollisionObjectf co_1(robot_1->getCollisionGeometry(p1));
            co_1.setTranslation(transform_1.translation());
            co_1.setRotation(transform_1.rotation());
            co_1.computeAABB();

            fcl::CollisionObjectf co_2(robot_2->getCollisionGeometry(p2));
            co_2.setTranslation(transform_2.translation());
            co_2.setRotation(transform_2.rotation());
            co_2.computeAABB();

            fcl::CollisionRequestf request;
            fcl::CollisionResultf result;
            fcl::collide(&co_1, &co_2, request, result);

            if (result.isCollision()) {
                part_1 = p1;
                part_2 = p2;
                return true;
            }
        }
    }

    return false;
}

// Private conflict resolution strategies
void CipherGeometricPlanner::updateDecomposition() {
    std::cout << "Updating decomposition..." << std::endl;
}

void CipherGeometricPlanner::expandSubproblem() {
    std::cout << "Expanding subproblem..." << std::endl;
}

GeometricPlanningResult CipherGeometricPlanner::useCompositePlanner(
    const std::vector<size_t>& robot_indices,
    const std::vector<std::vector<double>>& subproblem_starts,
    const std::vector<std::vector<double>>& subproblem_goals,
    const std::vector<double>& subproblem_env_min,
    const std::vector<double>& subproblem_env_max) {
    std::cout << "Using composite planner..." << std::endl;
    return GeometricPlanningResult();
}

bool CipherGeometricPlanner::resolveConflictWithStrategies(const SegmentConflict& conflict,
                                                            ConflictResolutionEntry& log_entry) {
    std::cout << "Resolving conflict with strategies..." << std::endl;
    const auto& config = config_.conflict_resolution_config;

    // Calculate max expansion layers if auto-detect
    int max_expansion_layers = config.max_expansion_layers;
    if (max_expansion_layers < 0) {
        max_expansion_layers = calculateMaxExpansionLayers();
    }

    // Cycle detection: track conflict counts per robot pair and escalate
    // the minimum expansion layer when the same pair keeps colliding
    int min_expansion_layer = 0;
    if (config.escalation_frequency > 0) {
        auto pair_key = std::make_tuple(
            std::min(conflict.robot_index_1, conflict.robot_index_2),
            std::max(conflict.robot_index_1, conflict.robot_index_2),
            conflict.timestep);
        int pair_count = ++robot_pair_conflict_counts_[pair_key];
        min_expansion_layer = (pair_count - 1) / config.escalation_frequency;
        min_expansion_layer = std::min(min_expansion_layer, max_expansion_layers);

        if (min_expansion_layer > 0) {
            std::cout << "  Cycle detection: robot pair (" << std::get<0>(pair_key) << ", "
                      << std::get<1>(pair_key) << ") at timestep " << std::get<2>(pair_key)
                      << " conflict #" << pair_count
                      << ", escalating to min expansion layer " << min_expansion_layer
                      << std::endl;
        }
    }

    // Strategy 1: Hierarchical Expansion + Refinement
    // This combines the old decomposition refinement and subproblem expansion into one
    // unified hierarchical approach:
    // - expansion_layer=0: refine just the collision cell (K times)
    // - expansion_layer=1,2,...: expand to neighbors, then refine all cells (K times each)
    // - Continue until expansion covers the whole decomposition
    if (config.max_refinement_levels > 0) {
        std::cout << "  Trying hierarchical expansion+refinement (max "
                  << config.max_refinement_levels << " refinement levels, max "
                  << max_expansion_layers << " expansion layers, min "
                  << min_expansion_layer << " expansion layer)..." << std::endl;

        if (resolveWithHierarchicalExpansionRefinement(
                conflict,
                config.max_refinement_levels,
                max_expansion_layers,
                min_expansion_layer,
                log_entry)) {
            std::cout << "  Hierarchical expansion+refinement resolved the collision" << std::endl;
            return true;
        }

        // Check if we timed out during hierarchical resolution
        if (isTimeoutExceeded()) {
            std::cerr << "  Timeout during hierarchical expansion+refinement" << std::endl;
            return false;
        }

        std::cout << "  Hierarchical expansion+refinement exhausted, escalating to local composite..." << std::endl;
    }

    // Strategy 2: Local Composite Planner (plans colliding robots jointly in local bounds)
    // if (conflict.type == SegmentConflict::ROBOT_ROBOT) {
    //     std::cout << "  Trying local composite planner (joint planning of colliding robots)..." << std::endl;

    //     if (resolveWithLocalCompositePlanner(conflict, log_entry)) {
    //         std::cout << "  Local composite planner resolved the collision" << std::endl;
    //         return true;
    //     }

    //     // Check if we timed out during local composite planning
    //     if (isTimeoutExceeded()) {
    //         std::cerr << "  Timeout during local composite planner" << std::endl;
    //         return false;
    //     }

    //     std::cout << "  Local composite planner failed, escalating to full-problem composite..." << std::endl;
    // }

    // Strategy 3: Full-Problem Composite Planner (ALL robots, original starts/goals)
    if (config.max_composite_attempts > 0) {
        std::cout << "  Trying full-problem composite planner (max "
                  << config.max_composite_attempts << " attempts)..." << std::endl;
        resolution_stats_.composite_planner_attempts++;

        if (resolveWithFullProblemCompositePlanner(config.max_composite_attempts, log_entry)) {
            std::cout << "  Full-problem composite planner resolved the conflict" << std::endl;
            resolution_stats_.composite_planner_successes++;
            return true;
        }

        std::cerr << "  Full-problem composite planner failed after " << config.max_composite_attempts
                  << " attempts for collision at timestep " << conflict.timestep << std::endl;
    }

    // All strategies exhausted - conflict could not be resolved
    std::cerr << "  All collision resolution strategies exhausted for conflict at timestep "
              << conflict.timestep << " between robots " << conflict.robot_index_1
              << " and " << conflict.robot_index_2 << std::endl;
    return false;
}

bool CipherGeometricPlanner::conflictPersistsForRobots(size_t robot_1, size_t robot_2, int timestep) const {
    std::cout << "Checking if conflict persists for robots..." << std::endl;
    return false;
}

bool CipherGeometricPlanner::resolveWithHierarchicalExpansionRefinement(
    const SegmentConflict& conflict,
    int max_refinement_levels,
    int max_expansion_layers,
    int min_expansion_layer,
    ConflictResolutionEntry& log_entry) {
    std::cout << "Resolving with hierarchical expansion/refinement..." << std::endl;
    size_t robot_1 = conflict.robot_index_1;
    size_t robot_2 = conflict.robot_index_2;

    // Allocate temporary states for conflict location
    ob::State* state_1 = robots_[robot_1]->getSpaceInformation()->getStateSpace()->allocState();
    ob::State* state_2 = robots_[robot_2]->getSpaceInformation()->getStateSpace()->allocState();

    // Locate conflict region - handle robots that have finished their paths (stationary at goal)
    const PathSegment* seg_1 = findSegmentAtTimestep(robot_1, conflict.timestep);
    const PathSegment* seg_2 = findSegmentAtTimestep(robot_2, conflict.timestep);

    // Get state for robot 1 at conflict time
    if (seg_1) {
        propagateToTimestep(robot_1, seg_1->segment_index, conflict.timestep, state_1);
    } else if (!path_segments_[robot_1].empty()) {
        // Robot has reached its goal - use final state
        const auto& last_seg = path_segments_[robot_1].back();
        robots_[robot_1]->getSpaceInformation()->copyState(state_1, last_seg.end_state);
    } else {
        std::cout << "    Robot " << robot_1 << " has no path segments" << std::endl;
        robots_[robot_1]->getSpaceInformation()->getStateSpace()->freeState(state_1);
        robots_[robot_2]->getSpaceInformation()->getStateSpace()->freeState(state_2);
        return false;
    }

    // Get state for robot 2 at conflict time (needed for potential future use)
    if (seg_2) {
        propagateToTimestep(robot_2, seg_2->segment_index, conflict.timestep, state_2);
    } else if (!path_segments_[robot_2].empty()) {
        // Robot has reached its goal - use final state
        const auto& last_seg = path_segments_[robot_2].back();
        robots_[robot_2]->getSpaceInformation()->copyState(state_2, last_seg.end_state);
    } else {
        std::cout << "    Robot " << robot_2 << " has no path segments" << std::endl;
        robots_[robot_1]->getSpaceInformation()->getStateSpace()->freeState(state_1);
        robots_[robot_2]->getSpaceInformation()->getStateSpace()->freeState(state_2);
        return false;
    }
    int conflict_region = decomp_->locateRegion(state_1);

    std::cout << "    Conflict in region " << conflict_region << std::endl;

    robots_[robot_1]->getSpaceInformation()->getStateSpace()->freeState(state_1);
    robots_[robot_2]->getSpaceInformation()->getStateSpace()->freeState(state_2);

    // Outer loop: expansion layers (0 = just conflict cell, 1 = 9 cells, 2 = 25 cells, etc.)
    // min_expansion_layer is set by cycle detection to skip small expansions that keep failing
    for (int expansion_layer = min_expansion_layer; expansion_layer <= max_expansion_layers; ++expansion_layer) {

        // Check for timeout before each expansion layer
        if (isTimeoutExceeded()) {
            std::cerr << "    Timeout during hierarchical expansion at layer " << expansion_layer << std::endl;
            return false;
        }

        // Check termination: expansion covers the whole decomposition
        if (expansion_layer > 0 && expansionCoversFullDecomposition(expansion_layer)) {
            std::cout << "    Expansion layer " << expansion_layer
                      << " covers entire decomposition, stopping expansion" << std::endl;
            break;
        }

        // Reset cells refined in previous expansion layers before considering the wider region set.
        // Refinement from a failed layer should not persist into the next expansion attempt.
        if (expansion_layer > min_expansion_layer) {
            auto grid_decomp = std::static_pointer_cast<GridDecompositionImpl>(decomp_);

            // Collect decomposed parent cells from previous expansion layers, deepest first.
            std::vector<std::pair<int,int>> to_reset; // (depth, rid)
            for (auto& [rid, level_pair] : region_refinement_level_) {
                if (level_pair.first < expansion_layer && grid_decomp->hasDecomposed(rid)) {
                    to_reset.push_back({grid_decomp->getDecompositionDepth(rid), rid});
                }
            }
            std::sort(to_reset.begin(), to_reset.end(), [](const auto& a, const auto& b) {
                return a.first > b.first;
            });

            for (auto& [depth, r] : to_reset) {
                if (grid_decomp->hasDecomposed(r)) {
                    for (int child : grid_decomp->getChildRegions(r)) {
                        region_viz_id_.erase(child);
                        region_refinement_level_.erase(child);
                    }
                }
                decomp_->resetCell(r);
                region_refinement_level_.erase(r);
            }

            if (!to_reset.empty()) {
                std::cout << "    Reset " << to_reset.size()
                          << " refined cell(s) from previous expansion layers" << std::endl;
            }
        }

        // Get expanded region for this layer
        std::vector<int> expanded_regions = getExpandedRegion(conflict_region, expansion_layer);

        std::cout << "    Expansion layer " << expansion_layer
                  << ": " << expanded_regions.size() << " regions" << std::endl;

        vizEmitCoupledPlanning({robot_1, robot_2}, expanded_regions);

        // Try K refinement levels at this expansion level
        if (attemptRefinementAtExpansionLevel(
                conflict,
                conflict_region,
                expanded_regions,
                expansion_layer,
                max_refinement_levels,
                log_entry)) {
            return true;
        }

        // Check if we timed out during refinement attempts
        if (isTimeoutExceeded()) {
            std::cerr << "    Timeout during refinement at expansion layer " << expansion_layer << std::endl;
            return false;
        }

        // Refinement at this expansion level failed, expand further
        std::cout << "    All refinement levels exhausted at expansion layer " << expansion_layer
                  << ", trying wider expansion..." << std::endl;
    }

    // All expansion levels exhausted
    std::cout << "    All expansion layers exhausted" << std::endl;
    return false;
}

bool CipherGeometricPlanner::attemptRefinementAtExpansionLevel(
    const SegmentConflict& conflict,
    int conflict_region,
    const std::vector<int>& expanded_regions,
    int expansion_layer,
    int max_refinement_levels,
    ConflictResolutionEntry& log_entry) {
    std::cout << "Attempting refinement at expansion level..." << std::endl;
    resolution_stats_.decomposition_refinement_attempts++;

    for (int refinement_level = 1; refinement_level <= max_refinement_levels; ++refinement_level) {
        // Check for timeout before each refinement attempt
        if (isTimeoutExceeded()) {
            std::cerr << "      Timeout before refinement level " << refinement_level
                      << " at expansion layer " << expansion_layer << std::endl;
            return false;
        }

        std::cout << "      Refinement level " << refinement_level << "/" << max_refinement_levels
                  << " at expansion layer " << expansion_layer << std::endl;

        StrategyAttempt attempt;
        attempt.strategy = "hierarchical_refinement";
        attempt.expansion_layer = expansion_layer;
        attempt.refinement_level = refinement_level;

        if (refineExpandedRegion(conflict, conflict_region, expanded_regions, expansion_layer, refinement_level)) {
            attempt.planning_succeeded = true;

            // Refinement and replanning succeeded, check if conflict is resolved
            if (!conflictPersistsForRobots(conflict.robot_index_1,
                                            conflict.robot_index_2,
                                            conflict.timestep)) {
                std::cout << "      Conflict resolved at expansion=" << expansion_layer
                          << ", refinement=" << refinement_level << std::endl;
                attempt.conflict_resolved = true;
                log_entry.attempts.push_back(attempt);
                resolution_stats_.decomposition_refinement_successes++;
                return true;
            }

            std::cout << "      Conflict persists after refinement level " << refinement_level
                      << ", trying higher refinement..." << std::endl;
        } else {
            attempt.planning_succeeded = false;
            std::cout << "      Refinement level " << refinement_level << " failed (planning failed)" << std::endl;
        }

        log_entry.attempts.push_back(attempt);
    }

    return false;
}

bool CipherGeometricPlanner::refineExpandedRegion(
    const SegmentConflict& conflict,
    int conflict_region,
    const std::vector<int>& expanded_regions,
    int expansion_layer,
    int refinement_level) {
    std::cout << "Refining expanded region..." << std::endl;

    size_t robot_1 = conflict.robot_index_1;
    size_t robot_2 = conflict.robot_index_2;

    // Calculate subdivision factor based on refinement level
    double subdivision_factor = std::pow(
        config_.conflict_resolution_config.decomposition_subdivision_factor,
        refinement_level);

    // Identify which LEAF cells within expanded_regions need refinement at this level.
    // We always descend to leaves so that higher refinement levels subdivide children
    // of previously-refined cells rather than re-decomposing already-split parents.
    auto grid_decomp = std::static_pointer_cast<GridDecompositionImpl>(decomp_);

    std::function<void(int, std::vector<int>&)> collectLeaves = [&](int rid, std::vector<int>& out) {
        if (!grid_decomp->hasDecomposed(rid)) { out.push_back(rid); return; }
        for (int child : grid_decomp->getChildRegions(rid)) collectLeaves(child, out);
    };

    std::vector<int> new_regions;
    for (int r : expanded_regions) {
        std::vector<int> leaves;
        collectLeaves(r, leaves);
        for (int leaf : leaves) {
            auto it = region_refinement_level_.find(leaf);
            if (it == region_refinement_level_.end() ||
                (it->second.first == expansion_layer && it->second.second < refinement_level))
                new_regions.push_back(leaf);
        }
    }

    if (new_regions.empty()) {
        std::cout << "        All " << expanded_regions.size()
                  << " cells already refined at level " << refinement_level
                  << ", skipping" << std::endl;
        return false;
    }

    std::cout << "        " << new_regions.size() << " new cell(s) to refine out of "
              << expanded_regions.size() << " total" << std::endl;

    // Capture viz IDs of cells-to-remove BEFORE they are erased from region_viz_id_.
    std::vector<std::string> removed_viz_ids;
    for (int r : new_regions)
        removed_viz_ids.push_back(region_viz_id_.count(r) ? region_viz_id_[r] : "c" + std::to_string(r));

    // Step 1: Refine the leaf cells in the global decomposition.
    for (int r : new_regions)
        decomp_->Decompose(r);

    std::cout << "        Decomposed " << new_regions.size() << " cell(s) in global decomposition" << std::endl;

    // Mark new cells as refined and register their children's viz IDs.
    for (int r : new_regions) {
        region_refinement_level_[r] = {expansion_layer, refinement_level};
        std::string parent_viz_id = region_viz_id_.count(r) ? region_viz_id_[r] : "c" + std::to_string(r);
        region_viz_id_.erase(r);
        for (int child : grid_decomp->getChildRegions(r)) {
            region_refinement_level_[child] = {expansion_layer, refinement_level};
            region_viz_id_[child] = parent_viz_id + "_" + std::to_string(child);
        }
    }

    // Step 2: Extract replanning bounds using top-level expanded_regions.
    // extractReplanningBoundsForExpandedRegion uses locateRegion() which returns
    // top-level cell IDs, so we must pass expanded_regions (not the leaf new_regions).
    PathUpdateInfo update_info_1, update_info_2;
    if (!extractReplanningBoundsForExpandedRegion(
            conflict, expanded_regions, update_info_1, update_info_2)) {
        std::cout << "        Failed to extract replanning bounds" << std::endl;
        return false;
    }

    // Add refined decomp to viz file — only show new_regions being replaced.
    if (do_viz_) {
        std::vector<std::tuple<std::string, std::vector<double>, std::vector<double>>> new_cells;
        for (int r : new_regions) {
            for (int child : grid_decomp->getChildRegions(r)) {
                auto cb = decomp_->getCellBounds(child);
                // region_viz_id_[child] was set in the marking step above
                new_cells.emplace_back(region_viz_id_[child],
                    std::vector<double>(cb.low.begin(), cb.low.end()),
                    std::vector<double>(cb.high.begin(), cb.high.end()));
            }
        }
        vizEmitGridUpdate(removed_viz_ids, new_cells);
    }

    // Handle ROBOT_OBSTACLE collisions (robot_1 == robot_2) as a single-robot replan.
    // Without this, the same robot would be replanned and integrated twice, corrupting the path.
    bool is_single_robot = (robot_1 == robot_2);

    // Determine which robots need replanning (stationary robots have entry == exit timestep)
    bool robot_1_stationary = (update_info_1.start_timestep == update_info_1.end_timestep);
    bool robot_2_stationary = is_single_robot ? true : (update_info_2.start_timestep == update_info_2.end_timestep);

    if (robot_1_stationary && robot_2_stationary) {
        // Both robots are stationary — can't replan either one
        std::cout << "        Both robots are stationary at goal, cannot refine" << std::endl;
        freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
        return false;
    }

    // Build lists of robots that need replanning
    std::vector<size_t> replan_robot_indices;
    std::vector<ob::State*> replan_starts;
    std::vector<ob::State*> replan_hl_starts;
    std::vector<ob::State*> replan_goals;
    std::vector<ob::State*> replan_hl_goals;
    // Track which index in replan arrays corresponds to which collision robot (0=robot_1, 1=robot_2)
    std::vector<int> replan_to_collision_idx;

    if (!robot_1_stationary) {
        replan_robot_indices.push_back(robot_1);
        replan_starts.push_back(update_info_1.planning_entry_state);
        replan_hl_starts.push_back(update_info_1.region_entry_state);
        replan_goals.push_back(update_info_1.planning_exit_state);
        replan_hl_goals.push_back(update_info_1.region_exit_state);
        replan_to_collision_idx.push_back(0);
    }
    if (!robot_2_stationary) {
        replan_robot_indices.push_back(robot_2);
        replan_starts.push_back(update_info_2.planning_entry_state);
        replan_hl_starts.push_back(update_info_2.region_entry_state);
        replan_goals.push_back(update_info_2.planning_exit_state);
        replan_hl_goals.push_back(update_info_2.region_exit_state);
        replan_to_collision_idx.push_back(1);
    }

    if (robot_1_stationary) {
        std::cout << "        Robot " << robot_1 << " is stationary at goal, only replanning robot " << robot_2 << std::endl;
    } else if (robot_2_stationary) {
        std::cout << "        Robot " << robot_2 << " is stationary at goal, only replanning robot " << robot_1 << std::endl;
    }

    std::vector<int> start_regions;
    std::vector<int> goal_regions;

    // Validate that entry/exit states of robots to replan are within local decomposition bounds
    bool states_in_bounds = true;
    for (const auto* state : replan_starts) {
        start_regions.push_back(decomp_->locateSubRegion(state));
        if (start_regions.back() < 0) {
            states_in_bounds = false;
            std::cout << "        Entry state outside local decomposition bounds" << std::endl;
            std::cout << "Local Decomp Bounds: ";
            for (size_t n =0; n < decomp_->getBounds().low.size(); ++n) {
                std::cout << "[" << decomp_->getBounds().low[n] << ":" << decomp_->getBounds().high[n] << "]";
            }
            std::cout << "" <<std::endl;
            
            robots_[0]->getSpaceInformation()->getStateSpace()->printState(state, std::cout);
            break;
        }
    }
    if (states_in_bounds) {
        for (const auto* state : replan_goals) {
            goal_regions.push_back(decomp_->locateSubRegion(state));
            if (goal_regions.back() < 0) {
                states_in_bounds = false;
                std::cout << "        Exit state outside local decomposition bounds" << std::endl;
                std::cout << "Local Decomp Bounds: ";
                for (size_t n =0; n < decomp_->getBounds().low.size(); ++n) {
                    std::cout << "[" << decomp_->getBounds().low[n] << ":" << decomp_->getBounds().high[n] << "]";
                }
                std::cout << "" <<std::endl;
                robots_[0]->getSpaceInformation()->getStateSpace()->printState(state, std::cout);
                break;
            }
        }
    }

    std::set<int> sts(start_regions.begin(), start_regions.end());
    std::set<int> gls(goal_regions.begin(), goal_regions.end());
    if (!states_in_bounds) {
        freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
        return false;
    }
    if (sts.size() != start_regions.size()) {
        std::cout << "!!!Same starts!!!" << std::endl;
        freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
        return false;
    }
    if (gls.size() != goal_regions.size()) {
        std::cout << "!!!Same goals!!!" << std::endl;
        freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
        return false;
    }

    // Step 3: MAPF replanning restricted to the expanded region's current leaf cells.
    // Collect leaf cells for all expanded_regions (post-refinement) so CBS cannot
    // route through regions outside the expanded area.
    std::set<int> expanded_leaf_regions;
    for (int r : expanded_regions) {
        std::vector<int> leaves;
        collectLeaves(r, leaves);
        expanded_leaf_regions.insert(leaves.begin(), leaves.end());
    }

    std::vector<std::vector<int>> local_high_level_paths;
    {
        CBS mapf_solver(1, config_.mapf_config.mapf_timeout,
                    obstacles_, config_.mapf_config.max_obstacle_volume_percent);

        for (size_t robot_idx = 0; robot_idx < replan_hl_starts.size(); ++robot_idx) {
            std::cout << "Robot " << robot_idx << " start region: " << decomp_->locateSubRegion(replan_hl_starts[robot_idx])
                << " -> end region: " << decomp_->locateSubRegion(replan_hl_goals[robot_idx]) << std::endl;
        }

        local_high_level_paths = mapf_solver.solve(
            decomp_, replan_hl_starts, replan_hl_goals, expanded_leaf_regions);
    }

    if (local_high_level_paths.empty()) {
        std::cout << "        MAPF failed" << std::endl;
        freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
        return false;
    } else {
        std::cout << "        MAPF succeeded, got high-level paths for " << local_high_level_paths.size() << " robot(s)" << std::endl;

        // Debug print the high-level paths
        for (size_t i = 0; i < local_high_level_paths.size(); ++i) {
            std::cout << "  Robot " << replan_robot_indices[i] << " high-level path: ";
            for (int sub_id : local_high_level_paths[i]) {
                std::cout << sub_id << " ";
            }
            std::cout << std::endl;
        }
    }


    for (size_t i = 0; i < replan_robot_indices.size(); ++i) {
        if (i >= local_high_level_paths.size() || local_high_level_paths[i].empty()) {
            std::cout << "        MAPF failed for robot " << replan_robot_indices[i] << std::endl;
            freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
            return false;
        }
    }

    if (do_viz_) {
        auto viz_id_for = [&](int sub_id) -> std::string {
            auto it = region_viz_id_.find(sub_id);
            return it != region_viz_id_.end() ? it->second : "c" + std::to_string(sub_id);
        };

        YAML::Node ev;
        ev["type"] = "local_mapf";
        YAML::Node paths;
        for (size_t i = 0; i < replan_robot_indices.size(); ++i) {
            YAML::Node cell_ids;
            for (int sub_id : local_high_level_paths[i]) {
                std::string cid = viz_id_for(sub_id);
                cell_ids.push_back(cid);
                // std::cout << "r" + std::to_string(replan_robot_indices[i]) << ": " << cid << std::endl;
            }
            paths["r" + std::to_string(replan_robot_indices[i])] = cell_ids;
        }
        ev["paths"] = paths;
        viz_events_.push_back(ev);
        vizWriteFile();
        std::cout << "[viz] local_mapf event written to " << viz_file_ << std::endl;
    }

    // Step 4: Guided planning (only for robots that need replanning)
    std::vector<GuidedPlanningResult> replan_results;
    bool all_succeeded = true;
    {
        for (size_t i = 0; i < replan_robot_indices.size(); ++i) {
            size_t robot_idx = replan_robot_indices[i];

            /// TODO: How to update problem def so that we can set new bounds
            auto robot_si = robot_sis_[robot_idx];

            replan_results.push_back(GuidedPlanningResult());
            replan_results[i].robot_index = robot_idx;

            auto pdef = std::make_shared<ob::ProblemDefinition>(robot_si);
            pdef->addStartState(replan_starts[i]);
            pdef->setGoal(std::make_shared<PositionGoalCondition>(
                robot_si, replan_goals[i], config_.goal_threshold
            ));

            auto planner = std::make_shared<GuidedGeometricRRT>(robot_si);
            planner->setIntermediateStates(true);
            planner->setDecomposition(decomp_);
            planner->setDecompositionPath(local_high_level_paths[i]);
            planner->setProblemDefinition(pdef);
            planner->setup();

            // ob::PlannerStatus status;
            ob::PlannerStatus status = planner->solve(
                ob::timedPlannerTerminationCondition(config_.planning_time_limit));
        
            if (status == ob::PlannerStatus::EXACT_SOLUTION ||
                status == ob::PlannerStatus::APPROXIMATE_SOLUTION) {
                auto path = pdef->getSolutionPath()->as<og::PathGeometric>();
                path->interpolate();

                replan_results[i].success = true;
                replan_results[i].path = std::make_shared<og::PathGeometric>(*path);
                std::cout << "  Robot " << robot_idx << ": solved with "
                        << replan_results[i].path->getStateCount() << " states" << std::endl;
            }
            else {
                replan_results[i].success = false;
                std::cout << "  Robot " << robot_idx << ": FAILED" << std::endl;
            }

            // GuidedPlanningResult result = guided_planner->solve(
            //     robots_[robot_idx],
            //     decomp_,
            //     replan_starts[i],
            //     replan_goals[i],
            //     local_high_level_paths[i],
            //     robot_idx);

            // replan_results.push_back(result);

            // if (!result.success) {
            if (!replan_results[i].success) {
                all_succeeded = false;
                break;
            }
        }
    }

    if (!all_succeeded) {
        std::cout << "        Guided planning failed" << std::endl;
        freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
        return false;
    }

    // Success! Integrate refined paths (only for robots that were replanned)
    std::cout << "        Planning succeeded, integrating refined paths" << std::endl;

    // Step 5: Integrate refined paths for robots that were replanned
    {
        for (size_t i = 0; i < replan_robot_indices.size(); ++i) {
            if (replan_results[i].path != nullptr) {
                std::vector<size_t> single_robot = {replan_robot_indices[i]};
                std::vector<GuidedPlanningResult> single_result = {replan_results[i]};
                PathUpdateInfo& update_info = (replan_to_collision_idx[i] == 0) ? update_info_1 : update_info_2;
                ///TODO: paths are not in correct regions... Also the paths are not from real start and goal
                integrateRefinedPaths(single_robot, single_result, update_info, update_info);
            }
        }

        // Record integrated paths
        if (do_viz_) {
            for (size_t i = 0; i < replan_robot_indices.size(); ++i) {
                if (replan_results[i].path == nullptr) continue;
                const size_t r = replan_robot_indices[i];
                auto si = robot_sis_[r];
                // const auto& path = replan_results[i].path;
                const auto& path = guided_planning_results_[r].path;
                YAML::Node ev;
                ev["type"] = "low_level_paths";
                YAML::Node paths;
                YAML::Node waypoints;
                const size_t n = path->getStateCount();
                for (size_t j = 0; j < n; ++j) {
                    std::vector<double> reals;
                    si->getStateSpace()->copyToReals(reals, path->getState(j));
                    YAML::Node wp;
                    YAML::Node state_node;
                    for (double v : reals) state_node.push_back(v);
                    while ((int)state_node.size() < 3) state_node.push_back(0.0);
                    wp["state"] = state_node;
                    YAML::Node ctrl; ctrl.push_back(0.0); ctrl.push_back(0.0);
                    wp["control"] = ctrl;
                    wp["duration"] = (j + 1 < n) ? 0.1 : 0.0;
                    waypoints.push_back(wp);
                }
                paths["r" + std::to_string(r)] = waypoints;
                ev["paths"] = paths;
                viz_events_.push_back(ev);
                vizWriteFile();
                std::cout << "[viz] refined low_level_paths event for robot " << r
                          << " written to " << viz_file_ << std::endl;
            }
        }
    }

    // Step 6: Re-check collisions
    {
        recheckConflictsFromTimestep(getRecheckStartTimestep(conflict));
    }

    freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);

    return true;
}

int CipherGeometricPlanner::calculateMaxExpansionLayers() const {
    std::cout << "Calculating max expansion layers..." << std::endl;

    // Find the largest bounding-box dimension across all robots and all their parts.
    float robot_max_dim = 0.0f;
    for (const auto& robot : robots_) {
        for (size_t part = 0; part < robot->numParts(); ++part) {
            auto geom = robot->getCollisionGeometry(part);
            if (geom) {
                geom->computeLocalAABB();
                const auto& aabb = geom->aabb_local;
                for (int d = 0; d < 3; ++d) {
                    float dim = aabb.max_[d] - aabb.min_[d];
                    robot_max_dim = std::max(robot_max_dim, dim);
                }
            }
        }
    }
    std::cout << "  Robot max geometry dimension: " << robot_max_dim << std::endl;

    return decomp_->getMaxDecompositions(0, robot_max_dim);

    // // Compute the base cell size from the decomposition bounds.
    // const auto& bounds = decomp_->getBounds();
    // int num_regions = decomp_->getNumRegions();
    // int grid_side = static_cast<int>(std::sqrt(num_regions));
    // float cell_size = std::numeric_limits<float>::max();
    // for (int d = 0; d < decomp_->getDimension(); ++d) {
    //     float span = static_cast<float>(bounds.high[d] - bounds.low[d]);
    //     float cs = (grid_side > 0) ? (span / grid_side) : span;
    //     cell_size = std::min(cell_size, cs);
    // }
    // std::cout << "  Base cell size (min dim): " << cell_size << std::endl;

    // // Cells must not be smaller than the robot's largest dimension.
    // // If the grid is already too fine, cap expansion to 1 layer and warn.
    // if (robot_max_dim > 0.0f && cell_size < robot_max_dim) {
    //     std::cout << "  WARNING: cell size (" << cell_size
    //               << ") is smaller than robot max dimension (" << robot_max_dim
    //               << "). Capping max expansion layers to 1." << std::endl;
    //     return 1;
    // }

    // // For a grid decomposition of NxN, the maximum useful expansion from center is N/2.
    // return (grid_side + 1) / 2;  // ceil(grid_side / 2)
}

bool CipherGeometricPlanner::expansionCoversFullDecomposition(int expansion_layers) const {
    std::cout << "Checking if expansion covers full decomposition..." << std::endl;
    int num_regions = decomp_->getNumRegions();

    // For a 2D grid, expansion by L layers from center gives at most (2L+1)^2 cells
    // If (2L+1)^2 >= num_regions, we've covered everything
    int max_cells_in_expansion = (2 * expansion_layers + 1) * (2 * expansion_layers + 1);

    return max_cells_in_expansion >= num_regions;
}

void CipherGeometricPlanner::freeUpdateInfoStates(
    size_t robot_1, size_t robot_2,
    PathUpdateInfo& update_info_1,
    PathUpdateInfo& update_info_2) {
    std::cout << "Freeing update info states..." << std::endl;
}

// Private helpers for all conflict strategies
std::shared_ptr<DecompositionImpl> CipherGeometricPlanner::createLocalDecomposition(
    int parent_region,
    double subdivision_factor) {
    std::cout << "Creating local decomposition..." << std::endl;
    const auto parent_bounds = decomp_->getCellBounds(parent_region);
    int sf = static_cast<int>(subdivision_factor);
    int dim = decomp_->getDimension();

    ob::RealVectorBounds local_bounds(dim);
    for (int i = 0; i < dim; ++i) {
        local_bounds.setLow(i, parent_bounds.low[i]);
        local_bounds.setHigh(i, parent_bounds.high[i]);
    }

    std::cout << "    Creating local decomposition: " << sf << "x" << sf << " grid" << std::endl;

    // Single original cell: sf sub-cells per dimension (square cell → square sub-cells)
    auto space = robots_[0]->getSpaceInformation()->getStateSpace();
    auto decomp_ = std::make_shared<RectGridDecompositionImpl>(
        std::vector<int>(dim, sf), local_bounds, space);

    recordRefinement(parent_region, decomp_);
    return decomp_;
}

std::shared_ptr<DecompositionImpl> CipherGeometricPlanner::createMultiCellDecomposition(
    const std::vector<int>& regions,
    double subdivision_factor) {
    std::cout << "Creating multi-cell decomposition..." << std::endl;
    int sf = static_cast<int>(subdivision_factor);
    int dim = decomp_->getDimension();

    // Get the original cell size (all cells are square with the same size)
    const auto first_bounds = decomp_->getCellBounds(regions[0]);
    double cell_size = first_bounds.high[0] - first_bounds.low[0];

    // Compute the bounding box of all original cells in the expanded region
    std::vector<double> env_min, env_max;
    computeExpandedBounds(regions, env_min, env_max);

    ob::RealVectorBounds expanded_bounds(dim);
    for (int i = 0; i < dim; ++i) {
        expanded_bounds.setLow(i, env_min[i]);
        expanded_bounds.setHigh(i, env_max[i]);
    }

    // For each original cell, the refinement produces sf sub-cells per dimension.
    // Compute the total sub-cell count per dimension from the number of original
    // cells along each axis (n_i * sf). This correctly handles non-square expanded
    // regions (e.g., 3×2 original cells at grid boundaries).
    std::vector<int> grid_lengths(dim);
    for (int i = 0; i < dim; ++i) {
        int n_i = static_cast<int>(std::round((env_max[i] - env_min[i]) / cell_size));
        grid_lengths[i] = n_i * sf;
    }

    std::cout << "      Multi-cell decomposition: " << grid_lengths[0] << "x" << grid_lengths[1]
              << " grid (" << (grid_lengths[0] * grid_lengths[1]) << " regions)" << std::endl;

    auto space = robots_[0]->getSpaceInformation()->getStateSpace();
    auto multi_cell_decomp = std::make_shared<RectGridDecompositionImpl>(
        grid_lengths, expanded_bounds, space);

    for (int region : regions)
        recordRefinement(region, multi_cell_decomp);

    return multi_cell_decomp;
}

bool CipherGeometricPlanner::extractReplanningBounds(
    const SegmentConflict& conflict,
    int conflict_region,
    PathUpdateInfo& update_info_1,
    PathUpdateInfo& update_info_2) {
    std::cout << "Extracting replanning bounds..." << std::endl;
    return false;
}

bool CipherGeometricPlanner::extractReplanningBoundsForExpandedRegion(
    const SegmentConflict& collision,
    const std::vector<int>& expanded_regions,
    PathUpdateInfo& update_info_1,
    PathUpdateInfo& update_info_2)
{
    size_t robot_1 = collision.robot_index_1;
    size_t robot_2 = collision.robot_index_2;

    // Convert expanded_regions to a set for O(1) lookup
    std::set<int> region_set(expanded_regions.begin(), expanded_regions.end());

    // Helper lambda to extract bounds for one robot
    auto extractForRobot = [&](size_t robot_idx, PathUpdateInfo& info) -> bool {
        info.robot_index = robot_idx;

        auto si = robots_[robot_idx]->getSpaceInformation();
        ob::State* temp_state = si->getStateSpace()->allocState();

        int collision_ts = collision.timestep;

        // Check if robot has finished its path before the collision timestep.
        // In that case, the robot is stationary at its goal — no replanning needed.
        int path_end_timestep = 0;
        if (!path_segments_[robot_idx].empty()) {
            path_end_timestep = path_segments_[robot_idx].back().end_timestep;
        }

        if (collision_ts >= path_end_timestep) {
            // Robot is stationary at goal — all three states collapse to the goal
            info.region_entry_state = si->getStateSpace()->allocState();
            info.region_exit_state = si->getStateSpace()->allocState();
            info.planning_entry_state = si->getStateSpace()->allocState();
            info.planning_exit_state = si->getStateSpace()->allocState();
            si->copyState(info.region_entry_state, goal_states_[robot_idx]);
            si->copyState(info.region_exit_state, goal_states_[robot_idx]);
            si->copyState(info.planning_entry_state, goal_states_[robot_idx]);
            si->copyState(info.planning_exit_state, goal_states_[robot_idx]);
            info.start_timestep = path_end_timestep;
            info.end_timestep = path_end_timestep;
            info.start_segment_idx = path_segments_[robot_idx].empty() ? 0 : path_segments_[robot_idx].size() - 1;
            info.end_segment_idx = info.start_segment_idx;
            si->getStateSpace()->freeState(temp_state);

            return true;
        }

        // Find entry to expanded region (scan backwards from collision)
        int entry_timestep = 0;
        size_t entry_segment = 0;
        bool found_entry = false;
        int pre_entry_timestep = -1;
        size_t pre_entry_segment = 0;

        for (int t = collision_ts; t >= 0; --t) {
            const PathSegment* seg = findSegmentAtTimestep(robot_idx, t);
            if (!seg) break;

            propagateToTimestep(robot_idx, seg->segment_index, t, temp_state);
            int region = decomp_->locateRegion(temp_state);

            // Check if region is NOT in the expanded set
            if (region_set.find(region) == region_set.end()) {
                // Found entry point (one timestep after leaving the region set)
                entry_timestep = t + 1;
                entry_segment = seg->segment_index;
                if (entry_timestep >= seg->end_timestep &&
                    seg->segment_index + 1 < path_segments_[robot_idx].size()) {
                    entry_segment = seg->segment_index + 1;
                }
                pre_entry_timestep = t;
                pre_entry_segment = seg->segment_index;
                found_entry = true;
                break;
            }
        }

        if (!found_entry) {
            // Robot starts in the expanded region
            entry_timestep = 0;
            entry_segment = 0;
        }

        // Find exit from expanded region (scan forwards).
        // Track the last timestep still inside the region so exit_state captures
        // the last in-region state (used by CBS to locateRegion the ending cell).
        int exit_timestep = -1;
        size_t exit_segment = 0;
        bool found_exit = false;

        int last_in_region_timestep = collision_ts;  // collision is inside by definition
        size_t last_in_region_segment = 0;
        if (const PathSegment* s = findSegmentAtTimestep(robot_idx, collision_ts))
            last_in_region_segment = s->segment_index;

        int max_timestep = path_end_timestep;

        for (int t = collision_ts; t < max_timestep; ++t) {
            const PathSegment* seg = findSegmentAtTimestep(robot_idx, t);
            if (!seg) break;

            propagateToTimestep(robot_idx, seg->segment_index, t, temp_state);
            int region = decomp_->locateRegion(temp_state);

            // Check if region is NOT in the expanded set
            if (region_set.find(region) == region_set.end()) {
                exit_timestep = t;
                exit_segment = seg->segment_index;
                found_exit = true;
                break;
            }

            // Still inside — record as candidate last-in-region state
            last_in_region_timestep = t;
            last_in_region_segment = seg->segment_index;
        }

        if (!found_exit) {
            // Robot's goal is in the expanded region (or robot has finished its path)
            exit_timestep = max_timestep;
            exit_segment = path_segments_[robot_idx].size() - 1;  // Use last valid segment index
        }

        // Allocate all four boundary states.
        info.region_entry_state = si->getStateSpace()->allocState();
        info.region_exit_state = si->getStateSpace()->allocState();
        info.planning_entry_state = si->getStateSpace()->allocState();
        info.planning_exit_state = si->getStateSpace()->allocState();

        // region_entry_state: first state inside the expanded region (at entry_timestep)
        // planning_entry_state: last state outside the expanded region (just before entry)
        if (!found_entry) {
            // Robot starts inside the region — no outside predecessor
            si->copyState(info.region_entry_state, start_states_[robot_idx]);
            si->copyState(info.planning_entry_state, start_states_[robot_idx]);
        } else {
            propagateToTimestep(robot_idx, entry_segment, entry_timestep, info.region_entry_state);
            propagateToTimestep(robot_idx, pre_entry_segment, pre_entry_timestep, info.planning_entry_state);
        }

        // planning_exit_state: first state outside the expanded region (at exit_timestep)
        // region_exit_state:   last state inside the expanded region (at last_in_region_timestep)
        if (exit_timestep >= max_timestep) {
            // Goal is inside the region — both collapse to the goal
            si->copyState(info.planning_exit_state, goal_states_[robot_idx]);
            si->copyState(info.region_exit_state, goal_states_[robot_idx]);
        } else {
            propagateToTimestep(robot_idx, exit_segment, exit_timestep, info.planning_exit_state);
            propagateToTimestep(robot_idx, last_in_region_segment, last_in_region_timestep, info.region_exit_state);
        }

        info.start_timestep = entry_timestep;
        info.end_timestep = exit_timestep;
        info.start_segment_idx = entry_segment;
        info.end_segment_idx = exit_segment;

        si->getStateSpace()->freeState(temp_state);
        return true;
    };

    auto robot_key = std::make_pair(robot_1, robot_2);
    auto& region_cache = robot_pair_refinement_info[robot_key];
    auto cache_it = region_cache.find(expanded_regions);
    if (cache_it != region_cache.end()) {
        std::cout << "Cache hit for robot pair (" << robot_1 << ", " << robot_2
              << ") with " << expanded_regions.size() << " regions" << std::endl;
        update_info_1 = cache_it->second.first;
        update_info_2 = cache_it->second.second;
        return true;
    }

    bool success_1 = extractForRobot(robot_1, update_info_1);
    bool success_2 = extractForRobot(robot_2, update_info_2);

    if (success_1 && success_2) {
        robot_pair_refinement_info[robot_key][expanded_regions] = {update_info_1, update_info_2};
    }

    return success_1 && success_2;
}

void CipherGeometricPlanner::integrateRefinedPaths(
    const std::vector<size_t>& robot_indices,
    const std::vector<GuidedPlanningResult>& local_results,
    const PathUpdateInfo& update_info_1,
    const PathUpdateInfo& update_info_2) {
    std::cout << "Integrating refined paths..." << std::endl;
    std::vector<PathUpdateInfo> update_infos = {update_info_1, update_info_2};

    for (size_t i = 0; i < robot_indices.size(); ++i) {
        size_t robot_idx = robot_indices[i];
        const auto& result = local_results[i];
        const auto& update_info = update_infos[i];

        std::cout << "    Integrating refined path for robot " << robot_idx << std::endl;

        // Note: We'll re-segment the entire path after splicing the PathControl below
        // This is necessary because segment state pointers become invalid after path updates

        // Splice the full PathControl for guided_planning_results
        // We need to construct a complete path: before + refined + after
        auto si = robots_[robot_idx]->getSpaceInformation();
        auto original_path = guided_planning_results_[robot_idx].path;

        if (original_path) {
            std::cout << "      Original path has " << original_path->getStateCount() << " states" << std::endl;
        }
        if (result.path) {
            std::cout << "      Refined path has " << result.path->getStateCount() << " states" << std::endl;
        }

        if (original_path && result.path) {
            // Create a new path that combines the three parts
            auto spliced_path = std::make_shared<og::PathGeometric>(si);

            // Part 1: Find the entry state index in the original path
            size_t entry_state_idx = 0;
            for (size_t s = 0; s < original_path->getStateCount(); ++s) {
                if (si->getStateSpace()->distance(original_path->getState(s), update_info.planning_entry_state) < 1e-3) {
                    std::cout << "!!Found Start!!" << std::endl;
                    entry_state_idx = s;
                    break;
                }
            }

            std::cout << "      Entry state found at index " << entry_state_idx << std::endl;

            // Part 1: Copy states from original path up to and including entry
            for (size_t s = 0; s <= entry_state_idx; ++s) {
                spliced_path->append(original_path->getState(s));
            }

            std::cout << "spliced path len: " << spliced_path->getStateCount() << std::endl;

            // Part 2: Add all states from the refined path
            for (size_t s = 0; s < result.path->getStateCount(); ++s) {
                spliced_path->append(result.path->getState(s));
            }

            std::cout << "spliced path len: " << spliced_path->getStateCount() << std::endl;

            // Part 3: Find exit state index in the original path
            size_t exit_state_idx = original_path->getStateCount() - 1;
            for (size_t s = entry_state_idx; s < original_path->getStateCount(); ++s) {
                if (si->getStateSpace()->distance(original_path->getState(s), update_info.planning_exit_state) < 1e-3) {
                    std::cout << "!!Found Exit!!" << std::endl;
                    exit_state_idx = s;
                    break;
                }
            }

            // Part 3: Copy states from original path after exit
            for (size_t s = exit_state_idx + 1; s < original_path->getStateCount(); ++s) {
                spliced_path->append(original_path->getState(s));
            }

            std::cout << "spliced path len: " << spliced_path->getStateCount() << std::endl;

            std::cout << "      Spliced path has " << spliced_path->getStateCount() << " states" << std::endl;

            // Update the guided planning result with the spliced path
            guided_planning_results_[robot_idx].path = spliced_path;

            // Re-segment the entire updated path from scratch
            std::vector<PathSegment> new_segments;
            segmentSinglePath(robot_idx, spliced_path, 0, new_segments);
            path_segments_[robot_idx] = new_segments;

            std::cout << "      Re-segmented path has " << new_segments.size() << " segments" << std::endl;
        } else {
            // If we can't splice, fall back torefined path
            guided_planning_results_[robot_idx] = result;

            // Re-segment the refined path
            std::vector<PathSegment> new_segments;
            segmentSinglePath(robot_idx, result.path, 0, new_segments);
            path_segments_[robot_idx] = new_segments;
        }
    }
}

void CipherGeometricPlanner::recheckConflictsFromTimestep(int start_timestep) {
    std::cout << "Re-checking conflicts from timestep..." << std::endl;
}

int CipherGeometricPlanner::getRecheckStartTimestep(const SegmentConflict& conflict) {
    std::cout << "Getting recheck start timestep..." << std::endl;
    return 0;
}

void CipherGeometricPlanner::segmentSinglePath(
    size_t robot_idx,
    const std::shared_ptr<og::PathGeometric>& path,
    int start_timestep_offset,
    std::vector<PathSegment>& segments) {
    std::cout << "Segmenting single path..." << std::endl;
}

// Private helpers for subproblem expansion strategies
std::vector<int> CipherGeometricPlanner::getExpandedRegion(int center_region, int expansion_layers) {
    std::cout << "Getting expanded region..." << std::endl;
    std::set<int> visited;
    std::queue<std::pair<int, int>> frontier;  // (region_id, distance)

    frontier.push({center_region, 0});
    visited.insert(center_region);

    while (!frontier.empty()) {
        auto [current_region, distance] = frontier.front();
        frontier.pop();

        if (distance < expansion_layers) {
            std::vector<int> neighbors;
            decomp_->getAllNeighbors(current_region, neighbors);

            for (int neighbor : neighbors) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    frontier.push({neighbor, distance + 1});
                }
            }
        }
    }

    return std::vector<int>(visited.begin(), visited.end());
}

void CipherGeometricPlanner::computeExpandedBounds(
    const std::vector<int>& regions,
    std::vector<double>& env_min,
    std::vector<double>& env_max) {
    std::cout << "Computing expanded bounds..." << std::endl;
    if (regions.empty()) {
        return;
    }

    auto grid_decomp = std::dynamic_pointer_cast<GridDecompositionImpl>(decomp_);
    int dim = decomp_->getDimension();

    env_min.resize(dim);
    env_max.resize(dim);

    // Initialize with first region's bounds
    const auto& first_bounds = grid_decomp->getCellBounds(regions[0]);
    for (int i = 0; i < dim; ++i) {
        env_min[i] = first_bounds.low[i];
        env_max[i] = first_bounds.high[i];
    }

    // Expand to include all regions
    for (size_t j = 1; j < regions.size(); ++j) {
        const auto& bounds = grid_decomp->getCellBounds(regions[j]);
        for (int i = 0; i < dim; ++i) {
            env_min[i] = std::min(env_min[i], bounds.low[i]);
            env_max[i] = std::max(env_max[i], bounds.high[i]);
        }
    }
}

// Private helpers for composite planner strategy
bool CipherGeometricPlanner::extractIndividualPaths(
    const std::shared_ptr<og::PathGeometric>& compound_path,
    std::vector<std::shared_ptr<og::PathGeometric>>& individual_paths) {
    std::cout << "Extracting individual paths from compound path..." << std::endl;
    return false;
}

// Private helpers for decomposition hierarchy tracking
void CipherGeometricPlanner::initializeDecompositionHierarchy() {
    std::cout << "Initializing decomposition hierarchy..." << std::endl;
}

DecompositionCell* CipherGeometricPlanner::findCellByRegion(int region_id) {
    std::cout << "Finding cell by region ID..." << std::endl;
    return nullptr;
}

DecompositionCell* CipherGeometricPlanner::findCellByRegionRecursive(DecompositionCell& cell, int region_id) {
    std::cout << "Finding cell by region ID recursively..." << std::endl;
    return nullptr;
}

void CipherGeometricPlanner::recordRefinement(int parent_region, const std::shared_ptr<DecompositionImpl> decomp_) {
    std::cout << "Recording refinement in decomposition hierarchy..." << std::endl;
}

bool CipherGeometricPlanner::resolveWithLocalCompositePlanner(
    const SegmentConflict& conflict,
    ConflictResolutionEntry& log_entry) {
    std::cout << "Resolving with local composite planner..." << std::endl;

    size_t robot_1 = conflict.robot_index_1;
    size_t robot_2 = conflict.robot_index_2;

    std::cout << "    Local composite planner: jointly planning robots "
              << robot_1 << " and " << robot_2 << std::endl;

    // Check for timeout
    if (isTimeoutExceeded()) {
        std::cerr << "    Timeout before local composite planner" << std::endl;
        return false;
    }

    StrategyAttempt attempt;
    attempt.strategy = "local_composite";

    // Locate conflict region and get expanded region for the subproblem
    // Handle robots that have finished their paths (stationary at goal)
    ob::State* temp_state = robots_[robot_1]->getSpaceInformation()->getStateSpace()->allocState();
    const PathSegment* seg_1 = findSegmentAtTimestep(robot_1, conflict.timestep);
    if (seg_1) {
        propagateToTimestep(robot_1, seg_1->segment_index, conflict.timestep, temp_state);
    } else if (!path_segments_[robot_1].empty()) {
        // Robot has reached its goal - use final state
        const auto& last_seg = path_segments_[robot_1].back();
        robots_[robot_1]->getSpaceInformation()->copyState(temp_state, last_seg.end_state);
    } else {
        robots_[robot_1]->getSpaceInformation()->getStateSpace()->freeState(temp_state);
        std::cout << "    Robot " << robot_1 << " has no path segments" << std::endl;
        attempt.planning_succeeded = false;
        log_entry.attempts.push_back(attempt);
        return false;
    }
    int conflict_region = decomp_->locateRegion(temp_state);
    robots_[robot_1]->getSpaceInformation()->getStateSpace()->freeState(temp_state);

    // Use 2 layers of expansion around the conflict region for local bounds
    std::vector<int> expanded_regions = getExpandedRegion(conflict_region, 2);

    // Extract replanning bounds (entry/exit states) for both robots
    PathUpdateInfo update_info_1, update_info_2;
    if (!extractReplanningBoundsForExpandedRegion(
            conflict, expanded_regions, update_info_1, update_info_2)) {
        std::cout << "    Failed to extract replanning bounds" << std::endl;
        attempt.planning_succeeded = false;
        log_entry.attempts.push_back(attempt);
        return false;
    }

    // Convert OMPL entry/exit states to std::vector<double>
    auto si_1 = robots_[robot_1]->getSpaceInformation();
    auto si_2 = robots_[robot_2]->getSpaceInformation();

    std::vector<double> start_1, goal_1, start_2, goal_2;
    si_1->getStateSpace()->copyToReals(start_1, update_info_1.planning_entry_state);
    si_1->getStateSpace()->copyToReals(goal_1, update_info_1.planning_exit_state);
    si_2->getStateSpace()->copyToReals(start_2, update_info_2.planning_entry_state);
    si_2->getStateSpace()->copyToReals(goal_2, update_info_2.planning_exit_state);

    // Compute local bounds from expanded region
    std::vector<double> local_env_min, local_env_max;
    computeExpandedBounds(expanded_regions, local_env_min, local_env_max);

    // Call useCompositePlanner to jointly plan both robots
    std::vector<size_t> robot_indices = {robot_1, robot_2};
    std::vector<std::vector<double>> subproblem_starts = {start_1, start_2};
    std::vector<std::vector<double>> subproblem_goals = {goal_1, goal_2};

    GeometricPlanningResult result;
    {
        result = useCompositePlanner(
            robot_indices, subproblem_starts, subproblem_goals,
            local_env_min, local_env_max);
    }

    if (result.solved && result.individual_paths.size() == 2) {
        attempt.planning_succeeded = true;
        std::cout << "    Local composite planning succeeded" << std::endl;

        // Convert PlanningResult individual paths to GuidedPlanningResult format
        std::vector<GuidedPlanningResult> local_results;
        for (size_t i = 0; i < robot_indices.size(); ++i) {
            GuidedPlanningResult guided_result;
            guided_result.success = true;
            guided_result.planning_time = result.planning_time;
            guided_result.robot_index = robot_indices[i];
            guided_result.path = result.individual_paths[i];
            local_results.push_back(guided_result);
        }

        // Integrate refined paths and re-check conflict
        {
            integrateRefinedPaths(robot_indices, local_results, update_info_1, update_info_2);
        }
        {
            recheckConflictsFromTimestep(getRecheckStartTimestep(conflict));
        }

        // Check if the conflict is resolved
        if (!conflictPersistsForRobots(robot_1, robot_2, conflict.timestep)) {
            std::cout << "    Local composite planner resolved the collision" << std::endl;
            attempt.conflict_resolved = true;
            log_entry.attempts.push_back(attempt);
            freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
            return true;
        }

        std::cout << "    Local composite planner: collision persists after replanning" << std::endl;
    } else {
        attempt.planning_succeeded = false;
        std::cout << "    Local composite planner failed to find solution" << std::endl;
    }

    log_entry.attempts.push_back(attempt);
    freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
    return false;
}

bool CipherGeometricPlanner::resolveWithFullProblemCompositePlanner(
    int max_attempts,
    ConflictResolutionEntry& log_entry) {
    std::cout << "Resolving with full-problem composite planner..." << std::endl;

    // Plan all robots jointly from their original starts to goals
    std::vector<size_t> all_robot_indices;
    for (size_t i = 0; i < robots_.size(); ++i) {
        all_robot_indices.push_back(i);
    }

    for (int attempt_num = 1; attempt_num <= max_attempts; ++attempt_num) {
        if (isTimeoutExceeded()) {
            std::cerr << "    Timeout during full-problem composite planner attempt "
                      << attempt_num << std::endl;
            return false;
        }

        std::cout << "    Full-problem composite attempt " << attempt_num
                  << "/" << max_attempts << std::endl;

        StrategyAttempt attempt;
        attempt.strategy = "full_problem_composite_planner";
        attempt.attempt_number = attempt_num;

        GeometricPlanningResult result = useCompositePlanner(
            all_robot_indices, starts_, goals_, env_min_, env_max_);

        attempt.planning_succeeded = result.solved;
        if (result.solved) {
            // All conflicts should be gone after a full re-plan
            attempt.conflict_resolved = true;
            log_entry.attempts.push_back(attempt);
            return true;
        }

        log_entry.attempts.push_back(attempt);
    }

    return false;
}


int main(int argc, char** argv)
{
    std::string inputFile;
    std::string outputFile;
    std::string configFile;
    std::string vizFile;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Show help message")
        ("input,i", po::value<std::string>(&inputFile)->required(), "Input YAML file")
        ("output,o", po::value<std::string>(&outputFile)->required(), "Output YAML file")
        ("cfg,c", po::value<std::string>(&configFile), "Configuration YAML file")
        ("viz,v", po::value<std::string>(&vizFile), "Visualization log output YAML file");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc << std::endl; return 0; }
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }

    CipherGeometricConfig config;

    if (vm.count("cfg")) {
        try {
            YAML::Node cfg = YAML::LoadFile(configFile);
            if (cfg["timelimit"])       config.planning_time_limit = cfg["timelimit"].as<double>();
            if (cfg["max_total_time"])  config.max_total_time = cfg["max_total_time"].as<double>();
            if (cfg["seed"])            config.seed = cfg["seed"].as<int>();
            if (cfg["region_size"])     config.decomposition_region_length = cfg["region_size"].as<int>();
            if (cfg["cbs_capacity"])    config.mapf_config.region_capacity = cfg["cbs_capacity"].as<int>();
            if (cfg["max_obstacle_volume_percent"])
                config.mapf_config.max_obstacle_volume_percent = cfg["max_obstacle_volume_percent"].as<double>();
        } catch (const YAML::Exception& e) {
            std::cerr << "ERROR loading config file: " << e.what() << std::endl;
            return 1;
        }
    }

    if (config.seed >= 0) {
        std::cout << "Setting random seed to: " << config.seed << std::endl;
        ompl::RNG::setSeed(config.seed);
    }

    std::cout << "Loading YAML file: " << inputFile << std::endl;
    YAML::Node env;
    try {
        env = YAML::LoadFile(inputFile);
    } catch (const YAML::Exception& e) {
        std::cerr << "ERROR loading YAML file: " << e.what() << std::endl;
        return 1;
    }

    // Parse environment bounds
    const auto& env_min_node = env["environment"]["min"];
    const auto& env_max_node = env["environment"]["max"];
    std::vector<double> env_min = {env_min_node[0].as<double>(), env_min_node[1].as<double>()};
    std::vector<double> env_max = {env_max_node[0].as<double>(), env_max_node[1].as<double>()};

    // Build FCL obstacle list
    std::vector<fcl::CollisionObjectf*> obstacles;
    if (env["environment"]["obstacles"]) {
        for (const auto& obs : env["environment"]["obstacles"]) {
            if (obs["type"].as<std::string>() == "box") {
                const auto& sz  = obs["size"];
                const auto& ctr = obs["center"];
                auto box = std::make_shared<fcl::Boxf>(sz[0].as<float>(), sz[1].as<float>(), 1.0f);
                auto* co = new fcl::CollisionObjectf(box);
                co->setTranslation(fcl::Vector3f(ctr[0].as<float>(), ctr[1].as<float>(), 0.0f));
                co->computeAABB();
                obstacles.push_back(co);
            }
        }
    }
    std::cout << "Loaded " << obstacles.size() << " obstacles" << std::endl;

    // Parse robots
    std::vector<std::string> robot_types;
    std::vector<std::vector<double>> starts, goals;
    int robot_idx = 0;
    for (const auto& robot_node : env["robots"]) {
        robot_types.push_back(robot_node["type"].as<std::string>());

        std::vector<double> s, g;
        for (const auto& v : robot_node["start"]) s.push_back(v.as<double>());
        for (const auto& v : robot_node["goal"])  g.push_back(v.as<double>());
        starts.push_back(s);
        goals.push_back(g);

        std::cout << "  Robot " << robot_idx << " (" << robot_types.back() << ")"
                  << "  Start: (" << s[0] << ", " << s[1] << ")"
                  << "  Goal: ("  << g[0] << ", " << g[1] << ")" << std::endl;
        ++robot_idx;
    }
    std::cout << "Planning for " << robot_types.size() << " robots" << std::endl;

    CipherGeometricPlanner planner(config);
    if (!vizFile.empty()) planner.setVizFile(vizFile);
    planner.loadProblem(robot_types, starts, goals, obstacles, env_min, env_max);
    CipherGeometricResult result = planner.plan();

    // Write output YAML
    YAML::Node output;
    output["solved"] = result.success;
    output["planning_time"] = result.planning_time;
    if (!result.failure_reason.empty())
        output["failure_reason"] = result.failure_reason;

    try {
        std::ofstream fout(outputFile);
        fout << output;
        fout.close();
        std::cout << "Output written to " << outputFile << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR writing output: " << e.what() << std::endl;
        return 1;
    }

    for (auto* co : obstacles) delete co;

    std::cout << "Done! time=" << result.planning_time << "s  solved=" << result.success << std::endl;
    return result.success ? 0 : 1;
}
