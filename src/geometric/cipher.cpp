#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <ompl/util/Console.h>

#ifndef NDEBUG
#define DOUT std::cout
#else
namespace {
struct NullBuf : std::streambuf { int overflow(int c) override { return c; } };
struct NullStream : std::ostream { NullStream() : std::ostream(&_buf) {} NullBuf _buf; } _null_stream;
}
#define DOUT _null_stream
#endif
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
    DOUT << "[viz] Header written (" << num_regions << " cells) to " << viz_file_ << std::endl;
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
    DOUT << "[viz] coupled_planning event: " << robot_indices.size()
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
    DOUT << "[viz] grid_update event: " << removed_viz_ids.size()
              << " removed, " << new_cells.size() << " added" << std::endl;
}

void CipherGeometricPlanner::vizEmitConflicts(const std::vector<SegmentConflict>& conflicts)
{
    if (!do_viz_ || conflicts.empty()) return;

    // For each unique robot pair, record only the first (minimum) timestep.
    std::map<std::pair<size_t,size_t>, int> pair_first_timestep;
    for (const auto& c : conflicts) {
        if (c.type != SegmentConflict::ROBOT_ROBOT) continue;
        auto key = std::make_pair(
            std::min(c.robot_index_1, c.robot_index_2),
            std::max(c.robot_index_1, c.robot_index_2));
        auto it = pair_first_timestep.find(key);
        if (it == pair_first_timestep.end())
            pair_first_timestep[key] = c.timestep;
        else
            it->second = std::min(it->second, c.timestep);
    }

    for (const auto& [pair, timestep] : pair_first_timestep) {
        YAML::Node ev;
        ev["type"] = "collision";
        YAML::Node robots_node;
        robots_node.push_back("r" + std::to_string(pair.first));
        robots_node.push_back("r" + std::to_string(pair.second));
        ev["robots"] = robots_node;
        ev["time"] = timestep * 0.1;
        viz_events_.push_back(ev);
    }

    vizWriteFile();
    DOUT << "[viz] collision events: " << pair_first_timestep.size()
              << " pair(s)" << std::endl;
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
    separateStartCells();

    // Build and write visualization header now that robots + decomp are ready
    initVizHeader();

    problem_loaded_ = true;
}

CipherGeometricResult CipherGeometricPlanner::plan() {
    DOUT << "Planning with CipherGeometricPlanner..." << std::endl;

    // Check if problem is loaded
    if (!problem_loaded_) {
        throw std::runtime_error("Problem not loaded. Call loadProblem() first.");
    }
    
    CipherGeometricResult result;
    planning_start_time_ = std::chrono::steady_clock::now();

    forbidden_edges_.clear();

    try {
        // Phase 1+2: CBS high-level paths + guided paths, with forbidden-edge retry
        int blocked_edge_attempts = 0;
        while (true) {
            DOUT << "[Phase 1] Computing high-level paths (attempt "
                 << blocked_edge_attempts + 1 << ")..." << std::endl;

            bool cbs_ok = false;
            try {
                computeHighLevelPaths();
                cbs_ok = true;
            } catch (const std::exception& e) {
                if (blocked_edge_attempts == 0) {
                    std::cerr << "[Phase 1] CBS failed (" << e.what()
                              << "); decomposing all leaf cells one level and retrying" << std::endl;
                    if (!decomposeAllLeavesOneLevel()) {
                        std::cerr << "[Phase 1] All cells at maximum decomposition depth; giving up" << std::endl;
                        throw;
                    }
                } else {
                    throw;
                }
            }
            if (!cbs_ok) continue;

            DOUT << "[Phase 1] High-level paths computed" << std::endl;

            if (isTimeoutExceeded()) {
                std::cerr << "Planning timeout exceeded after computing high-level paths" << std::endl;
                result.success = false;
                result.failure_reason = "timeout_high_level_paths";
                auto end_time = std::chrono::steady_clock::now();
                result.planning_time = std::chrono::duration<double>(end_time - planning_start_time_).count();
                result.resolution_stats = resolution_stats_;
                return result;
            }

            size_t prev_forbidden_count = forbidden_edges_.size();

            DOUT << "[Phase 2] Computing guided paths..." << std::endl;
            computeGuidedPaths();
            DOUT << "[Phase 2] Guided paths computed" << std::endl;

            if (isTimeoutExceeded()) {
                std::cerr << "Planning timeout exceeded after computing guided paths" << std::endl;
                result.success = false;
                result.failure_reason = "timeout_guided_paths";
                auto end_time = std::chrono::steady_clock::now();
                result.planning_time = std::chrono::duration<double>(end_time - planning_start_time_).count();
                result.resolution_stats = resolution_stats_;
                return result;
            }

            bool new_edges_added = forbidden_edges_.size() > prev_forbidden_count;
            if (!new_edges_added || blocked_edge_attempts >= config_.max_blocked_edge_retries)
                break;

            ++blocked_edge_attempts;
            DOUT << "[Retry] " << forbidden_edges_.size()
                 << " forbidden edge(s); replanning high-level paths" << std::endl;
            high_level_paths_.clear();
        }

        DOUT << "[Phase 3] Checking paths for conflicts..." << std::endl;
        bool conflicts_found = checkPathsForConflicts();
        DOUT << "[Phase 3] Conflict checking complete: " << segment_conflicts_.size() << " conflicts found." << std::endl;

        // Phase 4: Resolve conflicts 
        if (conflicts_found) {
            DOUT << "[Phase 5] Resolving conflicts..." << std::endl;
            bool conflicts_resolved = resolveConflicts();
            if (!conflicts_resolved) {
                std::cerr << "Planning failed: could not resolve all conflicts" << std::endl;
                result.success = false;
                if (isTimeoutExceeded()) {
                    result.failure_reason = "timeout_conflict_resolution";
                } else {
                    result.failure_reason = "strategies_exhausted";
                }
            } else {
                DOUT << "[Phase 5] All conflicts resolved" << std::endl;
                result.success = true;
            }
        } else {
            DOUT << "[Phase 5] No conflicts to resolve" << std::endl;
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
    DOUT << "Computing high-level paths..." << std::endl;
    CBS cbs_solver(config_.mapf_config.region_capacity, config_.mapf_config.mapf_timeout,
                    obstacles_, config_.mapf_config.max_obstacle_volume_percent);
    high_level_paths_ = cbs_solver.solve(decomp_, start_states_, goal_states_,
                                         /*allowed_regions=*/{}, forbidden_edges_);

    if (static_cast<int>(high_level_paths_.size()) < start_states_.size()) {
        std::cerr << "CBS failed to find paths for all robots." << std::endl;
    }
    for (int r = 0; r < (int)start_states_.size(); ++r) {
        if (high_level_paths_[r].empty()) {
            std::cerr << "CBS returned empty path for robot " << r << std::endl;
        }
        DOUT << "  Robot " << r << " CBS path: "
                  << high_level_paths_[r].size() << " regions" << std::endl;
    }

    if (do_viz_) {
        YAML::Node ev;
        ev["type"] = "mapf";
        YAML::Node paths;
        for (int r = 0; r < (int)high_level_paths_.size(); ++r) {
            YAML::Node cell_ids;
            for (int rid : high_level_paths_[r])
                cell_ids.push_back(region_viz_id_.count(rid)
                    ? region_viz_id_.at(rid) : "c" + std::to_string(rid));
            paths["r" + std::to_string(r)] = cell_ids;
        }
        ev["paths"] = paths;
        viz_events_.push_back(ev);
        vizWriteFile();
        DOUT << "[viz] mapf event written to " << viz_file_ << std::endl;
    }
}

void CipherGeometricPlanner::computeGuidedPaths() {
    DOUT << "Computing guided paths..." << std::endl;

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
        std::cout << "Guided planning for robot " << robot_idx << "..." << std::endl;

        auto robot_si = robot_sis_[robot_idx];

        guided_planning_results_.push_back(GuidedPlanningResult());

        guided_planning_results_[robot_idx].robot_index = robot_idx;

        auto pdef = std::make_shared<ob::ProblemDefinition>(robot_si);
        pdef->addStartState(start_states_[robot_idx]);
        pdef->setGoal(std::make_shared<PositionGoalCondition>(
            robot_si, goal_states_[robot_idx], config_.goal_threshold));
        
        // Compute per-robot XY circumradius (worst-case footprint for a rotating robot)
        float robot_inflation = 0.0f;
        if (config_.restrict_sampling_to_cell) {
            auto robot = robots_[robot_idx];
            for (size_t part = 0; part < robot->numParts(); ++part) {
                auto geom = robot->getCollisionGeometry(part);
                if (geom) {
                    geom->computeLocalAABB();
                    const auto& aabb = geom->aabb_local;
                    float hx = (aabb.max_[0] - aabb.min_[0]) / 2.0f;
                    float hy = (aabb.max_[1] - aabb.min_[1]) / 2.0f;
                    robot_inflation = std::max(robot_inflation, std::sqrt(hx * hx + hy * hy));
                }
            }
        }

        // Create guided planner instance
        auto planner = std::make_shared<GuidedGeometricRRT>(robot_si);
        planner->setIntermediateStates(true);
        planner->setDecomposition(decomp_);
        planner->setDecompositionPath(high_level_paths_[robot_idx]);
        planner->setRobotInflation(static_cast<double>(robot_inflation));
        if (config_.max_initial_extensions > 0)
            planner->setMaxExtensions(config_.max_initial_extensions);
        if (config_.max_no_progress_iters > 0)
            planner->setMaxNoProgressIters(config_.max_no_progress_iters);
        planner->setProblemDefinition(pdef);
        planner->setup();

        {
            const auto& path = high_level_paths_[robot_idx];
            int last_region = path.back();
            auto last_bounds = decomp_->getCellBounds(last_region);
            int goal_region = decomp_->locateSubRegion(goal_states_[robot_idx]);
            std::vector<double> goal_reals;
            robot_si->getStateSpace()->copyToReals(goal_reals, goal_states_[robot_idx]);
            bool goal_in_last = (goal_reals[0] >= last_bounds.low[0] && goal_reals[0] < last_bounds.high[0] &&
                                 goal_reals[1] >= last_bounds.low[1] && goal_reals[1] < last_bounds.high[1]);
            std::cout << "[cipher] robot " << robot_idx
                      << " goal=(" << goal_reals[0] << "," << goal_reals[1] << ")"
                      << " locateSubRegion(goal)=" << goal_region
                      << " path.back()=" << last_region
                      << " last_bounds=[" << last_bounds.low[0] << "," << last_bounds.high[0]
                      << "]x[" << last_bounds.low[1] << "," << last_bounds.high[1] << "]"
                      << " goal_in_last=" << goal_in_last << std::endl;
        }

        ob::PlannerStatus status = planner->solve(
            ob::timedPlannerTerminationCondition(config_.planning_time_limit));

        std::cout << "  Robot " << robot_idx << ": guided planner status: " << status << std::endl;
        std::cout << "  Robot " << robot_idx << ": extensions: " << planner->hitExtensionLimit() << std::endl;

        if (config_.max_initial_extensions > 0 &&
            planner->hitExtensionLimit() &&
            status != ob::PlannerStatus::EXACT_SOLUTION)
        {
            int stuck_idx = planner->getStuckRegionIdx();
            const auto& dpath = high_level_paths_[robot_idx];
            if (stuck_idx >= 0 && stuck_idx < (int)dpath.size() - 1) {
                int from_r = dpath[stuck_idx];
                int to_r   = dpath[stuck_idx + 1];
                DOUT << "  Robot " << robot_idx << ": stuck on edge ("
                     << from_r << " -> " << to_r << "), marking forbidden" << std::endl;
                forbidden_edges_.insert({from_r, to_r});
                forbidden_edges_.insert({to_r, from_r});
            }
        }

        if (config_.max_no_progress_iters > 0 &&
            planner->hitNoProgressLimit() &&
            status != ob::PlannerStatus::EXACT_SOLUTION)
        {
            int stuck_idx = planner->getNoProgressStuckRegionIdx();
            const auto& dpath = high_level_paths_[robot_idx];
            if (stuck_idx > 0 && stuck_idx < (int)dpath.size()) {
                int from_r = dpath[stuck_idx - 1];
                int to_r   = dpath[stuck_idx];
                DOUT << "  Robot " << robot_idx << ": no coverage progress, marking edge ("
                     << from_r << " -> " << to_r << ") forbidden" << std::endl;
                forbidden_edges_.insert({from_r, to_r});
                forbidden_edges_.insert({to_r, from_r});
            }
        }

        // std::cout << "  Robot " << robot_idx << ": guided planner status: " << status << std::endl;

        if (status == ob::PlannerStatus::EXACT_SOLUTION || status == ob::PlannerStatus::APPROXIMATE_SOLUTION) {
            auto path = pdef->getSolutionPath()->as<og::PathGeometric>();
            path->interpolate();

            guided_planning_results_[robot_idx].success = true;
            // guided_planning_results_[robot_idx].planning_time = ;
            guided_planning_results_[robot_idx].path = std::make_shared<og::PathGeometric>(*path);
            DOUT << "  Robot " << robot_idx << ": solved with "
                      << guided_planning_results_[robot_idx].path->getStateCount() << " states" << std::endl;
        }
        else {
            guided_planning_results_[robot_idx].success = false;
            DOUT << "  Robot " << robot_idx << ": FAILED" << std::endl;
            return;
        }
    }

    DOUT << "Guided planner results: " << guided_planning_results_.size() << std::endl;

    // Populate robot_paths_ directly from guided results
    robot_paths_.resize(robots_.size());
    for (size_t r = 0; r < guided_planning_results_.size(); ++r) {
        robot_paths_[r] = guided_planning_results_[r].success ? guided_planning_results_[r].path : nullptr;
    }

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
            DOUT << "[viz] low_level_paths event for robot " << r
                      << " written to " << viz_file_ << std::endl;
        }
    }
}

ob::State* CipherGeometricPlanner::getStateAtTimestep(size_t robot_idx, int timestep) const {
    if (robot_idx >= robot_paths_.size() || !robot_paths_[robot_idx]) return nullptr;
    const auto& path = robot_paths_[robot_idx];
    size_t count = path->getStateCount();
    if (count == 0) return nullptr;
    size_t idx = static_cast<size_t>(std::max(0, std::min(timestep, static_cast<int>(count) - 1)));
    return path->getState(idx);
}

bool CipherGeometricPlanner::checkPathsForConflicts() {
    segment_conflicts_.clear();

    if (robot_paths_.empty()) return false;

    bool any_path = false;
    for (const auto& p : robot_paths_) {
        if (p && p->getStateCount() > 0) { any_path = true; break; }
    }
    if (!any_path) return false;

    int max_timestep = 0;
    for (size_t r = 0; r < robot_paths_.size(); ++r) {
        if (robot_paths_[r])
            max_timestep = std::max(max_timestep, static_cast<int>(robot_paths_[r]->getStateCount()));
    }
    if (max_timestep == 0) return false;

    if (robots_.size() < 2) return false;

    for (int timestep = 0; timestep < max_timestep; ++timestep) {
        for (size_t i = 0; i < robots_.size(); ++i) {
            ob::State* si = getStateAtTimestep(i, timestep);
            if (!si) continue;
            for (size_t j = i + 1; j < robots_.size(); ++j) {
                ob::State* sj = getStateAtTimestep(j, timestep);
                if (!sj) continue;
                size_t part_i, part_j;
                if (checkTwoRobotConflict(i, si, j, sj, part_i, part_j)) {
                    SegmentConflict conflict;
                    conflict.type = SegmentConflict::ROBOT_ROBOT;
                    conflict.robot_index_1 = i;
                    conflict.robot_index_2 = j;
                    conflict.timestep = timestep;
                    conflict.part_index_1 = part_i;
                    conflict.part_index_2 = part_j;
                    segment_conflicts_.push_back(conflict);
                }
            }
        }
    }

    vizEmitConflicts(segment_conflicts_);
    return !segment_conflicts_.empty();
}

bool CipherGeometricPlanner::resolveConflicts() {
    DOUT << "Resolving conflicts..." << std::endl;
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

        DOUT << "Resolving conflict " << resolution_stats_.total_conflicts_encountered << ": Robots "
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

    DOUT << "All conflicts resolved successfully after " << resolution_stats_.total_conflicts_resolved << " conflict resolutions" << std::endl;
    return true;
}

// Private helpers
void CipherGeometricPlanner::setupDecomposition() {
    DOUT << "Setting up decomposition..." << std::endl;
    /// TODO: We need to remove the hardcoded 2D assumption. Need a way to get the dimension of the workspace.
    auto decomp = std::make_shared<GridDecompositionImpl>(2, workspace_bounds_, config_.decomposition_region_length);
    decomp->setStateSpace(robots_[0]->getSpaceInformation()->getStateSpace());
    decomp_ = decomp;

    // Seed the viz ID map: every initial coarse cell maps to "c{id}".
    region_viz_id_.clear();
    for (int r = 0; r < decomp_->getNumRegions(); ++r)
        region_viz_id_[r] = "c" + std::to_string(r);

    config_.conflict_resolution_config.max_refinement_levels = calculateMaxRefinementLevels();
}

void CipherGeometricPlanner::separateStartCells() {
    DOUT << "Checking for robots sharing start cells..." << std::endl;

    auto grid_decomp = std::dynamic_pointer_cast<GridDecompositionImpl>(decomp_);
    int max_levels = config_.conflict_resolution_config.max_refinement_levels;

    // Tracks cells that were produced by a prior start-separation split.
    // A cell in this set must not be decomposed again here — each original
    // conflicting start cell is split at most once.
    std::set<int> once_decomposed;

    while (true) {
        // Locate the finest-grained (leaf) cell for each robot's start state.
        std::vector<int> start_regions(start_states_.size());
        for (size_t i = 0; i < start_states_.size(); ++i)
            start_regions[i] = decomp_->locateSubRegion(start_states_[i]);

        // Collect cells that contain more than one robot.
        std::map<int, std::vector<size_t>> cell_to_robots;
        for (size_t i = 0; i < start_regions.size(); ++i)
            cell_to_robots[start_regions[i]].push_back(i);

        std::vector<int> to_decompose;
        for (auto& [cell, robots] : cell_to_robots)
            if (robots.size() > 1)
                to_decompose.push_back(cell);

        if (to_decompose.empty()) break;  // All robots in distinct cells.

        // Before decomposing, check we haven't hit the depth limit.
        for (int cell : to_decompose) {
            if (decomp_->getDecompositionDepth(cell) >= max_levels) {
                throw std::runtime_error(
                    "separateStartCells: cannot separate robots — cell " +
                    std::to_string(cell) + " reached maximum decomposition depth (" +
                    std::to_string(max_levels) + ")");
            }
        }

        // Filter out cells already produced by a prior split — each cell is
        // only ever decomposed once for start separation.
        std::vector<int> filtered;
        for (int cell : to_decompose)
            if (!once_decomposed.count(cell))
                filtered.push_back(cell);

        if (filtered.empty()) {
            throw std::runtime_error(
                "separateStartCells: cannot separate robots — start positions "
                "are too close; cells already decomposed once");
        }

        // Decompose each conflicting cell and update bookkeeping.
        std::vector<std::string> removed_viz_ids;
        std::vector<std::tuple<std::string, std::vector<double>, std::vector<double>>> new_cells;

        for (int r : filtered) {
            std::string parent_viz_id =
                region_viz_id_.count(r) ? region_viz_id_[r] : "c" + std::to_string(r);
            removed_viz_ids.push_back(parent_viz_id);

            decomp_->Decompose(r);

            region_viz_id_.erase(r);
            for (int child : grid_decomp->getChildRegions(r)) {
                once_decomposed.insert(child);
                region_viz_id_[child] = parent_viz_id + "_" + std::to_string(child);
                if (do_viz_) {
                    auto cb = decomp_->getCellBounds(child);
                    new_cells.emplace_back(
                        region_viz_id_[child],
                        std::vector<double>(cb.low.begin(), cb.low.end()),
                        std::vector<double>(cb.high.begin(), cb.high.end()));
                }
            }
        }

        if (do_viz_)
            vizEmitGridUpdate(removed_viz_ids, new_cells);

        DOUT << "  Decomposed " << filtered.size()
             << " start cell(s); re-checking..." << std::endl;
    }

    DOUT << "  All robots in distinct start cells." << std::endl;
}

bool CipherGeometricPlanner::decomposeAllLeavesOneLevel() {
    DOUT << "  Decomposing all leaf cells one level deeper..." << std::endl;
    auto grid_decomp = std::dynamic_pointer_cast<GridDecompositionImpl>(decomp_);
    int max_levels = config_.conflict_resolution_config.max_refinement_levels;

    std::function<void(int, std::vector<int>&)> collectLeaves = [&](int rid, std::vector<int>& out) {
        if (!grid_decomp->hasDecomposed(rid)) { out.push_back(rid); return; }
        for (int child : grid_decomp->getChildRegions(rid)) collectLeaves(child, out);
    };

    std::vector<int> leaves;
    for (int r = 0; r < decomp_->getNumRegions(); ++r)
        collectLeaves(r, leaves);

    // Only decompose the shallowest leaves so that cells already split by
    // separateStartCells() are not pushed one level ahead of everything else.
    int min_depth = std::numeric_limits<int>::max();
    for (int r : leaves)
        min_depth = std::min(min_depth, decomp_->getDecompositionDepth(r));

    if (min_depth >= max_levels)
        return false;

    std::vector<int> to_decompose;
    for (int r : leaves)
        if (decomp_->getDecompositionDepth(r) == min_depth)
            to_decompose.push_back(r);

    std::vector<std::string> removed_viz_ids;
    std::vector<std::tuple<std::string, std::vector<double>, std::vector<double>>> new_cells;

    for (int r : to_decompose) {
        std::string parent_viz_id =
            region_viz_id_.count(r) ? region_viz_id_[r] : "c" + std::to_string(r);
        removed_viz_ids.push_back(parent_viz_id);

        decomp_->Decompose(r);
        region_viz_id_.erase(r);

        for (int child : grid_decomp->getChildRegions(r)) {
            region_viz_id_[child] = parent_viz_id + "_" + std::to_string(child);
            if (do_viz_) {
                auto cb = decomp_->getCellBounds(child);
                new_cells.emplace_back(
                    region_viz_id_[child],
                    std::vector<double>(cb.low.begin(), cb.low.end()),
                    std::vector<double>(cb.high.begin(), cb.high.end()));
            }
        }
    }

    if (do_viz_)
        vizEmitGridUpdate(removed_viz_ids, new_cells);

    DOUT << "  Decomposed " << to_decompose.size() << " leaf cell(s)." << std::endl;
    return true;
}

void CipherGeometricPlanner::setupCollisionManager() {
    DOUT << "Setting up conflict manager..." << std::endl;
    collision_manager_ = std::make_shared<fcl::DynamicAABBTreeCollisionManagerf>();
    collision_manager_->registerObjects(obstacles_);
    collision_manager_->setup();
}

void CipherGeometricPlanner::setupRobots() {
    DOUT << "Setting up robots..." << std::endl;
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
    DOUT << "Cleaning up..." << std::endl;
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
    robot_paths_.clear();
    collision_manager_.reset();
    problem_loaded_ = false;
    resolution_stats_ = ResolutionStats();  // Reset resolution statistics
    robot_pair_conflict_counts_.clear();   // Reset cycle detection counters
    decomposition_hierarchy_.clear();       // Clear decomposition hierarchy
    region_viz_id_.clear();                 // Reset viz ID map
}

// Private conflict checking helpers
// std::vector<fcl::CollisionObjectf*> CipherGeometricPlanner::getObstaclesInRegion(
//     const std::vector<double>& region_min,
//     const std::vector<double>& region_max) const {
//     DOUT << "Getting obstacles in region..." << std::endl;
//     return {};
// }


bool CipherGeometricPlanner::checkTwoRobotConflict(size_t robot_idx_1, const ob::State* state_1,
                                               size_t robot_idx_2, const ob::State* state_2,
                                               size_t& part_1, size_t& part_2) const {
    // DOUT << "Checking two-robot conflict..." << std::endl;
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
// void CipherGeometricPlanner::updateDecomposition() {
//     DOUT << "Updating decomposition..." << std::endl;
// }

// void CipherGeometricPlanner::expandSubproblem() {
//     DOUT << "Expanding subproblem..." << std::endl;
// }

GeometricPlanningResult CipherGeometricPlanner::useCompositePlanner(
    const std::vector<size_t>& robot_indices,
    const std::vector<std::vector<double>>& subproblem_starts,
    const std::vector<std::vector<double>>& subproblem_goals,
    const std::vector<double>& subproblem_env_min,
    const std::vector<double>& subproblem_env_max) {
    DOUT << "Using composite planner for " << robot_indices.size() << " robots..." << std::endl;

    GeometricPlanningResult result;
    result.solved = false;
    result.planning_time = 0.0;

    // Build the YAML environment node that CoupledRRTPlanner::plan() expects
    YAML::Node env_yaml;

    // Environment bounds
    YAML::Node env_min_node, env_max_node;
    for (double v : subproblem_env_min) env_min_node.push_back(v);
    for (double v : subproblem_env_max) env_max_node.push_back(v);
    env_yaml["environment"]["min"] = env_min_node;
    env_yaml["environment"]["max"] = env_max_node;

    // Obstacles — extract box geometry from the FCL objects
    YAML::Node obs_list;
    for (const auto* co : obstacles_) {
        const auto* box = dynamic_cast<const fcl::Boxf*>(co->getCollisionGeometry());
        if (!box) continue;

        YAML::Node obs;
        obs["type"] = "box";

        YAML::Node size_node;
        size_node.push_back(static_cast<double>(box->side[0]));
        size_node.push_back(static_cast<double>(box->side[1]));
        obs["size"] = size_node;

        YAML::Node center_node;
        center_node.push_back(static_cast<double>(co->getTranslation()[0]));
        center_node.push_back(static_cast<double>(co->getTranslation()[1]));
        obs["center"] = center_node;

        obs_list.push_back(obs);
    }
    env_yaml["environment"]["obstacles"] = obs_list;

    // Per-robot entries
    YAML::Node robots_node;
    for (size_t i = 0; i < robot_indices.size(); ++i) {
        size_t r = robot_indices[i];
        YAML::Node robot_node;
        robot_node["type"] = robot_types_[r];

        YAML::Node start_node;
        for (double v : subproblem_starts[i]) start_node.push_back(v);
        robot_node["start"] = start_node;

        YAML::Node goal_node;
        for (double v : subproblem_goals[i]) goal_node.push_back(v);
        robot_node["goal"] = goal_node;

        robots_node.push_back(robot_node);
    }
    env_yaml["robots"] = robots_node;

    // Configure and run the coupled RRT planner
    CoupledRRTConfig rrt_config;
    rrt_config.time_limit     = config_.planning_time_limit;
    rrt_config.goal_threshold = config_.goal_threshold;
    rrt_config.seed           = config_.seed;

    CoupledRRTPlanner composite_planner(rrt_config);
    YAML::Node output = composite_planner.plan(env_yaml);

    result.solved        = output["solved"].as<bool>();
    result.planning_time = output["planning_time"].as<double>();

    if (!result.solved || !output["result"]) {
        return result;
    }

    // Reconstruct individual OMPL paths from the YAML state sequences
    const auto& yaml_result = output["result"];
    for (size_t i = 0; i < robot_indices.size(); ++i) {
        size_t r = robot_indices[i];
        auto si = robots_[r]->getSpaceInformation();
        auto path = std::make_shared<og::PathGeometric>(si);

        for (const auto& state_node : yaml_result[i]["states"]) {
            std::vector<double> reals;
            for (const auto& v : state_node) reals.push_back(v.as<double>());

            ob::State* s = si->getStateSpace()->allocState();
            si->getStateSpace()->copyFromReals(s, reals);
            path->append(s);
            si->getStateSpace()->freeState(s);
        }

        result.individual_paths.push_back(path);
    }

    return result;
}

bool CipherGeometricPlanner::resolveConflictWithStrategies(const SegmentConflict& conflict,
                                                            ConflictResolutionEntry& log_entry) {
    DOUT << "Resolving conflict with strategies..." << std::endl;
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
            DOUT << "  Cycle detection: robot pair (" << std::get<0>(pair_key) << ", "
                      << std::get<1>(pair_key) << ") at timestep " << std::get<2>(pair_key)
                      << " conflict #" << pair_count
                      << ", escalating to min expansion layer " << min_expansion_layer
                      << std::endl;
        }
    }

    // Strategy 1: Hierarchical Expansion + Refinement
    // This combines the old decomposition refinement and subproblem expansion into one
    // unified hierarchical approach:
    // - expansion_layer=0: refine just the conflict cell (K times)
    // - expansion_layer=1,2,...: expand to neighbors, then refine all cells (K times each)
    // - Continue until expansion covers the whole decomposition
    if (config.max_refinement_levels > 0) {
        DOUT << "  Trying hierarchical expansion+refinement (max "
                  << config.max_refinement_levels << " refinement levels, max "
                  << max_expansion_layers << " expansion layers, min "
                  << min_expansion_layer << " expansion layer)..." << std::endl;

        if (resolveWithHierarchicalExpansionRefinement(
                conflict,
                config.max_refinement_levels,
                max_expansion_layers,
                min_expansion_layer,
                log_entry)) {
            DOUT << "  Hierarchical expansion+refinement resolved the conflict" << std::endl;
            return true;
        }

        // Check if we timed out during hierarchical resolution
        if (isTimeoutExceeded()) {
            std::cerr << "  Timeout during hierarchical expansion+refinement" << std::endl;
            return false;
        }

        DOUT << "  Hierarchical expansion+refinement exhausted, escalating to local composite..." << std::endl;
    }

    // Strategy 2: Full-Problem Composite Planner (ALL robots, original starts/goals)
    if (config.max_composite_attempts > 0) {
        DOUT << "  Trying full-problem composite planner (max "
                  << config.max_composite_attempts << " attempts)..." << std::endl;
        resolution_stats_.composite_planner_attempts++;

        if (resolveWithFullProblemCompositePlanner(config.max_composite_attempts, log_entry)) {
            DOUT << "  Full-problem composite planner resolved the conflict" << std::endl;
            resolution_stats_.composite_planner_successes++;
            return true;
        }

        std::cerr << "  Full-problem composite planner failed after " << config.max_composite_attempts
                  << " attempts for conflict at timestep " << conflict.timestep << std::endl;
    }

    // All strategies exhausted - conflict could not be resolved
    std::cerr << "  All conflict resolution strategies exhausted for conflict at timestep "
              << conflict.timestep << " between robots " << conflict.robot_index_1
              << " and " << conflict.robot_index_2 << std::endl;
    return false;
}

bool CipherGeometricPlanner::conflictPersistsForRobots(size_t robot_1, size_t robot_2, int timestep) const {
    DOUT << "Checking if conflict persists for robots..." << std::endl;
    for (const auto& coll : segment_conflicts_) {
        if (coll.timestep == timestep) {
            if (coll.type == SegmentConflict::ROBOT_ROBOT) {
                if ((coll.robot_index_1 == robot_1 && coll.robot_index_2 == robot_2) ||
                    (coll.robot_index_1 == robot_2 && coll.robot_index_2 == robot_1)) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool CipherGeometricPlanner::resolveWithHierarchicalExpansionRefinement(
    const SegmentConflict& conflict,
    int max_refinement_levels,
    int max_expansion_layers,
    int min_expansion_layer,
    ConflictResolutionEntry& log_entry) {
    DOUT << "Resolving with hierarchical expansion/refinement..." << std::endl;
    size_t robot_1 = conflict.robot_index_1;
    size_t robot_2 = conflict.robot_index_2;

    // Allocate temporary states for conflict location
    ob::State* state_1 = robots_[robot_1]->getSpaceInformation()->getStateSpace()->allocState();
    ob::State* state_2 = robots_[robot_2]->getSpaceInformation()->getStateSpace()->allocState();

    // Locate conflict region
    ob::State* s1 = getStateAtTimestep(robot_1, conflict.timestep);
    ob::State* s2 = getStateAtTimestep(robot_2, conflict.timestep);
    if (!s1 || !s2) {
        DOUT << "    Robot has no path at conflict timestep" << std::endl;
        robots_[robot_1]->getSpaceInformation()->getStateSpace()->freeState(state_1);
        robots_[robot_2]->getSpaceInformation()->getStateSpace()->freeState(state_2);
        return false;
    }
    robots_[robot_1]->getSpaceInformation()->copyState(state_1, s1);
    robots_[robot_2]->getSpaceInformation()->copyState(state_2, s2);
    int conflict_region = decomp_->locateRegion(state_1);

    DOUT << "    Conflict in region " << conflict_region << std::endl;

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

        // Get expanded region for this layer
        std::vector<int> expanded_regions = getExpandedRegion(conflict_region, expansion_layer);
        bool covers_full_decomp = ((int)expanded_regions.size() >= decomp_->getNumRegions());

        DOUT << "    Expansion layer " << expansion_layer
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
        DOUT << "    All refinement levels exhausted at expansion layer " << expansion_layer
                  << ", trying wider expansion..." << std::endl;

        if (covers_full_decomp) {
            DOUT << "    Expansion layer " << expansion_layer
                      << " covers entire decomposition, stopping expansion" << std::endl;
            break;
        }
    }

    // All expansion levels exhausted
    DOUT << "    All expansion layers exhausted" << std::endl;
    return false;
}

bool CipherGeometricPlanner::attemptRefinementAtExpansionLevel(
    const SegmentConflict& conflict,
    int conflict_region,
    const std::vector<int>& expanded_regions,
    int expansion_layer,
    int max_refinement_levels,
    ConflictResolutionEntry& log_entry) {
    DOUT << "Attempting refinement at expansion level..." << std::endl;
    resolution_stats_.decomposition_refinement_attempts++;

    for (int refinement_level = 1; refinement_level <= max_refinement_levels; ++refinement_level) {
        // Check for timeout before each refinement attempt
        if (isTimeoutExceeded()) {
            std::cerr << "      Timeout before refinement level " << refinement_level
                      << " at expansion layer " << expansion_layer << std::endl;
            return false;
        }

        DOUT << "      Refinement level " << refinement_level << "/" << max_refinement_levels
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
                DOUT << "      Conflict resolved at expansion=" << expansion_layer
                          << ", refinement=" << refinement_level << std::endl;
                attempt.conflict_resolved = true;
                log_entry.attempts.push_back(attempt);
                resolution_stats_.decomposition_refinement_successes++;
                return true;
            }

            DOUT << "      Conflict persists after refinement level " << refinement_level
                      << ", trying higher refinement..." << std::endl;
        } else {
            attempt.planning_succeeded = false;
            DOUT << "      Refinement level " << refinement_level << " failed (planning failed)" << std::endl;
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
    DOUT << "Refining expanded region..." << std::endl;

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
        DOUT << "        All " << expanded_regions.size()
                  << " cells already refined at level " << refinement_level
                  << ", skipping" << std::endl;
        return false;
    }

    DOUT << "        " << new_regions.size() << " new cell(s) to refine out of "
              << expanded_regions.size() << " total" << std::endl;

    // Capture viz IDs of cells-to-remove BEFORE they are erased from region_viz_id_.
    std::vector<std::string> removed_viz_ids;
    for (int r : new_regions)
        removed_viz_ids.push_back(region_viz_id_.count(r) ? region_viz_id_[r] : "c" + std::to_string(r));

    // Step 1: Refine the leaf cells in the global decomposition.
    for (int r : new_regions)
        decomp_->Decompose(r);

    DOUT << "        Decomposed " << new_regions.size() << " cell(s) in global decomposition" << std::endl;

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
        DOUT << "        Failed to extract replanning bounds" << std::endl;
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

    // Handle ROBOT_OBSTACLE conflicts (robot_1 == robot_2) as a single-robot replan.
    // Without this, the same robot would be replanned and integrated twice, corrupting the path.
    bool is_single_robot = (robot_1 == robot_2);

    // Determine which robots need replanning (stationary robots have entry == exit timestep)
    bool robot_1_stationary = (update_info_1.start_timestep == update_info_1.end_timestep);
    bool robot_2_stationary = is_single_robot ? true : (update_info_2.start_timestep == update_info_2.end_timestep);

    if (robot_1_stationary && robot_2_stationary) {
        // Both robots are stationary — can't replan either one
        DOUT << "        Both robots are stationary at goal, cannot refine" << std::endl;
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
        DOUT << "        Robot " << robot_1 << " is stationary at goal, only replanning robot " << robot_2 << std::endl;
    } else if (robot_2_stationary) {
        DOUT << "        Robot " << robot_2 << " is stationary at goal, only replanning robot " << robot_1 << std::endl;
    }

    std::vector<int> start_regions;
    std::vector<int> goal_regions;

    // Validate that entry/exit states of robots to replan are within local decomposition bounds
    bool states_in_bounds = true;

    for (int i = 0; i < replan_hl_starts.size(); ++i) {
        // const auto* start_state = replan_starts[i];
        const auto* start_region_state = replan_hl_starts[i];

        // auto planning_region = decomp_->locateSubRegion(start_state);
        start_regions.push_back(decomp_->locateSubRegion(start_region_state));

        if (start_regions.back() < 0) {
            states_in_bounds = false;
            DOUT << "        Entry state outside local decomposition bounds" << std::endl;
            DOUT << "Local Decomp Bounds: ";
            for (size_t n =0; n < decomp_->getBounds().low.size(); ++n) {
                DOUT << "[" << decomp_->getBounds().low[n] << ":" << decomp_->getBounds().high[n] << "]";
            }
            DOUT << " " << std::endl;
            
            robots_[0]->getSpaceInformation()->getStateSpace()->printState(start_region_state, DOUT);
            break;
        }
    }
    if (states_in_bounds) {
        for (int i = 0; i < replan_hl_goals.size(); ++i) {
            // const auto* start_state = replan_starts[i];
            const auto* goal_region_state = replan_hl_goals[i];

            // auto planning_region = decomp_->locateSubRegion(start_state);
            goal_regions.push_back(decomp_->locateSubRegion(goal_region_state));
            if (goal_regions.back() < 0) {
                states_in_bounds = false;
                DOUT << "        Exit state outside local decomposition bounds" << std::endl;
                DOUT << "Local Decomp Bounds: ";
                for (size_t n =0; n < decomp_->getBounds().low.size(); ++n) {
                    DOUT << "[" << decomp_->getBounds().low[n] << ":" << decomp_->getBounds().high[n] << "]";
                }
                DOUT << " " <<std::endl;
                robots_[0]->getSpaceInformation()->getStateSpace()->printState(goal_region_state, DOUT);
                break;
            }
        }
    }

    for (size_t i = 0; i < start_regions.size() || i < goal_regions.size(); ++i) {
        if (i < start_regions.size()) DOUT << "Start[" << i << "]: " << start_regions[i] << "  ";
        if (i < goal_regions.size())  DOUT << "Goal[" << i << "]: " << goal_regions[i];
        DOUT << std::endl;
    }

    std::set<int> sts(start_regions.begin(), start_regions.end());
    std::set<int> gls(goal_regions.begin(), goal_regions.end());
    if (!states_in_bounds) {
        freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
        return false;
    }
    if (sts.size() != start_regions.size()) {
        DOUT << "!!!Same starts!!!" << std::endl;
        freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
        return false;
    }
    if (gls.size() != goal_regions.size()) {
        DOUT << "!!!Same goals!!!" << std::endl;
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
            DOUT << "Robot " << robot_idx << " start region: " << decomp_->locateSubRegion(replan_hl_starts[robot_idx])
                << " -> end region: " << decomp_->locateSubRegion(replan_hl_goals[robot_idx]) << std::endl;
        }

        local_high_level_paths = mapf_solver.solve(
            decomp_, replan_hl_starts, replan_hl_goals, expanded_leaf_regions);
    }

    if (local_high_level_paths.empty()) {
        DOUT << "        MAPF failed" << std::endl;
        freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
        return false;
    }

    // CBS routed to locateSubRegion(region_exit_state), but the low-level planner
    // goal (planning_exit_state) may be one region further (just past the conflict
    // boundary). Extend each path by that region when they differ so the planner
    // is allowed to reach the actual goal.
    for (size_t i = 0; i < local_high_level_paths.size(); ++i) {
        int goal_region = decomp_->locateSubRegion(replan_goals[i]);
        if (goal_region >= 0 && !local_high_level_paths[i].empty() &&
            local_high_level_paths[i].back() != goal_region) {
            local_high_level_paths[i].push_back(goal_region);
            DOUT << "        Extended path for robot " << i << " by goal region " << goal_region << std::endl;
        }
    }

    {
        DOUT << "        MAPF succeeded, got high-level paths for " << local_high_level_paths.size() << " robot(s)" << std::endl;

        // Debug print the high-level paths
        for (size_t i = 0; i < local_high_level_paths.size(); ++i) {
            DOUT << "  Robot " << replan_robot_indices[i] << " high-level path: ";
            for (int sub_id : local_high_level_paths[i]) {
                DOUT << sub_id << " ";
            }
            DOUT << std::endl;
        }
    }


    for (size_t i = 0; i < replan_robot_indices.size(); ++i) {
        if (i >= local_high_level_paths.size() || local_high_level_paths[i].empty()) {
            DOUT << "        MAPF failed for robot " << replan_robot_indices[i] << std::endl;
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
                // DOUT << "r" + std::to_string(replan_robot_indices[i]) << ": " << cid << std::endl;
            }
            paths["r" + std::to_string(replan_robot_indices[i])] = cell_ids;
        }
        ev["paths"] = paths;
        viz_events_.push_back(ev);
        vizWriteFile();
        DOUT << "[viz] local_mapf event written to " << viz_file_ << std::endl;
    }

    // Step 4: Guided planning (only for robots that need replanning)
    std::vector<GuidedPlanningResult> replan_results;
    bool all_succeeded = true;
    {
        for (size_t i = 0; i < replan_robot_indices.size(); ++i) {
            size_t robot_idx = replan_robot_indices[i];

            std::cout << "Guided planning for robot " << robot_idx << " with start state ";
            robots_[robot_idx]->getSpaceInformation()->getStateSpace()->printState(replan_starts[i], std::cout);
            std::cout << " and goal state ";            robots_[robot_idx]->getSpaceInformation()->getStateSpace()->printState(replan_goals[i], std::cout);
            std::cout << std::endl;

            /// TODO: How to update problem def so that we can set new bounds
            auto robot_si = robot_sis_[robot_idx];

            replan_results.push_back(GuidedPlanningResult());
            replan_results[i].robot_index = robot_idx;

            auto pdef = std::make_shared<ob::ProblemDefinition>(robot_si);
            pdef->addStartState(replan_starts[i]);
            pdef->setGoal(std::make_shared<PositionGoalCondition>(
                robot_si, replan_goals[i], config_.goal_threshold
            ));

            float robot_inflation = 0.0f;
            if (config_.restrict_sampling_to_cell) {
                auto robot = robots_[robot_idx];
                for (size_t part = 0; part < robot->numParts(); ++part) {
                    auto geom = robot->getCollisionGeometry(part);
                    if (geom) {
                        geom->computeLocalAABB();
                        const auto& aabb = geom->aabb_local;
                        float hx = (aabb.max_[0] - aabb.min_[0]) / 2.0f;
                        float hy = (aabb.max_[1] - aabb.min_[1]) / 2.0f;
                        robot_inflation = std::max(robot_inflation, std::sqrt(hx * hx + hy * hy));
                    }
                }
            }

            auto planner = std::make_shared<GuidedGeometricRRT>(robot_si);
            planner->setIntermediateStates(true);
            planner->setDecomposition(decomp_);
            planner->setDecompositionPath(local_high_level_paths[i]);
            planner->setRobotInflation(static_cast<double>(robot_inflation));
            planner->setProblemDefinition(pdef);
            planner->setup();

            {
                const auto& lpath = local_high_level_paths[i];
                int last_region = lpath.back();
                auto last_bounds = decomp_->getCellBounds(last_region);
                int goal_region = decomp_->locateSubRegion(replan_goals[i]);
                std::vector<double> goal_reals;
                robot_si->getStateSpace()->copyToReals(goal_reals, replan_goals[i]);
                bool goal_in_last = (goal_reals[0] >= last_bounds.low[0] && goal_reals[0] < last_bounds.high[0] &&
                                     goal_reals[1] >= last_bounds.low[1] && goal_reals[1] < last_bounds.high[1]);
                std::cout << "[cipher] replan robot " << robot_idx
                          << " goal=(" << goal_reals[0] << "," << goal_reals[1] << ")"
                          << " locateSubRegion(goal)=" << goal_region
                          << " path.back()=" << last_region
                          << " last_bounds=[" << last_bounds.low[0] << "," << last_bounds.high[0]
                          << "]x[" << last_bounds.low[1] << "," << last_bounds.high[1] << "]"
                          << " goal_in_last=" << goal_in_last << std::endl;
            }

            // ob::PlannerStatus status;
            ob::PlannerStatus status = planner->solve(
                ob::timedPlannerTerminationCondition(config_.planning_time_limit));
        
            if (status == ob::PlannerStatus::EXACT_SOLUTION ||
                status == ob::PlannerStatus::APPROXIMATE_SOLUTION) {
                auto path = pdef->getSolutionPath()->as<og::PathGeometric>();
                path->interpolate();

                replan_results[i].success = true;
                replan_results[i].path = std::make_shared<og::PathGeometric>(*path);
                DOUT << "  Robot " << robot_idx << ": solved with "
                        << replan_results[i].path->getStateCount() << " states" << std::endl;
            }
            else {
                replan_results[i].success = false;
                DOUT << "  Robot " << robot_idx << ": FAILED" << std::endl;
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
        DOUT << "        Guided planning failed" << std::endl;
        freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
        return false;
    }

    // Success! Integrate refined paths (only for robots that were replanned)
    DOUT << "        Planning succeeded, integrating refined paths" << std::endl;

    // Step 5: Integrate refined paths for robots that were replanned
    {
        for (size_t i = 0; i < replan_robot_indices.size(); ++i) {
            if (replan_results[i].path != nullptr) {
                std::vector<size_t> single_robot = {replan_robot_indices[i]};
                std::vector<GuidedPlanningResult> single_result = {replan_results[i]};
                PathUpdateInfo& update_info = (replan_to_collision_idx[i] == 0) ? update_info_1 : update_info_2;
                integrateRefinedPaths(single_robot, single_result, update_info, update_info);
            }
        }

        // Invalidate cached bounds for this robot pair — paths were just updated,
        // so any stored PathUpdateInfo (including planning_exit_state) is now stale.
        robot_pair_refinement_info.erase(std::make_pair(robot_1, robot_2));
        robot_pair_refinement_info.erase(std::make_pair(robot_2, robot_1));

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
                DOUT << "[viz] refined low_level_paths event for robot " << r
                          << " written to " << viz_file_ << std::endl;
            }
        }
    }

    // Step 6: Re-check collisions
    {
        recheckConflictsFromTimestep(getRecheckStartTimestep(conflict, update_info_1, update_info_2));
    }

    freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);

    return true;
}

int CipherGeometricPlanner::calculateMaxExpansionLayers() const {
    DOUT << "Calculating max expansion layers..." << std::endl;

    // getExpandedRegion uses getAllNeighbors (8-connectivity / Chebyshev distance).
    // The diameter of an NxN grid under Chebyshev distance is N-1 (corner to corner).
    // We need this worst-case bound because the conflict cell can be anywhere.
    int dim = decomp_->getDimension();
    int num_regions = decomp_->getNumRegions();
    int grid_side = static_cast<int>(std::round(std::pow(num_regions, 1.0 / dim)));
    int max_layers = grid_side - 1;

    DOUT << "  Grid: " << num_regions << " regions, side=" << grid_side
              << ", max expansion layers=" << max_layers << std::endl;
    return max_layers;
}

int CipherGeometricPlanner::calculateMaxRefinementLevels() const {
    DOUT << "Calculating max refinement levels..." << std::endl;

    float robot_max_dim = 0.0f;
    for (const auto& robot : robots_) {
        for (size_t part = 0; part < robot->numParts(); ++part) {
            auto geom = robot->getCollisionGeometry(part);
            if (geom) {
                geom->computeLocalAABB();
                const auto& aabb = geom->aabb_local;
                for (int d = 0; d < 3; ++d)
                    robot_max_dim = std::max(robot_max_dim, aabb.max_[d] - aabb.min_[d]);
            }
        }
    }

    // All grid cells are the same size, so region 0 is a valid representative.
    int max_levels = decomp_->getMaxDecompositions(0, static_cast<double>(robot_max_dim), config_.robot_cell_size_ratio);
    DOUT << "  Robot max dim: " << robot_max_dim
              << ", max refinement levels=" << max_levels << std::endl;
    return max_levels;
}


void CipherGeometricPlanner::freeUpdateInfoStates(
    size_t robot_1, size_t robot_2,
    PathUpdateInfo& update_info_1,
    PathUpdateInfo& update_info_2) {
    DOUT << "Freeing update info states..." << std::endl;
}

// Private helpers for all conflict strategies
// std::shared_ptr<DecompositionImpl> CipherGeometricPlanner::createLocalDecomposition(
//     int parent_region,
//     double subdivision_factor) {
//     DOUT << "Creating local decomposition..." << std::endl;
//     const auto parent_bounds = decomp_->getCellBounds(parent_region);
//     int sf = static_cast<int>(subdivision_factor);
//     int dim = decomp_->getDimension();

//     ob::RealVectorBounds local_bounds(dim);
//     for (int i = 0; i < dim; ++i) {
//         local_bounds.setLow(i, parent_bounds.low[i]);
//         local_bounds.setHigh(i, parent_bounds.high[i]);
//     }

//     DOUT << "    Creating local decomposition: " << sf << "x" << sf << " grid" << std::endl;

//     // Single original cell: sf sub-cells per dimension (square cell → square sub-cells)
//     auto space = robots_[0]->getSpaceInformation()->getStateSpace();
//     auto decomp_ = std::make_shared<RectGridDecompositionImpl>(
//         std::vector<int>(dim, sf), local_bounds, space);

//     // recordRefinement(parent_region, decomp_);
//     return decomp_;
// }

// std::shared_ptr<DecompositionImpl> CipherGeometricPlanner::createMultiCellDecomposition(
//     const std::vector<int>& regions,
//     double subdivision_factor) {
//     DOUT << "Creating multi-cell decomposition..." << std::endl;
//     int sf = static_cast<int>(subdivision_factor);
//     int dim = decomp_->getDimension();

//     // Get the original cell size (all cells are square with the same size)
//     const auto first_bounds = decomp_->getCellBounds(regions[0]);
//     double cell_size = first_bounds.high[0] - first_bounds.low[0];

//     // Compute the bounding box of all original cells in the expanded region
//     std::vector<double> env_min, env_max;
//     computeExpandedBounds(regions, env_min, env_max);

//     ob::RealVectorBounds expanded_bounds(dim);
//     for (int i = 0; i < dim; ++i) {
//         expanded_bounds.setLow(i, env_min[i]);
//         expanded_bounds.setHigh(i, env_max[i]);
//     }

//     // For each original cell, the refinement produces sf sub-cells per dimension.
//     // Compute the total sub-cell count per dimension from the number of original
//     // cells along each axis (n_i * sf). This correctly handles non-square expanded
//     // regions (e.g., 3×2 original cells at grid boundaries).
//     std::vector<int> grid_lengths(dim);
//     for (int i = 0; i < dim; ++i) {
//         int n_i = static_cast<int>(std::round((env_max[i] - env_min[i]) / cell_size));
//         grid_lengths[i] = n_i * sf;
//     }

//     DOUT << "      Multi-cell decomposition: " << grid_lengths[0] << "x" << grid_lengths[1]
//               << " grid (" << (grid_lengths[0] * grid_lengths[1]) << " regions)" << std::endl;

//     auto space = robots_[0]->getSpaceInformation()->getStateSpace();
//     auto multi_cell_decomp = std::make_shared<RectGridDecompositionImpl>(
//         grid_lengths, expanded_bounds, space);

//     for (int region : regions)
//         recordRefinement(region, multi_cell_decomp);

//     return multi_cell_decomp;
// }

// bool CipherGeometricPlanner::extractReplanningBounds(
//     const SegmentConflict& conflict,
//     int conflict_region,
//     PathUpdateInfo& update_info_1,
//     PathUpdateInfo& update_info_2) {
//     DOUT << "Extracting replanning bounds..." << std::endl;
//     return false;
// }

bool CipherGeometricPlanner::extractReplanningBoundsForExpandedRegion(
    const SegmentConflict& conflict,
    const std::vector<int>& expanded_regions,
    PathUpdateInfo& update_info_1,
    PathUpdateInfo& update_info_2)
{
    size_t robot_1 = conflict.robot_index_1;
    size_t robot_2 = conflict.robot_index_2;

    // Convert expanded_regions to a set for O(1) lookup
    std::set<int> region_set(expanded_regions.begin(), expanded_regions.end());

    // Helper lambda to extract bounds for one robot
    auto extractForRobot = [&](size_t robot_idx, PathUpdateInfo& info) -> bool {
        info.robot_index = robot_idx;

        auto si = robots_[robot_idx]->getSpaceInformation();

        int conflict_ts = conflict.timestep;

        // Check if robot has finished its path before the conflict timestep.
        int path_end_timestep = robot_paths_[robot_idx]
            ? static_cast<int>(robot_paths_[robot_idx]->getStateCount()) : 0;

        if (conflict_ts >= path_end_timestep) {
            // Robot is stationary at goal — all states collapse to the goal
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
            info.planning_entry_timestep = path_end_timestep;
            return true;
        }

        // Find entry to expanded region (scan backwards from conflict)
        int entry_timestep = 0;
        bool found_entry = false;
        int pre_entry_timestep = -1;

        for (int t = conflict_ts; t >= 0; --t) {
            ob::State* st = getStateAtTimestep(robot_idx, t);
            if (!st) break;
            int region = decomp_->locateRegion(st);
            if (region_set.find(region) == region_set.end()) {
                entry_timestep = t + 1;
                pre_entry_timestep = t;
                found_entry = true;
                break;
            }
        }

        if (!found_entry) {
            entry_timestep = 0;
        }

        // Find exit from expanded region (scan forwards).
        int exit_timestep = -1;
        bool found_exit = false;
        int last_in_region_timestep = -1;
        int max_timestep = path_end_timestep;

        for (int t = conflict_ts; t < max_timestep; ++t) {
            ob::State* st = getStateAtTimestep(robot_idx, t);
            if (!st) break;
            int region = decomp_->locateRegion(st);
            if (region_set.find(region) == region_set.end()) {
                exit_timestep = t;
                found_exit = true;
                last_in_region_timestep = t-1;
                break;
            }
            
        }

        if (!found_exit) {
            exit_timestep = max_timestep;
        }

        // Allocate all four boundary states.
        info.region_entry_state = si->getStateSpace()->allocState();
        info.region_exit_state = si->getStateSpace()->allocState();
        info.planning_entry_state = si->getStateSpace()->allocState();
        info.planning_exit_state = si->getStateSpace()->allocState();

        if (!found_entry) {
            si->copyState(info.region_entry_state, start_states_[robot_idx]);
            si->copyState(info.planning_entry_state, start_states_[robot_idx]);
        } else {
            si->copyState(info.region_entry_state, getStateAtTimestep(robot_idx, entry_timestep));
            si->copyState(info.planning_entry_state, getStateAtTimestep(robot_idx, pre_entry_timestep));
        }

        if (exit_timestep >= max_timestep) {
            si->copyState(info.planning_exit_state, goal_states_[robot_idx]);
            si->copyState(info.region_exit_state, goal_states_[robot_idx]);
        } else {
            si->copyState(info.planning_exit_state, getStateAtTimestep(robot_idx, exit_timestep));
            if (last_in_region_timestep >= 0) {
                si->copyState(info.region_exit_state, getStateAtTimestep(robot_idx, last_in_region_timestep));
            } else {
                si->copyState(info.region_exit_state, info.planning_exit_state);
            }
        }

        si->getStateSpace()->printState(info.planning_entry_state, std::cout);
        si->getStateSpace()->printState(info.region_entry_state, std::cout);

        si->getStateSpace()->printState(info.planning_exit_state, std::cout);
        si->getStateSpace()->printState(info.region_exit_state, std::cout);

        info.start_timestep = entry_timestep;
        info.end_timestep = exit_timestep;
        info.planning_entry_timestep = (pre_entry_timestep >= 0) ? pre_entry_timestep : 0;

        return true;
    };

    auto robot_key = std::make_pair(robot_1, robot_2);
    auto& region_cache = robot_pair_refinement_info[robot_key];
    auto cache_it = region_cache.find(expanded_regions);
    if (cache_it != region_cache.end()) {
        DOUT << "Cache hit for robot pair (" << robot_1 << ", " << robot_2
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
    DOUT << "Integrating refined paths..." << std::endl;
    std::vector<PathUpdateInfo> update_infos = {update_info_1, update_info_2};

    for (size_t i = 0; i < robot_indices.size(); ++i) {
        size_t robot_idx = robot_indices[i];
        const auto& result = local_results[i];
        const auto& update_info = update_infos[i];

        DOUT << "    Integrating refined path for robot " << robot_idx << std::endl;

        // Note: We'll re-segment the entire path after splicing the PathControl below
        // This is necessary because segment state pointers become invalid after path updates

        // Splice the full PathControl for guided_planning_results
        // We need to construct a complete path: before + refined + after
        auto si = robots_[robot_idx]->getSpaceInformation();
        auto original_path = guided_planning_results_[robot_idx].path;

        if (original_path) {
            DOUT << "      Original path has " << original_path->getStateCount() << " states" << std::endl;
        }
        if (result.path) {
            DOUT << "      Refined path has " << result.path->getStateCount() << " states" << std::endl;
        }

        if (original_path && result.path) {
            auto spliced_path = std::make_shared<og::PathGeometric>(si);

            // Part 1: copy original up to and including the entry state
            size_t entry_state_idx = 0;
            for (size_t s = 0; s < original_path->getStateCount(); ++s) {
                if (si->getStateSpace()->distance(original_path->getState(s), update_info.planning_entry_state) < 1e-3) {
                    DOUT << "!!Found Start!!" << std::endl;
                    entry_state_idx = s;
                    break;
                }
            }
            for (size_t s = 0; s <= entry_state_idx; ++s)
                spliced_path->append(original_path->getState(s));

            // Part 2: all refined path states
            for (size_t s = 0; s < result.path->getStateCount(); ++s)
                spliced_path->append(result.path->getState(s));

            // Part 3: original path after the exit state
            size_t exit_state_idx = original_path->getStateCount() - 1;
            for (size_t s = entry_state_idx; s < original_path->getStateCount(); ++s) {
                if (si->getStateSpace()->distance(original_path->getState(s), update_info.planning_exit_state) < 1e-3) {
                    DOUT << "!!Found Exit!!" << std::endl;
                    exit_state_idx = s;
                    break;
                }
            }
            for (size_t s = exit_state_idx + 1; s < original_path->getStateCount(); ++s)
                spliced_path->append(original_path->getState(s));

            DOUT << "      Spliced path has " << spliced_path->getStateCount() << " states" << std::endl;

            guided_planning_results_[robot_idx].path = spliced_path;
            robot_paths_[robot_idx] = spliced_path;
        } else {
            guided_planning_results_[robot_idx] = result;
            robot_paths_[robot_idx] = result.path;
        }
    }
}

void CipherGeometricPlanner::recheckConflictsFromTimestep(int start_timestep) {
    DOUT << "Re-checking conflicts from timestep " << start_timestep << std::endl;

    segment_conflicts_.clear();

    int max_timestep = 0;
    for (size_t r = 0; r < robot_paths_.size(); ++r) {
        if (robot_paths_[r])
            max_timestep = std::max(max_timestep, static_cast<int>(robot_paths_[r]->getStateCount()));
    }

    if (robots_.size() < 2) return;

    for (int timestep = start_timestep; timestep < max_timestep; ++timestep) {
        for (size_t i = 0; i < robots_.size(); ++i) {
            ob::State* si = getStateAtTimestep(i, timestep);
            if (!si) continue;
            for (size_t j = i + 1; j < robots_.size(); ++j) {
                ob::State* sj = getStateAtTimestep(j, timestep);
                if (!sj) continue;
                size_t part_i, part_j;
                if (checkTwoRobotConflict(i, si, j, sj, part_i, part_j)) {
                    SegmentConflict coll;
                    coll.type = SegmentConflict::ROBOT_ROBOT;
                    coll.robot_index_1 = i;
                    coll.robot_index_2 = j;
                    coll.timestep = timestep;
                    coll.part_index_1 = part_i;
                    coll.part_index_2 = part_j;
                    segment_conflicts_.push_back(coll);
                    DOUT << "    Found robot-robot conflict at timestep " << timestep << std::endl;
                    vizEmitConflicts(segment_conflicts_);
                    return;
                }
            }
        }
    }

    DOUT << "    Total conflicts found: " << segment_conflicts_.size() << std::endl;
}

int CipherGeometricPlanner::getRecheckStartTimestep(
    const SegmentConflict& conflict,
    const PathUpdateInfo& update_info_1,
    const PathUpdateInfo& update_info_2) {
    DOUT << "Getting recheck start timestep..." << std::endl;

    int result = std::min(update_info_1.planning_entry_timestep,
                          update_info_2.planning_entry_timestep);

    DOUT << "    Recheck start: using planning_entry_timestep " << result
              << " (conflict timestep was " << conflict.timestep << ")" << std::endl;

    return result;
}

// Private helpers for subproblem expansion strategies
std::vector<int> CipherGeometricPlanner::getExpandedRegion(int center_region, int expansion_layers) {
    DOUT << "Getting expanded region..." << std::endl;
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
    DOUT << "Computing expanded bounds..." << std::endl;
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
// bool CipherGeometricPlanner::extractIndividualPaths(
//     const std::shared_ptr<og::PathGeometric>& compound_path,
//     std::vector<std::shared_ptr<og::PathGeometric>>& individual_paths) {
//     DOUT << "Extracting individual paths from compound path..." << std::endl;
//     return false;
// }

// // Private helpers for decomposition hierarchy tracking
// void CipherGeometricPlanner::initializeDecompositionHierarchy() {
//     DOUT << "Initializing decomposition hierarchy..." << std::endl;
// }

// DecompositionCell* CipherGeometricPlanner::findCellByRegion(int region_id) {
//     DOUT << "Finding cell by region ID..." << std::endl;
//     return nullptr;
// }

// DecompositionCell* CipherGeometricPlanner::findCellByRegionRecursive(DecompositionCell& cell, int region_id) {
//     DOUT << "Finding cell by region ID recursively..." << std::endl;
//     return nullptr;
// }

// void CipherGeometricPlanner::recordRefinement(int parent_region, const std::shared_ptr<DecompositionImpl> decomp_) {
//     DOUT << "Recording refinement in decomposition hierarchy..." << std::endl;
// }

// bool CipherGeometricPlanner::resolveWithLocalCompositePlanner(
//     const SegmentConflict& conflict,
//     ConflictResolutionEntry& log_entry) {
//     DOUT << "Resolving with local composite planner..." << std::endl;

//     size_t robot_1 = conflict.robot_index_1;
//     size_t robot_2 = conflict.robot_index_2;

//     DOUT << "    Local composite planner: jointly planning robots "
//               << robot_1 << " and " << robot_2 << std::endl;

//     // Check for timeout
//     if (isTimeoutExceeded()) {
//         std::cerr << "    Timeout before local composite planner" << std::endl;
//         return false;
//     }

//     StrategyAttempt attempt;
//     attempt.strategy = "local_composite";

//     // Locate conflict region and get expanded region for the subproblem
//     // Handle robots that have finished their paths (stationary at goal)
//     ob::State* s1_at_ts = getStateAtTimestep(robot_1, conflict.timestep);
//     if (!s1_at_ts) {
//         DOUT << "    Robot " << robot_1 << " has no path at conflict timestep" << std::endl;
//         attempt.planning_succeeded = false;
//         log_entry.attempts.push_back(attempt);
//         return false;
//     }
//     int conflict_region = decomp_->locateRegion(s1_at_ts);

//     // Use 2 layers of expansion around the conflict region for local bounds
//     std::vector<int> expanded_regions = getExpandedRegion(conflict_region, 2);

//     // Extract replanning bounds (entry/exit states) for both robots
//     PathUpdateInfo update_info_1, update_info_2;
//     if (!extractReplanningBoundsForExpandedRegion(
//             conflict, expanded_regions, update_info_1, update_info_2)) {
//         DOUT << "    Failed to extract replanning bounds" << std::endl;
//         attempt.planning_succeeded = false;
//         log_entry.attempts.push_back(attempt);
//         return false;
//     }

//     // Convert OMPL entry/exit states to std::vector<double>
//     auto si_1 = robots_[robot_1]->getSpaceInformation();
//     auto si_2 = robots_[robot_2]->getSpaceInformation();

//     std::vector<double> start_1, goal_1, start_2, goal_2;
//     si_1->getStateSpace()->copyToReals(start_1, update_info_1.planning_entry_state);
//     si_1->getStateSpace()->copyToReals(goal_1, update_info_1.planning_exit_state);
//     si_2->getStateSpace()->copyToReals(start_2, update_info_2.planning_entry_state);
//     si_2->getStateSpace()->copyToReals(goal_2, update_info_2.planning_exit_state);

//     // Compute local bounds from expanded region
//     std::vector<double> local_env_min, local_env_max;
//     computeExpandedBounds(expanded_regions, local_env_min, local_env_max);

//     // Call useCompositePlanner to jointly plan both robots
//     std::vector<size_t> robot_indices = {robot_1, robot_2};
//     std::vector<std::vector<double>> subproblem_starts = {start_1, start_2};
//     std::vector<std::vector<double>> subproblem_goals = {goal_1, goal_2};

//     GeometricPlanningResult result;
//     {
//         result = useCompositePlanner(
//             robot_indices, subproblem_starts, subproblem_goals,
//             local_env_min, local_env_max);
//     }

//     if (result.solved && result.individual_paths.size() == 2) {
//         attempt.planning_succeeded = true;
//         DOUT << "    Local composite planning succeeded" << std::endl;

//         // Convert PlanningResult individual paths to GuidedPlanningResult format
//         std::vector<GuidedPlanningResult> local_results;
//         for (size_t i = 0; i < robot_indices.size(); ++i) {
//             GuidedPlanningResult guided_result;
//             guided_result.success = true;
//             guided_result.planning_time = result.planning_time;
//             guided_result.robot_index = robot_indices[i];
//             guided_result.path = result.individual_paths[i];
//             local_results.push_back(guided_result);
//         }

//         // Integrate refined paths and re-check conflict
//         {
//             integrateRefinedPaths(robot_indices, local_results, update_info_1, update_info_2);
//         }
//         {
//             recheckConflictsFromTimestep(getRecheckStartTimestep(conflict, update_info_1, update_info_2));
//         }

//         // Check if the conflict is resolved
//         if (!conflictPersistsForRobots(robot_1, robot_2, conflict.timestep)) {
//             DOUT << "    Local composite planner resolved the conflict" << std::endl;
//             attempt.conflict_resolved = true;
//             log_entry.attempts.push_back(attempt);
//             freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
//             return true;
//         }

//         DOUT << "    Local composite planner: conflict persists after replanning" << std::endl;
//     } else {
//         attempt.planning_succeeded = false;
//         DOUT << "    Local composite planner failed to find solution" << std::endl;
//     }

//     log_entry.attempts.push_back(attempt);
//     freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
//     return false;
// }

bool CipherGeometricPlanner::resolveWithFullProblemCompositePlanner(
    int max_attempts,
    ConflictResolutionEntry& log_entry) {
    DOUT << "Resolving with full-problem composite planner..." << std::endl;

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

        DOUT << "    Full-problem composite attempt " << attempt_num
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
        if (vm.count("help")) { DOUT << desc << std::endl; return 0; }
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        DOUT << desc << std::endl;
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
            if (cfg["cbs_timeout"])     config.mapf_config.mapf_timeout = cfg["cbs_timeout"].as<double>();
            if (cfg["max_obstacle_volume_percent"])
                config.mapf_config.max_obstacle_volume_percent = cfg["max_obstacle_volume_percent"].as<double>();
            if (cfg["robot_cell_size_ratio"])
                config.robot_cell_size_ratio = cfg["robot_cell_size_ratio"].as<double>();
            if (cfg["max_initial_extensions"])
                config.max_initial_extensions = cfg["max_initial_extensions"].as<int>();
            if (cfg["max_blocked_edge_retries"])
                config.max_blocked_edge_retries = cfg["max_blocked_edge_retries"].as<int>();
            if (cfg["max_no_progress_iters"])
                config.max_no_progress_iters = cfg["max_no_progress_iters"].as<int>();
            if (cfg["restrict_sampling_to_cell"])
                config.restrict_sampling_to_cell = cfg["restrict_sampling_to_cell"].as<bool>();
        } catch (const YAML::Exception& e) {
            std::cerr << "ERROR loading config file: " << e.what() << std::endl;
            return 1;
        }
    }

    if (config.seed >= 0) {
        DOUT << "Setting random seed to: " << config.seed << std::endl;
        ompl::RNG::setSeed(config.seed);
    }

    DOUT << "Loading YAML file: " << inputFile << std::endl;
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
    DOUT << "Loaded " << obstacles.size() << " obstacles" << std::endl;

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

        DOUT << "  Robot " << robot_idx << " (" << robot_types.back() << ")"
                  << "  Start: (" << s[0] << ", " << s[1] << ")"
                  << "  Goal: ("  << g[0] << ", " << g[1] << ")" << std::endl;
        ++robot_idx;
    }
    DOUT << "Planning for " << robot_types.size() << " robots" << std::endl;

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

    if (result.success) {
        const auto& guided_paths = planner.getGuidedPaths();
        YAML::Node result_node;
        for (const auto& gpr : guided_paths) {
            YAML::Node robot_node;
            YAML::Node states_list;
            if (gpr.path) {
                auto si = gpr.path->getSpaceInformation();
                for (size_t s = 0; s < gpr.path->getStateCount(); ++s) {
                    std::vector<double> reals;
                    si->getStateSpace()->copyToReals(reals, gpr.path->getState(s));
                    YAML::Node state_node;
                    for (double v : reals) state_node.push_back(v);
                    states_list.push_back(state_node);
                }
            }
            robot_node["states"] = states_list;
            result_node.push_back(robot_node);
        }
        output["result"] = result_node;
    }

    try {
        std::ofstream fout(outputFile);
        fout << output;
        fout.close();
        DOUT << "Output written to " << outputFile << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR writing output: " << e.what() << std::endl;
        return 1;
    }

    for (auto* co : obstacles) delete co;

    DOUT << "Done! time=" << result.planning_time << "s  solved=" << result.success << std::endl;
    return result.success ? 0 : 1;
}
