#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <queue>
#include <set>
#include <boost/heap/d_ary_heap.hpp>
#include <boost/program_options.hpp>
#include <ompl/util/Console.h>
#include <ompl/util/RandomNumbers.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#ifdef DOUT_ENABLED
#define DOUT std::cout
#else
namespace {
struct NullBuf : std::streambuf { int overflow(int c) override { return c; } };
struct NullStream : std::ostream { NullStream() : std::ostream(&_buf) {} NullBuf _buf; } _null_stream;
}
#define DOUT _null_stream
#endif

#include "robotStatePropagator.hpp"
#include "fclStateValidityChecker.hpp"
#include "dynoplan/optimization/ocp.hpp"
#include "dynoplan/optimization/multirobot_optimization.hpp"

#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include "dynobench/motions.hpp"
#include "dynobench/robot_models.hpp"
#include "dynoplan/ompl/robots.h"

#include "cipher_kino.h"
#include "fclStateValidityChecker.hpp"
#include "robots.h"

#include "utils/decomposition.h"
#include "utils/grid_decomposition.h"
#include "guided/guided_geometric_rrt.h"
#include "mapf/cbs.h"

namespace po = boost::program_options;

// ============================================================================
// Constructor / Destructor
// ============================================================================

CipherKinoPlanner::CipherKinoPlanner(const CipherKinoConfig& config)
    : config_(config),
      decomp_(nullptr),
      workspace_bounds_(3)
{
}

CipherKinoPlanner::~CipherKinoPlanner()
{
    cleanup();
}

// ============================================================================
// Problem Loading
// ============================================================================

void CipherKinoPlanner::loadProblem(
    const std::vector<std::string>& robot_types,
    const std::vector<std::vector<double>>& starts,
    const std::vector<std::vector<double>>& goals,
    const std::vector<fcl::CollisionObjectf*>& obstacles,
    const std::vector<double>& env_min,
    const std::vector<double>& env_max)
{
    // Clean up any previous problem
    cleanup();

    robot_types_ = robot_types;
    starts_      = starts;
    goals_       = goals;
    obstacles_   = obstacles;
    env_min_     = env_min;
    env_max_     = env_max;

    // Setup workspace bounds
    workspace_bounds_.setLow(0, env_min[0]);
    workspace_bounds_.setLow(1, env_min[1]);
    workspace_bounds_.setHigh(0, env_max[0]);
    workspace_bounds_.setHigh(1, env_max[1]);

    setupCollisionManager();
    setupRobots();
    setupDecomposition();
    if (config_.check_transition_feasibility)
        computeTransitionFeasibility();

    // Build and write visualization header now that robots + decomp are ready
    initVizHeader();

    problem_loaded_ = true;
}

// ============================================================================
// Main Planning Entry Point
// ============================================================================

CipherKinoResult CipherKinoPlanner::plan()
{
    DOUT << "Planning with CipherKinoPlanner..." << std::endl;

    // Check if problem is loaded
    if (!problem_loaded_) {
        throw std::runtime_error("Problem not loaded. Call loadProblem() first.");
    }
    
    CipherKinoResult result;
    planning_start_time_ = std::chrono::steady_clock::now();

    forbidden_edges_.clear();

    try {
        // Phase 1: Compute high-level paths over decomposition
        bool cbs_exhausted = false;
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
                        std::cerr << "[Phase 1] All cells at maximum decomposition depth; falling back to composite planner" << std::endl;
                        cbs_exhausted = true;
                        break;
                    }
                } else {
                    std::cerr << "[Phase 1] CBS failed again after decomposition; falling back to composite planner" << std::endl;
                    cbs_exhausted = true;
                    break;
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

        if (cbs_exhausted) {
            if (isTimeoutExceeded()) {
                std::cerr << "Planning timeout exceeded before composite fallback" << std::endl;
                result.success = false;
                result.failure_reason = "timeout_cbs_failed";
            } else if (config_.conflict_resolution_config.max_composite_attempts <= 0) {
                std::cerr << "CBS exhausted and composite fallback disabled (max_composite_attempts=0)" << std::endl;
                result.success = false;
                result.failure_reason = "cbs_failed_composite_disabled";
            } else {
                DOUT << "[Fallback] CBS exhausted; attempting full-problem composite planner ("
                     << config_.conflict_resolution_config.max_composite_attempts << " attempt(s))..." << std::endl;
                std::vector<size_t> all_robot_indices;
                for (size_t i = 0; i < robots_.size(); ++i) all_robot_indices.push_back(i);
                KinoPlanningResult composite_result =
                    useCompositePlanner(all_robot_indices, starts_, goals_, env_min_, env_max_);
                result.success = composite_result.solved;
                if (!composite_result.solved) {
                    std::cerr << "CBS fallback: composite planner also failed" << std::endl;
                    result.failure_reason = "cbs_failed_composite_failed";
                } else {
                    DOUT << "[Fallback] Composite planner succeeded" << std::endl;
                }
            }
        } else {
            bool all_paths_found = true;
            for (size_t r = 0; r < robot_paths_.size(); ++r) {
                if (!robot_paths_[r]) {
                    std::cerr << "Robot " << r << " failed to find a guided path" << std::endl;
                    all_paths_found = false;
                }
            }
            if (!all_paths_found) {
                result.success = false;
                result.failure_reason = "guided_path_failed";
                auto end_time = std::chrono::steady_clock::now();
                result.planning_time = std::chrono::duration<double>(end_time - planning_start_time_).count();
                result.resolution_stats = resolution_stats_;
                return result;
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

// ============================================================================
// Individual Planning Phases
// ============================================================================

void CipherKinoPlanner::computeHighLevelPaths()
{
    // TODO: run MAPF solver (CBS / A* / decoupled) to fill high_level_paths_
    DOUT << "Computing high-level paths..." << std::endl;
    CBS cbs_solver(config_.mapf_config.region_capacity, config_.mapf_config.mapf_timeout,
                    obstacles_, config_.mapf_config.max_obstacle_volume_percent);\
    ForbiddenEdgeSet all_forbidden = forbidden_edges_;
    all_forbidden.insert(structurally_forbidden_edges_.begin(), structurally_forbidden_edges_.end());
    high_level_paths_ = cbs_solver.solve(decomp_, start_states_, goal_states_,
                                         /*allowed_regions=*/{}, all_forbidden);

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
                cell_ids.push_back("c" + std::to_string(rid));
            paths["r" + std::to_string(r)] = cell_ids;
        }
        ev["paths"] = paths;
        viz_events_.push_back(ev);
        vizWriteFile();
        DOUT << "[viz] mapf event written to " << viz_file_ << std::endl;
    }
}

// ============================================================================
// Private helpers (file scope)
// ============================================================================

// Converts a dynobench::Trajectory to oc::PathControl using the robot's oc::SpaceInformation.
static std::shared_ptr<oc::PathControl> trajectoryToPathControl(
    const dynobench::Trajectory& traj,
    const std::shared_ptr<oc::SpaceInformation>& oc_si)
{
    if (traj.states.empty()) return nullptr;

    auto path = std::make_shared<oc::PathControl>(oc_si);
    auto rv_cs = std::dynamic_pointer_cast<oc::RealVectorControlSpace>(oc_si->getControlSpace());
    double default_dur = oc_si->getPropagationStepSize();

    auto eigenToState = [&](const Eigen::VectorXd& v) -> ob::State* {
        ob::State* s = oc_si->getStateSpace()->allocState();
        std::vector<double> reals(v.data(), v.data() + v.size());
        oc_si->getStateSpace()->copyFromReals(s, reals);
        return s;
    };

    ob::State* s0 = eigenToState(traj.states[0]);
    path->append(s0);
    oc_si->getStateSpace()->freeState(s0);

    for (size_t i = 0; i + 1 < traj.states.size(); ++i) {
        double dur = (traj.times.size() > (Eigen::Index)(i + 1))
                     ? traj.times[i + 1] - traj.times[i]
                     : default_dur;

        oc::Control* ctrl = oc_si->allocControl();
        if (rv_cs && i < traj.actions.size()) {
            auto* rv = ctrl->as<oc::RealVectorControlSpace::ControlType>();
            for (int d = 0; d < traj.actions[i].size(); ++d)
                rv->values[d] = traj.actions[i][d];
        }
        ob::State* sn = eigenToState(traj.states[i + 1]);
        path->append(sn, ctrl, dur);
        oc_si->getStateSpace()->freeState(sn);
        oc_si->freeControl(ctrl);
    }
    return path;
}

// Builds a dynobench::Problem from in-memory planner state for a single robot.
static dynobench::Problem buildDynobenchProblem(
    const std::string& models_base_path,
    const std::vector<std::string>& robot_types,
    const std::vector<double>& start,
    const std::vector<double>& goal,
    const std::vector<double>& env_min,
    const std::vector<double>& env_max,
    const std::vector<fcl::CollisionObjectf*>& obstacles)
{
    dynobench::Problem problem;
    problem.robotType = robot_types[0];
    problem.robotTypes = robot_types;
    problem.models_base_path = models_base_path + "/";
    problem.start = Eigen::VectorXd::Map(start.data(), start.size());
    problem.goal  = Eigen::VectorXd::Map(goal.data(),  goal.size());
    problem.p_lb  = Eigen::VectorXd::Map(env_min.data(), env_min.size());
    problem.p_ub  = Eigen::VectorXd::Map(env_max.data(), env_max.size());

    for (const auto* co : obstacles) {
        const auto* box = dynamic_cast<const fcl::Boxf*>(co->getCollisionGeometry());
        if (!box) continue;
        dynobench::Obstacle obs;
        obs.type   = "box";
        obs.size   = Eigen::Vector2d(box->side[0], box->side[1]);
        obs.center = Eigen::Vector2d(co->getTranslation()[0], co->getTranslation()[1]);
        problem.obstacles.push_back(obs);
    }
    return problem;
}

void CipherKinoPlanner::computeGuidedPaths()
{
    DOUT << "Computing guided paths..." << std::endl;

    if (!problem_loaded_) {
        throw std::runtime_error("Problem not loaded. Call loadProblem() first.");
    }
    if (high_level_paths_.empty()) {
        throw std::runtime_error("High-level paths not computed. Call computeHighLevelPaths() first.");
    }

    guided_planning_results_.clear();

    for (size_t robot_idx = 0; robot_idx < robots_.size(); ++robot_idx) {
        DOUT << "!!!!! COMPUTING GUIDED PATH FOR ROBOT: " << robot_idx << "!!!!!" << std::endl;
        guided_planning_results_.push_back(KinoGuidedPlanningResult());
        guided_planning_results_[robot_idx].robot_index = robot_idx;
        guided_planning_results_[robot_idx].success = false;

        auto robot_si = std::dynamic_pointer_cast<oc::SpaceInformation>(robot_sis_[robot_idx]);
        if (!robot_si) {
            DOUT << "  Robot " << robot_idx << ": no oc::SpaceInformation, skipping" << std::endl;
            continue;
        }

        std::vector<std::string> robot_type;
        robot_type.push_back(robot_types_[robot_idx]);

        // Build dynobench::Problem from in-memory state
        dynobench::Problem problem = buildDynobenchProblem(
            config_.models_base_path, robot_type,
            starts_[robot_idx], goals_[robot_idx],
            env_min_, env_max_, obstacles_);
        

        // Load dynobench robot model (robot_factory returns unique_ptr; wrap in shared_ptr)
        std::shared_ptr<dynobench::Model_robot> robot_model(
            dynobench::robot_factory(
                (config_.models_base_path + "/" + robot_types_[robot_idx] + ".yaml").c_str(),
                problem.p_lb, problem.p_ub).release());
        dynobench::load_env(*robot_model, problem);

        // Configure guided planner options
        std::vector<dynoplan::Motion> motions;
        dynoplan::Options_dbrrt options_dbrrt = config_.options_dbrrt;
        options_dbrrt.motionsFile = config_.motions_file;
        options_dbrrt.timelimit   = config_.planning_time_limit * 1000.0;  // dbrrt uses ms
        options_dbrrt.do_optimization = config_.options_dbrrt.do_optimization;
        options_dbrrt.verbose     = false;
        options_dbrrt.debug       = false;

        dynoplan::load_motion_primitives_new(
            options_dbrrt.motionsFile, *robot_model, motions,
            options_dbrrt.max_motions, options_dbrrt.cut_actions,
            /*shuffle=*/false, options_dbrrt.check_cols);
        options_dbrrt.motions_ptr = &motions;

        // Run guided kinodynamic planner
        dynobench::Trajectory traj_out;
        dynobench::Info_out out_info;
        dynoplan::Options_trajopt options_trajopt = config_.options_trajopt;
        options_trajopt.region_bounds_weight = config_.region_bounds_weight;
        // options_trajopt.max_iter = 100;
        // options_trajopt.init_reg = 10e2;
        // options_trajopt.states_reg = true;
        // options_trajopt.use_finite_diff = true;
        // options_trajopt.time_ref = 0.1;

        try {
            if (config_.guided_planner_method == "guided_idbrrt")
                dynoplan::guided_idbrrt(problem, robot_model, options_dbrrt, options_trajopt,
                    robots_[robot_idx], decomp_, high_level_paths_[robot_idx],
                    traj_out, out_info);
            else if (config_.guided_planner_method == "guided_dbrrt")
                dynoplan::guided_dbrrt(problem, robot_model, options_dbrrt, options_trajopt,
                    robots_[robot_idx], decomp_, high_level_paths_[robot_idx],
                    traj_out, out_info);
        } catch (const std::exception& e) {
            DOUT << "  Robot " << robot_idx << ": guided_idbrrt threw: " << e.what() << std::endl;
            continue;
        }

        bool success = (out_info.solved_raw || out_info.solved) && !traj_out.states.empty();
        if (success) {
            // Use optimized trajectory if available, otherwise raw
            if (out_info.solved)
                DOUT << "!!!!!!DBRRT OPTIMIZATION SOLVED!!!!!!" << std::endl;
            else
                DOUT << "!!!!!!DBRRT OPTIMIZATION NOT SOLVED!!!!!!" << std::endl;
            const auto& final_traj = (!out_info.trajs_opt.empty()) ? out_info.trajs_opt[0] : traj_out;
            guided_planning_results_[robot_idx].path = trajectoryToPathControl(final_traj, robot_si);
            guided_planning_results_[robot_idx].success = (guided_planning_results_[robot_idx].path != nullptr);
            DOUT << "  Robot " << robot_idx << ": solved with "
                 << guided_planning_results_[robot_idx].path->getStateCount() << " states" << std::endl;
        } else {
            DOUT << "  Robot " << robot_idx << ": FAILED" << std::endl;
        }
    }

    DOUT << "Guided planner results: " << guided_planning_results_.size() << std::endl;

    // Populate robot_paths_ from guided results
    robot_paths_.resize(robots_.size());
    for (size_t r = 0; r < guided_planning_results_.size(); ++r) {
        robot_paths_[r] = guided_planning_results_[r].success ? guided_planning_results_[r].path : nullptr;
    }

    // Emit low_level_paths viz events
    if (do_viz_ && !guided_planning_results_.empty()) {
        for (size_t r = 0; r < guided_planning_results_.size(); ++r) {
            if (!guided_planning_results_[r].path) continue;
            auto si = robot_sis_[r];
            const auto& path = guided_planning_results_[r].path;
            auto* oc_si_raw = dynamic_cast<oc::SpaceInformation*>(si.get());
            auto rv_cs = oc_si_raw
                ? std::dynamic_pointer_cast<oc::RealVectorControlSpace>(oc_si_raw->getControlSpace())
                : nullptr;

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
                YAML::Node ctrl_node;
                double duration = 0.0;
                if (j < path->getControlCount() && rv_cs) {
                    auto* rv_ctrl = path->getControl(j)->as<oc::RealVectorControlSpace::ControlType>();
                    for (unsigned int d = 0; d < rv_cs->getDimension(); ++d)
                        ctrl_node.push_back(rv_ctrl->values[d]);
                    duration = path->getControlDuration(j);
                } else {
                    ctrl_node.push_back(0.0); ctrl_node.push_back(0.0);
                }
                wp["control"] = ctrl_node;
                wp["duration"] = duration;
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

bool CipherKinoPlanner::checkPathsForConflicts()
{
    segment_conflicts_.clear();
    
    if (robot_paths_.empty()) return false;

    bool any_path = false;
    for (const auto& p : robot_paths_) {
        if (p && p->getStateCount() > 0) { any_path = true; break; }
    }
    if (!any_path) return false;

    int max_timestep = 0;
    for (size_t r = 0; r < robot_paths_.size(); ++r) {
        if (!robot_paths_[r] || robot_paths_[r]->getStateCount() == 0) continue;
        const auto& path = robot_paths_[r];
        auto* oc_si = dynamic_cast<oc::SpaceInformation*>(robot_sis_[r].get());
        double step_size = oc_si ? oc_si->getPropagationStepSize() : 1.0;
        int total_steps = 0;
        for (size_t k = 0; k < path->getControlCount(); ++k)
            total_steps += (int)std::round(path->getControlDuration(k) / step_size);
        max_timestep = std::max(max_timestep, total_steps);
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

bool CipherKinoPlanner::resolveConflicts()
{
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

// ============================================================================
// Timeout
// ============================================================================

bool CipherKinoPlanner::isTimeoutExceeded() const
{
    if (config_.max_total_time <= 0.0) return false;
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - planning_start_time_).count();
    return elapsed >= config_.max_total_time;
}

// ============================================================================
// Setup Helpers
// ============================================================================

void CipherKinoPlanner::setupDecomposition()
{
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

void CipherKinoPlanner::computeTransitionFeasibility() {
    structurally_forbidden_edges_.clear();

    if (obstacles_.empty()) return;

    // Compute max robot circumradius across all robots and parts.
    float max_radius = 0.0f;
    for (auto& robot : robots_) {
        for (size_t p = 0; p < robot->numParts(); ++p) {
            auto geom = robot->getCollisionGeometry(p);
            if (!geom) continue;
            geom->computeLocalAABB();
            const auto& aabb = geom->aabb_local;
            float hx = (aabb.max_[0] - aabb.min_[0]) / 2.0f;
            float hy = (aabb.max_[1] - aabb.min_[1]) / 2.0f;
            max_radius = std::max(max_radius, std::sqrt(hx * hx + hy * hy));
        }
    }
    double threshold = config_.transition_feasibility_robot_size_multiplier * 2.0 * max_radius;

    int total = decomp_->getTotalNumRegions();
    for (int a = 0; a < total; ++a) {
        if (!decomp_->isLeafRegion(a)) continue;
        const auto ba = decomp_->getCellBounds(a);

        std::vector<int> neighbors;
        decomp_->getNeighbors(a, neighbors);

        for (int b : neighbors) {
            if (!decomp_->isLeafRegion(b)) continue;
            const auto bb = decomp_->getCellBounds(b);

            // Determine the shared boundary axis and the 1D free interval on the boundary.
            double seg_lo, seg_hi;
            bool shared_in_y;
            if (std::abs(ba.high[0] - bb.low[0]) < 1e-9 || std::abs(bb.high[0] - ba.low[0]) < 1e-9) {
                // Vertical boundary: shared extent in Y
                shared_in_y = true;
                seg_lo = std::max(ba.low[1], bb.low[1]);
                seg_hi = std::min(ba.high[1], bb.high[1]);
            } else {
                // Horizontal boundary: shared extent in X
                shared_in_y = false;
                seg_lo = std::max(ba.low[0], bb.low[0]);
                seg_hi = std::min(ba.high[0], bb.high[0]);
            }

            if (seg_hi <= seg_lo) continue;

            // Subtract obstacle AABB projections to find free intervals.
            std::vector<std::pair<double,double>> covered;
            for (auto* obs : obstacles_) {
                const auto& aabb = obs->getAABB();
                double lo = shared_in_y ? (double)aabb.min_[1] : (double)aabb.min_[0];
                double hi = shared_in_y ? (double)aabb.max_[1] : (double)aabb.max_[0];
                double clo = std::max(lo, seg_lo);
                double chi = std::min(hi, seg_hi);
                if (chi > clo)
                    covered.emplace_back(clo, chi);
            }

            // Find the longest free sub-interval.
            std::sort(covered.begin(), covered.end());
            double longest_free = 0.0;
            double cur = seg_lo;
            for (auto& [clo, chi] : covered) {
                if (clo > cur)
                    longest_free = std::max(longest_free, clo - cur);
                cur = std::max(cur, chi);
            }
            longest_free = std::max(longest_free, seg_hi - cur);

            if (longest_free < threshold) {
                structurally_forbidden_edges_.insert({a, b});
                structurally_forbidden_edges_.insert({b, a});
            }
        }
    }

    DOUT << "computeTransitionFeasibility: " << structurally_forbidden_edges_.size() / 2
         << " impassable cell transitions (threshold=" << threshold << ")" << std::endl;
}

void CipherKinoPlanner::setupCollisionManager()
{
    DOUT << "Setting up conflict manager..." << std::endl;
    collision_manager_ = std::make_shared<fcl::DynamicAABBTreeCollisionManagerf>();
    collision_manager_->registerObjects(obstacles_);
    collision_manager_->setup();
}

void CipherKinoPlanner::setupRobots()
{
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
        // auto si = std::make_shared<ob::SpaceInformation>(state_space);
        auto si = robot->getSpaceInformation();

        si->setStateValidityChecker(
            std::make_shared<fclStateValidityChecker>(si, collision_manager_, robot));
        si->setStatePropagator(std::make_shared<RobotStatePropagator>(si, robot));
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

        KinoGuidedPlanningResult result;
        result.robot_index = i;
        result.planning_time = 0.0;
        result.success = false;
        guided_planning_results_.push_back(result);
    }
}

void CipherKinoPlanner::cleanup()
{
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

// ============================================================================
// Conflict-Checking Helpers
// ============================================================================

ob::State* CipherKinoPlanner::getStateAtTimestep(size_t robot_idx, int timestep) const
{
    if (robot_idx >= robot_paths_.size() || !robot_paths_[robot_idx]) return nullptr;
    const auto& path = robot_paths_[robot_idx];
    if (path->getStateCount() == 0) return nullptr;

    auto* oc_si = dynamic_cast<oc::SpaceInformation*>(robot_sis_[robot_idx].get());
    double step_size = oc_si ? oc_si->getPropagationStepSize() : 1.0;

    int cumulative = 0;
    for (size_t k = 0; k < path->getControlCount(); ++k) {
        int steps_k = (int)std::round(path->getControlDuration(k) / step_size);
        if (timestep < cumulative + steps_k)
            return path->getState(k);
        cumulative += steps_k;
    }
    return path->getState(path->getStateCount() - 1);
}

bool CipherKinoPlanner::checkTwoRobotConflict(
    size_t robot_idx_1, const ob::State* state_1,
    size_t robot_idx_2, const ob::State* state_2,
    size_t& part_1, size_t& part_2) const
{
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

// ============================================================================
// Conflict Resolution – Top-Level Strategies
// ============================================================================

KinoPlanningResult CipherKinoPlanner::useCompositePlanner(
    const std::vector<size_t>& robot_indices,
    const std::vector<std::vector<double>>& subproblem_starts,
    const std::vector<std::vector<double>>& subproblem_goals,
    const std::vector<double>& subproblem_env_min,
    const std::vector<double>& subproblem_env_max)
{
    DOUT << "Using composite planner for " << robot_indices.size() << " robots..." << std::endl;

    KinoPlanningResult result;
    result.solved        = false;
    result.planning_time = 0.0;

    // Collect selected robots
    std::vector<std::shared_ptr<Robot>> selected_robots;
    for (size_t r : robot_indices)
        selected_robots.push_back(robots_[r]);

    // Build compound state and control spaces
    auto compound_ss = std::make_shared<ob::CompoundStateSpace>();
    auto compound_cs = std::make_shared<oc::CompoundControlSpace>(compound_ss);
    for (size_t r : robot_indices) {
        auto robot_si_kino = std::dynamic_pointer_cast<oc::SpaceInformation>(robot_sis_[r]);
        if (!robot_si_kino) continue;
        auto ss = robot_si_kino->getStateSpace();
        ss->setName("Robot " + std::to_string(r));
        compound_ss->addSubspace(ss, 1.0);
        compound_cs->addSubspace(robot_si_kino->getControlSpace());
    }

    // Create compound SpaceInformation
    auto compound_si = std::make_shared<oc::SpaceInformation>(compound_ss, compound_cs);
    compound_si->setStateValidityChecker(
        std::make_shared<CompoundStateValidityChecker>(compound_si, collision_manager_, selected_robots));
    compound_si->setStatePropagator(
        std::make_shared<CompoundStatePropagator>(compound_si, selected_robots));

    // Use the propagation step size from the first robot
    if (!robot_indices.empty()) {
        auto first_si = std::dynamic_pointer_cast<oc::SpaceInformation>(robot_sis_[robot_indices[0]]);
        if (first_si) {
            compound_si->setPropagationStepSize(first_si->getPropagationStepSize());
            compound_si->setMinMaxControlDuration(
                first_si->getMinControlDuration(), first_si->getMaxControlDuration());
        }
    }
    compound_si->setup();

    // Build compound start state
    auto compound_start = compound_si->allocState();
    {
        auto* cs = compound_start->as<ob::CompoundState>();
        for (size_t i = 0; i < robot_indices.size(); ++i) {
            size_t r = robot_indices[i];
            auto robot_si_r = robot_sis_[r];
            ob::State* sub = robot_si_r->getStateSpace()->allocState();
            robot_si_r->getStateSpace()->copyFromReals(sub, subproblem_starts[i]);
            robot_si_r->getStateSpace()->copyState(cs->components[i], sub);
            robot_si_r->getStateSpace()->freeState(sub);
        }
    }

    // Build goal states for compound goal condition
    std::vector<ob::State*> goal_states_local;
    for (size_t i = 0; i < robot_indices.size(); ++i) {
        size_t r = robot_indices[i];
        ob::State* gs = robot_sis_[r]->getStateSpace()->allocState();
        robot_sis_[r]->getStateSpace()->copyFromReals(gs, subproblem_goals[i]);
        goal_states_local.push_back(gs);
    }

    auto pdef = std::make_shared<ob::ProblemDefinition>(compound_si);
    pdef->addStartState(compound_start);
    pdef->setGoal(std::make_shared<CompoundGoalCondition>(
        compound_si, goal_states_local, config_.goal_threshold));

    // Plan with oc::RRT
    auto planner = std::make_shared<oc::RRT>(compound_si);
    planner->setProblemDefinition(pdef);
    planner->setup();

    auto t0 = std::chrono::steady_clock::now();
    ob::PlannerStatus status = planner->solve(
        ob::timedPlannerTerminationCondition(config_.planning_time_limit));
    result.planning_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    result.solved = (status == ob::PlannerStatus::EXACT_SOLUTION ||
                     status == ob::PlannerStatus::APPROXIMATE_SOLUTION);

    if (result.solved) {
        auto compound_path = pdef->getSolutionPath()->as<oc::PathControl>();
        auto compound_path_ptr = std::make_shared<oc::PathControl>(*compound_path);
        result.path = compound_path_ptr;
        extractIndividualPaths(compound_path_ptr, robot_indices, result.individual_paths);
        DOUT << "  Composite planner solved in " << result.planning_time << "s" << std::endl;
    } else {
        DOUT << "  Composite planner failed" << std::endl;
    }

    // Cleanup
    compound_si->freeState(compound_start);
    for (size_t i = 0; i < robot_indices.size(); ++i)
        robot_sis_[robot_indices[i]]->getStateSpace()->freeState(goal_states_local[i]);

    return result;
}

bool CipherKinoPlanner::resolveConflictWithStrategies(
    const SegmentConflict& conflict,
    ConflictResolutionEntry& log_entry)
{
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

bool CipherKinoPlanner::conflictPersistsForRobots(
    size_t robot_1, size_t robot_2, int timestep) const
{
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

// ============================================================================
// Hierarchical Expansion / Refinement
// ============================================================================

bool CipherKinoPlanner::resolveWithHierarchicalExpansionRefinement(
    const SegmentConflict& conflict,
    int max_refinement_levels,
    int max_expansion_layers,
    int min_expansion_layer,
    ConflictResolutionEntry& log_entry)
{
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

bool CipherKinoPlanner::attemptRefinementAtExpansionLevel(
    const SegmentConflict& conflict,
    int conflict_region,
    const std::vector<int>& expanded_regions,
    int expansion_layer,
    int max_refinement_levels,
    ConflictResolutionEntry& log_entry)
{
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

bool CipherKinoPlanner::refineExpandedRegion(
    const SegmentConflict& conflict,
    int conflict_region,
    const std::vector<int>& expanded_regions,
    int expansion_layer,
    int refinement_level)
{
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
            decomp_, replan_hl_starts, replan_hl_goals, expanded_leaf_regions, structurally_forbidden_edges_);
    }

    if (local_high_level_paths.empty()) {
        DOUT << "        MAPF failed" << std::endl;
        freeUpdateInfoStates(robot_1, robot_2, update_info_1, update_info_2);
        return false;
    } else {
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
    std::vector<KinoGuidedPlanningResult> replan_results;
    bool all_succeeded = true;
    {
        for (size_t i = 0; i < replan_robot_indices.size(); ++i) {
            size_t robot_idx = replan_robot_indices[i];

            auto robot_si_kino = std::dynamic_pointer_cast<oc::SpaceInformation>(robot_sis_[robot_idx]);

            replan_results.push_back(KinoGuidedPlanningResult());
            replan_results[i].robot_index = robot_idx;

            // Convert OMPL start/goal states to std::vector<double> for Problem construction
            std::vector<double> start_reals, goal_reals;
            robot_si_kino->getStateSpace()->copyToReals(start_reals, replan_starts[i]);
            robot_si_kino->getStateSpace()->copyToReals(goal_reals,  replan_goals[i]);

            std::vector<std::string> robot_type;
            robot_type.push_back(robot_types_[robot_idx]);


            // Build dynobench::Problem with the local start/goal for this replan
            dynobench::Problem problem = buildDynobenchProblem(
                config_.models_base_path, robot_type,
                start_reals, goal_reals,
                env_min_, env_max_, obstacles_);

            std::shared_ptr<dynobench::Model_robot> robot_model(
                dynobench::robot_factory(
                    (config_.models_base_path + "/" + robot_types_[robot_idx] + ".yaml").c_str(),
                    problem.p_lb, problem.p_ub).release());
            dynobench::load_env(*robot_model, problem);

            std::vector<dynoplan::Motion> motions;
            dynoplan::Options_dbrrt options_dbrrt = config_.options_dbrrt;
            options_dbrrt.motionsFile = config_.motions_file;
            options_dbrrt.timelimit   = config_.planning_time_limit * 1000.0;
            options_dbrrt.do_optimization = config_.options_dbrrt.do_optimization;
            options_dbrrt.verbose     = false;
            options_dbrrt.debug       = false;

            dynoplan::load_motion_primitives_new(
                options_dbrrt.motionsFile, *robot_model, motions,
                options_dbrrt.max_motions, options_dbrrt.cut_actions,
                /*shuffle=*/false, options_dbrrt.check_cols);
            options_dbrrt.motions_ptr = &motions;

            dynobench::Trajectory traj_out;
            dynobench::Info_out out_info;
            dynoplan::Options_trajopt options_trajopt = config_.options_trajopt;
            options_trajopt.region_bounds_weight = config_.region_bounds_weight;
            // options_trajopt.max_iter = 100;
            // options_trajopt.init_reg = 10e2;
            // options_trajopt.states_reg = true;
            // options_trajopt.use_finite_diff = true;
            // options_trajopt.time_ref = 0.5;

            try {
                if (config_.guided_planner_method == "guided_idbrrt")
                    dynoplan::guided_idbrrt(problem, robot_model, options_dbrrt, options_trajopt,
                        robots_[robot_idx], decomp_, local_high_level_paths[i],
                        traj_out, out_info);
                else if (config_.guided_planner_method == "guided_dbrrt")
                    dynoplan::guided_dbrrt(problem, robot_model, options_dbrrt, options_trajopt,
                        robots_[robot_idx], decomp_, local_high_level_paths[i],
                        traj_out, out_info);
            } catch (const std::exception& e) {
                DOUT << "  Robot " << robot_idx << ": guided_idbrrt threw: " << e.what() << std::endl;
                replan_results[i].success = false;
                all_succeeded = false;
                break;
            }

            bool success = (out_info.solved_raw || out_info.solved) && !traj_out.states.empty();
            if (success) {
                if (out_info.solved)
                    DOUT << "!!!!!!DBRRT OPTIMIZATION SOLVED!!!!!!" << std::endl;
                else
                    DOUT << "!!!!!!DBRRT OPTIMIZATION NOT SOLVED!!!!!!" << std::endl;
                const auto& final_traj = (!out_info.trajs_opt.empty()) ? out_info.trajs_opt[0] : traj_out;
                replan_results[i].path = trajectoryToPathControl(final_traj, robot_si_kino);
                replan_results[i].success = (replan_results[i].path != nullptr);
                if (replan_results[i].success) {
                    DOUT << "  Robot " << robot_idx << ": solved with "
                         << replan_results[i].path->getStateCount() << " states" << std::endl;
                } else {
                    replan_results[i].success = false;
                }
            } else {
                replan_results[i].success = false;
                DOUT << "  Robot " << robot_idx << ": FAILED" << std::endl;
            }

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
                std::vector<KinoGuidedPlanningResult> single_result = {replan_results[i]};
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
                const auto& path = guided_planning_results_[r].path;
                auto* oc_si = dynamic_cast<oc::SpaceInformation*>(si.get());
                auto rv_cs = oc_si
                    ? std::dynamic_pointer_cast<oc::RealVectorControlSpace>(oc_si->getControlSpace())
                    : nullptr;

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
                    YAML::Node ctrl_node;
                    double duration = 0.0;
                    if (j < path->getControlCount() && rv_cs) {
                        auto* rv_ctrl = path->getControl(j)->as<oc::RealVectorControlSpace::ControlType>();
                        for (unsigned int d = 0; d < rv_cs->getDimension(); ++d)
                            ctrl_node.push_back(rv_ctrl->values[d]);
                        duration = path->getControlDuration(j);
                    } else {
                        ctrl_node.push_back(0.0); ctrl_node.push_back(0.0);
                    }
                    wp["control"] = ctrl_node;
                    wp["duration"] = duration;
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

int CipherKinoPlanner::calculateMaxExpansionLayers() const
{
    // Auto-detect: ceil(sqrt(num_regions) / 2)
    if (!decomp_) return 0;
    int num_regions = decomp_->getNumRegions();
    return static_cast<int>(std::ceil(std::sqrt(static_cast<double>(num_regions)) / 2.0));
}

int CipherKinoPlanner::calculateMaxRefinementLevels() const
{
    return config_.conflict_resolution_config.max_refinement_levels;
}

// ============================================================================
// Composite Planner Strategies
// ============================================================================

// bool CipherKinoPlanner::resolveWithLocalCompositePlanner(
//     const SegmentConflict& conflict,
//     ConflictResolutionEntry& log_entry)
// {
//     // TODO: compute local bounds around conflict, call useCompositePlanner,
//     //       integrate result via integrateRefinedPaths
//     return false;
// }

bool CipherKinoPlanner::resolveWithFullProblemCompositePlanner(
    int max_attempts,
    ConflictResolutionEntry& log_entry)
{
    DOUT << "Resolving with full-problem composite planner..." << std::endl;

    std::vector<size_t> all_robot_indices;
    for (size_t i = 0; i < robots_.size(); ++i)
        all_robot_indices.push_back(i);

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

        KinoPlanningResult result = useCompositePlanner(
            all_robot_indices, starts_, goals_, env_min_, env_max_);

        attempt.planning_succeeded = result.solved;
        if (result.solved && result.individual_paths.size() == robots_.size()) {
            // Replace all robot paths with the jointly-planned ones
            for (size_t i = 0; i < all_robot_indices.size(); ++i) {
                guided_planning_results_[i].path    = result.individual_paths[i];
                guided_planning_results_[i].success = true;
                robot_paths_[i]                     = result.individual_paths[i];
            }
            recheckConflictsFromTimestep(0);
            attempt.conflict_resolved = segment_conflicts_.empty();
            log_entry.attempts.push_back(attempt);
            return attempt.conflict_resolved;
        }

        log_entry.attempts.push_back(attempt);
    }

    return false;
}

// ============================================================================
// Replanning Bounds Extraction
// ============================================================================

bool CipherKinoPlanner::extractReplanningBoundsForExpandedRegion(
    const SegmentConflict& conflict,
    const std::vector<int>& expanded_regions,
    PathUpdateInfo& update_info_1,
    PathUpdateInfo& update_info_2)
{
    size_t robot_1 = conflict.robot_index_1;
    size_t robot_2 = conflict.robot_index_2;

    std::set<int> region_set(expanded_regions.begin(), expanded_regions.end());

    auto extractForRobot = [&](size_t robot_idx, PathUpdateInfo& info) -> bool {
        info.robot_index = robot_idx;

        auto si = robot_sis_[robot_idx];

        // Kinodynamic: path_end_timestep = total steps accumulated from control durations
        auto* oc_si_raw = dynamic_cast<oc::SpaceInformation*>(si.get());
        double step_size = oc_si_raw ? oc_si_raw->getPropagationStepSize() : 1.0;
        int path_end_timestep = 0;
        if (robot_paths_[robot_idx]) {
            const auto& rp = robot_paths_[robot_idx];
            for (size_t k = 0; k < rp->getControlCount(); ++k)
                path_end_timestep += (int)std::round(rp->getControlDuration(k) / step_size);
        }

        int conflict_ts = conflict.timestep;

        if (conflict_ts >= path_end_timestep) {
            // Robot is stationary at goal — all boundary states collapse to goal
            info.region_entry_state   = si->getStateSpace()->allocState();
            info.region_exit_state    = si->getStateSpace()->allocState();
            info.planning_entry_state = si->getStateSpace()->allocState();
            info.planning_exit_state  = si->getStateSpace()->allocState();
            si->copyState(info.region_entry_state,   goal_states_[robot_idx]);
            si->copyState(info.region_exit_state,    goal_states_[robot_idx]);
            si->copyState(info.planning_entry_state, goal_states_[robot_idx]);
            si->copyState(info.planning_exit_state,  goal_states_[robot_idx]);
            info.start_timestep         = path_end_timestep;
            info.end_timestep           = path_end_timestep;
            info.planning_entry_timestep = path_end_timestep;
            return true;
        }

        // Scan backwards from conflict to find region entry
        int entry_timestep = 0;
        bool found_entry = false;
        int pre_entry_timestep = -1;

        for (int t = conflict_ts; t >= 0; --t) {
            ob::State* st = getStateAtTimestep(robot_idx, t);
            if (!st) break;
            int region = decomp_->locateRegion(st);
            if (region_set.find(region) == region_set.end()) {
                entry_timestep     = t + 1;
                pre_entry_timestep = t;
                found_entry        = true;
                break;
            }
        }
        if (!found_entry)
            entry_timestep = 0;

        // Scan forward from conflict to find region exit
        int exit_timestep = -1;
        bool found_exit = false;
        int last_in_region_timestep = -1;

        for (int t = conflict_ts; t < path_end_timestep; ++t) {
            ob::State* st = getStateAtTimestep(robot_idx, t);
            if (!st) break;
            int region = decomp_->locateRegion(st);
            if (region_set.find(region) == region_set.end()) {
                exit_timestep             = t;
                last_in_region_timestep   = t - 1;
                found_exit                = true;
                break;
            }
        }
        if (!found_exit)
            exit_timestep = path_end_timestep;

        info.region_entry_state   = si->getStateSpace()->allocState();
        info.region_exit_state    = si->getStateSpace()->allocState();
        info.planning_entry_state = si->getStateSpace()->allocState();
        info.planning_exit_state  = si->getStateSpace()->allocState();

        if (!found_entry) {
            si->copyState(info.region_entry_state,   start_states_[robot_idx]);
            si->copyState(info.planning_entry_state, start_states_[robot_idx]);
        } else {
            si->copyState(info.region_entry_state,   getStateAtTimestep(robot_idx, entry_timestep));
            si->copyState(info.planning_entry_state, getStateAtTimestep(robot_idx, pre_entry_timestep));
        }

        if (exit_timestep >= path_end_timestep) {
            si->copyState(info.planning_exit_state, goal_states_[robot_idx]);
            si->copyState(info.region_exit_state,   goal_states_[robot_idx]);
        } else {
            si->copyState(info.planning_exit_state, getStateAtTimestep(robot_idx, exit_timestep));
            if (last_in_region_timestep >= 0)
                si->copyState(info.region_exit_state, getStateAtTimestep(robot_idx, last_in_region_timestep));
            else
                si->copyState(info.region_exit_state, info.planning_exit_state);
        }

        si->getStateSpace()->printState(info.planning_entry_state, DOUT);
        si->getStateSpace()->printState(info.region_entry_state,   DOUT);
        si->getStateSpace()->printState(info.planning_exit_state,  DOUT);
        si->getStateSpace()->printState(info.region_exit_state,    DOUT);

        info.start_timestep          = entry_timestep;
        info.end_timestep            = exit_timestep;
        info.planning_entry_timestep = (pre_entry_timestep >= 0) ? pre_entry_timestep : 0;
        return true;
    };

    auto robot_key  = std::make_pair(robot_1, robot_2);
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

    if (success_1 && success_2)
        robot_pair_refinement_info[robot_key][expanded_regions] = {update_info_1, update_info_2};

    return success_1 && success_2;
}

void CipherKinoPlanner::freeUpdateInfoStates(
    size_t robot_1, size_t robot_2,
    PathUpdateInfo& update_info_1,
    PathUpdateInfo& update_info_2)
{
    // auto freeIfNotNull = [](ob::SpaceInformation* si, ob::State*& s) {
    //     if (s) { si->freeState(s); s = nullptr; }
    // };
    // if (robot_1 < robot_sis_.size()) {
    //     auto* si = robot_sis_[robot_1].get();
    //     freeIfNotNull(si, update_info_1.region_entry_state);
    //     freeIfNotNull(si, update_info_1.region_exit_state);
    //     freeIfNotNull(si, update_info_1.planning_entry_state);
    //     freeIfNotNull(si, update_info_1.planning_exit_state);
    // }
    // if (robot_2 < robot_sis_.size()) {
    //     auto* si = robot_sis_[robot_2].get();
    //     freeIfNotNull(si, update_info_2.region_entry_state);
    //     freeIfNotNull(si, update_info_2.region_exit_state);
    //     freeIfNotNull(si, update_info_2.planning_entry_state);
    //     freeIfNotNull(si, update_info_2.planning_exit_state);
    // }
}

// ============================================================================
// Decomposition Helpers
// ============================================================================

// std::shared_ptr<DecompositionImpl> CipherKinoPlanner::createLocalDecomposition(
//     int parent_region,
//     double subdivision_factor)
// {
//     // TODO: look up parent region bounds, create finer DecompositionImpl inside them
//     return nullptr;
// }

// std::shared_ptr<DecompositionImpl> CipherKinoPlanner::createMultiCellDecomposition(
//     const std::vector<int>& regions,
//     double subdivision_factor)
// {
//     // TODO: compute union of region bounds, create finer DecompositionImpl inside them
//     return nullptr;
// }

// bool CipherKinoPlanner::extractReplanningBounds(
//     const SegmentConflict& conflict,
//     int conflict_region,
//     PathUpdateInfo& update_info_1,
//     PathUpdateInfo& update_info_2)
// {
//     // TODO: single-region variant of extractReplanningBoundsForExpandedRegion
//     return false;
// }

void CipherKinoPlanner::integrateRefinedPaths(
    const std::vector<size_t>& robot_indices,
    const std::vector<KinoGuidedPlanningResult>& local_results,
    const PathUpdateInfo& update_info_1,
    const PathUpdateInfo& update_info_2)
{
    // TODO: Need to adjust for controls and durations
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
            auto spliced_path = std::make_shared<oc::PathControl>(si);

            // Part 1: copy original up to and including the entry state
            size_t entry_state_idx = 0;
            for (size_t s = 0; s < original_path->getStateCount(); ++s) {
                if (si->getStateSpace()->distance(original_path->getState(s), update_info.planning_entry_state) < 1e-3) {
                    DOUT << "!!Found Start!!" << std::endl;
                    entry_state_idx = s;
                    break;
                }
            }
            // Part 1: original path up to and including the entry state (with controls)
            spliced_path->append(original_path->getState(0));
            for (size_t s = 0; s < entry_state_idx && s < original_path->getControlCount(); ++s)
                spliced_path->append(original_path->getState(s + 1),
                                     original_path->getControl(s),
                                     original_path->getControlDuration(s));

            // Part 2: refined path controls (entry state already appended, skip it)
            for (size_t s = 0; s < result.path->getControlCount(); ++s)
                spliced_path->append(result.path->getState(s + 1),
                                     result.path->getControl(s),
                                     result.path->getControlDuration(s));

            // Part 3: original path after the exit state (with controls)
            size_t exit_state_idx = original_path->getStateCount() - 1;
            for (size_t s = entry_state_idx; s < original_path->getStateCount(); ++s) {
                if (si->getStateSpace()->distance(original_path->getState(s), update_info.planning_exit_state) < 1e-3) {
                    DOUT << "!!Found Exit!!" << std::endl;
                    exit_state_idx = s;
                    break;
                }
            }
            for (size_t s = exit_state_idx; s < original_path->getControlCount(); ++s)
                spliced_path->append(original_path->getState(s + 1),
                                     original_path->getControl(s),
                                     original_path->getControlDuration(s));

            DOUT << "      Spliced path has " << spliced_path->getStateCount() << " states" << std::endl;

            guided_planning_results_[robot_idx].path = spliced_path;
            robot_paths_[robot_idx] = spliced_path;
        } else {
            guided_planning_results_[robot_idx] = result;
            robot_paths_[robot_idx] = result.path;
        }
    }
}

void CipherKinoPlanner::recheckConflictsFromTimestep(int start_timestep)
{
    DOUT << "Re-checking conflicts from timestep " << start_timestep << std::endl;

    segment_conflicts_.clear();

    int max_timestep = 0;
    for (size_t r = 0; r < robot_paths_.size(); ++r) {
        if (!robot_paths_[r] || robot_paths_[r]->getStateCount() == 0) continue;
        auto* oc_si = dynamic_cast<oc::SpaceInformation*>(robot_sis_[r].get());
        double step_size = oc_si ? oc_si->getPropagationStepSize() : 1.0;
        int total_steps = 0;
        for (size_t k = 0; k < robot_paths_[r]->getControlCount(); ++k)
            total_steps += (int)std::round(robot_paths_[r]->getControlDuration(k) / step_size);
        max_timestep = std::max(max_timestep, total_steps);
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
                    return;
                }
            }
        }
    }

    DOUT << "    Total conflicts found: " << segment_conflicts_.size() << std::endl;

}

int CipherKinoPlanner::getRecheckStartTimestep(
    const SegmentConflict& conflict,
    const PathUpdateInfo& update_info_1,
    const PathUpdateInfo& update_info_2)
{
    DOUT << "Getting recheck start timestep..." << std::endl;

    int result = std::min(update_info_1.planning_entry_timestep,
                          update_info_2.planning_entry_timestep);

    DOUT << "    Recheck start: using planning_entry_timestep " << result
              << " (conflict timestep was " << conflict.timestep << ")" << std::endl;

    return result;
}

// ============================================================================
// Subproblem Expansion Helpers
// ============================================================================

std::vector<int> CipherKinoPlanner::getExpandedRegion(int center_region, int expansion_layers)
{
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

void CipherKinoPlanner::computeExpandedBounds(
    const std::vector<int>& regions,
    std::vector<double>& env_min,
    std::vector<double>& env_max)
{
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

// ============================================================================
// Composite Planner Path Extraction
// ============================================================================

bool CipherKinoPlanner::extractIndividualPaths(
    const std::shared_ptr<oc::PathControl>& compound_path,
    const std::vector<size_t>& robot_indices,
    std::vector<std::shared_ptr<oc::PathControl>>& individual_paths)
{
    if (!compound_path || compound_path->getStateCount() == 0) return false;

    const size_t n_robots = robot_indices.size();
    individual_paths.clear();
    individual_paths.reserve(n_robots);

    for (size_t i = 0; i < n_robots; ++i) {
        size_t r = robot_indices[i];
        auto robot_si_kino = std::dynamic_pointer_cast<oc::SpaceInformation>(robot_sis_[r]);
        if (!robot_si_kino) { individual_paths.push_back(nullptr); continue; }

        auto path_i = std::make_shared<oc::PathControl>(robot_si_kino);

        // Append first state
        const auto* cs0 = compound_path->getState(0)->as<ob::CompoundState>();
        path_i->append(cs0->components[i]);

        // Append subsequent states with their controls and durations
        for (size_t k = 0; k < compound_path->getControlCount(); ++k) {
            const auto* cs_next = compound_path->getState(k + 1)->as<ob::CompoundState>();
            const auto* cc = compound_path->getControl(k)
                                 ->as<oc::CompoundControlSpace::ControlType>();
            double dur = compound_path->getControlDuration(k);
            path_i->append(cs_next->components[i], cc->components[i], dur);
        }

        individual_paths.push_back(path_i);
    }
    return true;
}

// ============================================================================
// Visualization
// ============================================================================

void CipherKinoPlanner::initVizHeader()
{
    if (!do_viz_) return;

    viz_header_ = YAML::Node();
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
        geom["type"]   = "sphere";
        geom["radius"] = 0.15;
        rn["geometry"] = geom;

        YAML::Node start_node; start_node.push_back(s_reals[0]); start_node.push_back(s_reals[1]); start_node.push_back(0.0);
        YAML::Node goal_node;  goal_node.push_back(g_reals[0]);  goal_node.push_back(g_reals[1]);  goal_node.push_back(0.0);
        rn["start"] = start_node;
        rn["goal"]  = goal_node;
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
        YAML::Node bmin; bmin.push_back(rb.low[0]);  bmin.push_back(rb.low[1]);  bmin.push_back(0.0);
        YAML::Node bmax; bmax.push_back(rb.high[0]); bmax.push_back(rb.high[1]); bmax.push_back(0.0);
        bounds["min"] = bmin;
        bounds["max"] = bmax;
        cn["bounds"] = bounds;
        viz_cells.push_back(cn);
    }
    viz_grid["cells"] = viz_cells;
    viz_header_["grid"] = viz_grid;

    vizWriteFile();
    DOUT << "[viz] Header written (" << num_regions << " cells) to " << viz_file_ << std::endl;
}

void CipherKinoPlanner::vizWriteFile() const
{
    if (!do_viz_ || viz_file_.empty()) return;
    YAML::Node doc;
    doc["header"] = viz_header_;
    doc["events"]  = YAML::Node();
    for (const auto& ev : viz_events_)
        doc["events"].push_back(ev);
    std::ofstream f(viz_file_);
    f << doc;
}

void CipherKinoPlanner::vizEmitCoupledPlanning(
    const std::vector<size_t>& robot_indices,
    const std::vector<int>& cell_ids)
{
    if (!do_viz_) return;
    YAML::Node ev;
    ev["type"] = "coupled_planning";
    for (auto r : robot_indices)  ev["robots"].push_back(static_cast<int>(r));
    for (auto c : cell_ids)       ev["cells"].push_back(c);
    viz_events_.push_back(ev);
}

void CipherKinoPlanner::vizEmitGridUpdate(
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
        YAML::Node c;
        c["id"] = id;
        YAML::Node bounds;
        bounds["min"] = bmin;
        bounds["max"] = bmax;
        c["bounds"] = bounds;
        cells_node.push_back(c);
    }
    ev["cells"] = cells_node;
    viz_events_.push_back(ev);
    vizWriteFile();
}

void CipherKinoPlanner::vizEmitConflicts(const std::vector<SegmentConflict>& conflicts)
{
    if (!do_viz_ || conflicts.empty()) return;

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

void CipherKinoPlanner::separateStartCells()
{
    DOUT << "Checking for robots sharing start cells..." << std::endl;

    auto grid_decomp = std::dynamic_pointer_cast<GridDecompositionImpl>(decomp_);
    int max_levels = config_.conflict_resolution_config.max_refinement_levels;

    std::set<int> once_decomposed;

    while (true) {
        std::vector<int> start_regions(start_states_.size());
        for (size_t i = 0; i < start_states_.size(); ++i)
            start_regions[i] = decomp_->locateSubRegion(start_states_[i]);

        std::map<int, std::vector<size_t>> cell_to_robots;
        for (size_t i = 0; i < start_regions.size(); ++i)
            cell_to_robots[start_regions[i]].push_back(i);

        std::vector<int> to_decompose;
        for (auto& [cell, robots] : cell_to_robots)
            if (robots.size() > 1)
                to_decompose.push_back(cell);

        if (to_decompose.empty()) break;

        for (int cell : to_decompose) {
            if (decomp_->getDecompositionDepth(cell) >= max_levels) {
                throw std::runtime_error(
                    "separateStartCells: cannot separate robots — cell " +
                    std::to_string(cell) + " reached maximum decomposition depth (" +
                    std::to_string(max_levels) + ")");
            }
        }

        std::vector<int> filtered;
        for (int cell : to_decompose)
            if (!once_decomposed.count(cell))
                filtered.push_back(cell);

        if (filtered.empty()) {
            throw std::runtime_error(
                "separateStartCells: cannot separate robots — start positions "
                "are too close; cells already decomposed once");
        }

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

bool CipherKinoPlanner::decomposeAllLeavesOneLevel()
{
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

// ============================================================================
// Entry Point
// ============================================================================

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

    CipherKinoConfig config;

    if (vm.count("cfg")) {
        try {
            YAML::Node cfg = YAML::LoadFile(configFile);
            if (cfg["timelimit"])             config.planning_time_limit = cfg["timelimit"].as<double>();
            if (cfg["max_total_time"])        config.max_total_time = cfg["max_total_time"].as<double>();
            if (cfg["seed"])                  config.seed = cfg["seed"].as<int>();
            if (cfg["region_size"])           config.decomposition_region_length = cfg["region_size"].as<int>();
            if (cfg["cbs_capacity"])          config.mapf_config.region_capacity = cfg["cbs_capacity"].as<int>();
            if (cfg["goal_threshold"])        config.goal_threshold = cfg["goal_threshold"].as<double>();
            if (cfg["robot_cell_size_ratio"]) config.robot_cell_size_ratio = cfg["robot_cell_size_ratio"].as<double>();
            if (cfg["models_base_path"])      config.models_base_path = cfg["models_base_path"].as<std::string>();
            if (cfg["motions_file"])          config.motions_file = cfg["motions_file"].as<std::string>();
            if (cfg["guided_planner_method"]) config.guided_planner_method = cfg["guided_planner_method"].as<std::string>();
            if (cfg["composite_planner_method"]) config.composite_planner_method = cfg["composite_planner_method"].as<std::string>();
            if (cfg["max_obstacle_volume_percent"])
                config.mapf_config.max_obstacle_volume_percent = cfg["max_obstacle_volume_percent"].as<double>();
            if (cfg["max_refinement_levels"])
                config.conflict_resolution_config.max_refinement_levels = cfg["max_refinement_levels"].as<int>();
            if (cfg["max_expansion_layers"])
                config.conflict_resolution_config.max_expansion_layers = cfg["max_expansion_layers"].as<int>();
            if (cfg["max_composite_attempts"])
                config.conflict_resolution_config.max_composite_attempts = cfg["max_composite_attempts"].as<int>();
            if (cfg["escalation_frequency"])
                config.conflict_resolution_config.escalation_frequency = cfg["escalation_frequency"].as<int>();
            if (cfg["do_optimization"])
                config.options_dbrrt.do_optimization = cfg["do_optimization"].as<bool>();
            if (cfg["solver_id"])
                config.options_trajopt.solver_id = cfg["solver_id"].as<int>();
            if (cfg["region_bounds_weight"])
                config.options_trajopt.region_bounds_weight = cfg["region_bounds_weight"].as<double>();
            if (cfg["check_transition_feasibility"])
                config.check_transition_feasibility = cfg["check_transition_feasibility"].as<bool>();
            if (cfg["transition_feasibility_robot_size_multiplier"])
                config.transition_feasibility_robot_size_multiplier =
                    cfg["transition_feasibility_robot_size_multiplier"].as<double>();
        } catch (const YAML::Exception& e) {
            std::cerr << "ERROR loading config file: " << e.what() << std::endl;
            return 1;
        }
    }

    if (config.seed >= 0) {
        DOUT << "Setting random seed to: " << config.seed << std::endl;
        ompl::RNG::setSeed(config.seed);
        config.options_dbrrt.seed = config.seed;
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

    CipherKinoPlanner planner(config);
    if (!vizFile.empty()) planner.setVizFile(vizFile);
    planner.loadProblem(robot_types, starts, goals, obstacles, env_min, env_max);
    CipherKinoResult result = planner.plan();

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
            YAML::Node waypoints;
            if (gpr.path) {
                const auto& path = *gpr.path;
                auto si = path.getSpaceInformation();
                const size_t n = path.getStateCount();
                for (size_t s = 0; s < n; ++s) {
                    YAML::Node wp;

                    std::vector<double> reals;
                    si->getStateSpace()->copyToReals(reals, path.getState(s));
                    YAML::Node state_node;
                    for (double v : reals) state_node.push_back(v);
                    wp["state"] = state_node;

                    YAML::Node ctrl;
                    if (s < path.getControlCount()) {
                        auto* oc_si = dynamic_cast<oc::SpaceInformation*>(si.get());
                        if (oc_si) {
                            auto ctrl_space = oc_si->getControlSpace()->as<oc::RealVectorControlSpace>();
                            const double* ctrl_vals = path.getControl(s)->as<oc::RealVectorControlSpace::ControlType>()->values;
                            for (unsigned int k = 0; k < ctrl_space->getDimension(); ++k)
                                ctrl.push_back(ctrl_vals[k]);
                        } else {
                            ctrl.push_back(0.0); ctrl.push_back(0.0);
                        }
                    } else {
                        ctrl.push_back(0.0); ctrl.push_back(0.0);
                    }
                    wp["control"] = ctrl;
                    wp["duration"] = (s < path.getControlCount()) ? path.getControlDuration(s) : 0.0;

                    waypoints.push_back(wp);
                }
            }
            robot_node["waypoints"] = waypoints;
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
