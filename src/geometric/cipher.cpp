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
    setupRobots();
    setupCollisionManager();
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

    } catch (const std::exception& e) {
        std::cerr << "Planning failed with exception: " << e.what() << std::endl;
        result.success = false;
        result.failure_reason = std::string("exception: ") + e.what();
    }

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
    // TODO: plan each robot with GuidedGeometricRRT and populate guided_planning_results_

    // Emit low_level_paths event per robot once paths are available
    if (do_viz_ && !guided_planning_results_.empty()) {
        for (int r = 0; r < (int)guided_planning_results_.size(); ++r) {
            auto si = robots_[r]->getSpaceInformation();
            const auto& path = guided_planning_results_[r];
            YAML::Node ev;
            ev["type"] = "low_level_paths";
            YAML::Node paths;
            YAML::Node waypoints;
            const size_t n = path.getStateCount();
            for (size_t i = 0; i < n; ++i) {
                std::vector<double> reals;
                si->getStateSpace()->copyToReals(reals, path.getState(i));
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
}

bool CipherGeometricPlanner::checkSegmentsForConflicts() {
    std::cout << "Checking segments for conflicts..." << std::endl;
    return true;
}

bool CipherGeometricPlanner::resolveConflicts() {
    std::cout << "Resolving conflicts..." << std::endl;
    return true;
}

// Private helpers
void CipherGeometricPlanner::setupDecomposition() {
    std::cout << "Setting up decomposition..." << std::endl;
    /// TODO: We need to remove the hardcoded 2D assumption. Need a way to get the dimension of the workspace.
    auto decomp = std::make_shared<GridDecompositionImpl>(2, workspace_bounds_, config_.decomposition_region_length);
    decomp->setStateSpace(robots_[0]->getSpaceInformation()->getStateSpace());
    decomp_ = decomp;
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

        // Create start state
        ob::State* start = si->getStateSpace()->allocState();
        si->getStateSpace()->copyFromReals(start, starts_[i]);
        start_states_.push_back(start);

        // Create goal state
        ob::State* goal = si->getStateSpace()->allocState();
        si->getStateSpace()->copyFromReals(goal, goals_[i]);
        goal_states_.push_back(goal);
    }
}

void CipherGeometricPlanner::cleanup() {
    std::cout << "Cleaning up..." << std::endl;
}

// Private conflict checking helpers
std::vector<fcl::CollisionObjectf*> CipherGeometricPlanner::getObstaclesInRegion(
    const std::vector<double>& region_min,
    const std::vector<double>& region_max) const {
    std::cout << "Getting obstacles in region..." << std::endl;
    return {};
}

const PathSegment* CipherGeometricPlanner::findSegmentAtTimestep(size_t robot_idx, int timestep) const {
    std::cout << "Finding segment at timestep..." << std::endl;
    return nullptr;
}

void CipherGeometricPlanner::propagateToTimestep(size_t robot_idx, size_t segment_idx, int timestep, ob::State* result) const {
    std::cout << "Propagating to timestep..." << std::endl;
}

bool CipherGeometricPlanner::checkTwoRobotConflict(size_t robot_idx_1, const ob::State* state_1,
                                               size_t robot_idx_2, const ob::State* state_2,
                                               size_t& part_1, size_t& part_2) const {
    std::cout << "Checking two-robot conflict..." << std::endl;
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
    return false;
}

bool attemptRefinementAtExpansionLevel(
    const SegmentConflict& conflict,
    int conflict_region,
    const std::vector<int>& expanded_regions,
    int expansion_layer,
    int max_refinement_levels,
    ConflictResolutionEntry& log_entry) {
    std::cout << "Attempting refinement at expansion level..." << std::endl;
    return false;
}

bool CipherGeometricPlanner::refineExpandedRegion(
    const SegmentConflict& conflict,
    const std::vector<int>& expanded_regions,
    int refinement_level) {
    std::cout << "Refining expanded region..." << std::endl;
    return false;
}

int CipherGeometricPlanner::calculateMaxExpansionLayers() const {
    std::cout << "Calculating max expansion layers..." << std::endl;
    return 0;
}

bool CipherGeometricPlanner::expansionCoversFullDecomposition(int expansion_layers) const {
    std::cout << "Checking if expansion covers full decomposition..." << std::endl;
    return false;
}

void CipherGeometricPlanner::freeUpdateInfoStates(
    size_t robot_1, size_t robot_2,
    PathUpdateInfo& update_info_1,
    PathUpdateInfo& update_info_2) {
    std::cout << "Freeing update info states..." << std::endl;
}

// Private helpers for all conflict strategies
DecompositionImpl* CipherGeometricPlanner::createLocalDecomposition(
    int parent_region,
    double subdivision_factor) {
    std::cout << "Creating local decomposition..." << std::endl;
    return nullptr;
}

DecompositionImpl* CipherGeometricPlanner::createMultiCellDecomposition(
    const std::vector<int>& regions,
    double subdivision_factor) {
    std::cout << "Creating multi-cell decomposition..." << std::endl;
    return nullptr;
}

bool CipherGeometricPlanner::extractReplanningBounds(
    const SegmentConflict& conflict,
    int conflict_region,
    PathUpdateInfo& update_info_1,
    PathUpdateInfo& update_info_2) {
    std::cout << "Extracting replanning bounds..." << std::endl;
    return false;
}

void CipherGeometricPlanner::integrateRefinedPaths(
    const std::vector<size_t>& robot_indices,
    const std::vector<og::PathGeometric>& local_results,
    const PathUpdateInfo& update_info_1,
    const PathUpdateInfo& update_info_2) {
    std::cout << "Integrating refined paths..." << std::endl;
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
    return {};
}

void CipherGeometricPlanner::computeExpandedBounds(
    const std::vector<int>& regions,
    std::vector<double>& env_min,
    std::vector<double>& env_max) {
    std::cout << "Computing expanded bounds..." << std::endl;
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

void CipherGeometricPlanner::recordRefinement(int parent_region, const DecompositionImpl* local_decomp) {
    std::cout << "Recording refinement in decomposition hierarchy..." << std::endl;
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
