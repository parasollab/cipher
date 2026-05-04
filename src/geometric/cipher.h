#ifndef CIPHER_GEOMETRIC_H
#define CIPHER_GEOMETRIC_H

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/SpaceInformation.h>
#include <fcl/fcl.h>
#include <chrono>
#include <map>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <tuple>
#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>

#include "utils/decomposition.h"
#include "src/geometric/coupled_rrt.h"
#include "src/guided/guided_geometric_rrt.h"


namespace ob = ompl::base;
namespace oc = ompl::control;
namespace og = ompl::geometric;

// Goal condition that checks only the first two reals (x, y) of any state space.
// This lets geometric planning ignore velocity/angle components of kinodynamic states.
class PositionGoalCondition : public ob::GoalRegion
{
public:
    PositionGoalCondition(
        const ob::SpaceInformationPtr& si,
        const ob::State* goal_state,
        double threshold)
        : ob::GoalRegion(si), goal_state_(goal_state)
    {
        threshold_ = threshold;
    }

    double distanceGoal(const ob::State* st) const override
    {
        std::vector<double> s_reals, g_reals;
        si_->getStateSpace()->copyToReals(s_reals, st);
        si_->getStateSpace()->copyToReals(g_reals, goal_state_);
        double dx = s_reals[0] - g_reals[0];
        double dy = s_reals[1] - g_reals[1];
        return std::sqrt(dx * dx + dy * dy);
    }

private:
    const ob::State* goal_state_;  // non-owning; lifetime managed by goal_states in main
};


// ============================================================================
// Configuration Structure
// ============================================================================

struct MAPFConfig {
    std::string method = "cbs";  // Options: "decoupled", "astar", "cbs"
    int region_capacity = 1;           // Robots per vertex/edge (for CBS)
    double max_obstacle_volume_percent = 0.5;  // Maximum obstacle volume in a region (0.0 to 1.0)
    double mapf_timeout = 30.0;          // Timeout for MAPF solver in seconds
};

struct ConflictResolutionConfig {
    // Hierarchical refinement parameters
    int max_refinement_levels = 3;                // K: refinement levels per expansion (0=skip)
    double decomposition_subdivision_factor = 2.0; // Subdivision multiplier per refinement level

    // Expansion parameters
    int max_expansion_layers = -1;                // Max expansion layers (-1 = auto-detect based on grid)
                                                  // Auto-detect: ceil(sqrt(num_regions)/2)

    // Cycle detection: after this many conflicts for the same robot pair,
    // increase the minimum expansion layer by 1 (0 = disable escalation)
    int escalation_frequency = 3;

    // Composite planner fallback
    int max_composite_attempts = 1;               // Composite planner attempts (0=skip)
    bool composite_uses_full_problem = true;      // If true, composite plans ALL robots from original starts/goals

    // Re-checking behavior
    bool recheck_from_prior_segment = false;      // If true, start conflict re-checking from prior segment start
};

struct CipherGeometricConfig {
    int decomposition_region_length = 5.0;
    std::vector<int> decomposition_resolution = {10, 10, 1};  // Grid cells in [x, y, z]
    double planning_time_limit = 30.0;
    double max_total_time = 0.0;  // Maximum total planning time in seconds (0 = no limit)
    int seed = -1;  // Random seed (-1 for random)
    double goal_threshold = 0.5;  // Distance threshold for goal satisfaction

    // Decomposition output directory (empty string disables saving)
    std::string decomposition_output_dir = "";

    // MAPF configuration
    MAPFConfig mapf_config;

    // Composite planner config
    std::string composite_planner_method = "coupled_rrt";

    // Guided planner configuration
    std::string guided_planner_method = "guided_geometric_rrt";

    // Segmentation configuration
    int segment_timesteps = 30;  // Number of timesteps per segment

    // Conflict resolution configuration
    ConflictResolutionConfig conflict_resolution_config;
};

// ============================================================================
// Path Segment Structure
// ============================================================================

struct PathSegment {
    size_t robot_index;           // Which robot this segment belongs to
    size_t segment_index;         // Index of this segment in the robot's path
    ob::State* start_state;       // Start state of segment (owned by original path)
    ob::State* end_state;         // End state of segment (owned by original path)
    std::vector<ob::State*> states;      // States of segment (owned by original path)
    double total_duration;        // Total duration of this segment
    int start_timestep;           // Starting timestep of this segment
    int end_timestep;             // Ending timestep of this segment
};

// ============================================================================
// Segment Conflict Structure
// ============================================================================

struct SegmentConflict {
    enum ConflictType {
        ROBOT_ROBOT,      // Conflict between two robots
        ROBOT_OBSTACLE    // Conflict between robot and environment
    };

    ConflictType type;
    size_t robot_index_1;      // Primary robot (or only robot for ROBOT_OBSTACLE)
    size_t robot_index_2;      // Secondary robot (only for ROBOT_ROBOT)
    size_t segment_index_1;    // Segment of robot 1
    size_t segment_index_2;    // Segment of robot 2 (only for ROBOT_ROBOT)
    int timestep;              // Timestep where conflict occurred
    size_t part_index_1;       // Which part of robot 1 collided
    size_t part_index_2;       // Which part of robot 2 collided (only for ROBOT_ROBOT)

    SegmentConflict()
        : type(ROBOT_OBSTACLE), robot_index_1(0), robot_index_2(0),
          segment_index_1(0), segment_index_2(0), timestep(0),
          part_index_1(0), part_index_2(0) {}
};

// ============================================================================
// Path Update Info (for conflict resolution)
// ============================================================================

struct PathUpdateInfo {
    size_t robot_index;
    int start_timestep;            // first timestep inside the expanded region
    int end_timestep;              // first timestep outside the expanded region
    int planning_entry_timestep;   // timestep of planning_entry_state (last timestep outside region before entry, or 0)
    size_t start_segment_idx;
    size_t end_segment_idx;
    ob::State* region_entry_state;  // first state inside the expanded region
    ob::State* region_exit_state;   // last state inside the expanded region
    ob::State* planning_entry_state;    // last state outside the expanded region (before entering)
    ob::State* planning_exit_state;     // first state outside the expanded region (after exiting)
};

// ============================================================================
// Hierarchical Decomposition Cell Structure
// ============================================================================

// Represents a cell in the decomposition hierarchy
// If children is empty, this is a leaf cell with the given region_id
// If children is non-empty, this cell was refined and region_id is the parent
struct DecompositionCell {
    int region_id;                              // Original region ID in parent decomposition
    std::vector<DecompositionCell> children;    // Child cells if refined (empty = leaf)

    // Bounds of this cell (for visualization)
    std::vector<double> bounds_min;
    std::vector<double> bounds_max;

    DecompositionCell() : region_id(-1) {}
    explicit DecompositionCell(int id) : region_id(id) {}

    bool isRefined() const { return !children.empty(); }
};

// ============================================================================
// Strategy Attempt Log Entry
// ============================================================================

struct StrategyAttempt {
    std::string strategy;          // "hierarchical_refinement" or "composite_planner"
    int expansion_layer = -1;      // For hierarchical (-1 if N/A)
    int refinement_level = -1;     // For hierarchical (-1 if N/A)
    int attempt_number = -1;       // For composite planner attempt index (-1 if N/A)
    bool planning_succeeded = false;  // Did the replanning itself succeed?
    bool conflict_resolved = false;  // Did the conflict get resolved after this attempt?
};

// ============================================================================
// Per-Conflict Resolution Log Entry
// ============================================================================

struct ConflictResolutionEntry {
    int conflict_number = 0;      // 1-based index
    size_t robot_1 = 0;
    size_t robot_2 = 0;
    int timestep = 0;
    std::vector<StrategyAttempt> attempts;  // All strategy attempts for this conflict
    bool resolved = false;
    std::string outcome;           // "resolved", "strategies_exhausted", "timeout"
};

// ============================================================================
// Resolution Statistics Structure
// ============================================================================

struct ResolutionStats {
    // Counts of how many times each strategy was attempted
    int decomposition_refinement_attempts = 0;
    int subproblem_expansion_attempts = 0;
    int composite_planner_attempts = 0;

    // Counts of how many times each strategy successfully resolved a conflict
    int decomposition_refinement_successes = 0;
    int subproblem_expansion_successes = 0;
    int composite_planner_successes = 0;

    // Total conflicts encountered and resolved
    int total_conflicts_encountered = 0;
    int total_conflicts_resolved = 0;

    // Detailed per-conflict resolution log
    std::vector<ConflictResolutionEntry> conflict_log;
};

// ============================================================================
// Planning Result Structure
// ============================================================================

struct CipherGeometricResult {
    bool success = false;
    double planning_time = 0.0;
    std::string failure_reason;    // Empty on success; e.g. "timeout_high_level_paths",
                                   // "timeout_guided_paths", "timeout_conflict_resolution",
                                   // "strategies_exhausted", "exception: ..."
    ResolutionStats resolution_stats;
};

struct GeometricPlanningResult {
    bool solved;                               // Whether exact solution was found
    double planning_time;                      // Time spent planning (seconds)
    std::shared_ptr<og::PathGeometric> path;     // Solution path (compound path for all robots)
    std::vector<std::shared_ptr<og::PathGeometric>> individual_paths;  // Individual paths per robot
};

struct GuidedPlanningResult {
    bool success;
    double planning_time;
    std::shared_ptr<og::PathGeometric> path;
    size_t robot_index;
};

// ============================================================================
// CipherGeometricPlanner Class (Geometric Version)
// ============================================================================

class CipherGeometricPlanner {
public:
    // Constructor
    explicit CipherGeometricPlanner(const CipherGeometricConfig& config);

    // Destructor
    ~CipherGeometricPlanner();

    // Load problem from data structures
    void loadProblem(
        const std::vector<std::string>& robot_types,
        const std::vector<std::vector<double>>& starts,
        const std::vector<std::vector<double>>& goals,
        const std::vector<fcl::CollisionObjectf*>& obstacles,
        const std::vector<double>& env_min,
        const std::vector<double>& env_max);

    // Main planning method - runs full SyCLoP pipeline
    CipherGeometricResult plan();

    // Individual planning phases (can be called separately if needed)
    void computeHighLevelPaths();
    void computeGuidedPaths();
    void segmentGuidedPaths();
    bool checkSegmentsForConflicts();  // Checks robot-robot conflicts only; obstacle avoidance is handled by guided planner
    bool resolveConflicts();  // Returns true if all conflicts were resolved

    // Accessors
    const std::vector<std::vector<int>>& getHighLevelPaths() const { return high_level_paths_; }
    const std::shared_ptr<DecompositionImpl> getDecomposition() const { return decomp_; }
    const std::vector<GuidedPlanningResult>& getGuidedPaths() const { return guided_planning_results_; }
    const std::vector<std::vector<PathSegment>>& getPathSegments() const { return path_segments_; }
    const std::vector<std::shared_ptr<Robot>>& getRobots() const { return robots_; }
    const std::vector<SegmentConflict>& getConflicts() const { return segment_conflicts_; }

private:
    // Configuration
    CipherGeometricConfig config_;

    // Environment data
    std::vector<fcl::CollisionObjectf*> obstacles_;
    std::vector<double> env_min_;
    std::vector<double> env_max_;
    ob::RealVectorBounds workspace_bounds_;
    std::shared_ptr<DecompositionImpl> decomp_;

    // Robot data
    std::vector<std::shared_ptr<Robot>> robots_;
    std::vector<std::shared_ptr<ob::SpaceInformation>> robot_sis_;
    std::vector<std::string> robot_types_;
    std::vector<std::vector<double>> starts_;
    std::vector<std::vector<double>> goals_;
    std::vector<ob::State*> start_states_;
    std::vector<ob::State*> goal_states_;

    // Planning state
    std::vector<std::vector<int>> high_level_paths_;
    std::vector<GuidedPlanningResult> guided_planning_results_;
    std::vector<std::vector<PathSegment>> path_segments_;  // Segments for each robot
    std::vector<SegmentConflict> segment_conflicts_;     // Detected conflicts
    bool problem_loaded_ = false;
    ResolutionStats resolution_stats_;  // Track conflict resolution statistics
    std::map<std::tuple<size_t, size_t, int>, int> robot_pair_conflict_counts_;  // Cycle detection: (robot1, robot2, timestep) -> count

    // Timeout tracking
    std::chrono::steady_clock::time_point planning_start_time_;
    bool isTimeoutExceeded() const;

    // Collision manager for obstacles (shared with guided planners)
    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> collision_manager_;

    // Hierarchical decomposition tracking
    std::vector<DecompositionCell> decomposition_hierarchy_;  // One cell per initial region

    std::map<std::pair<int, int>, std::map<std::vector<int>, std::pair<PathUpdateInfo, PathUpdateInfo>>> robot_pair_refinement_info;

    // Tracks (expansion_layer, refinement_level) at which each leaf cell was last decomposed.
    // A cell may only be re-decomposed within the same expansion_layer at a strictly higher
    // refinement_level; cells refined in any earlier expansion_layer are never touched again.
    std::unordered_map<int, std::pair<int,int>> region_refinement_level_;

    // Maps every current leaf region integer ID → the viz cell ID string used in the
    // YAML event log (e.g. 42 → "c17_42").  Populated at setup and updated on every
    // Decompose() call so local_mapf events always emit IDs that viz.py recognises.
    std::unordered_map<int, std::string> region_viz_id_;

    // Helper methods
    void setupDecomposition();
    void setupCollisionManager();
    void setupRobots();
    void cleanup();

    std::vector<fcl::CollisionObjectf*> getObstaclesInRegion(
        const std::vector<double>& region_min,
        const std::vector<double>& region_max) const;

    // Conflict checking helpers
    const PathSegment* findSegmentAtTimestep(size_t robot_idx, int timestep) const;
    void propagateToTimestep(size_t robot_idx, size_t segment_idx, int timestep, ob::State* result) const;
    bool checkTwoRobotConflict(size_t robot_idx_1, const ob::State* state_1,
                               size_t robot_idx_2, const ob::State* state_2,
                               size_t& part_1, size_t& part_2) const;

    // Conflict resolution strategies
    void updateDecomposition();
    void expandSubproblem();
    GeometricPlanningResult useCompositePlanner(
        const std::vector<size_t>& robot_indices,
        const std::vector<std::vector<double>>& subproblem_starts,
        const std::vector<std::vector<double>>& subproblem_goals,
        const std::vector<double>& subproblem_env_min,
        const std::vector<double>& subproblem_env_max);

    // Conflict resolution - modular strategy system
    bool resolveConflictWithStrategies(const SegmentConflict& conflict,
                                        ConflictResolutionEntry& log_entry);
    bool conflictPersistsForRobots(size_t robot_1, size_t robot_2, int timestep) const;

    // Hierarchical conflict resolution strategy
    bool resolveWithHierarchicalExpansionRefinement(
        const SegmentConflict& conflict,
        int max_refinement_levels,
        int max_expansion_layers,
        int min_expansion_layer,
        ConflictResolutionEntry& log_entry);

    bool attemptRefinementAtExpansionLevel(
        const SegmentConflict& conflict,
        int conflict_region,
        const std::vector<int>& expanded_regions,
        int expansion_layer,
        int max_refinement_levels,
        ConflictResolutionEntry& log_entry);

    bool refineExpandedRegion(
        const SegmentConflict& conflict,
        int conflict_region,
        const std::vector<int>& expanded_regions,
        int expansion_layer,
        int refinement_level);

    int calculateMaxExpansionLayers() const;
    bool expansionCoversFullDecomposition(int expansion_layers) const;

    // Local composite planner (plans colliding robots jointly in local bounds)
    bool resolveWithLocalCompositePlanner(const SegmentConflict& conflict,
                                          ConflictResolutionEntry& log_entry);

    // Full-problem composite planner fallback
    bool resolveWithFullProblemCompositePlanner(int max_attempts,
                                                ConflictResolutionEntry& log_entry);

    // Replanning bounds for expanded region
    bool extractReplanningBoundsForExpandedRegion(
        const SegmentConflict& conflict,
        const std::vector<int>& expanded_regions,
        PathUpdateInfo& update_info_1,
        PathUpdateInfo& update_info_2);

    void freeUpdateInfoStates(
        size_t robot_1, size_t robot_2,
        PathUpdateInfo& update_info_1,
        PathUpdateInfo& update_info_2);

    // Helper methods for all strategies
    std::shared_ptr<DecompositionImpl> createLocalDecomposition(
        int parent_region,
        double subdivision_factor);
    std::shared_ptr<DecompositionImpl> createMultiCellDecomposition(
        const std::vector<int>& regions,
        double subdivision_factor);
    bool extractReplanningBounds(
        const SegmentConflict& conflict,
        int conflict_region,
        PathUpdateInfo& update_info_1,
        PathUpdateInfo& update_info_2);
    // bool extractReplanningBoundsForExpandedRegion(
    //     const SegmentConflict& collision,
    //     const std::vector<int>& expanded_regions,
    //     PathUpdateInfo& update_info_1,
    //     PathUpdateInfo& update_info_2);
    void integrateRefinedPaths(
        const std::vector<size_t>& robot_indices,
        const std::vector<GuidedPlanningResult>& local_results,
        const PathUpdateInfo& update_info_1,
        const PathUpdateInfo& update_info_2);
    void recheckConflictsFromTimestep(int start_timestep);
    int getRecheckStartTimestep(const SegmentConflict& conflict,
                                const PathUpdateInfo& update_info_1,
                                const PathUpdateInfo& update_info_2);
    void segmentSinglePath(
        size_t robot_idx,
        const std::shared_ptr<og::PathGeometric>& path,
        int start_timestep_offset,
        std::vector<PathSegment>& segments);

    // Subproblem expansion helpers
    std::vector<int> getExpandedRegion(int center_region, int expansion_layers);
    void computeExpandedBounds(
        const std::vector<int>& regions,
        std::vector<double>& env_min,
        std::vector<double>& env_max);

    // Composite planner helpers
    bool extractIndividualPaths(
        const std::shared_ptr<og::PathGeometric>& compound_path,
        std::vector<std::shared_ptr<og::PathGeometric>>& individual_paths);

    // Decomposition hierarchy tracking
    void initializeDecompositionHierarchy();
    DecompositionCell* findCellByRegion(int region_id);
    DecompositionCell* findCellByRegionRecursive(DecompositionCell& cell, int region_id);
    void recordRefinement(int parent_region, const std::shared_ptr<DecompositionImpl> local_decomp);
    YAML::Node serializeDecompositionHierarchy() const;
    YAML::Node serializeCellRecursive(const DecompositionCell& cell) const;

    // Visualization
    bool do_viz_ = false;
    std::string viz_file_;
    YAML::Node viz_header_;
    std::vector<YAML::Node> viz_events_;
    void initVizHeader();
    void vizWriteFile() const;
    void vizEmitCoupledPlanning(const std::vector<size_t>& robot_indices,
                                const std::vector<int>& cell_ids);
    // new_cells: (id_string, bounds_min, bounds_max) for each sub-cell created
    void vizEmitGridUpdate(const std::vector<std::string>& removed_viz_ids,
                           const std::vector<std::tuple<std::string,
                                                        std::vector<double>,
                                                        std::vector<double>>>& new_cells);

public:
    void setVizFile(const std::string& path) { viz_file_ = path; do_viz_ = !path.empty(); }
};

#endif //CIPHER_GEOMETRIC_H