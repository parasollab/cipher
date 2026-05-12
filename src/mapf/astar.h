#ifndef MAPF_ASTAR_H
#define MAPF_ASTAR_H

#include "cbs.h"
#include "../utils/decomposition.h"
#include <fcl/fcl.h>
#include <ompl/base/State.h>
#include <memory>
#include <set>
#include <vector>

// Independent A* (IA*) MAPF solver.
// Each robot plans its shortest path through the region graph independently,
// ignoring other robots. Conflicts are expected and handled downstream.
class IndependentAStar {
public:
    IndependentAStar(int region_capacity, double timeout,
                     const std::vector<fcl::CollisionObjectf*>& obstacles = {},
                     double max_obstacle_volume_percent = 1.0);

    // Same interface as CBS::solve() for drop-in substitution.
    std::vector<std::vector<int>> solve(
        std::shared_ptr<DecompositionImpl> decomp,
        const std::vector<ompl::base::State*>& start_states,
        const std::vector<ompl::base::State*>& goal_states,
        const std::set<int>& allowed_regions = {},
        const ForbiddenEdgeSet& forbidden_edges = {});

    std::string getName() const { return "IndependentAStar"; }

private:
    int region_capacity_;
    double timeout_;
    std::vector<fcl::CollisionObjectf*> obstacles_;
    double max_obstacle_volume_percent_;

    RegionGraph buildRegionGraph(std::shared_ptr<DecompositionImpl> decomp,
                                 const std::set<int>& allowed_regions,
                                 const ForbiddenEdgeSet& forbidden_edges);
    std::set<int> computeInvalidRegions(std::shared_ptr<DecompositionImpl> decomp);

    // Returns region sequence from start to goal, or empty vector if no path.
    std::vector<int> findPath(const RegionGraph& graph,
                              int start_region, int goal_region);
};

#endif // MAPF_ASTAR_H
