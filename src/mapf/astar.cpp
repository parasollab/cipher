#include "astar.h"
#include <queue>
#include <stdexcept>
#include <algorithm>
#include <vector>

IndependentAStar::IndependentAStar(int region_capacity, double timeout,
                                   const std::vector<fcl::CollisionObjectf*>& obstacles,
                                   double max_obstacle_volume_percent)
    : region_capacity_(region_capacity), timeout_(timeout),
      obstacles_(obstacles), max_obstacle_volume_percent_(max_obstacle_volume_percent)
{}

std::vector<std::vector<int>> IndependentAStar::solve(
    std::shared_ptr<DecompositionImpl> decomp,
    const std::vector<ompl::base::State*>& start_states,
    const std::vector<ompl::base::State*>& goal_states,
    const std::set<int>& allowed_regions,
    const ForbiddenEdgeSet& forbidden_edges)
{
    RegionGraph graph = buildRegionGraph(decomp, allowed_regions, forbidden_edges);

    int num_robots = (int)start_states.size();
    std::vector<std::vector<int>> paths(num_robots);

    for (int r = 0; r < num_robots; ++r) {
        int start_region = decomp->locateSubRegion(start_states[r]);
        int goal_region  = decomp->locateSubRegion(goal_states[r]);

        if (start_region < 0)
            throw std::runtime_error("IndependentAStar: start state for robot " +
                                     std::to_string(r) + " is outside decomposition bounds");
        if (goal_region < 0)
            throw std::runtime_error("IndependentAStar: goal state for robot " +
                                     std::to_string(r) + " is outside decomposition bounds");

        paths[r] = findPath(graph, start_region, goal_region);
    }

    return paths;
}

// Duplicated from CBS — builds a directed region graph filtered by allowed regions,
// forbidden edges, and obstacle volume threshold.
RegionGraph IndependentAStar::buildRegionGraph(std::shared_ptr<DecompositionImpl> decomp,
                                               const std::set<int>& allowed_regions,
                                               const ForbiddenEdgeSet& forbidden_edges)
{
    int total = decomp->getTotalNumRegions();
    RegionGraph graph(total);

    std::set<int> invalid = computeInvalidRegions(decomp);

    for (int i = 0; i < total; ++i) {
        if (!decomp->isLeafRegion(i)) continue;
        if (!allowed_regions.empty() && !allowed_regions.count(i)) continue;
        if (invalid.count(i)) continue;

        std::vector<int> neighbors;
        decomp->getNeighbors(i, neighbors);

        for (int neighbor : neighbors) {
            if (!decomp->isLeafRegion(neighbor)) continue;
            if (!allowed_regions.empty() && !allowed_regions.count(neighbor)) continue;
            if (invalid.count(neighbor)) continue;
            if (!forbidden_edges.empty() && forbidden_edges.count({i, neighbor})) continue;
            EdgeProperty ep;
            ep.weight = 1.0;
            boost::add_edge(i, neighbor, ep, graph);
        }
    }

    return graph;
}

std::set<int> IndependentAStar::computeInvalidRegions(std::shared_ptr<DecompositionImpl> decomp)
{
    std::set<int> invalid;
    if (obstacles_.empty() || max_obstacle_volume_percent_ >= 1.0)
        return invalid;

    int total = decomp->getTotalNumRegions();
    for (int i = 0; i < total; ++i) {
        if (!decomp->isLeafRegion(i)) continue;
        double region_vol = decomp->getRegionVolume(i);
        const auto bounds = decomp->getCellBounds(i);

        double obstacle_area = 0.0;
        for (auto* obs : obstacles_) {
            const auto& aabb = obs->getAABB();
            double ix = std::max(0.0, std::min((double)aabb.max_[0], bounds.high[0]) -
                                      std::max((double)aabb.min_[0], bounds.low[0]));
            double iy = std::max(0.0, std::min((double)aabb.max_[1], bounds.high[1]) -
                                      std::max((double)aabb.min_[1], bounds.low[1]));
            obstacle_area += ix * iy;
        }

        if (obstacle_area / region_vol > max_obstacle_volume_percent_)
            invalid.insert(i);
    }
    return invalid;
}

// Dijkstra's shortest path over the region graph (uniform edge weights = 1).
// Returns the region sequence from start to goal, or empty if no path exists.
std::vector<int> IndependentAStar::findPath(const RegionGraph& graph,
                                            int start_region, int goal_region)
{
    if (start_region == goal_region)
        return {start_region};

    int total = (int)boost::num_vertices(graph);

    struct Node {
        int g_cost;
        int region;
        bool operator>(const Node& o) const { return g_cost > o.g_cost; }
    };

    // parent[r] == -2: unvisited; -1: start node; >= 0: parent region
    std::vector<int> parent(total, -2);
    std::vector<bool> closed(total, false);
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;

    parent[start_region] = -1;
    open.push({0, start_region});

    while (!open.empty()) {
        Node current = open.top();
        open.pop();

        if (closed[current.region]) continue;
        closed[current.region] = true;

        if (current.region == goal_region) {
            std::vector<int> path;
            for (int r = goal_region; r != -1; r = parent[r])
                path.push_back(r);
            std::reverse(path.begin(), path.end());
            return path;
        }

        auto out_edges = boost::out_edges(current.region, graph);
        for (auto it = out_edges.first; it != out_edges.second; ++it) {
            int neighbor = (int)boost::target(*it, graph);
            if (closed[neighbor]) continue;
            if (parent[neighbor] == -2) {
                parent[neighbor] = current.region;
                open.push({current.g_cost + 1, neighbor});
            }
        }
    }

    return {};
}
