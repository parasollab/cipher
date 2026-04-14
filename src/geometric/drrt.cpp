#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <limits>
#include <algorithm>

#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>

#include <fcl/fcl.h>
#include <fcl/narrowphase/collision.h>

#include <ompl/geometric/planners/prm/SPARStwo.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/State.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/ScopedState.h>
#include <ompl/util/RandomNumbers.h>

#include "robots.h"

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace po = boost::program_options;

// Forward declarations
class CompositeVertex;
class IndividualRoadmap;
class DiscreteRRTTree;
class DiscreteRRTPlanner;
class Robot;

// Represents a vertex in the tensor product graph
// Stores indices into each individual robot's roadmap
class CompositeVertex {
public:
    std::vector<unsigned int> vertex_indices;  // One index per robot
    CompositeVertex* parent;

    CompositeVertex(const std::vector<unsigned int>& indices)
        : vertex_indices(indices), parent(nullptr) {}

    // Hash function for use in unordered_set/map
    struct Hash {
        std::size_t operator()(const CompositeVertex& v) const;
    };

    // Equality comparison
    bool operator==(const CompositeVertex& other) const {
        return vertex_indices == other.vertex_indices;
    }

    // Check if this vertex differs from another in exactly one coordinate
    bool isNeighbor(const CompositeVertex& other) const;

    // Get the coordinate that differs (returns -1 if not a neighbor)
    int getDifferingCoordinate(const CompositeVertex& other) const;
};

// Stores a sparse roadmap for a single robot
class IndividualRoadmap {
public:
    struct SparsParams {
        bool has_max_failures = false;
        unsigned int max_failures = 0;
        bool has_sparse_delta_fraction = false;
        double sparse_delta_fraction = 0.0;
        bool has_dense_delta_fraction = false;
        double dense_delta_fraction = 0.0;
        bool has_stretch_factor = false;
        double stretch_factor = 0.0;
    };

    IndividualRoadmap(
        const std::shared_ptr<ob::SpaceInformation>& si,
        unsigned int num_samples = 500,
        double connection_radius = 1.0);

    IndividualRoadmap(
        const std::shared_ptr<ob::SpaceInformation>& si,
        unsigned int num_samples,
        double connection_radius,
        const SparsParams& spars_params);

    ~IndividualRoadmap() = default;

    // Build the roadmap with actual start and goal states
    void build(const ob::State* start, const ob::State* goal, double max_time = 10.0);

    // Get the number of vertices in the roadmap
    unsigned int getNumVertices() const;

    // Get the state associated with a vertex index
    const ob::State* getState(unsigned int index) const;

    // Get neighbors of a vertex
    std::vector<unsigned int> getNeighbors(unsigned int index) const;

    // Find the nearest vertex in the roadmap to a given state
    unsigned int getNearestVertex(const ob::State* state) const;

    // Find the neighbor of a vertex that moves closest to a target state
    // Returns the vertex index if found, or the same vertex if no valid neighbor
    unsigned int getNeighborToward(unsigned int vertex_idx, const ob::State* target_state) const;

    // Check if there's an edge between two vertices
    bool hasEdge(unsigned int v1, unsigned int v2) const;

private:
    void ensureVertexCache() const;

    std::shared_ptr<ob::SpaceInformation> si_;
    std::shared_ptr<og::SPARStwo> spars_;
    og::SimpleSetup ss_;

    unsigned int num_samples_;
    double connection_radius_;

    // Cache for neighbor lookups
    mutable std::unordered_map<unsigned int, std::vector<unsigned int>> neighbor_cache_;
    mutable bool vertex_cache_valid_ = false;
    mutable std::vector<og::SPARStwo::Vertex> vertex_cache_;
    mutable std::unordered_map<og::SPARStwo::Vertex, unsigned int> vertex_index_map_;
};

// Tree structure for discrete RRT
class DiscreteRRTTree {
public:
    DiscreteRRTTree(const std::vector<unsigned int>& root_indices);

    ~DiscreteRRTTree() = default;

    // Add a vertex to the tree with a parent
    void addVertex(const std::vector<unsigned int>& indices, CompositeVertex* parent);

    // Check if a vertex is in the tree
    bool contains(const std::vector<unsigned int>& indices) const;

    // Get a vertex from the tree (returns nullptr if not found)
    CompositeVertex* getVertex(const std::vector<unsigned int>& indices) const;

    // Get the root vertex
    CompositeVertex* getRoot() const { return root_.get(); }

    // Get all vertices in the tree
    const std::vector<std::unique_ptr<CompositeVertex>>& getVertices() const {
        return vertices_;
    }

    // Get size of tree
    size_t size() const { return vertex_map_.size(); }

private:
    std::unique_ptr<CompositeVertex> root_;
    std::vector<std::unique_ptr<CompositeVertex>> vertices_;

    // Hash table for fast lookup
    struct VectorHash {
        std::size_t operator()(const std::vector<unsigned int>& v) const {
            std::size_t seed = v.size();
            for (auto& i : v) {
                seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    std::unordered_map<std::vector<unsigned int>, CompositeVertex*, VectorHash> vertex_map_;
};

// Main discrete RRT planner
class DiscreteRRTPlanner {
public:
    DiscreteRRTPlanner(
        const std::vector<std::shared_ptr<IndividualRoadmap>>& roadmaps,
        const std::vector<std::shared_ptr<ob::SpaceInformation>>& space_info_list,
        const std::vector<std::shared_ptr<Robot>>& robots = {});

    ~DiscreteRRTPlanner() = default;

    // Set the start and goal configurations (as vertex indices)
    void setStartGoal(
        const std::vector<unsigned int>& start_indices,
        const std::vector<unsigned int>& goal_indices);

    // Set algorithm parameters
    void setExpansionsPerIteration(unsigned int n_it) { n_it_ = n_it; }
    void setSeed(int seed);

    // Run the planner
    bool solve(double max_time);

    // Get the solution path (if found)
    std::vector<std::vector<unsigned int>> getSolutionPath() const;

    // Get the solution as actual states
    std::vector<std::vector<const ob::State*>> getSolutionStates() const;

private:
    // Core algorithm functions
    void expandTree();

    // Sample a random continuous configuration
    std::vector<ob::ScopedState<>> sampleRandomConfiguration();

    // Find the nearest vertex in the tree to a continuous configuration
    CompositeVertex* nearestNeighborInTree(const std::vector<ob::ScopedState<>>& config);

    // Neighbor oracle: find a neighbor of v_near in direction of q_rand
    std::vector<unsigned int> neighborOracle(
        const CompositeVertex* v_near,
        const std::vector<ob::ScopedState<>>& q_rand);

    // Check if a composite vertex is collision-free
    bool isCollisionFree(const std::vector<unsigned int>& indices) const;

    // Check if the motion between two composite vertices is collision-free
    bool isCompositeMotionValid(
        const std::vector<unsigned int>& from_indices,
        const std::vector<unsigned int>& to_indices) const;

    // Check robot-robot collisions for a set of robot states
    bool isRobotRobotCollisionFree(const std::vector<const ob::State*>& states) const;

    // Check robot-robot collisions along the moving robot's edge
    bool isRobotRobotMotionValid(
        const std::vector<const ob::State*>& from_states,
        const std::vector<const ob::State*>& to_states) const;

    // Distance between two states
    double distance(const ob::State* s1, const ob::State* s2, size_t robot_idx) const;

    // Distance between composite configurations
    double compositeDistance(
        const std::vector<unsigned int>& indices,
        const std::vector<ob::ScopedState<>>& continuous_config) const;

    // Extract path from tree
    std::vector<std::vector<unsigned int>> extractPath(CompositeVertex* goal_vertex) const;

    // Member variables
    std::vector<std::shared_ptr<IndividualRoadmap>> roadmaps_;
    std::vector<std::shared_ptr<ob::SpaceInformation>> space_info_list_;
    std::vector<std::shared_ptr<Robot>> robots_;

    unsigned int num_robots_;

    std::vector<unsigned int> start_indices_;
    std::vector<unsigned int> goal_indices_;

    std::unique_ptr<DiscreteRRTTree> tree_;

    // Algorithm parameters
    unsigned int n_it_ = 5;  // Expansions per iteration

    // Random number generation
    std::mt19937 rng_;

    // Fallback collision radius (used when robots_ is not provided)
    double collision_radius_ = 0.5;

    // Statistics
    unsigned int num_expansions_ = 0;
    bool solution_found_ = false;
};

//=============================================================================
// CompositeVertex Implementation
//=============================================================================

std::size_t CompositeVertex::Hash::operator()(const CompositeVertex& v) const {
    std::size_t seed = v.vertex_indices.size();
    for (auto& i : v.vertex_indices) {
        seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

bool CompositeVertex::isNeighbor(const CompositeVertex& other) const {
    if (vertex_indices.size() != other.vertex_indices.size()) {
        return false;
    }

    for (size_t i = 0; i < vertex_indices.size(); ++i) {
        if (vertex_indices[i] == other.vertex_indices[i]) {
            return false;
        }
    }
    return true;
}

int CompositeVertex::getDifferingCoordinate(const CompositeVertex& other) const {
    if (vertex_indices.size() != other.vertex_indices.size()) {
        return -1;
    }

    int diff_coord = -1;
    int diff_count = 0;
    for (size_t i = 0; i < vertex_indices.size(); ++i) {
        if (vertex_indices[i] != other.vertex_indices[i]) {
            diff_coord = i;
            diff_count++;
            if (diff_count > 1) return -1;
        }
    }
    return diff_coord;
}

//=============================================================================
// IndividualRoadmap Implementation
//=============================================================================

IndividualRoadmap::IndividualRoadmap(
    const std::shared_ptr<ob::SpaceInformation>& si,
    unsigned int num_samples,
    double connection_radius)
    : IndividualRoadmap(si, num_samples, connection_radius, SparsParams())
{
}

IndividualRoadmap::IndividualRoadmap(
    const std::shared_ptr<ob::SpaceInformation>& si,
    unsigned int num_samples,
    double connection_radius,
    const SparsParams& spars_params)
    : si_(si)
    , ss_(si)
    , num_samples_(num_samples)
    , connection_radius_(connection_radius)
{
    spars_ = std::make_shared<og::SPARStwo>(si_);
    if (spars_params.has_max_failures) {
        spars_->setMaxFailures(spars_params.max_failures);
    }
    if (spars_params.has_sparse_delta_fraction) {
        spars_->setSparseDeltaFraction(spars_params.sparse_delta_fraction);
    }
    if (spars_params.has_dense_delta_fraction) {
        spars_->setDenseDeltaFraction(spars_params.dense_delta_fraction);
    }
    if (spars_params.has_stretch_factor) {
        spars_->setStretchFactor(spars_params.stretch_factor);
    }
    ss_.setPlanner(spars_);
}

void IndividualRoadmap::build(const ob::State* start, const ob::State* goal, double max_time) {
    vertex_cache_valid_ = false;
    neighbor_cache_.clear();

    // Use the actual problem start and goal states
    ob::ScopedState<> start_state(si_);
    ob::ScopedState<> goal_state(si_);
    si_->copyState(start_state.get(), start);
    si_->copyState(goal_state.get(), goal);

    ss_.setStartAndGoalStates(start_state, goal_state);

    // Phase 1: Find initial solution (use up to 50% of time budget)
    auto time_limit = ob::timedPlannerTerminationCondition(max_time);
    ob::PlannerStatus status = ss_.solve(time_limit);

    // Check if we found a solution
    if (!ss_.haveSolutionPath()) {
        std::cout << "  Error: No solution found between start and goal!" << std::endl;
        std::cout << "  Built roadmap with " << getNumVertices() << " vertices" << std::endl;

        // If no solution, we can't solve the problem. Throw an error.
        throw std::runtime_error("No solution found between start and goal");
    }

    std::cout << "  Built roadmap with " << getNumVertices() << " vertices" << std::endl;
}

unsigned int IndividualRoadmap::getNumVertices() const {
    ensureVertexCache();
    return static_cast<unsigned int>(vertex_cache_.size());
}

const ob::State* IndividualRoadmap::getState(unsigned int index) const {
    ensureVertexCache();
    if (index >= vertex_cache_.size()) {
        throw std::runtime_error("Roadmap state index out of range");
    }

    const og::SPARStwo::Graph& graph = spars_->getRoadmap();
    auto state_property = boost::get(og::SPARStwo::vertex_state_t(), graph);
    return state_property[vertex_cache_[index]];
}

std::vector<unsigned int> IndividualRoadmap::getNeighbors(unsigned int index) const {
    ensureVertexCache();
    if (index >= vertex_cache_.size()) {
        throw std::runtime_error("Roadmap vertex index out of range");
    }

    // Check cache first
    auto it = neighbor_cache_.find(index);
    if (it != neighbor_cache_.end()) {
        return it->second;
    }

    // Build neighbor list
    std::vector<unsigned int> neighbors;
    const og::SPARStwo::Graph& graph = spars_->getRoadmap();

    auto edge_range = boost::out_edges(vertex_cache_[index], graph);
    for (auto ei = edge_range.first; ei != edge_range.second; ++ei) {
        auto neighbor_vertex = boost::target(*ei, graph);
        auto map_it = vertex_index_map_.find(neighbor_vertex);
        if (map_it != vertex_index_map_.end()) {
            neighbors.push_back(map_it->second);
        }
    }

    // Cache it
    neighbor_cache_[index] = neighbors;
    return neighbors;
}

unsigned int IndividualRoadmap::getNearestVertex(const ob::State* state) const {
    ensureVertexCache();
    unsigned int num_vertices = static_cast<unsigned int>(vertex_cache_.size());

    if (num_vertices == 0) {
        throw std::runtime_error("Cannot find nearest vertex in empty roadmap");
    }

    const og::SPARStwo::Graph& graph = spars_->getRoadmap();
    auto state_property = boost::get(og::SPARStwo::vertex_state_t(), graph);

    double min_dist = std::numeric_limits<double>::infinity();
    unsigned int nearest_idx = 0;

    for (unsigned int i = 0; i < num_vertices; ++i) {
        double dist = si_->distance(state, state_property[vertex_cache_[i]]);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_idx = i;
        }
    }

    return nearest_idx;
}

unsigned int IndividualRoadmap::getNeighborToward(
    unsigned int vertex_idx,
    const ob::State* target_state) const
{
    std::vector<unsigned int> neighbors = getNeighbors(vertex_idx);

    if (neighbors.empty()) {
        return vertex_idx;  // No neighbors, return same vertex
    }

    double min_dist = std::numeric_limits<double>::infinity();
    unsigned int best_neighbor = vertex_idx;

    for (unsigned int neighbor_idx : neighbors) {
        const ob::State* neighbor_state = getState(neighbor_idx);
        double dist = si_->distance(neighbor_state, target_state);
        if (dist < min_dist) {
            min_dist = dist;
            best_neighbor = neighbor_idx;
        }
    }

    return best_neighbor;
}

bool IndividualRoadmap::hasEdge(unsigned int v1, unsigned int v2) const {
    ensureVertexCache();
    if (v1 >= vertex_cache_.size() || v2 >= vertex_cache_.size()) {
        return false;
    }

    const og::SPARStwo::Graph& graph = spars_->getRoadmap();
    return boost::edge(vertex_cache_[v1], vertex_cache_[v2], graph).second;
}

void IndividualRoadmap::ensureVertexCache() const {
    if (vertex_cache_valid_) {
        return;
    }

    vertex_cache_.clear();
    vertex_index_map_.clear();

    const og::SPARStwo::Graph& graph = spars_->getRoadmap();
    auto state_property = boost::get(og::SPARStwo::vertex_state_t(), graph);

    auto vertex_range = boost::vertices(graph);
    for (auto it = vertex_range.first; it != vertex_range.second; ++it) {
        auto v = *it;
        if (state_property[v] == nullptr) {
            continue;
        }
        unsigned int index = static_cast<unsigned int>(vertex_cache_.size());
        vertex_cache_.push_back(v);
        vertex_index_map_[v] = index;
    }

    vertex_cache_valid_ = true;
}

//=============================================================================
// DiscreteRRTTree Implementation
//=============================================================================

DiscreteRRTTree::DiscreteRRTTree(const std::vector<unsigned int>& root_indices) {
    root_ = std::make_unique<CompositeVertex>(root_indices);
    vertex_map_[root_indices] = root_.get();
}

void DiscreteRRTTree::addVertex(
    const std::vector<unsigned int>& indices,
    CompositeVertex* parent)
{
    auto new_vertex = std::make_unique<CompositeVertex>(indices);
    new_vertex->parent = parent;

    vertex_map_[indices] = new_vertex.get();
    vertices_.push_back(std::move(new_vertex));
}

bool DiscreteRRTTree::contains(const std::vector<unsigned int>& indices) const {
    return vertex_map_.find(indices) != vertex_map_.end();
}

CompositeVertex* DiscreteRRTTree::getVertex(const std::vector<unsigned int>& indices) const {
    auto it = vertex_map_.find(indices);
    if (it != vertex_map_.end()) {
        return it->second;
    }
    return nullptr;
}

//=============================================================================
// DiscreteRRTPlanner Implementation
//=============================================================================

DiscreteRRTPlanner::DiscreteRRTPlanner(
    const std::vector<std::shared_ptr<IndividualRoadmap>>& roadmaps,
    const std::vector<std::shared_ptr<ob::SpaceInformation>>& space_info_list,
    const std::vector<std::shared_ptr<Robot>>& robots)
    : roadmaps_(roadmaps)
    , space_info_list_(space_info_list)
    , robots_(robots)
    , num_robots_(roadmaps.size())
    , rng_(std::random_device{}())
{
    if (roadmaps_.size() != space_info_list_.size()) {
        throw std::runtime_error("Number of roadmaps must match number of space infos");
    }
    if (!robots_.empty() && robots_.size() != roadmaps_.size()) {
        throw std::runtime_error("Number of robots must match number of roadmaps");
    }
}

void DiscreteRRTPlanner::setSeed(int seed)
{
    if (seed >= 0) {
        rng_ = std::mt19937(seed);
    }
}

void DiscreteRRTPlanner::setStartGoal(
    const std::vector<unsigned int>& start_indices,
    const std::vector<unsigned int>& goal_indices)
{
    if (start_indices.size() != num_robots_ || goal_indices.size() != num_robots_) {
        throw std::runtime_error("Start/goal must have one index per robot");
    }

    start_indices_ = start_indices;
    goal_indices_ = goal_indices;

    // Initialize tree with start configuration
    tree_ = std::make_unique<DiscreteRRTTree>(start_indices_);
}

bool DiscreteRRTPlanner::solve(double max_time) {
    auto start_time = std::chrono::high_resolution_clock::now();
    solution_found_ = false;
    num_expansions_ = 0;

    std::cout << "Starting discrete-RRT search..." << std::endl;
    std::cout << "  Start: [";
    for (size_t i = 0; i < start_indices_.size(); ++i) {
        std::cout << start_indices_[i];
        if (i < start_indices_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Goal:  [";
    for (size_t i = 0; i < goal_indices_.size(); ++i) {
        std::cout << goal_indices_[i];
        if (i < goal_indices_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    unsigned int iteration = 0;
    while (true) {
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - start_time).count();

        if (elapsed > max_time) {
            std::cout << "Time limit reached" << std::endl;
            break;
        }

        // Expansion phase
        for (unsigned int i = 0; i < n_it_; ++i) {
            expandTree();
        }

        // Check if goal is in tree
        if (tree_->contains(goal_indices_)) {
            solution_found_ = true;
            std::cout << "Solution found!" << std::endl;
            std::cout << "  Tree size: " << tree_->size() << " vertices" << std::endl;
            std::cout << "  Expansions: " << num_expansions_ << std::endl;
            std::cout << "  Time: " << elapsed << " seconds" << std::endl;
            return true;
        }

        iteration++;

        // Progress update every 100 iterations
        if (iteration % 100 == 0) {
            std::cout << "  Iteration " << iteration
                      << ", tree size: " << tree_->size()
                      << ", time: " << elapsed << "s" << std::endl;
        }
    }

    std::cout << "No solution found" << std::endl;
    std::cout << "  Final tree size: " << tree_->size() << " vertices" << std::endl;
    std::cout << "  Total expansions: " << num_expansions_ << std::endl;
    return false;
}

void DiscreteRRTPlanner::expandTree() {
    num_expansions_++;

    // Sample random continuous configuration
    auto q_rand = sampleRandomConfiguration();

    // Find nearest vertex in tree
    CompositeVertex* v_near = nearestNeighborInTree(q_rand);

    // Use neighbor oracle to find new vertex
    std::vector<unsigned int> v_new_indices = neighborOracle(v_near, q_rand);

    // Check if it's new and valid
    if (v_new_indices != v_near->vertex_indices &&
        !tree_->contains(v_new_indices) &&
        isCompositeMotionValid(v_near->vertex_indices, v_new_indices))
    {
        tree_->addVertex(v_new_indices, v_near);
    }
}

std::vector<ob::ScopedState<>> DiscreteRRTPlanner::sampleRandomConfiguration() {
    std::vector<ob::ScopedState<>> config;
    config.reserve(num_robots_);

    for (size_t i = 0; i < num_robots_; ++i) {
        ob::ScopedState<> state(space_info_list_[i]);
        auto sampler = space_info_list_[i]->allocStateSampler();
        sampler->sampleUniform(state.get());
        config.push_back(state);
    }

    return config;
}

CompositeVertex* DiscreteRRTPlanner::nearestNeighborInTree(
    const std::vector<ob::ScopedState<>>& config)
{
    double min_dist = std::numeric_limits<double>::infinity();
    CompositeVertex* nearest = tree_->getRoot();

    // Check root
    double dist = compositeDistance(tree_->getRoot()->vertex_indices, config);
    if (dist < min_dist) {
        min_dist = dist;
        nearest = tree_->getRoot();
    }

    // Check all other vertices
    for (const auto& vertex_ptr : tree_->getVertices()) {
        dist = compositeDistance(vertex_ptr->vertex_indices, config);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = vertex_ptr.get();
        }
    }

    return nearest;
}

std::vector<unsigned int> DiscreteRRTPlanner::neighborOracle(
    const CompositeVertex* v_near,
    const std::vector<ob::ScopedState<>>& q_rand)
{
    // Move all robots simultaneously to neighbors (tensor product)
    std::vector<unsigned int> v_new_indices = v_near->vertex_indices;

    for (size_t i = 0; i < num_robots_; ++i) {
        unsigned int current_vertex = v_near->vertex_indices[i];
        std::vector<unsigned int> neighbors = roadmaps_[i]->getNeighbors(current_vertex);

        if (neighbors.empty()) {
            return v_near->vertex_indices;
        }

        unsigned int new_vertex = roadmaps_[i]->getNeighborToward(
            current_vertex, q_rand[i].get());

        if (new_vertex == current_vertex) {
            std::uniform_int_distribution<size_t> pick(0, neighbors.size() - 1);
            new_vertex = neighbors[pick(rng_)];
        }

        v_new_indices[i] = new_vertex;
    }

    return v_new_indices;
}

bool DiscreteRRTPlanner::isCollisionFree(const std::vector<unsigned int>& indices) const {
    std::vector<const ob::State*> states;
    states.reserve(num_robots_);

    for (size_t i = 0; i < num_robots_; ++i) {
        const ob::State* state = roadmaps_[i]->getState(indices[i]);
        if (!space_info_list_[i]->isValid(state)) {
            return false;
        }
        states.push_back(state);
    }

    return isRobotRobotCollisionFree(states);
}

bool DiscreteRRTPlanner::isCompositeMotionValid(
    const std::vector<unsigned int>& from_indices,
    const std::vector<unsigned int>& to_indices) const
{
    if (from_indices.size() != num_robots_ || to_indices.size() != num_robots_) {
        return false;
    }

    if (from_indices == to_indices) {
        return false;
    }

    if (!isCollisionFree(to_indices)) {
        return false;
    }

    std::vector<const ob::State*> from_states;
    std::vector<const ob::State*> to_states;
    from_states.reserve(num_robots_);
    to_states.reserve(num_robots_);

    for (size_t i = 0; i < num_robots_; ++i) {
        if (from_indices[i] == to_indices[i]) {
            return false;
        }
        if (!roadmaps_[i]->hasEdge(from_indices[i], to_indices[i])) {
            return false;
        }

        const ob::State* from_state = roadmaps_[i]->getState(from_indices[i]);
        const ob::State* to_state = roadmaps_[i]->getState(to_indices[i]);
        if (!space_info_list_[i]->checkMotion(from_state, to_state)) {
            return false;
        }

        from_states.push_back(from_state);
        to_states.push_back(to_state);
    }

    return isRobotRobotMotionValid(from_states, to_states);
}

bool DiscreteRRTPlanner::isRobotRobotCollisionFree(
    const std::vector<const ob::State*>& states) const
{
    bool use_fcl = robots_.size() == num_robots_;

    for (size_t i = 0; i < num_robots_; ++i) {
        for (size_t j = i + 1; j < num_robots_; ++j) {
            bool collision = false;

            if (use_fcl) {
                const auto& robot_i = robots_[i];
                const auto& robot_j = robots_[j];

                for (size_t part_i = 0; part_i < robot_i->numParts() && !collision; ++part_i) {
                    const auto& transform_i = robot_i->getTransform(states[i], part_i);
                    fcl::CollisionObjectf co_i(robot_i->getCollisionGeometry(part_i));
                    co_i.setTranslation(transform_i.translation());
                    co_i.setRotation(transform_i.rotation());
                    co_i.computeAABB();

                    for (size_t part_j = 0; part_j < robot_j->numParts(); ++part_j) {
                        const auto& transform_j = robot_j->getTransform(states[j], part_j);
                        fcl::CollisionObjectf co_j(robot_j->getCollisionGeometry(part_j));
                        co_j.setTranslation(transform_j.translation());
                        co_j.setRotation(transform_j.rotation());
                        co_j.computeAABB();

                        fcl::CollisionRequestf request;
                        request.num_max_contacts = 1;
                        fcl::CollisionResultf result;
                        fcl::collide(&co_i, &co_j, request, result);

                        if (result.isCollision()) {
                            collision = true;
                            break;
                        }
                    }
                }
            } else {
                auto* se2_state_i = states[i]->as<ob::SE2StateSpace::StateType>();
                auto* se2_state_j = states[j]->as<ob::SE2StateSpace::StateType>();

                double dx = se2_state_i->getX() - se2_state_j->getX();
                double dy = se2_state_i->getY() - se2_state_j->getY();
                double dist = std::sqrt(dx * dx + dy * dy);
                collision = (dist < 2.0 * collision_radius_);
            }

            if (collision) {
                return false;
            }
        }
    }

    return true;
}

bool DiscreteRRTPlanner::isRobotRobotMotionValid(
    const std::vector<const ob::State*>& from_states,
    const std::vector<const ob::State*>& to_states) const
{
    if (from_states.size() != num_robots_ || to_states.size() != num_robots_) {
        return false;
    }

    int steps = 1;
    for (size_t i = 0; i < num_robots_; ++i) {
        auto si = space_info_list_[i];
        double dist = si->distance(from_states[i], to_states[i]);
        double max_extent = si->getStateSpace()->getMaximumExtent();
        double resolution = si->getStateValidityCheckingResolution();
        double step = max_extent * resolution;
        if (step <= 0.0) {
            step = dist;
        }

        int robot_steps = 1;
        if (step > 0.0 && dist > 0.0) {
            robot_steps = static_cast<int>(std::ceil(dist / step));
        }
        if (robot_steps < 1) {
            robot_steps = 1;
        }
        steps = std::max(steps, robot_steps);
    }

    std::vector<ob::State*> temp_states(num_robots_, nullptr);
    for (size_t i = 0; i < num_robots_; ++i) {
        temp_states[i] = space_info_list_[i]->allocState();
    }

    std::vector<const ob::State*> states(num_robots_, nullptr);
    for (int i = 1; i <= steps; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(steps);
        for (size_t r = 0; r < num_robots_; ++r) {
            space_info_list_[r]->getStateSpace()->interpolate(
                from_states[r], to_states[r], t, temp_states[r]);
            states[r] = temp_states[r];
        }

        if (!isRobotRobotCollisionFree(states)) {
            for (size_t r = 0; r < num_robots_; ++r) {
                space_info_list_[r]->freeState(temp_states[r]);
            }
            return false;
        }
    }

    for (size_t r = 0; r < num_robots_; ++r) {
        space_info_list_[r]->freeState(temp_states[r]);
    }
    return true;
}

double DiscreteRRTPlanner::distance(
    const ob::State* s1,
    const ob::State* s2,
    size_t robot_idx) const
{
    return space_info_list_[robot_idx]->distance(s1, s2);
}

double DiscreteRRTPlanner::compositeDistance(
    const std::vector<unsigned int>& indices,
    const std::vector<ob::ScopedState<>>& continuous_config) const
{
    double total_dist = 0.0;

    for (size_t i = 0; i < num_robots_; ++i) {
        const ob::State* roadmap_state = roadmaps_[i]->getState(indices[i]);
        double d = distance(roadmap_state, continuous_config[i].get(), i);
        total_dist += d * d;  // Sum of squared distances
    }

    return std::sqrt(total_dist);
}

std::vector<std::vector<unsigned int>> DiscreteRRTPlanner::getSolutionPath() const {
    if (!solution_found_) {
        return {};
    }

    CompositeVertex* goal_vertex = tree_->getVertex(goal_indices_);
    if (!goal_vertex) {
        return {};
    }

    return extractPath(goal_vertex);
}

std::vector<std::vector<const ob::State*>> DiscreteRRTPlanner::getSolutionStates() const {
    std::vector<std::vector<unsigned int>> path = getSolutionPath();
    if (path.empty()) {
        return {};
    }

    std::vector<std::vector<const ob::State*>> state_path;
    state_path.reserve(path.size());

    for (const auto& indices : path) {
        std::vector<const ob::State*> states;
        states.reserve(num_robots_);

        for (size_t i = 0; i < num_robots_; ++i) {
            states.push_back(roadmaps_[i]->getState(indices[i]));
        }

        state_path.push_back(states);
    }

    return state_path;
}

std::vector<std::vector<unsigned int>> DiscreteRRTPlanner::extractPath(
    CompositeVertex* goal_vertex) const
{
    std::vector<std::vector<unsigned int>> path;

    CompositeVertex* current = goal_vertex;
    while (current != nullptr) {
        path.push_back(current->vertex_indices);
        current = current->parent;
    }

    std::reverse(path.begin(), path.end());
    return path;
}

//=============================================================================
// Helper Functions for Main
//=============================================================================

// Simple state validity checker for individual robots (geometric planning)
class GeometricStateValidityChecker : public ob::StateValidityChecker {
public:
    GeometricStateValidityChecker(
        const ob::SpaceInformationPtr& si,
        const std::shared_ptr<fcl::BroadPhaseCollisionManagerf>& env_manager,
        const std::shared_ptr<Robot>& robot)
        : ob::StateValidityChecker(si)
        , env_manager_(env_manager)
        , robot_(robot)
    {
    }

    bool isValid(const ob::State* state) const override {
        // Check bounds
        if (!si_->satisfiesBounds(state)) {
            return false;
        }

        // Check collision with environment
        for (size_t part = 0; part < robot_->numParts(); ++part) {
            fcl::Transform3f transform = robot_->getTransform(state, part);
            auto geom = robot_->getCollisionGeometry(part);

            fcl::CollisionObjectf obj(geom);
            obj.setTranslation(transform.translation());
            obj.setRotation(transform.rotation());
            obj.computeAABB();
            fcl::DefaultCollisionData<float> collision_data;
            collision_data.request.num_max_contacts = 1;

            env_manager_->collide(&obj, &collision_data, fcl::DefaultCollisionFunction<float>);

            if (collision_data.result.isCollision()) {
                return false;
            }
        }

        return true;
    }

private:
    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> env_manager_;
    std::shared_ptr<Robot> robot_;
};

// Load obstacles from YAML and create FCL manager
std::shared_ptr<fcl::BroadPhaseCollisionManagerf> createEnvironmentManager(
    const YAML::Node& config,
    std::vector<fcl::CollisionObjectf*>& obstacle_objects)
{
    auto manager = std::make_shared<fcl::DynamicAABBTreeCollisionManagerf>();

    if (config["environment"]["obstacles"]) {
        for (const auto& obs : config["environment"]["obstacles"]) {
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
                manager->registerObject(co);
            }
        }
    }

    manager->setup();
    return manager;
}

//=============================================================================
// Main Function
//=============================================================================

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string input_file;
    std::string output_file;
    std::string config_file;
    double time_limit = 60.0;
    double roadmap_time = 10.0;
    unsigned int expansions_per_iter = 5;
    int seed = -1;
    IndividualRoadmap::SparsParams spars_params;

    po::options_description desc("Discrete-RRT Multi-Robot Motion Planning");
    desc.add_options()
        ("help,h", "Show help message")
        ("input,i", po::value<std::string>(&input_file)->required(), "Input YAML file")
        ("output,o", po::value<std::string>(&output_file)->required(), "Output YAML file")
        ("cfg,c", po::value<std::string>(&config_file), "Configuration YAML file");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    // Load configuration file if provided
    if (vm.count("cfg")) {
        try {
            YAML::Node cfg = YAML::LoadFile(config_file);
            if (cfg["roadmap_time"]) {
                roadmap_time = cfg["roadmap_time"].as<double>();
            }
            if (cfg["expansions_per_iter"]) {
                expansions_per_iter = cfg["expansions_per_iter"].as<unsigned int>();
            }
            if (cfg["spars_max_failures"]) {
                spars_params.has_max_failures = true;
                spars_params.max_failures = cfg["spars_max_failures"].as<unsigned int>();
            }
            if (cfg["spars_sparse_delta_fraction"]) {
                spars_params.has_sparse_delta_fraction = true;
                spars_params.sparse_delta_fraction = cfg["spars_sparse_delta_fraction"].as<double>();
            }
            if (cfg["spars_dense_delta_fraction"]) {
                spars_params.has_dense_delta_fraction = true;
                spars_params.dense_delta_fraction = cfg["spars_dense_delta_fraction"].as<double>();
            }
            if (cfg["spars_stretch_factor"]) {
                spars_params.has_stretch_factor = true;
                spars_params.stretch_factor = cfg["spars_stretch_factor"].as<double>();
            }
            if (cfg["time_limit"]) {
                time_limit = cfg["time_limit"].as<double>();
            }
            if (cfg["seed"]) {
                seed = cfg["seed"].as<int>();
            }
        } catch (const YAML::Exception& e) {
            std::cerr << "ERROR loading config file: " << e.what() << std::endl;
            return 1;
        }
    }

    // Set the random seed
    if (seed >= 0) {
        std::cout << "Setting random seed to: " << seed << std::endl;
        ompl::RNG::setSeed(seed);
    } else {
        std::cout << "Using random seed" << std::endl;
    }

    std::cout << "Discrete-RRT Multi-Robot Motion Planning" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Input: " << input_file << std::endl;
    std::cout << "Output: " << output_file << std::endl;
    if (vm.count("cfg")) {
        std::cout << "Config: " << config_file << std::endl;
    }
    std::cout << "Time limit: " << time_limit << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "Algorithm Parameters:" << std::endl;
    std::cout << "  Roadmap build time: " << roadmap_time << "s per robot" << std::endl;
    std::cout << "  Expansions per iteration: " << expansions_per_iter << std::endl;
    if (spars_params.has_max_failures) {
        std::cout << "  SPARStwo max failures: " << spars_params.max_failures << std::endl;
    }
    if (spars_params.has_sparse_delta_fraction) {
        std::cout << "  SPARStwo sparse delta fraction: " << spars_params.sparse_delta_fraction << std::endl;
    }
    if (spars_params.has_dense_delta_fraction) {
        std::cout << "  SPARStwo dense delta fraction: " << spars_params.dense_delta_fraction << std::endl;
    }
    if (spars_params.has_stretch_factor) {
        std::cout << "  SPARStwo stretch factor: " << spars_params.stretch_factor << std::endl;
    }
    std::cout << std::endl;

    // Load YAML configuration
    YAML::Node config = YAML::LoadFile(input_file);

    // Create environment collision manager
    std::vector<fcl::CollisionObjectf*> obstacle_objects;
    auto env_manager = createEnvironmentManager(config, obstacle_objects);
    std::cout << "Loaded " << obstacle_objects.size() << " obstacles" << std::endl;

    // Get environment bounds
    ob::RealVectorBounds bounds(2);
    bounds.setLow(0, config["environment"]["min"][0].as<double>());
    bounds.setLow(1, config["environment"]["min"][1].as<double>());
    bounds.setHigh(0, config["environment"]["max"][0].as<double>());
    bounds.setHigh(1, config["environment"]["max"][1].as<double>());

    // Create robots
    std::vector<std::shared_ptr<Robot>> robots;
    std::vector<ob::ScopedState<ob::SE2StateSpace>> start_states;
    std::vector<ob::ScopedState<ob::SE2StateSpace>> goal_states;

    std::cout << "Creating robots..." << std::endl;
    for (const auto& robot_config : config["robots"]) {
        std::string type = robot_config["type"].as<std::string>();
        auto robot = create_robot(type, bounds);
        robots.push_back(robot);

        // Get start and goal
        auto si = robot->getSpaceInformation();
        ob::ScopedState<ob::SE2StateSpace> start(si->getStateSpace());
        ob::ScopedState<ob::SE2StateSpace> goal(si->getStateSpace());

        start->setX(robot_config["start"][0].as<double>());
        start->setY(robot_config["start"][1].as<double>());
        start->setYaw(robot_config["start"][2].as<double>());

        goal->setX(robot_config["goal"][0].as<double>());
        goal->setY(robot_config["goal"][1].as<double>());
        goal->setYaw(robot_config["goal"][2].as<double>());

        start_states.push_back(start);
        goal_states.push_back(goal);

        std::cout << "  Robot " << robots.size() - 1 << " (" << type << ")" << std::endl;
        std::cout << "    Start: (" << start->getX() << ", " << start->getY() << ", "
                  << start->getYaw() << ")" << std::endl;
        std::cout << "    Goal:  (" << goal->getX() << ", " << goal->getY() << ", "
                  << goal->getYaw() << ")" << std::endl;
    }

    size_t num_robots = robots.size();
    std::cout << "Total robots: " << num_robots << std::endl;
    std::cout << std::endl;

    // Build individual roadmaps for each robot
    std::cout << "Building individual roadmaps (geometric SPARStwo)..." << std::endl;
    std::vector<std::shared_ptr<IndividualRoadmap>> roadmaps;
    std::vector<std::shared_ptr<ob::SpaceInformation>> space_infos;
    std::vector<unsigned int> start_indices;
    std::vector<unsigned int> goal_indices;

    auto roadmap_start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_robots; ++i) {
        std::cout << "Robot " << i << ":" << std::endl;

        // Create geometric space info (note: using geometric, not control)
        auto state_space = std::make_shared<ob::SE2StateSpace>();
        state_space->setBounds(bounds);

        auto si = std::make_shared<ob::SpaceInformation>(state_space);

        // Set validity checker
        auto validity_checker = std::make_shared<GeometricStateValidityChecker>(
            si, env_manager, robots[i]);
        si->setStateValidityChecker(validity_checker);
        si->setup();

        space_infos.push_back(si);

        // Build roadmap with actual start and goal states
        auto roadmap = std::make_shared<IndividualRoadmap>(si, 0, 1.0, spars_params);
        roadmap->build(start_states[i].get(), goal_states[i].get(), roadmap_time);
        roadmaps.push_back(roadmap);

        // Find start and goal vertex indices
        unsigned int start_idx = roadmap->getNearestVertex(start_states[i].get());
        unsigned int goal_idx = roadmap->getNearestVertex(goal_states[i].get());

        start_indices.push_back(start_idx);
        goal_indices.push_back(goal_idx);

        const ob::State* roadmap_start = roadmap->getState(start_idx);
        const ob::State* roadmap_goal = roadmap->getState(goal_idx);
        auto* rs = roadmap_start->as<ob::SE2StateSpace::StateType>();
        auto* rg = roadmap_goal->as<ob::SE2StateSpace::StateType>();

        std::cout << "  Start vertex " << start_idx << ": ("
                  << rs->getX() << ", " << rs->getY() << ", " << rs->getYaw() << ")" << std::endl;
        std::cout << "  Goal vertex " << goal_idx << ": ("
                  << rg->getX() << ", " << rg->getY() << ", " << rg->getYaw() << ")" << std::endl;
    }

    auto roadmap_end_time = std::chrono::high_resolution_clock::now();
    double roadmap_build_time = std::chrono::duration<double>(
        roadmap_end_time - roadmap_start_time).count();
    std::cout << "Roadmap construction completed in " << roadmap_build_time << "s" << std::endl;
    std::cout << std::endl;

    // Create discrete-RRT planner
    std::cout << "Initializing discrete-RRT planner..." << std::endl;
    DiscreteRRTPlanner planner(roadmaps, space_infos, robots);
    planner.setSeed(seed);
    planner.setStartGoal(start_indices, goal_indices);
    planner.setExpansionsPerIteration(expansions_per_iter);
    std::cout << std::endl;

    // Solve
    auto planning_start = std::chrono::high_resolution_clock::now();
    bool solved = planner.solve(time_limit);
    auto planning_end = std::chrono::high_resolution_clock::now();
    double planning_time = std::chrono::duration<double>(planning_end - planning_start).count();

    // Write output
    YAML::Node output;
    output["solved"] = solved;
    output["planning_time"] = planning_time;
    output["roadmap_build_time"] = roadmap_build_time;
    output["total_time"] = planning_time + roadmap_build_time;

    if (solved) {
        auto solution_states = planner.getSolutionStates();

        std::cout << "\nSolution path length: " << solution_states.size() << " waypoints" << std::endl;

        for (size_t i = 0; i < num_robots; ++i) {
            YAML::Node robot_solution;
            std::vector<std::vector<double>> states_list;

            for (const auto& waypoint : solution_states) {
                const ob::State* state = waypoint[i];
                auto* se2_state = state->as<ob::SE2StateSpace::StateType>();

                std::vector<double> state_vec = {
                    se2_state->getX(),
                    se2_state->getY(),
                    se2_state->getYaw()
                };
                states_list.push_back(state_vec);
            }

            robot_solution["states"] = states_list;
            output["result"].push_back(robot_solution);
        }
    }

    std::ofstream fout(output_file);
    fout << output;
    fout.close();

    std::cout << "\nResults written to: " << output_file << std::endl;

    // Cleanup
    for (auto* co : obstacle_objects) {
        delete co;
    }

    return solved ? 0 : 1;
}
