#include "grid_decomposition.h"

#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <cmath>
#include <numeric>
#include <stdexcept>

// ─── RectGridDecompositionImpl ────────────────────────────────────────────────

RectGridDecompositionImpl::RectGridDecompositionImpl(
    const std::vector<int>& lengths,
    const ompl::base::RealVectorBounds& bounds,
    const ompl::base::StateSpacePtr& space)
    : dim_(static_cast<int>(lengths.size())),
      bounds_(bounds),
      lengths_(lengths),
      state_space_(space)
{
    num_regions_ = 1;
    cell_size_.resize(dim_);
    for (int i = 0; i < dim_; ++i) {
        num_regions_ *= lengths_[i];
        cell_size_[i] = (bounds_.high[i] - bounds_.low[i]) / lengths_[i];
    }
}

std::vector<int> RectGridDecompositionImpl::ridToCoord(int rid) const
{
    std::vector<int> coord(dim_);
    for (int i = 0; i < dim_; ++i) {
        coord[i] = rid % lengths_[i];
        rid /= lengths_[i];
    }
    return coord;
}

int RectGridDecompositionImpl::coordToRid(const std::vector<int>& coord) const
{
    for (int i = 0; i < dim_; ++i)
        if (coord[i] < 0 || coord[i] >= lengths_[i]) return -1;
    int rid = 0, stride = 1;
    for (int i = 0; i < dim_; ++i) {
        rid += coord[i] * stride;
        stride *= lengths_[i];
    }
    return rid;
}

ompl::base::RealVectorBounds RectGridDecompositionImpl::getCellBounds(int rid) const
{
    auto coord = ridToCoord(rid);
    ompl::base::RealVectorBounds b(dim_);
    for (int i = 0; i < dim_; ++i) {
        b.setLow(i,  bounds_.low[i] + coord[i]       * cell_size_[i]);
        b.setHigh(i, bounds_.low[i] + (coord[i] + 1) * cell_size_[i]);
    }
    return b;
}

double RectGridDecompositionImpl::getRegionVolume(int /*rid*/)
{
    double vol = 1.0;
    for (int i = 0; i < dim_; ++i) vol *= cell_size_[i];
    return vol;
}

void RectGridDecompositionImpl::project(const ompl::base::State* s,
                                         std::vector<double>& coord) const
{
    std::vector<double> reals;
    state_space_->copyToReals(reals, s);
    coord.resize(dim_);
    for (int i = 0; i < dim_; ++i) coord[i] = reals[i];
}

int RectGridDecompositionImpl::locateRegion(const ompl::base::State* s) const
{
    std::vector<double> coord;
    project(s, coord);
    std::vector<int> idx(dim_);
    for (int i = 0; i < dim_; ++i) {
        idx[i] = static_cast<int>((coord[i] - bounds_.low[i]) / cell_size_[i]);
        if (idx[i] < 0 || idx[i] >= lengths_[i]) return -1;
    }
    return coordToRid(idx);
}

void RectGridDecompositionImpl::getNeighbors(int rid, std::vector<int>& neighbors) const
{
    neighbors.clear();
    auto coord = ridToCoord(rid);
    for (int d = 0; d < dim_; ++d) {
        for (int delta : {-1, 1}) {
            auto nc = coord;
            nc[d] += delta;
            int nid = coordToRid(nc);
            if (nid >= 0) neighbors.push_back(nid);
        }
    }
}

void RectGridDecompositionImpl::getAllNeighbors(int rid, std::vector<int>& neighbors) const
{
    neighbors.clear();
    auto coord = ridToCoord(rid);
    // Iterate over all combinations of delta in {-1,0,+1} per dimension, skip self
    std::function<void(int, std::vector<int>&)> recurse = [&](int d, std::vector<int>& nc) {
        if (d == dim_) {
            if (nc == coord) return;
            int nid = coordToRid(nc);
            if (nid >= 0) neighbors.push_back(nid);
            return;
        }
        for (int delta : {-1, 0, 1}) {
            nc[d] = coord[d] + delta;
            recurse(d + 1, nc);
        }
    };
    std::vector<int> nc(dim_);
    recurse(0, nc);
}

void RectGridDecompositionImpl::sampleFromRegion(int rid, ompl::RNG& rng,
                                                  std::vector<double>& coord) const
{
    auto cb = getCellBounds(rid);
    coord.resize(dim_);
    for (int i = 0; i < dim_; ++i)
        coord[i] = rng.uniformReal(cb.low[i], cb.high[i]);
}

void RectGridDecompositionImpl::sampleFullState(const ompl::base::StateSamplerPtr& sampler,
                                                 const std::vector<double>& coord,
                                                 ompl::base::State* s) const
{
    sampler->sampleUniform(s);
    std::vector<double> reals;
    state_space_->copyToReals(reals, s);
    for (int i = 0; i < dim_; ++i) reals[i] = coord[i];
    state_space_->copyFromReals(s, reals);
}
#include <limits>

static int computeLength(int dim, const ompl::base::RealVectorBounds& bounds, double region_size)
{
    int length = 1;
    for (int i = 0; i < dim; ++i)
    {
        int cells = static_cast<int>(std::ceil((bounds.high[i] - bounds.low[i]) / region_size));
        length = std::max(length, cells);
    }
    return length;
}

GridDecompositionImpl::GridDecompositionImpl(const int length, const int dim, const ompl::base::RealVectorBounds& bounds)
        : ompl::control::GridDecomposition(length, dim, bounds),
          DecompositionImpl(),
          nextVirtualId_(0)
{
    nextVirtualId_ = ompl::control::GridDecomposition::getNumRegions();
}

GridDecompositionImpl::GridDecompositionImpl(const int dim, const ompl::base::RealVectorBounds& bounds, double region_size)
        : ompl::control::GridDecomposition(computeLength(dim, bounds, region_size), dim, bounds),
          DecompositionImpl(),

          nextVirtualId_(0)
{
    nextVirtualId_ = ompl::control::GridDecomposition::getNumRegions();
}

void GridDecompositionImpl::project(const ompl::base::State* s, std::vector<double>& coord) const
{
    if (state_space_)
    {
        // Generic path: first two reals are always x, y for supported state spaces
        std::vector<double> reals;
        state_space_->copyToReals(reals, s);
        coord.resize(dimension_);
        for (int i = 0; i < dimension_; ++i)
            coord[i] = reals[i];
        return;
    }

    // Legacy SE2 / SE3 path
    if (dimension_ == 2)
    {
        coord.resize(2);
        coord[0] = s->as<ompl::base::SE2StateSpace::StateType>()->getX();
        coord[1] = s->as<ompl::base::SE2StateSpace::StateType>()->getY();
    }
    else if (dimension_ == 3)
    {
        coord.resize(3);
        coord[0] = s->as<ompl::base::SE3StateSpace::StateType>()->getX();
        coord[1] = s->as<ompl::base::SE3StateSpace::StateType>()->getY();
        coord[2] = s->as<ompl::base::SE3StateSpace::StateType>()->getZ();
    }
}

void GridDecompositionImpl::sampleFullState(const ompl::base::StateSamplerPtr& sampler, const std::vector<double>& coord, ompl::base::State* s) const
{
    sampler->sampleUniform(s);
    if (state_space_)
    {
        // Generic path: override the first `dimension_` reals (x, y, ...) in the state
        std::vector<double> reals;
        state_space_->copyToReals(reals, s);
        for (int i = 0; i < dimension_; ++i)
            reals[i] = coord[i];
        state_space_->copyFromReals(s, reals);
        return;
    }

    // Legacy SE2 / SE3 path
    if (dimension_ == 2)
    {
        s->as<ompl::base::SE2StateSpace::StateType>()->setXY(coord[0], coord[1]);
    }
    else if (dimension_ == 3)
    {
        s->as<ompl::base::SE3StateSpace::StateType>()->setXYZ(coord[0], coord[1], coord[2]);
    }
}

// Returns true if two axis-aligned cells share a face (touch in exactly one
// dimension, strictly overlap in all others).
static bool spatiallyAdjacent(const ompl::base::RealVectorBounds& a,
                               const ompl::base::RealVectorBounds& b,
                               int dim)
{
    const double eps = 1e-9;
    int touching = -1;
    for (int i = 0; i < dim; ++i) {
        bool touch_ab = std::abs(a.high[i] - b.low[i]) < eps;
        bool touch_ba = std::abs(b.high[i] - a.low[i]) < eps;
        if (touch_ab || touch_ba) {
            if (touching >= 0) return false;  // touching in >1 dim = corner/edge only
            touching = i;
        } else {
            // Must strictly overlap in this dimension
            if (a.low[i] >= b.high[i] - eps || b.low[i] >= a.high[i] - eps)
                return false;
        }
    }
    return touching >= 0;
}

void GridDecompositionImpl::getNeighbors(int rid, std::vector<int>& neighbors) const
{
    neighbors.clear();

    if (children_.count(rid)) return;  // parent cell — not navigable

    int base_count = ompl::control::GridDecomposition::getNumRegions();

    // Fast path: use the grid formula when no sub-regions have been created and
    // rid is an original-grid cell.
    if (virtualBounds_.empty() && rid < base_count)
    {
        if (dimension_ == 2)
        {
            int row = rid / length_;
            int col = rid % length_;
            const int drow[] = {-1, 1, 0, 0};
            const int dcol[] = {0, 0, -1, 1};
            for (int i = 0; i < 4; ++i) {
                int nrow = row + drow[i];
                int ncol = col + dcol[i];
                if (nrow >= 0 && nrow < length_ && ncol >= 0 && ncol < length_)
                    neighbors.push_back(nrow * length_ + ncol);
            }
        }
        else if (dimension_ == 3)
        {
            int z = rid / (length_ * length_);
            int y = (rid / length_) % length_;
            int x = rid % length_;
            const int dx[] = {-1, 1, 0, 0, 0, 0};
            const int dy[] = {0, 0, -1, 1, 0, 0};
            const int dz[] = {0, 0, 0, 0, -1, 1};
            for (int i = 0; i < 6; ++i) {
                int nx = x + dx[i];
                int ny = y + dy[i];
                int nz = z + dz[i];
                if (nx >= 0 && nx < length_ && ny >= 0 && ny < length_ && nz >= 0 && nz < length_)
                    neighbors.push_back(nz * length_ * length_ + ny * length_ + nx);
            }
        }
        return;
    }

    // Slow path: spatial adjacency — needed when sub-regions exist.
    // Iterates all leaf cells and returns those that share a face with rid.
    const auto& bounds = getBoundsForRegion(rid);

    for (int i = 0; i < base_count; ++i) {
        if (i == rid || children_.count(i)) continue;
        if (spatiallyAdjacent(bounds, getBoundsForRegion(i), dimension_))
            neighbors.push_back(i);
    }
    for (const auto& [vid, vbounds] : virtualBounds_) {
        if (vid == rid || children_.count(vid)) continue;
        if (spatiallyAdjacent(bounds, *vbounds, dimension_))
            neighbors.push_back(vid);
    }
}

void GridDecompositionImpl::getAllNeighbors(int rid, std::vector<int>& neighbors) const
{
    neighbors.clear();

    if (dimension_ == 2)
    {
        int row = rid / length_;
        int col = rid % length_;

        // All 8 directions: cardinal + diagonal
        const int drow[] = {-1, -1, -1, 0, 0, 1, 1, 1};
        const int dcol[] = {-1,  0,  1, -1, 1, -1, 0, 1};

        for (int i = 0; i < 8; ++i)
        {
            int nrow = row + drow[i];
            int ncol = col + dcol[i];

            if (nrow >= 0 && nrow < length_ && ncol >= 0 && ncol < length_)
            {
                neighbors.push_back(nrow * length_ + ncol);
            }
        }
    }
    else if (dimension_ == 3)
    {
        int z = rid / (length_ * length_);
        int y = (rid / length_) % length_;
        int x = rid % length_;

        // All 26 directions: face + edge + corner neighbors
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0 && dz == 0) continue;
            int nx = x + dx, ny = y + dy, nz = z + dz;
            if (nx >= 0 && nx < length_ && ny >= 0 && ny < length_ && nz >= 0 && nz < length_)
                neighbors.push_back(nz * length_ * length_ + ny * length_ + nx);
        }
    }
}

void GridDecompositionImpl::Decompose(int rid)
{
    // If this region has already been decomposed, recurse into its children.
    if (children_.count(rid)) {
        // std::cout << "Num children " << rid << ": " << children_[rid].size() << std::endl;
        for (int child : children_.at(rid)) {
            Decompose(child);
            // std::cout << "Num children " << child << ": " << child << std::endl;
        }
        // std::cout << "Num children " << rid << ": " << children_[rid].size() << std::endl;
        return;
    }

    // Break the region into smaller subregions (split into 4 in 2D, 8 in 3D).
    // Works on both flat-grid IDs and virtual IDs from prior Decompose calls.
    const ompl::base::RealVectorBounds& parentBounds = getBoundsForRegion(rid);

    std::vector<double> mid(dimension_);
    for (int i = 0; i < dimension_; ++i)
        mid[i] = 0.5 * (parentBounds.low[i] + parentBounds.high[i]);

    const int numChildren = 1 << dimension_;  // 4 in 2D, 8 in 3D
    std::vector<int> childIds;
    childIds.reserve(numChildren);

    for (int q = 0; q < numChildren; ++q)
    {
        auto sb = std::make_shared<ompl::base::RealVectorBounds>(dimension_);
        for (int i = 0; i < dimension_; ++i)
        {
            if (q & (1 << i))  // bit i: 1 → upper half, 0 → lower half
            {
                sb->low[i]  = mid[i];
                sb->high[i] = parentBounds.high[i];
            }
            else
            {
                sb->low[i]  = parentBounds.low[i];
                sb->high[i] = mid[i];
            }
        }
        int childId = nextVirtualId_++;
        virtualBounds_[childId] = std::move(sb);
        parent_[childId] = rid;
        childIds.push_back(childId);
    }

    children_[rid] = std::move(childIds);
}

int GridDecompositionImpl::locateSubRegion(const ompl::base::State* s) const
{
    int rid = locateRegion(s);
    if (rid < 0) return -1;

    // Walk down the decomposition tree until we reach a leaf.
    while (children_.count(rid)) {
        std::vector<double> coord;
        project(s, coord);
        int matched = -1;
        for (int child : children_.at(rid)) {
            const auto& cb = getBoundsForRegion(child);
            bool inside = true;
            for (int i = 0; i < dimension_ && inside; ++i)
                inside = (coord[i] >= cb.low[i] && coord[i] < cb.high[i]);
            if (inside) { matched = child; break; }
        }
        if (matched < 0) break;  // state on a boundary edge; return parent
        rid = matched;
    }
    return rid;
}

int GridDecompositionImpl::getDecompositionDepth(int rid) const
{
    int depth = 0;
    int current = rid;
    while (parent_.count(current))
    {
        current = parent_.at(current);
        ++depth;
    }
    return depth;
}

int GridDecompositionImpl::getMaxDecompositions(int rid, double minSideLength) const
{
    const auto& bounds = getBoundsForRegion(rid);
    double minSide = std::numeric_limits<double>::max();
    for (int i = 0; i < dimension_; ++i)
        minSide = std::min(minSide, bounds.high[i] - bounds.low[i]);
    int n = 0;
    double side = minSide / 2.0;
    while (side > minSideLength)
    {
        ++n;
        side /= 2.0;
    }
    return n;
}

void GridDecompositionImpl::resetCell(int rid)
{
    auto it = children_.find(rid);
    if (it == children_.end()) return;

    for (int child : it->second)
        parent_.erase(child);

    children_.erase(it);
}