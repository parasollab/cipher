#ifndef GRID_DECOMPOSITION_IMPL_H
#define GRID_DECOMPOSITION_IMPL_H

#include "decomposition.h"
#include <ompl/base/StateSpace.h>
#include <ompl/control/planners/syclop/GridDecomposition.h>
#include <unordered_map>
#include <vector>

// Rectangular grid decomposition with independent per-dimension cell counts.
// Unlike GridDecompositionImpl (which requires a single length for all dims),
// this correctly handles non-square expanded regions where n_x != n_y.
// Each cell has size (span_x/length_x) × (span_y/length_y), which equals
// original_cell_size/sf in both dimensions when used for refinement.
class RectGridDecompositionImpl : public DecompositionImpl {
  public:
    // lengths[i] = number of cells along dimension i
    RectGridDecompositionImpl(const std::vector<int>& lengths,
                               const ompl::base::RealVectorBounds& bounds,
                               const ompl::base::StateSpacePtr& space);

    int getNumRegions() const override { return num_regions_; }
    int getDimension() const override { return dim_; }
    const ompl::base::RealVectorBounds& getBounds() const override { return bounds_; }
    double getRegionVolume(int rid) override;
    int locateRegion(const ompl::base::State* s) const override;
    void project(const ompl::base::State* s, std::vector<double>& coord) const override;
    void getNeighbors(int rid, std::vector<int>& neighbors) const override;
    void getAllNeighbors(int rid, std::vector<int>& neighbors) const override;
    void sampleFromRegion(int rid, ompl::RNG& rng, std::vector<double>& coord) const override;
    void sampleFullState(const ompl::base::StateSamplerPtr& sampler,
                         const std::vector<double>& coord, ompl::base::State* s) const override;
    ompl::base::RealVectorBounds getCellBounds(int rid) const override;
    void Decompose(int rid) override {}  // not needed for local refinement decomps
    int getDecompositionDepth(int /*rid*/) const override { return 0; }
    int getMaxDecompositions(int /*rid*/, double /*minSideLength*/) const override { return 0; }

  private:
    int dim_;
    int num_regions_;
    std::vector<int> lengths_;       // cells per dimension
    std::vector<double> cell_size_;  // (bounds.high[i] - bounds.low[i]) / lengths_[i]
    ompl::base::RealVectorBounds bounds_;
    ompl::base::StateSpacePtr state_space_;

    // Convert flat region id to per-dimension indices
    std::vector<int> ridToCoord(int rid) const;
    // Convert per-dimension indices to flat region id (-1 if out of range)
    int coordToRid(const std::vector<int>& coord) const;
};

class GridDecompositionImpl : public ompl::control::GridDecomposition, public DecompositionImpl {
  public:
    GridDecompositionImpl(const int length, const int dim, const ompl::base::RealVectorBounds& bounds);
    GridDecompositionImpl(const int dim, const ompl::base::RealVectorBounds& bounds, double region_size);

    // Bridge overrides: satisfy DecompositionImpl's pure virtuals using GridDecomposition's implementations
    int getNumRegions() const override { return ompl::control::GridDecomposition::getNumRegions(); }
    int getDimension() const override { return ompl::control::GridDecomposition::getDimension(); }
    const ompl::base::RealVectorBounds& getBounds() const override { return ompl::control::GridDecomposition::getBounds(); }
    double getRegionVolume(int rid) override {
        const auto& b = getBoundsForRegion(rid);
        double vol = 1.0;
        for (int i = 0; i < dimension_; ++i)
            vol *= (b.high[i] - b.low[i]);
        return vol;
    }
    int locateRegion(const ompl::base::State* s) const override { return ompl::control::GridDecomposition::locateRegion(s); }
    void sampleFromRegion(int rid, ompl::RNG& rng, std::vector<double>& coord) const override { ompl::control::GridDecomposition::sampleFromRegion(rid, rng, coord); }

    void project(const ompl::base::State* s, std::vector<double>& coord) const override;

    void sampleFullState(const ompl::base::StateSamplerPtr& sampler, const std::vector<double>& coord, ompl::base::State* s) const override;

    const ompl::base::RealVectorBounds& getRegionBounds(int rid) const {
        return ompl::control::GridDecomposition::getRegionBounds(rid);
    }

    // Public wrapper to access protected getRegionBounds method
    // const ompl::base::RealVectorBounds& getRegionBoundsPublic(int rid) const {
    //     return getRegionBounds(rid);
    // }

    ompl::base::RealVectorBounds getCellBounds(int rid) const override {
        return getBoundsForRegion(rid);
    }

    // Override to only return cardinal neighbors (no diagonals)
    void getNeighbors(int rid, std::vector<int>& neighbors) const override;

    // Returns all 8 neighbors (cardinal + diagonal)
    void getAllNeighbors(int rid, std::vector<int>& neighbors) const override;

    void Decompose(int rid) override;
    int getDecompositionDepth(int rid) const override;
    int getMaxDecompositions(int rid, double minSideLength) const override;

    int locateSubRegion(const ompl::base::State* s) const override;

    int getTotalNumRegions() const override { return nextVirtualId_; }
    bool isLeafRegion(int rid) const override { return !children_.count(rid); }

    // Set the robot state space so that project/sampleFullState work with any
    // state type (SE2, RealVector, etc.) via copyToReals/copyFromReals.
    // When not set, falls back to the original SE2 hard-cast behaviour.
    void setStateSpace(const ompl::base::StateSpacePtr& space) { state_space_ = space; }

    // Returns the child region IDs produced by Decompose(rid).
    const std::vector<int>& getChildRegions(int rid) const {
        return children_.at(rid);
    }

    bool hasDecomposed(int rid) const {
        return children_.count(rid) > 0;
    }

    // Returns bounds for any region ID — both flat-grid and virtual sub-region IDs.
    const ompl::base::RealVectorBounds& getBoundsForRegion(int rid) const {
        if (rid < ompl::control::GridDecomposition::getNumRegions())
            return getRegionBounds(rid);
        return *virtualBounds_.at(rid);
    }

  private:
    int nextVirtualId_;
    std::unordered_map<int, std::vector<int>> children_;
    std::unordered_map<int, int> parent_;
    std::unordered_map<int, std::shared_ptr<ompl::base::RealVectorBounds>> virtualBounds_;
    ompl::base::StateSpacePtr state_space_;  // optional; enables generic project/sampleFullState
};

#endif // GRID_DECOMPOSITION_IMPL_H