#ifndef GRID_DECOMPOSITION_IMPL_H
#define GRID_DECOMPOSITION_IMPL_H

#include "decomposition.h"
#include <ompl/base/StateSpace.h>
#include <ompl/control/planners/syclop/GridDecomposition.h>
#include <unordered_map>
#include <vector>

class GridDecompositionImpl : public ompl::control::GridDecomposition, public DecompositionImpl {
  public:
    GridDecompositionImpl(const int length, const int dim, const ompl::base::RealVectorBounds& bounds);
    GridDecompositionImpl(const int dim, const ompl::base::RealVectorBounds& bounds, double region_size);

    // Bridge overrides: satisfy DecompositionImpl's pure virtuals using GridDecomposition's implementations
    int getNumRegions() const override { return ompl::control::GridDecomposition::getNumRegions(); }
    int getDimension() const override { return ompl::control::GridDecomposition::getDimension(); }
    const ompl::base::RealVectorBounds& getBounds() const override { return ompl::control::GridDecomposition::getBounds(); }
    double getRegionVolume(int rid) override { return ompl::control::GridDecomposition::getRegionVolume(rid); }
    int locateRegion(const ompl::base::State* s) const override { return ompl::control::GridDecomposition::locateRegion(s); }
    void sampleFromRegion(int rid, ompl::RNG& rng, std::vector<double>& coord) const override { ompl::control::GridDecomposition::sampleFromRegion(rid, rng, coord); }

    void project(const ompl::base::State* s, std::vector<double>& coord) const override;

    void sampleFullState(const ompl::base::StateSamplerPtr& sampler, const std::vector<double>& coord, ompl::base::State* s) const override;

    // Public wrapper to access protected getRegionBounds method
    const ompl::base::RealVectorBounds& getRegionBoundsPublic(int rid) const {
        return getRegionBounds(rid);
    }

    // Override to only return cardinal neighbors (no diagonals)
    void getNeighbors(int rid, std::vector<int>& neighbors) const override;

    void Decompose(int rid) override;

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
    std::unordered_map<int, std::shared_ptr<ompl::base::RealVectorBounds>> virtualBounds_;
    ompl::base::StateSpacePtr state_space_;  // optional; enables generic project/sampleFullState
};

#endif // GRID_DECOMPOSITION_IMPL_H