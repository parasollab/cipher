#ifndef GRID_DECOMPOSITION_IMPL_H
#define GRID_DECOMPOSITION_IMPL_H

#include <ompl/control/planners/syclop/GridDecomposition.h>

class GridDecompositionImpl : public ompl::control::GridDecomposition {
  public:
    GridDecompositionImpl(const int length, const int dim, const ompl::base::RealVectorBounds& bounds);

    void project(const ompl::base::State* s, std::vector<double>& coord) const override;

    void sampleFullState(const ompl::base::StateSamplerPtr& sampler, const std::vector<double>& coord, ompl::base::State* s) const override;

    // Public wrapper to access protected getRegionBounds method
    const ompl::base::RealVectorBounds& getRegionBoundsPublic(int rid) const {
        return getRegionBounds(rid);
    }

    // Override to only return cardinal neighbors (no diagonals)
    void getNeighbors(int rid, std::vector<int>& neighbors) const override;
};

#endif // GRID_DECOMPOSITION_IMPL_H