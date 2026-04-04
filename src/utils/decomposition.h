#ifndef DECOMPOSITION_IMPL_H
#define DECOMPOSITION_IMPL_H

#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/base/StateSampler.h>
#include <ompl/base/State.h>
#include <ompl/util/RandomNumbers.h>
#include <vector>

class DecompositionImpl {
  public:
    virtual ~DecompositionImpl() = default;

    // OMPL Decomposition interface
    virtual int getNumRegions() const = 0;
    virtual int getDimension() const = 0;
    virtual const ompl::base::RealVectorBounds& getBounds() const = 0;
    virtual double getRegionVolume(int rid) = 0;
    virtual int locateRegion(const ompl::base::State* s) const = 0;
    virtual void project(const ompl::base::State* s, std::vector<double>& coord) const = 0;
    virtual void getNeighbors(int rid, std::vector<int>& neighbors) const = 0;
    virtual void sampleFromRegion(int rid, ompl::RNG& rng, std::vector<double>& coord) const = 0;
    virtual void sampleFullState(const ompl::base::StateSamplerPtr& sampler, const std::vector<double>& coord, ompl::base::State* s) const = 0;

    // Extended interface
    virtual void Decompose(int rid) = 0;
};

#endif // DECOMPOSITION_IMPL_H