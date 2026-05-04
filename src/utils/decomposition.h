#ifndef DECOMPOSITION_IMPL_H
#define DECOMPOSITION_IMPL_H

#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/base/StateSampler.h>
#include <ompl/base/State.h>
#include <ompl/util/RandomNumbers.h>
#include <vector>

class DecompositionImpl {
  public:
    // Constructor based on robot size and environment bounds
    DecompositionImpl(int dim, const ompl::base::RealVectorBounds& bounds, double region_size) {};
    DecompositionImpl() = default;

    virtual ~DecompositionImpl() = default;

    // OMPL Decomposition interface
    virtual int getNumRegions() const = 0;
    virtual int getDimension() const = 0;
    virtual const ompl::base::RealVectorBounds& getBounds() const = 0;
    virtual double getRegionVolume(int rid) = 0;
    virtual int locateRegion(const ompl::base::State* s) const = 0;
    // Like locateRegion, but recursively resolves into child sub-regions when
    // the containing region has been decomposed. Returns the finest-grained
    // region ID that contains s, or -1 if s is out of bounds.
    virtual int locateSubRegion(const ompl::base::State* s) const { return locateRegion(s); }
    virtual void project(const ompl::base::State* s, std::vector<double>& coord) const = 0;
    virtual void getNeighbors(int rid, std::vector<int>& neighbors) const = 0;
    virtual void getAllNeighbors(int rid, std::vector<int>& neighbors) const = 0;
    virtual void sampleFromRegion(int rid, ompl::RNG& rng, std::vector<double>& coord) const = 0;
    virtual void sampleFullState(const ompl::base::StateSamplerPtr& sampler, const std::vector<double>& coord, ompl::base::State* s) const = 0;
    virtual ompl::base::RealVectorBounds getCellBounds(int rid) const = 0;

    // Extended interface
    virtual void Decompose(int rid) = 0;
    virtual int getDecompositionDepth(int rid) const = 0;
    virtual int getMaxDecompositions(int rid, double minSideLength) const = 0;
    virtual void resetCell(int rid) = 0;

    // Returns the total number of regions including virtual sub-regions created
    // by Decompose(). Base default is the same as getNumRegions().
    virtual int getTotalNumRegions() const { return getNumRegions(); }

    // Returns true if rid is a leaf region (has not been further decomposed).
    virtual bool isLeafRegion(int rid) const { return true; }
};

#endif // DECOMPOSITION_IMPL_H