#define BOOST_TEST_MODULE "GridDecomposition"
#include <boost/test/unit_test.hpp>

#include "utils/grid_decomposition.h"
#include <ompl/base/spaces/RealVectorBounds.h>

// Helper: build a square [0,1]^dim decomposition with given length
static GridDecompositionImpl makeDecomp(int length, int dim)
{
    ompl::base::RealVectorBounds bounds(dim);
    for (int i = 0; i < dim; ++i)
    {
        bounds.low[i]  = 0.0;
        bounds.high[i] = 1.0;
    }
    return GridDecompositionImpl(length, dim, bounds);
}

// Helper: build a decomposition from region_size
static GridDecompositionImpl makeDecompFromSize(int dim, double low, double high, double region_size)
{
    ompl::base::RealVectorBounds bounds(dim);
    for (int i = 0; i < dim; ++i)
    {
        bounds.low[i]  = low;
        bounds.high[i] = high;
    }
    return GridDecompositionImpl(dim, bounds, region_size);
}

// ─── region_size constructor tests ───────────────────────────────────────────

BOOST_AUTO_TEST_CASE(RegionSize2D_CellsAtMostRegionSize)
{
    // [0,1]^2 with region_size=0.25 → length=4, 16 regions
    auto d = makeDecompFromSize(2, 0.0, 1.0, 0.25);
    BOOST_CHECK_EQUAL(d.getNumRegions(), 16);

    // Every region's extent in each dimension should be <= region_size
    for (int rid = 0; rid < d.getNumRegions(); ++rid)
    {
        const auto& b = d.getBoundsForRegion(rid);
        BOOST_CHECK_LE(b.high[0] - b.low[0], 0.25 + 1e-9);
        BOOST_CHECK_LE(b.high[1] - b.low[1], 0.25 + 1e-9);
    }
}

BOOST_AUTO_TEST_CASE(RegionSize2D_NonDivisibleExtent)
{
    // [0,1]^2 with region_size=0.3 → ceil(1.0/0.3)=4 cells → length=4, 16 regions
    auto d = makeDecompFromSize(2, 0.0, 1.0, 0.3);
    BOOST_CHECK_EQUAL(d.getNumRegions(), 16);
}

BOOST_AUTO_TEST_CASE(RegionSize2D_LargeRegionSize)
{
    // region_size >= extent → length=1, 1 region covers the whole space
    auto d = makeDecompFromSize(2, 0.0, 1.0, 2.0);
    BOOST_CHECK_EQUAL(d.getNumRegions(), 1);
    const auto& b = d.getBoundsForRegion(0);
    BOOST_CHECK_CLOSE(b.low[0],  0.0, 1e-9);
    BOOST_CHECK_CLOSE(b.high[0], 1.0, 1e-9);
}

BOOST_AUTO_TEST_CASE(RegionSize2D_AsymmetricBounds)
{
    // X: [0,4], Y: [0,1], region_size=1.0 → length=max(4,1)=4, 16 regions
    ompl::base::RealVectorBounds bounds(2);
    bounds.low[0]  = 0.0; bounds.high[0] = 4.0;
    bounds.low[1]  = 0.0; bounds.high[1] = 1.0;
    GridDecompositionImpl d(2, bounds, 1.0);
    BOOST_CHECK_EQUAL(d.getNumRegions(), 16);
}

BOOST_AUTO_TEST_CASE(RegionSize2D_MatchesLengthConstructor)
{
    // Both constructors with equivalent parameters should produce the same grid
    auto dSize   = makeDecompFromSize(2, 0.0, 1.0, 0.25);  // length=4
    auto dLength = makeDecomp(4, 2);
    BOOST_CHECK_EQUAL(dSize.getNumRegions(),  dLength.getNumRegions());
    BOOST_CHECK_EQUAL(dSize.getDimension(),   dLength.getDimension());
    for (int rid = 0; rid < dSize.getNumRegions(); ++rid)
    {
        const auto& bs = dSize.getBoundsForRegion(rid);
        const auto& bl = dLength.getBoundsForRegion(rid);
        for (int i = 0; i < 2; ++i)
        {
            BOOST_CHECK_CLOSE(bs.low[i],  bl.low[i],  1e-9);
            BOOST_CHECK_CLOSE(bs.high[i], bl.high[i], 1e-9);
        }
    }
}

BOOST_AUTO_TEST_CASE(RegionSize3D_CellsAtMostRegionSize)
{
    // [0,1]^3 with region_size=0.5 → length=2, 8 regions
    auto d = makeDecompFromSize(3, 0.0, 1.0, 0.5);
    BOOST_CHECK_EQUAL(d.getNumRegions(), 8);

    for (int rid = 0; rid < d.getNumRegions(); ++rid)
    {
        const auto& b = d.getBoundsForRegion(rid);
        for (int i = 0; i < 3; ++i)
            BOOST_CHECK_LE(b.high[i] - b.low[i], 0.5 + 1e-9);
    }
}

// ─── 2D tests ────────────────────────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(Decompose2D_ChildCount)
{
    auto d = makeDecomp(4, 2);
    d.Decompose(0);
    BOOST_CHECK(d.hasDecomposed(0));
    BOOST_CHECK_EQUAL(d.getChildRegions(0).size(), 4u);
}

BOOST_AUTO_TEST_CASE(Decompose2D_ChildBoundsHalved)
{
    auto d = makeDecomp(4, 2);
    // Region 0 spans [0, 0.25] x [0, 0.25] (first cell of a 4x4 grid)
    const auto& parentBounds = d.getBoundsForRegion(0);
    d.Decompose(0);

    const auto& children = d.getChildRegions(0);
    // Each child should cover half the parent in each dimension
    double midX = 0.5 * (parentBounds.low[0] + parentBounds.high[0]);
    double midY = 0.5 * (parentBounds.low[1] + parentBounds.high[1]);

    for (int q = 0; q < 4; ++q)
    {
        const auto& cb = d.getBoundsForRegion(children[q]);
        double expectedLowX  = (q & 1) ? midX : parentBounds.low[0];
        double expectedHighX = (q & 1) ? parentBounds.high[0] : midX;
        double expectedLowY  = (q & 2) ? midY : parentBounds.low[1];
        double expectedHighY = (q & 2) ? parentBounds.high[1] : midY;

        BOOST_CHECK_CLOSE(cb.low[0],  expectedLowX,  1e-9);
        BOOST_CHECK_CLOSE(cb.high[0], expectedHighX, 1e-9);
        BOOST_CHECK_CLOSE(cb.low[1],  expectedLowY,  1e-9);
        BOOST_CHECK_CLOSE(cb.high[1], expectedHighY, 1e-9);
    }
}

BOOST_AUTO_TEST_CASE(Decompose2D_Recursive)
{
    auto d = makeDecomp(4, 2);
    d.Decompose(0);
    int firstChild = d.getChildRegions(0)[0];

    d.Decompose(firstChild);
    BOOST_CHECK(d.hasDecomposed(firstChild));
    BOOST_CHECK_EQUAL(d.getChildRegions(firstChild).size(), 4u);

    // Grandchildren bounds should be 1/4 the size of the original region
    const auto& parentBounds = d.getBoundsForRegion(0);
    double quarterX = (parentBounds.high[0] - parentBounds.low[0]) / 4.0;
    double quarterY = (parentBounds.high[1] - parentBounds.low[1]) / 4.0;

    for (int gc : d.getChildRegions(firstChild))
    {
        const auto& gcb = d.getBoundsForRegion(gc);
        BOOST_CHECK_CLOSE(gcb.high[0] - gcb.low[0], quarterX, 1e-9);
        BOOST_CHECK_CLOSE(gcb.high[1] - gcb.low[1], quarterY, 1e-9);
    }
}

BOOST_AUTO_TEST_CASE(Decompose2D_VirtualIdsDontOverlapGridIds)
{
    auto d = makeDecomp(4, 2);  // numGridCells = 16
    d.Decompose(0);
    for (int childId : d.getChildRegions(0))
        BOOST_CHECK_GE(childId, d.getNumRegions());
}

BOOST_AUTO_TEST_CASE(Decompose2D_MultipleRegions)
{
    auto d = makeDecomp(4, 2);
    d.Decompose(0);
    d.Decompose(5);

    BOOST_CHECK(d.hasDecomposed(0));
    BOOST_CHECK(d.hasDecomposed(5));
    BOOST_CHECK(!d.hasDecomposed(1));

    // Children of different regions should have distinct IDs
    const auto& c0 = d.getChildRegions(0);
    const auto& c5 = d.getChildRegions(5);
    for (int a : c0)
        for (int b : c5)
            BOOST_CHECK_NE(a, b);
}

// ─── 3D tests ────────────────────────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(Decompose3D_ChildCount)
{
    auto d = makeDecomp(3, 3);
    d.Decompose(0);
    BOOST_CHECK(d.hasDecomposed(0));
    BOOST_CHECK_EQUAL(d.getChildRegions(0).size(), 8u);
}

BOOST_AUTO_TEST_CASE(Decompose3D_ChildBoundsHalved)
{
    auto d = makeDecomp(3, 3);
    const auto& parentBounds = d.getBoundsForRegion(0);
    d.Decompose(0);

    double mid[3];
    for (int i = 0; i < 3; ++i)
        mid[i] = 0.5 * (parentBounds.low[i] + parentBounds.high[i]);

    const auto& children = d.getChildRegions(0);
    for (int q = 0; q < 8; ++q)
    {
        const auto& cb = d.getBoundsForRegion(children[q]);
        for (int i = 0; i < 3; ++i)
        {
            double expectedLow  = (q & (1 << i)) ? mid[i] : parentBounds.low[i];
            double expectedHigh = (q & (1 << i)) ? parentBounds.high[i] : mid[i];
            BOOST_CHECK_CLOSE(cb.low[i],  expectedLow,  1e-9);
            BOOST_CHECK_CLOSE(cb.high[i], expectedHigh, 1e-9);
        }
    }
}

BOOST_AUTO_TEST_CASE(Decompose3D_Recursive)
{
    auto d = makeDecomp(3, 3);
    d.Decompose(0);
    int firstChild = d.getChildRegions(0)[0];

    d.Decompose(firstChild);
    BOOST_CHECK_EQUAL(d.getChildRegions(firstChild).size(), 8u);

    const auto& parentBounds = d.getBoundsForRegion(0);
    for (int i = 0; i < 3; ++i)
    {
        double quarterSize = (parentBounds.high[i] - parentBounds.low[i]) / 4.0;
        for (int gc : d.getChildRegions(firstChild))
        {
            const auto& gcb = d.getBoundsForRegion(gc);
            BOOST_CHECK_CLOSE(gcb.high[i] - gcb.low[i], quarterSize, 1e-9);
        }
    }
}

// ─── getDecompositionDepth tests ─────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(DecompositionDepth_FlatGridIsZero)
{
    auto d = makeDecomp(4, 2);
    for (int rid = 0; rid < d.getNumRegions(); ++rid)
        BOOST_CHECK_EQUAL(d.getDecompositionDepth(rid), 0);
}

BOOST_AUTO_TEST_CASE(DecompositionDepth_ChildrenAreOne)
{
    auto d = makeDecomp(4, 2);
    d.Decompose(0);
    for (int child : d.getChildRegions(0))
        BOOST_CHECK_EQUAL(d.getDecompositionDepth(child), 1);
}

BOOST_AUTO_TEST_CASE(DecompositionDepth_GrandchildrenAreTwo)
{
    auto d = makeDecomp(4, 2);
    d.Decompose(0);
    int firstChild = d.getChildRegions(0)[0];
    d.Decompose(firstChild);
    for (int gc : d.getChildRegions(firstChild))
        BOOST_CHECK_EQUAL(d.getDecompositionDepth(gc), 2);
}

BOOST_AUTO_TEST_CASE(DecompositionDepth_ParentUnchangedAfterDecompose)
{
    // Decomposing a region should not change its own reported depth
    auto d = makeDecomp(4, 2);
    d.Decompose(0);
    BOOST_CHECK_EQUAL(d.getDecompositionDepth(0), 0);
}

BOOST_AUTO_TEST_CASE(DecompositionDepth_MultipleRootsIndependent)
{
    auto d = makeDecomp(4, 2);
    d.Decompose(0);
    d.Decompose(5);
    for (int child : d.getChildRegions(5))
        BOOST_CHECK_EQUAL(d.getDecompositionDepth(child), 1);
}

BOOST_AUTO_TEST_CASE(DecompositionDepth_3D)
{
    auto d = makeDecomp(3, 3);
    d.Decompose(0);
    for (int child : d.getChildRegions(0))
        BOOST_CHECK_EQUAL(d.getDecompositionDepth(child), 1);

    int firstChild = d.getChildRegions(0)[0];
    d.Decompose(firstChild);
    for (int gc : d.getChildRegions(firstChild))
        BOOST_CHECK_EQUAL(d.getDecompositionDepth(gc), 2);
}

// ─── getMaxDecompositions tests ──────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(MaxDecompositions_CountHalvingsAboveThreshold)
{
    // Single cell [0,1]^2, side=1.0, minSideLength=0.1:
    // 1.0→0.5→0.25→0.125 (all >0.1), 0.0625 not >0.1 → 3 halvings
    auto d = makeDecomp(1, 2);
    BOOST_CHECK_EQUAL(d.getMaxDecompositions(0, 0.1), 3);
}

BOOST_AUTO_TEST_CASE(MaxDecompositions_AlreadyTooSmall)
{
    // 4x4 grid, cells have side 0.25; 0.25/2=0.125 not >0.2 → 0
    auto d = makeDecomp(4, 2);
    BOOST_CHECK_EQUAL(d.getMaxDecompositions(0, 0.2), 0);
}

BOOST_AUTO_TEST_CASE(MaxDecompositions_ExactHalfIsNotStrictlyAbove)
{
    // 2x2 grid, cells have side 0.5; 0.5/2=0.25 which is not strictly >0.25 → 0
    auto d = makeDecomp(2, 2);
    BOOST_CHECK_EQUAL(d.getMaxDecompositions(0, 0.25), 0);
}

BOOST_AUTO_TEST_CASE(MaxDecompositions_VirtualChildRegion)
{
    // After decomposing the single [0,1]^2 cell (side=1.0), children have side=0.5.
    // 0.5→0.25→0.125 (all >0.1), 0.0625 not >0.1 → 2
    auto d = makeDecomp(1, 2);
    d.Decompose(0);
    int child = d.getChildRegions(0)[0];
    BOOST_CHECK_EQUAL(d.getMaxDecompositions(child, 0.1), 2);
    // Parent answer unchanged
    BOOST_CHECK_EQUAL(d.getMaxDecompositions(0, 0.1), 3);
}

BOOST_AUTO_TEST_CASE(MaxDecompositions_AsymmetricBoundsUsesMinSide)
{
    // X: [0,4], Y: [0,1], length=4 → cells are 1.0 wide (X) × 0.25 tall (Y)
    // Min side = 0.25; 0.25/2=0.125>0.1 (n=1), 0.0625 not >0.1 → 1
    ompl::base::RealVectorBounds bounds(2);
    bounds.low[0] = 0.0; bounds.high[0] = 4.0;
    bounds.low[1] = 0.0; bounds.high[1] = 1.0;
    GridDecompositionImpl d(2, bounds, 1.0);
    BOOST_CHECK_EQUAL(d.getMaxDecompositions(0, 0.1), 1);
}

BOOST_AUTO_TEST_CASE(MaxDecompositions_3D)
{
    // Single cell [0,1]^3, side=1.0 — same halving logic as 2D
    auto d = makeDecomp(1, 3);
    BOOST_CHECK_EQUAL(d.getMaxDecompositions(0, 0.1), 3);
}
