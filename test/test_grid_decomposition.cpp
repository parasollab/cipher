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
