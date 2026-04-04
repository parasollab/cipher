#define BOOST_TEST_MODULE "CBS"
#include <boost/test/unit_test.hpp>

#include "mapf/cbs.h"
#include "utils/grid_decomposition.h"
#include <ompl/base/spaces/SE2StateSpace.h>
#include <algorithm>

// ─── Helpers ────────────────────────────────────────────────────────────────

static ompl::base::StateSpacePtr makeSE2Space()
{
    auto space = std::make_shared<ompl::base::SE2StateSpace>();
    ompl::base::RealVectorBounds bounds(2);
    bounds.low[0]  = bounds.low[1]  = 0.0;
    bounds.high[0] = bounds.high[1] = 1.0;
    space->setBounds(bounds);
    return space;
}

static std::shared_ptr<GridDecompositionImpl> makeDecomp(int length = 4)
{
    ompl::base::RealVectorBounds bounds(2);
    bounds.low[0]  = bounds.low[1]  = 0.0;
    bounds.high[0] = bounds.high[1] = 1.0;
    return std::make_shared<GridDecompositionImpl>(length, 2, bounds);
}

static void setXY(ompl::base::State* s, double x, double y)
{
    s->as<ompl::base::SE2StateSpace::StateType>()->setXY(x, y);
    s->as<ompl::base::SE2StateSpace::StateType>()->setYaw(0.0);
}

// ─── Single-robot tests ──────────────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(SingleRobot_TrivialPath)
{
    // When start and goal are the same region the path should be a single region.
    auto space  = makeSE2Space();
    auto decomp = makeDecomp();
    CBS solver(1, 10.0);

    auto* s = space->allocState();
    setXY(s, 0.125, 0.125);

    auto result = solver.solve(decomp, {s}, {s});

    BOOST_CHECK_EQUAL(result.size(), 1u);
    BOOST_CHECK_EQUAL(result[0].size(), 1u);
    BOOST_CHECK_EQUAL(result[0][0], decomp->locateRegion(s));

    space->freeState(s);
}

BOOST_AUTO_TEST_CASE(SingleRobot_CorrectEndpoints)
{
    auto space  = makeSE2Space();
    auto decomp = makeDecomp();
    CBS solver(1, 10.0);

    auto* start = space->allocState(); setXY(start, 0.125, 0.125);
    auto* goal  = space->allocState(); setXY(goal,  0.875, 0.875);

    int start_region = decomp->locateRegion(start);
    int goal_region  = decomp->locateRegion(goal);

    auto result = solver.solve(decomp, {start}, {goal});

    BOOST_CHECK_EQUAL(result.size(), 1u);
    BOOST_CHECK_EQUAL(result[0].front(), start_region);
    BOOST_CHECK_EQUAL(result[0].back(),  goal_region);

    space->freeState(start);
    space->freeState(goal);
}

BOOST_AUTO_TEST_CASE(SingleRobot_PathIsConnected)
{
    // Every consecutive pair of regions must be neighbors, or equal (wait action).
    auto space  = makeSE2Space();
    auto decomp = makeDecomp();
    CBS solver(1, 10.0);

    auto* start = space->allocState(); setXY(start, 0.125, 0.125);
    auto* goal  = space->allocState(); setXY(goal,  0.875, 0.875);

    auto result = solver.solve(decomp, {start}, {goal});
    const auto& path = result[0];

    for (size_t i = 0; i + 1 < path.size(); ++i)
    {
        if (path[i] == path[i+1])
            continue;  // valid wait

        std::vector<int> neighbors;
        decomp->getNeighbors(path[i], neighbors);
        bool adjacent = std::find(neighbors.begin(), neighbors.end(), path[i+1])
                        != neighbors.end();
        BOOST_CHECK_MESSAGE(adjacent,
            "step " << i << ": " << path[i] << " -> " << path[i+1] << " is not a valid move");
    }

    space->freeState(start);
    space->freeState(goal);
}

BOOST_AUTO_TEST_CASE(SingleRobot_AdjacentRegions)
{
    // One-step path: start and goal are immediate neighbors.
    auto space  = makeSE2Space();
    auto decomp = makeDecomp();
    CBS solver(1, 10.0);

    auto* start = space->allocState(); setXY(start, 0.125, 0.125);
    auto* goal  = space->allocState(); setXY(goal,  0.375, 0.125);

    int sr = decomp->locateRegion(start);
    int gr = decomp->locateRegion(goal);

    // Verify these are actually neighbors in the decomposition
    std::vector<int> neighbors;
    decomp->getNeighbors(sr, neighbors);
    BOOST_REQUIRE_MESSAGE(
        std::find(neighbors.begin(), neighbors.end(), gr) != neighbors.end(),
        "Test setup: " << sr << " and " << gr << " must be neighbors");

    auto result = solver.solve(decomp, {start}, {goal});

    BOOST_CHECK_EQUAL(result[0].front(), sr);
    BOOST_CHECK_EQUAL(result[0].back(),  gr);

    space->freeState(start);
    space->freeState(goal);
}

// ─── Multi-robot tests ───────────────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(TwoRobots_ResultSizeMatchesRobotCount)
{
    auto space  = makeSE2Space();
    auto decomp = makeDecomp();
    CBS solver(1, 10.0);

    auto* s0 = space->allocState(); setXY(s0, 0.125, 0.125);
    auto* g0 = space->allocState(); setXY(g0, 0.875, 0.125);
    auto* s1 = space->allocState(); setXY(s1, 0.125, 0.875);
    auto* g1 = space->allocState(); setXY(g1, 0.875, 0.875);

    auto result = solver.solve(decomp, {s0, s1}, {g0, g1});

    BOOST_CHECK_EQUAL(result.size(), 2u);
    BOOST_CHECK(!result[0].empty());
    BOOST_CHECK(!result[1].empty());

    space->freeState(s0); space->freeState(g0);
    space->freeState(s1); space->freeState(g1);
}

BOOST_AUTO_TEST_CASE(TwoRobots_CorrectEndpoints)
{
    auto space  = makeSE2Space();
    auto decomp = makeDecomp();
    CBS solver(1, 10.0);

    auto* s0 = space->allocState(); setXY(s0, 0.125, 0.125);
    auto* g0 = space->allocState(); setXY(g0, 0.875, 0.125);
    auto* s1 = space->allocState(); setXY(s1, 0.125, 0.875);
    auto* g1 = space->allocState(); setXY(g1, 0.875, 0.875);

    int sr0 = decomp->locateRegion(s0), gr0 = decomp->locateRegion(g0);
    int sr1 = decomp->locateRegion(s1), gr1 = decomp->locateRegion(g1);

    auto result = solver.solve(decomp, {s0, s1}, {g0, g1});

    BOOST_CHECK_EQUAL(result[0].front(), sr0);
    BOOST_CHECK_EQUAL(result[0].back(),  gr0);
    BOOST_CHECK_EQUAL(result[1].front(), sr1);
    BOOST_CHECK_EQUAL(result[1].back(),  gr1);

    space->freeState(s0); space->freeState(g0);
    space->freeState(s1); space->freeState(g1);
}

BOOST_AUTO_TEST_CASE(TwoRobots_ConflictingPaths_FindsSolution)
{
    // Both robots travel the same row in opposite directions — CBS must resolve
    // the head-on conflict via constraints.
    auto space  = makeSE2Space();
    auto decomp = makeDecomp();
    CBS solver(1, 30.0);

    auto* s0 = space->allocState(); setXY(s0, 0.125, 0.125);
    auto* g0 = space->allocState(); setXY(g0, 0.875, 0.125);
    auto* s1 = space->allocState(); setXY(s1, 0.875, 0.125);
    auto* g1 = space->allocState(); setXY(g1, 0.125, 0.125);

    int sr0 = decomp->locateRegion(s0), gr0 = decomp->locateRegion(g0);
    int sr1 = decomp->locateRegion(s1), gr1 = decomp->locateRegion(g1);

    auto result = solver.solve(decomp, {s0, s1}, {g0, g1});

    BOOST_CHECK_EQUAL(result.size(), 2u);
    BOOST_CHECK_EQUAL(result[0].front(), sr0);
    BOOST_CHECK_EQUAL(result[0].back(),  gr0);
    BOOST_CHECK_EQUAL(result[1].front(), sr1);
    BOOST_CHECK_EQUAL(result[1].back(),  gr1);

    space->freeState(s0); space->freeState(g0);
    space->freeState(s1); space->freeState(g1);
}

BOOST_AUTO_TEST_CASE(TwoRobots_NoSharedRegionAtSameTime)
{
    // With capacity=1, reconstruct per-timestep occupancy and verify no two
    // robots share a region at the same timestep.
    auto space  = makeSE2Space();
    auto decomp = makeDecomp();
    CBS solver(1, 30.0);

    auto* s0 = space->allocState(); setXY(s0, 0.125, 0.125);
    auto* g0 = space->allocState(); setXY(g0, 0.875, 0.125);
    auto* s1 = space->allocState(); setXY(s1, 0.875, 0.125);
    auto* g1 = space->allocState(); setXY(g1, 0.125, 0.125);

    auto result = solver.solve(decomp, {s0, s1}, {g0, g1});

    // Extend paths to equal length by holding at the final region.
    size_t max_len = 0;
    for (const auto& p : result) max_len = std::max(max_len, p.size());

    auto regionAt = [&](int robot, size_t t) {
        const auto& p = result[robot];
        return p[std::min(t, p.size() - 1)];
    };

    for (size_t t = 0; t < max_len; ++t)
    {
        int r0 = regionAt(0, t);
        int r1 = regionAt(1, t);
        BOOST_CHECK_MESSAGE(r0 != r1,
            "robots share region " << r0 << " at timestep " << t);
    }

    space->freeState(s0); space->freeState(g0);
    space->freeState(s1); space->freeState(g1);
}

BOOST_AUTO_TEST_CASE(ThreeRobots_FindsSolution)
{
    auto space  = makeSE2Space();
    auto decomp = makeDecomp();
    CBS solver(1, 30.0);

    auto* s0 = space->allocState(); setXY(s0, 0.125, 0.125);
    auto* g0 = space->allocState(); setXY(g0, 0.875, 0.875);
    auto* s1 = space->allocState(); setXY(s1, 0.875, 0.125);
    auto* g1 = space->allocState(); setXY(g1, 0.125, 0.875);
    auto* s2 = space->allocState(); setXY(s2, 0.125, 0.875);
    auto* g2 = space->allocState(); setXY(g2, 0.875, 0.125);

    auto result = solver.solve(decomp, {s0, s1, s2}, {g0, g1, g2});

    BOOST_CHECK_EQUAL(result.size(), 3u);
    for (size_t i = 0; i < 3; ++i)
        BOOST_CHECK(!result[i].empty());

    space->freeState(s0); space->freeState(g0);
    space->freeState(s1); space->freeState(g1);
    space->freeState(s2); space->freeState(g2);
}

BOOST_AUTO_TEST_CASE(HighCapacity_AllowsSharing)
{
    // capacity=2 lets two robots occupy the same region simultaneously.
    // Two robots with identical start/goal should both find a direct path.
    auto space  = makeSE2Space();
    auto decomp = makeDecomp();
    CBS solver(2, 10.0);

    auto* s0 = space->allocState(); setXY(s0, 0.125, 0.125);
    auto* g0 = space->allocState(); setXY(g0, 0.875, 0.125);
    auto* s1 = space->allocState(); setXY(s1, 0.125, 0.125);
    auto* g1 = space->allocState(); setXY(g1, 0.875, 0.125);

    BOOST_CHECK_NO_THROW(solver.solve(decomp, {s0, s1}, {g0, g1}));

    space->freeState(s0); space->freeState(g0);
    space->freeState(s1); space->freeState(g1);
}
