#ifndef GUIDED_GEOMETRIC_RRT_H
#define GUIDED_GEOMETRIC_RRT_H

#include <iostream>
#include <unordered_set>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>


#include "utils/decomposition.h"

namespace og = ompl::geometric;
namespace ob = ompl::base;

class GuidedGeometricRRT : public og::RRT {
public:
    GuidedGeometricRRT(const ob::SpaceInformationPtr &si)
        : og::RRT(si) {}

    ~GuidedGeometricRRT() override = default;

    void setDecomposition(std::shared_ptr<DecompositionImpl> decomposition)
    {
        decomposition_ = decomposition;
    }

    void setDecompositionPath(const std::vector<int> &path)
    {
        decomp_path = path;
    }

    void setMaxExtensions(int n)            { max_extensions_ = n; }
    bool hitExtensionLimit() const          { return hit_extension_limit_; }
    int  getStuckRegionIdx() const          { return stuck_region_idx_; }

    void setMaxNoProgressIters(int n)       { max_no_progress_iters_ = n; }
    bool hitNoProgressLimit() const         { return stuck_no_progress_; }
    int  getNoProgressStuckRegionIdx() const{ return no_progress_stuck_region_idx_; }

    void setRobotInflation(double inflation) { robot_inflation_ = inflation; }

    ob::PlannerStatus solve(const ob::PlannerTerminationCondition &ptc) override
    {
        std::cout << "[GuidedGeoRRT] solve() called" << std::endl;
        checkValidity();
        ob::Goal *goal = pdef_->getGoal().get();
        auto *goal_s = dynamic_cast<ob::GoalSampleableRegion *>(goal);

        while (const ob::State *st = pis_.nextStart())
        {
            auto *motion = new Motion(si_);
            si_->copyState(motion->state, st);
            nn_->add(motion);
        }

        if (nn_->size() == 0)
        {
            OMPL_ERROR("%s: There are no valid initial states!", getName().c_str());
            return ob::PlannerStatus::INVALID_START;
        }

        if (!sampler_)
            sampler_ = si_->allocStateSampler();

        OMPL_INFORM("%s: Starting planning with %u states already in datastructure", getName().c_str(), nn_->size());
        std::cout << "[GuidedGeoRRT] decomp_path size: " << decomp_path.size() << std::endl;
        if (decomp_path.empty()) {
            std::cout << "[GuidedGeoRRT] ERROR: decomp_path is empty!" << std::endl;
            return ob::PlannerStatus::INVALID_START;
        }

        hit_extension_limit_         = false;
        stuck_region_idx_            = -1;
        stuck_no_progress_           = false;
        no_progress_stuck_region_idx_ = -1;

        valid_regions.insert(decomp_path[0]);

        Motion *solution = nullptr;
        Motion *approxsol = nullptr;
        double approxdif = std::numeric_limits<double>::infinity();
        auto *rmotion = new Motion(si_);
        ob::State *rstate = rmotion->state;
        ob::State *xstate = si_->allocState();

        int region_idx = 0;
        int iteration  = 0;

        int iters_without_progress = 0;
        int last_coverage          = 0;
        int tracked_region_idx     = 0;

        while (!ptc)
        {
            ++iteration;
            if (max_extensions_ > 0 && iteration > max_extensions_) {
                std::cout << "[GuidedGeoRRT] hit extension limit=" << max_extensions_
                          << " at region_idx=" << region_idx
                          << " (region=" << decomp_path[region_idx] << ")" << std::endl;
                hit_extension_limit_ = true;
                stuck_region_idx_    = region_idx;
                break;
            }

            // No-progress detection: count consecutive iterations with no coverage increase
            if (region_idx != tracked_region_idx || coverage_map[region_idx] > last_coverage) {
                iters_without_progress = 0;
                last_coverage      = coverage_map[region_idx];
                tracked_region_idx = region_idx;
            } else {
                ++iters_without_progress;
            }
            if (max_no_progress_iters_ > 0 &&
                iters_without_progress >= max_no_progress_iters_ &&
                region_idx > 0)
            {
                std::cout << "[GuidedGeoRRT] no coverage progress at region_idx=" << region_idx
                          << " (region=" << decomp_path[region_idx] << ") after "
                          << iters_without_progress << " iterations" << std::endl;
                stuck_no_progress_            = true;
                no_progress_stuck_region_idx_ = region_idx;
                break;
            }

            bool sample_goal = false;
            if (iteration % 500 == 1)
                std::cout << "[GuidedGeoRRT] iteration=" << iteration
                          << " region_idx=" << region_idx
                          << " nn_size=" << nn_->size()
                          << " coverage=" << coverage_map[region_idx] << std::endl;

            /* sample random state (with goal biasing) */
            // if ((goal_s != nullptr) && rng_.uniform01() < goalBias_ && goal_s->canSample())
            //     goal_s->sampleGoal(rstate);
            // else
            //     sampler_->sampleUniform(rstate);

            // Check if current region is sufficiently covered
            // if (coverage_map[region_idx] > min_coverage) {
            //     // std::cout << "[GuidedGeoRRT] region " << decomp_path[region_idx]
            //     //           << " covered (count=" << coverage_map[region_idx]
            //     //           << "), advancing to region_idx=" << region_idx + 1 << std::endl;
            //     // Move to the next region in the path
            //     region_idx++;
            //     if (region_idx >= (int)decomp_path.size()) {
            //         std::cout << "[GuidedGeoRRT] ERROR: region_idx=" << region_idx
            //                   << " exceeds decomp_path size=" << decomp_path.size() << std::endl;
            //         break;
            //     }
            // }
            if (coverage_map[region_idx] > min_coverage) {
                if (region_idx < (int)decomp_path.size()-1) {
                    std::cout << "[GuidedGeoRRT] region " << decomp_path[region_idx]
                              << " covered, advancing to region_idx=" << region_idx + 1
                              << " (region=" << decomp_path[region_idx + 1] << ")" << std::endl;
                    region_idx++;
                    valid_regions.insert(decomp_path[region_idx]);
                } else {
                    if (!sample_goal)
                        std::cout << "[GuidedGeoRRT] all regions covered, biasing toward goal" << std::endl;
                    sample_goal = true;
                }
            }

            // Periodic goal-bias diagnostics (placed after sample_goal is set)
            if (sample_goal && iteration % 500 == 1) {
                Motion *near = nn_->nearest(rmotion);
                double dist;
                goal->isSatisfied(near->state, &dist);
                int nn_region = decomposition_->locateSubRegion(near->state);
                std::cout << "[GuidedGeoRRT] goal-bias iter=" << iteration
                          << " best_dist=" << approxdif
                          << " nn_dist_to_goal=" << dist
                          << " nn_region=" << nn_region
                          << " (last region=" << decomp_path.back() << ")" << std::endl;
            }

            if (sample_goal && (goal_s != nullptr) && rng_.uniform01() < goalBias_ && goal_s->canSample()) {
                goal_s->sampleGoal(rstate);
            } else {
                // Sample from the current region (inset by robot radius so body stays inside)
                // std::cout << "[GuidedGeoRRT] sampling from region " << decomp_path[region_idx]
                //           << " (region_idx=" << region_idx << ")" << std::endl;
                std::vector<double> coord(decomposition_->getDimension());
                if (region_idx < (int)decomp_path.size() - 1)
                    sampleFromRegionPathAware(decomp_path[region_idx], coord);
                else
                    decomposition_->sampleFromRegion(decomp_path[region_idx], rng_, coord);
                // std::cout << "[GuidedGeoRRT] sampled coord:";
                // for (double v : coord) std::cout << " " << v;
                // std::cout << std::endl;
                decomposition_->sampleFullState(sampler_, coord, rstate);
            }
            

            // si_->copyFromReals(rstate, coord);
            // std::cout << "[GuidedGeoRRT] sampleFullState done, rstate=";
            // si_->getStateSpace()->printState(rstate, std::cout);

            /* find closest state in the tree */
            Motion *nmotion = nn_->nearest(rmotion);
            ob::State *dstate = rstate;

            /* find state to add */
            double d = si_->distance(nmotion->state, rstate);
            // std::cout << "[GuidedGeoRRT] nearest neighbor distance d=" << d << std::endl;
            if (d > maxDistance_)
            {
                si_->getStateSpace()->interpolate(nmotion->state, rstate, maxDistance_ / d, xstate);
                dstate = xstate;
            }

            // si_->getStateSpace()->printState(dstate, std::cout);

            std::vector<Motion *> extension;

            bool motion_valid = si_->checkMotionTest(nmotion->state, dstate, nmotion->step);
            // if (iteration % 500 == 1)
            //     std::cout << "[GuidedGeoRRT] checkMotionTest=" << motion_valid
            //               << " d=" << d << std::endl;

            // std::cout << "[GuidedGeoRRT] motion_valid=" << motion_valid
            //           << " nmotion->step=" << nmotion->step
            //           << std::endl;

            if (motion_valid)
            {
                if (addIntermediateStates_)
                {
                    std::vector<ob::State *> states;
                    const unsigned int count = si_->getStateSpace()->validSegmentCount(nmotion->state, dstate);

                    if (si_->getMotionStates(nmotion->state, dstate, states, count, true, true))
                        si_->freeState(states[0]);

                    // std::cout << "[GuidedGeoRRT] intermediate states count=" << states.size() << std::endl;
                    int skipped_center = 0, skipped_body = 0;
                    int skipped_center_region = -1;
                    for (std::size_t i = 1; i < states.size(); ++i)
                    {
                        auto *motion = new Motion;
                        motion->state = states[i];

                        // Check if state enters any other regions
                        int new_region_idx = decomposition_->locateSubRegion(motion->state);
                        // std::cout << "[GuidedGeoRRT] new_region_idx=" << new_region_idx << "!=" << decomp_path[region_idx] << std::endl;
                        if ((region_idx > 0 && new_region_idx != decomp_path[region_idx - 1]) && new_region_idx != decomp_path[region_idx]) {
                        // if (valid_regions.find(new_region_idx) == valid_regions.end()) {
                            // Remove state from tree and continue
                            skipped_center_region = new_region_idx;
                            si_->freeState(motion->state);
                            delete motion;
                            ++skipped_center;
                            continue;
                        }
                        // Reject if robot body would extend into a non-path region
                        if (region_idx > 0 && region_idx < (int)decomp_path.size() - 1 && !isStateBodyInPath(motion->state, new_region_idx, decomp_path)) {
                            si_->freeState(motion->state);
                            delete motion;
                            ++skipped_body;
                            continue;
                        }
                        // else {
                        //     // Add state to tree and update coverage map
                        //     coverage_map[region_idx]++;
                        // }

                        motion->parent = nmotion;
                        motion->step = nmotion->step + 1;
                        // nn_->add(motion);
                        extension.push_back(motion);

                        nmotion = motion;
                    }
                    if (skipped_center > 0)
                        std::cout << "[GuidedGeoRRT] skipped " << skipped_center
                                  << " states (center check): got region " << skipped_center_region
                                  << ", expected " << decomp_path[region_idx]
                                  << (region_idx > 0 ? " or " + std::to_string(decomp_path[region_idx-1]) : "")
                                  << std::endl;
                    if (skipped_body > 0)
                        std::cout << "[GuidedGeoRRT] skipped " << skipped_body
                                  << " states (body check) in region_idx=" << region_idx
                                  << " (region=" << decomp_path[region_idx] << ")" << std::endl;
                    if (skipped_center == 0 && skipped_body == 0) {
                        // coverage_map[region_idx] += 1;
                        int end_region_idx = decomposition_->locateSubRegion(extension.back()->state);
                        if (end_region_idx == decomp_path[region_idx]) {
                            coverage_map[region_idx]++;
                        }

                        for (std::size_t i = 0; i < extension.size(); ++i)
                        {
                            nn_->add(extension[i]);
                        }
                    }
                }
                else
                {
                    auto *motion = new Motion(si_);
                    si_->copyState(motion->state, dstate);
                    motion->parent = nmotion;
                    motion->step = nmotion->step + 1;
                    nn_->add(motion);

                    nmotion = motion;
                }

                double dist = 0.0;
                bool sat = goal->isSatisfied(nmotion->state, &dist);
                if (sat)
                {
                    std::cout << "[GuidedGeoRRT] goal satisfied at iteration=" << iteration << std::endl;
                    approxdif = dist;
                    solution = nmotion;
                    break;
                }
                if (dist < approxdif)
                {
                    approxdif = dist;
                    approxsol = nmotion;
                }
            }
        }
        std::cout << "[GuidedGeoRRT] planning loop ended at iteration=" << iteration
                  << " solution=" << (solution ? "found" : "not found") << std::endl;

        bool solved = false;
        bool approximate = false;
        if (solution == nullptr)
        {
            solution = approxsol;
            approximate = true;
        }

        if (solution != nullptr)
        {
            lastGoalMotion_ = solution;

            /* construct the solution path */
            std::vector<Motion *> mpath;
            while (solution != nullptr)
            {
                mpath.push_back(solution);
                solution = solution->parent;
            }

            /* set the solution path */
            auto path(std::make_shared<og::PathGeometric>(si_));
            for (int i = mpath.size() - 1; i >= 0; --i)
                path->append(mpath[i]->state);
            pdef_->addSolutionPath(path, approximate, approxdif, getName());
            solved = true;
        }

        si_->freeState(xstate);
        if (rmotion->state != nullptr)
            si_->freeState(rmotion->state);
        delete rmotion;

        OMPL_INFORM("%s: Created %u states", getName().c_str(), nn_->size());

        return {solved, approximate};
    }

protected:
    std::shared_ptr<DecompositionImpl> decomposition_;

    std::vector<int> decomp_path;

    std::set<int> valid_regions;

    std::unordered_map<int, int> coverage_map;

    int min_coverage = 2;

    int  max_extensions_      = 0;
    bool hit_extension_limit_ = false;
    int  stuck_region_idx_    = -1;

    int  max_no_progress_iters_       = 0;
    bool stuck_no_progress_           = false;
    int  no_progress_stuck_region_idx_ = -1;

    double robot_inflation_{0.0};

    // Samples a position from rid, inset from boundaries facing non-path regions but
    // leaving transition sides (shared with path neighbors) open at full extent.
    void sampleFromRegionPathAware(int rid, std::vector<double>& coord) {
        auto cb = decomposition_->getCellBounds(rid);
        int dim = decomposition_->getDimension();
        std::vector<double> lo(dim), hi(dim);
        for (int d = 0; d < dim; ++d) { lo[d] = cb.low[d]; hi[d] = cb.high[d]; }

        if (robot_inflation_ > 0.0) {
            for (int d = 0; d < dim; ++d) { lo[d] += robot_inflation_; hi[d] -= robot_inflation_; }

            std::unordered_set<int> path_set(decomp_path.begin(), decomp_path.end());
            std::vector<int> neighbors;
            decomposition_->getNeighbors(rid, neighbors);
            for (int nbr : neighbors) {
                if (!path_set.count(nbr)) continue;
                auto nb = decomposition_->getCellBounds(nbr);
                for (int d = 0; d < dim; ++d) {
                    if (std::abs(cb.high[d] - nb.low[d]) < 1e-9) hi[d] = cb.high[d];
                    else if (std::abs(nb.high[d] - cb.low[d]) < 1e-9) lo[d] = cb.low[d];
                }
            }
            for (int d = 0; d < dim; ++d)
                if (lo[d] > hi[d]) lo[d] = hi[d] = (cb.low[d] + cb.high[d]) * 0.5;
        }

        coord.resize(dim);
        for (int d = 0; d < dim; ++d)
            coord[d] = rng_.uniformReal(lo[d], hi[d]);
    }

    // Returns false if the robot body (inflated center) would extend into a region
    // that is not part of decomp_path. Boundaries shared with path regions are allowed.
    bool isStateBodyInPath(const ob::State* state, int current_region,
                           const std::vector<int>& path) const {
        if (robot_inflation_ <= 0.0) return true;
        std::vector<double> coord;
        decomposition_->project(state, coord);
        auto cb = decomposition_->getCellBounds(current_region);
        std::unordered_set<int> path_set(path.begin(), path.end());
        std::vector<int> neighbors;
        decomposition_->getNeighbors(current_region, neighbors);
        int dim = decomposition_->getDimension();
        for (int nbr : neighbors) {
            if (path_set.count(nbr)) continue;
            auto nb = decomposition_->getCellBounds(nbr);
            for (int d = 0; d < dim; ++d) {
                bool is_hi_face = std::abs(cb.high[d] - nb.low[d]) < 1e-9;
                bool is_lo_face = std::abs(nb.high[d] - cb.low[d]) < 1e-9;
                if (!is_hi_face && !is_lo_face) continue;
                // Only apply the face inset when the state's position in perpendicular
                // dimensions actually overlaps with this neighbor's extent. If not, the
                // robot body cannot reach this neighbor regardless of the face distance.
                bool perp_overlap = true;
                for (int d2 = 0; d2 < dim; ++d2) {
                    if (d2 == d) continue;
                    if (coord[d2] < nb.low[d2] - robot_inflation_ ||
                        coord[d2] > nb.high[d2] + robot_inflation_) {
                        perp_overlap = false;
                        break;
                    }
                }
                if (!perp_overlap) continue;
                if (is_hi_face && coord[d] > cb.high[d] - robot_inflation_) return false;
                if (is_lo_face && coord[d] < cb.low[d] + robot_inflation_) return false;
            }
        }
        return true;
    }

};

#endif // GUIDED_GEOMETRIC_RRT_H
