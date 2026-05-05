#ifndef GUIDED_GEOMETRIC_RRT_H
#define GUIDED_GEOMETRIC_RRT_H

#include <iostream>
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

        valid_regions.insert(decomp_path[0]);

        Motion *solution = nullptr;
        Motion *approxsol = nullptr;
        double approxdif = std::numeric_limits<double>::infinity();
        auto *rmotion = new Motion(si_);
        ob::State *rstate = rmotion->state;
        ob::State *xstate = si_->allocState();

        int region_idx = 0;
        int iteration = 0;

        while (!ptc)
        {
            ++iteration;
            bool sample_goal = false;
            // if (iteration % 500 == 1)
            //     std::cout << "[GuidedGeoRRT] iteration=" << iteration
            //               << " region_idx=" << region_idx
            //               << " nn_size=" << nn_->size()
            //               << " coverage=" << coverage_map[region_idx] << std::endl;

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
                    region_idx++;
                    valid_regions.insert(decomp_path[region_idx]);
                } else {
                    sample_goal = true;
                }
            }

            if (sample_goal && (goal_s != nullptr) && rng_.uniform01() < goalBias_ && goal_s->canSample()) {
                goal_s->sampleGoal(rstate);
            } else {
                // Sample from the current region
                // std::cout << "[GuidedGeoRRT] sampling from region " << decomp_path[region_idx]
                //           << " (region_idx=" << region_idx << ")" << std::endl;
                std::vector<double> coord(decomposition_->getDimension());
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
                    int skipped = 0;
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
                            si_->freeState(motion->state);
                            delete motion;
                            ++skipped;
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
                    if (skipped > 0)
                        // skipped = 10;
                        std::cout << "[GuidedGeoRRT] skipped " << skipped
                                  << " intermediate states (wrong region)" << std::endl;
                    else {
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

};

#endif // GUIDED_GEOMETRIC_RRT_H
