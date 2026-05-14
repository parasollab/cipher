#ifndef GUIDED_GEOMETRIC_RRT_H
#define GUIDED_GEOMETRIC_RRT_H

#include <iostream>
#include <set>
#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>


#include "utils/decomposition.h"

namespace oc = ompl::control;
namespace ob = ompl::base;

class GuidedControlRRT : public oc::RRT {
public:
    GuidedControlRRT(const oc::SpaceInformationPtr &si)
        : oc::RRT(si) {}

    ~GuidedControlRRT() override = default;

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
        checkValidity();
        ob::Goal *goal = pdef_->getGoal().get();
        auto *goal_s = dynamic_cast<ob::GoalSampleableRegion *>(goal);

        while (const ob::State *st = pis_.nextStart())
        {
            auto *motion = new Motion(siC_);
            si_->copyState(motion->state, st);
            siC_->nullControl(motion->control);
            nn_->add(motion);
        }

        if (nn_->size() == 0)
        {
            OMPL_ERROR("%s: There are no valid initial states!", getName().c_str());
            return ob::PlannerStatus::INVALID_START;
        }

        if (!sampler_)
            sampler_ = si_->allocStateSampler();
        if (!controlSampler_)
            controlSampler_ = siC_->allocDirectedControlSampler();

        OMPL_INFORM("%s: Starting planning with %u states already in datastructure", getName().c_str(), nn_->size());

        Motion *solution = nullptr;
        Motion *approxsol = nullptr;
        double approxdif = std::numeric_limits<double>::infinity();

        auto *rmotion = new Motion(siC_);
        ob::State *rstate = rmotion->state;
        oc::Control *rctrl = rmotion->control;
        ob::State *xstate = si_->allocState();

        valid_regions.insert(decomp_path[0]);

        int region_idx = 0;
        int iteration = 0;
        bool sample_goal = false;

        while (ptc == false)
        {
            ++iteration;

            if (coverage_map[region_idx] > min_coverage) {
                if (region_idx < (int)decomp_path.size()-1) {
                    std::cout << "[GuidedControlRRT] region " << decomp_path[region_idx]
                              << " covered, advancing to region_idx=" << region_idx + 1
                              << " (region=" << decomp_path[region_idx + 1] << ")" << std::endl;
                    region_idx++;
                    valid_regions.insert(decomp_path[region_idx]);
                } else {
                    if (!sample_goal)
                        std::cout << "[GuidedControlRRT] all regions covered, biasing toward goal" << std::endl;
                    sample_goal = true;
                }
            }

            if (sample_goal && (goal_s != nullptr) && rng_.uniform01() < goalBias_ && goal_s->canSample()) {
                goal_s->sampleGoal(rstate);
            } else {
                std::vector<double> coord(decomposition_->getDimension());
                decomposition_->sampleFromRegion(decomp_path[region_idx], rng_, coord);
                decomposition_->sampleFullState(sampler_, coord, rstate);
            }

            /* find closest state in the tree */
            Motion *nmotion = nn_->nearest(rmotion);

            /* get time of nmotion */
            unsigned int nsteps = 0;
            Motion* nmotionCpy = nmotion;
            while (nmotionCpy)
            {
                nsteps += nmotionCpy->steps;
                nmotionCpy = nmotionCpy->parent;
            }
            /* sample a random control that attempts to go towards the random state, and also sample a control duration */
            unsigned int cd = controlSampler_->sampleToTest(rctrl, nmotion->control, nmotion->state, rmotion->state, nsteps);

            std::cout << "[GuidedControlRRT] iteration=" << iteration
                      << " region_idx=" << region_idx
                      << " nmotion steps=" << nmotion->steps
                      << " sampled control duration cd=" << cd
                      << std::endl;
        
            std::vector<Motion *> extension;

            if (addIntermediateStates_)
            {
                // this code is contributed by Jennifer Barry
                std::vector<ob::State *> pstates;
                cd = siC_->propagateWhileValid(nmotion->state, rctrl, cd, pstates, true);

                // std::cout << "[GuidedControlRRT] propagateWhileValid returned cd=" << cd
                //           << " valid intermediate states=" << pstates.size() << std::endl;

                if (cd >= siC_->getMinControlDuration())
                {

                    bool ends_in_region = false;
                    std::cout << "EndRegion: " << decomposition_->locateSubRegion(pstates.back()) << "!=" << decomp_path[region_idx] << std::endl;
                    if (decomposition_->locateSubRegion(pstates.back()) == decomp_path[region_idx]) {
                        // coverage_map[region_idx] += 1;
                        ends_in_region = true;
                    }

                    int skipped = 0;
                    Motion *lastmotion = nmotion;
                    bool solved = false;
                    size_t p = 0;
                    for (; p < pstates.size(); ++p)
                    {
                        /* create a motion */
                        auto *motion = new Motion();
                        motion->state = pstates[p];

                        // Check if state enters any other regions. If so, pstates are invalid
                        int new_region_idx = decomposition_->locateSubRegion(motion->state);
                        // std::cout << "[GuidedControlRRT] new_region_idx=" << new_region_idx << "!=" << decomp_path[region_idx] << std::endl;
                        
                        // If state enters a different region than the current one or the previous one (if > 0), skip it
                        if ((region_idx > 0 && new_region_idx != decomp_path[region_idx - 1]) && new_region_idx != decomp_path[region_idx]) {
                            si_->freeState(motion->state);
                            delete motion;
                            ++skipped;
                            continue;
                        }

                        // we need multiple copies of rctrl
                        motion->control = siC_->allocControl();
                        siC_->copyControl(motion->control, rctrl);
                        motion->steps = 1;
                        motion->parent = lastmotion;
                        lastmotion = motion;
                        // nn_->add(motion);
                        extension.push_back(motion);
                        double dist = 0.0;
                        solved = goal->isSatisfied(motion->state, &dist);
                        if (solved)
                        {
                            approxdif = dist;
                            solution = motion;
                            break;
                        }
                        if (dist < approxdif)
                        {
                            approxdif = dist;
                            approxsol = motion;
                        }
                    }

                    // if (skipped > 0)
                    //     std::cout << "[GuidedGeoRRT] skipped " << skipped
                    //               << " intermediate states (wrong region)" << std::endl;

                    if (skipped == 0) {
                        for (std::size_t i = 0; i < extension.size(); ++i)
                        {
                            nn_->add(extension[i]);
                        }

                        if (ends_in_region)
                        {
                            coverage_map[region_idx] += 1;
                        }
                    }

                    // free any states after we hit the goal
                    while (++p < pstates.size())
                        si_->freeState(pstates[p]);
                    if (solved)
                        break;
                }
                else
                    for (auto &pstate : pstates)
                        si_->freeState(pstate);
            }
            else
            {
                if (cd >= siC_->getMinControlDuration())
                {
                    /* create a motion */
                    auto *motion = new Motion(siC_);
                    si_->copyState(motion->state, rmotion->state);
                    siC_->copyControl(motion->control, rctrl);
                    motion->steps = cd;
                    motion->parent = nmotion;

                    nn_->add(motion);
                    double dist = 0.0;
                    bool solv = goal->isSatisfied(motion->state, &dist);
                    if (solv)
                    {
                        approxdif = dist;
                        solution = motion;
                        break;
                    }
                    if (dist < approxdif)
                    {
                        approxdif = dist;
                        approxsol = motion;
                    }
                }
            }
        }

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
            auto path(std::make_shared<oc::PathControl>(si_));
            for (int i = mpath.size() - 1; i >= 0; --i)
                if (mpath[i]->parent)
                    path->append(mpath[i]->state, mpath[i]->control, mpath[i]->steps * siC_->getPropagationStepSize());
                else
                    path->append(mpath[i]->state);
            solved = true;
            pdef_->addSolutionPath(path, approximate, approxdif, getName());
        }

        if (rmotion->state)
            si_->freeState(rmotion->state);
        if (rmotion->control)
            siC_->freeControl(rmotion->control);
        delete rmotion;
        si_->freeState(xstate);

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
