#ifndef GUIDED_GEOMETRIC_RRT_CONNECT_H
#define GUIDED_GEOMETRIC_RRT_CONNECT_H

#include <iostream>
#include <unordered_set>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/base/goals/GoalSampleableRegion.h>

#include "utils/decomposition.h"

namespace og = ompl::geometric;
namespace ob = ompl::base;

class GuidedGeometricRRTConnect : public og::RRTConnect {
public:
    GuidedGeometricRRTConnect(const ob::SpaceInformationPtr &si)
        : og::RRTConnect(si) {}

    ~GuidedGeometricRRTConnect() override = default;

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
        auto *goal = dynamic_cast<ob::GoalSampleableRegion *>(pdef_->getGoal().get());

        if (goal == nullptr) {
            OMPL_ERROR("%s: Unknown type of goal", getName().c_str());
            return ob::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
        }

        while (const ob::State *st = pis_.nextStart()) {
            auto *motion = new Motion(si_);
            si_->copyState(motion->state, st);
            motion->root = motion->state;
            tStart_->add(motion);
        }

        if (tStart_->size() == 0) {
            OMPL_ERROR("%s: Motion planning start tree could not be initialized!", getName().c_str());
            return ob::PlannerStatus::INVALID_START;
        }

        if (!goal->couldSample()) {
            OMPL_ERROR("%s: Insufficient states in sampleable goal region", getName().c_str());
            return ob::PlannerStatus::INVALID_GOAL;
        }

        if (decomp_path.empty()) {
            OMPL_ERROR("%s: Decomposition path is empty!", getName().c_str());
            return ob::PlannerStatus::INVALID_START;
        }

        if (!sampler_)
            sampler_ = si_->allocStateSampler();

        OMPL_INFORM("%s: Starting planning with %d states already in datastructure", getName().c_str(),
                    (int)(tStart_->size() + tGoal_->size()));

        const int N = (int)decomp_path.size();
        int region_idx_start = 0;
        int region_idx_goal = 0;
        std::unordered_map<int, int> coverage_map_start;
        std::unordered_map<int, int> coverage_map_goal;

        // All regions on the decomp path — used to filter connect-step states
        const std::unordered_set<int> valid_regions(decomp_path.begin(), decomp_path.end());

        ob::State *xstate = si_->allocState();
        auto *rmotion = new Motion(si_);
        ob::State *rstate = rmotion->state;

        Motion *approxsol = nullptr;
        double approxdif = std::numeric_limits<double>::infinity();
        bool solved = false;

        while (!ptc) {
            // Ensure goal tree has at least one state
            if (tGoal_->size() == 0 || pis_.getSampledGoalsCount() < tGoal_->size() / 2) {
                const ob::State *st = tGoal_->size() == 0 ? pis_.nextGoal(ptc) : pis_.nextGoal();
                if (st != nullptr) {
                    auto *motion = new Motion(si_);
                    si_->copyState(motion->state, st);
                    motion->root = motion->state;
                    tGoal_->add(motion);
                }
                if (tGoal_->size() == 0) {
                    OMPL_ERROR("%s: Unable to sample any valid states for goal tree", getName().c_str());
                    break;
                }
            }

            // Alternate trees
            bool growing_start = startTree_;
            startTree_ = !startTree_;
            TreeData &tree = growing_start ? tStart_ : tGoal_;
            TreeData &otherTree = growing_start ? tGoal_ : tStart_;

            int &region_idx = growing_start ? region_idx_start : region_idx_goal;
            auto &cov_map = growing_start ? coverage_map_start : coverage_map_goal;

            // Advance region once it is sufficiently covered
            if (cov_map[region_idx] > min_coverage && region_idx < N - 1)
                region_idx++;

            // path_idx: index into decomp_path for this tree's current target region
            // path_prev: index of the previously covered region (-1 = none)
            int path_idx  = growing_start ? region_idx       : (N - 1 - region_idx);
            int path_prev = growing_start ? (region_idx - 1) : (N - region_idx);

            // Guided sample from current target region
            std::vector<double> coord(decomposition_->getDimension());
            decomposition_->sampleFromRegion(decomp_path[path_idx], rng_, coord);
            decomposition_->sampleFullState(sampler_, coord, rstate);

            // --- Grow step ---
            Motion *nmotion = tree->nearest(rmotion);
            ob::State *dstate = rstate;

            double d = si_->distance(nmotion->state, rstate);
            if (d > maxDistance_) {
                si_->getStateSpace()->interpolate(nmotion->state, rstate, maxDistance_ / d, xstate);
                if (si_->equalStates(nmotion->state, xstate))
                    continue;
                dstate = xstate;
            }

            // Region filter: new state must be in current or previous allowed region
            int new_region = decomposition_->locateRegion(dstate);
            if (new_region != decomp_path[path_idx] &&
                !(path_prev >= 0 && new_region == decomp_path[path_prev]))
                continue;

            // Directional motion validity: start tree checks forward, goal tree checks backward
            bool valid = growing_start ?
                si_->checkMotion(nmotion->state, dstate) :
                si_->isValid(dstate) && si_->checkMotion(dstate, nmotion->state);

            if (!valid)
                continue;

            // Check every interpolated state along the edge stays within the
            // allowed regions — prevents edges from cutting through wrong cells.
            {
                const unsigned int count =
                    si_->getStateSpace()->validSegmentCount(nmotion->state, dstate);
                std::vector<ob::State *> iStates;
                si_->getMotionStates(nmotion->state, dstate, iStates, count, true, true);
                bool edgeOk = true;
                for (auto *ist : iStates) {
                    int ir = decomposition_->locateRegion(ist);
                    if (ir != decomp_path[path_idx] &&
                        !(path_prev >= 0 && ir == decomp_path[path_prev]))
                        edgeOk = false;
                    si_->freeState(ist);
                }
                if (!edgeOk)
                    continue;
            }

            auto *addedMotion = new Motion(si_);
            si_->copyState(addedMotion->state, dstate);
            addedMotion->parent = nmotion;
            addedMotion->root = nmotion->root;
            tree->add(addedMotion);
            cov_map[region_idx]++;

            // Track closest start-tree node to goal for approximate solution
            if (growing_start) {
                double dist = 0.0;
                goal->isSatisfied(addedMotion->state, &dist);
                if (dist < approxdif) {
                    approxdif = dist;
                    approxsol = addedMotion;
                }
            }

            // --- Connect step: greedily extend otherTree toward addedMotion ---
            // States must stay within the decomp_path region set (looser than the
            // grow step, but prevents connect edges from leaving the planned corridor).
            si_->copyState(rstate, addedMotion->state);
            bool other_is_start = !growing_start;
            Motion *connectMotion = nullptr;
            GrowState gsc = ADVANCED;

            while (gsc == ADVANCED) {
                Motion *cnmotion = otherTree->nearest(rmotion);
                ob::State *cdstate = rstate;
                double cd = si_->distance(cnmotion->state, rstate);
                bool creached = true;

                if (cd > maxDistance_) {
                    si_->getStateSpace()->interpolate(cnmotion->state, rstate, maxDistance_ / cd, xstate);
                    if (si_->equalStates(cnmotion->state, xstate)) {
                        gsc = TRAPPED;
                        break;
                    }
                    cdstate = xstate;
                    creached = false;
                }

                // Endpoint must be in a decomp_path region
                if (valid_regions.find(decomposition_->locateRegion(cdstate)) == valid_regions.end()) {
                    gsc = TRAPPED;
                    break;
                }

                // Every interpolated state along the connect edge must also stay in-path
                {
                    const unsigned int count =
                        si_->getStateSpace()->validSegmentCount(cnmotion->state, cdstate);
                    std::vector<ob::State *> iStates;
                    si_->getMotionStates(cnmotion->state, cdstate, iStates, count, true, true);
                    bool edgeOk = true;
                    for (auto *ist : iStates) {
                        if (valid_regions.find(decomposition_->locateRegion(ist)) == valid_regions.end())
                            edgeOk = false;
                        si_->freeState(ist);
                    }
                    if (!edgeOk) {
                        gsc = TRAPPED;
                        break;
                    }
                }

                bool cvalid = other_is_start ?
                    si_->checkMotion(cnmotion->state, cdstate) :
                    si_->isValid(cdstate) && si_->checkMotion(cdstate, cnmotion->state);

                if (!cvalid) {
                    gsc = TRAPPED;
                    break;
                }

                auto *cmotion = new Motion(si_);
                si_->copyState(cmotion->state, cdstate);
                cmotion->parent = cnmotion;
                cmotion->root = cnmotion->root;
                otherTree->add(cmotion);
                connectMotion = cmotion;
                gsc = creached ? REACHED : ADVANCED;
            }

            // Update tracked distance between trees
            const double newDist = tree->getDistanceFunction()(addedMotion, otherTree->nearest(addedMotion));
            if (newDist < distanceBetweenTrees_)
                distanceBetweenTrees_ = newDist;

            if (gsc == REACHED && connectMotion != nullptr) {
                Motion *startMotion = growing_start ? addedMotion : connectMotion;
                Motion *goalMotion  = growing_start ? connectMotion : addedMotion;

                if (!goal->isStartGoalPairValid(startMotion->root, goalMotion->root))
                    continue;

                // Step back one node to avoid a duplicate state at the connection point
                if (startMotion->parent != nullptr)
                    startMotion = startMotion->parent;
                else
                    goalMotion = goalMotion->parent;

                connectionPoint_ = std::make_pair(startMotion->state, goalMotion->state);

                // Backtrack start side
                std::vector<Motion *> mpath1;
                for (Motion *sol = startMotion; sol != nullptr; sol = sol->parent)
                    mpath1.push_back(sol);

                // Backtrack goal side
                std::vector<Motion *> mpath2;
                for (Motion *sol = goalMotion; sol != nullptr; sol = sol->parent)
                    mpath2.push_back(sol);

                auto path = std::make_shared<og::PathGeometric>(si_);
                path->getStates().reserve(mpath1.size() + mpath2.size());
                for (int i = (int)mpath1.size() - 1; i >= 0; --i)
                    path->append(mpath1[i]->state);
                for (auto *m : mpath2)
                    path->append(m->state);

                pdef_->addSolutionPath(path, false, 0.0, getName());
                solved = true;

                std::cout << "[GuidedGeoRRTConnect] solution found:"
                          << " start region_idx=" << region_idx_start
                          << " goal region_idx=" << region_idx_goal
                          << " start_tree_size=" << tStart_->size()
                          << " goal_tree_size=" << tGoal_->size() << std::endl;
                break;
            }
        }

        si_->freeState(xstate);
        si_->freeState(rstate);
        delete rmotion;

        OMPL_INFORM("%s: Created %u states (%u start + %u goal)", getName().c_str(),
                    tStart_->size() + tGoal_->size(), tStart_->size(), tGoal_->size());

        if (approxsol != nullptr && !solved) {
            std::vector<Motion *> mpath;
            for (Motion *sol = approxsol; sol != nullptr; sol = sol->parent)
                mpath.push_back(sol);
            auto path = std::make_shared<og::PathGeometric>(si_);
            for (int i = (int)mpath.size() - 1; i >= 0; --i)
                path->append(mpath[i]->state);
            pdef_->addSolutionPath(path, true, approxdif, getName());
            return ob::PlannerStatus::APPROXIMATE_SOLUTION;
        }

        return solved ? ob::PlannerStatus::EXACT_SOLUTION : ob::PlannerStatus::TIMEOUT;
    }

protected:
    std::shared_ptr<DecompositionImpl> decomposition_;
    std::vector<int> decomp_path;
    int min_coverage = 10;
};

#endif // GUIDED_GEOMETRIC_RRT_CONNECT_H
