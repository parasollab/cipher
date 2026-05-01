/*********************************************************************
* Kinodynamic Adaptive Robot Coordination (K-ARC)
*
* A hybrid approach for multi-robot kinodynamic planning that uses
* trajectory optimization (Crocoddyl/FDDP) for individual segments and
* sampling-based methods (kinodynamic RRT) for conflict resolution.
*
* Based on the paper:
* "K-ARC: Adaptive Robot Coordination for Multi-Robot Kinodynamic Planning"
* by Qin et al., arXiv 2501.01559v1, 2025
*********************************************************************/

// OMPL - Base
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/State.h>
#include <ompl/base/goals/GoalRegion.h>

// OMPL - Geometric (for initial kinematic paths)
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/PathGeometric.h>

// OMPL - Control (for kinodynamic planning)
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/StatePropagator.h>
#include <ompl/control/PathControl.h>
#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

// OMPL - Utilities
#include <ompl/util/RandomNumbers.h>

// Crocoddyl trajectory optimization
#include <crocoddyl/core/fwd.hpp>
#include <crocoddyl/core/action-base.hpp>
#include <crocoddyl/core/numdiff/action.hpp>
#include <crocoddyl/core/states/euclidean.hpp>
#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/solvers/fddp.hpp>

// Eigen
#include <Eigen/Dense>

// FCL
#include <fcl/fcl.h>

// YAML
#include <yaml-cpp/yaml.h>

// Boost
#include <boost/program_options.hpp>

// Project headers
#include "robots.h"
#include "fclStateValidityChecker.hpp"
#include "robotStatePropagator.hpp"

// Standard library
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>

namespace ob = ompl::base;
namespace oc = ompl::control;
namespace og = ompl::geometric;
namespace po = boost::program_options;

// ============================================================================
// Configuration
// ============================================================================

struct KARCConfig {
    double time_limit = 60.0;
    double goal_threshold = 1.5;
    int seed = -1;

    // Segmentation
    int num_segments = 5;

    // FDDP trajectory optimizer
    double dt = 0.1;            // timestep matching robot's internal dt
    int T_horizon = 50;         // steps per segment
    double beta_control = 0.01; // control effort weight
    double goal_weight = 100.0; // terminal goal weight
    int fddp_max_iter = 100;

    // Subproblem solver timeouts
    double prioritized_traj_opt_timeout = 5.0;
    double decoupled_kino_rrt_timeout = 10.0;
    double composite_kino_rrt_timeout = 20.0;

    // Termination
    int max_conflicts_resolved = 100;

    // Inter-robot separation used as soft constraint in coupled FDDP
    double collision_margin = 0.5;
    double obstacle_weight = 500.0;  // penalty weight for static obstacle avoidance

    // RRT fallback parameters
    double propagation_step_size = 0.1;
    int control_duration_min = 1;
    int control_duration_max = 10;
};

// ============================================================================
// Data Structures
// ============================================================================

struct Conflict {
    size_t robot_i = 0;
    size_t robot_j = 0;
    int timestep = 0;
};

struct Subproblem {
    std::vector<size_t> robot_indices;
    std::vector<ob::State*> local_starts;
    std::vector<ob::State*> local_goals;
    std::vector<double> env_min;
    std::vector<double> env_max;
    std::vector<fcl::CollisionObjectf*> local_obstacles;
    int segment_idx = 0;

    // AdaptSubProblem support (paper Algorithm 2, line 10)
    bool adapted = false;
    std::vector<ob::State*> expanded_starts;  // previous segment's starts
    std::vector<ob::State*> expanded_goals;   // next segment's goals

    void freeStates(const std::vector<std::shared_ptr<Robot>>& robots) {
        for (size_t k = 0; k < robot_indices.size(); ++k) {
            size_t ri = robot_indices[k];
            auto si = robots[ri]->getSpaceInformation();
            if (k < local_starts.size() && local_starts[k])
                si->freeState(local_starts[k]);
            if (k < local_goals.size() && local_goals[k])
                si->freeState(local_goals[k]);
            if (k < expanded_starts.size() && expanded_starts[k])
                si->freeState(expanded_starts[k]);
            if (k < expanded_goals.size() && expanded_goals[k])
                si->freeState(expanded_goals[k]);
        }
        local_starts.clear();
        local_goals.clear();
        expanded_starts.clear();
        expanded_goals.clear();
    }
};

struct SubproblemSolution {
    bool solved = false;
    double planning_time = 0.0;
    // Per-robot-in-subproblem: state sequence (as Eigen vectors)
    std::vector<std::vector<Eigen::VectorXd>> state_seqs;
    std::string method_used = "none";
};

struct KARCResult {
    bool solved = false;
    double planning_time = 0.0;
    // Final per-robot state sequences (all segments concatenated)
    std::vector<std::vector<Eigen::VectorXd>> paths;
    int num_conflicts_found = 0;
    int num_conflicts_resolved = 0;
    std::vector<std::string> methods_used;
};

struct RobotSpec {
    std::string type;
    std::vector<double> start;
    std::vector<double> goal;
};

struct PlanningProblem {
    std::vector<double> env_min;
    std::vector<double> env_max;
    std::vector<fcl::CollisionObjectf*> obstacles;
    std::vector<RobotSpec> robots;
};

// ============================================================================
// OMPL <-> Eigen Helpers
// ============================================================================

static int getStateDim(const std::shared_ptr<ob::SpaceInformation>& si) {
    auto* state = si->allocState();
    std::vector<double> reals;
    si->getStateSpace()->copyToReals(reals, state);
    si->freeState(state);
    return static_cast<int>(reals.size());
}

static int getControlDim(const std::shared_ptr<oc::SpaceInformation>& csi) {
    return static_cast<int>(
        csi->getControlSpace()->as<oc::RealVectorControlSpace>()->getDimension());
}

static Eigen::VectorXd omplToEigen(const ob::State* state,
                                   const std::shared_ptr<ob::SpaceInformation>& si) {
    std::vector<double> reals;
    si->getStateSpace()->copyToReals(reals, state);
    return Eigen::Map<Eigen::VectorXd>(reals.data(), reals.size());
}

static void eigenToOmpl(const Eigen::VectorXd& x, ob::State* state,
                        const std::shared_ptr<ob::SpaceInformation>& si) {
    std::vector<double> reals(x.data(), x.data() + x.size());
    si->getStateSpace()->copyFromReals(state, reals);
}

static void eigenToControl(const Eigen::VectorXd& u, oc::Control* ctrl) {
    auto* rv = ctrl->as<oc::RealVectorControlSpace::ControlType>();
    for (int i = 0; i < u.size(); ++i) rv->values[i] = u[i];
}

// ============================================================================
// Crocoddyl: Running Action Model (wraps Robot::propagate)
// ============================================================================

class RobotRunningModel : public crocoddyl::ActionModelAbstract {
public:
    RobotRunningModel(std::shared_ptr<Robot> robot,
                      std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng,
                      double dt, double beta,
                      double obstacle_weight = 0.0,
                      double obstacle_margin = 0.0)
        : crocoddyl::ActionModelAbstract(
              std::make_shared<crocoddyl::StateVector>(computeNx(robot)),
              computeNu(robot),
              computeNu(robot))  // nr = nu (control residual)
        , robot_(robot)
        , col_mng_(col_mng)
        , dt_(dt)
        , beta_(beta)
        , obstacle_weight_(obstacle_weight)
        , obstacle_margin_(obstacle_margin)
    {
        csi_ = robot->getSpaceInformation();
        si_ = std::dynamic_pointer_cast<ob::SpaceInformation>(csi_);
        state_in_ = csi_->allocState();
        state_out_ = csi_->allocState();
        ctrl_tmp_ = csi_->allocControl();
        nu_val_ = computeNu(robot);
    }

    ~RobotRunningModel() override {
        csi_->freeState(state_in_);
        csi_->freeState(state_out_);
        csi_->freeControl(ctrl_tmp_);
    }

    void calc(const std::shared_ptr<crocoddyl::ActionDataAbstract>& data,
              const Eigen::Ref<const Eigen::VectorXd>& x,
              const Eigen::Ref<const Eigen::VectorXd>& u) override {
        // Convert Eigen → OMPL
        std::vector<double> xv(x.data(), x.data() + x.size());
        csi_->getStateSpace()->copyFromReals(state_in_, xv);
        eigenToControl(u, ctrl_tmp_);

        // Propagate dynamics
        robot_->propagate(state_in_, ctrl_tmp_, dt_, state_out_);

        // Convert result → xnext
        std::vector<double> xnext_v;
        csi_->getStateSpace()->copyToReals(xnext_v, state_out_);
        data->xnext = Eigen::Map<Eigen::VectorXd>(xnext_v.data(), xnext_v.size());

        // Cost: control effort residual + per-step time cost (matches paper: J = β||u||² + Δt)
        data->r = std::sqrt(beta_) * u;
        data->cost = 0.5 * data->r.squaredNorm() + dt_;

        // Obstacle avoidance penalty: paper constraint (5), c_{i,k} ∈ W_free
        if (obstacle_weight_ > 0.0) {
            for (size_t part = 0; part < robot_->numParts(); ++part) {
                const auto& tf = robot_->getTransform(state_in_, part);
                fcl::CollisionObjectf robot_co(robot_->getCollisionGeometry(part));
                robot_co.setTranslation(tf.translation());
                robot_co.setRotation(tf.rotation());
                robot_co.computeAABB();

                fcl::DefaultDistanceData<float> dist_data;
                dist_data.request.enable_signed_distance = true;
                col_mng_->distance(&robot_co, &dist_data,
                                   fcl::DefaultDistanceFunction<float>);

                double d = static_cast<double>(dist_data.result.min_distance);
                if (d < obstacle_margin_) {
                    double violation = obstacle_margin_ - d;
                    data->cost += 0.5 * obstacle_weight_ * violation * violation;
                }
            }
        }
    }

    void calc(const std::shared_ptr<crocoddyl::ActionDataAbstract>& data,
              const Eigen::Ref<const Eigen::VectorXd>& x) override {
        data->xnext = x;
        data->r.setZero();
        data->cost = 0.0;
    }

    void calcDiff(const std::shared_ptr<crocoddyl::ActionDataAbstract>& data,
                  const Eigen::Ref<const Eigen::VectorXd>& x,
                  const Eigen::Ref<const Eigen::VectorXd>& u) override {
        // Left zero — ActionModelNumDiff computes these numerically
        data->Lx.setZero();
        data->Lu.setZero();
        data->Lxx.setZero();
        data->Luu.setZero();
        data->Lxu.setZero();
        data->Fx.setIdentity();
        data->Fu.setZero();
    }

    void calcDiff(const std::shared_ptr<crocoddyl::ActionDataAbstract>& data,
                  const Eigen::Ref<const Eigen::VectorXd>& x) override {
        data->Lx.setZero();
        data->Lxx.setZero();
        data->Fx.setIdentity();
    }

    std::shared_ptr<crocoddyl::ActionDataAbstract> createData() override {
        return std::make_shared<crocoddyl::ActionDataAbstract>(this);
    }

    // Required stubs for Crocoddyl's scalar-casting machinery
    std::shared_ptr<crocoddyl::ActionModelBase> cloneAsDouble() const override {
        return nullptr;
    }
    std::shared_ptr<crocoddyl::ActionModelBase> cloneAsFloat() const override {
        return nullptr;
    }

    static int computeNx(const std::shared_ptr<Robot>& robot) {
        auto si = robot->getSpaceInformation();
        auto* st = si->allocState();
        std::vector<double> reals;
        si->getStateSpace()->copyToReals(reals, st);
        si->freeState(st);
        return static_cast<int>(reals.size());
    }

    static int computeNu(const std::shared_ptr<Robot>& robot) {
        return static_cast<int>(
            robot->getSpaceInformation()->getControlSpace()
                ->as<oc::RealVectorControlSpace>()->getDimension());
    }

private:
    std::shared_ptr<Robot> robot_;
    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng_;
    std::shared_ptr<oc::SpaceInformation> csi_;
    std::shared_ptr<ob::SpaceInformation> si_;
    double dt_, beta_;
    double obstacle_weight_, obstacle_margin_;
    int nu_val_;
    mutable ob::State* state_in_;
    mutable ob::State* state_out_;
    mutable oc::Control* ctrl_tmp_;
};

// ============================================================================
// Crocoddyl: Running Action Model with inter-robot distance penalty
// Used by prioritised trajectory optimisation to constrain robot r_k's path
// away from already-planned robots' paths (paper Section IV-C).
// ============================================================================

class RobotRunningModelCoupled : public RobotRunningModel {
public:
    // other_x: already-planned robot's state at this specific timestep k
    RobotRunningModelCoupled(std::shared_ptr<Robot> robot,
                             std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng,
                             double dt, double beta,
                             double obstacle_weight,
                             double obstacle_margin,
                             const Eigen::VectorXd& other_x,
                             double collision_weight,
                             double collision_margin)
        : RobotRunningModel(robot, col_mng, dt, beta, obstacle_weight, obstacle_margin)
        , other_x_(other_x)
        , col_weight_(collision_weight)
        , margin_(collision_margin)
    {}

    void calc(const std::shared_ptr<crocoddyl::ActionDataAbstract>& data,
              const Eigen::Ref<const Eigen::VectorXd>& x,
              const Eigen::Ref<const Eigen::VectorXd>& u) override {
        RobotRunningModel::calc(data, x, u);
        // Positional separation (first two state components = x,y position)
        int pos_dim = std::min(2, static_cast<int>(x.size()));
        double dx = (x.head(pos_dim) - other_x_.head(pos_dim)).norm();
        if (dx < margin_) {
            double violation = margin_ - dx;
            data->cost += 0.5 * col_weight_ * violation * violation;
        }
    }

    std::shared_ptr<crocoddyl::ActionModelBase> cloneAsDouble() const override { return nullptr; }
    std::shared_ptr<crocoddyl::ActionModelBase> cloneAsFloat() const override { return nullptr; }

private:
    Eigen::VectorXd other_x_;
    double col_weight_;
    double margin_;
};

// ============================================================================
// Crocoddyl: Terminal Action Model (goal reaching)
// ============================================================================

class RobotTerminalModel : public crocoddyl::ActionModelAbstract {
public:
    RobotTerminalModel(std::shared_ptr<Robot> robot,
                       const Eigen::VectorXd& x_goal,
                       double goal_weight)
        : crocoddyl::ActionModelAbstract(
              std::make_shared<crocoddyl::StateVector>(
                  RobotRunningModel::computeNx(robot)),
              0,   // nu = 0 (no control at terminal)
              RobotRunningModel::computeNx(robot))  // nr = nx
        , x_goal_(x_goal)
        , goal_weight_(goal_weight)
    {}

    void calc(const std::shared_ptr<crocoddyl::ActionDataAbstract>& data,
              const Eigen::Ref<const Eigen::VectorXd>& x,
              const Eigen::Ref<const Eigen::VectorXd>& /*u*/) override {
        calc(data, x);
    }

    void calc(const std::shared_ptr<crocoddyl::ActionDataAbstract>& data,
              const Eigen::Ref<const Eigen::VectorXd>& x) override {
        data->xnext = x;
        data->r = std::sqrt(goal_weight_) * (x - x_goal_);
        data->cost = 0.5 * data->r.squaredNorm();
    }

    void calcDiff(const std::shared_ptr<crocoddyl::ActionDataAbstract>& data,
                  const Eigen::Ref<const Eigen::VectorXd>& x,
                  const Eigen::Ref<const Eigen::VectorXd>& u) override {
        calcDiff(data, x);
    }

    void calcDiff(const std::shared_ptr<crocoddyl::ActionDataAbstract>& data,
                  const Eigen::Ref<const Eigen::VectorXd>& x) override {
        data->Lx.setZero();
        data->Lxx.setZero();
        data->Fx.setIdentity();
    }

    std::shared_ptr<crocoddyl::ActionDataAbstract> createData() override {
        return std::make_shared<crocoddyl::ActionDataAbstract>(this);
    }

    // Required stubs for Crocoddyl's scalar-casting machinery
    std::shared_ptr<crocoddyl::ActionModelBase> cloneAsDouble() const override {
        return nullptr;
    }
    std::shared_ptr<crocoddyl::ActionModelBase> cloneAsFloat() const override {
        return nullptr;
    }

private:
    Eigen::VectorXd x_goal_;
    double goal_weight_;
};

// ============================================================================
// Validity Checker for kinodynamic decoupled subproblem solver
// (checks static obstacles + dynamic robot obstacles from previously planned paths)
// ============================================================================

class KinoDecoupledValidityChecker : public ob::StateValidityChecker {
public:
    KinoDecoupledValidityChecker(
        const ob::SpaceInformationPtr& si,
        const std::shared_ptr<fcl::BroadPhaseCollisionManagerf>& col_mng,
        const std::shared_ptr<Robot>& robot,
        const std::vector<std::shared_ptr<Robot>>& other_robots,
        const std::vector<std::vector<Eigen::VectorXd>>& other_seqs)
        : ob::StateValidityChecker(si)
        , col_mng_(col_mng)
        , robot_(robot)
        , other_robots_(other_robots)
        , other_seqs_(other_seqs)
    {}

    bool isValid(const ob::State* state) const override {
        if (!si_->satisfiesBounds(state)) return false;

        // Obstacle check
        for (size_t part = 0; part < robot_->numParts(); ++part) {
            const auto& tf = robot_->getTransform(state, part);
            fcl::CollisionObjectf co(robot_->getCollisionGeometry(part));
            co.setTranslation(tf.translation());
            co.setRotation(tf.rotation());
            co.computeAABB();
            fcl::DefaultCollisionData<float> cd;
            col_mng_->collide(&co, &cd, fcl::DefaultCollisionFunction<float>);
            if (cd.result.isCollision()) return false;
        }

        // Dynamic obstacle check against all timesteps of other robots
        for (size_t oi = 0; oi < other_robots_.size(); ++oi) {
            const auto& other = other_robots_[oi];
            const auto& other_si = other->getSpaceInformation();
            auto* other_state = other_si->allocState();

            for (const auto& xo : other_seqs_[oi]) {
                std::vector<double> rv(xo.data(), xo.data() + xo.size());
                other_si->getStateSpace()->copyFromReals(other_state, rv);

                for (size_t pi = 0; pi < robot_->numParts(); ++pi) {
                    for (size_t pj = 0; pj < other->numParts(); ++pj) {
                        const auto& tfi = robot_->getTransform(state, pi);
                        const auto& tfj = other->getTransform(other_state, pj);

                        fcl::CollisionObjectf coi(robot_->getCollisionGeometry(pi));
                        coi.setTranslation(tfi.translation());
                        coi.setRotation(tfi.rotation());
                        coi.computeAABB();

                        fcl::CollisionObjectf coj(other->getCollisionGeometry(pj));
                        coj.setTranslation(tfj.translation());
                        coj.setRotation(tfj.rotation());
                        coj.computeAABB();

                        fcl::CollisionRequestf req;
                        fcl::CollisionResultf res;
                        fcl::collide(&coi, &coj, req, res);
                        if (res.isCollision()) {
                            other_si->freeState(other_state);
                            return false;
                        }
                    }
                }
            }
            other_si->freeState(other_state);
        }
        return true;
    }

private:
    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng_;
    std::shared_ptr<Robot> robot_;
    std::vector<std::shared_ptr<Robot>> other_robots_;
    std::vector<std::vector<Eigen::VectorXd>> other_seqs_;
};

// ============================================================================
// Compound Validity Checker (for composite kinodynamic RRT)
// ============================================================================

class KinoCompoundValidityChecker : public ob::StateValidityChecker {
public:
    KinoCompoundValidityChecker(
        const ob::SpaceInformationPtr& si,
        const std::shared_ptr<fcl::BroadPhaseCollisionManagerf>& col_mng,
        const std::vector<std::shared_ptr<Robot>>& robots)
        : ob::StateValidityChecker(si)
        , col_mng_(col_mng)
        , robots_(robots)
    {}

    bool isValid(const ob::State* state) const override {
        if (!si_->satisfiesBounds(state)) return false;
        auto compound = state->as<ob::CompoundState>();

        for (size_t i = 0; i < robots_.size(); ++i) {
            for (size_t part = 0; part < robots_[i]->numParts(); ++part) {
                const auto& tf = robots_[i]->getTransform(compound->components[i], part);
                fcl::CollisionObjectf co(robots_[i]->getCollisionGeometry(part));
                co.setTranslation(tf.translation());
                co.setRotation(tf.rotation());
                co.computeAABB();
                fcl::DefaultCollisionData<float> cd;
                col_mng_->collide(&co, &cd, fcl::DefaultCollisionFunction<float>);
                if (cd.result.isCollision()) return false;
            }
        }

        for (size_t i = 0; i < robots_.size(); ++i) {
            for (size_t j = i + 1; j < robots_.size(); ++j) {
                for (size_t pi = 0; pi < robots_[i]->numParts(); ++pi) {
                    for (size_t pj = 0; pj < robots_[j]->numParts(); ++pj) {
                        const auto& tfi = robots_[i]->getTransform(compound->components[i], pi);
                        const auto& tfj = robots_[j]->getTransform(compound->components[j], pj);

                        fcl::CollisionObjectf coi(robots_[i]->getCollisionGeometry(pi));
                        coi.setTranslation(tfi.translation());
                        coi.setRotation(tfi.rotation());
                        coi.computeAABB();

                        fcl::CollisionObjectf coj(robots_[j]->getCollisionGeometry(pj));
                        coj.setTranslation(tfj.translation());
                        coj.setRotation(tfj.rotation());
                        coj.computeAABB();

                        fcl::CollisionRequestf req;
                        fcl::CollisionResultf res;
                        fcl::collide(&coi, &coj, req, res);
                        if (res.isCollision()) return false;
                    }
                }
            }
        }
        return true;
    }

private:
    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng_;
    std::vector<std::shared_ptr<Robot>> robots_;
};

class KinoCompoundPropagator : public oc::StatePropagator {
public:
    KinoCompoundPropagator(const oc::SpaceInformationPtr& si,
                           const std::vector<std::shared_ptr<Robot>>& robots)
        : oc::StatePropagator(si), robots_(robots) {}

    void propagate(const ob::State* state, const oc::Control* control,
                   double duration, ob::State* result) const override {
        auto cs = state->as<ob::CompoundState>();
        auto cc = control->as<oc::CompoundControlSpace::ControlType>();
        auto cr = result->as<ob::CompoundState>();
        for (size_t i = 0; i < robots_.size(); ++i) {
            robots_[i]->propagate(cs->components[i], cc->components[i],
                                  duration, cr->components[i]);
        }
    }

    bool canPropagateBackward() const override { return false; }

private:
    std::vector<std::shared_ptr<Robot>> robots_;
};

class KinoCompoundGoal : public ob::GoalRegion {
public:
    KinoCompoundGoal(const ob::SpaceInformationPtr& si,
                     const std::vector<ob::State*>& goals,
                     double threshold)
        : ob::GoalRegion(si), goals_(goals) {
        threshold_ = threshold;
    }

    double distanceGoal(const ob::State* st) const override {
        auto cs = st->as<ob::CompoundState>();
        auto css = si_->getStateSpace()->as<ob::CompoundStateSpace>();
        double max_d = 0.0;
        for (size_t i = 0; i < goals_.size(); ++i) {
            double d = css->getSubspace(i)->distance(cs->components[i], goals_[i]);
            if (d > max_d) max_d = d;
        }
        return max_d;
    }

private:
    std::vector<ob::State*> goals_;
};

// Forward declaration — defined after the planner methods.
static std::vector<Eigen::VectorXd> resamplePath(
    const std::vector<Eigen::VectorXd>& seq, int target_size);

// ============================================================================
// KARCPlanner
// ============================================================================

class KARCPlanner {
public:
    explicit KARCPlanner(const KARCConfig& config) : config_(config), position_bounds_(2) {}
    ~KARCPlanner() { cleanup(); }

    KARCResult plan(const PlanningProblem& problem);

private:
    // Setup
    void setupEnvironment(const PlanningProblem& problem);
    void setupRobots(const PlanningProblem& problem);
    void cleanup();

    // Step 1: kinematic paths ignoring dynamics
    void computeInitialPaths();

    // Step 2: segment goals from kinematic paths
    // Returns [robot][segment] = Eigen goal state
    std::vector<std::vector<Eigen::VectorXd>> computeSegmentGoals(int m);

    // Step 3: single-robot trajectory optimizer (FDDP)
    // Returns sequence of states for the segment.
    // coupled_robots / coupled_seqs: already-planned robots whose paths become
    // soft distance constraints (used by prioritised traj-opt, empty otherwise).
    std::vector<Eigen::VectorXd> optimizeSegmentFDDP(
        size_t robot_idx,
        const Eigen::VectorXd& x_start,
        const Eigen::VectorXd& x_goal,
        const std::vector<Eigen::VectorXd>& guide,
        const std::vector<std::shared_ptr<Robot>>& coupled_robots = {},
        const std::vector<std::vector<Eigen::VectorXd>>& coupled_seqs = {});

    // Fallback: OMPL kinodynamic RRT for one robot
    std::vector<Eigen::VectorXd> rrtSegment(
        size_t robot_idx,
        const ob::State* start,
        const ob::State* goal,
        double timeout);

    // Convert PathControl to Eigen sequence
    std::vector<Eigen::VectorXd> pathControlToEigen(
        const oc::PathControl& path, size_t robot_idx) const;

    // Conflict detection
    Conflict* findConflict(const std::vector<std::vector<Eigen::VectorXd>>& seqs);
    bool checkFCLCollision(size_t ri, const ob::State* si,
                           size_t rj, const ob::State* sj) const;
    // Time-aware collision check: returns true iff seq for robot ri is
    // collision-free against other_seqs at each matching timestep.
    bool timeAwareCollisionCheck(
        size_t ri,
        const std::vector<Eigen::VectorXd>& seq,
        const std::vector<std::shared_ptr<Robot>>& other_robots,
        const std::vector<std::vector<Eigen::VectorXd>>& other_seqs) const;

    // Subproblem
    Subproblem createSubproblem(const Conflict& conflict,
                                const std::vector<Eigen::VectorXd>& seg_starts,
                                const std::vector<Eigen::VectorXd>& seg_goals,
                                int seg_idx,
                                const std::vector<Eigen::VectorXd>& prev_seg_starts,
                                const std::vector<Eigen::VectorXd>& next_seg_goals);

    SubproblemSolution solveSubproblem(const Subproblem& sp);
    SubproblemSolution tryPrioritizedTrajOpt(const Subproblem& sp);
    SubproblemSolution tryDecoupledKinoRRT(const Subproblem& sp);
    SubproblemSolution tryCompositeKinoRRT(const Subproblem& sp);

    // Time remaining
    double timeRemaining() const;

    // Member variables
    KARCConfig config_;
    PlanningProblem problem_;
    ob::RealVectorBounds position_bounds_;
    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng_environment_;
    std::vector<std::shared_ptr<Robot>> robots_;
    std::vector<ob::State*> start_states_;
    std::vector<ob::State*> goal_states_;
    std::vector<std::shared_ptr<og::PathGeometric>> kinematic_paths_;
    KARCResult result_;
    std::chrono::steady_clock::time_point plan_start_;
};

// ============================================================================
// Utility: time remaining
// ============================================================================

double KARCPlanner::timeRemaining() const {
    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - plan_start_).count();
    return config_.time_limit - elapsed;
}

// ============================================================================
// Setup / Cleanup
// ============================================================================

void KARCPlanner::setupEnvironment(const PlanningProblem& problem) {
    position_bounds_.setLow(0, problem.env_min[0]);
    position_bounds_.setLow(1, problem.env_min[1]);
    position_bounds_.setHigh(0, problem.env_max[0]);
    position_bounds_.setHigh(1, problem.env_max[1]);

    col_mng_environment_ = std::make_shared<fcl::DynamicAABBTreeCollisionManagerf>();
    col_mng_environment_->registerObjects(problem.obstacles);
    col_mng_environment_->setup();
}

void KARCPlanner::setupRobots(const PlanningProblem& problem) {
    robots_.clear();
    start_states_.clear();
    goal_states_.clear();

    for (size_t i = 0; i < problem.robots.size(); ++i) {
        const auto& spec = problem.robots[i];
        auto robot = create_robot(spec.type, position_bounds_);
        auto si = robot->getSpaceInformation();

        si->setStateValidityChecker(
            std::make_shared<fclStateValidityChecker>(si, col_mng_environment_, robot));
        si->setStatePropagator(
            std::make_shared<RobotStatePropagator>(si, robot));
        si->setPropagationStepSize(config_.propagation_step_size);
        si->setMinMaxControlDuration(config_.control_duration_min,
                                     config_.control_duration_max);
        si->setup();

        robots_.push_back(robot);

        auto* s = si->allocState();
        si->getStateSpace()->copyFromReals(s, spec.start);
        start_states_.push_back(s);

        auto* g = si->allocState();
        si->getStateSpace()->copyFromReals(g, spec.goal);
        goal_states_.push_back(g);
    }
}

void KARCPlanner::cleanup() {
    for (size_t i = 0; i < robots_.size(); ++i) {
        auto si = robots_[i]->getSpaceInformation();
        if (i < start_states_.size() && start_states_[i])
            si->freeState(start_states_[i]);
        if (i < goal_states_.size() && goal_states_[i])
            si->freeState(goal_states_[i]);
    }
    robots_.clear();
    start_states_.clear();
    goal_states_.clear();
    kinematic_paths_.clear();
    col_mng_environment_.reset();
}

// ============================================================================
// Step 1: Compute initial kinematic paths (ignoring dynamics)
// Uses RRTConnect with a trivial static propagator (geometric planning mode)
// ============================================================================

void KARCPlanner::computeInitialPaths() {
    kinematic_paths_.resize(robots_.size());
    double per_robot_time = std::max(1.0, timeRemaining() / (2.0 * robots_.size()));

    for (size_t i = 0; i < robots_.size(); ++i) {
        auto robot = robots_[i];
        auto si = robot->getSpaceInformation();

        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        pdef->addStartState(start_states_[i]);
        auto goal = std::make_shared<ob::GoalState>(si);
        goal->setState(goal_states_[i]);
        goal->setThreshold(config_.goal_threshold);
        pdef->setGoal(goal);

        auto planner = std::make_shared<og::RRTConnect>(si);
        planner->setProblemDefinition(pdef);
        planner->setup();

        ob::PlannerStatus solved = planner->solve(
            ob::timedPlannerTerminationCondition(per_robot_time));

        if (!solved) {
            throw std::runtime_error("K-ARC: Failed to find kinematic path for robot "
                                     + std::to_string(i));
        }

        auto path = std::dynamic_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
        path->interpolate();
        kinematic_paths_[i] = std::make_shared<og::PathGeometric>(*path);
        std::cout << "  Robot " << i << ": kinematic path with "
                  << path->getStateCount() << " states\n";
    }
}

// ============================================================================
// Step 2: Segment goals
// For robot i, segment j's goal is the state at index (j+1)*K/m in path
// ============================================================================

std::vector<std::vector<Eigen::VectorXd>>
KARCPlanner::computeSegmentGoals(int m) {
    std::vector<std::vector<Eigen::VectorXd>> goals(robots_.size());

    for (size_t i = 0; i < robots_.size(); ++i) {
        auto& path = *kinematic_paths_[i];
        size_t K = path.getStateCount();
        auto si = robots_[i]->getSpaceInformation();
        goals[i].resize(m);

        for (int j = 0; j < m; ++j) {
            size_t idx;
            if (j == m - 1) {
                idx = K - 1;
            } else {
                idx = std::min(K - 1, static_cast<size_t>((j + 1) * K / m));
            }
            goals[i][j] = omplToEigen(path.getState(idx), si);
        }
    }
    return goals;
}

// ============================================================================
// Step 3: Single-robot FDDP trajectory optimizer
// ============================================================================

std::vector<Eigen::VectorXd> KARCPlanner::optimizeSegmentFDDP(
    size_t robot_idx,
    const Eigen::VectorXd& x_start,
    const Eigen::VectorXd& x_goal,
    const std::vector<Eigen::VectorXd>& guide,
    const std::vector<std::shared_ptr<Robot>>& coupled_robots,
    const std::vector<std::vector<Eigen::VectorXd>>& coupled_seqs)
{
    auto robot = robots_[robot_idx];
    int T = config_.T_horizon;
    int nu = RobotRunningModel::computeNu(robot);

    // Base uncoupled running model (used when no coupled sequences provided)
    auto running_raw = std::make_shared<RobotRunningModel>(
        robot, col_mng_environment_, config_.dt, config_.beta_control,
        config_.obstacle_weight, config_.collision_margin);
    auto running_uncoupled = std::make_shared<crocoddyl::ActionModelNumDiff>(running_raw, true);

    auto terminal_raw = std::make_shared<RobotTerminalModel>(
        robot, x_goal, config_.goal_weight);
    auto terminal = std::make_shared<crocoddyl::ActionModelNumDiff>(terminal_raw, true);

    // Build per-timestep running models.
    // When coupled sequences are provided (prioritised traj-opt), each timestep k
    // gets a model that penalises proximity to the already-planned robot at that k.
    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models;
    running_models.reserve(T);
    for (int k = 0; k < T; ++k) {
        if (!coupled_seqs.empty()) {
            // Sum contributions from all already-planned robots
            // For simplicity, use only the first coupled sequence (subproblems
            // always involve exactly 2 robots, so there is at most 1 planned ahead).
            size_t t_idx = std::min(static_cast<size_t>(k), coupled_seqs[0].size() - 1);
            auto raw_k = std::make_shared<RobotRunningModelCoupled>(
                robot, col_mng_environment_, config_.dt, config_.beta_control,
                config_.obstacle_weight, config_.collision_margin,
                coupled_seqs[0][t_idx],
                config_.goal_weight,
                config_.collision_margin);
            running_models.push_back(
                std::make_shared<crocoddyl::ActionModelNumDiff>(raw_k, true));
        } else {
            running_models.push_back(running_uncoupled);
        }
    }

    // Build shooting problem
    auto problem = std::make_shared<crocoddyl::ShootingProblem>(
        x_start, running_models, terminal);

    // Warm start: interpolate guide to T+1 waypoints
    std::vector<Eigen::VectorXd> xs_init(T + 1);
    if (guide.empty()) {
        for (int k = 0; k <= T; ++k) {
            double alpha = static_cast<double>(k) / T;
            xs_init[k] = (1.0 - alpha) * x_start + alpha * x_goal;
        }
    } else {
        for (int k = 0; k <= T; ++k) {
            double alpha = static_cast<double>(k) / T;
            double guide_idx = alpha * (static_cast<int>(guide.size()) - 1);
            int lo = static_cast<int>(guide_idx);
            int hi = std::min(lo + 1, static_cast<int>(guide.size()) - 1);
            double t = guide_idx - lo;
            xs_init[k] = (1.0 - t) * guide[lo] + t * guide[hi];
        }
    }
    std::vector<Eigen::VectorXd> us_init(T, Eigen::VectorXd::Zero(nu));

    // Solve
    auto solver = std::make_shared<crocoddyl::SolverFDDP>(problem);
    bool converged = solver->solve(xs_init, us_init, config_.fddp_max_iter, false);

    const auto& xs = solver->get_xs();
    const auto& us = solver->get_us();

    // Check if terminal state is close enough to goal.
    // Use only final_dist (not converged flag) — Crocoddyl sets converged=false
    // whenever it hits fddp_max_iter, even if the trajectory already reached the goal.
    double final_dist = (xs.back() - x_goal).norm();
    std::cout << "    FDDP converged=" << converged << " dist=" << final_dist << "\n";
    if (final_dist > config_.goal_threshold * 3.0) {
        std::cout << "    FDDP dist too large, falling back to RRT\n";
        // Fall back to OMPL RRT
        auto si = robots_[robot_idx]->getSpaceInformation();
        auto* s_ompl = si->allocState();
        auto* g_ompl = si->allocState();
        std::vector<double> sv(x_start.data(), x_start.data() + x_start.size());
        std::vector<double> gv(x_goal.data(), x_goal.data() + x_goal.size());
        si->getStateSpace()->copyFromReals(s_ompl, sv);
        si->getStateSpace()->copyFromReals(g_ompl, gv);
        auto seq = rrtSegment(robot_idx, s_ompl, g_ompl,
                              std::min(config_.prioritized_traj_opt_timeout, timeRemaining()));
        si->freeState(s_ompl);
        si->freeState(g_ompl);
        return seq;
    }

    // Convert xs to sequence of Eigen vectors; truncate at first state within
    // goal tolerance to minimise trajectory time (mirrors paper's Δt minimisation).
    auto seq = std::vector<Eigen::VectorXd>(xs.begin(), xs.end());
    for (size_t k = 1; k < seq.size(); ++k) {
        if ((seq[k] - x_goal).norm() <= config_.goal_threshold) {
            seq.resize(k + 1);
            break;
        }
    }
    return seq;
}

// ============================================================================
// RRT fallback for a single robot segment
// ============================================================================

std::vector<Eigen::VectorXd> KARCPlanner::rrtSegment(
    size_t robot_idx,
    const ob::State* start,
    const ob::State* goal,
    double timeout)
{
    auto robot = robots_[robot_idx];
    auto si = robot->getSpaceInformation();

    auto pdef = std::make_shared<ob::ProblemDefinition>(si);
    pdef->addStartState(start);
    auto goal_cond = std::make_shared<ob::GoalState>(si);
    goal_cond->setState(goal);
    goal_cond->setThreshold(config_.goal_threshold);
    pdef->setGoal(goal_cond);

    auto planner = std::make_shared<oc::RRT>(si);
    planner->setGoalBias(0.05);
    planner->setProblemDefinition(pdef);
    planner->setup();

    double t = std::max(0.5, timeout);
    ob::PlannerStatus status = planner->solve(ob::timedPlannerTerminationCondition(t));

    if (!status) return {};

    auto path = pdef->getSolutionPath()->as<oc::PathControl>();
    path->interpolate();
    return pathControlToEigen(*path, robot_idx);
}

std::vector<Eigen::VectorXd> KARCPlanner::pathControlToEigen(
    const oc::PathControl& path, size_t robot_idx) const
{
    auto si = robots_[robot_idx]->getSpaceInformation();
    std::vector<Eigen::VectorXd> seq;
    seq.reserve(path.getStateCount());
    for (size_t s = 0; s < path.getStateCount(); ++s) {
        seq.push_back(omplToEigen(path.getState(s), si));
    }
    return seq;
}

// ============================================================================
// FCL collision check between two robot states
// ============================================================================

bool KARCPlanner::checkFCLCollision(size_t ri, const ob::State* si,
                                     size_t rj, const ob::State* sj) const
{
    auto& robot_i = robots_[ri];
    auto& robot_j = robots_[rj];
    for (size_t pi = 0; pi < robot_i->numParts(); ++pi) {
        for (size_t pj = 0; pj < robot_j->numParts(); ++pj) {
            const auto& tfi = robot_i->getTransform(si, pi);
            const auto& tfj = robot_j->getTransform(sj, pj);

            fcl::CollisionObjectf coi(robot_i->getCollisionGeometry(pi));
            coi.setTranslation(tfi.translation());
            coi.setRotation(tfi.rotation());
            coi.computeAABB();

            fcl::CollisionObjectf coj(robot_j->getCollisionGeometry(pj));
            coj.setTranslation(tfj.translation());
            coj.setRotation(tfj.rotation());
            coj.computeAABB();

            fcl::CollisionRequestf req;
            fcl::CollisionResultf res;
            fcl::collide(&coi, &coj, req, res);
            if (res.isCollision()) return true;
        }
    }
    return false;
}

// ============================================================================
// Time-aware collision check across a resampled path
// ============================================================================

bool KARCPlanner::timeAwareCollisionCheck(
    size_t ri,
    const std::vector<Eigen::VectorXd>& seq,
    const std::vector<std::shared_ptr<Robot>>& other_robots,
    const std::vector<std::vector<Eigen::VectorXd>>& other_seqs) const
{
    if (seq.empty() || other_robots.empty()) return true;

    auto si_i = robots_[ri]->getSpaceInformation();
    auto* st_i = si_i->allocState();
    bool ok = true;

    for (size_t t = 0; t < seq.size() && ok; ++t) {
        std::vector<double> vi(seq[t].data(), seq[t].data() + seq[t].size());
        si_i->getStateSpace()->copyFromReals(st_i, vi);

        for (size_t oi = 0; oi < other_robots.size() && ok; ++oi) {
            // Find robot index in robots_ vector for checkFCLCollision
            size_t rj = robots_.size();
            for (size_t idx = 0; idx < robots_.size(); ++idx) {
                if (robots_[idx] == other_robots[oi]) { rj = idx; break; }
            }
            if (rj >= robots_.size()) continue;

            size_t t_o = std::min(t, other_seqs[oi].size() - 1);
            auto si_j = other_robots[oi]->getSpaceInformation();
            auto* st_j = si_j->allocState();
            std::vector<double> vj(other_seqs[oi][t_o].data(),
                                   other_seqs[oi][t_o].data() + other_seqs[oi][t_o].size());
            si_j->getStateSpace()->copyFromReals(st_j, vj);

            if (checkFCLCollision(ri, st_i, rj, st_j)) ok = false;

            si_j->freeState(st_j);
        }
    }

    si_i->freeState(st_i);
    return ok;
}

// ============================================================================
// Find first inter-robot conflict in a set of per-robot state sequences
// ============================================================================

Conflict* KARCPlanner::findConflict(
    const std::vector<std::vector<Eigen::VectorXd>>& seqs)
{
    size_t max_t = 0;
    for (const auto& seq : seqs) max_t = std::max(max_t, seq.size());

    std::vector<ob::State*> tmp_states(robots_.size(), nullptr);
    for (size_t i = 0; i < robots_.size(); ++i) {
        if (!seqs[i].empty())
            tmp_states[i] = robots_[i]->getSpaceInformation()->allocState();
    }

    Conflict* found = nullptr;
    for (size_t t = 0; t < max_t && !found; ++t) {
        // Fill states at timestep t
        for (size_t i = 0; i < robots_.size(); ++i) {
            if (seqs[i].empty()) continue;
            size_t idx = std::min(t, seqs[i].size() - 1);
            std::vector<double> rv(seqs[i][idx].data(),
                                   seqs[i][idx].data() + seqs[i][idx].size());
            robots_[i]->getSpaceInformation()->getStateSpace()->copyFromReals(
                tmp_states[i], rv);
        }

        for (size_t i = 0; i < robots_.size() && !found; ++i) {
            for (size_t j = i + 1; j < robots_.size() && !found; ++j) {
                if (seqs[i].empty() || seqs[j].empty()) continue;
                if (checkFCLCollision(i, tmp_states[i], j, tmp_states[j])) {
                    found = new Conflict();
                    found->robot_i = i;
                    found->robot_j = j;
                    found->timestep = static_cast<int>(t);
                }
            }
        }
    }

    for (size_t i = 0; i < robots_.size(); ++i) {
        if (tmp_states[i])
            robots_[i]->getSpaceInformation()->freeState(tmp_states[i]);
    }

    return found;
}

// ============================================================================
// Create subproblem from a conflict
// ============================================================================

Subproblem KARCPlanner::createSubproblem(
    const Conflict& conflict,
    const std::vector<Eigen::VectorXd>& seg_starts,
    const std::vector<Eigen::VectorXd>& seg_goals,
    int seg_idx,
    const std::vector<Eigen::VectorXd>& prev_seg_starts,
    const std::vector<Eigen::VectorXd>& next_seg_goals)
{
    Subproblem sp;
    sp.robot_indices = {conflict.robot_i, conflict.robot_j};
    sp.segment_idx = seg_idx;
    sp.env_min = problem_.env_min;
    sp.env_max = problem_.env_max;
    sp.local_obstacles = problem_.obstacles;

    for (size_t k = 0; k < sp.robot_indices.size(); ++k) {
        size_t ri = sp.robot_indices[k];
        auto si = robots_[ri]->getSpaceInformation();

        auto* ls = si->allocState();
        std::vector<double> sv(seg_starts[ri].data(),
                               seg_starts[ri].data() + seg_starts[ri].size());
        si->getStateSpace()->copyFromReals(ls, sv);
        sp.local_starts.push_back(ls);

        auto* lg = si->allocState();
        std::vector<double> gv(seg_goals[ri].data(),
                               seg_goals[ri].data() + seg_goals[ri].size());
        si->getStateSpace()->copyFromReals(lg, gv);
        sp.local_goals.push_back(lg);

        // Expanded start/goal for AdaptSubProblem (paper Algorithm 2, line 10)
        auto* es = si->allocState();
        std::vector<double> esv(prev_seg_starts[ri].data(),
                                prev_seg_starts[ri].data() + prev_seg_starts[ri].size());
        si->getStateSpace()->copyFromReals(es, esv);
        sp.expanded_starts.push_back(es);

        auto* eg = si->allocState();
        std::vector<double> egv(next_seg_goals[ri].data(),
                                next_seg_goals[ri].data() + next_seg_goals[ri].size());
        si->getStateSpace()->copyFromReals(eg, egv);
        sp.expanded_goals.push_back(eg);
    }

    return sp;
}

// ============================================================================
// Solve Subproblem: hierarchy of solvers
// ============================================================================

SubproblemSolution KARCPlanner::solveSubproblem(const Subproblem& sp) {
    // Try all three solvers in order.
    auto tryAll = [&](const Subproblem& s) -> SubproblemSolution {
        SubproblemSolution sol;
        std::cout << "    Trying prioritized trajectory optimization...\n";
        sol = tryPrioritizedTrajOpt(s);
        if (sol.solved) { sol.method_used = "PrioritizedTrajOpt"; return sol; }

        std::cout << "    Trying decoupled kinodynamic RRT...\n";
        sol = tryDecoupledKinoRRT(s);
        if (sol.solved) { sol.method_used = "DecoupledKinoRRT"; return sol; }

        std::cout << "    Trying composite kinodynamic RRT...\n";
        sol = tryCompositeKinoRRT(s);
        if (sol.solved) { sol.method_used = "CompositeKinoRRT"; return sol; }

        sol.solved = false;
        return sol;
    };

    SubproblemSolution sol = tryAll(sp);
    if (sol.solved || sp.adapted) return sol;

    // AdaptSubProblem (paper Algorithm 2, line 10): expand start→prev-segment
    // start and goal→next-segment goal, then retry the solver hierarchy once.
    if (!sp.expanded_starts.empty() && !sp.expanded_goals.empty()) {
        std::cout << "    All solvers failed; adapting subproblem to expanded query...\n";

        // Build an expanded copy of the subproblem with widened queries.
        Subproblem expanded = sp;
        expanded.adapted = true;
        for (size_t k = 0; k < sp.robot_indices.size(); ++k) {
            size_t ri = sp.robot_indices[k];
            auto si = robots_[ri]->getSpaceInformation();
            si->copyState(expanded.local_starts[k], sp.expanded_starts[k]);
            si->copyState(expanded.local_goals[k],  sp.expanded_goals[k]);
        }

        sol = tryAll(expanded);
        if (sol.solved) sol.method_used += "+Adapted";
    }

    return sol;
}

// ============================================================================
// Level 1: Prioritized Trajectory Optimization
// Plans for each robot sequentially; earlier robots become obstacles for later ones
// ============================================================================

SubproblemSolution KARCPlanner::tryPrioritizedTrajOpt(const Subproblem& sp) {
    SubproblemSolution sol;
    sol.state_seqs.resize(sp.robot_indices.size());

    std::vector<std::vector<Eigen::VectorXd>> planned_seqs;
    std::vector<std::shared_ptr<Robot>> planned_robots;

    auto t0 = std::chrono::steady_clock::now();

    for (size_t k = 0; k < sp.robot_indices.size(); ++k) {
        size_t ri = sp.robot_indices[k];
        auto x_start = omplToEigen(sp.local_starts[k], robots_[ri]->getSpaceInformation());
        auto x_goal  = omplToEigen(sp.local_goals[k],  robots_[ri]->getSpaceInformation());

        // Pass already-planned sequences as soft distance constraints (paper Sec. IV-C):
        // r₂'s optimiser receives the constraint that each state stays ≥ collision_margin
        // away from each state in r₁'s path.
        auto seq = optimizeSegmentFDDP(ri, x_start, x_goal, {}, planned_robots, planned_seqs);

        if (seq.empty()) {
            sol.solved = false;
            sol.planning_time = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            return sol;
        }

        seq = resamplePath(seq, config_.T_horizon + 1);
        sol.state_seqs[k] = seq;
        planned_seqs.push_back(seq);
        planned_robots.push_back(robots_[ri]);
    }

    // Final time-aware collision check: verify the constrained FDDP actually
    // produced conflict-free paths before declaring success.
    for (size_t k = 0; k < sp.robot_indices.size(); ++k) {
        size_t ri = sp.robot_indices[k];
        std::vector<std::shared_ptr<Robot>> others;
        std::vector<std::vector<Eigen::VectorXd>> other_seqs;
        for (size_t qi = 0; qi < sp.robot_indices.size(); ++qi) {
            if (qi == k) continue;
            others.push_back(robots_[sp.robot_indices[qi]]);
            other_seqs.push_back(sol.state_seqs[qi]);
        }
        if (!timeAwareCollisionCheck(ri, sol.state_seqs[k], others, other_seqs)) {
            std::cout << "    Prioritized traj-opt still collides after coupling; escalating\n";
            sol.solved = false;
            sol.planning_time = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            return sol;
        }
    }

    sol.solved = true;
    sol.planning_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    return sol;
}

// ============================================================================
// Level 2: Decoupled Kinodynamic RRT
// Plans each robot with previously planned robots as dynamic obstacles
// ============================================================================

SubproblemSolution KARCPlanner::tryDecoupledKinoRRT(const Subproblem& sp) {
    SubproblemSolution sol;
    sol.state_seqs.resize(sp.robot_indices.size());

    std::vector<std::vector<Eigen::VectorXd>> planned_seqs;
    std::vector<std::shared_ptr<Robot>> planned_robots;

    auto t0 = std::chrono::steady_clock::now();
    double timeout = std::min(config_.decoupled_kino_rrt_timeout, timeRemaining());

    for (size_t k = 0; k < sp.robot_indices.size(); ++k) {
        size_t ri = sp.robot_indices[k];
        auto si = robots_[ri]->getSpaceInformation();

        auto decoupled_checker = std::make_shared<KinoDecoupledValidityChecker>(
            si, col_mng_environment_, robots_[ri], planned_robots, planned_seqs);

        // Skip if start/goal is in the time-unaware forbidden zone.
        if (!decoupled_checker->isValid(sp.local_starts[k]) ||
            !decoupled_checker->isValid(sp.local_goals[k])) {
            std::cout << "    Skipping RRT: start/goal blocked by time-unaware checker\n";
            sol.solved = false;
            sol.planning_time = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            return sol;
        }

        auto old_checker = si->getStateValidityChecker();
        si->setStateValidityChecker(decoupled_checker);
        si->setup();

        double t_left = std::max(0.5, timeout - std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count());

        auto seq = rrtSegment(ri, sp.local_starts[k], sp.local_goals[k], t_left);

        si->setStateValidityChecker(old_checker);
        si->setup();

        if (seq.empty()) {
            sol.solved = false;
            sol.planning_time = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            return sol;
        }

        // Post-hoc time-aware validation: the during-planning checker is time-
        // unaware (conservative). Verify the returned path is actually free at
        // each matching timestep before accepting it.
        auto resampled = resamplePath(seq, config_.T_horizon + 1);
        if (!timeAwareCollisionCheck(ri, resampled, planned_robots, planned_seqs)) {
            std::cout << "    Decoupled RRT path fails time-aware check; discarding\n";
            sol.solved = false;
            sol.planning_time = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            return sol;
        }
        seq = resampled;

        sol.state_seqs[k] = seq;
        planned_seqs.push_back(seq);
        planned_robots.push_back(robots_[ri]);
    }

    sol.solved = true;
    sol.planning_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    return sol;
}

// ============================================================================
// Level 3: Composite Kinodynamic RRT (compound state+control space)
// ============================================================================

SubproblemSolution KARCPlanner::tryCompositeKinoRRT(const Subproblem& sp) {
    SubproblemSolution sol;
    auto t0 = std::chrono::steady_clock::now();
    double timeout = std::min(config_.composite_kino_rrt_timeout, timeRemaining());

    // Collect subproblem robots
    std::vector<std::shared_ptr<Robot>> sp_robots;
    for (size_t ri : sp.robot_indices) sp_robots.push_back(robots_[ri]);

    // Build compound state+control spaces
    auto compound_ss = std::make_shared<ob::CompoundStateSpace>();
    auto compound_cs = std::make_shared<oc::CompoundControlSpace>(compound_ss);
    for (auto& r : sp_robots) {
        compound_ss->addSubspace(r->getSpaceInformation()->getStateSpace(), 1.0);
        compound_cs->addSubspace(r->getSpaceInformation()->getControlSpace());
    }

    auto compound_si = std::make_shared<oc::SpaceInformation>(compound_ss, compound_cs);
    compound_si->setStateValidityChecker(
        std::make_shared<KinoCompoundValidityChecker>(
            compound_si, col_mng_environment_, sp_robots));
    compound_si->setStatePropagator(
        std::make_shared<KinoCompoundPropagator>(compound_si, sp_robots));
    compound_si->setPropagationStepSize(config_.propagation_step_size);
    compound_si->setMinMaxControlDuration(config_.control_duration_min,
                                          config_.control_duration_max);
    compound_si->setup();

    // Build compound start
    auto compound_start = compound_si->allocState();
    {
        auto cs = compound_start->as<ob::CompoundState>();
        for (size_t k = 0; k < sp_robots.size(); ++k) {
            sp_robots[k]->getSpaceInformation()->getStateSpace()->copyState(
                cs->components[k], sp.local_starts[k]);
        }
    }

    // Build goal condition
    std::vector<ob::State*> goals(sp.local_goals.begin(), sp.local_goals.end());
    auto goal_cond = std::make_shared<KinoCompoundGoal>(
        compound_si, goals, config_.goal_threshold);

    auto pdef = std::make_shared<ob::ProblemDefinition>(compound_si);
    pdef->addStartState(compound_start);
    pdef->setGoal(goal_cond);

    auto planner = std::make_shared<oc::RRT>(compound_si);
    planner->setGoalBias(0.05);
    planner->setProblemDefinition(pdef);
    planner->setup();

    ob::PlannerStatus status = planner->solve(
        ob::timedPlannerTerminationCondition(timeout));

    compound_si->freeState(compound_start);

    sol.planning_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    if (!status) {
        sol.solved = false;
        return sol;
    }

    // Extract per-robot state sequences from compound path
    auto path = pdef->getSolutionPath()->as<oc::PathControl>();
    path->interpolate();

    sol.state_seqs.resize(sp_robots.size());
    for (size_t k = 0; k < sp_robots.size(); ++k) {
        sol.state_seqs[k].reserve(path->getStateCount());
    }

    for (size_t s = 0; s < path->getStateCount(); ++s) {
        auto compound = path->getState(s)->as<ob::CompoundState>();
        for (size_t k = 0; k < sp_robots.size(); ++k) {
            sol.state_seqs[k].push_back(
                omplToEigen(compound->components[k],
                            sp_robots[k]->getSpaceInformation()));
        }
    }

    for (size_t k = 0; k < sp_robots.size(); ++k) {
        sol.state_seqs[k] = resamplePath(sol.state_seqs[k], config_.T_horizon + 1);
    }

    sol.solved = true;
    return sol;
}

// ============================================================================
// Resample a state sequence to exactly target_size states via linear interp.
// Ensures all segment paths have the same length so index t = same wall-clock
// time for every robot, which is required by the index-based findConflict.
// ============================================================================

static std::vector<Eigen::VectorXd> resamplePath(
    const std::vector<Eigen::VectorXd>& seq, int target_size)
{
    if (seq.empty()) return seq;
    if (static_cast<int>(seq.size()) == target_size) return seq;

    std::vector<Eigen::VectorXd> out;
    out.reserve(target_size);
    int n = static_cast<int>(seq.size()) - 1;
    for (int k = 0; k < target_size; ++k) {
        double t = static_cast<double>(k) / (target_size - 1) * n;
        int lo = static_cast<int>(t);
        int hi = std::min(lo + 1, n);
        double alpha = t - lo;
        out.push_back((1.0 - alpha) * seq[lo] + alpha * seq[hi]);
    }
    return out;
}

// ============================================================================
// Main Plan Function
// ============================================================================

KARCResult KARCPlanner::plan(const PlanningProblem& problem) {
    std::cout << "K-ARC: Starting planner...\n";
    plan_start_ = std::chrono::steady_clock::now();
    problem_ = problem;
    result_ = KARCResult();

    try {
        setupEnvironment(problem);
        setupRobots(problem);

        int n = static_cast<int>(robots_.size());
        int m = config_.num_segments;
        std::cout << "K-ARC: " << n << " robots, " << m << " segments\n";

        // Step 1: kinematic paths
        std::cout << "K-ARC: Computing kinematic paths...\n";
        computeInitialPaths();

        // Step 2: segment goals
        auto seg_goals = computeSegmentGoals(m);

        // Initialize per-robot paths and segment starts
        result_.paths.resize(n);
        std::vector<Eigen::VectorXd> seg_starts(n);
        for (int i = 0; i < n; ++i) {
            seg_starts[i] = omplToEigen(start_states_[i],
                                        robots_[i]->getSpaceInformation());
        }
        // prev_seg_starts is used by AdaptSubProblem to widen the query start.
        std::vector<Eigen::VectorXd> prev_seg_starts = seg_starts;

        // Step 3: process each segment
        for (int j = 0; j < m; ++j) {
            if (timeRemaining() < 1.0) {
                std::cout << "K-ARC: Time limit reached at segment " << j << "\n";
                result_.solved = false;
                break;
            }

            std::cout << "K-ARC: Processing segment " << j + 1 << "/" << m << "\n";

            // For last segment, override goal with actual goal state
            std::vector<Eigen::VectorXd> current_seg_goals(n);
            for (int i = 0; i < n; ++i) {
                if (j == m - 1) {
                    current_seg_goals[i] = omplToEigen(goal_states_[i],
                                                       robots_[i]->getSpaceInformation());
                } else {
                    current_seg_goals[i] = seg_goals[i][j];
                }
            }

            // Next-segment goals used by AdaptSubProblem to widen the query goal.
            std::vector<Eigen::VectorXd> next_seg_goals(n);
            for (int i = 0; i < n; ++i) {
                if (j + 1 < m) {
                    next_seg_goals[i] = seg_goals[i][j + 1];
                } else {
                    next_seg_goals[i] = omplToEigen(goal_states_[i],
                                                    robots_[i]->getSpaceInformation());
                }
            }

            // Kinematic guide for this segment (states from kinematic path)
            std::vector<std::vector<Eigen::VectorXd>> guides(n);
            for (int i = 0; i < n; ++i) {
                auto& path = *kinematic_paths_[i];
                size_t K = path.getStateCount();
                auto si = robots_[i]->getSpaceInformation();
                size_t lo = (j == 0) ? 0 : std::min(K - 1, static_cast<size_t>(j * K / m));
                size_t hi = (j == m - 1) ? K - 1
                    : std::min(K - 1, static_cast<size_t>((j + 1) * K / m));
                for (size_t idx = lo; idx <= hi; ++idx) {
                    guides[i].push_back(omplToEigen(path.getState(idx), si));
                }
                if (guides[i].empty()) guides[i].push_back(seg_starts[i]);
            }

            // Optimize each robot's segment
            std::vector<std::vector<Eigen::VectorXd>> seg_seqs(n);
            bool any_failed = false;
            for (int i = 0; i < n && !any_failed; ++i) {
                std::cout << "  Robot " << i << " FDDP...\n";
                seg_seqs[i] = optimizeSegmentFDDP(i, seg_starts[i],
                                                  current_seg_goals[i], guides[i]);
                if (seg_seqs[i].empty()) {
                    std::cout << "  Robot " << i << " failed to find segment path\n";
                    any_failed = true;
                } else {
                    int raw_len = static_cast<int>(seg_seqs[i].size());
                    int target = config_.T_horizon + 1;
                    if (raw_len != target) {
                        std::cout << "  Robot " << i << " path length " << raw_len
                                  << " != " << target << ", resampling\n";
                        seg_seqs[i] = resamplePath(seg_seqs[i], target);
                    } else {
                        std::cout << "  Robot " << i << " path length " << raw_len << "\n";
                    }
                }
            }

            if (any_failed) {
                result_.solved = false;
                break;
            }

            // Resolve all conflicts in this segment before moving on
            bool seg_failed = false;
            while (true) {
                Conflict* conflict = findConflict(seg_seqs);
                if (!conflict) break;

                std::cout << "  Conflict: robots " << conflict->robot_i
                          << " and " << conflict->robot_j
                          << " at step " << conflict->timestep << "\n";
                result_.num_conflicts_found++;

                if (result_.num_conflicts_resolved >= config_.max_conflicts_resolved) {
                    delete conflict;
                    std::cout << "K-ARC: Max conflicts reached\n";
                    seg_failed = true;
                    break;
                }

                Subproblem sp = createSubproblem(*conflict, seg_starts,
                                                 current_seg_goals, j,
                                                 prev_seg_starts, next_seg_goals);
                delete conflict;

                SubproblemSolution sol = solveSubproblem(sp);
                sp.freeStates(robots_);

                if (!sol.solved) {
                    std::cout << "  Subproblem unsolvable\n";
                    seg_failed = true;
                    break;
                }

                result_.num_conflicts_resolved++;
                result_.methods_used.push_back(sol.method_used);
                std::cout << "  Resolved via " << sol.method_used << "\n";

                // Update segment sequences with conflict-resolved paths
                for (size_t k = 0; k < sp.robot_indices.size(); ++k) {
                    seg_seqs[sp.robot_indices[k]] = sol.state_seqs[k];
                }
            }

            if (seg_failed) {
                result_.solved = false;
                break;
            }

            // Preserve current seg_starts before advancing (used next iteration by AdaptSubProblem)
            prev_seg_starts = seg_starts;

            // Append segment states to global paths and update starts for next segment
            for (int i = 0; i < n; ++i) {
                const auto& seq = seg_seqs[i];
                size_t start_idx = result_.paths[i].empty() ? 0 : 1; // skip duplicate boundary
                for (size_t s = start_idx; s < seq.size(); ++s) {
                    result_.paths[i].push_back(seq[s]);
                }
                if (result_.paths[i].empty() && !seq.empty()) {
                    result_.paths[i].push_back(seq.front());
                }
                if (!seq.empty()) seg_starts[i] = seq.back();
            }

            if (j == m - 1) {
                result_.solved = true;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "K-ARC: Exception: " << e.what() << "\n";
        result_.solved = false;
    }

    result_.planning_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - plan_start_).count();

    std::cout << "K-ARC: Done in " << result_.planning_time << "s"
              << " solved=" << result_.solved
              << " conflicts=" << result_.num_conflicts_found << "\n";
    return result_;
}

// ============================================================================
// YAML I/O
// ============================================================================

struct RobotSpecStorage { std::vector<fcl::CollisionObjectf*> obstacles; };

PlanningProblem loadProblemFromYAML(const std::string& file,
                                    std::vector<fcl::CollisionObjectf*>& owned_obstacles) {
    std::cout << "Loading: " << file << "\n";
    YAML::Node env = YAML::LoadFile(file);
    PlanningProblem prob;

    const auto& mn = env["environment"]["min"];
    const auto& mx = env["environment"]["max"];
    prob.env_min = {mn[0].as<double>(), mn[1].as<double>()};
    prob.env_max = {mx[0].as<double>(), mx[1].as<double>()};

    if (env["environment"]["obstacles"]) {
        for (const auto& obs : env["environment"]["obstacles"]) {
            if (obs["type"].as<std::string>() == "box") {
                const auto& sz = obs["size"];
                const auto& cn = obs["center"];
                auto box = std::make_shared<fcl::Boxf>(
                    sz[0].as<float>(), sz[1].as<float>(), 1.0f);
                auto* co = new fcl::CollisionObjectf(box);
                co->setTranslation(fcl::Vector3f(
                    cn[0].as<float>(), cn[1].as<float>(), 0.0f));
                co->computeAABB();
                owned_obstacles.push_back(co);
                prob.obstacles.push_back(co);
            }
        }
    }

    for (const auto& rn : env["robots"]) {
        RobotSpec spec;
        spec.type = rn["type"].as<std::string>();
        for (const auto& v : rn["start"]) spec.start.push_back(v.as<double>());
        for (const auto& v : rn["goal"])  spec.goal.push_back(v.as<double>());
        prob.robots.push_back(spec);
    }
    std::cout << "Loaded " << prob.robots.size() << " robots, "
              << prob.obstacles.size() << " obstacles\n";
    return prob;
}

KARCConfig loadConfigFromYAML(const std::string& file) {
    KARCConfig cfg;
    try {
        YAML::Node c = YAML::LoadFile(file);
        if (c["timelimit"])                     cfg.time_limit = c["timelimit"].as<double>();
        if (c["goal_threshold"])                cfg.goal_threshold = c["goal_threshold"].as<double>();
        if (c["num_segments"])                  cfg.num_segments = c["num_segments"].as<int>();
        if (c["dt"])                            cfg.dt = c["dt"].as<double>();
        if (c["T_horizon"])                     cfg.T_horizon = c["T_horizon"].as<int>();
        if (c["beta_control"])                  cfg.beta_control = c["beta_control"].as<double>();
        if (c["goal_weight"])                   cfg.goal_weight = c["goal_weight"].as<double>();
        if (c["fddp_max_iter"])                 cfg.fddp_max_iter = c["fddp_max_iter"].as<int>();
        if (c["prioritized_traj_opt_timeout"])  cfg.prioritized_traj_opt_timeout = c["prioritized_traj_opt_timeout"].as<double>();
        if (c["decoupled_kino_rrt_timeout"])    cfg.decoupled_kino_rrt_timeout = c["decoupled_kino_rrt_timeout"].as<double>();
        if (c["composite_kino_rrt_timeout"])    cfg.composite_kino_rrt_timeout = c["composite_kino_rrt_timeout"].as<double>();
        if (c["max_conflicts_resolved"])        cfg.max_conflicts_resolved = c["max_conflicts_resolved"].as<int>();
        if (c["propagation_step_size"])         cfg.propagation_step_size = c["propagation_step_size"].as<double>();
        if (c["control_duration_min"])          cfg.control_duration_min = c["control_duration_min"].as<int>();
        if (c["control_duration_max"])          cfg.control_duration_max = c["control_duration_max"].as<int>();
        if (c["collision_margin"])              cfg.collision_margin = c["collision_margin"].as<double>();
        if (c["seed"])                          cfg.seed = c["seed"].as<int>();
    } catch (const YAML::Exception& e) {
        std::cerr << "Config load error: " << e.what() << "\n";
        throw;
    }
    return cfg;
}

void writeResultToYAML(const std::string& file,
                       const KARCResult& result,
                       const PlanningProblem& problem) {
    YAML::Node out;
    out["solved"] = result.solved;
    out["planning_time"] = result.planning_time;
    out["num_conflicts_found"] = result.num_conflicts_found;
    out["num_conflicts_resolved"] = result.num_conflicts_resolved;

    if (result.solved && !result.paths.empty()) {
        YAML::Node res_node;
        for (size_t r = 0; r < problem.robots.size(); ++r) {
            YAML::Node robot_data;
            YAML::Node states_node;

            if (r < result.paths.size()) {
                for (const auto& xv : result.paths[r]) {
                    YAML::Node sn;
                    for (int d = 0; d < xv.size(); ++d) sn.push_back(xv[d]);
                    states_node.push_back(sn);
                }
            }
            robot_data["states"] = states_node;
            res_node.push_back(robot_data);
        }
        out["result"] = res_node;
    }

    std::ofstream fout(file);
    fout << out;
    std::cout << "Output written to " << file << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string inputFile, outputFile, configFile;

    po::options_description desc("K-ARC options");
    desc.add_options()
        ("help,h", "Show help")
        ("input,i",  po::value<std::string>(&inputFile)->required(),  "Input YAML")
        ("output,o", po::value<std::string>(&outputFile)->required(), "Output YAML")
        ("cfg,c",    po::value<std::string>(&configFile),             "Config YAML");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc; return 0; }
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "ERROR: " << e.what() << "\n" << desc;
        return 1;
    }

    std::vector<fcl::CollisionObjectf*> owned_obstacles;
    try {
        KARCConfig config;
        if (vm.count("cfg")) config = loadConfigFromYAML(configFile);

        if (config.seed >= 0) {
            ompl::RNG::setSeed(config.seed);
            std::cout << "Seed: " << config.seed << "\n";
        }

        std::cout << "=== K-ARC Configuration ===\n"
                  << "Time limit: " << config.time_limit << "s\n"
                  << "Segments: " << config.num_segments << "\n"
                  << "T_horizon: " << config.T_horizon << "\n"
                  << "dt: " << config.dt << "\n"
                  << "===========================\n";

        PlanningProblem prob = loadProblemFromYAML(inputFile, owned_obstacles);
        KARCPlanner planner(config);
        KARCResult result = planner.plan(prob);

        std::cout << "=== Results ===\n"
                  << "Solved: " << result.solved << "\n"
                  << "Time: " << result.planning_time << "s\n"
                  << "Conflicts found: " << result.num_conflicts_found << "\n"
                  << "Conflicts resolved: " << result.num_conflicts_resolved << "\n";

        writeResultToYAML(outputFile, result, prob);

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        for (auto* o : owned_obstacles) delete o;
        return 1;
    }

    for (auto* o : owned_obstacles) delete o;
    return 0;
}
