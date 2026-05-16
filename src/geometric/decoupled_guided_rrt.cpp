// Multi-Robot OMPL headers
#include <ompl/multirobot/base/SpaceInformation.h>
#include <ompl/multirobot/base/ProblemDefinition.h>
#include <ompl/multirobot/geometric/planners/pp/PP.h>
#include <ompl/multirobot/geometric/PlanGeometric.h>

// OMPL base headers
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/StateSpace.h>
#include <ompl/geometric/PathGeometric.h>

// Standard library
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <limits>
#include <functional>

// YAML
#include <yaml-cpp/yaml.h>

// Boost
#include <boost/program_options.hpp>

// db-CBS robot dynamics
#include "robots.h"
#include "fclStateValidityChecker.hpp"

// Guided planner + decomposition + CBS
#include "guided/guided_geometric_rrt.h"
#include "utils/grid_decomposition.h"
#include "mapf/cbs.h"

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace omrb = ompl::multirobot::base;
namespace omrg = ompl::multirobot::geometric;
namespace po = boost::program_options;


class IndividualStateValidityChecker : public fclStateValidityChecker
{
public:
    IndividualStateValidityChecker(
        ob::SpaceInformationPtr si,
        std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng_environment,
        std::shared_ptr<Robot> robot,
        std::map<const ob::SpaceInformationPtr, std::shared_ptr<Robot>> all_robots,
        std::vector<std::pair<const ob::State*, std::shared_ptr<Robot>>> other_goals)
        : fclStateValidityChecker(si, col_mng_environment, robot, false)
        , all_robots_(std::move(all_robots))
        , other_goals_(std::move(other_goals))
    {}

    bool isValid(const ob::State* state) const override
    {
        if (!fclStateValidityChecker::isValid(state))
            return false;

        const auto& t1 = robot_->getTransform(state, 0);
        fcl::CollisionObjectf co1(robot_->getCollisionGeometry(0));
        co1.setTranslation(t1.translation());
        co1.setRotation(t1.rotation());
        co1.computeAABB();

        for (const auto& [goal_state, other_robot] : other_goals_) {
            const auto& t2 = other_robot->getTransform(goal_state, 0);
            fcl::CollisionObjectf co2(other_robot->getCollisionGeometry(0));
            co2.setTranslation(t2.translation());
            co2.setRotation(t2.rotation());
            co2.computeAABB();

            fcl::CollisionRequest<float> request;
            fcl::CollisionResult<float> result;
            fcl::collide(&co1, &co2, request, result);
            if (result.isCollision())
                return false;
        }
        return true;
    }

    bool areStatesValid(
        const ob::State* state1,
        const std::pair<const ob::SpaceInformationPtr, const ob::State*> state2) const override
    {
        const int part = 0;

        const auto& t1 = robot_->getTransform(state1, part);
        fcl::CollisionObjectf co1(robot_->getCollisionGeometry(part));
        co1.setTranslation(t1.translation());
        co1.setRotation(t1.rotation());
        co1.computeAABB();

        auto iter = all_robots_.find(state2.first);
        if (iter == all_robots_.end())
            return false;

        auto other_robot = iter->second;
        if (!state2.second)
            return true;

        const auto& t2 = other_robot->getTransform(state2.second, part);
        fcl::CollisionObjectf co2(other_robot->getCollisionGeometry(part));
        co2.setTranslation(t2.translation());
        co2.setRotation(t2.rotation());
        co2.computeAABB();

        fcl::CollisionRequest<float> request;
        fcl::CollisionResult<float> result;
        fcl::collide(&co1, &co2, request, result);
        return !result.isCollision();
    }

private:
    std::map<const ob::SpaceInformationPtr, std::shared_ptr<Robot>> all_robots_;
    std::vector<std::pair<const ob::State*, std::shared_ptr<Robot>>> other_goals_;
};


class IndividualGoalCondition : public ob::GoalRegion
{
public:
    IndividualGoalCondition(
        const ob::SpaceInformationPtr& si,
        const ob::State* goal_state,
        double threshold)
        : ob::GoalRegion(si), goal_state_(goal_state)
    {
        threshold_ = threshold;
    }

    double distanceGoal(const ob::State* st) const override
    {
        return si_->distance(st, goal_state_);
    }

private:
    const ob::State* goal_state_;
};


inline bool statesOverlap(
    const ob::State* s1, std::shared_ptr<Robot> r1,
    const ob::State* s2, std::shared_ptr<Robot> r2)
{
    const auto& t1 = r1->getTransform(s1, 0);
    fcl::CollisionObjectf co1(r1->getCollisionGeometry(0));
    co1.setTranslation(t1.translation());
    co1.setRotation(t1.rotation());
    co1.computeAABB();

    const auto& t2 = r2->getTransform(s2, 0);
    fcl::CollisionObjectf co2(r2->getCollisionGeometry(0));
    co2.setTranslation(t2.translation());
    co2.setRotation(t2.rotation());
    co2.computeAABB();

    fcl::CollisionRequest<float> request;
    fcl::CollisionResult<float> result;
    fcl::collide(&co1, &co2, request, result);
    return result.isCollision();
}


int main(int argc, char** argv)
{
    std::string inputFile;
    std::string outputFile;
    std::string configFile;
    std::string vizFile;
    double timelimit                   = 60.0;
    double goal_threshold              = 0.5;
    double goal_bias                   = 0.05;
    double region_size                 = 5.0;
    double cbs_timeout                              = 30.0;
    int    cbs_capacity                             = 1;
    double max_obstacle_volume_percent              = 1.0;
    int    seed                                     = -1;
    int    max_refinement_levels                    = 3;
    bool   check_transition_feasibility             = false;
    double transition_feasibility_robot_size_multiplier = 1.0;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Show help message")
        ("input,i",  po::value<std::string>(&inputFile)->required(),  "Input YAML file")
        ("output,o", po::value<std::string>(&outputFile)->required(), "Output YAML file")
        ("cfg,c",    po::value<std::string>(&configFile),             "Configuration YAML file")
        ("viz,v",    po::value<std::string>(&vizFile),                "Visualization output YAML file");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc << std::endl; return 0; }
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }

    if (vm.count("cfg")) {
        try {
            YAML::Node cfg = YAML::LoadFile(configFile);
            if (cfg["goal_threshold"])              goal_threshold              = cfg["goal_threshold"].as<double>();
            if (cfg["goal_bias"])                   goal_bias                   = cfg["goal_bias"].as<double>();
            if (cfg["timelimit"])                   timelimit                   = cfg["timelimit"].as<double>();
            if (cfg["region_size"])                 region_size                 = cfg["region_size"].as<double>();
            if (cfg["cbs_timeout"])                 cbs_timeout                 = cfg["cbs_timeout"].as<double>();
            if (cfg["cbs_capacity"])                cbs_capacity                = cfg["cbs_capacity"].as<int>();
            if (cfg["max_obstacle_volume_percent"]) max_obstacle_volume_percent = cfg["max_obstacle_volume_percent"].as<double>();
            if (cfg["seed"])                        seed                        = cfg["seed"].as<int>();
            if (cfg["max_refinement_levels"])       max_refinement_levels       = cfg["max_refinement_levels"].as<int>();
            if (cfg["check_transition_feasibility"])
                check_transition_feasibility = cfg["check_transition_feasibility"].as<bool>();
            if (cfg["transition_feasibility_robot_size_multiplier"])
                transition_feasibility_robot_size_multiplier = cfg["transition_feasibility_robot_size_multiplier"].as<double>();
        } catch (const YAML::Exception& e) {
            std::cerr << "ERROR loading config file: " << e.what() << std::endl;
            return 1;
        }
    }

    if (seed >= 0) {
        ompl::RNG::setSeed(seed);
    }

    YAML::Node env;
    try {
        env = YAML::LoadFile(inputFile);
    } catch (const YAML::Exception& e) {
        std::cerr << "ERROR loading YAML file: " << e.what() << std::endl;
        return 1;
    }

    // Parse environment bounds
    const auto& env_min = env["environment"]["min"];
    const auto& env_max = env["environment"]["max"];

    ob::RealVectorBounds position_bounds(2);
    position_bounds.setLow(0,  env_min[0].as<double>());
    position_bounds.setLow(1,  env_min[1].as<double>());
    position_bounds.setHigh(0, env_max[0].as<double>());
    position_bounds.setHigh(1, env_max[1].as<double>());

    // Build FCL obstacle manager
    auto col_mng_environment = std::make_shared<fcl::DynamicAABBTreeCollisionManagerf>();
    std::vector<fcl::CollisionObjectf*> obstacles;

    if (env["environment"]["obstacles"]) {
        for (const auto& obs : env["environment"]["obstacles"]) {
            if (obs["type"].as<std::string>() == "box") {
                const auto& size   = obs["size"];
                const auto& center = obs["center"];
                auto box = std::make_shared<fcl::Boxf>(
                    size[0].as<float>(), size[1].as<float>(), 1.0f);
                auto* co = new fcl::CollisionObjectf(box);
                co->setTranslation(fcl::Vector3f(
                    center[0].as<float>(), center[1].as<float>(), 0.0f));
                co->computeAABB();
                obstacles.push_back(co);
                col_mng_environment->registerObject(co);
            }
        }
        col_mng_environment->setup();
    }

    // --- Pass 1: create robots ---
    std::vector<std::shared_ptr<Robot>> robots;
    std::vector<ob::SpaceInformationPtr> robot_sis;
    std::map<const ob::SpaceInformationPtr, std::shared_ptr<Robot>> all_robots;
    int robot_idx = 0;

    for (const auto& robot_node : env["robots"]) {
        auto robotType = robot_node["type"].as<std::string>();

        auto robot = create_robot(robotType, position_bounds);
        auto state_space = robot->getSpaceInformation()->getStateSpace();
        state_space->setName("Robot " + std::to_string(robot_idx));
        state_space->setup();

        auto robot_si = std::make_shared<ob::SpaceInformation>(state_space);
        all_robots[robot_si] = robot;
        robots.push_back(robot);
        robot_sis.push_back(robot_si);
        ++robot_idx;
    }

    const int num_robots = static_cast<int>(robots.size());

    // --- Pass 2a: parse start/goal states ---
    std::vector<ob::State*> start_states;
    std::vector<ob::State*> goal_states;
    robot_idx = 0;

    for (const auto& robot_node : env["robots"]) {
        auto robot_si = robot_sis[robot_idx];

        const auto& start_vec = robot_node["start"];
        auto* start_state = robot_si->getStateSpace()->allocState();
        auto* start_se2 = start_state->as<ob::SE2StateSpace::StateType>();
        start_se2->setX(start_vec[0].as<double>());
        start_se2->setY(start_vec[1].as<double>());
        start_se2->setYaw(start_vec.size() > 2 ? start_vec[2].as<double>() : 0.0);
        start_states.push_back(start_state);

        const auto& goal_vec = robot_node["goal"];
        auto* goal_state = robot_si->getStateSpace()->allocState();
        auto* goal_se2 = goal_state->as<ob::SE2StateSpace::StateType>();
        goal_se2->setX(goal_vec[0].as<double>());
        goal_se2->setY(goal_vec[1].as<double>());
        goal_se2->setYaw(goal_vec.size() > 2 ? goal_vec[2].as<double>() : 0.0);
        goal_states.push_back(goal_state);

        ++robot_idx;
    }

    // --- CBS high-level paths ---
    auto decomp = std::make_shared<GridDecompositionImpl>(2, position_bounds, region_size);
    decomp->setStateSpace(robot_sis[0]->getStateSpace());

    auto write_failure = [&]() {
        YAML::Node out;
        out["solved"] = false;
        out["planning_time"] = 0.0;
        std::ofstream fout(outputFile);
        fout << out;
    };

    auto decomposeAllLeavesOneLevel = [&]() -> bool {
        std::function<void(int, std::vector<int>&)> collectLeaves = [&](int rid, std::vector<int>& out) {
            if (!decomp->hasDecomposed(rid)) { out.push_back(rid); return; }
            for (int child : decomp->getChildRegions(rid)) collectLeaves(child, out);
        };
        std::vector<int> leaves;
        for (int r = 0; r < decomp->getNumRegions(); ++r) collectLeaves(r, leaves);

        int min_depth = std::numeric_limits<int>::max();
        for (int r : leaves) min_depth = std::min(min_depth, decomp->getDecompositionDepth(r));
        if (min_depth >= max_refinement_levels) return false;

        for (int r : leaves)
            if (decomp->getDecompositionDepth(r) == min_depth)
                decomp->Decompose(r);
        return true;
    };

    auto computeForbiddenEdges = [&]() -> ForbiddenEdgeSet {
        ForbiddenEdgeSet forbidden;
        if (!check_transition_feasibility) return forbidden;
        float max_radius = 0.0f;
        for (auto& robot : robots) {
            for (size_t p = 0; p < robot->numParts(); ++p) {
                auto geom = robot->getCollisionGeometry(p);
                if (!geom) continue;
                geom->computeLocalAABB();
                const auto& aabb = geom->aabb_local;
                float hx = (aabb.max_[0] - aabb.min_[0]) / 2.0f;
                float hy = (aabb.max_[1] - aabb.min_[1]) / 2.0f;
                max_radius = std::max(max_radius, std::sqrt(hx * hx + hy * hy));
            }
        }
        double threshold = transition_feasibility_robot_size_multiplier * 2.0 * max_radius;
        int total = decomp->getTotalNumRegions();
        for (int a = 0; a < total; ++a) {
            if (!decomp->isLeafRegion(a)) continue;
            const auto ba = decomp->getCellBounds(a);
            std::vector<int> neighbors;
            decomp->getNeighbors(a, neighbors);
            for (int b : neighbors) {
                if (!decomp->isLeafRegion(b)) continue;
                const auto bb = decomp->getCellBounds(b);
                double seg_lo, seg_hi;
                if (std::abs(ba.high[0] - bb.low[0]) < 1e-9 || std::abs(bb.high[0] - ba.low[0]) < 1e-9) {
                    seg_lo = std::max(ba.low[1], bb.low[1]);
                    seg_hi = std::min(ba.high[1], bb.high[1]);
                } else {
                    seg_lo = std::max(ba.low[0], bb.low[0]);
                    seg_hi = std::min(ba.high[0], bb.high[0]);
                }
                if (seg_hi <= seg_lo) continue;
                bool shared_in_y = (std::abs(ba.high[0] - bb.low[0]) < 1e-9 ||
                                     std::abs(bb.high[0] - ba.low[0]) < 1e-9);
                // Coordinate of the shared boundary in the perpendicular dimension.
                double boundary_perp = shared_in_y
                    ? ((ba.high[0] < bb.high[0]) ? ba.high[0] : ba.low[0])
                    : ((ba.high[1] < bb.high[1]) ? ba.high[1] : ba.low[1]);
                std::vector<std::pair<double,double>> covered;
                for (auto* obs : obstacles) {
                    const auto& aabb = obs->getAABB();
                    // Skip obstacles that don't straddle the shared boundary.
                    if (shared_in_y) {
                        if ((double)aabb.min_[0] > boundary_perp || (double)aabb.max_[0] < boundary_perp) continue;
                    } else {
                        if ((double)aabb.min_[1] > boundary_perp || (double)aabb.max_[1] < boundary_perp) continue;
                    }
                    double lo = shared_in_y ? (double)aabb.min_[1] : (double)aabb.min_[0];
                    double hi = shared_in_y ? (double)aabb.max_[1] : (double)aabb.max_[0];
                    double clo = std::max(lo, seg_lo);
                    double chi = std::min(hi, seg_hi);
                    if (chi > clo) covered.emplace_back(clo, chi);
                }
                std::sort(covered.begin(), covered.end());
                double longest_free = 0.0, cur = seg_lo;
                for (auto& [clo, chi] : covered) {
                    if (clo > cur) longest_free = std::max(longest_free, clo - cur);
                    cur = std::max(cur, chi);
                }
                longest_free = std::max(longest_free, seg_hi - cur);
                if (longest_free < threshold) {
                    forbidden.insert({a, b});
                    forbidden.insert({b, a});
                }
            }
        }
        return forbidden;
    };

    ForbiddenEdgeSet structurally_forbidden_edges = computeForbiddenEdges();

    std::vector<std::vector<int>> region_paths;
    YAML::Node viz_mapf_event;
    bool cbs_ok = false;
    bool cbs_failed = false;
    int cbs_attempts = 0;
    while (!cbs_ok) {
        try {
            CBS cbs(cbs_capacity, cbs_timeout, obstacles, max_obstacle_volume_percent);
            region_paths = cbs.solve(decomp, start_states, goal_states, {}, structurally_forbidden_edges);

            bool paths_valid = static_cast<int>(region_paths.size()) >= num_robots;
            if (paths_valid) {
                for (int r = 0; r < num_robots; ++r)
                    if (region_paths[r].empty()) { paths_valid = false; break; }
            }
            if (!paths_valid) throw std::runtime_error("incomplete paths");
            cbs_ok = true;
        } catch (const std::exception& e) {
            std::cerr << "[CBS] Failed (" << e.what()
                      << "); decomposing regions one level and retrying" << std::endl;
            if (!decomposeAllLeavesOneLevel()) {
                std::cerr << "[CBS] All cells at maximum decomposition depth; giving up" << std::endl;
                cbs_failed = true;
                break;
            }
            structurally_forbidden_edges = computeForbiddenEdges();
        }
        ++cbs_attempts;
    }

    if (cbs_failed) {
        write_failure();
        if (!vizFile.empty()) {
            YAML::Node viz_header;
            viz_header["dimensions"] = 2;
            YAML::Node viz_robots;
            for (int r = 0; r < num_robots; ++r) {
                const auto* s = start_states[r]->as<ob::SE2StateSpace::StateType>();
                const auto* g = goal_states[r]->as<ob::SE2StateSpace::StateType>();
                YAML::Node rn;
                rn["id"] = "r" + std::to_string(r);
                YAML::Node geom; geom["type"] = "sphere"; geom["radius"] = 0.15;
                rn["geometry"] = geom;
                YAML::Node start_node; start_node.push_back(s->getX()); start_node.push_back(s->getY()); start_node.push_back(0.0);
                YAML::Node goal_node;  goal_node.push_back(g->getX());  goal_node.push_back(g->getY());  goal_node.push_back(0.0);
                rn["start"] = start_node;
                rn["goal"]  = goal_node;
                viz_robots.push_back(rn);
            }
            viz_header["robots"] = viz_robots;
            YAML::Node viz_grid;
            YAML::Node viz_cells;
            const int num_regions = decomp->getTotalNumRegions();
            for (int rid = 0; rid < num_regions; ++rid) {
                const auto& rb = decomp->getCellBounds(rid);
                YAML::Node cn;
                cn["id"] = "c" + std::to_string(rid);
                YAML::Node bounds;
                YAML::Node bmin; bmin.push_back(rb.low[0]);  bmin.push_back(rb.low[1]);  bmin.push_back(0.0);
                YAML::Node bmax; bmax.push_back(rb.high[0]); bmax.push_back(rb.high[1]); bmax.push_back(0.0);
                bounds["min"] = bmin;
                bounds["max"] = bmax;
                cn["bounds"] = bounds;
                viz_cells.push_back(cn);
            }
            viz_grid["cells"] = viz_cells;
            viz_header["grid"] = viz_grid;
            YAML::Node viz_doc;
            viz_doc["header"] = viz_header;
            viz_doc["events"] = YAML::Node(YAML::NodeType::Sequence);
            try {
                std::ofstream vfout(vizFile);
                vfout << viz_doc;
            } catch (const std::exception& e) {
                std::cerr << "ERROR writing viz file: " << e.what() << std::endl;
            }
        }
        return 1;
    }

    if (!vizFile.empty()) {
        viz_mapf_event["type"] = "mapf";
        YAML::Node mapf_paths;
        for (int r = 0; r < num_robots; ++r) {
            YAML::Node cell_ids;
            for (int rid : region_paths[r])
                cell_ids.push_back("c" + std::to_string(rid));
            mapf_paths["r" + std::to_string(r)] = cell_ids;
        }
        viz_mapf_event["paths"] = mapf_paths;
    }

    // Map each robot's SI to its CBS region path for the planner allocator
    std::map<ob::SpaceInformationPtr, std::vector<int>> si_to_path;
    for (int r = 0; r < num_robots; ++r)
        si_to_path[robot_sis[r]] = region_paths[r];

    // --- Pass 2b: validity checkers and problem definitions ---
    auto ma_si   = std::make_shared<omrb::SpaceInformation>();
    auto ma_pdef = std::make_shared<omrb::ProblemDefinition>(ma_si);

    for (robot_idx = 0; robot_idx < num_robots; ++robot_idx) {
        auto robot    = robots[robot_idx];
        auto robot_si = robot_sis[robot_idx];

        std::vector<std::pair<const ob::State*, std::shared_ptr<Robot>>> other_goals;
        for (int j = 0; j < num_robots; ++j) {
            if (j != robot_idx) {
                if (!statesOverlap(goal_states[j], robots[j],
                                   start_states[robot_idx], robots[robot_idx])) {
                    other_goals.emplace_back(goal_states[j], robots[j]);
                }
            }
        }

        auto validity_checker = std::make_shared<IndividualStateValidityChecker>(
            robot_si, col_mng_environment, robot, all_robots, other_goals);
        robot_si->setStateValidityChecker(validity_checker);
        robot_si->setup();

        auto pdef = std::make_shared<ob::ProblemDefinition>(robot_si);
        pdef->addStartState(start_states[robot_idx]);
        pdef->setGoal(std::make_shared<IndividualGoalCondition>(
            robot_si, goal_states[robot_idx], goal_threshold));

        ma_si->addIndividual(robot_si);
        ma_pdef->addIndividual(pdef);
    }

    ma_si->lock();
    ma_pdef->lock();

    // Planner allocator: creates GuidedGeometricRRT with per-robot CBS region path
    ma_si->setPlannerAllocator([&](const ob::SpaceInformationPtr& si) -> ob::PlannerPtr {
        auto planner = std::make_shared<GuidedGeometricRRT>(si);
        planner->setGoalBias(goal_bias);
        planner->setDecomposition(decomp);
        planner->setDecompositionPath(si_to_path.at(si));
        return planner;
    });

    auto pp = std::make_shared<omrg::PP>(ma_si);
    pp->setProblemDefinition(ma_pdef);

    auto wall_start = std::chrono::steady_clock::now();
    bool solved = pp->as<omrb::Planner>()->solve(timelimit);
    auto wall_end = std::chrono::steady_clock::now();
    double planning_time = std::chrono::duration<double>(wall_end - wall_start).count();

    YAML::Node output;
    output["solved"]        = solved;
    output["planning_time"] = planning_time;

    if (solved) {
        omrb::PlanPtr solution = ma_pdef->getSolutionPlan();
        if (!solution) {
            std::cerr << "ERROR: solution plan is null!" << std::endl;
            return 1;
        }
        auto geom_plan = solution->as<omrg::PlanGeometric>();
        if (!geom_plan) {
            std::cerr << "ERROR: geom_plan cast failed!" << std::endl;
            return 1;
        }

        for (int r = 0; r < num_robots; ++r) {
            if (!geom_plan->getPath(r)) {
                solved = false;
            }
        }

        if (!solved) {
            output["solved"] = false;
        } else {
            YAML::Node result;
            for (int r = 0; r < num_robots; ++r) {
                og::PathGeometricPtr robot_path = geom_plan->getPath(r);
                robot_path->interpolate();

                YAML::Node states_node;
                for (size_t i = 0; i < robot_path->getStateCount(); ++i) {
                    const auto* se2 = robot_path->getState(i)->as<ob::SE2StateSpace::StateType>();
                    YAML::Node state_node;
                    state_node.push_back(se2->getX());
                    state_node.push_back(se2->getY());
                    state_node.push_back(se2->getYaw());
                    states_node.push_back(state_node);
                }
                YAML::Node robot_data;
                robot_data["states"] = states_node;
                result.push_back(robot_data);
            }
            output["result"] = result;
        }
    }

    try {
        std::ofstream fout(outputFile);
        fout << output;
        fout.close();
    } catch (const std::exception& e) {
        std::cerr << "ERROR writing output: " << e.what() << std::endl;
        return 1;
    }

    if (!vizFile.empty()) {
        YAML::Node viz_header;
        viz_header["dimensions"] = 2;

        // Robots
        YAML::Node viz_robots;
        for (int r = 0; r < num_robots; ++r) {
            const auto* s = start_states[r]->as<ob::SE2StateSpace::StateType>();
            const auto* g = goal_states[r]->as<ob::SE2StateSpace::StateType>();

            YAML::Node rn;
            rn["id"] = "r" + std::to_string(r);
            YAML::Node geom;
            geom["type"]   = "sphere";
            geom["radius"] = 0.15;
            rn["geometry"] = geom;

            YAML::Node start_node; start_node.push_back(s->getX()); start_node.push_back(s->getY()); start_node.push_back(0.0);
            YAML::Node goal_node;  goal_node.push_back(g->getX());  goal_node.push_back(g->getY());  goal_node.push_back(0.0);
            rn["start"] = start_node;
            rn["goal"]  = goal_node;
            viz_robots.push_back(rn);
        }
        viz_header["robots"] = viz_robots;

        // Grid cells
        YAML::Node viz_grid;
        YAML::Node viz_cells;
        const int num_regions = decomp->getTotalNumRegions();
        for (int rid = 0; rid < num_regions; ++rid) {
            const auto& rb = decomp->getCellBounds(rid);
            YAML::Node cn;
            cn["id"] = "c" + std::to_string(rid);
            YAML::Node bounds;
            YAML::Node bmin; bmin.push_back(rb.low[0]);  bmin.push_back(rb.low[1]);  bmin.push_back(0.0);
            YAML::Node bmax; bmax.push_back(rb.high[0]); bmax.push_back(rb.high[1]); bmax.push_back(0.0);
            bounds["min"] = bmin;
            bounds["max"] = bmax;
            cn["bounds"] = bounds;
            viz_cells.push_back(cn);
        }
        viz_grid["cells"] = viz_cells;
        viz_header["grid"] = viz_grid;

        // Events
        YAML::Node viz_events;
        viz_events.push_back(viz_mapf_event);

        if (solved) {
            omrb::PlanPtr solution = ma_pdef->getSolutionPlan();
            auto geom_plan = solution->as<omrg::PlanGeometric>();
            YAML::Node ev;
            ev["type"] = "low_level_paths";
            YAML::Node paths;
            for (int r = 0; r < num_robots; ++r) {
                og::PathGeometricPtr robot_path = geom_plan->getPath(r);
                YAML::Node waypoints;
                for (size_t j = 0; j < robot_path->getStateCount(); ++j) {
                    const auto* se2 = robot_path->getState(j)->as<ob::SE2StateSpace::StateType>();
                    YAML::Node wp;
                    YAML::Node state_node;
                    state_node.push_back(se2->getX());
                    state_node.push_back(se2->getY());
                    state_node.push_back(se2->getYaw());
                    wp["state"] = state_node;
                    waypoints.push_back(wp);
                }
                paths["r" + std::to_string(r)] = waypoints;
            }
            ev["paths"] = paths;
            viz_events.push_back(ev);
        }

        YAML::Node viz_doc;
        viz_doc["header"] = viz_header;
        viz_doc["events"] = viz_events;
        try {
            std::ofstream vfout(vizFile);
            vfout << viz_doc;
            vfout.close();
        } catch (const std::exception& e) {
            std::cerr << "ERROR writing viz file: " << e.what() << std::endl;
        }
    }

    // Cleanup
    for (auto* co : obstacles) delete co;
    for (int r = 0; r < num_robots; ++r) {
        robot_sis[r]->freeState(start_states[r]);
        robot_sis[r]->freeState(goal_states[r]);
    }

    return solved ? 0 : 1;
}
