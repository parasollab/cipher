// OMPL base headers
#include <ompl/base/goals/GoalRegion.h>
#include <ompl/base/spaces/SE2StateSpace.h>

// FCL
#include <fcl/fcl.h>

// Standard library
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>

// YAML
#include <yaml-cpp/yaml.h>

// Boost
#include <boost/program_options.hpp>

// db-CBS robot dynamics
#include "robots.h"
#include "fclStateValidityChecker.hpp"
#include "robotStatePropagator.hpp"

// Guided planner + decomposition + CBS
#include "guided_control_rrt.h"
#include "utils/grid_decomposition.h"
#include "mapf/cbs.h"

namespace ob = ompl::base;
namespace oc = ompl::control;
namespace po = boost::program_options;

// ---------------------------------------------------------------------------
// Helpers for building the visualization YAML log (see visualization/README.md)
// ---------------------------------------------------------------------------
static void vizWriteFile(const std::string& path, const YAML::Node& header,
                         const std::vector<YAML::Node>& events)
{
    YAML::Node doc;
    doc["header"] = header;
    YAML::Node evs;
    for (const auto& e : events) evs.push_back(e);
    doc["events"] = evs;
    std::ofstream fout(path);
    fout << doc;
}

static YAML::Node makeVec3(double x, double y, double z = 0.0)
{
    YAML::Node n;
    n.push_back(x); n.push_back(y); n.push_back(z);
    return n;
}


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
    const ob::State* goal_state_;  // non-owning; lifetime managed by goal_states vector in main
};


int main(int argc, char** argv)
{
    // Parse command line arguments
    std::string inputFile;
    std::string outputFile;
    std::string configFile;
    std::string vizFile;
    double timelimit = 60.0;
    double goal_threshold = 0.5;
    double region_size = 5.0;
    double cbs_timeout = 30.0;
    int cbs_capacity = 1;
    double max_obstacle_volume_percent = 1.0;
    int seed = -1;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Show help message")
        ("input,i", po::value<std::string>(&inputFile)->required(), "Input YAML file")
        ("output,o", po::value<std::string>(&outputFile)->required(), "Output YAML file")
        ("cfg,c", po::value<std::string>(&configFile), "Configuration YAML file")
        ("viz,v", po::value<std::string>(&vizFile), "Visualization log output YAML file");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }

    // Load configuration file if provided
    if (vm.count("cfg")) {
        try {
            YAML::Node cfg = YAML::LoadFile(configFile);
            if (cfg["goal_threshold"])  goal_threshold = cfg["goal_threshold"].as<double>();
            if (cfg["timelimit"])       timelimit       = cfg["timelimit"].as<double>();
            if (cfg["region_size"])     region_size     = cfg["region_size"].as<double>();
            if (cfg["cbs_timeout"])     cbs_timeout     = cfg["cbs_timeout"].as<double>();
            if (cfg["cbs_capacity"])    cbs_capacity    = cfg["cbs_capacity"].as<int>();
            if (cfg["max_obstacle_volume_percent"]) max_obstacle_volume_percent = cfg["max_obstacle_volume_percent"].as<double>();
            if (cfg["seed"])            seed            = cfg["seed"].as<int>();
        } catch (const YAML::Exception& e) {
            std::cerr << "ERROR loading config file: " << e.what() << std::endl;
            return 1;
        }
    }

    if (seed >= 0) {
        std::cout << "Setting random seed to: " << seed << std::endl;
        ompl::RNG::setSeed(seed);
    }

    // Load YAML problem
    std::cout << "Loading YAML file: " << inputFile << std::endl;
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
    position_bounds.setLow(0, env_min[0].as<double>());
    position_bounds.setLow(1, env_min[1].as<double>());
    position_bounds.setHigh(0, env_max[0].as<double>());
    position_bounds.setHigh(1, env_max[1].as<double>());

    // Create FCL collision manager for obstacles
    auto col_mng_environment = std::make_shared<fcl::DynamicAABBTreeCollisionManagerf>();
    std::vector<fcl::CollisionObjectf*> obstacles;

    if (env["environment"]["obstacles"]) {
        for (const auto& obs : env["environment"]["obstacles"]) {
            if (obs["type"].as<std::string>() == "box") {
                const auto& size   = obs["size"];
                const auto& center = obs["center"];

                auto box = std::make_shared<fcl::Boxf>(
                    size[0].as<float>(),
                    size[1].as<float>(),
                    1.0f);

                auto* co = new fcl::CollisionObjectf(box);
                co->setTranslation(fcl::Vector3f(
                    center[0].as<float>(),
                    center[1].as<float>(),
                    0.0f));
                co->computeAABB();

                obstacles.push_back(co);
                col_mng_environment->registerObject(co);
            }
        }
        col_mng_environment->setup();
    }

    std::cout << "Loaded " << obstacles.size() << " obstacles" << std::endl;

    // --- Pass 1: create robots and SpaceInformation ---
    // Each robot's state space comes from create_robot() so that the decomposition,
    // validity checker, and planner all work in the robot's native state space.
    std::vector<std::shared_ptr<Robot>> robots;
    std::vector<oc::SpaceInformationPtr> robot_sis;
    int robot_idx = 0;

    for (const auto& robot_node : env["robots"]) {
        auto robotType = robot_node["type"].as<std::string>();
        std::cout << "  Robot " << robot_idx << " (" << robotType << ")" << std::endl;

        auto robot = create_robot(robotType, position_bounds);
        auto state_space = robot->getSpaceInformation()->getStateSpace();
        state_space->setName("Robot " + std::to_string(robot_idx));

        // auto robot_si = std::make_shared<ob::SpaceInformation>(state_space);
        auto robot_si = robot->getSpaceInformation();

        robots.push_back(robot);
        robot_sis.push_back(robot_si);
        ++robot_idx;
    }

    // --- Pass 2: configure validity checkers, parse start/goal ---
    std::vector<ob::State*> start_states;
    std::vector<ob::State*> goal_states;
    robot_idx = 0;

    for (const auto& robot_node : env["robots"]) {
        auto robot    = robots[robot_idx];
        auto robot_si = robot_sis[robot_idx];

        // Environment-only validity checker (inter-robot separation handled by CBS)
        auto validity_checker = std::make_shared<fclStateValidityChecker>(
            robot_si, col_mng_environment, robot, false);
        robot_si->setStateValidityChecker(validity_checker);
        robot_si->setStatePropagator(std::make_shared<RobotStatePropagator>(robot_si, robot));
        robot_si->setup();

        // Parse start/goal generically via copyFromReals so any state space is supported
        const auto& start_vec = robot_node["start"];
        std::vector<double> start_reals;
        for (const auto& v : start_vec) start_reals.push_back(v.as<double>());
        auto* start_state = robot_si->getStateSpace()->allocState();
        robot_si->getStateSpace()->copyFromReals(start_state, start_reals);
        start_states.push_back(start_state);

        const auto& goal_vec = robot_node["goal"];
        std::vector<double> goal_reals;
        for (const auto& v : goal_vec) goal_reals.push_back(v.as<double>());
        auto* goal_state = robot_si->getStateSpace()->allocState();
        robot_si->getStateSpace()->copyFromReals(goal_state, goal_reals);
        goal_states.push_back(goal_state);

        std::cout << "  Robot " << robot_idx
                  << "  Start: (" << start_reals[0] << ", " << start_reals[1] << ")"
                  << "  Goal: ("  << goal_reals[0]  << ", " << goal_reals[1]  << ")"
                  << std::endl;

        ++robot_idx;
    }

    const int num_robots = static_cast<int>(robots.size());
    std::cout << "Planning for " << num_robots << " robots" << std::endl;

    // Create grid decomposition and wire up the robot state space so that
    // project() and sampleFullState() work generically (x,y = reals[0,1])
    auto decomp = std::make_shared<GridDecompositionImpl>(2, position_bounds, region_size);
    decomp->setStateSpace(robot_sis[0]->getStateSpace());  // all robots share the same type
    std::cout << "Grid decomposition: " << decomp->getNumRegions()
              << " regions (cell size " << region_size << ")" << std::endl;

    // ------------------------------------------------------------------
    // Build visualization header (robots + grid cells).
    // Written to vizFile whenever we have something meaningful to show.
    // ------------------------------------------------------------------
    const bool do_viz = !vizFile.empty();
    YAML::Node viz_header;
    std::vector<YAML::Node> viz_events;

    if (do_viz) {
        viz_header["dimensions"] = 2;

        // Robots
        YAML::Node viz_robots;
        for (int r = 0; r < num_robots; ++r) {
            const auto& robot_node = env["robots"][r];
            std::string rtype = robot_node["type"].as<std::string>();

            std::vector<double> s_reals, g_reals;
            robot_sis[r]->getStateSpace()->copyToReals(s_reals, start_states[r]);
            robot_sis[r]->getStateSpace()->copyToReals(g_reals, goal_states[r]);

            YAML::Node rn;
            rn["id"] = "r" + std::to_string(r);
            rn["dynamics"] = rtype;
            YAML::Node geom;
            geom["type"] = "sphere";
            geom["radius"] = 0.15;
            rn["geometry"] = geom;
            rn["start"] = makeVec3(s_reals[0], s_reals[1]);
            rn["goal"]  = makeVec3(g_reals[0], g_reals[1]);
            viz_robots.push_back(rn);
        }
        viz_header["robots"] = viz_robots;

        // Grid cells
        YAML::Node viz_grid;
        YAML::Node viz_cells;
        const int num_regions = decomp->getNumRegions();
        for (int rid = 0; rid < num_regions; ++rid) {
            const auto& rb = decomp->getBoundsForRegion(rid);
            YAML::Node cn;
            cn["id"] = "c" + std::to_string(rid);
            YAML::Node bounds;
            bounds["min"] = makeVec3(rb.low[0], rb.low[1]);
            bounds["max"] = makeVec3(rb.high[0], rb.high[1]);
            cn["bounds"] = bounds;
            viz_cells.push_back(cn);
        }
        viz_grid["cells"] = viz_cells;
        viz_header["grid"] = viz_grid;

        // Write immediately so the grid is visible even before planning
        vizWriteFile(vizFile, viz_header, viz_events);
        std::cout << "[viz] Header written (" << num_regions << " cells) to " << vizFile << std::endl;
    }

    // Run CBS to get high-level region paths
    std::cout << "Running CBS (capacity=" << cbs_capacity
              << ", timeout=" << cbs_timeout << "s)..." << std::endl;

    CBS cbs(cbs_capacity, cbs_timeout, obstacles, max_obstacle_volume_percent);
    std::vector<std::vector<int>> region_paths = cbs.solve(decomp, start_states, goal_states);

    YAML::Node output;
    auto write_failure = [&]() {
        output["solved"] = false;
        output["planning_time"] = 0.0;
        std::ofstream fout(outputFile);
        fout << output;
    };

    if (static_cast<int>(region_paths.size()) < num_robots) {
        std::cerr << "CBS failed to find paths for all robots." << std::endl;
        write_failure();
        return 1;
    }
    for (int r = 0; r < num_robots; ++r) {
        if (region_paths[r].empty()) {
            std::cerr << "CBS returned empty path for robot " << r << std::endl;
            write_failure();
            return 1;
        }
        std::cout << "  Robot " << r << " CBS path: "
                  << region_paths[r].size() << " regions" << std::endl;
    }

    // Emit mapf event
    if (do_viz) {
        YAML::Node ev;
        ev["type"] = "mapf";
        YAML::Node paths;
        for (int r = 0; r < num_robots; ++r) {
            YAML::Node cell_ids;
            for (int rid : region_paths[r])
                cell_ids.push_back("c" + std::to_string(rid));
            paths["r" + std::to_string(r)] = cell_ids;
        }
        ev["paths"] = paths;
        viz_events.push_back(ev);
        vizWriteFile(vizFile, viz_header, viz_events);
        std::cout << "[viz] mapf event written to " << vizFile << std::endl;
    }

    // Solve each robot with GuidedControlRRT (decoupled, sequential)
    auto wall_start = std::chrono::steady_clock::now();

    bool all_solved = true;
    std::vector<std::vector<std::vector<double>>> robot_paths(num_robots);

    for (int r = 0; r < num_robots; ++r) {
        std::cout << "Planning robot " << r << "..." << std::endl;

        auto robot_si = robot_sis[r];

        auto pdef = std::make_shared<ob::ProblemDefinition>(robot_si);
        pdef->addStartState(start_states[r]);
        pdef->setGoal(std::make_shared<IndividualGoalCondition>(
            robot_si, goal_states[r], goal_threshold));

        auto planner = std::make_shared<GuidedControlRRT>(robot_si);
        // auto planner = std::make_shared<oc::RRT>(robot_si);
        planner->setIntermediateStates(true);
        planner->setDecomposition(decomp);
        planner->setDecompositionPath(region_paths[r]);
        planner->setProblemDefinition(pdef);
        planner->setup();

        ob::PlannerStatus status = planner->solve(
            ob::timedPlannerTerminationCondition(timelimit));

        if (status == ob::PlannerStatus::EXACT_SOLUTION ||
            status == ob::PlannerStatus::APPROXIMATE_SOLUTION) {
            auto path = pdef->getSolutionPath()->as<oc::PathControl>();
            path->interpolate();

            for (size_t i = 0; i < path->getStateCount(); ++i) {
                std::vector<double> reals;
                robot_si->getStateSpace()->copyToReals(reals, path->getState(i));
                robot_paths[r].push_back(reals);
            }
            std::cout << "  Robot " << r << ": solved with "
                      << robot_paths[r].size() << " states" << std::endl;

            // Emit low_level_paths event for this robot
            if (do_viz) {
                YAML::Node ev;
                ev["type"] = "low_level_paths";
                YAML::Node paths;
                YAML::Node waypoints;
                const size_t n = robot_paths[r].size();
                for (size_t i = 0; i < n; ++i) {
                    const auto& st = robot_paths[r][i];
                    YAML::Node wp;
                    YAML::Node state_node;
                    for (double v : st) state_node.push_back(v);
                    // Pad to 3 components for the visualizer
                    while ((int)state_node.size() < 3) state_node.push_back(0.0);
                    wp["state"] = state_node;
                    YAML::Node ctrl; ctrl.push_back(0.0); ctrl.push_back(0.0);
                    wp["control"] = ctrl;
                    wp["duration"] = (i + 1 < n) ? 0.1 : 0.0;
                    waypoints.push_back(wp);
                }
                paths["r" + std::to_string(r)] = waypoints;
                ev["paths"] = paths;
                viz_events.push_back(ev);
                vizWriteFile(vizFile, viz_header, viz_events);
                std::cout << "[viz] low_level_paths event for robot " << r
                          << " written to " << vizFile << std::endl;
            }
        } else {
            std::cout << "  Robot " << r << ": FAILED" << std::endl;
            all_solved = false;
        }
    }

    auto wall_end = std::chrono::steady_clock::now();
    double planning_time = std::chrono::duration<double>(wall_end - wall_start).count();

    // Write output YAML
    output["solved"] = all_solved;
    output["planning_time"] = planning_time;

    if (all_solved) {
        YAML::Node result;
        for (int r = 0; r < num_robots; ++r) {
            YAML::Node robot_data;
            YAML::Node states_node;
            for (const auto& st : robot_paths[r]) {
                YAML::Node state_node;
                for (double v : st) state_node.push_back(v);
                states_node.push_back(state_node);
            }
            robot_data["states"] = states_node;
            result.push_back(robot_data);
        }
        output["result"] = result;
    }

    try {
        std::ofstream fout(outputFile);
        fout << output;
        fout.close();
        std::cout << "Output written to " << outputFile << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR writing output: " << e.what() << std::endl;
        return 1;
    }

    // Cleanup
    for (auto* co : obstacles) delete co;
    for (int r = 0; r < num_robots; ++r) {
        robot_sis[r]->freeState(start_states[r]);
        robot_sis[r]->freeState(goal_states[r]);
    }

    std::cout << "Done! time=" << planning_time << "s  solved=" << all_solved << std::endl;
    return all_solved ? 0 : 1;
}
