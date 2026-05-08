// OMPL base headers
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/control/planners/rrt/RRT.h>

// Standard library
#include <iostream>
#include <fstream>
#include <chrono>

// YAML
#include <yaml-cpp/yaml.h>

// Boost
#include <boost/program_options.hpp>

#include "coupled_rrt.h"

namespace po = boost::program_options;

KinoCoupledRRTConfig loadConfigFromYAML(const std::string& configFile)
{
    KinoCoupledRRTConfig config;

    try {
        YAML::Node cfg = YAML::LoadFile(configFile);
        if (cfg["timelimit"]) {
            config.time_limit = cfg["timelimit"].as<double>();
        }
        if (cfg["goal_threshold"]) {
            config.goal_threshold = cfg["goal_threshold"].as<double>();
        }
        if (cfg["propagation_step_size"]) {
            config.propagation_step_size = cfg["propagation_step_size"].as<double>();
        }
        if (cfg["control_duration_min"]) {
            config.control_duration_min = cfg["control_duration_min"].as<int>();
        }
        if (cfg["control_duration_max"]) {
            config.control_duration_max = cfg["control_duration_max"].as<int>();
        }
        if (cfg["seed"]) {
            config.seed = cfg["seed"].as<int>();
        }
        if (cfg["goal_bias"]) {
            config.goal_bias = cfg["goal_bias"].as<double>();
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "ERROR loading config file: " << e.what() << std::endl;
        throw;
    }

    return config;
}

#ifdef COUPLED_RRT_MAIN
int main(int argc, char** argv)
{
    // Parse command line arguments
    std::string inputFile;
    std::string outputFile;
    std::string configFile;
    KinoCoupledRRTConfig config;
    // double time_limit = 60.0;
    // double goal_threshold = 0.5;
    // double goal_bias = 0.05;
    // int seed = -1;
    // double propagation_step_size = 0.1;
    // int control_duration_min = 1;
    // int control_duration_max = 10;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Show help message")
        ("input,i", po::value<std::string>(&inputFile)->required(), "Input YAML file")
        ("output,o", po::value<std::string>(&outputFile)->required(), "Output YAML file")
        ("cfg,c", po::value<std::string>(&configFile), "Configuration YAML file");

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
            config = loadConfigFromYAML(configFile);
            // YAML::Node cfg = YAML::LoadFile(configFile);
            // if (cfg["goal_threshold"]) {
            //     goal_threshold = cfg["goal_threshold"].as<double>();
            // }
            // if (cfg["seed"]) {
            //     seed = cfg["seed"].as<int>();
            // }
            // if (cfg["time_limit"]) {
            //     time_limit = cfg["time_limit"].as<double>();
            // }
            // if (cfg["goal_bias"]) {
            //     goal_bias = cfg["goal_bias"].as<double>();
            // }
            // if (cfg["propagation_step_size"]) {
            //     propagation_step_size = cfg["propagation_step_size"].as<double>();
            // }
            // if (cfg["control_duration"]) {
            //     control_duration_min = cfg["control_duration"][0].as<int>();
            //     control_duration_max = cfg["control_duration"][1].as<int>();
            // }
        } catch (const YAML::Exception& e) {
            std::cerr << "ERROR loading config file: " << e.what() << std::endl;
            return 1;
        }
    }

    // Set the random seed
    if (config.seed >= 0) {
        std::cout << "Setting random seed to: " << config.seed << std::endl;
        ompl::RNG::setSeed(config.seed);
    } else {
        std::cout << "Using random seed" << std::endl;
    }

    // Load YAML configuration
    std::cout << "Loading YAML file: " << inputFile << std::endl;
    YAML::Node env;
    try {
        env = YAML::LoadFile(inputFile);
    } catch (const YAML::Exception& e) {
        std::cerr << "ERROR loading YAML file: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "YAML loaded successfully" << std::endl;

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
                const auto& size = obs["size"];
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

    // Build the compound state space and collect per-robot state spaces / robots
    auto compound_ss = std::make_shared<ob::CompoundStateSpace>();
    std::vector<std::shared_ptr<Robot>> robots;
    // Individual state spaces kept for allocating/freeing per-robot start/goal states
    std::vector<ob::StateSpacePtr> robot_spaces;

    std::cout << "Creating robots..." << std::endl;
    int robot_idx = 0;
    for (const auto& robot_node : env["robots"]) {
        auto robotType = robot_node["type"].as<std::string>();
        std::cout << "  Robot " << robot_idx << " (" << robotType << ")" << std::endl;

        auto robot = create_robot(robotType, position_bounds);
        auto state_space = robot->getSpaceInformation()->getStateSpace();
        state_space->setName("Robot " + std::to_string(robot_idx));

        compound_ss->addSubspace(state_space, 1.0);
        robots.push_back(robot);
        robot_spaces.push_back(state_space);
        robot_idx++;
    }

    const int num_robots = robots.size();
    std::cout << "Planning for " << num_robots << " robots" << std::endl;

    // Build compound control space from each robot's control space
    auto cspace = std::make_shared<oc::CompoundControlSpace>(compound_ss);
    for (auto& robot : robots) {
        cspace->addSubspace(robot->getSpaceInformation()->getControlSpace());
    }

    // Create kinodynamic SpaceInformation over the compound state+control spaces
    auto compound_si = std::make_shared<oc::SpaceInformation>(compound_ss, cspace);
    compound_si->setStateValidityChecker(
        std::make_shared<CompoundStateValidityChecker>(
            compound_si, col_mng_environment, robots));
    compound_si->setStatePropagator(
        std::make_shared<CompoundStatePropagator>(compound_si, robots));
    compound_si->setPropagationStepSize(config.propagation_step_size);
    compound_si->setMinMaxControlDuration(config.control_duration_min, config.control_duration_max);
    compound_si->setup();

    // Allocate per-robot start/goal states and fill from YAML
    std::vector<ob::State*> start_states;
    std::vector<ob::State*> goal_states;
    robot_idx = 0;

    for (const auto& robot_node : env["robots"]) {
        // auto& ss = robot_spaces[robot_idx];
        auto robot_si = robots[robot_idx]->getSpaceInformation();

        // const auto& start_vec = robot_node["start"];
        // auto start_state = ss->allocState();
        // auto start_se2 = start_state->as<ob::SE2StateSpace::StateType>();
        // start_se2->setX(start_vec[0].as<double>());
        // start_se2->setY(start_vec[1].as<double>());
        // start_se2->setYaw(start_vec.size() > 2 ? start_vec[2].as<double>() : 0.0);
        // start_states.push_back(start_state);

        // const auto& goal_vec = robot_node["goal"];
        // auto goal_state = ss->allocState();
        // auto goal_se2 = goal_state->as<ob::SE2StateSpace::StateType>();
        // goal_se2->setX(goal_vec[0].as<double>());
        // goal_se2->setY(goal_vec[1].as<double>());
        // goal_se2->setYaw(goal_vec.size() > 2 ? goal_vec[2].as<double>() : 0.0);
        // goal_states.push_back(goal_state);

        // std::cout << "  Robot " << robot_idx
        //           << "  Start: (" << start_se2->getX() << ", " << start_se2->getY() << ")"
        //           << "  Goal: ("  << goal_se2->getX()  << ", " << goal_se2->getY()  << ")"
        //           << std::endl;

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

        robot_idx++;
    }

    // Build compound start state by copying per-robot states into components
    auto compound_start = compound_si->allocState();
    {
        auto cs = compound_start->as<ob::CompoundState>();
        for (int r = 0; r < num_robots; ++r) {
            robot_spaces[r]->copyState(cs->components[r], start_states[r]);
        }
    }

    // Problem definition
    auto pdef = std::make_shared<ob::ProblemDefinition>(compound_si);
    pdef->addStartState(compound_start);
    pdef->setGoal(std::make_shared<CompoundGoalCondition>(
        compound_si, goal_states, config.goal_threshold));

    // Create and configure the RRT planner
    auto planner = std::make_shared<oc::RRT>(compound_si);
    planner->setGoalBias(config.goal_bias);
    planner->setProblemDefinition(pdef);
    planner->setup();

    std::cout << "Planner configured. Starting search..." << std::endl;
    std::cout << "  Goal threshold: " << config.goal_threshold << std::endl;
    std::cout << "  Goal bias: " << config.goal_bias << std::endl;
    std::cout << "  Total time limit: " << config.time_limit << " seconds" << std::endl;

    // Solve
    auto start_time = std::chrono::steady_clock::now();
    ob::PlannerStatus status = planner->solve(ob::timedPlannerTerminationCondition(config.time_limit));
    auto end_time = std::chrono::steady_clock::now();
    double planning_time = std::chrono::duration<double>(end_time - start_time).count();

    bool solved = (status == ob::PlannerStatus::EXACT_SOLUTION);

    std::cout << "Planning completed in " << planning_time << " seconds" << std::endl;
    std::cout << "Solution found: " << (solved ? "YES" : "NO") << std::endl;

    // Write output
    YAML::Node output;
    output["solved"] = solved;
    output["planning_time"] = planning_time;

    if (solved) {
        std::cout << "Extracting solution path..." << std::endl;

        auto path = pdef->getSolutionPath()->as<oc::PathControl>();
        path->interpolate();

        std::cout << "  Path has " << path->getStateCount() << " states after interpolation" << std::endl;

        // Build per-robot state lists from the compound path
        std::vector<YAML::Node> robot_states(num_robots);

        for (size_t s = 0; s < path->getStateCount(); ++s) {
            const auto* compound = path->getState(s)->as<ob::CompoundState>();
            for (int r = 0; r < num_robots; ++r) {
                std::vector<double> reals;
                robot_spaces[r]->copyToReals(reals, compound->components[r]);
                YAML::Node state_node;
                for (double v : reals) state_node.push_back(v);
                robot_states[r].push_back(state_node);
            }
        }

        YAML::Node result;
        for (int r = 0; r < num_robots; ++r) {
            YAML::Node robot_data;
            robot_data["states"] = robot_states[r];
            result.push_back(robot_data);
        }
        output["result"] = result;

        std::cout << "Solution extracted successfully" << std::endl;
    }

    // Write output file
    try {
        std::ofstream fout(outputFile);
        fout << output;
        fout.close();
        std::cout << "Output written to " << outputFile << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR writing output file: " << e.what() << std::endl;
        return 1;
    }

    // Cleanup
    compound_si->freeState(compound_start);

    for (int r = 0; r < num_robots; ++r) {
        robot_spaces[r]->freeState(start_states[r]);
        robot_spaces[r]->freeState(goal_states[r]);
    }

    for (auto* co : obstacles) {
        delete co;
    }

    std::cout << "Done!" << std::endl;
    return 0;
}
#endif // COUPLED_RRT_MAIN
