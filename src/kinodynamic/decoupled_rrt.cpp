// Multi-Robot OMPL headers
#include <ompl/multirobot/base/SpaceInformation.h>
#include <ompl/multirobot/base/ProblemDefinition.h>
#include <ompl/multirobot/control/planners/pp/PP.h>
#include <ompl/multirobot/control/PlanControl.h>

// OMPL base headers
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/StateSpace.h>
#include <ompl/control/planners/rrt/RRT.h>

// Standard library
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include <chrono>

// YAML
#include <yaml-cpp/yaml.h>

// Boost
#include <boost/program_options.hpp>

// db-CBS robot dynamics
#include "robotStatePropagator.hpp"
#include "decoupled_rrt_utils.h"

namespace oc = ompl::control;
namespace omrb = ompl::multirobot::base;
namespace omrc = ompl::multirobot::control;
namespace po = boost::program_options;
static ob::PlannerPtr plannerAllocator(const ob::SpaceInformationPtr& si)
{
    auto planner = std::make_shared<oc::RRT>(std::static_pointer_cast<oc::SpaceInformation>(si));
    planner->setIntermediateStates(true);
    return planner;
}

int main(int argc, char** argv)
{
    // Parse command line arguments
    std::string inputFile;
    std::string outputFile;
    std::string configFile;
    double timelimit = 60.0;
    double goal_threshold = 0.5;
    int seed = -1;

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
            YAML::Node cfg = YAML::LoadFile(configFile);
            if (cfg["goal_threshold"]) {
                goal_threshold = cfg["goal_threshold"].as<double>();
            }
            if (cfg["seed"]) {
                seed = cfg["seed"].as<int>();
            }
            if (cfg["timelimit"]) {
                timelimit = cfg["timelimit"].as<double>();
            }
        } catch (const YAML::Exception& e) {
            std::cerr << "ERROR loading config file: " << e.what() << std::endl;
            return 1;
        }
    }

    // Set the random seed
    if (seed >= 0) {
        std::cout << "Setting random seed to: " << seed << std::endl;
        ompl::RNG::setSeed(seed);
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

    // Create multi-robot space information and problem definition
    auto ma_si = std::make_shared<omrc::SpaceInformation>();
    auto ma_pdef = std::make_shared<omrb::ProblemDefinition>(ma_si);

    // --- Pass 1: Create robots and build all_robots map ---
    // Each robot from create_robot() has an oc::SpaceInformation (control-based).
    // For geometric planning we create a fresh ob::SpaceInformation from the same
    // state space, avoiding the propagator requirement of oc::SpaceInformation::setup().
    std::cout << "Creating robots..." << std::endl;
    std::vector<std::shared_ptr<Robot>> robots;
    std::vector<oc::SpaceInformationPtr> robot_sis;
    std::map<const ob::SpaceInformationPtr, std::shared_ptr<Robot>> all_robots;
    int robot_idx = 0;

    for (const auto& robot_node : env["robots"]) {
        auto robotType = robot_node["type"].as<std::string>();
        std::cout << "  Robot " << robot_idx << " (" << robotType << ")" << std::endl;

        auto robot = create_robot(robotType, position_bounds);
        auto state_space = robot->getSpaceInformation()->getStateSpace();
        state_space->setName("Robot " + std::to_string(robot_idx));
        state_space->setup();  // populate valueLocations_ before copyFromReals in Pass 2a

        auto robot_si = robot->getSpaceInformation();

        all_robots[robot_si] = robot;
        robots.push_back(robot);
        robot_sis.push_back(robot_si);
        robot_idx++;
    }

    // --- Pass 2a: Parse start/goal states for all robots ---
    std::vector<ob::State*> start_states;
    std::vector<ob::State*> goal_states;
    robot_idx = 0;

    for (const auto& robot_node : env["robots"]) {
        auto robot_si = robot_sis[robot_idx];

        const auto& start_vec = robot_node["start"];
        std::vector<double> start_reals;
        for (const auto& v : start_vec) start_reals.push_back(v.as<double>());
        auto start_state = robot_si->getStateSpace()->allocState();
        robot_si->getStateSpace()->copyFromReals(start_state, start_reals);
        start_states.push_back(start_state);

        const auto& goal_vec = robot_node["goal"];
        std::vector<double> goal_reals;
        for (const auto& v : goal_vec) goal_reals.push_back(v.as<double>());
        auto goal_state = robot_si->getStateSpace()->allocState();
        robot_si->getStateSpace()->copyFromReals(goal_state, goal_reals);
        goal_states.push_back(goal_state);

        std::cout << "    Start: (" << start_reals[0] << ", " << start_reals[1] << ")" << std::endl;
        std::cout << "    Goal:  (" << goal_reals[0]  << ", " << goal_reals[1]  << ")" << std::endl;

        robot_idx++;
    }

    const int num_robots = robots.size();

    // --- Pass 2b: Create validity checkers and problem definitions ---
    // All goals are known now, so each checker can forbid other robots' goal positions.
    for (robot_idx = 0; robot_idx < num_robots; ++robot_idx) {
        auto robot = robots[robot_idx];
        auto robot_si = robot_sis[robot_idx];

        // Build list of every other robot's goal state for collision avoidance.
        // Skip robot j's goal if it overlaps with robot K's start — robot K is
        // already there at t=0 and will move away (dynamic checks handle timing).
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

        robot_si->setStatePropagator(std::make_shared<RobotStatePropagator>(robot_si, robot));
        robot_si->setup();

        auto pdef = std::make_shared<ob::ProblemDefinition>(robot_si);
        pdef->addStartState(start_states[robot_idx]);
        pdef->setGoal(std::make_shared<IndividualGoalCondition>(
            robot_si, goal_states[robot_idx], goal_threshold));

        ma_si->addIndividual(robot_si);
        ma_pdef->addIndividual(pdef);
    }
    std::cout << "Planning for " << num_robots << " robots" << std::endl;

    // Lock the multi-robot structures
    ma_si->lock();
    ma_pdef->lock();

    // Set planner allocator
    ma_si->setPlannerAllocator(plannerAllocator);

    // Create control PP planner
    auto planner = std::make_shared<omrc::PP>(ma_si);
    planner->setProblemDefinition(ma_pdef);

    std::cout << "Planner configured. Starting search..." << std::endl;
    std::cout << "  Goal threshold: " << goal_threshold << std::endl;
    std::cout << "  Total time limit: " << timelimit << " seconds" << std::endl;

    // Solve
    auto start_time = std::chrono::steady_clock::now();
    bool solved = planner->as<omrb::Planner>()->solve(timelimit);
    auto end_time = std::chrono::steady_clock::now();
    double planning_time = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "Planning completed in " << planning_time << " seconds" << std::endl;
    std::cout << "Solution found: " << (solved ? "YES" : "NO") << std::endl;

    // Write output
    YAML::Node output;
    output["solved"] = solved;
    output["planning_time"] = planning_time;

    if (solved) {
        std::cout << "Extracting solution paths..." << std::endl;

        // Get solution plan
        omrb::PlanPtr solution = ma_pdef->getSolutionPlan();
        if (!solution) {
            std::cerr << "ERROR: solution plan is null!" << std::endl;
            return 1;
        }

        auto ctrl_plan = solution->as<omrc::PlanControl>();
        if (!ctrl_plan) {
            std::cerr << "ERROR: ctrl_plan cast failed!" << std::endl;
            return 1;
        }

        // Verify all robots have paths — treat as failure if any are missing
        for (int r = 0; r < num_robots; ++r) {
            if (!ctrl_plan->getPath(r)) {
                std::cout << "Robot " << r << " has no path — treating as failure." << std::endl;
                solved = false;
            }
        }

        if (!solved) {
            std::cout << "Not all robots were solved. Marking as failure." << std::endl;
            output["solved"] = false;
        } else {
            // Extract paths for each robot
            YAML::Node result;
            for (int r = 0; r < num_robots; ++r) {
                YAML::Node robot_data;

                oc::PathControlPtr robot_path = ctrl_plan->getPath(r);

                if (robot_path) {
                    std::cout << "  Robot " << r << ": " << robot_path->getStateCount()
                            << " states before interpolation" << std::endl;

                    robot_path->interpolate();

                    std::cout << "  Robot " << r << ": " << robot_path->getStateCount()
                            << " states after interpolation" << std::endl;

                    // Extract states
                    YAML::Node states_node;
                    auto robot_space = robot_sis[r]->getStateSpace();
                    for (size_t i = 0; i < robot_path->getStateCount(); ++i) {
                        const ob::State* robot_state = robot_path->getState(i);
                        if (!robot_state) {
                            std::cerr << "ERROR: null state at index " << i << " for robot " << r << std::endl;
                            break;
                        }
                        std::vector<double> reals;
                        robot_space->copyToReals(reals, robot_state);

                        YAML::Node state_node;
                        for (double v : reals) state_node.push_back(v);
                        states_node.push_back(state_node);
                    }
                    robot_data["states"] = states_node;
                }

                result.push_back(robot_data);
            }
            output["result"] = result;

            std::cout << "Solution extracted successfully" << std::endl;
        } // end else (all robots solved)
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
    for (auto* co : obstacles) {
        delete co;
    }

    for (size_t i = 0; i < start_states.size(); ++i) {
        robot_sis[i]->freeState(start_states[i]);
    }

    for (size_t i = 0; i < goal_states.size(); ++i) {
        robot_sis[i]->freeState(goal_states[i]);
    }

    std::cout << "Done!" << std::endl;
    return 0;
}
