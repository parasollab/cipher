// Multi-Robot OMPL headers
#include <ompl/multirobot/base/SpaceInformation.h>
#include <ompl/multirobot/base/ProblemDefinition.h>
#include <ompl/multirobot/control/planners/pp/PP.h>
#include <ompl/multirobot/control/PlanControl.h>


// OMPL base headers
#include <ompl/base/goals/GoalRegion.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/control/planners/rrt/RRT.h>

// FCL
#include <fcl/fcl.h>

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
#include "robots.h"
#include "fclStateValidityChecker.hpp"
#include "robotStatePropagator.hpp"

namespace ob = ompl::base;
namespace oc = ompl::control;
namespace omrb = ompl::multirobot::base;
namespace omrc = ompl::multirobot::control;
namespace po = boost::program_options;


class IndividualStateValidityChecker : public fclStateValidityChecker
{
public:
    IndividualStateValidityChecker(
        ob::SpaceInformationPtr si,
        std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng_environment,
        std::shared_ptr<Robot> robot,
        std::map<const ob::SpaceInformationPtr, std::shared_ptr<Robot>> all_robots)
        : fclStateValidityChecker(si, col_mng_environment, robot, false)
        , all_robots_(all_robots)
    {}

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
        if (iter != all_robots_.end()) {
            auto other_robot = iter->second;
            if (!state2.second) {
                return true;
            }
            // Try reading the state as SE2 to verify it's valid memory before getTransform
            const auto* se2 = state2.second->as<ob::SE2StateSpace::StateType>();
            const auto& t2 = other_robot->getTransform(state2.second, part);
            fcl::CollisionObjectf co2(other_robot->getCollisionGeometry(part));
            co2.setTranslation(t2.translation());
            co2.setRotation(t2.rotation());
            co2.computeAABB();

            fcl::CollisionRequest<float> request;
            fcl::CollisionResult<float> result;
            fcl::collide(&co1, &co2, request, result);
            bool collision = result.isCollision();

            if (collision) {
                return false;
            }
        } else {
            return false;
        }
        return true;
    }

private:
    std::map<const ob::SpaceInformationPtr, std::shared_ptr<Robot>> all_robots_;
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
    const ob::State* goal_state_;  // non-owning; lifetime managed by goal_states vector in main
};


ob::PlannerPtr plannerAllocator(const ob::SpaceInformationPtr& si)
{
    return std::make_shared<oc::RRT>(std::static_pointer_cast<oc::SpaceInformation>(si));
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

        auto robot_si = robot->getSpaceInformation();

        all_robots[robot_si] = robot;
        robots.push_back(robot);
        robot_sis.push_back(robot_si);
        robot_idx++;
    }

    // --- Pass 2: Configure each robot, parse start/goal, create problem definitions ---
    std::vector<ob::State*> start_states;
    std::vector<ob::State*> goal_states;
    robot_idx = 0;

    for (const auto& robot_node : env["robots"]) {
        auto robot = robots[robot_idx];
        auto robot_si = robot_sis[robot_idx];

        // Set state validity checker with dynamic obstacle support
        auto validity_checker = std::make_shared<IndividualStateValidityChecker>(
            robot_si, col_mng_environment, robot, all_robots);
        robot_si->setStateValidityChecker(validity_checker);

        robot_si->setStatePropagator(std::make_shared<RobotStatePropagator>(robot_si, robot));

        // Setup the space information
        robot_si->setup();

        // Parse start state
        const auto& start_vec = robot_node["start"];
        auto start_state = robot_si->getStateSpace()->allocState();
        auto start_se2 = start_state->as<ob::SE2StateSpace::StateType>();
        start_se2->setX(start_vec[0].as<double>());
        start_se2->setY(start_vec[1].as<double>());
        start_se2->setYaw(start_vec.size() > 2 ? start_vec[2].as<double>() : 0.0);
        start_states.push_back(start_state);

        // Parse goal state
        const auto& goal_vec = robot_node["goal"];
        auto goal_state = robot_si->getStateSpace()->allocState();
        auto goal_se2 = goal_state->as<ob::SE2StateSpace::StateType>();
        goal_se2->setX(goal_vec[0].as<double>());
        goal_se2->setY(goal_vec[1].as<double>());
        goal_se2->setYaw(goal_vec.size() > 2 ? goal_vec[2].as<double>() : 0.0);
        goal_states.push_back(goal_state);

        // Create problem definition for this robot
        auto pdef = std::make_shared<ob::ProblemDefinition>(robot_si);
        pdef->addStartState(start_state);
        pdef->setGoal(std::make_shared<IndividualGoalCondition>(
            robot_si, goal_state, goal_threshold));

        // Add to multi-robot space information and problem definition
        ma_si->addIndividual(robot_si);
        ma_pdef->addIndividual(pdef);

        std::cout << "    Start: (" << start_se2->getX() << ", " << start_se2->getY() << ")" << std::endl;
        std::cout << "    Goal:  (" << goal_se2->getX() << ", " << goal_se2->getY() << ")" << std::endl;

        robot_idx++;
    }

    const int num_robots = robots.size();
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
                    for (size_t i = 0; i < robot_path->getStateCount(); ++i) {
                        const ob::State* robot_state = robot_path->getState(i);
                        if (!robot_state) {
                            std::cerr << "ERROR: null state at index " << i << " for robot " << r << std::endl;
                            break;
                        }
                        const ob::SE2StateSpace::StateType* se2_state =
                            robot_state->as<ob::SE2StateSpace::StateType>();

                        YAML::Node state_node;
                        state_node.push_back(se2_state->getX());
                        state_node.push_back(se2_state->getY());
                        state_node.push_back(se2_state->getYaw());
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
