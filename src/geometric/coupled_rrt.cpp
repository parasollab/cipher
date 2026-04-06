// OMPL base headers
#include <ompl/base/goals/GoalRegion.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/PathGeometric.h>

// FCL
#include <fcl/fcl.h>

// Standard library
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>

// YAML
#include <yaml-cpp/yaml.h>

// Boost
#include <boost/program_options.hpp>

// db-CBS robot dynamics
#include "../../db-CBS/src/robots.h"

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace po = boost::program_options;


class CompoundStateValidityChecker : public ob::StateValidityChecker
{
public:
    CompoundStateValidityChecker(
        const ob::SpaceInformationPtr& si,
        const std::shared_ptr<fcl::BroadPhaseCollisionManagerf>& col_mng_environment,
        const std::vector<std::shared_ptr<Robot>>& robots)
        : ob::StateValidityChecker(si),
          col_mng_environment_(col_mng_environment),
          robots_(robots) {}

    bool isValid(const ob::State* state) const override
    {
        // Check bounds
        if (!si_->satisfiesBounds(state)) {
            return false;
        }

        auto compound = state->as<ob::CompoundState>();

        // Check each robot against obstacles
        for (size_t i = 0; i < robots_.size(); ++i) {
            for (size_t part = 0; part < robots_[i]->numParts(); ++part) {
                const auto& transform = robots_[i]->getTransform(
                    compound->components[i], part);

                fcl::CollisionObjectf robot_co(robots_[i]->getCollisionGeometry(part));
                robot_co.setTranslation(transform.translation());
                robot_co.setRotation(transform.rotation());
                robot_co.computeAABB();

                fcl::DefaultCollisionData<float> collision_data;
                col_mng_environment_->collide(&robot_co, &collision_data,
                    fcl::DefaultCollisionFunction<float>);

                if (collision_data.result.isCollision()) {
                    return false;
                }
            }
        }

        // Check robot-robot collisions
        for (size_t i = 0; i < robots_.size(); ++i) {
            for (size_t j = i + 1; j < robots_.size(); ++j) {
                if (checkRobotRobotCollision(
                    compound->components[i],
                    compound->components[j],
                    robots_[i],
                    robots_[j])) {
                    return false;
                }
            }
        }

        return true;
    }

private:
    bool checkRobotRobotCollision(
        const ob::State* state_i,
        const ob::State* state_j,
        const std::shared_ptr<Robot>& robot_i,
        const std::shared_ptr<Robot>& robot_j) const
    {
        for (size_t part_i = 0; part_i < robot_i->numParts(); ++part_i) {
            for (size_t part_j = 0; part_j < robot_j->numParts(); ++part_j) {
                const auto& transform_i = robot_i->getTransform(state_i, part_i);
                const auto& transform_j = robot_j->getTransform(state_j, part_j);

                fcl::CollisionObjectf co_i(robot_i->getCollisionGeometry(part_i));
                co_i.setTranslation(transform_i.translation());
                co_i.setRotation(transform_i.rotation());
                co_i.computeAABB();

                fcl::CollisionObjectf co_j(robot_j->getCollisionGeometry(part_j));
                co_j.setTranslation(transform_j.translation());
                co_j.setRotation(transform_j.rotation());
                co_j.computeAABB();

                fcl::CollisionRequestf request;
                fcl::CollisionResultf result;
                fcl::collide(&co_i, &co_j, request, result);

                if (result.isCollision()) {
                    return true;
                }
            }
        }
        return false;
    }

    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> col_mng_environment_;
    std::vector<std::shared_ptr<Robot>> robots_;
};


// Goal condition that is satisfied when all robots are within threshold of their
// respective goals (measured as max per-robot distance in the composite space).
class CompoundGoalCondition : public ob::GoalRegion
{
public:
    CompoundGoalCondition(
        const ob::SpaceInformationPtr& si,
        const ob::CompoundStateSpace* css,
        const std::vector<ob::State*>& goal_states,
        double threshold)
        : ob::GoalRegion(si), css_(css), goal_states_(goal_states)
    {
        threshold_ = threshold;
    }

    double distanceGoal(const ob::State* st) const override
    {
        auto compound = st->as<ob::CompoundState>();
        double max_dist = 0.0;
        for (size_t i = 0; i < goal_states_.size(); ++i) {
            double dist = css_->getSubspace(i)->distance(
                compound->components[i], goal_states_[i]);
            if (dist > max_dist) {
                max_dist = dist;
            }
        }
        return max_dist;
    }

private:
    const ob::CompoundStateSpace* css_;
    std::vector<ob::State*> goal_states_;  // non-owning; lifetime managed by main
};


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

    // Create single SpaceInformation over the compound space
    auto compound_si = std::make_shared<ob::SpaceInformation>(compound_ss);
    compound_si->setStateValidityChecker(
        std::make_shared<CompoundStateValidityChecker>(
            compound_si, col_mng_environment, robots));
    compound_si->setup();

    // Allocate per-robot start/goal states and fill from YAML
    std::vector<ob::State*> start_states;
    std::vector<ob::State*> goal_states;
    robot_idx = 0;

    for (const auto& robot_node : env["robots"]) {
        auto& ss = robot_spaces[robot_idx];

        const auto& start_vec = robot_node["start"];
        auto start_state = ss->allocState();
        auto start_se2 = start_state->as<ob::SE2StateSpace::StateType>();
        start_se2->setX(start_vec[0].as<double>());
        start_se2->setY(start_vec[1].as<double>());
        start_se2->setYaw(start_vec.size() > 2 ? start_vec[2].as<double>() : 0.0);
        start_states.push_back(start_state);

        const auto& goal_vec = robot_node["goal"];
        auto goal_state = ss->allocState();
        auto goal_se2 = goal_state->as<ob::SE2StateSpace::StateType>();
        goal_se2->setX(goal_vec[0].as<double>());
        goal_se2->setY(goal_vec[1].as<double>());
        goal_se2->setYaw(goal_vec.size() > 2 ? goal_vec[2].as<double>() : 0.0);
        goal_states.push_back(goal_state);

        std::cout << "  Robot " << robot_idx
                  << "  Start: (" << start_se2->getX() << ", " << start_se2->getY() << ")"
                  << "  Goal: ("  << goal_se2->getX()  << ", " << goal_se2->getY()  << ")"
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
        compound_si, compound_ss.get(), goal_states, goal_threshold));

    // Create and configure the RRT planner
    auto planner = std::make_shared<og::RRT>(compound_si);
    planner->setProblemDefinition(pdef);
    planner->setup();

    std::cout << "Planner configured. Starting search..." << std::endl;
    std::cout << "  Goal threshold: " << goal_threshold << std::endl;
    std::cout << "  Total time limit: " << timelimit << " seconds" << std::endl;

    // Solve
    auto start_time = std::chrono::steady_clock::now();
    ob::PlannerStatus status = planner->solve(ob::timedPlannerTerminationCondition(timelimit));
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

        auto path = pdef->getSolutionPath()->as<og::PathGeometric>();
        path->interpolate();

        std::cout << "  Path has " << path->getStateCount() << " states after interpolation" << std::endl;

        // Build per-robot state lists from the compound path
        std::vector<YAML::Node> robot_states(num_robots);

        for (size_t s = 0; s < path->getStateCount(); ++s) {
            const auto* compound = path->getState(s)->as<ob::CompoundState>();
            for (int r = 0; r < num_robots; ++r) {
                const auto* se2 = compound->components[r]->as<ob::SE2StateSpace::StateType>();
                YAML::Node state_node;
                state_node.push_back(se2->getX());
                state_node.push_back(se2->getY());
                state_node.push_back(se2->getYaw());
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
