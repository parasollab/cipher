#include <fstream>
#include <iostream>
#include <memory>

#include <boost/program_options.hpp>
#include <ompl/util/RandomNumbers.h>

#include <yaml-cpp/yaml.h>

#include "utils/decomposition.h"
#include "utils/grid_decomposition.h"

#include "dynobench/general_utils.hpp"
#include "dynobench/motions.hpp"
#include "dynobench/robot_models.hpp"
// #include "dynoplan/dbrrt/dbrrt.hpp"
#include "guided/guided_dbrrt.hpp"
#include "dynoplan/optimization/ocp.hpp"

#include "robots.h"
#include "mapf/cbs.h"
#include "fclStateValidityChecker.hpp"
#include "robotStatePropagator.hpp"


namespace ob = ompl::base;
namespace po = boost::program_options;
using namespace dynobench;
using namespace dynoplan;

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


int main(int argc, char *argv[]) {
  Options_trajopt options_trajopt;
  Options_dbrrt options_dbrrt;
  po::options_description desc("Allowed options");
  std::string cfg_file, results_file, env_file, models_base_path, vizFile;
  set_from_boostop(desc, VAR_WITH_NAME(cfg_file));
  set_from_boostop(desc, VAR_WITH_NAME(results_file));
  set_from_boostop(desc, VAR_WITH_NAME(env_file));
  set_from_boostop(desc, VAR_WITH_NAME(models_base_path));
  desc.add_options()("vizFile,z", po::value<std::string>(&vizFile), "Visualization log output YAML file");
  options_trajopt.add_options(desc);
  options_dbrrt.add_options(desc);

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
      return 0;
    }
  } catch (po::error &e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }

  if (cfg_file != "") {
    options_dbrrt.read_from_yaml(cfg_file.c_str());
    options_trajopt.read_from_yaml(cfg_file.c_str());
  }

  Problem problem(env_file.c_str());
  problem.models_base_path = models_base_path;
  Trajectory traj;
  Info_out out_db;

  std::cout << "*** options_dbrrt ***" << std::endl;
  options_dbrrt.print(std::cout);
  std::cout << "***" << std::endl;

  options_trajopt.smooth_traj = false;
  options_trajopt.region_bounds_weight = 0.0;
  // options_trajopt.solver_id = int(dynoplan::SOLVER::mpc);
  std::cout << "*** options_trajopt ***" << std::endl;
  options_trajopt.print(std::cout);
  std::cout << "***" << std::endl;

  std::vector<Motion> motions;

  std::shared_ptr<dynobench::Model_robot> robot = dynobench::robot_factory(
      (problem.models_base_path + problem.robotType + ".yaml").c_str(),
      problem.p_lb, problem.p_ub);

  load_env(*robot, problem);

  // Load YAML problem
  std::cout << "Loading YAML file: " << env_file << std::endl;
  YAML::Node env;
  try {
      env = YAML::LoadFile(env_file);
  } catch (const YAML::Exception& e) {
      std::cerr << "ERROR loading YAML file: " << e.what() << std::endl;
      return 1;
  }

  std::cout << "YAML file loaded successfully." << std::endl;

  const auto& env_min = env["environment"]["min"];
  const auto& env_max = env["environment"]["max"];

  ob::RealVectorBounds position_bounds(2);
  position_bounds.setLow(0, env_min[0].as<double>());
  position_bounds.setLow(1, env_min[1].as<double>());
  position_bounds.setHigh(0, env_max[0].as<double>());
  position_bounds.setHigh(1, env_max[1].as<double>());

  std::cout << "Set Bounds" << std::endl;

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

  auto r = env["robots"][0];
  auto robot_type = r["type"].as<std::string>();
  auto temp_robot = create_robot(robot_type, position_bounds);
  auto validity_checker = std::make_shared<fclStateValidityChecker>(
            temp_robot->getSpaceInformation(), col_mng_environment, temp_robot, false);
  temp_robot->getSpaceInformation()->setStateValidityChecker(validity_checker);
  temp_robot->getSpaceInformation()->setStatePropagator(std::make_shared<RobotStatePropagator>(temp_robot->getSpaceInformation(), temp_robot));
  temp_robot->getSpaceInformation()->setup();

  std::cout << "Created temp robot" << std::endl;

  auto decomp = std::make_shared<GridDecompositionImpl>(2, position_bounds, 5.0);
  decomp->setStateSpace(temp_robot->getSpaceInformation()->getStateSpace());

  std::cout << "Created decomposition" << std::endl;

  std::vector<ob::State*> start_states, goal_states;

  const auto& start_vec = r["start"];
  std::vector<double> start_reals;
  for (const auto& v : start_vec) start_reals.push_back(v.as<double>());
  auto* start_state = temp_robot->getSpaceInformation()->getStateSpace()->allocState();
  temp_robot->getSpaceInformation()->getStateSpace()->copyFromReals(start_state, start_reals);
  start_states.push_back(start_state);

  std::cout << "Loaded start state" << std::endl; 

  const auto& goal_vec = r["goal"];
  std::vector<double> goal_reals;
  for (const auto& v : goal_vec) goal_reals.push_back(v.as<double>());
  auto* goal_state = temp_robot->getSpaceInformation()->getStateSpace()->allocState();
  temp_robot->getSpaceInformation()->getStateSpace()->copyFromReals(goal_state, goal_reals);
  goal_states.push_back(goal_state);

  std::cout << "Loaded goal state" << std::endl;

  YAML::Node viz_header;
  std::vector<YAML::Node> viz_events;

  const bool do_viz = !vizFile.empty();
  const int num_robots = 1;

  if (do_viz) {
    viz_header["dimensions"] = 2;

    // Robots
    YAML::Node viz_robots;
    for (int r = 0; r < num_robots; ++r) {
        const auto& robot_node = env["robots"][r];
        std::string rtype = robot_node["type"].as<std::string>();

        std::vector<double> s_reals, g_reals;
        temp_robot->getSpaceInformation()->getStateSpace()->copyToReals(s_reals, start_states[r]);
        temp_robot->getSpaceInformation()->getStateSpace()->copyToReals(g_reals, goal_states[r]);

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

  load_motion_primitives_new(
      options_dbrrt.motionsFile, *robot, motions, options_dbrrt.max_motions,
      options_dbrrt.cut_actions, false, options_dbrrt.check_cols);

  options_dbrrt.motions_ptr = &motions;

  int cbs_capacity = 1;
  double cbs_timeout = 30.0;

  std::cout << "Running CBS to get region path..." << std::endl;  

  CBS cbs(cbs_capacity, cbs_timeout);
  std::vector<std::vector<int>> region_paths = cbs.solve(decomp, start_states, goal_states);

  std::cout << "CBS returned " << region_paths.size() << " region paths." << std::endl;

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

  dynoplan::guided_idbrrt(problem, robot, options_dbrrt, options_trajopt, temp_robot,
                          decomp, region_paths[0], traj, out_db);
  // dynoplan::guided_dbrrt(problem, robot, options_dbrrt, options_trajopt, temp_robot,
  //                         decomp, region_paths[0], traj, out_db);

  std::cout << "guided_dbrrt returned with traj of size " << traj.states.size() << std::endl;

  std::cout << "*** info_out ***" << std::endl;
  out_db.to_yaml(std::cout);
  std::cout << "***" << std::endl;

  std::cout << "Raw solved: " << out_db.solved_raw << std::endl;
  std::cout << "Optimized solved: " << out_db.solved << std::endl;

  auto make_waypoints = [](const dynobench::Trajectory& tr) {
    YAML::Node waypoints;
    const size_t n = tr.states.size();
    for (size_t i = 0; i < n; ++i) {
      YAML::Node wp;
      YAML::Node state_node;
      for (int k = 0; k < tr.states[i].size(); ++k)
        state_node.push_back(tr.states[i][k]);
      while ((int)state_node.size() < 3) state_node.push_back(0.0);
      wp["state"] = state_node;
      YAML::Node ctrl;
      if (i < tr.actions.size()) {
        for (int k = 0; k < tr.actions[i].size(); ++k)
          ctrl.push_back(tr.actions[i][k]);
      } else {
        ctrl.push_back(0.0); ctrl.push_back(0.0);
      }
      wp["control"] = ctrl;
      double dur = 0.0;
      if (i + 1 < n)
        dur = (tr.times.size() > 0) ? tr.times[i + 1] - tr.times[i] : 0.1;
      wp["duration"] = dur;
      waypoints.push_back(wp);
    }
    return waypoints;
  };

  if (do_viz && out_db.solved_raw && !out_db.solved) {
    const auto& raw_traj = !out_db.trajs_raw.empty() ? out_db.trajs_raw[0] : traj;
    YAML::Node ev;
    ev["type"] = "raw_trajectory";
    YAML::Node paths;
    paths["r0"] = make_waypoints(raw_traj);
    ev["paths"] = paths;
    viz_events.push_back(ev);
    vizWriteFile(vizFile, viz_header, viz_events);
    std::cout << "[viz] raw_trajectory event written to " << vizFile << std::endl;
  }

  if (do_viz && out_db.solved) {
    YAML::Node ev;
    ev["type"] = "low_level_paths";
    YAML::Node paths;
    paths["r0"] = make_waypoints(out_db.trajs_opt[0]);
    ev["paths"] = paths;
    viz_events.push_back(ev);
    vizWriteFile(vizFile, viz_header, viz_events);
    std::cout << "[viz] low_level_paths event (optimized) written to " << vizFile << std::endl;
  }

  CSTR_(results_file);
  std::ofstream results(results_file);
  results << "alg: guided_dbrrt" << std::endl;
  results << "time_stamp: " << get_time_stamp() << std::endl;
  results << "env_file: " << env_file << std::endl;
  results << "cfg_file: " << cfg_file << std::endl;
  results << "results_file: " << results_file << std::endl;
  results << "options dbrrt:" << std::endl;
  options_dbrrt.print(results, "  ");
  results << "options trajopt:" << std::endl;
  options_trajopt.print(results, "  ");
  out_db.to_yaml(results);

  return out_db.solved ? EXIT_SUCCESS : EXIT_FAILURE;
}
