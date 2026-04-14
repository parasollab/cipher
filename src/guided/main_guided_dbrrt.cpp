#include <fstream>
#include <iostream>
#include <memory>

#include <boost/program_options.hpp>

#include "dynobench/general_utils.hpp"
#include "dynobench/motions.hpp"
#include "dynobench/robot_models.hpp"
#include "dynoplan/dbrrt/dbrrt.hpp"
#include "dynoplan/optimization/ocp.hpp"

namespace po = boost::program_options;
using namespace dynobench;
using namespace dynoplan;

namespace dynoplan {
void guided_idbrrt(const dynobench::Problem &problem,
                   std::shared_ptr<dynobench::Model_robot> robot,
                   const Options_dbrrt &options_dbrrt,
                   const Options_trajopt &options_trajopt,
                   dynobench::Trajectory &traj_out,
                   dynobench::Info_out &info_out);
} // namespace dynoplan

int main(int argc, char *argv[]) {
  Options_trajopt options_trajopt;
  Options_dbrrt options_dbrrt;
  po::options_description desc("Allowed options");
  std::string cfg_file, results_file, env_file, models_base_path;
  set_from_boostop(desc, VAR_WITH_NAME(cfg_file));
  set_from_boostop(desc, VAR_WITH_NAME(results_file));
  set_from_boostop(desc, VAR_WITH_NAME(env_file));
  set_from_boostop(desc, VAR_WITH_NAME(models_base_path));
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

  std::vector<Motion> motions;

  std::shared_ptr<dynobench::Model_robot> robot = dynobench::robot_factory(
      (problem.models_base_path + problem.robotType + ".yaml").c_str(),
      problem.p_lb, problem.p_ub);

  load_env(*robot, problem);

  load_motion_primitives_new(
      options_dbrrt.motionsFile, *robot, motions, options_dbrrt.max_motions,
      options_dbrrt.cut_actions, false, options_dbrrt.check_cols);

  options_dbrrt.motions_ptr = &motions;

  dynoplan::guided_idbrrt(problem, robot, options_dbrrt, options_trajopt, traj,
                          out_db);

  std::cout << "*** info_out ***" << std::endl;
  out_db.to_yaml(std::cout);
  std::cout << "***" << std::endl;

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
