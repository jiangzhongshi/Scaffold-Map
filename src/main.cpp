#include "ReWeightedARAP.h"
#include "StateManager.h"
#include "UI/DeformGUI.h"
#include "UI/TextureGUI.h"
#include "util/triangle_utils.h"
#include <igl/Timer.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <memory>
#include <CLI/CLI.hpp>


int main(int argc, char* argv[]) {
  CLI::App command_line{"Scaffold Mapping"};
  std::string filename = "";
  int demo_type = 1;
  command_line.add_option("-m,--mesh", filename, "Mesh path")->check(CLI::ExistingFile);
  command_line.add_option("-t,--type", demo_type, "demo type");
  try {
      command_line.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
      return command_line.exit(e);
  }

  StateManager s_(filename, static_cast<DemoType>(demo_type));
#ifndef NOGUI
  bool complex_ui = false;
  if(complex_ui) {
    TextureGUI gui(s_);
    gui.launch();
  } else {
    igl::opengl::glfw::Viewer v;

    DeformGUI gui(v, s_);
    v.launch();
  }
#else
  using namespace Eigen;
  using namespace std;
  auto &d_ = s_.scaf_data;
  auto &ws = s_.ws_solver;
  double last_mesh_energy =  ws->compute_energy(d_.w_uv, false) / d_.mesh_measure;
  std::cout<<"Initial Energy"<<last_mesh_energy<<std::endl;
  cout << "Initial V_num: "<<d_.mv_num<<" F_num: "<<d_.mf_num<<endl;
  d_.energy = ws->compute_energy(d_.w_uv, true) / d_.mesh_measure;
  while (s_.iter_count < 50) {
    std::cout << "=============" << std::endl;
    std::cout << "Iteration:" << s_.iter_count++ << '\t';
    igl::Timer timer;

    timer.start();

    ws->mesh_improve();

    double new_weight = d_.mesh_measure * last_mesh_energy / (d_.sf_num*100 );
    ws->adjust_scaf_weight(new_weight);

    d_.energy = ws->perform_iteration(d_.w_uv);

    cout<<"Iteration time = "<<timer.getElapsedTime()<<endl;
    double current_mesh_energy =
        ws->compute_energy(d_.w_uv, false) / d_.mesh_measure - 4;
    double mesh_energy_decrease = last_mesh_energy - current_mesh_energy;
    cout << "Energy After:"
         << d_.energy - 4
         << "\tMesh Energy:"
         << current_mesh_energy
         << "\tEnergy Decrease"
         << mesh_energy_decrease
         << endl;
    cout << "V_num: "<<d_.v_num<<" F_num: "<<d_.f_num<<endl;
    last_mesh_energy = current_mesh_energy;
  }

  MatrixXd wuv3 = MatrixXd::Zero(d_.v_num,3);
  wuv3.leftCols(2) = d_.w_uv;
  s_.save(filename+".ser");
  igl::writeOBJ(filename+"_uv.obj", wuv3, d_.m_T);
  std::cout<<"END"<<std::endl;
#endif
  return 0;
}
