#include "ReWeightedARAP.h"
#include "StateManager.h"
#include "util/triangle_utils.h"
#include "util/tetgenio_parser.h"
#include "UI/DeformGUI.h"
#include "UI/TextureGUI.h"

#include <nanogui/formhelper.h>
#include <nanogui/screen.h>

#include <igl/viewer/Viewer.h>
#include <igl/triangle/triangulate.h>
#include <igl/boundary_loop.h>
#include <igl/cat.h>
#include <igl/slice_into.h>
#include <igl/local_basis.h>
#include <igl/read_triangle_mesh.h>
#include <igl/polar_svd.h>
#include <igl/write_triangle_mesh.h>
#include <igl/flip_avoiding_line_search.h>
#include <igl/adjacency_matrix.h>
#include <igl/png/writePNG.h>

#include <igl/slim.h>
#include <igl/slice.h>
#include <igl/colon.h>
#include <igl/Timer.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/flipped_triangles.h>
#include <igl/components.h>
#include <igl/euler_characteristic.h>
#include <igl/is_edge_manifold.h>
#include <igl/doublearea.h>
#include <igl/squared_edge_lengths.h>
#include <igl/PI.h>

#include <memory>
#include <algorithm>
#include <iostream>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/barycenter.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/readMESH.h>
#include <igl/writeMESH.h>
#include <igl/writeOBJ.h>
#define MODEL_PATH "../models/bumpy.off"
#define CUBE_PATH "../models/cube_cc1.obj"

int main(int argc, char* argv[]) {
  using namespace Eigen;
  using namespace std;
  std::string filename = argv[1];
  StateManager s_(filename);
#ifndef NOGUI
  bool complex_ui = (argc >= 3);
  if(complex_ui) {
    TextureGUI gui(s_);
    gui.launch();
  } else {
    igl::viewer::Viewer v;
    DeformGUI gui(v, s_);
    v.launch();
  }
#else
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
