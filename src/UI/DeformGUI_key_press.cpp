//
// Created by Zhongshi Jiang on 3/22/17.
//
#include "DeformGUI.h"
#include "../ReWeightedARAP.h"
#include "../StateManager.h"
#include "../util/triangle_utils.h"
#include <nanogui/formhelper.h>
#include <igl/slim.h>
#include <igl/file_dialog_save.h>
#include <igl/file_dialog_open.h>
#include <igl/boundary_loop.h>
#include <igl/colon.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/unproject.h>
#include <igl/project.h>
#include <igl/viewer/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/doublearea.h>
#include <igl/triangle_triangle_adjacency.h>

#include "../Newton/SchaeferNewtonParametrizer.h"

#include <igl/Timer.h>
#include <igl/flip_avoiding_line_search.h>
#include <igl/cat.h>

namespace igl {
namespace slim {
double compute_energy(igl::SLIMData &s, Eigen::MatrixXd &V_new);
};
};

bool DeformGUI::key_press(unsigned int key, int mod) {
  using namespace Eigen;
  using namespace std;
  ScafData &sd = s_.scaf_data;
  auto ws_solver = s_.ws_solver;
  bool &optimize_scaf = s_.optimize_scaffold;
  bool adjust_frame = true;
  int &iteration_count = s_.iter_count;

  static MatrixXd interp, starting_triangulation = sd.w_uv, cslim_uv,
      leap_reference, V_out;
  static double last_mesh_energy = ws_solver->compute_energy(sd.w_uv, false)/
      sd.mesh_measure - 2*d_.dim;
  static double accumulated_time = 0;
  if (key == ' ') {   /// Iterates.

    for (int i = 0; i < inner_iters; i++) {
      std::cout << "=============" << std::endl;
      std::cout << "Iteration:" << iteration_count++ << '\t';
      igl::Timer timer;

      timer.start();
      if(optimize_scaf) {
        d_.rect_frame_V.resize(0,0);
        d_.mesh_improve();
        if(!use_newton) ws_solver->after_mesh_improve();
      } else {
        d_.automatic_expand_frame(2,3);
        ws_solver->after_mesh_improve();
      }
      if(auto_weight)
      ws_solver->adjust_scaf_weight(
          (last_mesh_energy)*sd.mesh_measure /(sd.sf_num) / 100.0);
      if(!use_newton) {

        sd.energy = ws_solver->perform_iteration(sd.w_uv);
      }
      else {
        // no precomputation.
        SchaeferNewtonParametrizer snp(sd);

        Eigen::MatrixXd Vo = sd.w_uv;
        MatrixXi w_T;
        igl::cat(1,sd.m_T,sd.s_T,w_T);
        snp.newton_iteration(w_T, Vo);

        auto whole_E =
            [&](Eigen::MatrixXd &uv) { return snp.evaluate_energy(w_T, uv);};

//        sd.w_uv = Vo;
        sd.energy = igl::flip_avoiding_line_search(w_T, sd.w_uv, Vo,
                                                      whole_E, -1)
            / d_.mesh_measure;
      }

      double current_mesh_energy =
          ws_solver->compute_energy(sd.w_uv, false) / sd.mesh_measure- 2*d_.dim;
      double mesh_energy_decrease = last_mesh_energy - current_mesh_energy;
      double iter_timing = timer.getElapsedTime();
      accumulated_time += iter_timing;
      cout << "Timing = " << iter_timing << "Total = "<<accumulated_time<< endl;
      cout << "Energy After:"
           << sd.energy - 2*d_.dim
           << "\tMesh Energy:"
           << current_mesh_energy
           << "\tEnergy Decrease"
           << mesh_energy_decrease
           << endl;
      cout << "V_num: "<<d_.v_num<<" F_num: "<<d_.f_num<<endl;
      last_mesh_energy = current_mesh_energy;

    }
  }

  switch (key) {
    case '0':
    case ' ':v_.data.clear();
      v_.data.set_mesh(sd.w_uv, sd.surface_F);
      v_.data.set_face_based(true);
      break;
    default:return false;
  }

  scaffold_coloring();
  if (v_.ngui)
    v_.ngui->refresh();
  return true;
}
