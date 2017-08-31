//
// Created by Zhongshi Jiang on 5/18/17.
//

#include "TextureGUI.h"
#include "../ReWeightedARAP.h"
#include "../StateManager.h"

#include <vector>
#include <nanogui/formhelper.h>
#include <nanogui/slider.h>

#include <igl/unproject_onto_mesh.h>
#include <igl/unproject.h>
#include <igl/project.h>
#include <igl/file_dialog_save.h>
#include <igl/file_dialog_open.h>
#include <igl/writeOBJ.h>
#include <igl/Timer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/png/writePNG.h>
#include <stb_image_write.h>
#include <igl/png/readPNG.h>
#include <igl/slice.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/slice.h>
#include <igl/remove_unreferenced.h>


bool TextureGUI::mouse_down(int button) {
  using namespace Eigen;
  if(!uv_space) return false; // only deal with uv_space
  if(button != 0) return false; // do nothing: not left botton
  switch(mouse_click_mode) {
    case ClickMode::PLACE_PATCH: {

      Eigen::Vector3f coord =
          igl::project(
              Eigen::Vector3f(d_.w_uv(0, 0), d_.w_uv(0, 1), 0),
              (v_.core.view * v_.core.model).eval(),
              v_.core.proj,
              v_.core.viewport);

      Eigen::RowVector3d pos0 = igl::unproject(
          Eigen::Vector3f(v_.down_mouse_x,
                          v_.core.viewport[3] - v_.down_mouse_y,
                          coord[2]),
          (v_.core.view * v_.core.model).eval(),
          v_.core.proj, v_.core.viewport).cast<double>();

      mouse_click_mode = ClickMode::NONE;
      Eigen::RowVector2d center = pos0.head(2);

      if(picked_component == -1) { // adding new patch
        auto &V = cache_new_patch_V;
        auto &F = cache_new_patch_F;
        {
//    std::lock_guard<std::mutex> lock(mutex_);
          continue_computing_ = false;
          for (auto &t:threads_)
            if (t.joinable())
              t.join();
          threads_.clear();

          d_.add_new_patch(V, F, center);

          s_.ws_solver->has_pre_calc = false;
          s_.ws_solver->pre_calc();
//        for(auto&r: component_scalings_) r = 1;

          re_draw_ = true;
        }
      } else { // moving old patch
        std::cout<<"Moving Old..."<<std::endl;

        //
        Eigen::MatrixXi F_comp, F_temp;
        auto acc_size = 0;
        for(int i=0; i<picked_component; i++) {
          acc_size += d_.component_sizes[i];
        }
        Eigen::VectorXi F_ids = Eigen::VectorXi::LinSpaced(
            d_ .component_sizes[picked_component], acc_size,
            acc_size + d_.component_sizes[picked_component] - 1);

        igl::slice(d_.m_T, F_ids, 1, F_comp);
        int v_min = F_comp.minCoeff();
        int v_max = F_comp.maxCoeff();
        Eigen::VectorXi dummyI;
        Eigen::MatrixXd V_temp;

        // assume this does not change ordering info.
        igl::remove_unreferenced(d_.w_uv, F_comp, V_temp, F_temp, dummyI);

        RowVector2d old_center = V_temp.colwise().mean();
        for(int i=0; i<V_temp.rows(); i++) V_temp.row(i) -= old_center;

        std::cout<<"New V size"<<V_temp.rows()
                 <<std::endl
            <<"v_min"<<v_min<<"v_max"<<v_max<<std::endl;
        for(int i=0 ; i<v_max - v_min + 1; i++) {
          d_.w_uv.row(v_min + i) = V_temp.row(i) + center;
        }

        d_.mesh_improve();
        picked_component = -1;

        re_draw_ = true;

      }
    }
      return true;

    case ClickMode::CHOOSE_PATCH: {
      int picked_face = -1;
      Eigen::MatrixXd m_uv3 = Eigen::MatrixXd::Zero(d_.mv_num, 3);
      m_uv3.leftCols(d_.w_uv.cols()) = d_.w_uv.topRows(d_.mv_num);

      Eigen::Vector3f bc;

      float x = v_.current_mouse_x;
      float y = v_.core.viewport(3) - v_.current_mouse_y;
      if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y),
                                   v_.core.view * v_.core.model,
                                   v_.core.proj, v_.core.viewport,
                                   m_uv3,d_.m_T,
                                   picked_face,
                                   bc)) {
        auto acc_size = 0;
        for(int i=0; i<d_.component_sizes.size(); i++) {
          acc_size += d_.component_sizes[i];
          if(picked_face < acc_size) {
            picked_component = i;
            break;
          }
        }
        assert(picked_component != -1);
        std::cout<<"Picked"<<picked_component<<std::endl;
        mouse_click_mode = ClickMode::PLACE_PATCH;
        re_draw_ = true;


//        while(component_scalings_.size() < d_.component_sizes.size()) {
//          std::cout<<"Pushing"<<std::endl;
//          component_scalings_.push_back(1);
//        }

      } // else do nothing. Maybe wrongly pressed.
    }
      return true;
    case ClickMode::SOFT_CONSTRAINT : {
      int fid = 0;
      Eigen::Vector3f bc;
      // Cast a ray in the view direction starting from the mouse position
      float x = v_.current_mouse_x;
      float y = v_.core.viewport(3) - v_.current_mouse_y;

      // only picking the interior.
      Eigen::MatrixXd m_uv3 = Eigen::MatrixXd::Zero(d_.mv_num, 3);
      auto& picked_face = d_.m_T;
      m_uv3.leftCols(d_.w_uv.cols()) = d_.w_uv.topRows(d_.mv_num);
      if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y),
                                   v_.core.view * v_.core.model,
                                   v_.core.proj, v_.core.viewport,
                                   m_uv3, picked_face,
                                   fid, bc)) {
        // find min
        int vid = 0;
        if (bc(1) > bc(0)) {
          vid = 1;
          if (bc(2) > bc(1)) vid = 2;
        } else if (bc(2) > bc(0)) vid = 2;

        bc(vid) -= 1;
        offset_picking = bc(0) * d_.w_uv.row(picked_face(fid, 0)) +
            bc(1) * d_.w_uv.row(picked_face(fid, 1)) +
            bc(2) * d_.w_uv.row(picked_face(fid, 2));

        picked_vert_ = picked_face(fid, vid);

        Eigen::Vector3f coord =
            igl::project(
                Vector3f(m_uv3.row(vid).cast<float>()),
                (v_.core.view * v_.core.model).eval(),
                v_.core.proj,
                v_.core.viewport);
        down_z_ = coord[2];

        dragging_ = true;
        re_draw_ = true;
    }
    }
    return true;
    default: return true; // overwrite left button. Disable rotation.
  }
}

bool TextureGUI::mouse_up(int button) {
  if (dragging_ == false || button != 0)
    return false;

  int m_x = v_.current_mouse_x;
  int m_y = v_.current_mouse_y;

  Eigen::RowVector3f pos1 = igl::unproject(
      Eigen::Vector3f(m_x, v_.core.viewport[3] - m_y, down_z_),
      (v_.core.view * v_.core.model).eval(),
      v_.core.proj, v_.core.viewport);
  Eigen::RowVector3f pos0 = igl::unproject(
      Eigen::Vector3f(v_.down_mouse_x,
                      v_.core.viewport[3] - v_.down_mouse_y,
                      down_z_),
      (v_.core.view * v_.core.model).eval(),
      v_.core.proj, v_.core.viewport);

  Eigen::RowVector3d diff = Eigen::RowVector3d((pos1 - pos0).cast<double>());
//  std::cout << "pos0:" << pos0 << std::endl
//            << "pos1:" << pos1 << std::endl;

  int dim = d_.dim;
  Eigen::RowVector3d vert0(0,0,0);
  vert0.head(dim) = d_.w_uv.row(picked_vert_);
  Eigen::RowVector3d vert1 = vert0 + diff;

  v_.data.add_points(vert0, Eigen::RowVector3d(1, 0, 0));
  v_.data.add_points(vert1, Eigen::RowVector3d(0, 0, 1));
  v_.data
    .add_edges(vert0,
               vert1,
               Eigen::RowVector3d(0, 0, 1));

  v_.data.dirty |= v_.data.DIRTY_OVERLAY_LINES;
  d_.add_soft_constraints(picked_vert_, vert1.head(dim));

  dragging_ = false;
  mouse_click_mode = ClickMode::NONE;
  re_draw_= true;
  return true;
}
