//
// Created by Zhongshi Jiang on 1/25/17.
//

#include "DeformGUI.h"
#include "../ReWeightedARAP.h"
#include "../StateManager.h"
#include "../util/triangle_utils.h"
#include <igl/slim.h>
#include <igl/file_dialog_save.h>
#include <igl/file_dialog_open.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/unproject.h>
#include <igl/project.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeMESH.h>
#include <igl/boundary_facets.h>
#include <igl/writeOBJ.h>
#include <igl/png/readPNG.h>
#include <igl/slice.h>

#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>


bool DeformGUI::mouse_down(int button, int mod) {
  if (button != 0) // only overwrite left button
    return false;

  using namespace Eigen;
  // pick vertex.
  int fid = 0;
  Eigen::Vector3f bc;
  // Cast a ray in the view direction starting from the mouse position
  float x = v_.current_mouse_x;
  float y = v_.core().viewport(3) - v_.current_mouse_y;

  // only picking the interior.
  Eigen::MatrixXd m_uv3 = Eigen::MatrixXd::Zero(d_.mv_num, 3);
  auto& picked_face = d_.dim == 3?d_.surface_F:d_.m_T;
  m_uv3.leftCols(d_.w_uv.cols()) = d_.w_uv.topRows(d_.mv_num);
  if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y),
                               v_.core().view,
                               v_.core().proj, v_.core().viewport,
                               m_uv3, picked_face,
                               fid, bc)) {
    v_.data().set_colors(mesh_color);

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
            (v_.core().view).eval(),
            v_.core().proj,
            v_.core().viewport);
    down_z_ = coord[2];

    dragging_ = true;
    return true;
  }

  return false;
}


bool DeformGUI::render_to_png(const int width, const int height,
                               const std::string png_file) {

  Eigen::Vector4f viewport_ori = v_.core().viewport;
  v_.core().viewport << 0, 0, width, height;
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(width,height);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(width,height);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(width,height);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(width,height);

  // Draw the scene in the buffers
  v_.core().draw_buffer(
    v_.data(),false,R,G,B,A);

  // Save it to a PNG
  // igl::png::writePNG(R,G,B,A,png_file);
  v_.core().viewport = viewport_ori;

  return true;
}
bool DeformGUI::mouse_up(int button, int mod) {
  if (dragging_ == false || button != 0)
    return false;

  int m_x = v_.current_mouse_x;
  int m_y = v_.current_mouse_y;

  Eigen::RowVector3f pos1 = igl::unproject(
      Eigen::Vector3f(m_x, v_.core().viewport[3] - m_y, down_z_),
      v_.core().view,
      v_.core().proj, v_.core().viewport);
  Eigen::RowVector3f pos0 = igl::unproject(
      Eigen::Vector3f(v_.down_mouse_x,
                      v_.core().viewport[3] - v_.down_mouse_y,
                      down_z_),
      v_.core().view,
      v_.core().proj, v_.core().viewport);

  Eigen::RowVector3d diff = Eigen::RowVector3d((pos1 - pos0).cast<double>());

  int dim = d_.dim;
  Eigen::RowVector3d vert0(0,0,0);
  vert0.head(dim) = d_.w_uv.row(picked_vert_);
  Eigen::RowVector3d vert1 = vert0 + diff;

  v_.data().add_points(vert0, Eigen::RowVector3d(1, 0, 0));
  v_.data().add_points(vert1, Eigen::RowVector3d(0, 0, 1));
  v_.data()
          .add_edges(vert0,
                     vert1,
                     Eigen::RowVector3d(0, 0, 1));

  v_.data().dirty |= igl::opengl::MeshGL::DIRTY_OVERLAY_LINES;
  d_.add_soft_constraints(picked_vert_, vert1.head(dim));

  dragging_ = false;
  return true;
}

void DeformGUI::show_box()
{
       Eigen::MatrixXi box_F, plot_F;
       igl::boundary_facets(d_.s_T, box_F);
 //        igl::cat(1, d_.surface_F, box_F,  plot_F);
       v_.data().set_mesh(d_.w_uv, d_.surface_F);
       {
         for (int i = 0; i < box_F.rows(); i++) {
           Eigen::MatrixXi edges(3, 2);
           edges << 0, 1, 1, 2, 2, 0;
           for (int e = 0; e < 3; e++) {
             v_.data().add_edges
                 (
                     d_.w_uv.row(box_F(i, edges(e, 0))),
                     d_.w_uv.row(box_F(i, edges(e, 1))),
                     Eigen::RowVector3d(0, 0, 0)
                 );
           }
         }
       }
       {
         // Find the bounding box
         Eigen::Vector3d m = d_.w_uv.colwise().minCoeff();
         Eigen::Vector3d M = d_.w_uv.colwise().maxCoeff();

         // Corners of the bounding box
         Eigen::MatrixXd V_box(8, 3);
         V_box <<
               m(0), m(1), m(2),
             M(0), m(1), m(2),
             M(0), M(1), m(2),
             m(0), M(1), m(2),
             m(0), m(1), M(2),
             M(0), m(1), M(2),
             M(0), M(1), M(2),
             m(0), M(1), M(2);

         // Edges of the bounding box
         Eigen::MatrixXi E_box(12, 2);
         E_box <<
               0, 1,
             1, 2,
             2, 3,
             3, 0,
             4, 5,
             5, 6,
             6, 7,
             7, 4,
             0, 4,
             1, 5,
             2, 6,
             7, 3;

         // Plot the corners of the bounding box as points

         // Plot the edges of the bounding box
         for (unsigned i = 0; i < E_box.rows(); ++i)
           v_.data().add_edges
               (
                   V_box.row(E_box(i, 0)),
                   V_box.row(E_box(i, 1)),
                   Eigen::RowVector3d(0, 0, 0)
               );

         v_.data().dirty |= igl::opengl::MeshGL::DIRTY_ALL;

       }
       v_.data().set_face_based(true);
}


DeformGUI::DeformGUI(igl::opengl::glfw::Viewer &vi,
                     StateManager &state) :
    v_(vi),
    s_(state),
    d_(state.scaf_data){
//    menu_ = std::make_shared<igl::opengl::glfw::imgui::ImGuiMenu>();

  using namespace Eigen;
  v_.callback_mouse_down = [this](igl::opengl::glfw::Viewer &, int a, int b) {
    return this->mouse_down(a, b);
  };
  v_.callback_mouse_up = [this](igl::opengl::glfw::Viewer &, int a, int b) {
    return this->mouse_up(a, b);
  };

  v_.callback_key_pressed = [this](igl::opengl::glfw::Viewer &,
                                         unsigned int key,
                                         int mod) {
    return this->key_press(key, mod);
  };

//  v_.callback_init = [this](igl::opengl::glfw::Viewer &) {
//    return extended_menu();
//  };
  // extended_menu();
  v_.data().set_mesh(d_.w_uv, d_.surface_F);

    scaffold_coloring();
  v_.data().set_face_based(true);
  v_.data().show_overlay_depth = false;

  VectorXi b;
  MatrixXd bc, uv_init = d_.w_uv.topRows(d_.mv_num);
}

void DeformGUI::scaffold_coloring() {
  if(d_.dim == 2) {
    assert(v_.data().F.rows() == d_.f_num &&
        "Scaf Color Incompatible With Mesh");

    mesh_color.resize(d_.f_num, 3);
    for (int i = 0; i < d_.mf_num; i++)
      mesh_color.row(i) << 182. / 255., 215. / 255, 168 / 255.;

    for (int i = d_.mf_num; i < d_.f_num; i++)
      mesh_color.row(i) << 0.86, 0.86, 0.86;
    v_.data().set_colors(mesh_color);
    v_.core().lighting_factor = 0;
  }

  // render overlay with d_.soft_cons
  for (auto const &x: d_.soft_cons) {
    Eigen::RowVector3d vert0(0, 0, 0);
    vert0.head(d_.dim) = d_.w_uv.row(x.first);
    v_.data().add_points(vert0, Eigen::RowVector3d(1, 0, 0));
    v_.data().add_points(x.second, Eigen::RowVector3d(0, 0, 1));
    v_.data().dirty |= igl::opengl::MeshGL::DIRTY_OVERLAY_POINTS;
//    v_.data().dirty |= v_.data().DIRTY_OVERLAY_LINES;
  }
if(show_interior_boundary) {
  int acc_bnd = 0;
  for (int i = 0; i < d_.bnd_sizes.size(); i++) {
    Eigen::VectorXi this_piece_of_bnd =
        d_.internal_bnd.segment(acc_bnd, d_.bnd_sizes[i]);
    acc_bnd += d_.bnd_sizes[i];
    Eigen::MatrixXd E0, E1;
    igl::slice(d_.w_uv, this_piece_of_bnd, 1, E0);
    int n_e = E0.rows();
    E1.resizeLike(E0);
    E1.bottomRows(n_e - 1) = E0.topRows(n_e - 1);
    E1.row(0) = E0.row(n_e - 1);
    v_.data().add_edges(E0, E1, Eigen::RowVector3d(1, 0, 0));
  }
  v_.data().dirty |= igl::opengl::MeshGL::DIRTY_OVERLAY_LINES;
}
}

bool DeformGUI::extended_menu()
{
	using namespace Eigen;
	using namespace std;

  static igl::opengl::glfw::imgui::ImGuiMenu menu_;
	v_.plugins.push_back(&menu_);

	menu_.callback_draw_viewer_menu = [&]() {
	    menu_.draw_viewer_menu();
//     Add widgets to the sidebar.
    if (ImGui::CollapsingHeader("Scaffold Info", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Text("iteration count %d", s_.iter_count);
        ImGui::Text("V %ld F %ld", d_.mv_num, d_.mf_num);
        ImGui::Text("Scaf V %ld F %ld T %ld", d_.sv_num, d_.sf_num, d_.s_T.rows());
    }
    if (ImGui::CollapsingHeader("Serialization", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button("Save")) s_.save(igl::file_dialog_save());
        if (ImGui::Button("Load")) {
            std::string ser_file = igl::file_dialog_open();
            if(ser_file.empty()) return;
            s_.load(ser_file);

            VectorXi b(d_.soft_cons.size());
            MatrixXd bc(b.rows(), 3);
            int i = 0;
            for (auto const &x: d_.soft_cons) {
              b(i) = x.first;
              bc.row(i) = x.second;
              i++;
            }

            MatrixXd m_uv = d_.w_uv.topRows(d_.mv_num);

            // refresh after load
            v_.data().clear();
            v_.data().set_mesh(d_.w_uv, d_.surface_F);
            scaffold_coloring();
        }
        if (ImGui::Button("SaveMesh")) {
           if(d_.dim == 3) {
             Eigen::MatrixXi F;
             igl::boundary_facets(d_.s_T, F);
             igl::writeMESH(igl::file_dialog_save(), d_.w_uv, d_.s_T, F);
           } else {
             Eigen::MatrixXd w_uv3 = Eigen::MatrixXd::Zero(d_.w_uv.rows(),3);
             w_uv3.leftCols(2) = d_.w_uv;
             igl::writeOBJ(igl::file_dialog_save(),w_uv3, d_.surface_F);
           }
        }

		} // Serialization
	};

    menu_.callback_draw_custom_window = [&]()
    {
      // Define next window position + size
      ImGui::SetNextWindowPos(ImVec2(1000.f * menu_.menu_scaling(), 10), ImGuiCond_FirstUseEver);
      ImGui::SetNextWindowSize(ImVec2(200, 260), ImGuiCond_FirstUseEver);
      ImGui::Begin(
    "Scaffold Tweak", nullptr,
    ImGuiWindowFlags_NoSavedSettings
      );


    ImGui::Checkbox("Remesh", &s_.optimize_scaffold);
    ImGui::InputInt("It Per []", &s_.inner_iters);
    ImGui::Checkbox("Use Newton", &s_.use_newton);
    ImGui::Checkbox("Auto Weight", &s_.auto_weight);

    double val = d_.scaffold_factor;
    if (ImGui::InputDouble("Scaf Weight",&val)) {
        d_.set_scaffold_factor(val);
         std::cout << "Weight:" << val << std::endl;
    }
    if (ImGui::Button("Clear Constraints")) {
      d_.soft_cons.clear();
      scaffold_coloring();
    }
    if (ImGui::Button("SnapShot")) {
     render_to_png(2*1600, 2*900,
                   igl::file_dialog_save
                       ());
    }

    if(d_.dim==3 && ImGui::Button("ShowBox"))  show_box();

		ImGui::End();
	};
  return false;
}
