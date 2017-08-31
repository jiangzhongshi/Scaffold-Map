//
// Created by Zhongshi Jiang on 1/25/17.
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
#include <igl/writeMESH.h>
#include <igl/boundary_facets.h>
#include <igl/writeOBJ.h>
#include <stb_image_write.h>
#include <igl/png/readPNG.h>
#include <igl/slice.h>

bool DeformGUI::mouse_down(int button, int mod) {
  if (button != 0) // only overwrite left button
    return false;

  using namespace Eigen;
  // pick vertex.
  int fid = 0;
  Eigen::Vector3f bc;
  // Cast a ray in the view direction starting from the mouse position
  float x = v_.current_mouse_x;
  float y = v_.core.viewport(3) - v_.current_mouse_y;

  // only picking the interior.
  Eigen::MatrixXd m_uv3 = Eigen::MatrixXd::Zero(d_.mv_num, 3);
  auto& picked_face = d_.dim == 3?d_.surface_F:d_.m_T;
  m_uv3.leftCols(d_.w_uv.cols()) = d_.w_uv.topRows(d_.mv_num);
  if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y),
                               v_.core.view * v_.core.model,
                               v_.core.proj, v_.core.viewport,
                               m_uv3, picked_face,
                               fid, bc)) {
    v_.data.set_colors(mesh_color);

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
    return true;
  }

  return false;
}


bool DeformGUI::render_to_png(const int width, const int height,
                               const std::string png_file) {
  const int comp = 4;                                             // 4 Channels Red, Green, Blue, Alpha
  const int stride = width*comp;  // Length of one row in bytes
  unsigned char * data_fv = new unsigned char[comp * width * height];

  // Hack -- override viewport and redraw
  Eigen::Vector4f viewport_ori = v_.core.viewport;
  v_.core.viewport << 0, 0, width, height;

  glClearColor(0.0f,0.0f,0.0f,0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  v_.core.draw(v_.data,v_.opengl,false);

  // get the data
  glReadPixels(0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE,data_fv);

  v_.core.viewport = viewport_ori;

  // flip vertically
  std::vector<unsigned char> pixels(comp * width * height,0);     // The image itself;
  for (unsigned j = 0; j<height;++j){
    int j_fl = (height-1-j);
    for (unsigned i = 0; i< stride; ++i){
      pixels[j*stride + i] = data_fv[j_fl * stride + i];
    }
  }
  bool ret = stbi_write_png(png_file.c_str(), width, height, comp, pixels.data(), stride*sizeof(unsigned char));
  delete [] data_fv;
  return ret;
}
bool DeformGUI::mouse_up(int button, int mod) {
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
  return true;
}

DeformGUI::DeformGUI(igl::viewer::Viewer &vi,
                     StateManager &state) :
    v_(vi),
    s_(state),
    d_(state.scaf_data) {

  using namespace Eigen;
  v_.callback_mouse_down = [this](igl::viewer::Viewer &, int a, int b) {
    return this->mouse_down(a, b);
  };
  v_.callback_mouse_up = [this](igl::viewer::Viewer &, int a, int b) {
    return this->mouse_up(a, b);
  };

  v_.callback_key_pressed = [this](igl::viewer::Viewer &,
                                         unsigned int key,
                                         int mod) {
    return this->key_press(key, mod);
  };

  v_.callback_init = [this](igl::viewer::Viewer &) {
    return extended_menu();
  };

  v_.data.set_mesh(d_.w_uv, d_.surface_F);
  scaffold_coloring();
  v_.data.set_face_based(true);
  v_.core.show_overlay_depth = false;

  VectorXi b;
  MatrixXd bc, uv_init = d_.w_uv.topRows(d_.mv_num);
}

void DeformGUI::scaffold_coloring() {
  if(d_.dim == 2) {
    assert(v_.data.F.rows() == d_.f_num &&
        "Scaf Color Incompatible With Mesh");

    mesh_color.resize(d_.f_num, 3);
    for (int i = 0; i < d_.mf_num; i++)
      mesh_color.row(i) << 182. / 255., 215. / 255, 168 / 255.;

    for (int i = d_.mf_num; i < d_.f_num; i++)
      mesh_color.row(i) << 0.86, 0.86, 0.86;
    v_.data.set_colors(mesh_color);
    v_.core.lighting_factor = 0;
  }

  // render overlay with d_.soft_cons
  for (auto const &x: d_.soft_cons) {
    Eigen::RowVector3d vert0(0, 0, 0);
    vert0.head(d_.dim) = d_.w_uv.row(x.first);
    v_.data.add_points(vert0, Eigen::RowVector3d(1, 0, 0));
    v_.data.add_points(x.second, Eigen::RowVector3d(0, 0, 1));
    v_.data.dirty |= v_.data.DIRTY_OVERLAY_POINTS;
//    v_.data.dirty |= v_.data.DIRTY_OVERLAY_LINES;
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
    v_.data.add_edges(E0, E1, Eigen::RowVector3d(1, 0, 0));
  }
  v_.data.dirty |= igl::viewer::ViewerData::DIRTY_OVERLAY_LINES;
}
}

bool DeformGUI::extended_menu() {
  using namespace Eigen;
  using namespace std;

  auto& ws_solver = s_.ws_solver;
  bool &reg_scaf = s_.optimize_scaffold;
  int &iteration_count = s_.iter_count;

  v_.ngui->addGroup("Scaffold Info");

  v_.ngui->addVariable("It", iteration_count, false);

  v_.ngui->addGroup("Serialization");
  v_.ngui->addButton("Save", [&]() {
    s_.save(igl::file_dialog_save());
  });
  v_.ngui->addButton("Load", [&]() {
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

    ws_solver = s_.ws_solver;

    // refresh after load
    v_.data.clear();
    v_.data.set_mesh(d_.w_uv, d_.surface_F);
    scaffold_coloring();

    v_.ngui->refresh();
  });

  v_.ngui->addButton("SaveMesh", [&]() {
    if(d_.dim == 3) {
      Eigen::MatrixXi F;
      igl::boundary_facets(d_.s_T, F);
      igl::writeMESH(igl::file_dialog_save(), d_.w_uv, d_.s_T, F);
    } else {
      Eigen::MatrixXd w_uv3 = Eigen::MatrixXd::Zero(d_.w_uv.rows(),3);
      w_uv3.leftCols(2) = d_.w_uv;
      igl::writeOBJ(igl::file_dialog_save(),w_uv3, d_.surface_F);
    }
  });

  // Add an additional menu window
  v_.ngui->addWindow(Eigen::Vector2i(1280 - 220, 10), "Scaffold Tweak");
  v_.ngui->addVariable<bool>("ReMesh", [&reg_scaf](bool val) {
    reg_scaf = val; // set
  }, [&reg_scaf]() {
    return reg_scaf; // get
  });
  v_.ngui->addVariable("AutoWeight", auto_weight);
  v_.ngui->addVariable<float>("Scaf Weight",
                                    [&ws_solver](float val) {
                                      ws_solver->adjust_scaf_weight(val);
                                      std::cout << "Weight:" << val
                                                << std::endl;
                                    },
                                    [this]() {
                                      return d_.scaffold_factor;
                                    });

  // Expose the same variable directly ...
  // m_viewer.ngui->addVariable("float",floatVariable);

//    if(m_state.solver_type == StateManager::SolverType::LBFGS)
//      m_viewer.ngui->addVariable("BFGS per it",inner_iters );
//    else if(m_state.solver_type == StateManager::SolverType::ReweightedARAP)
  v_.ngui->addVariable("It Per [ ]", inner_iters);
  v_.ngui->addVariable("Use Newton", use_newton);

  v_.ngui->addButton("Clear Constraints", [&]() {
    d_.soft_cons.clear();
    scaffold_coloring();

    v_.ngui->refresh();
  });


  v_.ngui->addButton("Snapshot", [&]() {
    render_to_png(2*1600, 2*900,
                  igl::file_dialog_save
                      ());
  });

  // Generate menu
  v_.screen->performLayout();

  return false;
};

