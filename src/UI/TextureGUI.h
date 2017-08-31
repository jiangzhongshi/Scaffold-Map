//
// Created by Zhongshi Jiang on 5/13/17.
//

#ifndef SCAFFOLD_TEST_TEXTUREGUI_H
#define SCAFFOLD_TEST_TEXTUREGUI_H

#include <thread>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <condition_variable>

#define ENABLE_SERIALIZATION
#include <igl/viewer/Viewer.h>
#include <igl/viewer/ViewerCore.h>

class StateManager;
class ScafData;

class TextureGUI {
 public:
  explicit TextureGUI(StateManager &);

  bool pre_draw();
  bool post_draw();

  bool mouse_down(int button);
  bool mouse_up(int button);
  bool key_pressed(unsigned int key);
  bool extended_menu();

  bool computation_for_png_dumping();
  bool background_computation();
  bool launch();

  bool render_to_png(const int width, const int height,
                     const std::string png_file);

  // members
  ScafData &d_;
  StateManager &s_;
  igl::viewer::Viewer v_;
  double reference_scaling_ = 1;

  // coloring and display.
  Eigen::MatrixXd mesh_color_;
  std::vector<Eigen::RowVector3d> componet_colors_;
  Eigen::Matrix<unsigned char, -1,-1> texture_R,texture_G, texture_B, texture_A;
  bool uv_space = true;
  igl::viewer::ViewerCore viewer_core_3d_;
  igl::viewer::ViewerCore viewer_core_2d_;
  bool viewer_cores_init = false;
  double uv_scale = 1.;
  void scaffold_coloring();
  void show_uv_seam(bool uv_space);

  //
  enum class ClickMode {
    CHOOSE_PATCH,
    PLACE_PATCH,
    SOFT_CONSTRAINT,
    NONE
  } mouse_click_mode = ClickMode::NONE;

 public: // soft constraints related
  int picked_vert_; //assume mesh stay constant throughout;
  bool dragging_ = false;
  float down_z_;
  Eigen::RowVectorXd offset_picking; // clicked - nearest vert

 public:


  int picked_component = -1;
//  bool choosing_patch = false;
//  bool placing_new_patch = false;
  Eigen::MatrixXd cache_new_patch_V;
  Eigen::MatrixXi cache_new_patch_F;

  // parameters
  bool auto_weighting_ = true;

  // control
  bool continue_computing_ = false;
  bool re_draw_ = true;
  bool re_png_ = false;
  std::mutex mutex_;
  std::vector<std::thread> threads_;
  std::condition_variable cv_;


  //
};
#endif //SCAFFOLD_TEST_TEXTUREGUI_H
