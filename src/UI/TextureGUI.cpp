//
// Created by Zhongshi Jiang on 5/13/17.
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
#include <sstream>
#include <igl/cat.h>
#include <iomanip>

#include <igl/Timer.h>
#include <igl/flip_avoiding_line_search.h>
#include <igl/cat.h>
#include "../Newton/SchaeferNewtonParametrizer.h"



TextureGUI::TextureGUI(StateManager &state):
s_(state), d_(state.scaf_data) {
  using namespace Eigen;
  using igl_viewer = igl::viewer::Viewer;

  igl::png::readPNG("/Users/zhongshi/Desktop/texture_bb.png",texture_R,
                    texture_G, texture_B, texture_A);

  v_.callback_pre_draw = [this](igl_viewer&) {
    return this->pre_draw();
  };
  v_.callback_post_draw = [this](igl_viewer&) {
    return this->post_draw();
  };
  v_.callback_init = [this](igl_viewer&) {
    return extended_menu();
  };
  v_.callback_key_pressed = [this](igl_viewer&, auto a, auto b) {
    return key_pressed(a);
  };
  v_.callback_mouse_down = [this](igl_viewer&, auto a, auto b) {
    return mouse_down(a);
  };
  v_.callback_mouse_up = [this](igl_viewer&, auto a, auto b) {
    return mouse_up(a);
  };

}

bool TextureGUI::launch() {
  v_.launch();
  continue_computing_ = false;
  re_draw_ = false;
  for(auto & t : threads_) if(t.joinable()) t.join();
  return true;
}

bool TextureGUI::render_to_png(const int width, const int height,
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

typedef struct {
  double r;       // a fraction between 0 and 1
  double g;       // a fraction between 0 and 1
  double b;       // a fraction between 0 and 1
} rgb;

typedef struct {
  double h;       // angle in degrees
  double s;       // a fraction between 0 and 1
  double v;       // a fraction between 0 and 1
} hsv;

static hsv   rgb2hsv(rgb in);
static rgb   hsv2rgb(hsv in);

hsv rgb2hsv(rgb in)
{
  hsv         out;
  double      min, max, delta;

  min = in.r < in.g ? in.r : in.g;
  min = min  < in.b ? min  : in.b;

  max = in.r > in.g ? in.r : in.g;
  max = max  > in.b ? max  : in.b;

  out.v = max;                                // v
  delta = max - min;
  if (delta < 0.00001)
  {
    out.s = 0;
    out.h = 0; // undefined, maybe nan?
    return out;
  }
  if( max > 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
    out.s = (delta / max);                  // s
  } else {
    // if max is 0, then r = g = b = 0
    // s = 0, v is undefined
    out.s = 0.0;
    out.h = NAN;                            // its now undefined
    return out;
  }
  if( in.r >= max )                           // > is bogus, just keeps compilor happy
    out.h = ( in.g - in.b ) / delta;        // between yellow & magenta
  else
  if( in.g >= max )
    out.h = 2.0 + ( in.b - in.r ) / delta;  // between cyan & yellow
  else
    out.h = 4.0 + ( in.r - in.g ) / delta;  // between magenta & cyan

  out.h *= 60.0;                              // degrees

  if( out.h < 0.0 )
    out.h += 360.0;

  return out;
}


rgb hsv2rgb(hsv in)
{
  double      hh, p, q, t, ff;
  long        i;
  rgb         out;

  if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
    out.r = in.v;
    out.g = in.v;
    out.b = in.v;
    return out;
  }
  hh = in.h;
  if(hh >= 360.0) hh = 0.0;
  hh /= 60.0;
  i = (long)hh;
  ff = hh - i;
  p = in.v * (1.0 - in.s);
  q = in.v * (1.0 - (in.s * ff));
  t = in.v * (1.0 - (in.s * (1.0 - ff)));

  switch(i) {
    case 0:
      out.r = in.v;
      out.g = t;
      out.b = p;
      break;
    case 1:
      out.r = q;
      out.g = in.v;
      out.b = p;
      break;
    case 2:
      out.r = p;
      out.g = in.v;
      out.b = t;
      break;

    case 3:
      out.r = p;
      out.g = q;
      out.b = in.v;
      break;
    case 4:
      out.r = t;
      out.g = p;
      out.b = in.v;
      break;
    case 5:
    default:
      out.r = in.v;
      out.g = p;
      out.b = q;
      break;
  }
  return out;
}
void TextureGUI::scaffold_coloring() {
  mesh_color_ = Eigen::MatrixXd::Constant(d_.f_num, 3, 0.86);
  double saturation = .2;
  double bright_val = .8;
  auto& face_sizes = d_.component_sizes;
//  Eigen::VectorXi face_sizes(12);
//  face_sizes<< 90, 73,801, 73,222,478,224,272, 40, 198, 167,185;
  for(int i=0, cur_color = 0; i<12; i++) {
    int cur_size = face_sizes[i];
    rgb ci = hsv2rgb({i*360/12.,saturation,bright_val});
    mesh_color_.middleRows(cur_color, cur_size) = Eigen::RowVector3d(ci.r,ci.g,
                                                                  ci.b)
        .replicate(cur_size,1);
    cur_color += cur_size;
  }

//  auto acc_size = 0;
//  for(int i=0, j=0; i<d_.component_sizes.size(); i++) {
//    acc_size += d_.component_sizes[i];
//    if(i != picked_component) {
//      for (; j < acc_size; j++) {
//        mesh_color_.row(j) << 182. / 255., 215. / 255, 168 / 255.;
//      }
//    } else {
//      for (; j < acc_size; j++) {
//        mesh_color_.row(j) << 254. / 255., 216. / 255, 111 / 255.;
//      }
//    }
//  }

  // render overlay with d_.soft_cons
  for (auto const &x: d_.soft_cons) {
    Eigen::RowVector3d vert0(0, 0, 0);
    vert0.head(d_.dim) = d_.w_uv.row(x.first);
    v_.data.add_points(vert0, Eigen::RowVector3d(1, 0, 0));
    v_.data.add_points(x.second, Eigen::RowVector3d(0, 0, 1));
    v_.data.dirty |= v_.data.DIRTY_OVERLAY_POINTS;
//    v_.data.dirty |= v_.data.DIRTY_OVERLAY_LINES;
  }

  v_.data.set_colors(mesh_color_);
}

bool TextureGUI::post_draw() {
  if(re_png_) {

    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << s_.iter_count;
    std::string s = ss.str();
    render_to_png(2*1600, 2*900, "result" + s +".png");
    std::cout<<"Written:"<<s<<std::endl;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      re_png_ = false;
    }
  }
  return false;
}

bool TextureGUI::pre_draw() {
  if (re_draw_) {
    if (!viewer_cores_init) {
      viewer_core_3d_ = v_.core;
      viewer_core_2d_ = v_.core;
      viewer_cores_init = true;

      viewer_core_3d_.show_texture = true;
      viewer_core_3d_.show_lines = false;
      viewer_core_3d_.lighting_factor = 1.f;
      viewer_core_3d_.show_overlay_depth = true;
      viewer_core_3d_.is_animating = true;
      viewer_core_3d_.align_camera_center(d_.m_V, d_.m_T);

      viewer_core_2d_.show_texture = false;
      viewer_core_2d_.show_lines = true;
      viewer_core_2d_.show_overlay_depth = false;
      viewer_core_2d_.is_animating = true;
      viewer_core_2d_.lighting_factor = 0;
      viewer_core_2d_.align_camera_center(d_.w_uv, d_.surface_F);

      v_.core = uv_space ? viewer_core_2d_ : viewer_core_3d_;
    }

    v_.data.clear();
    {
      std::lock_guard<std::mutex> lock(mutex_);

      if (uv_space) {
        v_.data.set_mesh(d_.w_uv, d_.surface_F);
        scaffold_coloring();
      } else {
        v_.data.set_mesh(d_.m_V, d_.m_T);
        v_.data.set_colors(Eigen::RowVector3d::Constant(1));
        v_.data.set_uv(uv_scale * d_.w_uv.topRows(d_.mv_num));
        v_.data.texture_R = texture_R;
        v_.data.texture_G = texture_G;
        v_.data.texture_B = texture_B;
        v_.data.texture_A = texture_A;
      }
      show_uv_seam(uv_space);

      re_draw_ = false;
    }
    v_.ngui->refresh();
  }
  return false;
}

bool TextureGUI::key_pressed(unsigned int key) {
  switch (key) {
    case ' ':
      continue_computing_ = !continue_computing_;
      if(continue_computing_){
        threads_.emplace_back(&TextureGUI::background_computation, this);
      } else {
        for(auto&t:threads_)
          if(t.joinable())
            t.join();
        threads_.clear();
      }
      std::cout<< (continue_computing_ ?
      std::string("Continue...") : std::string("Pause."))
          <<std::endl;
      return true;
    case '=':
    case '+':
      if (continue_computing_) {
        continue_computing_ = false;
        for (auto &t:threads_)
          if (t.joinable())
            t.join();
        threads_.clear();
      }
      {double sc = reference_scaling_ + 1;
      double change_factor =  sc/ reference_scaling_;
      s_.ws_solver->enlarge_internal_reference(change_factor);
      reference_scaling_= sc;
      v_.ngui->refresh();}
      return true;
    case '-':
    case '_':
      if (continue_computing_) {
        continue_computing_ = false;
        for (auto &t:threads_)
          if (t.joinable())
            t.join();
        threads_.clear();
      }
      {double sc = reference_scaling_ - 1;
      double change_factor =  sc/ reference_scaling_;
      s_.ws_solver->enlarge_internal_reference(change_factor);
      reference_scaling_= sc;
      v_.ngui->refresh();}
      return true;
    case 'd':
    case 'D': // drag
      mouse_click_mode = ClickMode::SOFT_CONSTRAINT;
      return true;
    case 'm':
    case 'M': // move
      continue_computing_ = false;
      for(auto&t:threads_)
        if(t.joinable())
          t.join();
      threads_.clear();
      mouse_click_mode = ClickMode::CHOOSE_PATCH;
      v_.ngui->refresh();
      return true;
    case 'c':
    case 'C':
      continue_computing_ = false;
      for(auto&t:threads_)
        if(t.joinable())
          t.join();
      threads_.clear();
      d_.soft_cons.clear();
      picked_component = -1;
      mouse_click_mode = ClickMode::NONE;
      scaffold_coloring();
      continue_computing_ = true;
      threads_.emplace_back(&TextureGUI::background_computation, this);
      v_.ngui->refresh();
      return true;
    default: break;
  }
  return false;
}

bool TextureGUI::computation_for_png_dumping() {
  using namespace std;
  auto &ws = s_.ws_solver;
  double last_mesh_energy =  ws->compute_energy(d_.w_uv, false) / d_.mesh_measure;
  d_.energy = ws->compute_energy(d_.w_uv, true) / d_.mesh_measure;
  if(s_.iter_count == 0) {
    // print the initial
    while(re_png_)
    {
      std::this_thread::sleep_for(100ms);
    }
    re_png_ = true;
  }
  while (continue_computing_) {


    std::cout << "=============" << std::endl;
    std::cout << "Iteration:" << s_.iter_count+1 << '\t';
    if(s_.iter_count>50)//iter_countz
      return true;
    igl::Timer timer;

    timer.start();

    while(re_png_)
    {
      std::this_thread::sleep_for(100ms);
    }
    s_.iter_count ++; // separate here for threading.

    {
      std::lock_guard<std::mutex> lock(mutex_);
      bool auto_expand = false;
      if(auto_expand)
        d_.rect_frame_V.resize(0,0);
      ws->mesh_improve();
    }


    auto iteration_out = d_.w_uv;
  if(true) {
    double new_weight = d_.mesh_measure * (last_mesh_energy-4)
        / (d_.sf_num*100);
    ws->adjust_scaf_weight(new_weight);
    d_.energy = ws->perform_iteration(iteration_out);
  } else {
    SchaeferNewtonParametrizer snp(d_);

    Eigen::MatrixXd Vo =d_.w_uv;
    Eigen::MatrixXi w_T;
    igl::cat(1,d_.m_T,d_.s_T,w_T);
    snp.newton_iteration(w_T, Vo);

    auto whole_E =
        [&](Eigen::MatrixXd &uv) { return snp.evaluate_energy(w_T, uv);};

//        sd.w_uv = Vo;
    d_.energy = igl::flip_avoiding_line_search(w_T, iteration_out, Vo,
                                               whole_E, -1)
        / d_.mesh_measure;
  }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      d_.w_uv = iteration_out;
      re_draw_ = true;
      re_png_ = true;
    }

    double current_mesh_energy =
        ws->compute_energy(d_.w_uv, false) / d_.mesh_measure;
    double mesh_energy_decrease = last_mesh_energy - current_mesh_energy;
    cout << "Energy After:"
         << d_.energy - 4
         << "\tMesh Energy:"
         << current_mesh_energy - 4
         << "\tEnergy Decrease"
         << mesh_energy_decrease
         << endl;
    last_mesh_energy = current_mesh_energy;


  }
  return false;
}

bool TextureGUI::background_computation() {
  using namespace std;
  auto &ws = s_.ws_solver;
  double last_mesh_energy =  ws->compute_energy(d_.w_uv, false) / d_.mesh_measure;
  std::cout<<"last" <<last_mesh_energy<<std::endl;
  d_.energy = ws->compute_energy(d_.w_uv, true) / d_.mesh_measure;
  while (continue_computing_) {

    std::cout << "=============" << std::endl;
    std::cout << "Iteration:" << s_.iter_count+1 << '\t';
    igl::Timer timer;

    timer.start();


    s_.iter_count ++; // separate here for threading.

    if(s_.optimize_scaffold){
      std::lock_guard<std::mutex> lock(mutex_);
      bool auto_expand = false;
      if(auto_expand)
        d_.rect_frame_V.resize(0,0);
      ws->mesh_improve();
    } else {
      ws->after_mesh_improve();
    }


    if (auto_weighting_) {
      double new_weight = d_.mesh_measure * (last_mesh_energy - 4)
          / (100 * d_.sf_num);
      ws->adjust_scaf_weight(new_weight);
      std::cout<<"ScafWeight:"<<d_.scaffold_factor<<std::endl;
    }

    auto iteration_out = d_.w_uv;
    d_.energy = ws->perform_iteration(iteration_out);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      d_.w_uv = iteration_out;
      re_draw_ = true;
    }

    double current_mesh_energy =
        ws->compute_energy(d_.w_uv, false) / d_.mesh_measure;
    double mesh_energy_decrease = last_mesh_energy - current_mesh_energy;
    cout << "Energy After:"
         << d_.energy - 4
         << "\tMesh Energy:"
         << current_mesh_energy - 4
         << "\tEnergy Decrease"
         << mesh_energy_decrease
         << endl;
    last_mesh_energy = current_mesh_energy;
  }
  return false;
}

void TextureGUI::show_uv_seam(bool uv_space) {
  if(uv_space) return;
  using namespace Eigen;
  using namespace std;
  const auto& V_to_slice = uv_space?d_.w_uv:d_.m_V;

  int acc_bnd = 0;
  for(int i=0; i<d_.bnd_sizes.size(); i++) {
    Eigen::VectorXi this_piece_of_bnd =
        d_.internal_bnd.segment(acc_bnd, d_.bnd_sizes[i]);
    acc_bnd += d_.bnd_sizes[i];
    MatrixXd E0, E1;
    igl::slice(V_to_slice, this_piece_of_bnd, 1, E0);
    int n_e = E0.rows();
    E1.resizeLike(E0);
    E1.bottomRows(n_e - 1) = E0.topRows(n_e - 1);
    E1.row(0) = E0.row(n_e - 1);
    v_.data.add_edges(E0, E1, Eigen::RowVector3d(1, 0, 0));
  }
  v_.data.dirty |= igl::viewer::ViewerData::DIRTY_OVERLAY_LINES;
};
