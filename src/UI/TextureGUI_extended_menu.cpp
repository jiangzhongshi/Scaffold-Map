//
// Created by Zhongshi Jiang on 5/17/17.
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

bool TextureGUI::extended_menu() {
  using namespace Eigen;
  using namespace std;

  auto& ws_solver = s_.ws_solver;
  bool &reg_scaf = s_.optimize_scaffold;
  int &iteration_count = s_.iter_count;

  v_.ngui->addGroup("Scaffold Info");

  v_.ngui->addVariable("It", iteration_count, false);

  v_.ngui->addGroup("Serialization");
  v_.ngui->addButton("Save", [&]() {
    std::string filename = igl::file_dialog_save();
    s_.save(filename);
    if(uv_space) viewer_core_2d_ = v_.core;
    else viewer_core_3d_ = v_.core;
    igl::serialize(viewer_core_3d_, "vcore3", filename);
    igl::serialize(viewer_core_2d_, "vcore2", filename);
  });
  v_.ngui->addButton("Load", [&]() {
    std::string ser_file = igl::file_dialog_open();
    if(ser_file.empty()) return;
    s_.load(ser_file);
    igl::deserialize(viewer_core_3d_, "vcore3", ser_file);
    igl::deserialize(viewer_core_2d_, "vcore2", ser_file);
    viewer_cores_init = true;

    re_draw_ = true;
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

  v_.ngui->addButton("SaveToOBJ", [&]() {
    Eigen::MatrixXd w_uv3 = Eigen::MatrixXd::Zero(d_.w_uv.rows(),3);
    w_uv3.leftCols(d_.w_uv.cols()) = d_.w_uv;
    igl::writeOBJ(igl::file_dialog_save(),w_uv3, d_.surface_F);
  });

  v_.ngui->addButton("Snapshot", [&]() {
    render_to_png(2*1600, 2*900,
                  igl::file_dialog_save
                      ());
  });

  v_.ngui->addButton("LoadUV", [&]() {
    Eigen::MatrixXd UVV;
    Eigen::MatrixXi UVF;
    igl::read_triangle_mesh(igl::file_dialog_open(),UVV,UVF);
    if(!uv_space)
      v_.data.set_uv(30*UVV,UVF);
    else {
      v_.data.clear();
      v_.data.set_mesh(UVV, UVF);
      v_.data.set_colors(RowVector3d(182. / 255., 215. / 255, 168 / 255.));
      
    }


  });
  v_.ngui->addVariable("ClickMode", mouse_click_mode, true)
    ->setItems({"Choose", "Move", "Dragging","None"});

  // Add an additional menu window
  v_.ngui->addWindow(Eigen::Vector2i(1600 - 220, 10),
                     "Scaffold Tweak");
  v_.ngui->addVariable<bool>("ReMesh", [&reg_scaf](bool val) {
    reg_scaf = val; // set
  }, [&reg_scaf]() {
    return reg_scaf; // get
  });
  v_.ngui->addVariable("AutoWeight", auto_weighting_);
  v_.ngui->addVariable<float>("Scaf Weight",
                              [&ws_solver](float val) {
                                ws_solver->adjust_scaf_weight(val);
                                std::cout << "Weight:" << val
                                          << std::endl;
                              },
                              [this]() {
                                return d_.scaffold_factor;
                              });

  v_.ngui->addVariable<bool>("UV Space", [this](bool val) {
    this->uv_space = val;
    re_draw_ = true;
    if (viewer_cores_init) {
      if (!uv_space) {
        viewer_core_2d_ = v_.core;
        v_.core = viewer_core_3d_;
      } else {
        viewer_core_3d_ = v_.core;
        v_.core = viewer_core_2d_;
      }
    }

    if (continue_computing_) {
      continue_computing_ = false;
      for (auto &t:threads_)
        if (t.joinable())
          t.join();
      threads_.clear();
    }
  }, [this](){return this->uv_space;});

  v_.ngui->addVariable<float>("UV size", [this](float val) {
    this->uv_scale = val;
    re_draw_ = true;
  }, [this](){return this->uv_scale;});

  v_.ngui->addVariable("Current Patch", picked_component);
  v_.ngui->addVariable<double>("Scaling", [this](double sc) {
//    if (picked_component == -1) return; //nonsense
    // wait for computation to be done.
    if (continue_computing_) {
      continue_computing_ = false;
      for (auto &t:threads_)
        if (t.joinable())
          t.join();
      threads_.clear();
    }
// then adjust ws->Dx, Dy by a factor of.
    double change_factor = sc / reference_scaling_;
    s_.ws_solver->enlarge_internal_reference(change_factor);

// then assign to the current scaling.
     reference_scaling_= sc;
  }, [this]() {
    return reference_scaling_;
      });

  v_.ngui->addButton("Clear Constraints", [&]() {
    d_.soft_cons.clear();
    scaffold_coloring();

    v_.ngui->refresh();
  });

  auto window = v_.ngui->addWindow(Eigen::Vector2i(1200, 700),
                                   "Patches");
  auto tools = new nanogui::Widget(window);
  v_.ngui->addWidget("tools",tools);

  tools->setLayout(new nanogui::BoxLayout
                       (nanogui::Orientation::Horizontal,
                        nanogui::Alignment::Middle, 0, 6));

  auto button = new nanogui::Button(tools, "Add Patch");
  button->setCallback([&]() {
    if(!igl::read_triangle_mesh(igl::file_dialog_open(),
                                cache_new_patch_V,cache_new_patch_F)) {
      std::cerr<<"Cannot open mesh!"<<std::endl;
      return;
    }
    continue_computing_ = false;
    for(auto&t:threads_)
      if(t.joinable())
        t.join();
    threads_.clear();
    mouse_click_mode = ClickMode::PLACE_PATCH;
  });
  button = new nanogui::Button(tools, "Choose Patch");
  button->setCallback([&]() {
    continue_computing_ = false;
    for(auto&t:threads_)
      if(t.joinable())
        t.join();
    threads_.clear();
    mouse_click_mode = ClickMode::CHOOSE_PATCH;
  });

  v_.ngui->addWidget("",new nanogui::Label(window, "Image panel & scroll "
      "panel", "sans-bold"));

  auto slider_text = new nanogui::Widget(window);
  slider_text->setLayout(new nanogui::BoxLayout
                             (nanogui::Orientation::Horizontal,
                              nanogui::Alignment::Middle, 0, 20));

  v_.ngui->addWidget("Scale", slider_text);
  nanogui::Slider *slider = new nanogui::Slider(slider_text);
  slider->setRange(std::make_pair(0.5f,1.5f));
  slider->setValue(1.f);
  slider->setFixedWidth(200);

  auto textBox = new nanogui::TextBox(slider_text);
  textBox->setFixedSize(Vector2i(60, 25));
  textBox->setValue("100");
  textBox->setUnits("%");
  slider->setCallback([textBox](float value) {
    textBox->setValue(std::to_string((int) (value * 100)));
  });
  slider->setFinalCallback([&](float value) {
    cout << "Final slider value: " << (int) (value * 100) << endl;
  });
  textBox->setFixedSize(Vector2i(60,25));
  textBox->setFontSize(20);
  textBox->setAlignment(nanogui::TextBox::Alignment::Right);

  auto reset_scale_button = new nanogui::Button(slider_text, "1");
  reset_scale_button->setCallback([slider, textBox]() {
    slider->setValue(1.f);
    textBox->setValue("100");
    std::cout<<"Reset"<<std::endl;
  });
  reset_scale_button->setFixedSize(Vector2i(25,25));

  // Generate menu
  v_.screen->performLayout();

  return false;
}
