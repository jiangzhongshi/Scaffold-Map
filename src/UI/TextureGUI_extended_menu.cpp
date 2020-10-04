//
// Created by Zhongshi Jiang on 5/17/17.
//

#include "TextureGUI.h"
#include "../ReWeightedARAP.h"
#include "../StateManager.h"

#include <vector>

#include <igl/unproject_onto_mesh.h>
#include <igl/unproject.h>
#include <igl/project.h>
#include <igl/file_dialog_save.h>
#include <igl/file_dialog_open.h>
#include <igl/writeOBJ.h>
#include <igl/Timer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/png/writePNG.h>
#include <igl/png/readPNG.h>
#include <igl/slice.h>

bool TextureGUI::extended_menu() {
   using namespace Eigen;
   using namespace std;

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
       if (ImGui::Button("SaveUV")) {
            Eigen::MatrixXd w_uv3 = Eigen::MatrixXd::Zero(d_.w_uv.rows(),3);
            w_uv3.leftCols(2) = d_.w_uv;
            igl::writeOBJ(igl::file_dialog_save(),w_uv3, d_.surface_F);
       }

       } // Serialization
   };


//   v_.ngui->addGroup("Serialization");
//   v_.ngui->addButton("Save", [&]() {
//     std::string filename = igl::file_dialog_save();
//     s_.save(filename);
//     if(uv_space) viewer_core_2d_ = v_.core;
//     else viewer_core_3d_ = v_.core;
//     igl::serialize(viewer_core_3d_, "vcore3", filename);
//     igl::serialize(viewer_core_2d_, "vcore2", filename);
//   });
//   v_.ngui->addButton("Load", [&]() {
//     std::string ser_file = igl::file_dialog_open();
//     if(ser_file.empty()) return;
//     s_.load(ser_file);
//     igl::deserialize(viewer_core_3d_, "vcore3", ser_file);
//     igl::deserialize(viewer_core_2d_, "vcore2", ser_file);
//     viewer_cores_init = true;

//     re_draw_ = true;
//     VectorXi b(d_.soft_cons.size());
//     MatrixXd bc(b.rows(), 3);
//     int i = 0;
//     for (auto const &x: d_.soft_cons) {
//       b(i) = x.first;
//       bc.row(i) = x.second;
//       i++;
//     }

//     MatrixXd m_uv = d_.w_uv.topRows(d_.mv_num);

//     ws_solver = s_.ws_solver;

//     // refresh after load
//     v_.data().clear();
//     v_.data().set_mesh(d_.w_uv, d_.surface_F);
//     scaffold_coloring();

//     v_.ngui->refresh();
//   });


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
   if (ImGui::Checkbox("uv space", &uv_space))
   {
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
   }
   if (!uv_space)
         if(ImGui::SliderFloat("uv size", &uv_scale, 0.0f, 10.0f, "ratio = %.3f")) re_draw_ = true;

   ImGui::Text("Current Picked Patch %d", picked_component);

   float current_scaling = reference_scaling_;
   if(ImGui::SliderFloat("Reference Scaling", &current_scaling, 0.1f, 10.0f, "ratio = %.3f")){
           // wait for computation to be done.
           if (continue_computing_) {
             continue_computing_ = false;
             for (auto &t:threads_)
               if (t.joinable())
                 t.join();
             threads_.clear();
           }
       // then adjust ws->Dx, Dy by a factor of.
           float change_factor = current_scaling / reference_scaling_;
           s_.ws_solver->enlarge_internal_reference(
                       static_cast<double>(change_factor));
           reference_scaling_ = current_scaling;
   }

   if(ImGui::Button("Add Patch")) {
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
   }
   if(ImGui::Button("Choose Patch")) {
            continue_computing_ = false;
            for(auto&t:threads_)
              if(t.joinable())
                t.join();
            threads_.clear();
            mouse_click_mode = ClickMode::CHOOSE_PATCH;
   }
//   v_.ngui->addButton("SaveToOBJ", [&]() {
//     Eigen::MatrixXd w_uv3 = Eigen::MatrixXd::Zero(d_.w_uv.rows(),3);
//     w_uv3.leftCols(d_.w_uv.cols()) = d_.w_uv;
//     igl::writeOBJ(igl::file_dialog_save(),w_uv3, d_.surface_F);
//   });

//   v_.ngui->addButton("Snapshot", [&]() {
//     render_to_png(2*1600, 2*900,
//                   igl::file_dialog_save
//                       ());
//   });

//   v_.ngui->addButton("LoadUV", [&]() {
//     Eigen::MatrixXd UVV;
//     Eigen::MatrixXi UVF;
//     igl::read_triangle_mesh(igl::file_dialog_open(),UVV,UVF);
//     if(!uv_space)
//       v_.data().set_uv(30*UVV,UVF);
//     else {
//       v_.data().clear();
//       v_.data().set_mesh(UVV, UVF);
//       v_.data().set_colors(RowVector3d(182. / 255., 215. / 255, 168 / 255.));
      
//     }


//   });
//   v_.ngui->addVariable("ClickMode", mouse_click_mode, true)
//     ->setItems({"Choose", "Move", "Dragging","None"});// // then assign to the current scaling.
//   auto window = v_.ngui->addWindow(Eigen::Vector2i(1200, 700),
//                                    "Patches");

//   auto tools = new nanogui::Widget(window);
//   v_.ngui->addWidget("tools",tools);


//   auto button = new nanogui::Button(tools, "Add Patch");
//   button->setCallback([&]() {
//     if(!igl::read_triangle_mesh(igl::file_dialog_open(),
//                                 cache_new_patch_V,cache_new_patch_F)) {
//       std::cerr<<"Cannot open mesh!"<<std::endl;
//       return;
//     }
//     continue_computing_ = false;
//     for(auto&t:threads_)
//       if(t.joinable())
//         t.join();
//     threads_.clear();
//     mouse_click_mode = ClickMode::PLACE_PATCH;
//   });
//   button = new nanogui::Button(tools, "Choose Patch");
//   button->setCallback([&]() {
//     continue_computing_ = false;
//     for(auto&t:threads_)
//       if(t.joinable())
//         t.join();
//     threads_.clear();
//     mouse_click_mode = ClickMode::CHOOSE_PATCH;
//   });

//   v_.ngui->addWidget("",new nanogui::Label(window, "Image panel & scroll "
//       "panel", "sans-bold"));

//   auto slider_text = new nanogui::Widget(window);
//   slider_text->setLayout(new nanogui::BoxLayout
//                              (nanogui::Orientation::Horizontal,
//                               nanogui::Alignment::Middle, 0, 20));

//   v_.ngui->addWidget("Scale", slider_text);
//   nanogui::Slider *slider = new nanogui::Slider(slider_text);
//   slider->setRange(std::make_pair(0.5f,1.5f));
//   slider->setValue(1.f);
//   slider->setFixedWidth(200);

//   auto textBox = new nanogui::TextBox(slider_text);
//   textBox->setFixedSize(Vector2i(60, 25));
//   textBox->setValue("100");
//   textBox->setUnits("%");
//   slider->setCallback([textBox](float value) {
//     textBox->setValue(std::to_string((int) (value * 100)));
//   });
//   slider->setFinalCallback([&](float value) {
//     cout << "Final slider value: " << (int) (value * 100) << endl;
//   });
//   textBox->setFixedSize(Vector2i(60,25));
//   textBox->setFontSize(20);
//   textBox->setAlignment(nanogui::TextBox::Alignment::Right);

//   auto reset_scale_button = new nanogui::Button(slider_text, "1");
//   reset_scale_button->setCallback([slider, textBox]() {
//     slider->setValue(1.f);
//     textBox->setValue("100");
//     std::cout<<"Reset"<<std::endl;
//   });
//   reset_scale_button->setFixedSize(Vector2i(25,25));

       ImGui::End();
   };
  return false;
}
