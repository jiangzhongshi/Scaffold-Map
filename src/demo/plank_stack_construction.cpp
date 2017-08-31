//
// Created by Zhongshi Jiang on 4/17/17.
//
#include "../util/triangle_utils.h"

#include "../util/tetgenio_parser.h"
#include "../ScafData.h"

#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <Eigen/Core>
#include <igl/viewer/Viewer.h>
#include <igl/triangle/triangulate.h>
#include <igl/cat.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <igl/writeMESH.h>
#include <igl/readMESH.h>
#include <igl/barycenter.h>
#include <igl/read_triangle_mesh.h>
#include <iostream>
#include <igl/Timer.h>
#include <igl/doublearea.h>
#include <igl/slice.h>

void bars_stack_construction(ScafData& d_) {
  using namespace Eigen;
  using namespace std;

  d_.dim = 2;

  const double thick = 1;
  const double space = 4; // vertical
  const int length = 100;
  const int layer = 400; // this is to be going to infinity!
  const int num_hole = 1; // to prevent bug of triangle.
  assert(num_hole < length);

  auto v_num = 2 + length * 2;
  auto f_num = length * 2;
  Eigen::MatrixXd H(num_hole * layer, 2);
  Eigen::MatrixXd V(layer * v_num, 2);
  Eigen::MatrixXi F(layer * f_num, 3);
  Eigen::MatrixXi E(layer * v_num, 2);

  d_.internal_bnd.resize(v_num * layer);
  d_.component_sizes.resize(layer);
  for(auto &c:d_.component_sizes) c = f_num;

  d_.bnd_sizes.resize(layer);
  for(auto &c: d_.bnd_sizes) c = v_num;

  for (auto l = 0; l < layer; l++) {
    auto start_v = l * v_num;
    auto start_f = l * f_num;
    for (int i = 0; i < length + 1; i++) {
      V.row(start_v + 2 * i) << i * thick, l * space;
      V.row(start_v + 2 * i + 1) << i * thick, l * space + thick;
    }
    for (int i = 0; i < length; i++) {
      F.row(start_f + 2 * i) =
          Eigen::Array3i(2 * i, 2 * i + 2, 2 * i + 1) + start_v;
      F.row(start_f + 2 * i + 1) =
          Eigen::Array3i(2 * i + 1, 2 * i + 2, 2 * i + 3) + start_v;
    }

    for (int i = 0; i <= length; i++) {
      E.row(start_v + i) << start_v + 2 * i, start_v + 2 * i + 2;
      E.row(start_v + length + i + 1) <<
                                      start_v + (v_num-1 - 2*i),
          start_v + (v_num - 1 - 2 * i - 2);
    }
    E(start_v + length, 1) = start_v + (v_num-1);
    E(start_v + 2 * length + 1, 1) = start_v;

//    E.row(start_v + length) << start_v + 2 * length, start_v + 2 * length + 1;
//    E.row(start_v + 2 * length + 1) << start_v + 1, start_v + 0;

    for (int i = 0; i < num_hole; i++) {
      RowVector2d hh(0, 0);
      for (auto v : {0, 1, 2})
        hh += V.row(F(start_f + i, v));
      H.row(l * num_hole + i) = hh / 3;
    }
  }

  Matrix2d ob;// = rect_corners;
  {
    VectorXd uv_max = V.colwise().maxCoeff();
    VectorXd uv_min = V.colwise().minCoeff();
    VectorXd uv_mid = (uv_max + uv_min) / 2.;
    ob.row(0) = uv_mid;
    ob.row(1) = uv_mid;

    double scaf_range = 3;
    Array2d scaf_scale(2, 1.5);
    ob.row(0) += (scaf_scale * (uv_min - uv_mid).array()).matrix();
    ob.row(1) += (scaf_scale * (uv_max - uv_mid).array()).matrix();
  }
  Vector2d rect_len;
  rect_len << ob(1, 0) - ob(0, 0), ob(1, 1) - ob(0, 1);

  int rect_side = 6;
  MatrixXd V_rect(2 * rect_side, 2);
  MatrixXi E_rect(2 * rect_side, 2);
  for (int i = 0; i < rect_side; i++) {
    V_rect.row(i) << ob(0, 0) + i * rect_len(0) / (rect_side - 1), ob(0, 1);
    V_rect.row(rect_side + i) << ob(0, 0) + i * rect_len(0) / (rect_side - 1),
        ob(1, 1);

    E_rect.row(i) << i, i + 1;
    E_rect.row(rect_side + i) << 2 * rect_side - 1 - i, 2 * rect_side - 2 - i;
  }
  E_rect.row(rect_side - 1) << rect_side - 1, 2 * rect_side - 1;
  E_rect.row(2 * rect_side - 1) << rect_side, 0;

  MatrixXd whole_V;
  igl::cat(1, V, V_rect, whole_V);
  MatrixXi whole_E(V.rows() + V_rect.rows(), 2);
  whole_E.topRows(V.rows()) = E;
  whole_E.bottomRows(V_rect.rows()) = E_rect.array() + V.rows();

  MatrixXd s_uv;
  MatrixXi s_F;
//  igl::triangle::triangulate(whole_V,whole_E, H, "qQ", s_uv, s_F);
//  cout<<V<<endl<<E_rect<<endl;
//  mesh_cat(V,F,s_uv,s_F, w_uv,w_F);

  d_.mv_num = V.rows();
  d_.mf_num = F.rows();
  d_.internal_bnd = E.col(0);
  d_.m_V = Eigen::MatrixXd::Zero(d_.mv_num,3);
  d_.m_V.leftCols(2) = V;
  d_.w_uv = V;
  d_.m_T = F;
  igl::doublearea(d_.m_V, d_.m_T, d_.m_M);
  d_.m_M /= 2.;
  d_.surface_F = d_.m_T;
  d_.mesh_measure = d_.m_M.sum();

  // add constraints.
  Eigen::VectorXi bottom_ids = Eigen::VectorXi::LinSpaced(length+2, 0,
                                                          length+1);
  Eigen::MatrixXd bottom_coords;
  igl::slice(d_.w_uv, bottom_ids, 1, bottom_coords);
  d_.add_soft_constraints(bottom_ids, bottom_coords);

  int top_pick = d_.mv_num - length/2;
//  d_.add_soft_constraints(top_pick, );

  d_.mesh_improve();
//  v_.data.add_points(H,Eigen::RowVector3d(1,0,0) );
//  MatrixXd mesh_color(w_F.rows(),3);
//  for (int i = 0; i < mf_num; i++)
//    mesh_color.row(i) << 148/255., 195/255., 128/255.;
//
//  for (int i = mf_num; i < w_F.rows(); i++)
//    mesh_color.row(i) << 0.86, 0.86, 0.86;
//  v_.data.set_colors(mesh_color);
//  MatrixXd w_uv3 =MatrixXd::Zero(w_uv.rows(),3);
//  w_uv3.leftCols(2) = w_uv;
//  igl::writeOBJ("400layer.obj",w_uv3,w_F);
}

void bars_stack_construction(Eigen::MatrixXd w_uv, Eigen::MatrixXi w_F,
int mv_num, int mf_num) {
  using namespace Eigen;
  using namespace std;

  const double thick = 1;
  const double space = 4; // vertical
  const int length = 100;
  const int layer = 8; // this is to be going to infinity!
  const int num_hole = 1; // to prevent bug of triangle.
  assert(num_hole < length);

  auto v_num = 2 + length * 2;
  auto f_num = length * 2;
  Eigen::MatrixXd H(num_hole * layer, 2);
  Eigen::MatrixXd V(layer * v_num, 2);
  Eigen::MatrixXi F(layer * f_num, 3);
  Eigen::MatrixXi E(layer * v_num, 2);

  for (auto l = 0; l < layer; l++) {
    auto start_v = l * v_num;
    auto start_f = l * f_num;
    for (int i = 0; i < length + 1; i++) {
      V.row(start_v + 2 * i) << i * thick, l * space;
      V.row(start_v + 2 * i + 1) << i * thick, l * space + thick;
    }
    for (int i = 0; i < length; i++) {
      F.row(start_f + 2 * i) =
          Eigen::Array3i(2 * i, 2 * i + 2, 2 * i + 1) + start_v;
      F.row(start_f + 2 * i + 1) =
          Eigen::Array3i(2 * i + 1, 2 * i + 2, 2 * i + 3) + start_v;
    }

    for (int i = 0; i < length; i++) {
      E.row(start_v + i) << start_v + 2 * i, start_v + 2 * i + 2;
      E.row(start_v + length + i + 1) << start_v + 2 * i + 3, start_v + 2 * i
          + 1;
    }
    E.row(start_v + length) << start_v + 2 * length, start_v + 2 * length + 1;
    E.row(start_v + 2 * length + 1) << start_v + 1, start_v + 0;

    for (int i = 0; i < num_hole; i++) {
      RowVector2d hh(0, 0);
      for (auto v : {0, 1, 2})
        hh += V.row(F(start_f + i, v));
      H.row(l * num_hole + i) = hh / 3;
    }
  }

  Matrix2d ob;// = rect_corners;
  {
    VectorXd uv_max = V.colwise().maxCoeff();
    VectorXd uv_min = V.colwise().minCoeff();
    VectorXd uv_mid = (uv_max + uv_min) / 2.;
    ob.row(0) = uv_mid;
    ob.row(1) = uv_mid;

    double scaf_range = 3;
    Array2d scaf_scale(2, 1.5);
    ob.row(0) += (scaf_scale * (uv_min - uv_mid).array()).matrix();
    ob.row(1) += (scaf_scale * (uv_max - uv_mid).array()).matrix();
  }
  Vector2d rect_len;
  rect_len << ob(1, 0) - ob(0, 0), ob(1, 1) - ob(0, 1);

  int rect_side = 6;
  MatrixXd V_rect(2 * rect_side, 2);
  MatrixXi E_rect(2 * rect_side, 2);
  for (int i = 0; i < rect_side; i++) {
    V_rect.row(i) << ob(0, 0) + i * rect_len(0) / (rect_side - 1), ob(0, 1);
    V_rect.row(rect_side + i) << ob(0, 0) + i * rect_len(0) / (rect_side - 1),
        ob(1, 1);

    E_rect.row(i) << i, i + 1;
    E_rect.row(rect_side + i) << 2 * rect_side - 1 - i, 2 * rect_side - 2 - i;
  }
  E_rect.row(rect_side - 1) << rect_side - 1, 2 * rect_side - 1;
  E_rect.row(2 * rect_side - 1) << rect_side, 0;

  MatrixXd whole_V;
  igl::cat(1, V, V_rect, whole_V);
  MatrixXi whole_E(V.rows() + V_rect.rows(), 2);
  whole_E.topRows(V.rows()) = E;
  whole_E.bottomRows(V_rect.rows()) = E_rect.array() + V.rows();

  MatrixXd s_uv;
  MatrixXi s_F;
//  igl::triangle::triangulate(whole_V,whole_E, H, "qQ", s_uv, s_F);
//  cout<<V<<endl<<E_rect<<endl;
//  mesh_cat(V,F,s_uv,s_F, w_uv,w_F);

  mv_num = V.rows();
  mf_num = F.rows();

  igl::viewer::Viewer v_;
  v_.data.set_mesh(V, F);
//  v_.data.add_points(H,Eigen::RowVector3d(1,0,0) );
//  MatrixXd mesh_color(w_F.rows(),3);
//  for (int i = 0; i < mf_num; i++)
//    mesh_color.row(i) << 148/255., 195/255., 128/255.;
//
//  for (int i = mf_num; i < w_F.rows(); i++)
//    mesh_color.row(i) << 0.86, 0.86, 0.86;
//  v_.data.set_colors(mesh_color);
//  MatrixXd w_uv3 =MatrixXd::Zero(w_uv.rows(),3);
//  w_uv3.leftCols(2) = w_uv;
//  igl::writeOBJ("400layer.obj",w_uv3,w_F);
  v_.launch();
}

void plank_stack_construction(Eigen::MatrixXd &TV, Eigen::MatrixXi &TT,
                              Eigen::MatrixXi &TF, int layer,
                              Eigen::VectorXi &H) {
  using namespace Eigen;

  MatrixXd TV0, TV1;
  MatrixXi TT0, TT1, TF0, TF1;
  igl::readMESH("../models/plank.mesh", TV0, TT0, TF0);
  const int plank_vert = TV0.rows();
  int start_vert = 0;
  double height = 3;
  const int num_holes = 3;

  H.resize(layer);
  TV.resize(layer*TV0.rows(),3);
  TT.resize(layer*TT0.rows(),4);
  TF.resize(layer*TF0.rows(),3);

  for (int i = 0; i < layer; i++) {
    start_vert = i*TV0.rows();

    TV1 = TV0;
    TV1.col(0) = TV0.col(0).array() + i* height;

    TT1 = TT0.array() + start_vert;
    TF1 = TF0.array() + start_vert;

    TV.block(i*TV0.rows(), 0, TV0.rows(), 3) = TV1;
    TT.block(i*TT0.rows(), 0, TT0.rows(), 4) = TT1;
    TF.block(i*TF0.rows(), 0, TF0.rows(), 3) = TF1;
    H(i) = TT0.rows()*i;
  }

  Vector3d max_V = TV.colwise().maxCoeff();
  Vector3d min_V = TV.colwise().minCoeff();
  RowVector3d mean_V = (max_V + min_V)/2.0;
  for(int i=0; i<TV.rows(); i++){
    TV.row(i) -= mean_V;
  }

}


Eigen::MatrixXd g_cube_V;
Eigen::MatrixXi g_cube_F;
Eigen::VectorXi Hid;
Eigen::MatrixXi g_TF0;
#include <igl/readMESH.h>
void scaf_plank_stack_construction(Eigen::MatrixXd & mTV,
Eigen::MatrixXi &mTT, Eigen::MatrixXd &wTV, Eigen::MatrixXi &sTT,
Eigen::VectorXi& frame_id) {
  using namespace Eigen;
  using namespace std;
  MatrixXd cube_V, H, wV;
  MatrixXi wF, TF;

//  igl::readMESH("../models/bumpy.mesh", mTV,mTT, g_TF0);
//  Hid.resize(1); Hid(0) = 0;
  plank_stack_construction(mTV, mTT, g_TF0, 8, Hid);
  igl::read_triangle_mesh("../models/cube_cc1.obj", g_cube_V, g_cube_F);

  igl::Timer timer; timer.start();
  RowVector3d max_V0 = mTV.colwise().maxCoeff();
  cube_V.resizeLike(g_cube_V);
  for (int i = 0; i < 3; i++) cube_V.col(i) = 2*3 * max_V0(i) * g_cube_V.col(i);
//cV*=30;

  MatrixXi F2_m = g_cube_F.array() + mTV.rows();

  igl::cat(1, g_TF0, F2_m, wF);
  igl::cat(1, mTV, cube_V, wV);
  H = MatrixXd::Zero(Hid.size(),3);
  for(int i=0; i<Hid.size(); i++) {
    for(auto v:{0,1,2,3})  H.row(i) += (mTV.row(mTT(Hid(i),v)))/4;
  }
  Eigen::VectorXd TR;
  igl::dev::tetgen::tetrahedralize(wV, wF, H,
                                        "pqY", wTV, sTT, TF, TR);

  int bnd_size = cube_V.rows();
  frame_id = Eigen::VectorXi::LinSpaced(
      bnd_size,mTV.rows(),mTV.rows()+bnd_size-1);

}
