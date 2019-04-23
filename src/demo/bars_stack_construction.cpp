//
// Created by Zhongshi Jiang on 4/17/17.
//

#include <Eigen/Core>
#include <igl/viewer/Viewer.h>
#include <igl/triangle/triangulate.h>
#include <igl/cat.h>
#include <igl/writeOBJ.h>

void bars_stack_construction(Eigen::MatrixXd w_uv, Eigen::MatrixXi w_F,
int mv_num, int mf_num) {
  using namespace Eigen;
  using namespace std;

  const double thick =  1;
  const double space = 4; // vertical
  const int length = 100;
  const int layer = 400; // this is to be going to infinity!
  const int num_hole = 1; // to prevent bug of triangle.
  assert(num_hole < length);

  auto v_num = 2 + length*2;
  auto f_num = length*2;
  Eigen::MatrixXd H(num_hole*layer,2);
  Eigen::MatrixXd V(layer*v_num, 2);
  Eigen::MatrixXi F(layer*f_num, 3);
  Eigen::MatrixXi E(layer*v_num ,2);

  for (auto l=0; l<layer; l++) {
    auto start_v = l * v_num;
    auto start_f = l * f_num;
    for(int i=0; i<length+1; i++) {
      V.row(start_v + 2 * i) << i * thick, l * space;
      V.row(start_v + 2 * i + 1) << i * thick, l * space + thick;
    }
    for(int i=0; i<length; i++) {
      F.row(start_f + 2*i) = Eigen::Array3i(2*i, 2*i+2,2*i+1)+start_v;
      F.row(start_f + 2*i+1) = Eigen::Array3i(2*i+1, 2*i+2,2*i+3)+start_v;
    }

    for(int i=0; i<length; i++) {
      E.row(start_v+i) << start_v+ 2*i, start_v + 2*i+2;
      E.row(start_v+length+i+1)<<start_v+2*i+3, start_v+2*i+1;
    }
    E.row(start_v+length) << start_v+2*length, start_v+2*length+1;
    E.row(start_v+2*length+1) << start_v+1,start_v+0;

    for(int i=0 ; i<num_hole; i++) {
      RowVector2d hh(0,0);
      for(auto v : {0,1,2})
        hh += V.row(F(start_f + i, v));
      H.row(l*num_hole + i) = hh/3;
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
    Array2d scaf_scale(2,1.5);
    ob.row(0) += (scaf_scale * (uv_min - uv_mid).array()).matrix();
    ob.row(1) += (scaf_scale * (uv_max - uv_mid).array()).matrix();
  }
  Vector2d rect_len;
  rect_len << ob(1, 0) - ob(0, 0), ob(1, 1) - ob(0, 1);

  int rect_side = 6;
  MatrixXd V_rect(2*rect_side,2);
  MatrixXi E_rect(2*rect_side,2);
  for(int i=0; i<rect_side; i++) {
    V_rect.row(i) << ob(0, 0) + i * rect_len(0) / (rect_side-1), ob(0, 1);
    V_rect.row(rect_side + i) << ob(0, 0) + i * rect_len(0) / (rect_side-1),
        ob(1, 1);

    E_rect.row(i) << i, i + 1;
    E_rect.row(rect_side + i) << 2 * rect_side - 1 - i, 2 * rect_side - 2 - i;
  }
  E_rect.row(rect_side - 1) << rect_side - 1, 2*rect_side - 1;
  E_rect.row(2*rect_side - 1) << rect_side, 0;



  MatrixXd whole_V;
  igl::cat(1, V, V_rect, whole_V);
  MatrixXi whole_E(V.rows() + V_rect.rows(), 2);
  whole_E.topRows(V.rows()) = E;
  whole_E.bottomRows(V_rect.rows()) = E_rect.array() + V.rows();

  MatrixXd s_uv; MatrixXi s_F;
  igl::triangle::triangulate(whole_V,whole_E, H, "qQ", s_uv, s_F);
//  cout<<V<<endl<<E_rect<<endl;
  mesh_cat(V,F,s_uv,s_F, w_uv,w_F);

  mv_num = V.rows();
  mf_num = F.rows();

  igl::viewer::Viewer v_;
  v_.data.set_mesh(w_uv,w_F);
  v_.data.add_points(H,Eigen::RowVector3d(1,0,0) );
  MatrixXd mesh_color(w_F.rows(),3);
  for (int i = 0; i < mf_num; i++)
    mesh_color.row(i) << 148/255., 195/255., 128/255.;

  for (int i = mf_num; i < w_F.rows(); i++)
    mesh_color.row(i) << 0.86, 0.86, 0.86;
  v_.data.set_colors(mesh_color);
  MatrixXd w_uv3 =MatrixXd::Zero(w_uv.rows(),3);
  w_uv3.leftCols(2) = w_uv;
  igl::writeOBJ("400layer.obj",w_uv3,w_F);
  v_.launch();
}