//
// Created by Zhongshi Jiang on 5/8/17.
//

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
#include <fstream>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/doublearea.h>
#include <igl/PI.h>
#include <igl/flipped_triangles.h>
#include <igl/file_dialog_open.h>

void parameterization_init(std::string filename, Eigen::MatrixXd& V_ref,
                           Eigen::MatrixXi &F_ref,
                           Eigen::MatrixXd& V_all, Eigen::MatrixXi &F_scaf,
  Eigen::VectorXi &frame_id, Eigen::MatrixXi& display_F) {
  using namespace std;
  using namespace Eigen;
  while (!igl::read_triangle_mesh(filename, V_ref,
                                  F_ref)) {
    std::cerr << "Cannot Open Mesh!" << std::endl;
    filename = igl::file_dialog_open();
  }

  Eigen::MatrixXd uv_init;
  Eigen::VectorXi bnd;
  Eigen::MatrixXd bnd_uv;
  igl::Timer timer;
  timer.start();
  igl::boundary_loop(F_ref, bnd);
  cout << "bndloop = " << timer.getElapsedTime() << endl;
  timer.start();


  timer.start();
  VectorXd M;
  igl::doublearea(V_ref, F_ref, M);
 std::cout<<"sqrtM/2pi"<< sqrt(M.sum()/(2*igl::PI))<<std::endl;
//  M /= M.sum()/igl::PI;

  igl::map_vertices_to_circle(V_ref, bnd, bnd_uv);
  V_ref *= 2;
  bnd_uv *= sqrt(M.sum()/(2*igl::PI));
  cout << "v2circle = " << timer.getElapsedTime() << endl;
  igl::harmonic(V_ref, F_ref, bnd, bnd_uv, 1, uv_init);
  if (igl::flipped_triangles(uv_init, F_ref).size() != 0) {
    igl::harmonic(F_ref, bnd, bnd_uv, 1, uv_init); // use uniform laplacian
  }
  cout << "Harmonic = " << timer.getElapsedTime() << endl;

//  scaffold_generator(uv_init, F_ref, scaf_data.density, Vall, Fscaf);

  MatrixXd V_bnd;
  V_bnd.resize(bnd.size(), uv_init.cols());
  for (int i = 0; i < bnd.size(); i++) // redoing step 1.
  {
    V_bnd.row(i) = uv_init.row(bnd(i));
  }
  Matrix2d ob;// = rect_corners;
  {
    VectorXd uv_max = uv_init.colwise().maxCoeff();
    VectorXd uv_min = uv_init.colwise().minCoeff();
    VectorXd uv_mid = (uv_max + uv_min) / 2.;

    double scaf_range = 8;
    ob.row(0) = uv_mid + scaf_range * (uv_min - uv_mid);
    ob.row(1) = uv_mid + scaf_range * (uv_max - uv_mid);
  }
  Vector2d rect_len;
  rect_len << ob(1, 0) - ob(0, 0), ob(1, 1) - ob(0, 1);

  int frame_points = 5;
  MatrixXd V_rect;
  V_rect.resize(4 * frame_points, 2);
  for (int i = 0; i < frame_points; i++) {
    // 0,0;0,1
    V_rect.row(i) << ob(0, 0), ob(0, 1) + i * rect_len(1) / frame_points;
    // 0,0;1,1
    V_rect.row(i + frame_points)
        << ob(0, 0) + i * rect_len(0) / frame_points, ob(1, 1);
    // 1,0;1,1
    V_rect.row(i + 2 * frame_points) << ob(1, 0), ob(1, 1) - i * rect_len(1) /
        frame_points;
    // 1,0;0,1
    V_rect.row(i + 3 * frame_points)
        << ob(1, 0) - i * rect_len(0) / frame_points, ob(0, 1);
    // 0,0;0,1
  }

  // Concatenate Vert and Edge
  MatrixXd V;
  MatrixXi E;
  igl::cat(1, V_bnd, V_rect, V);
  E.resize(V.rows(), 2);
  for (int i = 0; i < E.rows(); i++)
    E.row(i) << i, i + 1;
  E(bnd.size() - 1, 1) = 0;
  E(V.rows() - 1, 1) = static_cast<int>(bnd.size());

  MatrixXd H = MatrixXd::Zero(10, 2);
  for (int f = 0; f < H.rows(); f++)
    for (int i = 0; i < 3; i++)
      H.row(f) += uv_init.row(F_ref(f, i)); // redoing step 2
  H /= 3.;
  timer.start();
  MatrixXd uv2;
  igl::triangle::triangulate(V, E, H, "qYYQ", uv2, F_scaf);

  auto bnd_n = bnd.size();
  V_all.resize(uv_init.rows() - bnd_n + uv2.rows(), 2);
  V_all.topRows(uv_init.rows()) = uv_init;
  V_all.bottomRows(uv2.rows() - bnd_n) = uv2.bottomRows(-bnd_n + uv2.rows());

  for (auto i = 0; i < F_scaf.rows(); i++)
    for (auto j = 0; j < F_scaf.cols(); j++) {
      auto &x = F_scaf(i, j);
      if (x < bnd_n) x = bnd(x);
      else x += uv_init.rows() - bnd_n;
    }

  cout << "New Cat = " << timer.getElapsedTime() << endl;

  frame_id = Eigen::VectorXi::LinSpaced(
      V_rect.rows(), V_ref.rows(), V_ref.rows() + V_rect.rows() - 1);
  igl::cat(1, F_ref, F_scaf, display_F);
}
