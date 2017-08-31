//
// Created by Zhongshi Jiang on 5/7/17.
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

void leg_flow_initializer(Eigen::MatrixXd &mTV,
                          Eigen::MatrixXi &mTT,
                          Eigen::MatrixXd &wTV,
                          Eigen::MatrixXi &sTT,
                          Eigen::VectorXi &frame_id,
                          Eigen::MatrixXi &surface_F) {
  using namespace Eigen;
  using namespace std;
  Eigen::MatrixXd V_init;

  igl::read_triangle_mesh("../models/leg-input.off", mTV, mTT);

  Eigen::MatrixXd cube_V, id_cube_V;
  Eigen::MatrixXi cube_F;
  igl::read_triangle_mesh("../models/cube_cc1.obj", id_cube_V, cube_F);

  MatrixXi mTF;
  igl::read_triangle_mesh("../models/leg-flow.off", V_init, surface_F);
  cube_V.resizeLike(id_cube_V);
  RowVector3d max_V0 = V_init.colwise().maxCoeff();
  for (int i = 0; i < 3; i++)
    cube_V.col(i) = 2 * 3 *
        max_V0(i) * id_cube_V.col(i);
  MatrixXd wV;
  MatrixXi wF, TF;
  MatrixXd H;

  MatrixXi F2_m = cube_F.array() + V_init.rows();
  igl::cat(1, surface_F, F2_m, wF);
  igl::cat(1, V_init, cube_V, wV);

  std::string name = "leg_cube.mesh";
  ifstream f(name.c_str());
  if (!f.good()) {
    igl::copyleft::tetgen::tetrahedralize(wV, wF, "pqYc", wTV, sTT, TF);
    f.close();
    igl::writeMESH(name, wTV, sTT, TF);
  } else {
    f.close();
    igl::readMESH(name, wTV, sTT, TF);
  }
  int bnd_size = cube_V.rows();
  frame_id = Eigen::VectorXi::LinSpaced(
      bnd_size, V_init.rows(), V_init.rows() + bnd_size - 1);

}