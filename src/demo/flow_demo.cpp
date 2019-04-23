//
// Created by Zhongshi Jiang on 5/7/17.
//

#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <Eigen/Core>
#include <igl/opengl/glfw/Viewer.h>
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
#include "../util/tetgenio_parser.h"

#include "../util/tet_utils.h"
void flow_initializer(std::string filename,
                      std::string target_file,
    Eigen::MatrixXd &mTV,
                          Eigen::MatrixXi &mTT,
                          Eigen::MatrixXd &wTV,
                          Eigen::MatrixXi &sTT,
                          Eigen::VectorXi &frame_id,
                          Eigen::MatrixXi &surface_F,
                          int& inner_scaf_tets) {
  using namespace Eigen;
  using namespace std;
  Eigen::MatrixXd V_init;


  Eigen::MatrixXd cube_V, id_cube_V;
  Eigen::MatrixXi cube_F;
  igl::read_triangle_mesh("../nice_cube.obj", id_cube_V, cube_F);

  MatrixXi mTF;
  igl::read_triangle_mesh(target_file, mTV, mTT);
  igl::read_triangle_mesh(filename, V_init, surface_F);

  cube_V.resizeLike(id_cube_V);
  double max_V0 = V_init.maxCoeff();
  for (int i = 0; i < 3; i++)
    cube_V.col(i) = 3 * max_V0 * id_cube_V.col(i);
  MatrixXd wV;
  MatrixXi wF, TF;

  MatrixXi F2_m = cube_F.array() + V_init.rows();
  igl::cat(1, surface_F, F2_m, wF);
  igl::cat(1, V_init, cube_V, wV);

  MatrixXd TR,R;
  igl::dev::tetgen::tetrahedralize(wV, wF, R, "pqYYA", wTV, sTT, TF,
                                   TR);
  vector<RowVector4i> TT_R1, TT_R2;
  for(int i=0; i<TR.rows(); i++) {
    if(TR(i) == 2)
      TT_R1.push_back(sTT.row(i));
    else
      TT_R2.push_back(sTT.row(i));
  }

  inner_scaf_tets = TT_R1.size();
  for(int i=0; i<inner_scaf_tets; i++) {
    sTT.row(i) = TT_R1[i];
  }
  for(int i=0; i<TT_R2.size(); i++) {
    sTT.row(i + inner_scaf_tets) = TT_R2[i];
  }

  auto tet_quality = [&wTV](int a, int b, int c, int d) -> double {
    return -quality_utils::quality(wTV.row(a), wTV.row(b),
                                   wTV.row(c), wTV.row(d));
  };
  {
    double min_q = INFINITY;
    for (auto i=0; i<sTT.rows(); i++) {
      RowVector4i r = sTT.row(i);
      min_q = (std::min)(min_q, tet_quality(r(0), r(1), r(2), r(3)));
    }
    cout << endl << "q_in" << min_q << '\t';
    assert(min_q > 0);
  }


  int bnd_size = cube_V.rows();
  frame_id = Eigen::VectorXi::LinSpaced(
      bnd_size, V_init.rows(), V_init.rows() + bnd_size - 1);

}
