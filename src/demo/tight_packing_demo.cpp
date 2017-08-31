//
// Created by Zhongshi Jiang on 5/19/17.
//

#include <igl/adjacency_matrix.h>
#include <igl/components.h>
#include <iostream>
#include <igl/remove_unreferenced.h>
#include <igl/doublearea.h>
#include <igl/PI.h>
#include "../ScafData.h"
#include "../util/triangle_utils.h"

void tight_packing_init(std::string filename, ScafData& d_) {
  using namespace Eigen;
  using namespace std;

  MatrixXd V; MatrixXi F;
  read_mesh_with_uv_seam(filename, V, F);
  std::cout<<"Vrows"<<V.rows()<<endl;
  std::cout<<"Frows"<<F.rows()<<endl;

  SparseMatrix<double> Adj;
  Eigen::MatrixXi V_conn_flag;
  MatrixXi component_vert_sizes;
  igl::adjacency_matrix(F,Adj);
  igl::components(Adj,V_conn_flag, component_vert_sizes);
  std::cout<<"counts:"<<component_vert_sizes<<std::endl;
  int component_number = component_vert_sizes.size();

  VectorXi component_face_sizes = Eigen::VectorXi::Zero(component_number);
  Eigen::VectorXi F_conn_flag(F.rows());
  double max_area = -1;
  double biggest_part = -1;
  for(int i=0; i<F.rows(); i++) {
    int flag = V_conn_flag(F(i,0));
    F_conn_flag(i) = flag;
    component_face_sizes(flag) ++;
  }
  std::cout<<"comp_face_counts:"<<component_face_sizes<<std::endl;
  std::vector<MatrixXd> separated_V(component_number);
  std::vector<MatrixXi> separated_F(component_number);

  for(int i=0; i<component_number; i++) {
    Eigen::MatrixXi F_temp(component_face_sizes(i),3);
    int filler = 0;
    for(int j=0; j<F.rows(); j++) {
      if(F_conn_flag(j)== i){
        F_temp.row(filler++) = F.row(j);
      }
    }
    VectorXi I;
    igl::remove_unreferenced(V, F_temp, separated_V[i], separated_F[i],I);

    VectorXd M;
    igl::doublearea(separated_V[i], separated_F[i], M);
    if(M.sum() > max_area) {
      max_area = M.sum()/2;
      biggest_part = i;
    }
  }

  double max_rad = sqrt((max_area)/igl::PI);

  int grid_size = std::ceil(sqrt(component_number));

  for(int i=0; i<component_number; i++) {
    int row = i%grid_size;
    int col = i/grid_size;
    RowVector2d center(row*max_rad*4.1, col*max_rad*4.1);
    d_.add_new_patch(separated_V[i], separated_F[i], center);
  }

  return;
}