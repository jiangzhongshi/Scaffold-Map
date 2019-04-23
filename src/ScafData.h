//
// Created by Zhongshi Jiang on 2/12/17.
//

#ifndef SCAFFOLD_TEST_SCAFDATA_H
#define SCAFFOLD_TEST_SCAFDATA_H

#include <Eigen/Dense>
#include <map>
#include <igl/MappingEnergyType.h>
#include <vector>

struct ScafData {
  // dimension for domain of parameterization/deformation
 public:
  ScafData();
  ScafData(Eigen::MatrixXd &mesh_V, Eigen::MatrixXi &mesh_F,
           Eigen::MatrixXd &all_V, Eigen::MatrixXi &scaf_T);
  void add_new_patch(const Eigen::MatrixXd&, const Eigen::MatrixXi&,
                     const Eigen::RowVectorXd &center);

  void mesh_improve(bool square_frame=false, bool expand_frame=true);
  void mesh_improve_3d(bool expand_frame);

  void automatic_expand_frame(double min=2.0, double max = 3.0);

  void add_soft_constraints(int b,
                            const Eigen::RowVectorXd &bc);
  void add_soft_constraints(const Eigen::VectorXi &b,
                            const Eigen::MatrixXd &bc);
  void update_scaffold();
  void set_scaffold_factor(double weight);

  double scaffold_factor = 10;

  typedef igl::MappingEnergyType SLIM_ENERGY;

  SLIM_ENERGY slim_energy = SLIM_ENERGY::SYMMETRIC_DIRICHLET;

  SLIM_ENERGY scaf_energy = SLIM_ENERGY::SYMMETRIC_DIRICHLET;

// Optional Input

  double exp_factor = 1.0; // used for exponential energies, ignored otherwise

// Output
  double energy; // objective value

// INTERNAL
  long mv_num, mf_num;
  long sv_num, sf_num;
  Eigen::MatrixXd m_V; // input initial mesh V
  Eigen::MatrixXi m_T; // input initial mesh F/T

  Eigen::MatrixXd w_uv; // whole domain uv: mesh + free vertices
  Eigen::MatrixXi s_T; // scaffold domain tets: scaffold tets
  Eigen::MatrixXi w_T;

  Eigen::VectorXd m_M; // mesh area or volume
  Eigen::VectorXd s_M; // scaffold area or volume
  Eigen::VectorXd w_M; // area/volume weights for whole
  double mesh_measure; // area or volume
  double avg_edge_length;
  long v_num;
  long f_num;
  double proximal_p = 1e-8; //unused

  std::map<int, Eigen::RowVectorXd> soft_cons;
  double soft_const_p = 1e4;

  Eigen::VectorXi frame_ids;

 private:
  Eigen::VectorXd soft_b;
  Eigen::MatrixXd soft_bc;

 public: // public for ser
  // caching
  Eigen::VectorXi internal_bnd;
  Eigen::MatrixXd rect_frame_V;

  // multi-chart support
  std::vector<int> component_sizes;
  std::vector<int> bnd_sizes;

  //3D
 public:
  Eigen::MatrixXi surface_F;

  int dim; // dimension for ambient space. Same for mesh/scaf

  // flow arap
  int inner_scaf_tets = 0; 
          
  };


  #endif //SCAFFOLD_TEST_SCAFDATA_H
