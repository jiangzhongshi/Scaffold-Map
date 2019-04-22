//
// Created by Zhongshi Jiang on 10/9/16.
//

#ifndef SCAFFOLD_TEST_WEIGHTEDSCAFFOLD_H
#define SCAFFOLD_TEST_WEIGHTEDSCAFFOLD_H

#include "ScafData.h"
#include <igl/serialize.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <igl/arap.h>

class ScafData;

class ReWeightedARAP
{

public:
  ReWeightedARAP(ScafData &state) : d_(state)
  {};

  void pre_calc() ;

  void solve_weighted_proxy(Eigen::MatrixXd &uv_new);

  double compute_energy(const Eigen::MatrixXd &V_new,
                                              bool whole = true) ;

  double perform_iteration(Eigen::MatrixXd &w_uv) ;
  double perform_iteration(Eigen::MatrixXd &w_uv,bool dummy) ;
  void change_scaffold_reference(const Eigen::MatrixXd &s_uv) ; // Go To ScafData
private: // utilities
  static void simplified_covariance_scatter_matrix(
      const Eigen::MatrixXd & V,
      const Eigen::MatrixXi & F,
      Eigen::SparseMatrix<double>& Dx,Eigen::SparseMatrix<double>& Dy,
      Eigen::SparseMatrix<double>& Dz );

  void compute_scaffold_gradient_matrix(Eigen::SparseMatrix<double> &D1,
                                        Eigen::SparseMatrix<double> &D2);

  void compute_jacobians(const Eigen::MatrixXd &V_o, bool whole = true);

  static void compute_jacobians(const Eigen::MatrixXd &uv,
                                         const Eigen::SparseMatrix<double>& Dx,
                                         const Eigen::SparseMatrix<double> &Dy,
                                         const Eigen::SparseMatrix<double> &Dz,
                                         Eigen::MatrixXd& Ji);

  double compute_soft_constraint_energy(const Eigen::MatrixXd &uv) const ;

  void solve_weighted_arap(Eigen::MatrixXd &uv);

 private: // build and solve linear system

  void build_linear_system(Eigen::SparseMatrix<double> &L,
                           Eigen::VectorXd &rhs) const;

  // util
  void build_surface_linear_system(Eigen::SparseMatrix<double> &L,
                                           Eigen::VectorXd &rhs) const;

  // util
  void build_scaffold_linear_system(Eigen::SparseMatrix<double> &L,
                                   Eigen::VectorXd &rhs) const;

  static void buildAm(const Eigen::VectorXd &sqrt_M,
               const Eigen::SparseMatrix<double> &Dx,
               const Eigen::SparseMatrix<double> &Dy,
               const Eigen::SparseMatrix<double> &Dz,
               const Eigen::MatrixXd &W,
               Eigen::SparseMatrix<double> &Am) ;


  static void buildRhs(const Eigen::VectorXd &sqrt_M,
                      const Eigen::MatrixXd &W,
                      const Eigen::MatrixXd& Ri,
                      Eigen::VectorXd &f_rhs) ;

 private: // variables
  ScafData &d_;
  Eigen::VectorXd M_m, M_s; // ScafData
  Eigen::MatrixXd Ri_m, Ji_m, Ri_s, Ji_s;
  Eigen::MatrixXd W_m, W_s;

  Eigen::SparseMatrix<double> Dx_s, Dy_s, Dz_s; //ScafData
  Eigen::SparseMatrix<double> Dx_m, Dy_m, Dz_m; // ScafData

  int f_n, v_n;

 public:
  bool has_pre_calc = false;

  // go to util
  static void compute_surface_gradient_matrix(
      const Eigen::MatrixXd &V,
      const Eigen::MatrixXi &F,
      const Eigen::MatrixXd &F1,
      const Eigen::MatrixXd &F2,
      Eigen::SparseMatrix<double, 0, int> &D1,
      Eigen::SparseMatrix<double, 0, int> &D2);
 public:
  // go to ScafData
  void after_mesh_improve();
  void enlarge_internal_reference(double); // ScafData

  void compute_jacobians(const Eigen::MatrixXd &V_new) const;


public: //arap
  igl::ARAPData arap_data;
  Eigen::SparseMatrix<double> CotMat; // semi-negative definite.
//  Eigen::MatrixXd arap_Bc;
  std::vector<Eigen::Matrix3d> arap_rots;
  Eigen::MatrixXd arap_cot_entries;
  double arap_energy_p = 1e5;
  double compute_surface_ARAP_energy(const Eigen::MatrixXd &uv) const;
  void update_surface_ARAP_rots();

};

#endif //SCAFFOLD_TEST_WEIGHTEDSCAFFOLD_H
