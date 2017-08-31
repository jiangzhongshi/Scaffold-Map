//
// Created by Zhongshi Jiang on 10/9/16.
//

#ifndef SCAFFOLD_TEST_WEIGHTEDSCAFFOLD_H
#define SCAFFOLD_TEST_WEIGHTEDSCAFFOLD_H

#include "ScaffoldSolver.h"
#include "ScafData.h"
#include <igl/serialize.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>

class ScafData;

class ReWeightedARAP
{

public:
  ReWeightedARAP(ScafData &state) : d_(state)
  {};

  void pre_calc() ;

  [[deprecated]] void regenerate_scaffold() ;

  void adjust_scaf_weight(double) ;

  void solve_weighted_proxy(Eigen::MatrixXd &uv_new);

  double compute_energy(const Eigen::MatrixXd &V_new,
                                              bool whole = true) ;

  double perform_iteration(Eigen::MatrixXd &w_uv) ;
  double perform_iteration(Eigen::MatrixXd &w_uv,bool dummy) ;
  void change_scaffold_reference(const Eigen::MatrixXd &s_uv) ;
  void mesh_improve() ;
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
                                         const Eigen::SparseMatrix<double> &Dx,
                                         const Eigen::SparseMatrix<double> &Dy,
                                         Eigen::MatrixXd& Ji);
  static void compute_jacobians(const Eigen::MatrixXd &uv,
                                         const Eigen::SparseMatrix<double>& Dx,
                                         const Eigen::SparseMatrix<double> &Dy,
                                         const Eigen::SparseMatrix<double> &Dz,
                                         Eigen::MatrixXd& Ji);

  static double compute_energy_from_jacobians(const Eigen::MatrixXd &Ji,
                                               const Eigen::VectorXd &areas,
                                               ScafData::SLIM_ENERGY energy_type);

  void add_soft_constraints(Eigen::SparseMatrix<double> &L,
                            Eigen::VectorXd &rhs) const;
  double compute_soft_constraint_energy(const Eigen::MatrixXd &uv) const ;

  [[deprecated]] void update_weights_and_closest_rotations(Eigen::MatrixXd
                                                           &uv){};

  template <int dim>
  void update_weights_and_closest_rotations(
      const Eigen::MatrixXd& Ji,
      ScafData::SLIM_ENERGY energy_type,
      Eigen::MatrixXd& W,
      Eigen::MatrixXd& Ri);

  void solve_weighted_arap(Eigen::MatrixXd &uv);

 private: // build and solve linear system

  void build_linear_system(Eigen::SparseMatrix<double> &L,
                           Eigen::VectorXd &rhs) const;

  void build_surface_linear_system(Eigen::SparseMatrix<double> &L,
                                           Eigen::VectorXd &rhs) const;

  void build_scaffold_linear_system(Eigen::SparseMatrix<double> &L,
                                   Eigen::VectorXd &rhs) const;

  [[deprecated]] void buildAm(const Eigen::VectorXd &sqrt_M,
                 Eigen::SparseMatrix<double> &Am) const{};

  static void buildAm(const Eigen::VectorXd &sqrt_M,
              const Eigen::SparseMatrix<double> &Dx,
              const Eigen::SparseMatrix<double> &Dy,
              const Eigen::MatrixXd &W,
              Eigen::SparseMatrix<double> &Am) ;

  static void buildAm(const Eigen::VectorXd &sqrt_M,
               const Eigen::SparseMatrix<double> &Dx,
               const Eigen::SparseMatrix<double> &Dy,
               const Eigen::SparseMatrix<double> &Dz,
               const Eigen::MatrixXd &W,
               Eigen::SparseMatrix<double> &Am) ;

  [[deprecated]] void buildRhs(const Eigen::VectorXd &sqrt_M,
             const Eigen::SparseMatrix<double> &At,
             Eigen::VectorXd &frhs) const;

  // static computing f_rhs = \sqrt(M) b = \sqrt(M)*W*R
  static void buildRhs(const Eigen::VectorXd &sqrt_M,
                      const Eigen::MatrixXd &W,
                      const Eigen::MatrixXd& Ri,
                      Eigen::VectorXd &f_rhs) ;

 private: // variables
  ScafData &d_;
  Eigen::VectorXd M_m, M_s;
  Eigen::MatrixXd Ri_m, Ji_m, Ri_s, Ji_s;
  Eigen::MatrixXd W_m, W_s;

  Eigen::SparseMatrix<double> Dx_s, Dy_s, Dz_s;
  Eigen::SparseMatrix<double> Dx_m, Dy_m, Dz_m;

  int f_n, v_n, sf_n, sv_n, mf_n, mv_n;

 public:
  bool has_pre_calc = false;

  static void compute_surface_gradient_matrix(
      const Eigen::MatrixXd &V,
      const Eigen::MatrixXi &F,
      const Eigen::MatrixXd &F1,
      const Eigen::MatrixXd &F2,
      Eigen::SparseMatrix<double, 0, int> &D1,
      Eigen::SparseMatrix<double, 0, int> &D2);
 public:
   void after_mesh_improve();
  void adjust_frame(double,double);
  void compute_jacobians(const Eigen::MatrixXd &V_new) const;

  void enlarge_internal_reference(double);
};

#endif //SCAFFOLD_TEST_WEIGHTEDSCAFFOLD_H
