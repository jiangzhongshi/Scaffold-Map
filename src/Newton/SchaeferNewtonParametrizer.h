#ifndef SCHAEFER_NEWTON_PARAMETRIZER_H
#define SCHAEFER_NEWTON_PARAMETRIZER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <set>
#include <vector>

#include "autodiff.h"

typedef Eigen::VectorXd                     Gradient;
typedef Eigen::MatrixXd                     Hessian;
typedef DScalar2<double, Gradient, Hessian> DScalar;

class ScafData;

class SchaeferNewtonParametrizer {

public:
  // does precomputation if it was not already done
  SchaeferNewtonParametrizer(ScafData &_sd);

  void newton_iteration(const Eigen::MatrixXi &F,
                          Eigen::MatrixXd &uv);

  double evaluate_energy(const Eigen::MatrixXi &F, Eigen::MatrixXd &uv);

private:

  double compute_energy_gradient_hessian(const Eigen::MatrixXi &F,
                                           Eigen::MatrixXd &uv,
                                           Eigen::VectorXd &grad,
                                           Eigen::SparseMatrix<double> &hessian);

  void precompute(const Eigen::MatrixXi &F);

  DScalar compute_face_energy_left_part(const Eigen::MatrixXi &F,
                                          const Eigen::MatrixXd &uv,
                                          int f);

  DScalar compute_face_energy_right_part(const Eigen::MatrixXi &F,
                                           const Eigen::MatrixXd &uv,
                                           int f_idx);

  void finiteGradient(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd &x, Eigen::VectorXd &grad, int accuracy = 0);
  bool check_gradient(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& X,
       const Eigen::VectorXd& grad, int accuracy = 3);

  void get_gradient(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                  Eigen::VectorXd& uv, Eigen::VectorXd& grad);

  void finiteHessian(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
        const Eigen::VectorXd & x, Eigen::MatrixXd & hessian, int accuracy = 0);

  void finiteHessian_with_grad(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
            const Eigen::VectorXd& x, Eigen::MatrixXd & hessian, int accuracy);

  bool checkHessian(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
          const Eigen::VectorXd & x, const Eigen::MatrixXd& actual_hessian, int accuracy = 3);

  // cached computations
  bool has_precomputed;
  std::vector<double> m_cached_edges_1;
  std::vector<double> m_cached_edges_2;
  std::vector<double> m_cached_dot_prod;
  Eigen::VectorXd m_dblArea_orig;
  Eigen::VectorXd m_dbl_area_weight;

  ScafData& d_;
};

#endif // SCHAEFER_NEWTON_PARAMETRIZER_H
