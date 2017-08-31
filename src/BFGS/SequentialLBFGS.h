#ifndef SCAFFOLDED_SYMMETRIC_DIRICHLET_H
#define SCAFFOLDED_SYMMETRIC_DIRICHLET_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <set>
#include <vector>

#include "../ScaffoldSolver.h"
#include "../ScafData.h"


class SequentialLBFGS: public ScaffoldSolver
{

public:
  // does precomputation if it was not already done
  SequentialLBFGS(ScafData &data);
  void pre_calc() override;

  void parametrize(Eigen::MatrixXd &uv);

  double parametrize_LBFGS(Eigen::MatrixXd &uv, int max_iter, int dummy);
  double parametrize_LBFGS(Eigen::MatrixXd &uv, int max_iter);

  double one_step(Eigen::MatrixXd &uv);

  double perform_iteration(Eigen::MatrixXd &w_uv) override ;

  // Modify m_cached_edges_1/2 and _dot_prod for compute energy.
  double compute_energy(const Eigen::MatrixXd &uv, bool whole) override ;

  // m_cot_entries.
  void compute_negative_gradient(
    const Eigen::MatrixXd &uv, Eigen::MatrixXd &neg_grad);

  double cur_energy;
  double cur_riemann_energy;
  bool has_converged;

  // cached computations
  std::vector<double> m_cached_l_energy_per_face;
  std::vector<double> m_cached_r_energy_per_face;


  // scaffold interface.
  void change_scaffold_reference(const Eigen::MatrixXd& s_uv) override;
  void regenerate_scaffold() override;
  void adjust_scaf_weight(double) override;
  void mesh_improve() override;
private:

  double single_gradient_descent(Eigen::MatrixXd &uv);

  double LineSearch_michael_armijo_imp(Eigen::MatrixXd &uv,
                                         const Eigen::MatrixXd &d,
                                         double max_step_size);

  // https://github.com/PatWie/CppNumericalSolvers/
  double LineSearch_patwie_armijo_imp(Eigen::MatrixXd &uv, const Eigen::MatrixXd &grad,
                                        const Eigen::MatrixXd &d, double max_step_size);

  double compute_energy(const Eigen::VectorXd &uv, bool whole);


void compute_negative_gradient(
    const Eigen::VectorXd &uv_vec, Eigen::VectorXd &neg_grad_vec);

  double compute_max_step_from_singularities(
      const Eigen::MatrixXd &uv, Eigen::MatrixXd &grad);

  double get_min_pos_root(const Eigen::MatrixXd &uv,
                            Eigen::MatrixXd &d,
                            int f);

  double compute_face_energy_left_part(
      const Eigen::MatrixXd &uv, int f_idx);

  double compute_face_energy_right_part(
      const Eigen::MatrixXd &uv, int f_idx, double orig_t_dbl_area);

  double compute_face_energy_part(
      const Eigen::MatrixXd &uv, bool is_left_grad);

  bool check_grad(const Eigen::MatrixXd &uv, int v_idx,
                    Eigen::RowVector2d grad,
                    bool is_left_grad);


  void zero_out_const_vertices_search_direction(Eigen::MatrixXd& d);

  void update_results(double new_energy);

  long m_iter;
  bool has_pre_calc;
  ScafData& m_data;

  long v_n, f_n, mv_n, mf_n;
  // cached computations
  std::vector<double> m_cached_edges_1;
  std::vector<double> m_cached_edges_2;
  std::vector<double> m_cached_dot_prod;
  Eigen::MatrixXd m_cot_entries;
  Eigen::MatrixXd m_m_cot_entries; // member-var_mesh_cot_entries.
  Eigen::VectorXd m_dbl_area;
  Eigen::VectorXd m_sq_dbl_area;

  std::vector<int> m_b; // constrained vertices
  int m_arap_iter;
};

#endif // SYMMETRIC_DIRICHLET_H
