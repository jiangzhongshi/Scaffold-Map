//
// Created by Zhongshi Jiang on 1/24/17.
//

#ifndef SCAFFOLD_TEST_SCAFFOLDSOLVER_H
#define SCAFFOLD_TEST_SCAFFOLDSOLVER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

class ScaffoldSolver
{
public:
  virtual void pre_calc() = 0;
  virtual double compute_energy(const Eigen::MatrixXd&, bool) = 0;
  virtual void change_scaffold_reference(const Eigen::MatrixXd &) = 0;
  virtual void regenerate_scaffold() = 0;
  virtual void adjust_scaf_weight(double) = 0;
  virtual double perform_iteration(Eigen::MatrixXd &w_uv) = 0;
  virtual void mesh_improve()=0;
};


#endif //SCAFFOLD_TEST_SCAFFOLDSOLVER_H
