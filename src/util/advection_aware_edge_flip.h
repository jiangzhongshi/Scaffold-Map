//
// Created by Zhongshi Jiang on 3/10/17.
//

#ifndef SCAFFOLD_TEST_ADVECTION_AWARE_EDGE_FLIP_H
#define SCAFFOLD_TEST_ADVECTION_AWARE_EDGE_FLIP_H
void simple_edge_flip(Eigen::MatrixXi &F, Eigen::MatrixXi &FF,
                      Eigen::MatrixXi &FFi, int f0,
                      int e0) ;

double advection_aware_edge_flip(const Eigen::MatrixXd &V,
                                 const Eigen::MatrixXd &d,
                                 double alpha_max,
                                 Eigen::MatrixXi &F,
                                 Eigen::MatrixXi &FF,
                                 Eigen::MatrixXi &FFi);

double advection_aware_edge_flip(const Eigen::MatrixXd &V,
                                 const Eigen::MatrixXd &d,
                                 Eigen::MatrixXi &F);

#endif //SCAFFOLD_TEST_ADVECTION_AWARE_EDGE_FLIP_H
