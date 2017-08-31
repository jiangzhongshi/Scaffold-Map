//
// Created by Zhongshi Jiang on 5/2/17.
//

#ifndef SCAFFOLD_TEST_TETRAHEDRAL_REFINEMENT_H
#define SCAFFOLD_TEST_TETRAHEDRAL_REFINEMENT_H

#include <vector>
#include <igl/igl_inline.h>
#include <functional>
#include <set>
#include <Eigen/Core>

namespace igl{ namespace dev {
bool tet_tuple_edge_contraction(int ti, int fi, int ei, bool along,
                                std::function<double(int, int,
                                                     int, int)> tet_quality,
                                std::function<bool(int)> vertex_editable,
                                std::vector<Eigen::RowVector4i> &T,
                                std::vector<Eigen::RowVector4i> &TT,
                                std::vector<Eigen::RowVector4i> &TTif,
                                std::vector<Eigen::Matrix<int, 4, 3>> &TTie,
                                std::vector<int> &new_tets_id);

bool laplacian_smart_smoothing(int ti, int fi, int ei, bool along,
                               std::function<double(int,
                                                    int,
                                                    int,
                                                    int)> tet_quality,
                               std::function<bool(int)> vertex_editable,
                               const std::vector<Eigen::RowVector4i> &T,
                               const std::vector<Eigen::RowVector4i> &TT,
                               const std::vector<Eigen::RowVector4i> &TTif,
                               const std::vector<Eigen::Matrix<int,
                                                               4,
                                                               3>> &TTie,
                               std::vector<Eigen::RowVector3d> &V,
                               std::vector<int> &new_tets_id);

bool tet_tuple_edge_split(int ti, int fi, int ei, bool ai,
                          std::function<double(int, int,
                                               int, int)> tet_quality,
                          std::vector<Eigen::RowVector3d> &V,
                          std::vector<Eigen::RowVector4i> &T,
                          std::vector<Eigen::RowVector4i> &TT,
                          std::vector<Eigen::RowVector4i> &TTif,
                          std::vector<Eigen::Matrix<int, 4, 3>> &TTie,
                          std::vector<int> &new_tets_id);
}
}
#endif //SCAFFOLD_TEST_TETRAHEDRAL_REFINEMENT_H
