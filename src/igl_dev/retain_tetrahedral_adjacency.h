//
// Created by Zhongshi Jiang on 3/31/17.
//

#ifndef SCAFFOLD_TEST_RETAIN_TETRAHEDRAL_ADJACENCY_H
#define SCAFFOLD_TEST_RETAIN_TETRAHEDRAL_ADJACENCY_H

#include <vector>
#include <set>
#include <Eigen/Core>
#include <igl/igl_inline.h>

IGL_INLINE void retain_tetrahedral_adjacency(
    const std::set<int>& delete_id, const std::set<int>& surround_id,
    const std::vector<Eigen::RowVector4i>& new_T,
    std::vector<Eigen::RowVector4i>& T,
    std::vector<Eigen::RowVector4i>& TT,
    std::vector<Eigen::RowVector4i>& TTif,
    std::vector<Eigen::Matrix<int, 4, 3>>& TTie);
#ifndef IGL_STATIC_LIBRARY
#endif
#endif //SCAFFOLD_TEST_RETAIN_TETRAHEDRAL_ADJACENCY_H
