//
// Created by Zhongshi Jiang on 9/22/17.
//

#include "scaf.h"

#include <igl/doublearea.h>
#include <iostream>
#include <igl/volume.h>
#include <igl/boundary_facets.h>
#include <igl/Timer.h>
#include <igl/massmatrix.h>
#include <igl/triangle/triangulate.h>
#include <igl/cat.h>
#include <igl/boundary_loop.h>
#include <igl/edge_flaps.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/flipped_triangles.h>
#include <igl/PI.h>

#include "igl/arap.h"
#include "igl/cat.h"
#include "igl/doublearea.h"
#include "igl/grad.h"
#include "igl/local_basis.h"
#include "igl/per_face_normals.h"
#include "igl/slice_into.h"
#include "igl/serialize.h"

#include <igl/ARAPEnergyType.h>
#include <igl/covariance_scatter_matrix.h>

#include <igl/flip_avoiding_line_search.h>
#include <igl/boundary_facets.h>
#include <igl/unique.h>
#include <igl/slim.h>
#include <igl/grad.h>
#include <igl/is_symmetric.h>
#include <igl/polar_svd.h>
#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/edge_lengths.h>
#include <igl/local_basis.h>
#include <igl/readOBJ.h>
#include <igl/repdiag.h>
#include <igl/vector_area_matrix.h>
#include <iostream>
#include <igl/slice.h>
#include <igl/colon.h>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <map>
#include <set>
#include <vector>
#include <igl/Timer.h>
#include <igl/edge_flaps.h>
#include <igl/arap_linear_block.h>

namespace igl
{
namespace scaf
{
void update_scaffold(igl::SCAFData &s)
{
  s.mv_num = s.m_V.rows();
  s.mf_num = s.m_T.rows();

  s.v_num = s.w_uv.rows();
  s.sf_num = s.s_T.rows();

  s.sv_num = s.v_num - s.mv_num;
  s.f_num = s.sf_num + s.mf_num;

  s.s_M = Eigen::VectorXd::Constant(s.sf_num, s.scaffold_factor);
}

void add_soft_constraints(igl::SCAFData &s, const Eigen::VectorXi &b,
                          const Eigen::MatrixXd &bc)
{
  assert(b.rows() == bc.rows() && "Constraint input incompatible");
  for (int i = 0; i < b.rows(); i++)
    s.soft_cons[b(i)] = bc.row(i);
}

void add_soft_constraints(igl::SCAFData &s, int b, const Eigen::RowVectorXd &bc)
{
  s.soft_cons[b] = bc;
}

void mesh_improve(igl::SCAFData &s)
{
  using namespace Eigen;
  MatrixXd m_uv = s.w_uv.topRows(s.mv_num);
  MatrixXd V_bnd;
  V_bnd.resize(s.internal_bnd.size(), 2);
  for (int i = 0; i < s.internal_bnd.size(); i++) // redoing step 1.
  {
    V_bnd.row(i) = m_uv.row(s.internal_bnd(i));
  }

  if (s.rect_frame_V.size() == 0)
  {
    Matrix2d ob; // = rect_corners;
    {
      VectorXd uv_max = m_uv.colwise().maxCoeff();
      VectorXd uv_min = m_uv.colwise().minCoeff();
      VectorXd uv_mid = (uv_max + uv_min) / 2.;

      Eigen::Array2d scaf_range(3, 3);
      ob.row(0) = uv_mid.array() + scaf_range * ((uv_min - uv_mid).array());
      ob.row(1) = uv_mid.array() + scaf_range * ((uv_max - uv_mid).array());
    }
    Vector2d rect_len;
    rect_len << ob(1, 0) - ob(0, 0), ob(1, 1) - ob(0, 1);
    int frame_points = 5;

    s.rect_frame_V.resize(4 * frame_points, 2);
    for (int i = 0; i < frame_points; i++)
    {
      // 0,0;0,1
      s.rect_frame_V.row(i) << ob(0, 0), ob(0, 1) + i * rect_len(1) / frame_points;
      // 0,0;1,1
      s.rect_frame_V.row(i + frame_points)
          << ob(0, 0) + i * rect_len(0) / frame_points,
          ob(1, 1);
      // 1,0;1,1
      s.rect_frame_V.row(i + 2 * frame_points) << ob(1, 0), ob(1, 1) - i * rect_len(1) / frame_points;
      // 1,0;0,1
      s.rect_frame_V.row(i + 3 * frame_points)
          << ob(1, 0) - i * rect_len(0) / frame_points,
          ob(0, 1);
      // 0,0;0,1
    }
    s.frame_ids = Eigen::VectorXi::LinSpaced(s.rect_frame_V.rows(), s.mv_num,
                                             s.mv_num +
                                                 s.rect_frame_V.rows());
  }

  // Concatenate Vert and Edge
  MatrixXd V;
  MatrixXi E;
  igl::cat(1, V_bnd, s.rect_frame_V, V);
  E.resize(V.rows(), 2);
  for (int i = 0; i < E.rows(); i++)
    E.row(i) << i, i + 1;
  int acc_bs = 0;
  for (auto bs : s.bnd_sizes)
  {
    E(acc_bs + bs - 1, 1) = acc_bs;
    acc_bs += bs;
  }
  E(V.rows() - 1, 1) = acc_bs;
  assert(acc_bs == s.internal_bnd.size());

  MatrixXd H = MatrixXd::Zero(s.component_sizes.size(), 2);
  {
    int hole_f = 0;
    int hole_i = 0;
    for (auto cs : s.component_sizes)
    {
      for (int i = 0; i < 3; i++)
        H.row(hole_i) += m_uv.row(s.m_T(hole_f, i)); // redoing step 2
      hole_f += cs;
      hole_i++;
    }
  }
  H /= 3.;

  MatrixXd uv2;
  igl::triangle::triangulate(V, E, H, "qYYQ", uv2, s.s_T);
  auto bnd_n = s.internal_bnd.size();

  for (auto i = 0; i < s.s_T.rows(); i++)
    for (auto j = 0; j < s.s_T.cols(); j++)
    {
      auto &x = s.s_T(i, j);
      if (x < bnd_n)
        x = s.internal_bnd(x);
      else
        x += m_uv.rows() - bnd_n;
    }

  igl::cat(1, s.m_T, s.s_T, s.w_T);
  s.w_uv.conservativeResize(m_uv.rows() - bnd_n + uv2.rows(), 2);
  s.w_uv.bottomRows(uv2.rows() - bnd_n) = uv2.bottomRows(-bnd_n + uv2.rows());

  update_scaffold(s);
}

void automatic_expand_frame(igl::SCAFData &s, double min2, double max3)
{
  // right top
  // left down
  using namespace Eigen;
  MatrixXd m_uv = s.w_uv.topRows(s.mv_num);
  MatrixXd frame(2, s.dim), bbox(2, s.dim);
  frame << s.w_uv.colwise().maxCoeff(), s.w_uv.colwise().minCoeff();
  bbox << m_uv.colwise().maxCoeff(), m_uv.colwise().minCoeff();
  RowVector2d center = bbox.colwise().mean();
  /*
        bbox.row(0) -= center;
        bbox.row(1) -= center;
        frame.row(0) -= center;
        frame.row(1) -= center;
      */
  struct line_func
  {
    double a, b;

    double operator()(double y)
    {
      return a * y + b;
    };
  };

  auto linear_stretch = [](double s0,
                           double t0,
                           double s1,
                           double t1) { // source0, target0, source1, target1
    Matrix2d S;
    S << s0, 1, s1, 1;
    Vector2d t;
    t << t0, t1;
    Vector2d coef = S.colPivHouseholderQr().solve(t);
    return line_func{coef[0], coef[1]};
  };

  double new_frame;
  double center_coord;
  for (auto d = 0; d < s.dim; d++)
  {
    center_coord = center(d);

    if (frame(0, d) - center_coord < min2 * (bbox(0, d) - center_coord))
    {
      new_frame = max3 * (bbox(0, d) - center_coord) + center_coord;
      auto expand = linear_stretch(bbox(0, d), bbox(0, d),
                                   frame(0, d), new_frame);
      for (auto v = s.mv_num; v < s.v_num; v++)
      {
        if (s.w_uv(v, d) > bbox(0, d))
          s.w_uv(v, d) = expand(s.w_uv(v, d));
      }
    }

    if (frame(1, d) - center_coord > min2 * (bbox(1, d) - center_coord))
    {
      new_frame = max3 * (bbox(1, d) - center_coord) + center_coord;
      auto expand = linear_stretch(bbox(1, d), bbox(1, d),
                                   frame(1, d), new_frame);
      for (auto v = s.mv_num; v < s.v_num; v++)
      {
        if (s.w_uv(v, d) < bbox(1, d))
          s.w_uv(v, d) = expand(s.w_uv(v, d));
      }
    }
  }
}

void add_new_patch(igl::SCAFData &s, const Eigen::MatrixXd &V_in,
                   const Eigen::MatrixXi &F_ref,
                   const Eigen::RowVectorXd &center)
{
  using namespace std;
  using namespace Eigen;

  VectorXd M;
  igl::doublearea(V_in, F_ref, M);

  Eigen::MatrixXd V_ref = V_in; // / sqrt(M.sum()/2/igl::PI);
  // M /= M.sum()/igl::PI;
  Eigen::MatrixXd uv_init;
  Eigen::VectorXi bnd;
  Eigen::MatrixXd bnd_uv;

  std::vector<std::vector<int>> all_bnds;
  igl::boundary_loop(F_ref, all_bnds);
  int num_holes = all_bnds.size() - 1;

  std::sort(all_bnds.begin(), all_bnds.end(), [](auto &a, auto &b) {
    return a.size() > b.size();
  });

  bnd = Map<Eigen::VectorXi>(all_bnds[0].data(),
                             all_bnds[0].size());

  igl::map_vertices_to_circle(V_ref, bnd, bnd_uv);
  bnd_uv *= sqrt(M.sum() / (2 * igl::PI));
  bnd_uv.rowwise() += center;
  s.mesh_measure += M.sum() / 2;
  std::cout << "Mesh Measure" << M.sum() / 2 << std::endl;

  if (num_holes == 0)
  {

    if (bnd.rows() == V_ref.rows())
    {
      std::cout << "All vert on boundary" << std::endl;
      uv_init.resize(V_ref.rows(), 2);
      for (int i = 0; i < bnd.rows(); i++)
      {
        uv_init.row(bnd(i)) = bnd_uv.row(i);
      }
    }
    else
    {
      igl::harmonic(V_ref, F_ref, bnd, bnd_uv, 1, uv_init);

      if (igl::flipped_triangles(uv_init, F_ref).size() != 0)
      {
        std::cout << "Using Uniform Laplacian" << std::endl;
        igl::harmonic(F_ref, bnd, bnd_uv, 1,
                      uv_init); // use uniform laplacian
      }
    }
  }
  else
  {
    auto &F = F_ref;
    auto &V = V_in;
    auto &primary_bnd = bnd;
    // fill holes
    int n_filled_faces = 0;
    int real_F_num = F.rows();
    for (int i = 0; i < num_holes; i++)
    {
      n_filled_faces += all_bnds[i + 1].size();
    }
    MatrixXi F_filled(n_filled_faces + real_F_num, 3);
    F_filled.topRows(real_F_num) = F;

    int new_vert_id = V.rows();
    int new_face_id = real_F_num;

    for (int i = 0; i < num_holes; i++)
    {
      int cur_bnd_size = all_bnds[i + 1].size();
      auto it = all_bnds[i + 1].begin();
      auto back = all_bnds[i + 1].end() - 1;
      F_filled.row(new_face_id++) << *it, *back, new_vert_id;
      while (it != back)
      {
        F_filled.row(new_face_id++)
            << *(it + 1),
            *(it), new_vert_id;
        it++;
      }
      new_vert_id++;
    }
    assert(new_face_id == F_filled.rows());
    assert(new_vert_id == V.rows() + num_holes);

    igl::harmonic(F_filled, primary_bnd, bnd_uv, 1, uv_init);
    uv_init.conservativeResize(V.rows(), 2);
    if (igl::flipped_triangles(uv_init, F_ref).size() != 0)
    {
      std::cout << "Wrong Choice of Outer bnd:" << std::endl;
      //      assert(false&&"Wrong Choice of outer bnd?");
    }
  }

  s.component_sizes.push_back(F_ref.rows());

  MatrixXd m_uv = s.w_uv.topRows(s.mv_num);
  igl::cat(1, m_uv, uv_init, s.w_uv);
  //  s.mv_num =  s.w_uv.rows();

  s.m_M.conservativeResize(s.mf_num + M.size());
  s.m_M.bottomRows(M.size()) = M / 2;

  //  internal_bnd.conservativeResize(internal_bnd.size()+ bnd.size());
  //  internal_bnd.bottomRows(bnd.size()) = bnd.array() + s.mv_num;
  //  bnd_sizes.push_back(bnd.size());

  for (auto cur_bnd : all_bnds)
  {
    s.internal_bnd.conservativeResize(s.internal_bnd.size() + cur_bnd.size());
    s.internal_bnd.bottomRows(cur_bnd.size()) =
        Map<ArrayXi>(cur_bnd.data(), cur_bnd.size()) + s.mv_num;
    s.bnd_sizes.push_back(cur_bnd.size());
  }

  s.m_T.conservativeResize(s.mf_num + F_ref.rows(), 3);
  s.m_T.bottomRows(F_ref.rows()) = F_ref.array() + s.mv_num;
  s.mf_num += F_ref.rows();

  s.m_V.conservativeResize(s.mv_num + V_ref.rows(), 3);
  s.m_V.bottomRows(V_ref.rows()) = V_ref;
  s.mv_num += V_ref.rows();

  s.rect_frame_V = MatrixXd();

  mesh_improve(s);
}

// functions from ReweightedARAP, a static function
void compute_surface_gradient_matrix(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &F1,
    const Eigen::MatrixXd &F2,
    Eigen::SparseMatrix<double> &D1,
    Eigen::SparseMatrix<double> &D2)
{
  Eigen::SparseMatrix<double> G;
  igl::grad(V, F, G);
  Eigen::SparseMatrix<double> Dx = G.block(0, 0, F.rows(), V.rows());
  Eigen::SparseMatrix<double> Dy = G.block(F.rows(), 0, F.rows(), V.rows());
  Eigen::SparseMatrix<double> Dz = G.block(2 * F.rows(), 0, F.rows(), V.rows());

  D1 = F1.col(0).asDiagonal() * Dx + F1.col(1).asDiagonal() * Dy +
       F1.col(2).asDiagonal() * Dz;
  D2 = F2.col(0).asDiagonal() * Dx + F2.col(1).asDiagonal() * Dy +
       F2.col(2).asDiagonal() * Dz;
}

template <typename DerivedV, typename DerivedF>
inline void adjusted_grad(const Eigen::MatrixBase<DerivedV> &V,
                          const Eigen::MatrixBase<DerivedF> &F,
                          Eigen::SparseMatrix<typename DerivedV::Scalar> &G,
                          double eps)
{
  Eigen::Matrix<typename DerivedV::Scalar, Eigen::Dynamic, 3>
      eperp21(F.rows(), 3), eperp13(F.rows(), 3);
  int fixed = 0;
  for (int i = 0; i < F.rows(); ++i)
  {
    // renaming indices of vertices of triangles for convenience
    int i1 = F(i, 0);
    int i2 = F(i, 1);
    int i3 = F(i, 2);

    // #F x 3 matrices of triangle edge vectors, named after opposite vertices
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3> v32 = V.row(i3) - V.row(i2);
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3> v13 = V.row(i1) - V.row(i3);
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3> v21 = V.row(i2) - V.row(i1);
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3> n = v32.cross(v13);
    // area of parallelogram is twice area of triangle
    // area of parallelogram is || v1 x v2 ||
    // This does correct l2 norm of rows, so that it contains #F list of twice
    // triangle areas
    double dblA = std::sqrt(n.dot(n));
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3> u;
    if (dblA > eps)
    {
      // now normalize normals to get unit normals
      u = n / dblA;
    }
    else
    {
      // Abstract equilateral triangle v1=(0,0), v2=(h,0), v3=(h/2, (sqrt(3)/2)*h)
      fixed++;
      // get h (by the area of the triangle)
      dblA = eps;
      double h = sqrt((dblA) / sin(
                                   M_PI / 3.0)); // (h^2*sin(60))/2. = Area => h = sqrt(2*Area/sin_60)

      Eigen::Vector3d v1, v2, v3;
      v1 << 0, 0, 0;
      v2 << h, 0, 0;
      v3 << h / 2., (sqrt(3) / 2.) * h, 0;

      // now fix v32,v13,v21 and the normal
      v32 = v3 - v2;
      v13 = v1 - v3;
      v21 = v2 - v1;
      n = v32.cross(v13);
    }

    // rotate each vector 90 degrees around normal
    double norm21 = std::sqrt(v21.dot(v21));
    double norm13 = std::sqrt(v13.dot(v13));
    eperp21.row(i) = u.cross(v21);
    eperp21.row(i) =
        eperp21.row(i) / std::sqrt(eperp21.row(i).dot(eperp21.row(i)));
    eperp21.row(i) *= norm21 / dblA;
    eperp13.row(i) = u.cross(v13);
    eperp13.row(i) =
        eperp13.row(i) / std::sqrt(eperp13.row(i).dot(eperp13.row(i)));
    eperp13.row(i) *= norm13 / dblA;
  }

  std::vector<int> rs;
  rs.reserve(F.rows() * 4 * 3);
  std::vector<int> cs;
  cs.reserve(F.rows() * 4 * 3);
  std::vector<double> vs;
  vs.reserve(F.rows() * 4 * 3);

  // row indices
  for (int r = 0; r < 3; r++)
  {
    for (int j = 0; j < 4; j++)
    {
      for (int i = r * F.rows(); i < (r + 1) * F.rows(); i++)
        rs.push_back(i);
    }
  }

  // column indices
  for (int r = 0; r < 3; r++)
  {
    for (int i = 0; i < F.rows(); i++)
      cs.push_back(F(i, 1));
    for (int i = 0; i < F.rows(); i++)
      cs.push_back(F(i, 0));
    for (int i = 0; i < F.rows(); i++)
      cs.push_back(F(i, 2));
    for (int i = 0; i < F.rows(); i++)
      cs.push_back(F(i, 0));
  }

  // values
  for (int i = 0; i < F.rows(); i++)
    vs.push_back(eperp13(i, 0));
  for (int i = 0; i < F.rows(); i++)
    vs.push_back(-eperp13(i, 0));
  for (int i = 0; i < F.rows(); i++)
    vs.push_back(eperp21(i, 0));
  for (int i = 0; i < F.rows(); i++)
    vs.push_back(-eperp21(i, 0));
  for (int i = 0; i < F.rows(); i++)
    vs.push_back(eperp13(i, 1));
  for (int i = 0; i < F.rows(); i++)
    vs.push_back(-eperp13(i, 1));
  for (int i = 0; i < F.rows(); i++)
    vs.push_back(eperp21(i, 1));
  for (int i = 0; i < F.rows(); i++)
    vs.push_back(-eperp21(i, 1));
  for (int i = 0; i < F.rows(); i++)
    vs.push_back(eperp13(i, 2));
  for (int i = 0; i < F.rows(); i++)
    vs.push_back(-eperp13(i, 2));
  for (int i = 0; i < F.rows(); i++)
    vs.push_back(eperp21(i, 2));
  for (int i = 0; i < F.rows(); i++)
    vs.push_back(-eperp21(i, 2));

  // create sparse gradient operator matrix
  G.resize(3 * F.rows(), V.rows());
  std::vector<Eigen::Triplet<typename DerivedV::Scalar>> triplets;
  for (int i = 0; i < (int)vs.size(); ++i)
  {
    triplets.push_back(Eigen::Triplet<typename DerivedV::Scalar>(rs[i],
                                                                 cs[i],
                                                                 vs[i]));
  }
  G.setFromTriplets(triplets.begin(), triplets.end());
  //  std::cout<<"Adjusted"<<fixed<<std::endl;
};

void simplified_covariance_scatter_matrix(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    Eigen::SparseMatrix<double> &Dx, Eigen::SparseMatrix<double> &Dy,
    Eigen::SparseMatrix<double> &Dz)
{
  using namespace Eigen;
  auto energy = igl::ARAP_ENERGY_TYPE_SPOKES_AND_RIMS;
  SparseMatrix<double> Kx, Ky, Kz;
  igl::arap_linear_block(V, F, 0, energy, Kx);
  igl::arap_linear_block(V, F, 1, energy, Ky);
  igl::arap_linear_block(V, F, 2, energy, Kz);
  Dx = Kx.transpose();
  Dy = Ky.transpose();
  Dz = Kz.transpose();
}

void compute_scaffold_gradient_matrix(SCAFData &d_,
                                      Eigen::SparseMatrix<double> &D1, Eigen::SparseMatrix<double> &D2)
{
  using namespace Eigen;
  Eigen::SparseMatrix<double> G;
  MatrixXi F_s = d_.s_T;
  int vn = d_.v_num;
  MatrixXd V = MatrixXd::Zero(vn, 3);
  V.leftCols(2) = d_.w_uv;
  //  std::cout<<"Avg Mesh Area"<<d_.mesh_measure/d_.mv_num<<std::endl;

  double min_bnd_edge_len = INFINITY;
  int acc_bnd = 0;
  for (int i = 0; i < d_.bnd_sizes.size(); i++)
  {
    int current_size = d_.bnd_sizes[i];

    for (int e = acc_bnd; e < acc_bnd + current_size - 1; e++)
    {
      min_bnd_edge_len = std::min(min_bnd_edge_len,
                                  (d_.w_uv.row(d_.internal_bnd(e)) -
                                   d_.w_uv.row(d_.internal_bnd(e + 1)))
                                      .squaredNorm());
    }
    min_bnd_edge_len = std::min(min_bnd_edge_len,
                                (d_.w_uv.row(d_.internal_bnd(acc_bnd)) -
                                 d_.w_uv.row(d_.internal_bnd(acc_bnd + current_size -
                                                             1)))
                                    .squaredNorm());
    acc_bnd += current_size;
  }

  //  std::cout<<"MinBndEdge"<<min_bnd_edge_len<<std::endl;
  double area_threshold = min_bnd_edge_len / 4.0;

  adjusted_grad(V, F_s, G, area_threshold);
  Eigen::SparseMatrix<double> Dx = G.block(0, 0, F_s.rows(), vn);
  Eigen::SparseMatrix<double> Dy = G.block(F_s.rows(), 0, F_s.rows(), vn);
  Eigen::SparseMatrix<double> Dz = G.block(2 * F_s.rows(), 0, F_s.rows(), vn);

  MatrixXd F1, F2, F3;
  igl::local_basis(V, F_s, F1, F2, F3);
  D1 = F1.col(0).asDiagonal() * Dx + F1.col(1).asDiagonal() * Dy +
       F1.col(2).asDiagonal() * Dz;
  D2 = F2.col(0).asDiagonal() * Dx + F2.col(1).asDiagonal() * Dy +
       F2.col(2).asDiagonal() * Dz;
}

void compute_jacobians(SCAFData &d_, bool whole)
{
  auto comp_J2 = [](const Eigen::MatrixXd &uv,
                    const Eigen::SparseMatrix<double> &Dx,
                    const Eigen::SparseMatrix<double> &Dy,
                    Eigen::MatrixXd &Ji) {
    // Ji=[D1*u,D2*u,D1*v,D2*v];
    Ji.resize(Dx.rows(), 4);
    Ji.col(0) = Dx * uv.col(0);
    Ji.col(1) = Dy * uv.col(0);
    Ji.col(2) = Dx * uv.col(1);
    Ji.col(3) = Dy * uv.col(1);
  };
  auto comp_J3 = [](const Eigen::MatrixXd &uv,
                    const Eigen::SparseMatrix<double> &Dx,
                    const Eigen::SparseMatrix<double> &Dy,
                    const Eigen::SparseMatrix<double> &Dz,
                    Eigen::MatrixXd &Ji) {
    // Ji=[D1*u,D2*u,D3*u, D1*v,D2*v, D3*v, D1*w,D2*w,D3*w];
    Ji.resize(Dx.rows(), 9);
    Ji.col(0) = Dx * uv.col(0);
    Ji.col(1) = Dy * uv.col(0);
    Ji.col(2) = Dz * uv.col(0);
    Ji.col(3) = Dx * uv.col(1);
    Ji.col(4) = Dy * uv.col(1);
    Ji.col(5) = Dz * uv.col(1);
    Ji.col(6) = Dx * uv.col(2);
    Ji.col(7) = Dy * uv.col(2);
    Ji.col(8) = Dz * uv.col(2);
  };
  auto V_new = d_.w_uv;
  Eigen::MatrixXd m_V_new = V_new.topRows(d_.mv_num);
  if (d_.dim == 2)
  {
    comp_J2(m_V_new, d_.Dx_m, d_.Dy_m, d_.Ji_m);
    if (whole)
      comp_J2(V_new, d_.Dx_s, d_.Dy_s, d_.Ji_s);
  }
  else
  {
    comp_J3(m_V_new, d_.Dx_m, d_.Dy_m, d_.Dz_m, d_.Ji_m);
    if (whole)
      comp_J3(V_new, d_.Dx_s, d_.Dy_s, d_.Dz_s, d_.Ji_s);
  }
}

double compute_energy_from_jacobians(const Eigen::MatrixXd &Ji,
                                   const Eigen::VectorXd &areas,
                                   igl::SLIMData::SLIM_ENERGY energy_type)
{
  double energy = 0;
  int dim = Ji.cols() == 4 ? 2 : 3;
  if (dim == 2)
  {
    Eigen::Matrix<double, 2, 2> ji;
    for (int i = 0; i < Ji.rows(); i++)
    {
      ji(0, 0) = Ji(i, 0);
      ji(0, 1) = Ji(i, 1);
      ji(1, 0) = Ji(i, 2);
      ji(1, 1) = Ji(i, 3);

      typedef Eigen::Matrix<double, 2, 2> Mat2;
      typedef Eigen::Matrix<double, 2, 1> Vec2;
      Mat2 ri, ti, ui, vi;
      Vec2 sing;
      igl::polar_svd(ji, ri, ti, ui, sing, vi);
      double s1 = sing(0);
      double s2 = sing(1);

      switch (energy_type)
      {
      case igl::SLIMData::ARAP:
      {
        energy += areas(i) * (pow(s1 - 1, 2) + pow(s2 - 1, 2));
        break;
      }
      case igl::SLIMData::SYMMETRIC_DIRICHLET:
      {
        energy +=
            areas(i) * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2));
        break;
      }
      case igl::SLIMData::LOG_ARAP:
      {
        energy += areas(i) * (pow(log(s1), 2) + pow(log(s2), 2));
        break;
      }
      case igl::SLIMData::CONFORMAL:
      {
        energy += areas(i) * ((pow(s1, 2) + pow(s2, 2)) / (2 * s1 * s2));
        break;
      }
      default:
        break;
      }
    }
  }
  else
  {
    Eigen::Matrix<double, 3, 3> ji;
    for (int i = 0; i < Ji.rows(); i++)
    {
      ji(0, 0) = Ji(i, 0);
      ji(0, 1) = Ji(i, 1);
      ji(0, 2) = Ji(i, 2);
      ji(1, 0) = Ji(i, 3);
      ji(1, 1) = Ji(i, 4);
      ji(1, 2) = Ji(i, 5);
      ji(2, 0) = Ji(i, 6);
      ji(2, 1) = Ji(i, 7);
      ji(2, 2) = Ji(i, 8);

      typedef Eigen::Matrix<double, 3, 3> Mat3;
      typedef Eigen::Matrix<double, 3, 1> Vec3;
      Mat3 ri, ti, ui, vi;
      Vec3 sing;
      igl::polar_svd(ji, ri, ti, ui, sing, vi);
      double s1 = sing(0);
      double s2 = sing(1);
      double s3 = sing(2);

      switch (energy_type)
      {
      case igl::SLIMData::ARAP:
      {
        energy += areas(i) * (pow(s1 - 1, 2) + pow(s2 - 1, 2) + pow(s3 - 1, 2));
        break;
      }
      case igl::SLIMData::SYMMETRIC_DIRICHLET:
      {
        energy += areas(i) * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2) + pow(s3, 2) + pow(s3, -2));
        break;
      }
      case igl::SLIMData::LOG_ARAP:
      {
        energy += areas(i) * (pow(log(s1), 2) + pow(log(std::abs(s2)), 2) + pow(log(std::abs(s3)), 2));
        break;
      }
      case igl::SLIMData::CONFORMAL:
      {
        energy += areas(i) * ((pow(s1, 2) + pow(s2, 2) + pow(s3, 2)) / (3 * pow(s1 * s2 * s3, 2. / 3.)));
        break;
      }
      case igl::SLIMData::EXP_CONFORMAL:
      {
        energy += areas(i) * exp((pow(s1, 2) + pow(s2, 2) + pow(s3, 2)) / (3 * pow(s1 * s2 * s3, 2. / 3.)));
        break;
      }
      default:
        assert(false);
      }
    }
  }
  return energy;
}

double compute_soft_constraint_energy(const SCAFData& d_) 
{
  double e = 0;
  for (auto const &x:d_.soft_cons)
    e += d_.soft_const_p * (x.second - d_.w_uv.row(x.first)).squaredNorm();

  return e;
}

double compute_energy(SCAFData &d_, bool whole)
{
  compute_jacobians(d_, whole);
  double energy = compute_energy_from_jacobians(d_.Ji_m, d_.m_M, d_.slim_energy);

  if (whole) 
   energy += compute_energy_from_jacobians(d_.Ji_s, d_.s_M, d_.scaf_energy);
  energy += compute_soft_constraint_energy(d_);
  return energy;
}

void adjust_scaf_weight(SCAFData &d_, double new_weight) 
{
  d_.scaffold_factor = new_weight;
  update_scaffold(d_);
}
/*
void perform_iteration(SCAFData &d_) 
{
  auto & w_uv = d_.w_uv;
  Eigen::MatrixXd V_out = w_uv;
  solve_weighted_proxy(V_out);
  auto whole_E =
      [&d_](Eigen::MatrixXd &uv) { return compute_energy(d_, uv); };

  igl::Timer timer;
  timer.start();
  Eigen::MatrixXi w_T;
  if(d_.m_T.cols() == d_.s_T.cols())
    igl::cat(1, d_.m_T, d_.s_T, w_T);
  else
    w_T = d_.s_T;
  double energy= igl::flip_avoiding_line_search(w_T, w_uv, V_out,
                                        whole_E, -1) / d_.mesh_measure;
  return energy;
}*/

}
}

IGL_INLINE void igl::scaf_precompute(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &V_init,
    igl::SCAFData &data,
    double soft_p)
{
  igl::scaf::add_new_patch(data, V, F, Eigen::RowVector2d(0, 0));
  data.soft_const_p = soft_p;

  using namespace Eigen;

  auto &d_ = data;
  auto &Dx_m = data.Dx_m;
  auto &Dy_m = data.Dy_m;
  auto &Dz_m = data.Dz_m;
  auto &Dx_s = data.Dx_s;
  auto &Dy_s = data.Dy_s;
  auto &Dz_s = data.Dz_s;
  auto &Ri_m = data.Ri_m;
  auto &Ji_m = data.Ji_m;
  auto &Ri_s = data.Ri_s;
  auto &Ji_s = data.Ji_s;
  auto &W_m = data.W_m;
  auto &W_s = data.W_s;

  if (!data.has_pre_calc)
  {
    int mv_n = d_.mv_num;
    int mf_n = d_.mf_num;
    int sv_n = d_.sv_num;
    int sf_n = d_.sf_num;

    int v_n = mv_n + sv_n;
    int f_n = mf_n + sf_n;
    if (d_.dim == 2)
    {
      Eigen::MatrixXd F1, F2, F3;
      igl::local_basis(d_.m_V, d_.m_T, F1, F2, F3);
      igl::scaf::compute_surface_gradient_matrix(d_.m_V, d_.m_T, F1, F2, Dx_m,
                                                 Dy_m);

      igl::scaf::compute_scaffold_gradient_matrix(d_, Dx_s, Dy_s);
    }
    else
    {

      if (d_.m_T.cols() == 3)
      {
        igl::scaf::simplified_covariance_scatter_matrix(d_.m_V, d_.m_T,
                                                        Dx_m, Dy_m, Dz_m);
      }
      else
      {
        Eigen::SparseMatrix<double> Gm;
        igl::grad(d_.m_V, d_.m_T, Gm);

        Dx_m = Gm.block(0, 0, mf_n, mv_n);
        Dy_m = Gm.block(mf_n, 0, mf_n, mv_n);
        Dz_m = Gm.block(2 * mf_n, 0, mf_n, mv_n);
      }

      Eigen::SparseMatrix<double> Gs;
      igl::grad(d_.w_uv, d_.s_T, Gs);

      Dx_s = Gs.block(0, 0, sf_n, v_n);
      Dy_s = Gs.block(sf_n, 0, sf_n, v_n);
      Dz_s = Gs.block(2 * sf_n, 0, sf_n, v_n);
    }
    int dim = d_.dim;

    Dx_m.makeCompressed();
    Dy_m.makeCompressed();
    Dz_m.makeCompressed();
    Ri_m = MatrixXd::Zero(Dx_m.rows(), dim * dim);
    Ji_m.resize(Dx_m.rows(), dim * dim);
    W_m.resize(Dx_m.rows(), dim * dim);

    Dx_s.makeCompressed();
    Dy_s.makeCompressed();
    Dz_s.makeCompressed();
    Ri_s = MatrixXd::Zero(Dx_s.rows(), dim * dim);
    Ji_s.resize(Dx_s.rows(), dim * dim);
    W_s.resize(Dx_s.rows(), dim * dim);

    data.has_pre_calc = true;
  }
}

IGL_INLINE Eigen::MatrixXd igl::scaf_solve(SCAFData &d_, int iter_num)
{
  using namespace std;
  using namespace Eigen;
  //  auto &ws = wssolver;
  double last_mesh_energy = igl::scaf::compute_energy(d_, false) / d_.mesh_measure;
  std::cout << "Initial Energy" << last_mesh_energy << std::endl;
  cout << "Initial V_num: " << d_.mv_num << " F_num: " << d_.mf_num << endl;
  d_.energy = igl::scaf::compute_energy(d_, true) / d_.mesh_measure;

  igl::Timer timer;
  timer.start();

  igl::scaf::mesh_improve(d_);

  double new_weight = d_.mesh_measure * last_mesh_energy / (d_.sf_num * 100);
  igl::scaf::adjust_scaf_weight(d_, new_weight);

  d_.energy = igl::scaf::perform_iteration(d_);

  cout << "Iteration time = " << timer.getElapsedTime() << endl;
  double current_mesh_energy =
      igl::scaf::compute_energy(d_, false) / d_.mesh_measure - 4;
  double mesh_energy_decrease = last_mesh_energy - current_mesh_energy;
  cout << "Energy After:"
       << d_.energy - 4
       << "\tMesh Energy:"
       << current_mesh_energy
       << "\tEnergy Decrease"
       << mesh_energy_decrease
       << endl;
  cout << "V_num: " << d_.v_num << " F_num: " << d_.f_num << endl;
  last_mesh_energy = current_mesh_energy;

  Eigen::MatrixXd wuv3 = Eigen::MatrixXd::Zero(d_.v_num, 3);
  wuv3.leftCols(2) = d_.w_uv;
}
