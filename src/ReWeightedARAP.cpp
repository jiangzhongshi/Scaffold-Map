//
// Created by Zhongshi Jiang on 10/9/16.
//

#include "ReWeightedARAP.h"
#include "ScafData.h"
#include "util/triangle_utils.h"
#include "util/tet_utils.h"

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
#include <igl/fit_rotations.h>
#include <igl/columnize.h>
#include <igl/arap_rhs.h>
#include <igl/cotmatrix_entries.h>

//#define NO_SCAFFOLD ;
using namespace std;
using namespace Eigen;

void ReWeightedARAP::compute_surface_gradient_matrix(
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

void ReWeightedARAP::simplified_covariance_scatter_matrix(
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

//only 2D
void ReWeightedARAP::compute_scaffold_gradient_matrix(
    Eigen::SparseMatrix<double> &D1, Eigen::SparseMatrix<double> &D2)
{
  assert(d_.dim == 2);
  Eigen::SparseMatrix<double> G;
  MatrixXi F_s = d_.s_T;
  int vn = d_.v_num;
  MatrixXd V = MatrixXd::Zero(vn, 3);
  V.leftCols(2) = d_.w_uv;

  double min_bnd_edge_len = INFINITY;
  int acc_bnd = 0;
  for (int i = 0; i < d_.bnd_sizes.size(); i++)
  {
    int current_size = d_.bnd_sizes[i];

    for (int e = acc_bnd; e < acc_bnd + current_size - 1; e++)
    {
      min_bnd_edge_len = (std::min)(min_bnd_edge_len,
                                  (d_.w_uv.row(d_.internal_bnd(e)) -
                                   d_.w_uv.row(d_.internal_bnd(e + 1)))
                                      .squaredNorm());
    }
    min_bnd_edge_len = (std::min)(min_bnd_edge_len,
                                (d_.w_uv.row(d_.internal_bnd(acc_bnd)) -
                                 d_.w_uv.row(d_.internal_bnd(acc_bnd + current_size -
                                                             1)))
                                    .squaredNorm());
    acc_bnd += current_size;
  }

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

void ReWeightedARAP::pre_calc()
{
  if (!has_pre_calc)
  {

    v_n = d_.mv_num + d_.sv_num;
    f_n = d_.mf_num + d_.sf_num;
    if (d_.dim == 2)
    {
      Eigen::MatrixXd F1, F2, F3;
      igl::local_basis(d_.m_V, d_.m_T, F1, F2, F3);
      compute_surface_gradient_matrix(d_.m_V, d_.m_T, F1, F2, Dx_m,
                                      Dy_m);

      compute_scaffold_gradient_matrix(Dx_s, Dy_s);
    }
    else
    {
      if (d_.m_T.cols() == 3)
      {
        auto &data = arap_data;
        using namespace Eigen;
        using namespace std;
        typedef double Scalar;

        // number of vertices
        data.n = d_.mv_num;
        //assert(F.cols() == 3 && "For now only triangles");
        // dimension
        //const int dim = V.cols();
        data.dim = 3;

        typedef SparseMatrix<Scalar> SparseMatrixS;
        SparseMatrixS ref_map, ref_map_dim;

        const MatrixXd &ref_V = (d_.m_V);
        const MatrixXi &ref_F = (d_.m_T);
        igl::cotmatrix(d_.m_V, d_.m_T, CotMat);
        igl::cotmatrix_entries(d_.m_V, d_.m_T, arap_cot_entries);

        igl::ARAPEnergyType eff_energy =
            igl::ARAP_ENERGY_TYPE_SPOKES_AND_RIMS;

        // Get covariance scatter matrix, when applied collects the covariance
        // matrices used to fit rotations to during optimization
        igl::covariance_scatter_matrix(ref_V, ref_F, eff_energy, data.CSM);

        igl::arap_rhs(ref_V, ref_F, data.dim, eff_energy, data.K);
        update_surface_ARAP_rots();
      }
      else
      {
        Eigen::SparseMatrix<double> Gm;
        igl::grad(d_.m_V, d_.m_T, Gm);

        Dx_m = Gm.block(0, 0, d_.mf_num, d_.mv_num);
        Dy_m = Gm.block(d_.mf_num, 0, d_.mf_num, d_.mv_num);
        Dz_m = Gm.block(2 * d_.mf_num, 0, d_.mf_num, d_.mv_num);
      }

      Eigen::SparseMatrix<double> Gs;
      igl::grad(d_.w_uv, d_.s_T, Gs);

      Dx_s = Gs.block(0, 0, d_.sf_num, v_n);
      Dy_s = Gs.block(d_.sf_num, 0, d_.sf_num, v_n);
      Dz_s = Gs.block(2 * d_.sf_num, 0, d_.sf_num, v_n);
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

    has_pre_calc = true;
  }
}

void ReWeightedARAP::solve_weighted_proxy(Eigen::MatrixXd &uv_new)
{
  igl::Timer timer;
  timer.start();
  compute_jacobians(uv_new);
  igl::slim_update_weights_and_closest_rotations_with_jacobians(Ji_m, d_.slim_energy,
                                                                0, W_m, Ri_m);
  igl::slim_update_weights_and_closest_rotations_with_jacobians(Ji_s, d_.scaf_energy,
                                                                0, W_s, Ri_s);
  //  cout << "update_weigths = "<<timer.getElapsedTime()<<endl;
  solve_weighted_arap(uv_new);
}

void ReWeightedARAP::compute_jacobians(const Eigen::MatrixXd &uv,
                                       const Eigen::SparseMatrix<double> &Dx,
                                       const Eigen::SparseMatrix<double> &Dy,
                                       const Eigen::SparseMatrix<double> &Dz,
                                       Eigen::MatrixXd &Ji)
{
  if (Dz.rows()==0) {
  // Ji=[D1*u,D2*u,D1*v,D2*v];
  Ji.resize(Dx.rows(), 4);
  Ji.col(0) = Dx * uv.col(0);
  Ji.col(1) = Dy * uv.col(0);
  Ji.col(2) = Dx * uv.col(1);
  Ji.col(3) = Dy * uv.col(1);
    } else {
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
    }
}

double ReWeightedARAP::compute_energy(const Eigen::MatrixXd &V_new,
                                      bool whole)
{
  compute_jacobians(V_new, whole);
  double energy = 0;
  if (d_.dim==2 || d_.m_T.cols() == 4) {
    energy += igl::mapping_energy_with_jacobians(Ji_m, d_.m_M, d_.slim_energy, 0);
  } else {    // arap
    energy += compute_surface_ARAP_energy(V_new);
  }
  if (whole)
    energy += igl::mapping_energy_with_jacobians(Ji_s, d_.s_M, d_.scaf_energy, 0);

  energy += compute_soft_constraint_energy(V_new);

  return energy;
}
void ReWeightedARAP::compute_jacobians(const MatrixXd &V_new, bool whole)
{
  Eigen::MatrixXd m_V_new = V_new.topRows(d_.mv_num);
  if (d_.m_T.cols() == 4 || d_.dim ==2)
    compute_jacobians(m_V_new, Dx_m, Dy_m, Dz_m, Ji_m);
  if (whole) compute_jacobians(V_new, Dx_s, Dy_s, Dz_s, Ji_s);
}

void ReWeightedARAP::change_scaffold_reference(const MatrixXd &s_uv)
{
  assert(s_uv.rows() == d_.v_num && "CHANGE_SCAFFOLD_REFERENCE");
  assert(s_uv.cols() == d_.dim && "DIMENSION_NOT_MATCH");

  MatrixXi F_s = d_.s_T;
  int vn = d_.v_num;

  MatrixXd V = MatrixXd::Zero(vn, 3);
  V.leftCols(s_uv.cols()) = s_uv;

  int dim = d_.dim;
  if (dim == 2)
  {
    compute_scaffold_gradient_matrix(Dx_s, Dy_s);
  }
  else
  {
    Eigen::SparseMatrix<double> G;
    igl::grad(V, F_s, G);

    Dx_s = G.block(0, 0, d_.sf_num, vn);
    Dy_s = G.block(d_.sf_num, 0, d_.sf_num, vn);
    Dz_s = G.block(2 * d_.sf_num, 0, d_.sf_num, vn);
  }

  Dx_s.makeCompressed();
  Dy_s.makeCompressed();
  Dz_s.makeCompressed();
}


double ReWeightedARAP::perform_iteration(MatrixXd &w_uv)
{
  Eigen::MatrixXd V_out = w_uv;
  solve_weighted_proxy(V_out);
  auto whole_E =
      [this](Eigen::MatrixXd &uv) { return this->compute_energy(uv); };

  igl::Timer timer;
  timer.start();
  Eigen::MatrixXi w_T;
  if (d_.m_T.cols() == d_.s_T.cols())
    igl::cat(1, d_.m_T, d_.s_T, w_T);
  else
    w_T = d_.s_T;
  double energy = igl::flip_avoiding_line_search(w_T, w_uv, V_out,
                                                 whole_E, -1) /
                  d_.mesh_measure;
  return energy;
}

double ReWeightedARAP::perform_iteration(Eigen::MatrixXd &w_uv,
                                         bool whole)
{
  Eigen::MatrixXd V_out = w_uv;
  solve_weighted_proxy(V_out);

  auto whole_E = [this, whole](Eigen::MatrixXd &uv) { return this->compute_energy(uv); }; // whole

  igl::Timer timer;
  timer.start();
  double energy = igl::flip_avoiding_line_search(d_.s_T, w_uv, V_out,
                                                 whole_E, -1) /
                  d_.mesh_measure;
  //  cout << "LineSearch = " << timer.getElapsedTime()<<endl;
  return energy;
}

double
ReWeightedARAP::compute_soft_constraint_energy(const Eigen::MatrixXd &uv) const
{
  double e = 0;
  for (auto const &x : d_.soft_cons)
    e += d_.soft_const_p * (x.second - uv.row(x.first)).squaredNorm();

  return e;
}

void ReWeightedARAP::after_mesh_improve()
{
  // differ to pre_calc in the sense of only updating scaffold ones

  v_n = d_.mv_num + d_.sv_num;
  f_n = d_.mf_num + d_.sf_num;
  if (d_.dim == 2)
  {
    compute_scaffold_gradient_matrix(Dx_s, Dy_s);
  }
  else
  {
    Eigen::SparseMatrix<double> Gs;
    igl::grad(d_.w_uv, d_.s_T, Gs);

    Dx_s = Gs.block(0, 0, d_.sf_num, v_n);
    Dy_s = Gs.block(d_.sf_num, 0, d_.sf_num, v_n);
    Dz_s = Gs.block(2 * d_.sf_num, 0, d_.sf_num, v_n);
  }
  int dim = d_.dim;

  Dx_s.makeCompressed();
  Dy_s.makeCompressed();
  Dz_s.makeCompressed();
  Ri_s = MatrixXd::Zero(Dx_s.rows(), dim * dim);
  Ji_s.resize(Dx_s.rows(), dim * dim);
  W_s.resize(Dx_s.rows(), dim * dim);
}


void ReWeightedARAP::enlarge_internal_reference(double scale)
{
  Dx_m /= scale;
  Dy_m /= scale;
}

void ReWeightedARAP::update_surface_ARAP_rots()
{
  // arap code block goes here
  using namespace Eigen;
  using namespace std;

  auto &data = arap_data;
  const int n = data.n;

  // changes each arap iteration
  MatrixXd U_prev = d_.w_uv.topRows(d_.mv_num);

  MatrixXd Udim = U_prev.replicate(data.dim, 1);
  // As if U.col(2) was 0
  MatrixXd S = data.CSM * Udim;
  // THIS NORMALIZATION IS IMPORTANT TO GET SINGLE PRECISION SVD CODE TO WORK
  // CORRECTLY.
  S /= S.array().abs().maxCoeff();

  const int Rdim = data.dim;
  MatrixXd R(Rdim, data.CSM.rows());

  igl::fit_rotations(S, true, R);

  arap_rots.resize(R.cols() / 3);
  for (int i = 0; i < arap_rots.size(); i++)
  {
    arap_rots[i] = R.block(0, 3 * i, 3, 3);
  }
  /* 
  // Number of rotations: #vertices or #elements 
  int num_rots = data.K.cols()/Rdim/Rdim; 
  // distribute group rotations to vertices in each group 
 
  VectorXd Rcol; 
  igl::columnize(R,num_rots,2,Rcol); 
  VectorXd Bcol = data.K * Rcol; 
  assert(Bcol.size() == data.n*data.dim); 
 
  arap_Bc = Map<MatrixXd>(Bcol.data(), data.n, data.dim);  //column order 
   */
}

double
ReWeightedARAP::compute_surface_ARAP_energy(const Eigen::MatrixXd &U)
    const
{
  using Scalar = double;
  using namespace igl;
  using namespace Eigen;

  double energy = 0;

  const auto &F = d_.m_T;
  const auto &V = d_.m_V;
  const auto &R = arap_rots;
  int m = F.rows();
  Matrix<int, Dynamic, 2> edges;
  edges.resize(3, 2);
  edges << 1, 2,
      2, 0,
      0, 1;

  auto &M = d_.m_M;

  // gather cotangent weights
  auto &C = arap_cot_entries;
  // should have weights for each edge
  assert(C.cols() == edges.rows());
  // loop over elements
  for (int i = 0; i < m; i++)
  {
    // loop over edges of element
    for (int e = 0; e < edges.rows(); e++)
    {
      int source = F(i, edges(e, 0));
      int dest = F(i, edges(e, 1));
      VectorXd edge_vec = (V.row(source) - V.row(dest));
      VectorXd edge_uec = (U.row(source) - U.row(dest));
      // loop over edges again
      for (int f = 0; f < edges.rows(); f++)
      {
        int Rs = F(i, edges(f, 0));
        int Rd = F(i, edges(f, 1));
        if (Rs == source && Rd == dest)
        {
          energy += M(Rs) * C(i, e) * (edge_uec - R[Rs] * edge_vec).squaredNorm();
          energy += M(Rd) * C(i, e) * (edge_uec - R[Rd] * edge_vec).squaredNorm();
        }
        else if (Rd == source)
        {
          energy += M(Rd) * C(i, (e + 2) % 3) * (edge_uec - R[Rd] * edge_vec).squaredNorm();
        }
        else if (Rs == dest)
        {
          energy += M(Rs) * C(i, (e + 1) % 3) * (edge_uec - R[Rs] * edge_vec).squaredNorm();
        }
      }
    }
  }

  return energy * arap_energy_p / 2.;
}
