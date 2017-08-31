//
// Created by Zhongshi Jiang on 10/9/16.
//

#include "ReWeightedARAP.h"
#include "ScafData.h"
#include "util/triangle_utils.h"

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

//#define NO_SCAFFOLD ;
using namespace std;
using namespace Eigen;

void ReWeightedARAP::compute_surface_gradient_matrix(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &F1,
    const Eigen::MatrixXd &F2,
    Eigen::SparseMatrix<double> &D1,
    Eigen::SparseMatrix<double> &D2) {
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
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    Eigen::SparseMatrix<double>& Dx,Eigen::SparseMatrix<double>& Dy,
    Eigen::SparseMatrix<double>& Dz) {
  using namespace Eigen;
  auto energy = igl::ARAP_ENERGY_TYPE_SPOKES_AND_RIMS;
  SparseMatrix<double> Kx,Ky,Kz;
  igl::arap_linear_block(V,F,0,energy,Kx);
  igl::arap_linear_block(V,F,1,energy,Ky);
  igl::arap_linear_block(V,F,2,energy,Kz);
  Dx = Kx.transpose();
  Dy = Ky.transpose();
  Dz = Kz.transpose();
}

//only 2D
void ReWeightedARAP::compute_scaffold_gradient_matrix(
    Eigen::SparseMatrix<double> &D1, Eigen::SparseMatrix<double> &D2) {
  Eigen::SparseMatrix<double> G;
  MatrixXi F_s = d_.s_T;
  int vn = d_.v_num;
  MatrixXd V = MatrixXd::Zero(vn, 3);
  V.leftCols(2) = d_.w_uv;
//  std::cout<<"Avg Mesh Area"<<d_.mesh_measure/d_.mv_num<<std::endl;

  double min_bnd_edge_len = INFINITY;
  int acc_bnd = 0;
  for(int i=0; i<d_.bnd_sizes.size(); i++) {
    int current_size = d_.bnd_sizes[i];

    for(int e=acc_bnd; e<acc_bnd + current_size - 1; e++) {
      min_bnd_edge_len = std::min(min_bnd_edge_len,
                                  (d_.w_uv.row(d_.internal_bnd(e)) -
                                          d_.w_uv.row(d_.internal_bnd(e+1)))
                                      .squaredNorm());
    }
    min_bnd_edge_len = std::min(min_bnd_edge_len,
                                (d_.w_uv.row(d_.internal_bnd(acc_bnd)) -
                            d_.w_uv.row(d_.internal_bnd(acc_bnd +current_size -
                                1))).squaredNorm());
    acc_bnd += current_size;
  }

//  std::cout<<"MinBndEdge"<<min_bnd_edge_len<<std::endl;
  double area_threshold = min_bnd_edge_len/4.0;

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

void ReWeightedARAP::pre_calc() {
  if (!has_pre_calc) {
    mv_n = d_.mv_num;
    mf_n = d_.mf_num;
    sv_n = d_.sv_num;
    sf_n = d_.sf_num;

    v_n = mv_n + sv_n;
    f_n = mf_n + sf_n;
    if (d_.dim == 2) {
      Eigen::MatrixXd F1, F2, F3;
      igl::local_basis(d_.m_V, d_.m_T, F1, F2, F3);
      compute_surface_gradient_matrix(d_.m_V, d_.m_T, F1, F2, Dx_m,
                                      Dy_m);

      compute_scaffold_gradient_matrix(Dx_s, Dy_s);
    } else {

      if(d_.m_T.cols() == 3) {
        simplified_covariance_scatter_matrix(d_.m_V, d_.m_T,
                                             Dx_m, Dy_m, Dz_m);
      } else {
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

    has_pre_calc = true;
  }
}

template <>
void ReWeightedARAP::update_weights_and_closest_rotations<2>(
    const Eigen::MatrixXd& Ji,
    ScafData::SLIM_ENERGY energy_type,
    Eigen::MatrixXd& W,
    Eigen::MatrixXd& Ri);
template <>
void ReWeightedARAP::update_weights_and_closest_rotations<3>(
    const Eigen::MatrixXd& Ji,
    ScafData::SLIM_ENERGY energy_type,
    Eigen::MatrixXd& W,
    Eigen::MatrixXd& Ri);
void ReWeightedARAP::solve_weighted_proxy(Eigen::MatrixXd &uv_new)
{
  igl::Timer timer;
  timer.start();
  compute_jacobians(uv_new);
  if(d_.dim==2) {
    update_weights_and_closest_rotations<2>(Ji_m, d_.slim_energy,
                                            W_m, Ri_m);
    update_weights_and_closest_rotations<2>(Ji_s, d_.scaf_energy,
                                            W_s, Ri_s);
  } else {
    update_weights_and_closest_rotations<3>(Ji_m, d_.slim_energy,
                                            W_m, Ri_m);
    update_weights_and_closest_rotations<3>(Ji_s, d_.scaf_energy,
                                            W_s, Ri_s);
  }
//  cout << "update_weigths = "<<timer.getElapsedTime()<<endl;
  solve_weighted_arap(uv_new);
}

void ReWeightedARAP::compute_jacobians(const Eigen::MatrixXd &uv,
                                       const Eigen::SparseMatrix<double>& Dx,
                                       const Eigen::SparseMatrix<double>& Dy,
                                       const Eigen::SparseMatrix<double>& Dz,
                                       Eigen::MatrixXd& Ji) {
    // Ji=[D1*u,D2*u,D3*u, D1*v,D2*v, D3*v, D1*w,D2*w,D3*w];
    Ji.resize(Dx.rows(),9);
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

void ReWeightedARAP::compute_jacobians(const Eigen::MatrixXd &uv,
                                       const Eigen::SparseMatrix<double>& Dx,
                                       const Eigen::SparseMatrix<double>& Dy,
                                       Eigen::MatrixXd& Ji) {
    // Ji=[D1*u,D2*u,D1*v,D2*v];
    Ji.resize(Dx.rows(),4);
    Ji.col(0) = Dx * uv.col(0);
    Ji.col(1) = Dy * uv.col(0);
    Ji.col(2) = Dx * uv.col(1);
    Ji.col(3) = Dy * uv.col(1);
}

template <>
void ReWeightedARAP::update_weights_and_closest_rotations<2>(
    const Eigen::MatrixXd& Ji,
    ScafData::SLIM_ENERGY energy_type,
    Eigen::MatrixXd& W,
    Eigen::MatrixXd& Ri) {

  const double eps = 1e-8;
  double exp_f = d_.exp_factor;

  W.resize(Ji.rows(),4);
  Ri.resize(Ji.rows(),4);
  for (int i = 0; i < Ji.rows(); ++i) {
    typedef Eigen::Matrix<double, 2, 2> Mat2;
    typedef Eigen::Matrix<double, 2, 1> Vec2;
    Mat2 ji, ri, ti, ui, vi;
    Vec2 sing;
    Vec2 closest_sing_vec;
    Mat2 mat_W;
    Vec2 m_sing_new;
    double s1, s2;

    ji(0, 0) = Ji(i, 0);
    ji(0, 1) = Ji(i, 1);
    ji(1, 0) = Ji(i, 2);
    ji(1, 1) = Ji(i, 3);

    igl::polar_svd(ji, ri, ti, ui, sing, vi);

    s1 = sing(0);
    s2 = sing(1);

    // Branch between mesh face and scaffold faces.
    switch (energy_type) {
      case ScafData::ARAP: {
        m_sing_new << 1, 1;
        break;
      }
      case ScafData::SYMMETRIC_DIRICHLET: {
        double s1_g = 2 * (s1 - pow(s1, -3));
        double s2_g = 2 * (s2 - pow(s2, -3));
        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(
            s2_g / (2 * (s2 - 1)));
        break;
      }
      case ScafData::LOG_ARAP: {
        double s1_g = 2 * (log(s1) / s1);
        double s2_g = 2 * (log(s2) / s2);
        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(
            s2_g / (2 * (s2 - 1)));
        break;
      }
      case ScafData::CONFORMAL: {
        double s1_g = 1 / (2 * s2) - s2 / (2 * pow(s1, 2));
        double s2_g = 1 / (2 * s1) - s1 / (2 * pow(s2, 2));

        double geo_avg = sqrt(s1 * s2);
        double s1_min = geo_avg;
        double s2_min = geo_avg;

        m_sing_new << sqrt(s1_g / (2 * (s1 - s1_min))), sqrt(
            s2_g / (2 * (s2 - s2_min)));

        // change local step
        closest_sing_vec << s1_min, s2_min;
        ri = ui * closest_sing_vec.asDiagonal() * vi.transpose();
        break;
      }
      case ScafData::EXP_CONFORMAL: {
        double s1_g = 2 * (s1 - pow(s1, -3));
        double s2_g = 2 * (s2 - pow(s2, -3));

        double geo_avg = sqrt(s1 * s2);
        double s1_min = geo_avg;
        double s2_min = geo_avg;

        double in_exp = exp_f * ((pow(s1, 2) + pow(s2, 2)) / (2 * s1 * s2));
        double exp_thing = exp(in_exp);

        s1_g *= exp_thing * exp_f;
        s2_g *= exp_thing * exp_f;

        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(
            s2_g / (2 * (s2 - 1)));
        break;
      }
      case ScafData::EXP_SYMMETRIC_DIRICHLET: {
        double s1_g = 2 * (s1 - pow(s1, -3));
        double s2_g = 2 * (s2 - pow(s2, -3));

        double in_exp =
            exp_f * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2));
        double exp_thing = exp(in_exp);

        s1_g *= exp_thing * exp_f;
        s2_g *= exp_thing * exp_f;

        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(
            s2_g / (2 * (s2 - 1)));
        break;
      }
      default:break;
    }

    if (abs(s1 - 1) < eps) m_sing_new(0) = 1;
    if (abs(s2 - 1) < eps) m_sing_new(1) = 1;

    mat_W = ui * m_sing_new.asDiagonal() * ui.transpose();

    W(i,0) = mat_W(0, 0);
    W(i,1) = mat_W(0, 1);
    W(i,2) = mat_W(1, 0);
    W(i,3) = mat_W(1, 1);

    // 2) Update local step (doesn't have to be a rotation, for instance in case of conformal energy)
    Ri(i, 0) = ri(0, 0);
    Ri(i, 1) = ri(1, 0);
    Ri(i, 2) = ri(0, 1);
    Ri(i, 3) = ri(1, 1);
  }
}

template <>
void ReWeightedARAP::update_weights_and_closest_rotations<3>(
    const Eigen::MatrixXd& Ji,
    ScafData::SLIM_ENERGY energy_type,
    Eigen::MatrixXd& W,
    Eigen::MatrixXd& Ri) {

  const double eps = 1e-8;
  double exp_f = d_.exp_factor;

  typedef Eigen::Matrix<double, 3, 1> Vec3;
  typedef Eigen::Matrix<double, 3, 3> Mat3;
  Mat3 ji;
  Vec3 m_sing_new;
  Vec3 closest_sing_vec;
  const double sqrt_2 = sqrt(2);

  W.resize(Ji.rows(),9);
  Ri.resize(Ji.rows(),9);
  for (int i = 0; i < Ji.rows(); ++i) {
    ji(0, 0) = Ji(i, 0);
    ji(0, 1) = Ji(i, 1);
    ji(0, 2) = Ji(i, 2);
    ji(1, 0) = Ji(i, 3);
    ji(1, 1) = Ji(i, 4);
    ji(1, 2) = Ji(i, 5);
    ji(2, 0) = Ji(i, 6);
    ji(2, 1) = Ji(i, 7);
    ji(2, 2) = Ji(i, 8);

    Mat3 ri, ti, ui, vi;
    Vec3 sing;
    igl::polar_svd(ji, ri, ti, ui, sing, vi);

    double s1 = sing(0);
    double s2 = sing(1);
    double s3 = sing(2);

    // 1) Update Weights
    switch (energy_type) {
      case ScafData::ARAP: {
        m_sing_new << 1, 1, 1;
        break;
      }
      case ScafData::LOG_ARAP: {
        double s1_g = 2 * (log(s1) / s1);
        double s2_g = 2 * (log(s2) / s2);
        double s3_g = 2 * (log(s3) / s3);
        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(
            s2_g / (2 * (s2 - 1))), sqrt(s3_g / (2 * (s3 - 1)));
        break;
      }
      case ScafData::SYMMETRIC_DIRICHLET: {
        double s1_g = 2 * (s1 - pow(s1, -3));
        double s2_g = 2 * (s2 - pow(s2, -3));
        double s3_g = 2 * (s3 - pow(s3, -3));
        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(
            s2_g / (2 * (s2 - 1))), sqrt(s3_g / (2 * (s3 - 1)));
        break;
      }
      case ScafData::EXP_SYMMETRIC_DIRICHLET: {
        double s1_g = 2 * (s1 - pow(s1, -3));
        double s2_g = 2 * (s2 - pow(s2, -3));
        double s3_g = 2 * (s3 - pow(s3, -3));
        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(
            s2_g / (2 * (s2 - 1))), sqrt(s3_g / (2 * (s3 - 1)));

        double in_exp = exp_f
            * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2)
                + pow(s3, 2) + pow(s3, -2));
        double exp_thing = exp(in_exp);

        s1_g *= exp_thing * exp_f;
        s2_g *= exp_thing * exp_f;
        s3_g *= exp_thing * exp_f;

        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(
            s2_g / (2 * (s2 - 1))), sqrt(s3_g / (2 * (s3 - 1)));

        break;
      }
      case ScafData::CONFORMAL: {
        double common_div = 9 * (pow(s1 * s2 * s3, 5. / 3.));

        double s1_g =
            (-2 * s2 * s3 * (pow(s2, 2) + pow(s3, 2) - 2 * pow(s1, 2)))
                / common_div;
        double s2_g =
            (-2 * s1 * s3 * (pow(s1, 2) + pow(s3, 2) - 2 * pow(s2, 2)))
                / common_div;
        double s3_g =
            (-2 * s1 * s2 * (pow(s1, 2) + pow(s2, 2) - 2 * pow(s3, 2)))
                / common_div;

        double closest_s = sqrt(pow(s1, 2) + pow(s3, 2)) / sqrt_2;
        double s1_min = closest_s;
        double s2_min = closest_s;
        double s3_min = closest_s;

        m_sing_new << sqrt(s1_g / (2 * (s1 - s1_min))), sqrt(
            s2_g / (2 * (s2 - s2_min))), sqrt(
            s3_g / (2 * (s3 - s3_min)));

        // change local step
        closest_sing_vec << s1_min, s2_min, s3_min;
        ri = ui * closest_sing_vec.asDiagonal() * vi.transpose();
        break;
      }
      case ScafData::EXP_CONFORMAL: {
        // E_conf = (s1^2 + s2^2 + s3^2)/(3*(s1*s2*s3)^(2/3) )
        // dE_conf/ds1 = (-2*(s2*s3)*(s2^2+s3^2 -2*s1^2) ) / (9*(s1*s2*s3)^(5/3))
        // Argmin E_conf(s1): s1 = sqrt(s1^2+s2^2)/sqrt(2)
        double common_div = 9 * (pow(s1 * s2 * s3, 5. / 3.));

        double s1_g =
            (-2 * s2 * s3 * (pow(s2, 2) + pow(s3, 2) - 2 * pow(s1, 2)))
                / common_div;
        double s2_g =
            (-2 * s1 * s3 * (pow(s1, 2) + pow(s3, 2) - 2 * pow(s2, 2)))
                / common_div;
        double s3_g =
            (-2 * s1 * s2 * (pow(s1, 2) + pow(s2, 2) - 2 * pow(s3, 2)))
                / common_div;

        double in_exp = exp_f * ((pow(s1, 2) + pow(s2, 2) + pow(s3, 2))
            / (3 * pow((s1 * s2 * s3), 2. / 3)));;
        double exp_thing = exp(in_exp);

        double closest_s = sqrt(pow(s1, 2) + pow(s3, 2)) / sqrt_2;
        double s1_min = closest_s;
        double s2_min = closest_s;
        double s3_min = closest_s;

        s1_g *= exp_thing * exp_f;
        s2_g *= exp_thing * exp_f;
        s3_g *= exp_thing * exp_f;

        m_sing_new << sqrt(s1_g / (2 * (s1 - s1_min))), sqrt(
            s2_g / (2 * (s2 - s2_min))), sqrt(
            s3_g / (2 * (s3 - s3_min)));

        // change local step
        closest_sing_vec << s1_min, s2_min, s3_min;
        ri = ui * closest_sing_vec.asDiagonal() * vi.transpose();
      }
    }
    if (std::abs(s1 - 1) < eps) m_sing_new(0) = 1;
    if (std::abs(s2 - 1) < eps) m_sing_new(1) = 1;
    if (std::abs(s3 - 1) < eps) m_sing_new(2) = 1;
    Mat3 mat_W;
    mat_W = ui * m_sing_new.asDiagonal() * ui.transpose();

    W(i,0) = mat_W(0, 0);
    W(i,1) = mat_W(0, 1);
    W(i,2) = mat_W(0, 2);
    W(i,3) = mat_W(1, 0);
    W(i,4) = mat_W(1, 1);
    W(i,5) = mat_W(1, 2);
    W(i,6) = mat_W(2, 0);
    W(i,7) = mat_W(2, 1);
    W(i,8) = mat_W(2, 2);

    // 2) Update closest rotations (not rotations in case of conformal energy)
    Ri(i, 0) = ri(0, 0);
    Ri(i, 1) = ri(1, 0);
    Ri(i, 2) = ri(2, 0);
    Ri(i, 3) = ri(0, 1);
    Ri(i, 4) = ri(1, 1);
    Ri(i, 5) = ri(2, 1);
    Ri(i, 6) = ri(0, 2);
    Ri(i, 7) = ri(1, 2);
    Ri(i, 8) = ri(2, 2);

  }

}

double ReWeightedARAP::compute_energy(const Eigen::MatrixXd &V_new,
                                      bool whole) {
  compute_jacobians(V_new, whole);

  double energy = compute_energy_from_jacobians(Ji_m,
                                                d_.m_M,
                                                d_.slim_energy);
  if (whole)
    energy += compute_energy_from_jacobians(Ji_s,
                                            d_.s_M,
                                            d_.scaf_energy);

  energy += compute_soft_constraint_energy(V_new);

  return energy;
}
void ReWeightedARAP::compute_jacobians(const MatrixXd &V_new, bool whole) {
  Eigen::MatrixXd m_V_new = V_new.topRows(mv_n);
  if (d_.dim == 2) {
    compute_jacobians(m_V_new, Dx_m, Dy_m, Ji_m);
   if(whole) compute_jacobians(V_new, Dx_s, Dy_s, Ji_s);
  } else {
    compute_jacobians(m_V_new, Dx_m, Dy_m, Dz_m, Ji_m);
   if(whole) compute_jacobians(V_new, Dx_s, Dy_s, Dz_s, Ji_s);
  }
}

double ReWeightedARAP::compute_energy_from_jacobians(const Eigen::MatrixXd &Ji,
                                                    const VectorXd &areas,
                                                    ScafData::SLIM_ENERGY energy_type) {
  double energy = 0;
  int dim = Ji.cols() == 4 ? 2:3;
  if(dim == 2) {
    Eigen::Matrix<double, 2, 2> ji;
    for (int i = 0; i < Ji.rows(); i++) {
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

      switch (energy_type) {
        case ScafData::ARAP: {
          energy += areas(i) * (pow(s1 - 1, 2) + pow(s2 - 1, 2));
          break;
        }
        case ScafData::SYMMETRIC_DIRICHLET: {
          energy +=
              areas(i)
                  * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2));
          break;
        }
        case ScafData::LOG_ARAP: {
          energy += areas(i) * (pow(log(s1), 2) + pow(log(s2), 2));
          break;
        }
        case ScafData::CONFORMAL: {
          energy += areas(i) * ((pow(s1, 2) + pow(s2, 2)) / (2 * s1 * s2));
          break;
        }
        default:break;

      }

    }
  } else {
    Eigen::Matrix<double, 3, 3> ji;
    for (int i = 0; i < Ji.rows(); i++) {
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

      switch (energy_type) {
        case ScafData::ARAP: {
          energy += areas(i)
              * (pow(s1 - 1, 2) + pow(s2 - 1, 2) + pow(s3 - 1, 2));
          break;
        }
        case ScafData::SYMMETRIC_DIRICHLET: {
          energy += areas(i)
              * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2)
                  + pow(s3, 2) + pow(s3, -2));
          break;
        }
        case ScafData::LOG_ARAP: {
          energy += areas(i)
              * (pow(log(s1), 2) + pow(log(std::abs(s2)), 2)
                  + pow(log(std::abs(s3)), 2));
          break;
        }
        case ScafData::CONFORMAL: {
          energy += areas(i) * ((pow(s1, 2) + pow(s2, 2) + pow(s3, 2))
              / (3 * pow(s1 * s2 * s3, 2. / 3.)));
          break;
        }
        case ScafData::EXP_CONFORMAL: {
          energy += areas(i) * exp((pow(s1, 2) + pow(s2, 2) + pow(s3, 2))
                                       / (3
                                           * pow(s1 * s2 * s3, 2. / 3.)));
          break;
        }
        default:
          assert(false);
      }

    }
  }
  return energy;
}


void ReWeightedARAP::change_scaffold_reference(const MatrixXd &s_uv) {
  assert(s_uv.rows() == d_.v_num && "CHANGE_SCAFFOLD_REFERENCE");
  assert(s_uv.cols() == d_.dim && "DIMENSION_NOT_MATCH");

  MatrixXi F_s = d_.s_T;
  int vn = d_.v_num;

  MatrixXd V = MatrixXd::Zero(vn, 3);
  V.leftCols(s_uv.cols()) = s_uv;


  int dim = d_.dim;
  if (dim == 2) {
    compute_scaffold_gradient_matrix(Dx_s,Dy_s);
  } else {
    Eigen::SparseMatrix<double> G;
    igl::grad(V, F_s, G);

    Dx_s = G.block(0, 0, sf_n, vn);
    Dy_s = G.block(sf_n, 0, sf_n, vn);
    Dz_s = G.block(2 * sf_n, 0, sf_n, vn);
  }

  Dx_s.makeCompressed();
  Dy_s.makeCompressed();
  Dz_s.makeCompressed();

}

void ReWeightedARAP::adjust_scaf_weight(double new_weight) {
  d_.scaffold_factor = new_weight;
  d_.update_scaffold();

//  for (int i = 0; i < d_.dim * d_.dim; i++)
//    M.segment(i * f_n, f_n) = d_.w_M;
}

double ReWeightedARAP::perform_iteration(MatrixXd &w_uv) {
  Eigen::MatrixXd V_out = w_uv;
  solve_weighted_proxy(V_out);
  auto whole_E =
      [this](Eigen::MatrixXd &uv) { return this->compute_energy(uv); };

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
}

double ReWeightedARAP::perform_iteration(Eigen::MatrixXd &w_uv,
                                         bool whole) {
  Eigen::MatrixXd V_out = w_uv;
  solve_weighted_proxy(V_out);

  auto whole_E = [this,whole](Eigen::MatrixXd& uv)
  {return this->compute_energy(uv);};  // whole

  igl::Timer timer;
  timer.start();
  double energy= igl::flip_avoiding_line_search(d_.s_T, w_uv, V_out,
                                                whole_E, -1) / d_.mesh_measure;
//  cout << "LineSearch = " << timer.getElapsedTime()<<endl;
  return energy;
}

double
ReWeightedARAP::compute_soft_constraint_energy(const Eigen::MatrixXd &uv) const {
  double e = 0;
  for (auto const &x:d_.soft_cons)
    e += d_.soft_const_p * (x.second - uv.row(x.first)).squaredNorm();

  return e;
}

void ReWeightedARAP::add_soft_constraints(Eigen::SparseMatrix<double> &L,
                                          Eigen::VectorXd &rhs) const {
  for (int d = 0; d < d_.dim; d++)
    for (auto const &x:d_.soft_cons) {
      int v_idx = x.first;
      rhs(d * v_n + v_idx) += d_.soft_const_p * x.second(d); // rhs
      L.coeffRef(d * v_n + v_idx, d * v_n + v_idx) +=
          d_.soft_const_p; // diagonal of matrix
    }
}

void ReWeightedARAP::mesh_improve() {
    d_.mesh_improve();
    after_mesh_improve();
}

void ReWeightedARAP::after_mesh_improve() {
// differ to pre_calc in the sense of only updating scaffold ones
    mv_n = d_.mv_num;
    mf_n = d_.mf_num;
    sv_n = d_.sv_num;
    sf_n = d_.sf_num;

    v_n = mv_n + sv_n;
    f_n = mf_n + sf_n;
    if (d_.dim == 2) {
      compute_scaffold_gradient_matrix(Dx_s, Dy_s);
    } else {
      Eigen::SparseMatrix<double> Gs;
      igl::grad(d_.w_uv, d_.s_T, Gs);

      Dx_s = Gs.block(0, 0, sf_n, v_n);
      Dy_s = Gs.block(sf_n, 0, sf_n, v_n);
      Dz_s = Gs.block(2 * sf_n, 0, sf_n, v_n);
    }
    int dim = d_.dim;

    Dx_s.makeCompressed();
    Dy_s.makeCompressed();
    Dz_s.makeCompressed();
    Ri_s = MatrixXd::Zero(Dx_s.rows(), dim * dim);
    Ji_s.resize(Dx_s.rows(), dim * dim);
    W_s.resize(Dx_s.rows(), dim * dim);
}

void ReWeightedARAP::adjust_frame(double a, double b) {
  d_.automatic_expand_frame(a,b);
}

void ReWeightedARAP::enlarge_internal_reference(double scale) {
  Dx_m /= scale;
  Dy_m /= scale;
}