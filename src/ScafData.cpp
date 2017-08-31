//
// Created by Zhongshi Jiang on 2/12/17.
//

#include "ScafData.h"
#include <igl/doublearea.h>
#include <iostream>
#include <igl/volume.h>
#include <igl/boundary_facets.h>
#include <igl/Timer.h>
#include <igl/massmatrix.h>
#include <igl/triangle/triangulate.h>
#include "util/tetrahedral_improvement.h"
#include "igl_dev/tetrahedron_tetrahedron_adjacency.h"
#include "util/tetgenio_parser.h"
#include "util/triangle_utils.h"
#include <igl/cat.h>
#include <igl/boundary_loop.h>
#include <igl/edge_flaps.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/flipped_triangles.h>
#include <igl/PI.h>

using namespace std;
using namespace Eigen;

ScafData::ScafData(Eigen::MatrixXd &mesh_V, Eigen::MatrixXi &mesh_F,
                   Eigen::MatrixXd &all_V,
                   Eigen::MatrixXi &scaf_T) :
    m_V(mesh_V), s_T(scaf_T), w_uv(all_V), m_T(mesh_F)
{
//  volumetric = (s_T.cols() == 4);
  mv_num = m_V.rows();
  mf_num = m_T.rows();
  sf_num = s_T.rows();

  dim = w_uv.cols();

  if (dim == 2) {
    igl::doublearea(m_V, m_T, m_M);
    m_M /= 2.;
    igl::doublearea(w_uv, s_T, s_M);
    s_M /= 2.;

    component_sizes.push_back(mf_num);
    surface_F = m_T;
    igl::boundary_loop(m_T, internal_bnd);
    bnd_sizes.push_back(internal_bnd.size());
  } else {
   if(m_T.cols() == 4) {
     igl::volume(w_uv, m_T, m_M);
     igl::boundary_facets(m_T, surface_F);
   } else {
     Eigen::SparseMatrix<double> mass;
     igl::massmatrix(m_V, m_T, igl::MASSMATRIX_TYPE_BARYCENTRIC,mass);
     m_M = mass.diagonal();

     surface_F = m_T;
   }
    igl::volume(w_uv, s_T, s_M);
  }

  mesh_measure = m_M.sum();
//  scaf_volume = s_M.sum();

  update_scaffold();
}

void ScafData::update_scaffold()
{
  mv_num = m_V.rows();
  mf_num = m_T.rows();

  v_num = w_uv.rows();
  sf_num = s_T.rows();

  sv_num = v_num - mv_num;
  f_num = sf_num + mf_num;

//  igl::doublearea(w_uv, s_T, s_M);
//  s_M *= 10*scaffold_factor;
  s_M = Eigen::VectorXd::Constant(sf_num, scaffold_factor);
}

void ScafData::add_soft_constraints(const Eigen::VectorXi &b,
                                    const Eigen::MatrixXd &bc)
{
  assert(b.rows() == bc.rows() && "Constraint input incompatible");
  for(int i=0; i < b.rows(); i++)
    soft_cons[b(i)] = bc.row(i);
}
void ScafData::add_soft_constraints(int b, const Eigen::RowVectorXd &bc) {
  soft_cons[b] = bc;
}


void ScafData::mesh_improve() {
  if (dim == 3) {
    igl::Timer timer;
    timer.start();
    automatic_expand_frame(2,3);
    std::vector<Eigen::RowVector3d> V(w_uv.rows());
    std::vector<Eigen::RowVector4i> T(sf_num);
    for (int i = 0; i < V.size(); i++) {
      V[i] = w_uv.row(i);
    }
    for (int i = 0; i < sf_num; i++) {
      T[i] = s_T.row(i);
    }

    decltype(T) TT, TTif;
    std::vector<Eigen::Matrix<int, 4, 3>> TTie;
    igl::dev::tetrahedron_tetrahedron_adjacency(T, TT, TTif, TTie);


    // https://github.com/janba/DSC/blob/master/is_mesh/util.h#L415
    struct test_utils {
      using vec3 = Eigen::RowVector3d;
      inline static double ms_length(const vec3 &a,
                                     const vec3 &b,
                                     const vec3 &c,
                                     const vec3 &d) {
        double result = 0.;
        result += (a - b).squaredNorm();
        result += (a - c).squaredNorm();
        result += (a - d).squaredNorm();
        result += (b - c).squaredNorm();
        result += (b - d).squaredNorm();
        result += (c - d).squaredNorm();
        return result / 6.;
      }

      inline static double rms_length(const vec3 &a,
                                      const vec3 &b,
                                      const vec3 &c,
                                      const vec3 &d) {
        return sqrt(ms_length(a, b, c, d));
      }

      inline static double signed_volume(const vec3 &a,
                                         const vec3 &b,
                                         const vec3 &c,
                                         const vec3 &d) {
        return (a - d).dot((b - d).cross(c - d)) / 6.;
      }

// https://hal.inria.fr/inria-00518327
      inline static double quality(const vec3 &a,
                                   const vec3 &b,
                                   const vec3 &c,
                                   const vec3 &d) {
        double v = signed_volume(a, b, c, d);
        double lrms = rms_length(a, b, c, d);

        double q = 8.48528 * v / (lrms * lrms * lrms);
#ifdef DEBUG
        assert(!isnan(q));
#endif
        return q;
      }
    };

    auto tet_quality = [&V](int a, int b, int c, int d) -> double {
      return -test_utils::quality(V[a], V[b], V[c], V[d]);
    };
    auto orient3D = [&V](int a, int b, int c, int d) -> bool {
      return -test_utils::signed_volume(V[a], V[b], V[c], V[d]) > 0;
    };

    std::cout << "Prepare Stellar" << timer.getElapsedTime()
              << std::endl;
    timer.start();
    {
      double old_q = INFINITY;
      for (auto r:T)
        old_q = std::min(old_q, tet_quality(r(0), r(1), r(2), r(3)));
      cout << endl << "q" << old_q << '\t';
    }
    combined_improvement_pass(tet_quality, orient3D,
                              [&](int vi) { return vi >= mv_num; },
                              V, T, TT, TTif, TTie);
    std::cout << "Combined Pass" << timer.getElapsedTime()
              << std::endl;
    timer.start();
    {
      double old_q = INFINITY;
      for (auto r:T)
        old_q = std::min(old_q, tet_quality(r(0), r(1), r(2), r(3)));
      cout << "q" << old_q << '\t';
    }
//  face_removal_pass(tet_quality, orient3D, T, TT, TTif, TTie);
    std::cout << "FacePass" << timer.getElapsedTime()
              << std::endl;
    timer.start();
    {
      double old_q = INFINITY;
      for (auto r:T)
        old_q = std::min(old_q, tet_quality(r(0), r(1), r(2), r(3)));
      cout << "q" << old_q << endl;
    }

    sf_num = T.size();
    s_T.conservativeResize(sf_num, 4);
    s_T = Eigen::Map<Eigen::Matrix<int, -1, -1, Eigen::RowMajor>>
        ((int *) &T[0], sf_num, 4);

    v_num = V.size();
    w_uv = Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>
        ((double *) &V[0], v_num, 3);
  } else {

    MatrixXd m_uv = w_uv.topRows(mv_num);
    MatrixXd V_bnd;
    V_bnd.resize(internal_bnd.size(), 2);
    for (int i = 0; i < internal_bnd.size(); i++) // redoing step 1.
    {
      V_bnd.row(i) = m_uv.row(internal_bnd(i));
    }

    if(rect_frame_V.size() == 0) {
      Matrix2d ob;// = rect_corners;
      {
        VectorXd uv_max = m_uv.colwise().maxCoeff();
        VectorXd uv_min = m_uv.colwise().minCoeff();
        VectorXd uv_mid = (uv_max + uv_min) / 2.;

//        double scaf_range = 3;
        Eigen::Array2d scaf_range(3,3);
        ob.row(0) = uv_mid.array() + scaf_range * ((uv_min - uv_mid).array());
        ob.row(1) = uv_mid.array() + scaf_range * ((uv_max - uv_mid).array());
      }
      Vector2d rect_len;
      rect_len << ob(1, 0) - ob(0, 0), ob(1, 1) - ob(0, 1);
      int frame_points = 5;

      if(false) {
        // adjust to square, as in packing
        if (rect_len(0) > rect_len(1)) {
          ob(1, 1) = ob(0, 1) + rect_len(0);
          rect_len(1) = rect_len(0);
        } else {
          ob(0, 1) = ob(0, 0) + rect_len(1);
          rect_len(0) = rect_len(1);
        }
        frame_points = 20;
      }
      
      rect_frame_V.resize(4 * frame_points, 2);
      for (int i = 0; i < frame_points; i++) {
        // 0,0;0,1
        rect_frame_V.row(i) << ob(0, 0), ob(0, 1) + i * rect_len(1) / frame_points;
        // 0,0;1,1
        rect_frame_V.row(i + frame_points)
            << ob(0, 0) + i * rect_len(0) / frame_points, ob(1, 1);
        // 1,0;1,1
        rect_frame_V.row(i + 2 * frame_points) << ob(1, 0), ob(1, 1)
            - i * rect_len(1) /
                frame_points;
        // 1,0;0,1
        rect_frame_V.row(i + 3 * frame_points)
            << ob(1, 0) - i * rect_len(0) / frame_points, ob(0, 1);
        // 0,0;0,1
      }
      frame_ids = Eigen::VectorXi::LinSpaced(rect_frame_V.rows(), mv_num,
      mv_num + rect_frame_V.rows());
    }

    // Concatenate Vert and Edge
    MatrixXd V;
    MatrixXi E;
    igl::cat(1, V_bnd, rect_frame_V, V);
    E.resize(V.rows(), 2);
    for (int i = 0; i < E.rows(); i++)
      E.row(i) << i, i + 1;
    int acc_bs = 0;
    for(auto bs:bnd_sizes) {
      E(acc_bs + bs - 1,1 ) = acc_bs;
      acc_bs += bs;
    }
    E(V.rows() - 1, 1) = acc_bs;
    assert(acc_bs== internal_bnd.size());

    MatrixXd H = MatrixXd::Zero(component_sizes.size(), 2);
    {
      int hole_f = 0;
      int hole_i = 0;
      for (auto cs:component_sizes) {
        for (int i = 0; i < 3; i++)
          H.row(hole_i) += m_uv.row(m_T(hole_f, i)); // redoing step 2
        hole_f += cs;
        hole_i++;
      }
    }
    H /= 3.;

    MatrixXd uv2;
    igl::triangle::triangulate(V, E, H, "qYYQ", uv2, s_T);
    auto bnd_n = internal_bnd.size();

    for (auto i = 0; i < s_T.rows(); i++)
      for (auto j = 0; j < s_T.cols(); j++) {
        auto &x = s_T(i, j);
        if (x < bnd_n) x = internal_bnd(x);
        else x += m_uv.rows() - bnd_n;
      }

    igl::cat(1, m_T, s_T, surface_F);
    w_uv.conservativeResize(m_uv.rows() - bnd_n + uv2.rows(), 2);
    w_uv.bottomRows(uv2.rows() - bnd_n) = uv2.bottomRows(-bnd_n + uv2.rows());
  }

  update_scaffold();
}

void ScafData::automatic_expand_frame(double min2, double max3) {
  // right top
  // left down
  using namespace Eigen;
  MatrixXd m_uv = w_uv.topRows(mv_num);
  MatrixXd frame(2,dim), bbox(2,dim);
  frame << w_uv.colwise().maxCoeff(), w_uv.colwise().minCoeff();
  bbox << m_uv.colwise().maxCoeff(), m_uv.colwise().minCoeff();
  RowVector2d center = bbox.colwise().mean();
/*
  bbox.row(0) -= center;
  bbox.row(1) -= center;
  frame.row(0) -= center;
  frame.row(1) -= center;
*/
  struct line_func {
    double a, b;

    double operator()(double y) { return a * y + b; };
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
  for (auto d=0; d<dim;d++) {
    center_coord = center(d);

    if (frame(0, d) - center_coord < min2 * (bbox(0, d) - center_coord)) {
      new_frame = max3 * (bbox(0, d) - center_coord) + center_coord;
      auto expand = linear_stretch(bbox(0, d), bbox(0, d),
                                   frame(0, d), new_frame);
      for (auto v = mv_num; v < v_num; v++) {
        if (w_uv(v, d) > bbox(0, d))
          w_uv(v, d) = expand(w_uv(v, d));
      }
    }

    if (frame(1, d) - center_coord > min2 * (bbox(1, d) - center_coord)) {
      new_frame = max3 * (bbox(1, d) - center_coord) + center_coord;
      auto expand = linear_stretch(bbox(1, d), bbox(1, d),
                                   frame(1, d), new_frame);
      for (auto v = mv_num; v < v_num; v++) {
        if (w_uv(v, d) < bbox(1, d))
          w_uv(v, d) = expand(w_uv(v, d));
      }
    }
  }
}

void ScafData::add_new_patch(const Eigen::MatrixXd &V_in,
                             const Eigen::MatrixXi &F_ref,
                              const Eigen::RowVectorXd &center) {
  using namespace std;
  using namespace Eigen;

  VectorXd M;
  igl::doublearea(V_in, F_ref, M);

  Eigen::MatrixXd V_ref = V_in;// / sqrt(M.sum()/2/igl::PI);
 // M /= M.sum()/igl::PI;
  Eigen::MatrixXd uv_init;
  Eigen::VectorXi bnd;
  Eigen::MatrixXd bnd_uv;

  std::vector<std::vector<int>> all_bnds;
  igl::boundary_loop(F_ref, all_bnds);
  int num_holes = all_bnds.size() - 1;

  std::sort(all_bnds.begin(), all_bnds.end(), [](auto& a, auto&b){
    return a.size() > b.size();
  });

  bnd =  Map<Eigen::VectorXi>(all_bnds[0].data(),
                              all_bnds[0].size());

  igl::map_vertices_to_circle(V_ref, bnd, bnd_uv);
  bnd_uv *= sqrt(M.sum()/(2*igl::PI));
  bnd_uv.rowwise() += center;
  mesh_measure += M.sum()/2;
  std::cout<<"Mesh Measure"<< M.sum()/2<<std::endl;

  if(num_holes == 0) {

    if (bnd.rows() == V_ref.rows()) {
      std::cout << "All vert on boundary" << std::endl;
      uv_init.resize(V_ref.rows(), 2);
      for (int i = 0; i < bnd.rows(); i++) {
        uv_init.row(bnd(i)) = bnd_uv.row(i);
      }
    } else {
      igl::harmonic(V_ref, F_ref, bnd, bnd_uv, 1, uv_init);

      if (igl::flipped_triangles(uv_init, F_ref).size() != 0) {
        std::cout << "Using Uniform Laplacian" << std::endl;
        igl::harmonic(F_ref, bnd, bnd_uv, 1, uv_init); // use uniform laplacian
      }
    }
  } else {
    auto &F = F_ref;
    auto &V = V_in;
    auto &primary_bnd = bnd;
    // fill holes
    int n_filled_faces = 0;
    int real_F_num = F.rows();
    for (int i = 0; i < num_holes; i++) {
      n_filled_faces += all_bnds[i + 1].size();
    }
    MatrixXi F_filled(n_filled_faces + real_F_num, 3);
    F_filled.topRows(real_F_num) = F;

    int new_vert_id = V.rows();
    int new_face_id = real_F_num;

    for (int i = 0; i < num_holes; i++) {
      int cur_bnd_size = all_bnds[i + 1].size();
      auto it = all_bnds[i + 1].begin();
      auto back = all_bnds[i + 1].end() - 1;
      F_filled.row(new_face_id++) << *it, *back, new_vert_id;
      while (it != back) {
        F_filled.row(new_face_id++)
            << *(it + 1), *(it), new_vert_id;
        it++;
      }
      new_vert_id++;
    }
    assert(new_face_id == F_filled.rows());
    assert(new_vert_id == V.rows() + num_holes);

    igl::harmonic(F_filled, primary_bnd, bnd_uv, 1, uv_init);
    uv_init.conservativeResize(V.rows(), 2);
    if (igl::flipped_triangles(uv_init, F_ref).size() != 0) {
      std::cout<<"Wrong Choice of Outer bnd:"<<std::endl;
//      assert(false&&"Wrong Choice of outer bnd?");
    }
  }

  component_sizes.push_back(F_ref.rows());

  MatrixXd m_uv = w_uv.topRows(mv_num);
  igl::cat(1, m_uv, uv_init, w_uv);
//  mv_num = w_uv.rows();

  m_M.conservativeResize(mf_num + M.size());
  m_M.bottomRows(M.size()) = M/2;

//  internal_bnd.conservativeResize(internal_bnd.size()+ bnd.size());
//  internal_bnd.bottomRows(bnd.size()) = bnd.array() + mv_num;
//  bnd_sizes.push_back(bnd.size());

  for(auto cur_bnd : all_bnds) {
    internal_bnd.conservativeResize(internal_bnd.size()+ cur_bnd.size());
    internal_bnd.bottomRows(cur_bnd.size()) =
        Map<ArrayXi>(cur_bnd.data(),cur_bnd.size()) + mv_num;
    bnd_sizes.push_back(cur_bnd.size());
  }

  m_T.conservativeResize(mf_num + F_ref.rows(), 3);
  m_T.bottomRows(F_ref.rows()) = F_ref.array() + mv_num;
  mf_num += F_ref.rows();

  m_V.conservativeResize(mv_num + V_ref.rows(), 3);
  m_V.bottomRows(V_ref.rows()) = V_ref;
  mv_num += V_ref.rows();

  rect_frame_V = MatrixXd();

  mesh_improve();
}

ScafData::ScafData() {
  dim = 2;
  mv_num = 0;
  mf_num = 0;
  sf_num= 0;
  sv_num = 0;
  mesh_measure = 0;
}