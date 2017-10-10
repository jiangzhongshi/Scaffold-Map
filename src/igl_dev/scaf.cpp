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
      for(int i = 0; i < b.rows(); i++)
        s.soft_cons[b(i)] = bc.row(i);
    }

    void
    add_soft_constraints(igl::SCAFData &s, int b, const Eigen::RowVectorXd &bc)
    {
      s.soft_cons[b] = bc;
    }


    void mesh_improve(igl::SCAFData &s)
    {
      using namespace Eigen;
      MatrixXd m_uv = s.w_uv.topRows(s.mv_num);
      MatrixXd V_bnd;
      V_bnd.resize(s.internal_bnd.size(), 2);
      for(int i = 0; i < s.internal_bnd.size(); i++) // redoing step 1.
      {
        V_bnd.row(i) = m_uv.row(s.internal_bnd(i));
      }

      if(s.rect_frame_V.size() == 0)
      {
        Matrix2d ob;// = rect_corners;
        {
          VectorXd uv_max = m_uv.colwise().maxCoeff();
          VectorXd uv_min = m_uv.colwise().minCoeff();
          VectorXd uv_mid = (uv_max + uv_min) / 2.;

//        double scaf_range = 3;
          Eigen::Array2d scaf_range(3, 3);
          ob.row(0) = uv_mid.array() + scaf_range * ((uv_min - uv_mid).array());
          ob.row(1) = uv_mid.array() + scaf_range * ((uv_max - uv_mid).array());
        }
        Vector2d rect_len;
        rect_len << ob(1, 0) - ob(0, 0), ob(1, 1) - ob(0, 1);
        int frame_points = 5;

        s.rect_frame_V.resize(4 * frame_points, 2);
        for(int i = 0; i < frame_points; i++)
        {
          // 0,0;0,1
          s.rect_frame_V.row(i) << ob(0, 0), ob(0, 1) +
                                           i * rect_len(1) / frame_points;
          // 0,0;1,1
          s.rect_frame_V.row(i + frame_points)
              << ob(0, 0) + i * rect_len(0) / frame_points, ob(1, 1);
          // 1,0;1,1
          s.rect_frame_V.row(i + 2 * frame_points) << ob(1, 0), ob(1, 1)
                                                              -
                                                              i * rect_len(1) /
                                                              frame_points;
          // 1,0;0,1
          s.rect_frame_V.row(i + 3 * frame_points)
              << ob(1, 0) - i * rect_len(0) / frame_points, ob(0, 1);
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
      for(int i = 0; i < E.rows(); i++)
        E.row(i) << i, i + 1;
      int acc_bs = 0;
      for(auto bs:s.bnd_sizes)
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
        for(auto cs:s.component_sizes)
        {
          for(int i = 0; i < 3; i++)
            H.row(hole_i) += m_uv.row(s.m_T(hole_f, i)); // redoing step 2
          hole_f += cs;
          hole_i++;
        }
      }
      H /= 3.;

      MatrixXd uv2;
      igl::triangle::triangulate(V, E, H, "qYYQ", uv2, s.s_T);
      auto bnd_n = s.internal_bnd.size();

      for(auto i = 0; i < s.s_T.rows(); i++)
        for(auto j = 0; j < s.s_T.cols(); j++)
        {
          auto &x = s.s_T(i, j);
          if(x < bnd_n) x = s.internal_bnd(x);
          else x += m_uv.rows() - bnd_n;
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
        { return a * y + b; };
      };

      auto linear_stretch = [](double s0,
                               double t0,
                               double s1,
                               double t1)
      { // source0, target0, source1, target1
        Matrix2d S;
        S << s0, 1, s1, 1;
        Vector2d t;
        t << t0, t1;
        Vector2d coef = S.colPivHouseholderQr().solve(t);
        return line_func{coef[0], coef[1]};
      };

      double new_frame;
      double center_coord;
      for(auto d = 0; d < s.dim; d++)
      {
        center_coord = center(d);

        if(frame(0, d) - center_coord < min2 * (bbox(0, d) - center_coord))
        {
          new_frame = max3 * (bbox(0, d) - center_coord) + center_coord;
          auto expand = linear_stretch(bbox(0, d), bbox(0, d),
                                       frame(0, d), new_frame);
          for(auto v = s.mv_num; v < s.v_num; v++)
          {
            if(s.w_uv(v, d) > bbox(0, d))
              s.w_uv(v, d) = expand(s.w_uv(v, d));
          }
        }

        if(frame(1, d) - center_coord > min2 * (bbox(1, d) - center_coord))
        {
          new_frame = max3 * (bbox(1, d) - center_coord) + center_coord;
          auto expand = linear_stretch(bbox(1, d), bbox(1, d),
                                       frame(1, d), new_frame);
          for(auto v = s.mv_num; v < s.v_num; v++)
          {
            if(s.w_uv(v, d) < bbox(1, d))
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

      Eigen::MatrixXd V_ref = V_in;// / sqrt(M.sum()/2/igl::PI);
      // M /= M.sum()/igl::PI;
      Eigen::MatrixXd uv_init;
      Eigen::VectorXi bnd;
      Eigen::MatrixXd bnd_uv;

      std::vector<std::vector<int>> all_bnds;
      igl::boundary_loop(F_ref, all_bnds);
      int num_holes = all_bnds.size() - 1;

      std::sort(all_bnds.begin(), all_bnds.end(), [](auto &a, auto &b)
      {
        return a.size() > b.size();
      });

      bnd = Map<Eigen::VectorXi>(all_bnds[0].data(),
                                 all_bnds[0].size());

      igl::map_vertices_to_circle(V_ref, bnd, bnd_uv);
      bnd_uv *= sqrt(M.sum() / (2 * igl::PI));
      bnd_uv.rowwise() += center;
      s.mesh_measure += M.sum() / 2;
      std::cout << "Mesh Measure" << M.sum() / 2 << std::endl;

      if(num_holes == 0)
      {

        if(bnd.rows() == V_ref.rows())
        {
          std::cout << "All vert on boundary" << std::endl;
          uv_init.resize(V_ref.rows(), 2);
          for(int i = 0; i < bnd.rows(); i++)
          {
            uv_init.row(bnd(i)) = bnd_uv.row(i);
          }
        }
        else
        {
          igl::harmonic(V_ref, F_ref, bnd, bnd_uv, 1, uv_init);

          if(igl::flipped_triangles(uv_init, F_ref).size() != 0)
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
        for(int i = 0; i < num_holes; i++)
        {
          n_filled_faces += all_bnds[i + 1].size();
        }
        MatrixXi F_filled(n_filled_faces + real_F_num, 3);
        F_filled.topRows(real_F_num) = F;

        int new_vert_id = V.rows();
        int new_face_id = real_F_num;

        for(int i = 0; i < num_holes; i++)
        {
          int cur_bnd_size = all_bnds[i + 1].size();
          auto it = all_bnds[i + 1].begin();
          auto back = all_bnds[i + 1].end() - 1;
          F_filled.row(new_face_id++) << *it, *back, new_vert_id;
          while(it != back)
          {
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
        if(igl::flipped_triangles(uv_init, F_ref).size() != 0)
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

      for(auto cur_bnd : all_bnds)
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
  }

}

IGL_INLINE void igl::scaf_precompute(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &V_init,
    igl::SCAFData &data,
    double soft_p)
{
  igl::scaf::add_new_patch(data, V, F, Eigen::RowVector2d(0,0));
  data.soft_const_p = soft_p;
}

IGL_INLINE Eigen::MatrixXd igl::scaf_solve(SCAFData &data, int iter_num){
  using namespace std;
  using namespace Eigen;
  auto &d_ = data;
  auto &ws = wssolver;
  double last_mesh_energy =  ws->compute_energy(d_.w_uv, false) / d_.mesh_measure;
  std::cout<<"Initial Energy"<<last_mesh_energy<<std::endl;
  cout << "Initial V_num: "<<d_.mv_num<<" F_num: "<<d_.mf_num<<endl;
  d_.energy = ws->compute_energy(d_.w_uv, true) / d_.mesh_measure;
    igl::Timer timer;

    timer.start();

    ws->mesh_improve();

    double new_weight = d_.mesh_measure * last_mesh_energy / (d_.sf_num*100 );
    ws->adjust_scaf_weight(new_weight);

    d_.energy = ws->perform_iteration(d_.w_uv);

    cout<<"Iteration time = "<<timer.getElapsedTime()<<endl;
    double current_mesh_energy =
        ws->compute_energy(d_.w_uv, false) / d_.mesh_measure - 4;
    double mesh_energy_decrease = last_mesh_energy - current_mesh_energy;
    cout << "Energy After:"
         << d_.energy - 4
         << "\tMesh Energy:"
         << current_mesh_energy
         << "\tEnergy Decrease"
         << mesh_energy_decrease
         << endl;
    cout << "V_num: "<<d_.v_num<<" F_num: "<<d_.f_num<<endl;
    last_mesh_energy = current_mesh_energy;
  }

  Eigen::MatrixXd wuv3 = Eigen::MatrixXd::Zero(d_.v_num,3);
  wuv3.leftCols(2) = d_.w_uv;
  

}