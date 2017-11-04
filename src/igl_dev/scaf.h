//
// Created by Zhongshi Jiang on 9/22/17.
//

#ifndef SCAFFOLD_MAP_SCAF_H
#define SCAFFOLD_MAP_SCAF_H

#include <igl/slim.h>
#include <igl/igl_inline.h>

namespace igl
{

    struct SCAFData
    {
      double scaffold_factor = 10;
      igl::SLIMData::SLIM_ENERGY scaf_energy = igl::SLIMData::SYMMETRIC_DIRICHLET;
      igl::SLIMData::SLIM_ENERGY slim_energy = igl::SLIMData::SYMMETRIC_DIRICHLET;

// Output
      int dim = 2;
      double energy; // objective value

      long mv_num, mf_num;
      long sv_num, sf_num;
      long v_num, f_num;
      Eigen::MatrixXd m_V; // input initial mesh V
      Eigen::MatrixXi m_T; // input initial mesh F/T
// INTERNAL
      Eigen::MatrixXd w_uv; // whole domain uv: mesh + free vertices
      Eigen::MatrixXi s_T; // scaffold domain tets: scaffold tets
      Eigen::MatrixXi w_T;

      Eigen::VectorXd m_M; // mesh area or volume
      Eigen::VectorXd s_M; // scaffold area or volume
      Eigen::VectorXd w_M; // area/volume weights for whole
      double mesh_measure; // area or volume
      double avg_edge_length;
      double proximal_p = 0;

      Eigen::VectorXi frame_ids;

      std::map<int, Eigen::RowVectorXd> soft_cons;
      double soft_const_p = 1e4;

    Eigen::VectorXi internal_bnd;
    Eigen::MatrixXd rect_frame_V;
    // multi-chart support
    std::vector<int> component_sizes;
    std::vector<int> bnd_sizes;
      /*
    public:
      SCAFData();
      SCAFData(Eigen::MatrixXd &mesh_V, Eigen::MatrixXi &mesh_F,
          Eigen::MatrixXd &all_V, Eigen::MatrixXi &scaf_T);
      void add_new_patch(const Eigen::MatrixXd&, const Eigen::MatrixXi&,
                         const Eigen::RowVectorXd &center);

      void mesh_improve();
      void automatic_expand_frame(double min=2.0, double max = 3.0);

      void add_soft_constraints(int b,
                                const Eigen::RowVectorXd &bc);
      void add_soft_constraints(const Eigen::VectorXi &b,
                                const Eigen::MatrixXd &bc);
      void update_scaffold();*/
        // reweightedARAP interior variables.
        bool has_pre_calc = false;
        Eigen::SparseMatrix<double> Dx_s, Dy_s, Dz_s;
        Eigen::SparseMatrix<double> Dx_m, Dy_m, Dz_m;
        Eigen::MatrixXd Ri_m, Ji_m, Ri_s, Ji_s;
        Eigen::MatrixXd W_m, W_s;
    };


// Compute necessary information to start using SCAF
// Inputs:
//		V           #V by 3 list of mesh vertex positions
//		F           #F by 3/3 list of mesh faces (triangles/tets)
//    b           list of boundary indices into V
//    bc          #b by dim list of boundary conditions
//    soft_p      Soft penalty factor (can be zero)
//    slim_energy Energy to minimize
    IGL_INLINE void scaf_precompute(
        const Eigen::MatrixXd &V,
        const Eigen::MatrixXi &F,
        const Eigen::MatrixXd &V_init,
        SCAFData &data,
        double soft_p);


// Run iter_num iterations of SCAF
// Outputs:
//    V_o (in SLIMData): #V by dim list of mesh vertex positions
    IGL_INLINE Eigen::MatrixXd scaf_solve(SCAFData &data, int iter_num);

  }

#endif //SCAFFOLD_MAP_SCAF_H
