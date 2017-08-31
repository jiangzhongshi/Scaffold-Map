#include "SchaeferNewtonParametrizer.h"

#include "eigen_stl_utils.h"

#include <igl/cotmatrix_entries.h>
#include <igl/doublearea.h>
#include <iostream>

#include "../ScafData.h"

//#undef NDEBUG
#include <assert.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
//#define NDEBUG

DECLARE_DIFFSCALAR_BASE();

using namespace std;
SchaeferNewtonParametrizer::SchaeferNewtonParametrizer(ScafData &_sd)
		: has_precomputed(false), d_(_sd) {
	// empty
}

void SchaeferNewtonParametrizer::newton_iteration(const Eigen::MatrixXi &F,
												  Eigen::MatrixXd &uv) {
	// flat uv (x1,y1,x2,y2,...)
	cout << "computing grad and hessian" << endl;
	Eigen::VectorXd x; mat2_to_vec(uv,x);
	// compute energy for hessian grad calculation
	//DScalar Fx;// = 0;
	Eigen::VectorXd grad; Eigen::SparseMatrix<double> hessian;
	double energy = compute_energy_gradient_hessian(F,
													uv,
													grad,
													hessian);

	// Grad and hessian finite-diff check (super slow, only perform on very small meshes)
	/*
	bool grad_ok = check_gradient(V,F,x,grad);
	cout << "grad ok = " << grad_ok << endl;
	assert(grad_ok);
	Eigen::MatrixXd dense_hessian(hessian);
	bool hessian_ok = checkHessian(V,F,x,dense_hessian,0);
	cout << "hessian ok = " << hessian_ok << endl;
	assert(hessian_ok);
	*/

	// perform newton iteration
	cout << "performing newton iteration" << endl;


// Modified by Zhongshi at May 11, 2017 to keep boundary
	// The flattening order is different from the rest!!!
#define KEEP_BND
#ifdef KEEP_BND
	const auto& bnd_ids = d_.frame_ids;
	using namespace Eigen;

	auto bnd_n = bnd_ids.size(); assert(bnd_n > 0);
	MatrixXd bnd_pos;
	int dim = 2;
	int v_n = d_.w_uv.rows();
	igl::slice(d_.w_uv, bnd_ids, 1, bnd_pos);

	VectorXi known_ids(bnd_n * dim);
	VectorXi unknown_ids((v_n - bnd_n) * dim);

	{ // get the complement of bnd_ids.
		int assign = 0, i = 0;
		for (int get = 0; i < v_n && get < bnd_ids.size(); i++) {
			if (bnd_ids(get) == i) get++;
			else unknown_ids(2*(assign++)) = 2*i;
		}
		while (i < v_n) unknown_ids(2*(assign++)) = 2*(i++);
		assert(assign + bnd_ids.size() == v_n);
	}

	VectorXd known_pos(bnd_n * dim);
	  for(int i=0; i<bnd_n; i++) {
		  known_ids(i * 2) = 2*bnd_ids(i);
		  known_ids(i * 2 + 1) = 2*bnd_ids(i) + 1;
	  }
	for(int i=0; i<v_n - bnd_n; i++) {
		unknown_ids(i*2 + 1) = unknown_ids(i*2) + 1;
	}

	Eigen::SparseMatrix<double> hessian_unknown;
  	igl::slice(hessian, unknown_ids, unknown_ids, hessian_unknown);

	Eigen::VectorXd grad_unknown;
	igl::slice(grad, unknown_ids, 1, grad_unknown);

	Eigen::SparseLU<Eigen::SparseMatrix<double> > solver;
	solver.compute(hessian_unknown);
	if(solver.info()!=Eigen::Success) {
		cout << "Eigen Failure!" << endl;
		exit(1);
	}
	Eigen::VectorXd res = solver.solve(grad_unknown);
	VectorXd Uc = VectorXd::Zero(2* v_n);
	igl::slice_into(res, unknown_ids.matrix(), 1, Uc);
	x -= Uc;
#else

	Eigen::SparseLU<Eigen::SparseMatrix<double> > solver;
  int n = hessian.rows();
  Eigen::SparseMatrix<double> id(n,n); id.setIdentity();
//	solver.compute(hessian + (1e-5) * id);
  solver.compute(hessian);
	if(solver.info()!=Eigen::Success) {
		cout << "Eigen Failure!" << endl;
        exit(1);
	}
	Eigen::VectorXd res = solver.solve(grad);
	x -= res;
#endif
	// unflatten uv
	vec_to_mat2(x,uv);
}

double SchaeferNewtonParametrizer::evaluate_energy(const Eigen::MatrixXi &F,
												   Eigen::MatrixXd &uv) {
	precompute(F);
	double energy = 0;

	for (int f_idx = 0; f_idx < F.rows(); f_idx++) {
		int v_1 = F(f_idx, 0);
		int v_2 = F(f_idx, 1);
		int v_3 = F(f_idx, 2);

		// compute current triangle squared area
		auto x1 = uv(v_1, 0);
		auto y1 = uv(v_1, 1);
		auto x2 = uv(v_2, 0);
		auto y2 = uv(v_2, 1); //DScalar x0(F(f,0),0); DScalar y0(F(f,0),1);
		auto x3 = uv(v_3, 0);
		auto y3 = uv(v_3, 1); //DScalar x0(F(f,0),0); DScalar y0(F(f,
		// 0),1);

		auto rx = x1 - x3;//uv(F(f,0),0)-uv(F(f,2),0);
		auto sx = x2 - x3;//uv(F(f,1),0)-uv(F(f,2),0);
		auto ry = y1 - y3;//uv(F(f,0),1)-uv(F(f,2),1);
		auto sy = y2 - y3;//uv(F(f,1),1)-uv(F(f,2),1);
		auto dblAd = rx * sy - ry * sx;
		auto uv_sqrt_dbl_area = dblAd * dblAd;

		auto l_part = (1 / (m_dblArea_orig(f_idx)) + (m_dblArea_orig(f_idx)
				/ uv_sqrt_dbl_area)) *
				m_dbl_area_weight(f_idx);

		//DScalar part_1 = (uv.row(v_3)-uv.row(v_1)).squaredNorm() * m_cached_edges_1[f_idx];
		auto part_1 =
				(pow(x3 - x1, 2) + pow(y3 - y1, 2)) * m_cached_edges_1[f_idx];
		//part_1 += (uv.row(v_2)-uv.row(v_1)).squaredNorm()* m_cached_edges_2[f_idx];
		part_1 += (pow(x2 - x1, 2) + pow(y2 - y1, 2)) * m_cached_edges_2[f_idx];
		part_1 /= (2 * m_dblArea_orig(f_idx));

		//DScalar part_2_1 = (uv.row(v_3)-uv.row(v_1)).dot(uv.row(v_2)-uv.row(v_1));
		auto part_2_1 = (x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1);
		double part_2_2 = m_cached_dot_prod[f_idx];
		auto part_2 = -(part_2_1 * part_2_2) / (m_dblArea_orig(f_idx));

		auto r_part = part_1 + part_2;

		energy += l_part * r_part;
	}
	return energy;
}

double SchaeferNewtonParametrizer::compute_energy_gradient_hessian(const Eigen::MatrixXi &F,
																   Eigen::MatrixXd &uv,
																   Eigen::VectorXd &grad,
																   Eigen::SparseMatrix<double> &hessian) {

	// can save some computation time
	hessian.resize(2*d_.v_num,2*d_.v_num);
	hessian.reserve(10*2*d_.v_num);
	std::vector<Eigen::Triplet<double> > IJV;//(10*2*6*d_.v_num);
	IJV.reserve(36*F.rows());
	grad.resize(2*d_.v_num); grad.setZero();

	precompute(F); // precompute if needed
	// uv is arranged by (x1,y1,x2,y2,...)
	double energy = 0;
	for (int i = 0; i < F.rows(); i++) {
		DiffScalarBase::setVariableCount(6); // 3 vertices with 2 rows for each
		auto l_part = compute_face_energy_left_part(F, uv, i);
		auto r_part =
				compute_face_energy_right_part(F, uv, i);

		auto temp = l_part * r_part;
		energy += temp.getValue();

		Eigen::VectorXd local_grad = temp.getGradient();
		for (int v_i = 0; v_i < 3; v_i++) {
			int v_global = F(i,v_i);

			grad(v_global*2) = grad(v_global*2) + local_grad(v_i*2); // x
			grad(v_global*2+1) = grad(v_global*2+1) + local_grad(v_i*2+1); // y
		}

		Eigen::MatrixXd local_hessian = temp.getHessian();
              Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(local_hessian);
              Eigen::MatrixXd D = es.eigenvalues();
              Eigen::MatrixXd U = es.eigenvectors();
              for (int i = 0; i < 6; i++)
                      D(i) = (D(i) < 0) ? 0 : D(i);
              local_hessian = U * D.asDiagonal()* U.inverse();
		for (int v1 = 0; v1 < 6; v1++) {
			for (int v2 = 0; v2 < 6; v2++) {
				int v1_global = F(i,v1/2)*2 + v1%2;
				int v2_global = F(i,v2/2)*2 + v2%2;

				IJV.push_back(Eigen::Triplet<double>(v1_global,v2_global, local_hessian(v1,v2)));
			}
		}
	}
	hessian.setFromTriplets(IJV.begin(),IJV.end());
	return energy;
}

DScalar SchaeferNewtonParametrizer::compute_face_energy_left_part(const
																  Eigen::MatrixXi &F,
																  const Eigen::MatrixXd &uv,
																  int f_idx) {

	int v_1 = F(f_idx,0); int v_2 = F(f_idx,1); int v_3 = F(f_idx,2);

	// compute current triangle squared area
	DScalar x1(0*2,uv(v_1,0)); DScalar y1(0*2+1,uv(v_1,1)); //DScalar x0(F(f,0),0); DScalar y0(F(f,0),1);
	DScalar x2(1*2,uv(v_2,0)); DScalar y2(1*2+1,uv(v_2,1)); //DScalar x0(F(f,0),0); DScalar y0(F(f,0),1);
	DScalar x3(2*2,uv(v_3,0)); DScalar y3(2*2+1,uv(v_3,1)); //DScalar x0(F(f,0),0); DScalar y0(F(f,0),1);

	
    auto rx = x1-x3;//uv(F(f,0),0)-uv(F(f,2),0);
    auto sx = x2-x3;//uv(F(f,1),0)-uv(F(f,2),0);
    auto ry = y1-y3;//uv(F(f,0),1)-uv(F(f,2),1);
    auto sy = y2-y3;//uv(F(f,1),1)-uv(F(f,2),1);
    auto dblAd = rx*sy - ry*sx;
	auto uv_sqrt_dbl_area = dblAd*dblAd;
    

    return (1/(m_dblArea_orig(f_idx)) + (m_dblArea_orig(f_idx)
			/uv_sqrt_dbl_area)) *
			m_dbl_area_weight(f_idx);
}

DScalar SchaeferNewtonParametrizer::compute_face_energy_right_part
		(const Eigen::MatrixXi &F, const Eigen::MatrixXd &uv, int f_idx) {
	int v_1 = F(f_idx,0); int v_2 = F(f_idx,1); int v_3 = F(f_idx,2);

	DScalar x1(0*2,uv(v_1,0)); DScalar y1(0*2+1,uv(v_1,1)); //DScalar x0(F(f,0),0); DScalar y0(F(f,0),1);
	DScalar x2(1*2,uv(v_2,0)); DScalar y2(1*2+1,uv(v_2,1)); //DScalar x0(F(f,0),0); DScalar y0(F(f,0),1);
	DScalar x3(2*2,uv(v_3,0)); DScalar y3(2*2+1,uv(v_3,1)); //DScalar x0(F(f,0),0); DScalar y0(F(f,0),1);

	//DScalar part_1 = (uv.row(v_3)-uv.row(v_1)).squaredNorm() * m_cached_edges_1[f_idx];
	auto part_1 =  ( pow(x3-x1,2) + pow(y3-y1,2) ) * m_cached_edges_1[f_idx];
	//part_1 += (uv.row(v_2)-uv.row(v_1)).squaredNorm()* m_cached_edges_2[f_idx];
	part_1 += ( pow(x2-x1,2) + pow(y2-y1,2) ) * m_cached_edges_2[f_idx];
	part_1 /= (2*m_dblArea_orig(f_idx));

	//DScalar part_2_1 = (uv.row(v_3)-uv.row(v_1)).dot(uv.row(v_2)-uv.row(v_1));
	auto part_2_1 = (x3-x1) * (x2-x1) + (y3-y1) * (y2-y1);
	double part_2_2 = m_cached_dot_prod[f_idx];
	auto part_2 = -(part_2_1 * part_2_2)/ (m_dblArea_orig(f_idx));

	return part_1+part_2;
}

void SchaeferNewtonParametrizer::precompute(const Eigen::MatrixXi &F) {
	using namespace Eigen;
	if (!has_precomputed) {

//    	igl::doublearea(V,F, m_dblArea_orig);
		m_dblArea_orig.resize(d_.f_num);
		m_dblArea_orig.head(d_.mf_num) = d_.m_M*2;

		VectorXd scaf_area;
		igl::doublearea(d_.w_uv, d_.s_T, scaf_area);
		m_dblArea_orig.tail(d_.sf_num) = scaf_area;

		m_dbl_area_weight = m_dblArea_orig;
		m_dbl_area_weight.tail(d_.sf_num) = d_.s_M*2;

		//m_cached_l_energy_per_face.resize(F.rows());
		//m_cached_r_energy_per_face.resize(F.rows());
		assert(F.rows() == d_.f_num);

		m_cached_edges_1.resize(F.rows());
		m_cached_edges_2.resize(F.rows());
		m_cached_dot_prod.resize(F.rows());

      	auto V = d_.m_V;
		for (int f = 0; f < d_.mf_num; f++) {
			int v_1 = F(f, 0);
			int v_2 = F(f, 1);
			int v_3 = F(f, 2);

			m_cached_edges_1[f] = (V.row(v_2) - V.row(v_1)).squaredNorm();
			m_cached_edges_2[f] = (V.row(v_3) - V.row(v_1)).squaredNorm();
			m_cached_dot_prod[f] =
					(V.row(v_3) - V.row(v_1)).dot(V.row(v_2) - V.row(v_1));
		}

		V = d_.w_uv;

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

      std::cout<<"MinBndEdge"<<min_bnd_edge_len<<std::endl;
      double area_threshold = min_bnd_edge_len/4.0;

		for(int f=d_.mf_num; f< d_.f_num; f++) {
			int v_1 = F(f, 0);
			int v_2 = F(f, 1);
			int v_3 = F(f, 2);

			if(m_dblArea_orig(f) <= area_threshold)
			{
				m_dblArea_orig(f) = area_threshold;
				auto dblA = m_dblArea_orig(f);
				double h = sqrt((dblA) / sin(
						M_PI / 3.0));
				Eigen::Vector3d v1, v2, v3;
				v1 << 0, 0, 0;
				v2 << h, 0, 0;
				v3 << h / 2., (sqrt(3) / 2.) * h, 0;

				m_cached_edges_1[f] = (v2 - v1).squaredNorm();
				m_cached_edges_2[f] = (v3 - v1).squaredNorm();
				m_cached_dot_prod[f] =(v3 - v1).dot(v2-v1);
			} else {
				m_cached_edges_1[f] = (V.row(v_2) - V.row(v_1)).squaredNorm();
				m_cached_edges_2[f] = (V.row(v_3) - V.row(v_1)).squaredNorm();
				m_cached_dot_prod[f] =
						(V.row(v_3) - V.row(v_1)).dot(V.row(v_2) - V.row(v_1));
			}
		}

		has_precomputed = true;
	}
}

double energy_value(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& xx) {
	Eigen::MatrixXd mat; vec_to_mat2(xx,mat);
	assert(false && "Seems to be only used in finite verification");
	return 0;
}

void SchaeferNewtonParametrizer::finiteGradient(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
				const Eigen::VectorXd &x, Eigen::VectorXd &grad, int accuracy) {
    // accuracy can be 0, 1, 2, 3

    const double eps = 2.2204e-8;
    const size_t D = x.rows();
    const int idx = (accuracy-3)/2;
    const std::vector< std::vector <double>> coeff =
    { {1, -1}, {1, -8, 8, -1}, {-1, 9, -45, 45, -9, 1}, {3, -32, 168, -672, 672, -168, 32, -3} };
    const std::vector< std::vector <double>> coeff2 =
    { {1, -1}, {-2, -1, 1, 2}, {-3, -2, -1, 1, 2, 3}, {-4, -3, -2, -1, 1, 2, 3, 4} };
    const std::vector<double> dd = {2, 12, 60, 840};

    Eigen::VectorXd finiteDiff(D);
    for (size_t d = 0; d < D; d++) {
      finiteDiff[d] = 0;
      for (int s = 0; s < 2*(accuracy+1); ++s)
      {
        Eigen::VectorXd xx = x.eval();
        xx[d] += coeff2[accuracy][s]*eps;
        
        finiteDiff[d] += coeff[accuracy][s]*energy_value(V,F,xx);
      }
      finiteDiff[d] /= (dd[accuracy]* eps);
    }
    grad = finiteDiff;
  }

bool SchaeferNewtonParametrizer::check_gradient(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
				const Eigen::VectorXd& x, const Eigen::VectorXd& actual_grad, int accuracy) {
   	const int D = x.rows();
    Eigen::VectorXd expected_grad(D);
    cout << "computing finite gradient" << endl;
    finiteGradient(V,F,x, expected_grad, accuracy);
    cout << "done computing finite gradient" << endl;

    bool correct = true;

    for (int d = 0; d < D; ++d) {
      double scale = std::max(std::max(fabs(actual_grad[d]), fabs(expected_grad[d])), 1.);
      if(fabs(actual_grad[d]-expected_grad[d])>1e-2 * scale)
        correct = false;
    	break;
    }
    return correct;
}

 void SchaeferNewtonParametrizer::finiteHessian(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
 			const Eigen::VectorXd & x, Eigen::MatrixXd & hessian, int accuracy) {
    const double eps = 2.2204e-08;
    const size_t DIM = x.rows();

    if(accuracy == 0) {
      for (size_t i = 0; i < DIM; i++) {
        for (size_t j = 0; j < DIM; j++) {
          
          Eigen::VectorXd xx = x;

          xx[i] += eps; xx[j] += eps;
          double f1 = energy_value(V,F,xx);
          xx[i] -= eps; xx[j] -= eps;
          
          xx[i] += eps;
          double f2 = energy_value(V,F,xx);
          xx[i] -= eps;

          xx[j] += eps;
          double f3 = energy_value(V,F,xx);
          xx[j] -= eps;

          
          double f4 = energy_value(V,F,xx);

          hessian(i, j) = (f1 - f2 - f3 + f4) / (eps * eps);
        }
      }
    } else {
      Eigen::VectorXd xx;
      for (size_t i = 0; i < DIM; i++) {
        for (size_t j = 0; j < DIM; j++) {

          double term_1 = 0;
          xx = x.eval(); xx[i] += 1*eps;  xx[j] += -2*eps;  term_1 += energy_value(V,F,xx);
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += -1*eps;  term_1 += energy_value(V,F,xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += 1*eps;   term_1 += energy_value(V,F,xx);
          xx = x.eval(); xx[i] += -1*eps; xx[j] += 2*eps;   term_1 += energy_value(V,F,xx);

          double term_2 = 0;
          xx = x.eval(); xx[i] += -1*eps; xx[j] += -2*eps;  term_2 += energy_value(V,F,xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += -1*eps;  term_2 += energy_value(V,F,xx);
          xx = x.eval(); xx[i] += 1*eps;  xx[j] += 2*eps;   term_2 += energy_value(V,F,xx);
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += 1*eps;   term_2 += energy_value(V,F,xx);

          double term_3 = 0;
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += -2*eps;  term_3 += energy_value(V,F,xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += 2*eps;   term_3 += energy_value(V,F,xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += -2*eps;  term_3 -= energy_value(V,F,xx);
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += 2*eps;   term_3 -= energy_value(V,F,xx);

          double term_4 = 0;
          xx = x.eval(); xx[i] += -1*eps; xx[j] += -1*eps;  term_4 += energy_value(V,F,xx);
          xx = x.eval(); xx[i] += 1*eps;  xx[j] += 1*eps;   term_4 += energy_value(V,F,xx);
          xx = x.eval(); xx[i] += 1*eps;  xx[j] += -1*eps;  term_4 -= energy_value(V,F,xx);
          xx = x.eval(); xx[i] += -1*eps; xx[j] += 1*eps;   term_4 -= energy_value(V,F,xx);

          hessian(i, j) = (-63 * term_1+63 * term_2+44 * term_3+74 * term_4)/(600.0 * eps * eps);

        }
      }
    }

  }

  void SchaeferNewtonParametrizer::get_gradient(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                  Eigen::VectorXd& uv, Eigen::VectorXd& grad) {
  	 Eigen::SparseMatrix<double> hessian;
  	 Eigen::MatrixXd uv_mat; vec_to_mat2(uv,uv_mat);
	  compute_energy_gradient_hessian(F,
									  uv_mat,
									  grad,
									  hessian);
  }

  void SchaeferNewtonParametrizer::finiteHessian_with_grad(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
  					const Eigen::VectorXd& x,
 					Eigen::MatrixXd & hessian, int accuracy) {
  	int var_num = V.rows()*2;
  	const double eps = 2.2204e-08;
  	hessian.resize(var_num,var_num);
  	
  	for (int i = 0; i < var_num; i++) {
  		for (int j = 0; j < var_num; j++) {
  			Eigen::VectorXd new_x = x;
  			
  			Eigen::VectorXd grad_cur; get_gradient(V,F,new_x, grad_cur);
  			double gi_x = grad_cur(i); double gj_x = grad_cur(j);

  			new_x(j) = new_x(j) + eps;
  			Eigen::VectorXd grad_i;  get_gradient(V,F,new_x, grad_i);
  			double gradj_plus_i = grad_i(i);
  			new_x(j) = new_x(j) - eps;

  			new_x(i) = new_x(i) + eps;
  			Eigen::VectorXd grad_j;  get_gradient(V,F,new_x, grad_j);
  			double gradi_plus_j = grad_j(j);
  			new_x(j) = new_x(j) - eps;

  			hessian(i,j) =  (gradj_plus_i - gj_x)/(2*eps) + (gradi_plus_j - gi_x)/(2*eps);
  		}
  	}
  }


bool SchaeferNewtonParametrizer::checkHessian(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
					const Eigen::VectorXd & x, const Eigen::MatrixXd& actual_hessian, int accuracy) {
    // TODO: check if derived class exists:
    // int(typeid(&Rosenbrock<double>::gradient) == typeid(&Problem<double>::gradient)) == 1 --> overwritten
    const int D = x.rows();
    bool correct = true;

    Eigen::MatrixXd expected_hessian = Eigen::MatrixXd::Zero(D, D);
    //finiteHessian(V,F,x, expected_hessian, accuracy);
    finiteHessian_with_grad(V,F,x, expected_hessian, accuracy);

    for (int d = 0; d < D; ++d) {
      for (int e = 0; e < D; ++e) {
        double scale = std::max(std::max(fabs(actual_hessian(d, e)), fabs(expected_hessian(d, e))), 1.);
        if(fabs(actual_hessian(d, e)- expected_hessian(d, e))>1e-1 * scale) {
        		cout << "not correct for d = " << d << " and e = " << e << endl;
        		correct = false;
        	}
      }
    }
    return correct;

  }
