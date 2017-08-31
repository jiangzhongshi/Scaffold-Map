// Created by Michael Rabinovich
// https://github.com/MichaelRabinovich/Scalable-Locally-Injective-Mappings/
//
// Modified by Zhongshi Jiang

#include "SequentialLBFGS.h"
#include "../ReWeightedARAP.h"
#include "eigen_stl_utils.h"

#include "igl/boundary_loop.h"
#include <igl/cotmatrix_entries.h>
#include <igl/doublearea.h>
#include "igl/harmonic.h"
#include "igl/map_vertices_to_circle.h"
#include <igl/cat.h>

#include <queue>
#include <assert.h>
#include "../util/triangle_utils.h"

SequentialLBFGS::SequentialLBFGS(ScafData &data)
    : m_iter(0),
      v_n(data.v_num),
      f_n(data.f_num),
      mv_n(data.mv_num),
      mf_n(data.mf_num),
      m_data(data)
{
  cur_energy = INFINITY;
  has_pre_calc = false;
  has_converged = false;
  m_arap_iter = 0;

}


void SequentialLBFGS::parametrize(Eigen::MatrixXd &uv)
{

  // currently empty (only support "one step" interface)
}

double
SequentialLBFGS::parametrize_LBFGS(Eigen::MatrixXd &uv, int max_iter, int dummy)
{


  // init
  pre_calc();
  cur_energy = compute_energy(uv, true);
  double old_energy = INFINITY;

  const int m = 10;

  std::vector<Eigen::MatrixXd> s(m), y(m);
  std::vector<double> alpha(m);
  Eigen::MatrixXd q;
  Eigen::MatrixXd grad(uv.rows(), 2);
  compute_negative_gradient(uv, grad);
  grad = -1 * grad;
  Eigen::MatrixXd uv_old, grad_old;
  double H0k = 1; // start with a gradient descent

  int m_iter = 0;
  int overall_iters = 0;
  do
  {
    const double relativeEpsilon = 0.0001 * max(1.0, uv.norm());
    q = grad;
    const int k = min(m, m_iter);

    for(int i = k - 1; i >= 0; i--)
    {
      // cwise product + sum is just like dot product if the matrices were flattened
      double rho = 1.0 / ((s[i].cwiseProduct(y[i])).sum());
      alpha[i] = rho * (s[i].cwiseProduct(q)).sum();
      q = q - alpha[i] * y[i];
    }
    q = H0k * q;
    for(int i = 0; i < k; i++)
    {
      double rho = 1.0 / ((s[i].cwiseProduct(y[i])).sum());
      double beta = rho * (y[i].cwiseProduct(q)).sum();
      q = q + s[i] * (alpha[i] - beta);
    }
    // now q = Hk*grad(fk)

    // make sure we are still in a descent direction
    double descent = grad.cwiseProduct(q).sum();
    double alpha_init = 1;//1.0/grad.norm();
    if(descent < 0.0001 * relativeEpsilon)
    {
      cout << "lost direction!" << endl;
      q = grad;
      m_iter = 0;
      alpha_init = 1.0;
    }

    // minimize with the search direction
    q = -1 * q; // descent direction
    //q = q/q.norm();
    uv_old = uv;
    grad_old = grad;
    zero_out_const_vertices_search_direction(q);
    double min_step_to_singularity = compute_max_step_from_singularities(uv, q);
    //cout << "min step to singularity = " << min_step_to_singularity << endl; //int t; cin >> t;
    double max_step_size = min(1., min_step_to_singularity * 0.9);
    //cur_energy = LineSearch_patwie_armijo_imp(V,F,uv,grad,q,max_step_size);
    old_energy = cur_energy;
    cur_energy = LineSearch_michael_armijo_imp(uv, q, max_step_size);

    // compute new gradient,s and y
    Eigen::MatrixXd grad_old = grad;
    compute_negative_gradient(uv, grad);
    grad = -1 * grad;


    Eigen::MatrixXd s_new = uv - uv_old;
    Eigen::MatrixXd y_new = grad - grad_old;

    // update the history
    if(m_iter < m)
    {
      s[m_iter] = s_new;
      y[m_iter] = y_new;
    }
    else
    {
      s.erase(s.begin());
      s.push_back(s_new);

      y.erase(y.begin());
      y.push_back(y_new);
    }
    // update the scaling factor
    H0k = y_new.cwiseProduct(s_new).sum() /
          static_cast<double>(y_new.cwiseProduct(y_new).sum());

    m_iter++;
    overall_iters++;

  } while((grad.norm() > 1.0e-3) && (overall_iters < max_iter) &&
          ((old_energy - cur_energy) > 1.0e-8));

  cout << "LBFGS_v1 iter = "
       << overall_iters << endl;
  if(overall_iters < max_iter)
  {
    has_converged = true;
  }
  return cur_energy;
}

double SequentialLBFGS::parametrize_LBFGS(Eigen::MatrixXd &uv, int max_iter)
{

  cout << "init p lbfgs vercot" << endl;
  // init
  pre_calc();
  cur_energy = compute_energy(uv, true);

  Eigen::VectorXd x;
  mat2_to_vec(uv, x);
  cout << "x.rows = " << x.rows() << endl;
  const size_t m = 10;
  const size_t DIM = x.rows();

  Eigen::MatrixXd sVector = Eigen::MatrixXd::Zero(DIM, m);
  Eigen::MatrixXd yVector = Eigen::MatrixXd::Zero(DIM, m);

  Eigen::VectorXd alpha = Eigen::VectorXd::Zero(m);
  Eigen::VectorXd grad(DIM), q(DIM), grad_old(DIM), s(DIM), y(DIM);
  compute_negative_gradient(x, grad);
  grad = -1 * grad;
  //cout << "got negative grad: " << endl << grad << endl;
  Eigen::VectorXd x_old = x;
  Eigen::VectorXd x_old2 = x;

  double x_start = compute_energy(x, true), x_end;

  cout << "current energy is " << x_start << endl;
  size_t iter = 0, j = 0;

  double H0k = 1;


  double oscillate_diff = 0, oscillate_diff2 = 0;

  do
  {

    const double relativeEpsilon = 0.0001 * max(1.0, x.norm());

    if(grad.norm() < relativeEpsilon)
      break;

    //Algorithm 7.4 (L-BFGS two-loop recursion)
    q = grad;
    const int k = min(m, iter);

    // for i k − 1, k − 2, . . . , k − m
    for(int i = k - 1; i >= 0; i--)
    {
      // alpha_i <- rho_i*s_i^T*q
      const double rho = 1.0 / static_cast<Eigen::VectorXd>(sVector.col(i)).dot(
          static_cast<Eigen::VectorXd>(yVector.col(i)));
      alpha(i) = rho * static_cast<Eigen::VectorXd>(sVector.col(i)).dot(q);
      // q <- q - alpha_i*y_i
      q = q - alpha(i) * yVector.col(i);
    }
    // r <- H_k^0*q
    q = H0k * q;
    //for i k − m, k − m + 1, . . . , k − 1
    for(int i = 0; i < k; i++)
    {
      // beta <- rho_i * y_i^T * r
      const double rho = 1.0 / static_cast<Eigen::VectorXd>(sVector.col(i)).dot(
          static_cast<Eigen::VectorXd>(yVector.col(i)));
      const double beta =
          rho * static_cast<Eigen::VectorXd>(yVector.col(i)).dot(q);
      // r <- r + s_i * ( alpha_i - beta)
      q = q + sVector.col(i) * (alpha(i) - beta);
    }
    // stop with result "H_k*f_f'=q"

    // any issues with the descent direction ?
    double descent = grad.dot(q);
    double alpha_init = 1.0 / grad.norm();
    if(descent < 0.0001 * relativeEpsilon)
    {
      cout << "hopa!" << endl; //int blabla; cin >> blabla;
      q = grad;
      iter = 0;
      alpha_init = 1.0;
    }

    // find steplength
    //const double rate = WolfeRule::linesearch(x, -q,  FunctionValue, FunctionGradient, alpha_init) ;
    //x = x - rate * q;

    // update guess
    //q=q/q.norm(); //TODO: remove me!
    x_old = x;
    Eigen::MatrixXd x_mat;
    vec_to_mat2(x, x_mat);
    Eigen::MatrixXd grad_m;
    vec_to_mat2(grad, grad_m);
    Eigen::MatrixXd q_m;
    vec_to_mat2(q, q_m);
    q_m = -q_m;
    zero_out_const_vertices_search_direction(q_m);
    double min_step_to_singularity = compute_max_step_from_singularities(x_mat,
                                                                         q_m);
    //cout << "q norm = " << q.norm() << endl;
    //cout << "min point to singularity = " << min_step_to_singularity << endl;
    double max_step_size = min(1., min_step_to_singularity);
    //double max_step_size = min_step_to_singularity;
    /*
    if (min_step_to_singularity > 1) {
        //cout << "grad = " << endl << grad << endl; int temp2; cin >> temp2;
        int temp2; cin >> temp2;
    }
    */
    //cur_energy = LineSearch_patwie_armijo_imp(V,F,x_mat,grad_m,q_m,max_step_size);
    cur_energy = LineSearch_michael_armijo_imp(x_mat, q_m, max_step_size);

    //assert(count_flips(V,F, x_mat) == 0);

    // update x
    mat2_to_vec(x_mat, x);


    grad_old = grad;
    compute_negative_gradient(x, grad);
    grad = -1 * grad;

    s = x - x_old;
    y = grad - grad_old;

    // update the history
    if(iter < m)
    {
      sVector.col(iter) = s;
      yVector.col(iter) = y;
    }
    else
    {

      sVector.leftCols(m - 1) = sVector.rightCols(m - 1).eval();
      sVector.rightCols(1) = s;
      yVector.leftCols(m - 1) = yVector.rightCols(m - 1).eval();
      yVector.rightCols(1) = y;
    }
    // update the scaling factor
    H0k = y.dot(s) / static_cast<double>(y.dot(y));

    /*
    // now the ugly part : detect convergence
    // observation: L-BFGS seems to oscillate
    x_start = FunctionValue(x_old);
    x_old2 = x_old;
    x_old = x;
    x_end = FunctionValue(x);

    oscillate_diff2 = oscillate_diff;
    oscillate_diff = static_cast<Vector>(x_old2-x).norm();*/

    iter++;
    j++;
    /*
    if(fabs(oscillate_diff-oscillate_diff2)<1.0e-7)
      break;
    */
  } while((grad.norm() > 1.0e-3) && (j < max_iter));

  vec_to_mat2(x, uv);
  cout << endl << endl << endl << "----ver2: parametrize_LBFGS iter = " << j
       << endl;
  return cur_energy;

}

double
SequentialLBFGS::one_step(Eigen::MatrixXd &uv)
{
  double new_energy;
  if(m_iter == 0)
  {
    // compute things we'll need in every gradient descent (cotentries, areas, etc)
    pre_calc();
    cur_energy = compute_energy(uv, true);
    m_iter++;
  }
  else
  {
    cout << "single gradient descent: cur energy before = " << cur_energy
         << endl;
    double new_energy = single_gradient_descent(uv);
    cout << "single gradient descent: cur energy after = " << new_energy
         << endl;
    update_results(new_energy);
  }

  return cur_energy;
}

void SequentialLBFGS::update_results(double new_energy)
{
  if(new_energy < cur_energy)
  {
    cout << "Energy is lower : " << new_energy << endl;
    cur_energy = new_energy;
    m_iter++;
  }
  else
  {
    cout << "Error: could not lower energy!" << endl;
  }
}

double SequentialLBFGS::single_gradient_descent(Eigen::MatrixXd &uv)
{
  Eigen::MatrixXd d(uv.rows(), 2);
  compute_negative_gradient(uv, d);

  double min_step_to_singularity = compute_max_step_from_singularities(uv, d);
  double max_step_size = min(1., min_step_to_singularity * 0.8);

  //return LineSearch_michael_armijo_imp(V,F,uv,d,max_step_size);
  return LineSearch_patwie_armijo_imp(uv, -d, d,
                                      max_step_size);
}

double
SequentialLBFGS::LineSearch_patwie_armijo_imp(Eigen::MatrixXd &uv, const Eigen::MatrixXd &grad,
                                                           const Eigen::MatrixXd &p, double step_size)
{
  const double c = 0.2;
  const double rho = 0.7;
  int cur_iter = 0;
  int MAX_STEP_SIZE_ITER = 60;

  Eigen::MatrixXd new_uv = uv + step_size * p;
  double f = compute_energy(new_uv, true);
  const double f_in = cur_energy;
  // tiny hack cause we work with matrices and not vectors
  const double Cache =
      c * (grad.col(0).dot(p.col(0)) + grad.col(1).dot(p.col(1)));
  while((f > f_in + step_size * Cache) && (cur_iter < MAX_STEP_SIZE_ITER))
  {
    step_size *= rho;
    new_uv = uv + step_size * p;
    f = compute_energy(new_uv, true);
    cur_iter++;
  }
  if(cur_iter < MAX_STEP_SIZE_ITER)
  {
    uv = new_uv;
    cout << "lowered at iter = " << cur_iter << " and step = " << step_size
         << endl;
    cout << "previous energy = " << f_in << " and new = " << f << endl;
  }
  else
  {
    cout << "could not lower energy!" << endl;
  }
  return f;
}

double
SequentialLBFGS::LineSearch_michael_armijo_imp(Eigen::MatrixXd &uv,
                                                            const Eigen::MatrixXd &d,
                                                            double step_size)
{

  double old_energy = cur_energy;
  double new_energy = old_energy;
  int cur_iter = 0;
  int MAX_STEP_SIZE_ITER = 60;

  //cout << "current step size = " << step_size << endl;
  while(new_energy >= old_energy && cur_iter < MAX_STEP_SIZE_ITER)
  {
    Eigen::MatrixXd new_uv = uv + step_size * d;

    new_energy = compute_energy(new_uv, true);
    //cout << "new energy = " << new_energy << " and old energy = " << old_energy << endl;
    if(new_energy >= old_energy)
    {
      step_size /= 2;
      //cout << "new step size of " << step_size << endl;
    }
    else
    {
      uv = new_uv;
    }
    cur_iter++;
  }
  return new_energy;
}

void SequentialLBFGS::compute_negative_gradient(
    const Eigen::MatrixXd &uv, Eigen::MatrixXd &neg_grad)
{

  //cout << "computing gradient the old way" << endl;
  auto& F = m_data.s_T;
  Eigen::VectorXd dblArea_p;
  igl::doublearea(uv, F, dblArea_p);
  //cout << "V area sum = " << m_dbl_area.sum() << " and uv area sum = " << dblArea_p.sum() << endl;

  auto scaf_w = m_data.scaffold_factor;
  neg_grad.setZero();
  /* for DBG
  Eigen::MatrixXd left_grad(neg_grad.rows(),neg_grad.cols()); left_grad.setZero();
  Eigen::MatrixXd right_grad(neg_grad.rows(),neg_grad.cols()); right_grad.setZero(); */
  for(int i = 0; i < f_n; i++)
  {
    // add to the vertices gradient

    double energy_left_part = m_cached_l_energy_per_face[i];
    double energy_right_part = m_cached_r_energy_per_face[i];

    double t_orig_area = m_dbl_area(i) / 2;
    double t_uv_area = dblArea_p(i) / 2;
    double left_grad_const = -pow(t_orig_area, 2) / pow(t_uv_area, 3);

    for(int j = 0; j < 3; j++)
    {
      int v1 = j;
      int v2 = (j + 1) % 3;
      int v3 = (j + 2) % 3;
      int v1_i = F(i, j);
      int v2_i = F(i, (j + 1) % 3);
      int v3_i = F(i, (j + 2) % 3);
      // compute left gradient
      Eigen::RowVector2d c_left_grad;
      Eigen::RowVector2d rotated_left_grad = (left_grad_const *
                                              (uv.row(v2_i) - uv.row(v3_i)));
      c_left_grad(0) = rotated_left_grad(1);
      c_left_grad(1) = -rotated_left_grad(0);

      // compute right gradient
      Eigen::RowVector2d c_right_grad;
      //− cot(θ2)U3 − cot(θ3)U2 + (cot(θ2) + cot(θ3))U1
      // note: the entries for this function are half of the contangents

      c_right_grad = -m_cot_entries(i, v2) * uv.row(v3_i) -
                     m_cot_entries(i, v3) * uv.row(v2_i)
                     + (m_cot_entries(i, v2) + m_cot_entries(i, v3)) *
                       uv.row(v1_i);

      // product rule (and subtract from vector cause we compute the negative of the gradient)
      neg_grad.row(v1_i) = neg_grad.row(v1_i) -
                           (c_left_grad * energy_right_part +
                            c_right_grad * energy_left_part)
                           * (i < mf_n ? 1 : 2 * scaf_w / m_dbl_area(i) );

      /* for DBG
      //left_grad.row(v1_i) = left_grad.row(v1_i) + c_left_grad;
      //right_grad.row(v1_i) = right_grad.row(v1_i) + c_right_grad;*/
    }
  }

  //zero_out_const_vertices_search_direction(neg_grad);
}

double SequentialLBFGS::compute_max_step_from_singularities(
    const Eigen::MatrixXd &uv, Eigen::MatrixXd &d)
{
  double max_step = INFINITY;

  for(int f = 0; f < f_n; f++)
  {
    double min_positive_root = get_min_pos_root(uv, d, f);

    if(max_step > min_positive_root)
    {
      //cout << "min pos root at f = " << f << " is " << min_positive_root << endl;
    }

    max_step = min(max_step, min_positive_root);
  }
  return max_step;
}

double SequentialLBFGS::get_min_pos_root(const Eigen::MatrixXd &uv,
                                                      Eigen::MatrixXd &d,
                                                      int f)
{
  /*
  Symbolic matlab for equation 4 at the paper (this is how to recreate the formulas below)
  U11 = sym('U11');
  U12 = sym('U12');
  U21 = sym('U21');
  U22 = sym('U22');
  U31 = sym('U31');
  U32 = sym('U32');

  V11 = sym('V11');
  V12 = sym('V12');
  V21 = sym('V21');
  V22 = sym('V22');
  V31 = sym('V31');
  V32 = sym('V32');

  t = sym('t');

  U1 = [U11,U12];
  U2 = [U21,U22];
  U3 = [U31,U32];

  V1 = [V11,V12];
  V2 = [V21,V22];
  V3 = [V31,V32];

  A = [(U2+V2*t) - (U1+ V1*t)];
  B = [(U3+V3*t) - (U1+ V1*t)];
  C = [A;B];

  solve(det(C), t);
  cf = coeffs(det(C),t); % Now cf(1),cf(2),cf(3) holds the coefficients for the polynom
*/

  auto& F = m_data.s_T;
  int v1 = F(f, 0);
  int v2 = F(f, 1);
  int v3 = F(f, 2);
  // get quadratic coefficients (ax^2 + b^x + c)
#define U11 uv(v1,0)
#define U12 uv(v1,1)
#define U21 uv(v2,0)
#define U22 uv(v2,1)
#define U31 uv(v3,0)
#define U32 uv(v3,1)

#define V11 d(v1,0)
#define V12 d(v1,1)
#define V21 d(v2,0)
#define V22 d(v2,1)
#define V31 d(v3,0)
#define V32 d(v3,1)


  double a =
      V11 * V22 - V12 * V21 - V11 * V32 + V12 * V31 + V21 * V32 - V22 * V31;
  double b =
      U11 * V22 - U12 * V21 - U21 * V12 + U22 * V11 - U11 * V32 + U12 * V31 +
      U31 * V12 - U32 * V11 + U21 * V32 - U22 * V31 - U31 * V22 + U32 * V21;
  double c =
      U11 * U22 - U12 * U21 - U11 * U32 + U12 * U31 + U21 * U32 - U22 * U31;


  double delta_in = pow(b, 2) - 4 * a * c;
  if(delta_in < 0)
  {
    return INFINITY;
  }
  double delta = sqrt(delta_in);
  double t1 = (-b + delta) / (2 * a);
  double t2 = (-b - delta) / (2 * a);

  double tmp_n = min(t1, t2);
  t1 = max(t1, t2);
  t2 = tmp_n;
  // return the smallest negative root if it exists, otherwise return infinity
  if(t1 > 0)
  {
    if(t2 > 0)
    {
      return t2;
    }
    else
    {
      return t1;
    }
  }
  else
  {
    return INFINITY;
  }
}

double
SequentialLBFGS::compute_energy(const Eigen::MatrixXd &uv, bool whole)
{
  pre_calc(); // in case we need precomputation
  double energy = 0;
  //cout << "normal compute energy!" << endl;
  long loop_f_n = whole ? f_n : mf_n;
  auto scaf_w = m_data.scaffold_factor;
  for(int i = 0; i < loop_f_n; i++)
  {
    double l_part = compute_face_energy_left_part(uv, i);
    double r_part = compute_face_energy_right_part(uv, i, m_dbl_area(i));

    energy += l_part * r_part * (i < mf_n ? 1 : 2 * scaf_w/m_dbl_area(i) );

    if(whole)
    {
      // cache results for the gradient use
      m_cached_l_energy_per_face[i] = l_part;
      m_cached_r_energy_per_face[i] = r_part;
    }

  }
  return energy;
}

double
SequentialLBFGS::compute_face_energy_left_part(
    const Eigen::MatrixXd &uv, int f)
{
  // compute current triangle squared area
  auto& F = m_data.s_T;
  auto rx = uv(F(f, 0), 0) - uv(F(f, 2), 0);
  auto sx = uv(F(f, 1), 0) - uv(F(f, 2), 0);
  auto ry = uv(F(f, 0), 1) - uv(F(f, 2), 1);
  auto sy = uv(F(f, 1), 1) - uv(F(f, 2), 1);
  double dblAd = rx * sy - ry * sx;
  double uv_sq_dbl_area = dblAd * dblAd;


  return (1 + (m_sq_dbl_area(f) / uv_sq_dbl_area));
}

double
SequentialLBFGS::compute_face_energy_right_part(
    const Eigen::MatrixXd &uv, int f_idx, double orig_t_dbl_area)
{
  auto& F = m_data.s_T;
  int v_1 = F(f_idx, 0);
  int v_2 = F(f_idx, 1);
  int v_3 = F(f_idx, 2);

  double part_1 =
      (uv.row(v_3) - uv.row(v_1)).squaredNorm() * m_cached_edges_1[f_idx];
  part_1 += (uv.row(v_2) - uv.row(v_1)).squaredNorm() * m_cached_edges_2[f_idx];
  part_1 /= (2 * orig_t_dbl_area);

  double part_2_1 = (uv.row(v_3) - uv.row(v_1)).dot(uv.row(v_2) - uv.row(v_1));
  double part_2_2 = m_cached_dot_prod[f_idx];
  double part_2 = -(part_2_1 * part_2_2) / (orig_t_dbl_area);

  return part_1 + part_2;
}

bool
SequentialLBFGS::check_grad(const Eigen::MatrixXd &uv, int v_idx,
                                         Eigen::RowVector2d grad,
                                         bool is_left_grad)
{
  bool grad_ok = true;
  //double h = 0.001;
  double h = 0.00001;

  if(is_left_grad)
  {
    cout << endl << endl << "-------computing left grad----" << endl;
  }
  else
  {
    cout << endl << endl << "-------computing right grad----" << endl;
  }

  double cur_energy = compute_face_energy_part(uv, is_left_grad);
  Eigen::MatrixXd uv_cpy = uv;
  //cout << "current energy = " << cur_energy << endl;

  Eigen::RowVector2d finite_diff_grad;

  uv_cpy(v_idx, 0) = uv(v_idx, 0) + h;
  double forward_x = compute_face_energy_part(uv_cpy, is_left_grad);
  //cout << "forward_x energy = " << forward_x << endl;

  uv_cpy(v_idx, 0) = uv(v_idx, 0) - h;
  double back_x = compute_face_energy_part(uv_cpy, is_left_grad);
  //cout << "back_x energy = " << back_x << endl;

  finite_diff_grad(0) = (forward_x - back_x) / (2 * h);

  uv_cpy.row(v_idx) = uv.row(v_idx);

  uv_cpy(v_idx, 1) = uv(v_idx, 1) + h;
  double forward_y = compute_face_energy_part(uv_cpy, is_left_grad);
  //cout << "forward_y energy = " << forward_y << endl;
  uv_cpy(v_idx, 1) = uv(v_idx, 1) - h;
  double back_y = compute_face_energy_part(uv_cpy, is_left_grad);
  finite_diff_grad(1) = (forward_y - back_y) / (2 * h);
  //cout << "back_y energy = " << back_y << endl;

  cout << "analytic grad = " << endl << grad << endl << " finite diff grad = "
       << endl << finite_diff_grad << endl;

  grad_ok = (finite_diff_grad - grad).norm() / (grad.norm()) < 0.001;
  return grad_ok;
}

double SequentialLBFGS::compute_face_energy_part(
    const Eigen::MatrixXd &uv, bool is_left_grad)
{
  double energy = 0;
  for(int f = 0; f < f_n; f++)
  {
    if(is_left_grad)
    {
      energy += compute_face_energy_left_part(uv, f);
    }
    else
    {
      energy += compute_face_energy_right_part(uv, f, m_dbl_area(f));
    }
  }
  return energy;
}

void SequentialLBFGS::pre_calc()
{
  auto& m_v = m_data.m_V;
  auto& F = m_data.m_T;
  if(!has_pre_calc)
  {

    // igl returns half a cot, so we need to double it by 2
    igl::cotmatrix_entries(m_v, F, m_m_cot_entries);
    m_m_cot_entries *= 2;

//    igl::doublearea(m_v, F, m_dbl_area);
    igl::doublearea(m_data.w_uv,m_data.s_T, m_dbl_area);
    m_dbl_area.head(mf_n) = 2 * m_data.m_M;
    m_sq_dbl_area.resize(f_n, 1);
    for(int i = 0; i < f_n; i++)
    {
      m_sq_dbl_area(i) = pow(m_dbl_area(i), 2);
    }

    m_cached_l_energy_per_face.resize(f_n);
    m_cached_r_energy_per_face.resize(f_n);

    m_cached_edges_1.resize(f_n);
    m_cached_edges_2.resize(f_n);
    m_cached_dot_prod.resize(f_n);

    for(int f = 0; f < mf_n; f++)
    {
      int v_1 = F(f, 0);
      int v_2 = F(f, 1);
      int v_3 = F(f, 2);

      m_cached_edges_1[f] = (m_v.row(v_2) - m_v.row(v_1)).squaredNorm();
      m_cached_edges_2[f] = (m_v.row(v_3) - m_v.row(v_1)).squaredNorm();
      m_cached_dot_prod[f] = (m_v.row(v_3) - m_v.row(v_1)).dot(
          m_v.row(v_2) - m_v.row(v_1));
    }

    change_scaffold_reference(m_data.w_uv);

    has_pre_calc = true;
  }
}

void
SequentialLBFGS::change_scaffold_reference(const Eigen::MatrixXd& s_uv)
{
  assert(s_uv.rows() == v_n &&
             "New Reference's vertices should be equal to the original.");

  Eigen::MatrixXi F_s = m_data.s_T.bottomRows(f_n - mf_n);

  Eigen::VectorXd scaffold_dbl_area;
  igl::doublearea(s_uv,F_s, scaffold_dbl_area);
  //m_dbl_area = Eigen::VectorXd::Ones(f_n) * (2 * d_.scaffold_factor);
  m_dbl_area.tail(f_n - mf_n) = scaffold_dbl_area;

  m_sq_dbl_area.resize(f_n, 1);
  for(int i = mf_n; i < f_n; i++)
  {
    m_sq_dbl_area(i) = pow(m_dbl_area(i), 2);
  }

  Eigen::MatrixXd m_s_cot_entries;
  igl::cotmatrix_entries(s_uv, F_s, m_s_cot_entries);
  igl::cat(1, m_m_cot_entries, m_s_cot_entries, m_cot_entries);



  for(int f = 0; f < f_n - mf_n; f++)
  {
    int v_1 = F_s(f, 0);
    int v_2 = F_s(f, 1);
    int v_3 = F_s(f, 2);

    m_cached_edges_1[f+mf_n] = (s_uv.row(v_2) - s_uv.row(v_1)).squaredNorm();
    m_cached_edges_2[f+mf_n] = (s_uv.row(v_3) - s_uv.row(v_1)).squaredNorm();
    m_cached_dot_prod[f+mf_n] = (s_uv.row(v_3) - s_uv.row(v_1)).dot(
        s_uv.row(v_2) - s_uv.row(v_1));
  }

}

double
SequentialLBFGS::compute_energy(const Eigen::VectorXd &uv_vec, bool whole)
{
  assert (uv_vec.rows() % 2 == 0);
  //cout << "computing energy from a vector!" << endl;
  Eigen::MatrixXd uv_mat;
  vec_to_mat2(uv_vec, uv_mat);
  return compute_energy(uv_mat, whole);
}

void SequentialLBFGS::compute_negative_gradient(
    const Eigen::VectorXd &uv_vec, Eigen::VectorXd &neg_grad_vec)
{
  assert (uv_vec.rows() % 2 == 0);
  //cout << "computing gradient from a vector!" << endl;
  Eigen::MatrixXd uv_mat;
  vec_to_mat2(uv_vec, uv_mat);
  Eigen::MatrixXd neg_grad_mat(uv_mat.rows(), 2);
  compute_negative_gradient(uv_mat, neg_grad_mat);
  mat2_to_vec(neg_grad_mat, neg_grad_vec);
}

void
SequentialLBFGS::zero_out_const_vertices_search_direction(Eigen::MatrixXd &d)
{

  int v_num = m_b.size();
  for(int i = 0; i < v_num; i++)
  {
    d.row(m_b[i]) << 0, 0;
  }
}

void SequentialLBFGS::adjust_scaf_weight(double weight)
{
  m_data.scaffold_factor = weight;
  m_data.update_scaffold();

  igl::doublearea(m_data.w_uv,m_data.s_T, m_dbl_area);
  m_dbl_area.head(mf_n) = 2 * m_data.m_M;

  for(int i = 0; i < f_n; i++)
  {
    m_sq_dbl_area(i) = pow(m_dbl_area(i), 2);
  }

  m_data.energy = compute_energy(m_data.w_uv, true);
}

void SequentialLBFGS::regenerate_scaffold()
{
  using namespace Eigen;
  MatrixXd m_uv = m_data.w_uv.topRows(mv_n);
  Matrix2d rect;
  rect << -10, -20, 20, 20;

  scaffold_generator(m_uv, m_data.m_T, m_data.density, m_data.w_uv,
                     m_data.s_T);

  m_data.update_scaffold();

  mv_n = m_data.mv_num;
  mf_n = m_data.mf_num;
  v_n =  m_data.v_num;
  f_n = m_data.f_num;

  // to be guaranteed conservative!
  m_cached_edges_1.resize(f_n);
  m_cached_edges_1.resize(f_n);
  m_cached_edges_2.resize(f_n);
  m_cached_dot_prod.resize(f_n);
  m_cached_l_energy_per_face.resize(f_n);
  m_cached_r_energy_per_face.resize(f_n);

  igl::doublearea(m_data.w_uv,m_data.s_T, m_dbl_area);
  //m_dbl_area = Eigen::VectorXd::Ones(f_n) * (2 * d_.scaffold_factor);
  m_dbl_area.head(mf_n) = 2 * m_data.m_M;

  m_sq_dbl_area.resize(f_n, 1);
  for(int i = 0; i < f_n; i++)
  {
    m_sq_dbl_area(i) = pow(m_dbl_area(i), 2);
  }

  change_scaffold_reference(m_data.w_uv);
}

double SequentialLBFGS::perform_iteration(Eigen::MatrixXd &w_uv)
{
  return parametrize_LBFGS(w_uv,100,0)/m_data.mesh_measure;
}

#include <igl/edge_flaps.h>
void SequentialLBFGS::mesh_improve()
{
  Eigen::MatrixXd& V = m_data.w_uv;
  Eigen::MatrixXi F = m_data.s_T.bottomRows(m_data.sf_num);

  Eigen::MatrixXi E, EF, EV, EI;
  Eigen::VectorXi EMAP_vec;

  igl::edge_flaps(F, E, EMAP_vec, EF, EI); // boundary has -1 EF and EV;

  // from EI to EV
  EV = EI;
  for(size_t e=0; e<EV.rows(); e++)
    for(auto i:{0,1})
      if(EI(e,i)!= -1)
        EV(e,i) = F(EF(e,i), EI(e,i));

  triangle_improving_edge_flip(V,F,E,EF,EV,EMAP_vec);

  m_data.s_T.bottomRows(m_data.sf_num) = F;

  m_data.update_scaffold();
  igl::doublearea(m_data.w_uv,m_data.s_T, m_dbl_area);
  m_dbl_area.head(mf_n) = 2 * m_data.m_M;

  for(int i = 0; i < f_n; i++)
  {
    m_sq_dbl_area(i) = pow(m_dbl_area(i), 2);
  }

  m_data.energy = compute_energy(m_data.w_uv, true);

}

