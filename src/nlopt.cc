// Copyright (c) 2014 CNRS
// Authors: Benjamin Chretien


// This file is part of roboptim-core-plugin-nlopt
// roboptim-core-plugin-nlopt is free software: you can redistribute it
// and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version.

// roboptim-core-plugin-nlopt is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Lesser Public License for more details.  You should have
// received a copy of the GNU Lesser General Public License along with
// roboptim-core-plugin-nlopt  If not, see
// <http://www.gnu.org/licenses/>.

#include <cstring>
#include <map>
#include <limits> // epsilon

#include <boost/assign/list_of.hpp>
#include <boost/preprocessor/array/elem.hpp>
#include <boost/preprocessor/iteration/local.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <roboptim/core/function.hh>
#include <roboptim/core/linear-function.hh>
#include <roboptim/core/problem.hh>
#include <roboptim/core/solver-error.hh>
#include <roboptim/core/function/constant.hh>
#include <roboptim/core/filter/plus.hh>
#include <roboptim/core/filter/minus.hh>

#include "roboptim/core/plugin/nlopt/nlopt.hh"

namespace roboptim
{
  namespace nlopt
  {
    namespace detail
    {
      using namespace Eigen;

      template <typename S>
      class CallbackHandler
      {
      public:
        typedef S solver_t;
        typedef typename solver_t::problem_t problem_t;
        typedef typename solver_t::problem_t::function_t::argument_t argument_t;
        typedef typename solver_t::problem_t::function_t::value_type value_type;
        typedef typename solver_t::callback_t callback_t;
        typedef typename solver_t::solverState_t solverState_t;

        CallbackHandler (const solver_t& solver)
          : problem_ (solver.problem ()),
            callback_ (solver.callback ()),
            solverState_ (solver.problem ())
        {}
        virtual ~CallbackHandler (){}

        void callback (const argument_t& x, value_type cost)
        {
          solverState_.x () = x;
          solverState_.cost () = cost;

          callback_ (problem_, solverState_);
        }

      private:
        /// \brief Intermediate callback (called at each end
        /// of iteration).
        const problem_t& problem_;

        /// \brief Callback function.
        callback_t callback_;

        /// \brief Solver state.
        solverState_t solverState_;
      };

      /// \brief Wrapper for NLopt functions.
      /// TODO: add an optional callback in the wrapper (not available in NLopt)
      template <typename F>
      class Wrapper
      {
      public:
	typedef Matrix<double,Dynamic,Dynamic,RowMajor> jacobian_t;
	typedef Map<jacobian_t> map_jacobian_t;
	typedef DifferentiableFunction::result_t result_t;
	typedef Map<result_t> map_result_t;
	typedef DifferentiableFunction::argument_t argument_t;
	typedef Map<const argument_t> map_argument_t;
	typedef argument_t::Index index_t;

	typedef CallbackHandler<SolverNlp> callbackHandler_t;

	Wrapper (const F& f) : f_ (f) {}
	~Wrapper () {}

	double compute(const std::vector<double>& x,
		       std::vector<double>& grad)
	{
	  Map<const VectorXd> eigen_x (x.data (),
				       static_cast<index_t> (x.size ()));
	  Map<VectorXd> eigen_grad (grad.data (),
				    static_cast<index_t> (grad.size ()));
	  // Compute grad_f(x)
	  if (!grad.empty ())
	    {
	      eigen_grad = f_.gradient (eigen_x);
	    }

	  // Compute f(x)
	  double res = f_ (eigen_x)[0];

	  // Callback (for cost function)
	  if (callbackHandler_)
	    {
	      // Call user-defined callback
	      callbackHandler_->callback (eigen_x, res);
	    }

	  return res;
	}

	void compute(const map_argument_t& x,
		     map_result_t& res,
		     boost::optional<map_jacobian_t>& jac)
	{
	  // Compute the Jacobian of f
	  if (jac) *jac = f_.jacobian (x);

	  // Compute f(x)
	  res = f_ (x);

	  // Callback (for cost function)
	  if (callbackHandler_)
	    {
	      // Call user-defined callback
	      callbackHandler_->callback (x, res[0]);
	    }
	}

	// TODO: use C-style NLopt function to prevent copy to STL vector
	/// \brief Wrap function
	static double wrap(const std::vector<double>& x,
			   std::vector<double>& grad,
			   void *data)
	{
	  return (*reinterpret_cast<Wrapper<F>*> (data)).compute (x, grad);
	}

	/// \brief Wrap vector-valued function
	static void vwrap(unsigned m, double* res,
			  unsigned n, const double* x,
			  double* grad, void *data)
	{
	  map_argument_t map_x (x, n);
	  map_result_t map_res (res, m);
	  // Note: gradients are stored contiguously, i.e. dci/dxj is
	  // stored in grad[i*n+j] where c: R^n -> R^m
	  boost::optional<map_jacobian_t> map_jac;
	  if (grad != NULL) map_jac = map_jacobian_t (grad, m, n);
	  (*reinterpret_cast<Wrapper<F>*> (data)).compute (map_x,
							   map_res,
							   map_jac);
	}

	boost::optional<callbackHandler_t&>& callbackHandler ()
	{
	  return callbackHandler_;
	}

	const boost::optional<callbackHandler_t&>& callbackHandler () const
	{
	  return callbackHandler_;
	}

      protected:
	const F& f_;

	boost::optional<callbackHandler_t&> callbackHandler_;
      };


      /// \brief Visitor used to load the last values of constraints.
      struct constraintLoader : public boost::static_visitor<void>
      {
        typedef Function::vector_t vector_t;
        typedef vector_t::Index index_t;
        typedef Function::argument_t argument_t;

        constraintLoader (const argument_t& x,
                          vector_t& constraintValues) :
	  x_ (x),
	  constraintValues_ (constraintValues),
	  i_ (0)
        {}

        template <typename U>
        void operator () (const U& g)
        {
	  assert (constraintValues_.size ()
		  >= static_cast<index_t> (g->outputSize ()) + i_);
	  constraintValues_.segment (i_, g->outputSize ()) = (*g) (x_);
	  i_ += g->outputSize ();
        }

      private:
        const argument_t& x_;
        vector_t& constraintValues_;
        vector_t::Index i_;
      };
    } // namespace detail

    SolverNlp::SolverNlp (const problem_t& problem) :
      parent_t (problem),
      n_ (problem.function ().inputSize ()),
      m_ (problem.function ().outputSize ()),
      x_ (n_),
      solverState_ (problem),
      algo_map_ (),
      global_algos_ ()
    {
      // Initialize x
      x_.setZero ();

      epsilon_ = std::numeric_limits<double>::epsilon ();

      // Initialize solver parameters
      initializeParameters ();

      // Load <Status, warning message> map
      result_map_ = boost::assign::map_list_of
	(::nlopt::FAILURE,
	 "Failure")
	(::nlopt::INVALID_ARGS,
	 "Invalid arguments")
	(::nlopt::OUT_OF_MEMORY,
	 "Out of memory")
	(::nlopt::ROUNDOFF_LIMITED,
	 "Roundoff limited")
	(::nlopt::FORCED_STOP,
	 "Forced stop")
	(::nlopt::SUCCESS,
	 "Optimization success")
	(::nlopt::STOPVAL_REACHED,
	 "Stop value reached")
	(::nlopt::FTOL_REACHED,
	 "f tolerance reached")
	(::nlopt::XTOL_REACHED,
	 "x tolerance reached")
	(::nlopt::MAXEVAL_REACHED,
	 "Maximum number of evaluations reached")
	(::nlopt::MAXTIME_REACHED,
	 "Maximum time reached");

      // Load <algo string, algo> map
      algo_map_ = boost::assign::map_list_of
#define N_ALGO 21
#define ALGO_LIST (N_ALGO, (LD_MMA, LD_SLSQP, LD_LBFGS, LD_VAR1, LD_VAR2, \
                            LD_TNEWTON_PRECOND_RESTART, LD_TNEWTON_PRECOND, \
                            LD_TNEWTON_RESTART, LD_TNEWTON,		\
                            GD_MLSL, GD_MLSL_LDS, GD_STOGO,		\
                            GN_ISRES, GN_ORIG_DIRECT, GN_ORIG_DIRECT_L, \
                            AUGLAG, AUGLAG_EQ, LD_AUGLAG, LD_AUGLAG_EQ, \
                            GD_MLSL, GD_MLSL_LDS))
#define GET_ALGO(n) BOOST_PP_ARRAY_ELEM(n,ALGO_LIST)
#define BOOST_PP_LOCAL_MACRO(n)				\
	(std::string (BOOST_PP_STRINGIZE(GET_ALGO(n))), \
	 ::nlopt::GET_ALGO(n))
#define BOOST_PP_LOCAL_LIMITS (0,N_ALGO-1)
#include BOOST_PP_LOCAL_ITERATE()
	;
#undef ALGO_LIST
#undef N_ALGO

      global_algos_.insert ("AUGLAG");
      global_algos_.insert ("AUGLAG_EQ");
      global_algos_.insert ("LD_AUGLAG");
      global_algos_.insert ("LD_AUGLAG_EQ");
      global_algos_.insert ("GD_MLSL");
      global_algos_.insert ("GD_MLSL_LDS");
    }

    SolverNlp::~SolverNlp ()
    {
    }

#define DEFINE_PARAMETER(KEY, DESCRIPTION, VALUE)	\
    do {						\
      parameters ()[KEY].description = DESCRIPTION;	\
      parameters ()[KEY].value = VALUE;			\
    } while (0)

    void SolverNlp::initializeParameters ()
    {
      // Clear parameters
      parameters ().clear ();

      // Shared parameters
      DEFINE_PARAMETER ("max-iterations", "number of iterations", 10000);

      // NLopt-specific parameters
      DEFINE_PARAMETER ("nlopt.algorithm",
			"optimization algorithm",
			std::string ("LD_AUGLAG"));
      DEFINE_PARAMETER ("nlopt.local_algorithm",
			"local optimization algorithm",
			std::string ("LD_MMA"));
      DEFINE_PARAMETER ("nlopt.xtol_rel",
			"relative tolerance on optimization parameters",
			epsilon_);
      DEFINE_PARAMETER ("nlopt.xtol_abs",
			"absolute tolerance on optimization parameters",
			epsilon_);
    }

#define LOAD_RESULT_CONSTRAINTS()					\
    /* Return state of constraints at the end of the optimization */	\
    size_t n_cstr = 0;							\
    for (size_t i = 0; i < problem ().constraints ().size (); ++i) {	\
      n_cstr += problem ().boundsVector ()[i].size ();			\
    }									\
    result.constraints.resize (static_cast<index_t> (n_cstr));		\
    detail::constraintLoader cl (result.x, result.constraints);		\
    for (size_t i = 0; i < problem ().constraints ().size (); ++i) {	\
      boost::apply_visitor (cl, problem ().constraints ()[i]);		\
    }

    // Utility macro to print result with warning message
#define LOAD_RESULT_WARNINGS(STATUS)					\
    case STATUS:							\
    {									\
      ResultWithWarnings result (n_, 1);				\
      result.x = map_x;							\
      result.value = problem ().function () (result.x);			\
      LOAD_RESULT_CONSTRAINTS();					\
      result.warnings.push_back (SolverWarning (result_map_[STATUS]));	\
      result_ = result;							\
      if (!callback_.empty ())						\
        {								\
          solverState_.x () = result.x;					\
          solverState_.cost () = result.value[0];			\
          callback_ (problem (), solverState_);				\
        }								\
    }									\
    break;

    // Utility macro to print error message
#define LOAD_RESULT_ERROR(STATUS)			\
    case STATUS:					\
    {							\
      result_ = SolverError (result_map_[STATUS]);	\
    }							\
    break;

    void SolverNlp::solve ()
    {
      using namespace Eigen;
      typedef VectorXd::Index index_t;

      // Load optional starting point
      if (problem ().startingPoint ())
	{
	  x_ = *(problem ().startingPoint ());
	}

      // Create NLopt solver
      // Check mandatory NLopt optimization algorithm
      if (parameters ().find ("nlopt.algorithm") == parameters ().end ())
	{
	  result_ = SolverError ("Undefined NLopt algorithm.");
	  return;
	}

      std::string opt_algo
	= boost::get<std::string> (parameters ()["nlopt.algorithm"].value);
      ::nlopt::opt opt (algo_map_[opt_algo], static_cast<unsigned int> (n_));

      // Set appropriate tolerances
      if (parameters ().find ("nlopt.xtol_rel") != parameters ().end ())
	opt.set_xtol_rel (boost::get<double>
			  (parameters ()["nlopt.xtol_rel"].value));
      if (parameters ().find ("nlopt.xtol_abs") != parameters ().end ())
	opt.set_xtol_abs (boost::get<double>
			  (parameters ()["nlopt.xtol_abs"].value));
      if (parameters ().find ("max-iterations") != parameters ().end ())
	opt.set_maxeval (boost::get<int>
			 (parameters ()["max-iterations"].value));

      // If using a global algorithm that relies on a local algorithm
      if (global_algos_.find (opt_algo) != global_algos_.end ())
	{
          // We need to set a local optimizer (a copy is made into opt)
          std::string local_algo = boost::get<std::string>
	    (parameters ()["nlopt.local_algorithm"].value);
          ::nlopt::opt local_opt (algo_map_[local_algo],
                                  static_cast<unsigned int> (n_));
          local_opt.set_xtol_rel (opt.get_xtol_rel ());
          local_opt.set_xtol_abs (opt.get_xtol_abs ());
          local_opt.set_maxeval (opt.get_maxeval ());
          opt.set_local_optimizer (local_opt);
	}

      // Set objective function
      typedef detail::Wrapper<function_t> obj_wrapper_t;
      obj_wrapper_t obj (problem ().function ());

      typedef detail::Wrapper<function_t>::callbackHandler_t callbackHandler_t;
      callbackHandler_t callbackHandler (*this);
      obj.callbackHandler () = callbackHandler;
      opt.set_min_objective (obj_wrapper_t::wrap, &obj);

      // Add bound constraints
      std::vector<double> lb, ub;
      const intervals_t& bounds = problem ().argumentBounds ();
      for (std::size_t i = 0; i < bounds.size (); ++i)
	{
	  index_t ii = static_cast<index_t> (i);

	  double lower = bounds[i].first;
	  double upper = bounds[i].second;

	  // If starting point outside of bounds: move x to bounds
	  // (else the solver may fail at start)
	  if (x_[ii] < lower) x_[ii] = lower;
	  if (x_[ii] > upper) x_[ii] = upper;

	  lb.push_back (lower);
	  ub.push_back (upper);
	}
      opt.set_lower_bounds (lb);
      opt.set_upper_bounds (ub);

      typedef detail::Wrapper<DifferentiableFunction> constraint_wrapper_t;
      typedef boost::shared_ptr<constraint_wrapper_t> constraint_wrapper_ptr;
      size_t iter = 0;
      std::vector<boost::shared_ptr<DifferentiableFunction> > constraints;
      std::vector<constraint_wrapper_ptr> constraint_wrappers;
      constraints.reserve (problem ().constraints ().size ());
      constraint_wrappers.reserve (problem ().constraints ().size ());

      // Iterate over constraints
      for (constraints_t::const_iterator
	     cstr = problem ().constraints ().begin ();
	   cstr != problem ().constraints ().end ();
	   ++cstr)
	{
	  // Set constraints
	  boost::shared_ptr<DifferentiableFunction> g;
	  if (cstr->which () == linearFunctionId)
	    g = boost::get<boost::shared_ptr<LinearFunction> > (*cstr);
	  else
	    g = boost::get<boost::shared_ptr<DifferentiableFunction> > (*cstr);
	  assert (!!g);

	  constraints.push_back (g);

	  const intervals_t& bounds = problem ().boundsVector ()[iter];

	  // Vector of tolerances
	  index_t m = g->outputSize ();
	  std::size_t mm = static_cast<std::size_t> (m);
	  std::vector<double> vec_tol (mm, epsilon_);

	  ConstantFunction::result_t lowerBoundValues (m);
	  ConstantFunction::result_t upperBoundValues (m);

	  for (std::size_t i = 0; i < mm; ++i)
	    {
	      index_t ii = static_cast<index_t> (i);
	      lowerBoundValues[ii] = bounds[i].first;
	      upperBoundValues[ii] = bounds[i].second;
	    }
	  boost::shared_ptr<ConstantFunction> cst_lb =
	    boost::make_shared<ConstantFunction> (g->inputSize (),
						  lowerBoundValues);
	  boost::shared_ptr<ConstantFunction> cst_ub =
	    boost::make_shared<ConstantFunction> (g->inputSize (),
						  upperBoundValues);

	  boost::shared_ptr<DifferentiableFunction> g_lb = cst_lb - g;
	  boost::shared_ptr<DifferentiableFunction> g_ub = g - cst_ub;

	  constraint_wrapper_ptr wrapper_ub (new constraint_wrapper_t (*g_ub));
	  constraint_wrappers.push_back (wrapper_ub);
	  constraints.push_back (g_ub);

	  constraint_wrapper_ptr wrapper_lb (new constraint_wrapper_t (*g_lb));
	  constraint_wrappers.push_back (wrapper_lb);
	  constraints.push_back (g_lb);

	  // Inequality constraint
	  // Process: c <= ub
	  opt.add_inequality_mconstraint (constraint_wrapper_t::vwrap,
					  wrapper_ub.get (), vec_tol);

	  // Process: lb <= c, i.e. -c <= -lb
	  opt.add_inequality_mconstraint (constraint_wrapper_t::vwrap,
					  wrapper_lb.get (), vec_tol);
	  ++iter;

	  // TODO: handle equality constraints
	}

      // Use callback (even if mostly empty)
      if (!callback_.empty ())
	{
	  solverState_.x () = x_;
          callback_ (problem (), solverState_);
	}

      double res_min;
      std::vector<double> stl_x (static_cast<std::size_t> (n_));
      Map<argument_t> map_x (stl_x.data (), n_);
      map_x = x_;

      // Solve problem with initial x
      ::nlopt::result opt_result;
      try {
	opt_result = opt.optimize (stl_x, res_min);
      }
      // Result may still be correct when a roundoff exception is thrown.
      catch (::nlopt::roundoff_limited& e)
	{
          opt_result = ::nlopt::ROUNDOFF_LIMITED;
	}

      switch (opt_result)
	{
	case ::nlopt::SUCCESS:
	  {
	    Result result (n_, 1);
	    result.x = map_x;
	    result.value = problem ().function () (result.x);
	    LOAD_RESULT_CONSTRAINTS();
	    result_ = result;

	    // Use callback for last iteration
	    if (!callback_.empty ())
	      {
		solverState_.x () = result.x;
		solverState_.cost () = result.value[0];
		callback_ (problem (), solverState_);
	      }
	  }
	  break;

	  LOAD_RESULT_WARNINGS (::nlopt::STOPVAL_REACHED);
	  LOAD_RESULT_WARNINGS (::nlopt::FTOL_REACHED);
	  LOAD_RESULT_WARNINGS (::nlopt::XTOL_REACHED);
	  LOAD_RESULT_WARNINGS (::nlopt::MAXEVAL_REACHED);
	  LOAD_RESULT_WARNINGS (::nlopt::MAXTIME_REACHED);
	  LOAD_RESULT_WARNINGS (::nlopt::ROUNDOFF_LIMITED);
	  LOAD_RESULT_ERROR (::nlopt::FAILURE);
	  LOAD_RESULT_ERROR (::nlopt::INVALID_ARGS);
	  LOAD_RESULT_ERROR (::nlopt::OUT_OF_MEMORY);
	  LOAD_RESULT_ERROR (::nlopt::FORCED_STOP);

	default:
	  {
	    result_ = SolverError ("Error");
	  }
	}
    }

  } // namespace nlopt
} // end of namespace roboptim

extern "C"
{
  using namespace roboptim::nlopt;
  typedef SolverNlp::parent_t solver_t;

  ROBOPTIM_DLLEXPORT unsigned getSizeOfProblem ();
  ROBOPTIM_DLLEXPORT const char* getTypeIdOfConstraintsList ();
  ROBOPTIM_DLLEXPORT solver_t* create (const SolverNlp::problem_t& pb);
  ROBOPTIM_DLLEXPORT void destroy (solver_t* p);

  ROBOPTIM_DLLEXPORT unsigned getSizeOfProblem ()
  {
    return sizeof (solver_t::problem_t);
  }

  ROBOPTIM_DLLEXPORT const char* getTypeIdOfConstraintsList ()
  {
    return typeid (solver_t::problem_t::constraintsList_t).name ();
  }

  ROBOPTIM_DLLEXPORT solver_t* create (const SolverNlp::problem_t& pb)
  {
    return new SolverNlp (pb);
  }

  ROBOPTIM_DLLEXPORT void destroy (solver_t* p)
  {
    delete p;
  }
}
