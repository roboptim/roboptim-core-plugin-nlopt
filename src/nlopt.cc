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

#include <roboptim/core/function.hh>
#include <roboptim/core/linear-function.hh>
#include <roboptim/core/problem.hh>
#include <roboptim/core/solver-error.hh>

#include "roboptim/core/plugin/nlopt/nlopt.hh"

namespace roboptim
{
  namespace nlopt
  {
    namespace detail
    {
      /// \brief Wrapper for NLopt functions.
      template <typename F>
      class Wrapper
      {
      public:
	Wrapper (const F& f) : f_ (f) {}
	~Wrapper () {}

	double compute(const std::vector<double>& x,
		       std::vector<double>& grad)
	{
	  using namespace Eigen;

	  Map<const VectorXd> eigen_x (x.data (), x.size ());
	  Map<VectorXd> eigen_grad (grad.data (), grad.size ());
	  // Compute grad_f(x)
	  if (!grad.empty ())
	    {
	      eigen_grad = f_.gradient (eigen_x);
	    }

	  // Compute f(x)
	  return f_ (eigen_x)[0];
	}

	static double wrap(const std::vector<double>& x,
			   std::vector<double>& grad,
			   void *data)
	{
	  return (*reinterpret_cast<Wrapper<F>*> (data)).compute (x, grad);
	}

      protected:
	const F& f_;
      };
    } // namespace detail

    SolverNlp::SolverNlp (const problem_t& problem) :
      parent_t (problem),
      n_ (problem.function ().inputSize ()),
      m_ (problem.function ().outputSize ()),
      x_ (n_),
      solverState_ (problem)
    {
      // Initialize x
      x_.setZero ();

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
#define N_ALGO 9
#define ALGO_LIST (N_ALGO, (LD_MMA, LD_SLSQP, LD_LBFGS, LD_VAR1, LD_VAR2, \
                            LD_TNEWTON_PRECOND_RESTART, LD_TNEWTON_PRECOND, \
                            LD_TNEWTON_RESTART, LD_TNEWTON))
#define GET_ALGO(n) BOOST_PP_ARRAY_ELEM(n,ALGO_LIST)
#define BOOST_PP_LOCAL_MACRO(n)				\
	(std::string (BOOST_PP_STRINGIZE(GET_ALGO(n))), \
	 ::nlopt::GET_ALGO(n))
#define BOOST_PP_LOCAL_LIMITS (0,N_ALGO-1)
#include BOOST_PP_LOCAL_ITERATE()
	;
#undef ALGO_LIST
#undef N_ALGO
    }

    SolverNlp::~SolverNlp () throw ()
    {
    }

#define DEFINE_PARAMETER(KEY, DESCRIPTION, VALUE)	\
    do {						\
      parameters ()[KEY].description = DESCRIPTION;	\
      parameters ()[KEY].value = VALUE;			\
    } while (0)

    void SolverNlp::initializeParameters () throw ()
    {
      // Clear parameters
      parameters ().clear ();

      double epsilon = std::numeric_limits<double>::epsilon ();

      // Shared parameters
      DEFINE_PARAMETER ("max-iterations", "number of iterations", 3000);

      // NLopt-specific parameters
      DEFINE_PARAMETER ("nlopt.algorithm",
			"optimization algorithm",
			std::string ("LD_MMA"));
      DEFINE_PARAMETER ("nlopt.xtol_rel",
			"relative tolerance on optimization parameters",
			epsilon);
      DEFINE_PARAMETER ("nlopt.xtol_abs",
			"absolute tolerance on optimization parameters",
			epsilon);
    }

    // Utility macro to print result with warning message
#define LOAD_RESULT_WARNINGS(STATUS)					\
    case STATUS:							\
    {									\
      ResultWithWarnings result (n_, 1);				\
      result.x = map_x;							\
      result.value = problem ().function () (result.x);			\
      result.warnings.push_back (SolverWarning (result_map_[STATUS]));	\
      result_ = result;							\
    }									\
    break;

    // Utility macro to print error message
#define LOAD_RESULT_ERROR(STATUS)			\
    case STATUS:					\
    {							\
      result_ = SolverError (result_map_[STATUS]);	\
    }							\
    break;

    void SolverNlp::solve () throw ()
    {
      using namespace Eigen;

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

      ::nlopt::opt opt (algo_map_[boost::get<std::string>
                                  (parameters ()["nlopt.algorithm"].value)],
                        static_cast<unsigned int> (n_));

      // Set appropriate tolerances
      if (parameters ().find ("nlopt.xtol_rel") != parameters ().end ())
	opt.set_xtol_rel (boost::get<double>
			  (parameters ()["nlopt.xtol_rel"].value));
      if (parameters ().find ("nlopt.xtol_abs") != parameters ().end ())
	opt.set_xtol_abs (boost::get<double>
			  (parameters ()["nlopt.xtol_abs"].value));

      // Set objective function
      detail::Wrapper<function_t> obj (problem ().function ());
      opt.set_min_objective (detail::Wrapper<function_t>::wrap,
                             &obj);

      double res_min;
      std::vector<double> stl_x (n_);
      Map<argument_t> map_x (stl_x.data (), n_);
      map_x = x_;

      // Solve problem
      ::nlopt::result result = opt.optimize (stl_x, res_min);

      switch (result)
	{
	case ::nlopt::SUCCESS:
	  {
	    Result result (n_, 1);
	    result.x = map_x;
	    result.value = problem ().function () (result.x);
	    result_ = result;
	  }
	  break;

	  LOAD_RESULT_WARNINGS (::nlopt::STOPVAL_REACHED)
	    LOAD_RESULT_WARNINGS (::nlopt::FTOL_REACHED)
	    LOAD_RESULT_WARNINGS (::nlopt::XTOL_REACHED)
	    LOAD_RESULT_WARNINGS (::nlopt::MAXEVAL_REACHED)
	    LOAD_RESULT_WARNINGS (::nlopt::MAXTIME_REACHED)

	    LOAD_RESULT_ERROR (::nlopt::FAILURE)
	    LOAD_RESULT_ERROR (::nlopt::INVALID_ARGS)
	    LOAD_RESULT_ERROR (::nlopt::OUT_OF_MEMORY)
	    LOAD_RESULT_ERROR (::nlopt::ROUNDOFF_LIMITED)
	    LOAD_RESULT_ERROR (::nlopt::FORCED_STOP)

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
