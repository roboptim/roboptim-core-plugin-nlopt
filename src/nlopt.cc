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

#include <boost/assign/list_of.hpp>

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
      // Initialize this class parameters
      x_.setZero ();

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
    }

    SolverNlp::~SolverNlp () throw ()
    {
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
      // TODO: choose appropriate solver
      ::nlopt::opt opt(::nlopt::LD_MMA, static_cast<unsigned int> (n_));

      // TODO: set appropriate tolerances
      opt.set_xtol_rel (1e-4);

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
