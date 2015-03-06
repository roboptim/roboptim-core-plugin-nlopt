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

#ifndef ROBOPTIM_CORE_PLUGIN_NLOPT_NLOPT_HH
# define ROBOPTIM_CORE_PLUGIN_NLOPT_NLOPT_HH

# include <map>
# include <set>

# include <boost/mpl/vector.hpp>

# include <roboptim/core/solver.hh>
# include <roboptim/core/solver-state.hh>

// Use C++ interface for NLopt
# include <nlopt.hpp>

namespace roboptim {
  namespace nlopt {
    /// \brief Solver implementing a variant of Levenberg-Marquardt algorithm.
    ///
    /// This solver tries to minimize the euclidean norm of a vector valued
    /// function.
    class SolverNlp :
      public Solver<DifferentiableFunction,
                    boost::mpl::vector<LinearFunction,
                                       DifferentiableFunction> >
    {
    public:
      /// \brief Parent type
      typedef Solver<DifferentiableFunction,
                     boost::mpl::vector<LinearFunction,
                                        DifferentiableFunction> > parent_t;
      /// \brief Cost function type
      typedef problem_t::function_t function_t;
      /// \brief Argument type
      typedef function_t::argument_t argument_t;
      /// \brief type of result
      typedef function_t::result_t result_t;
      /// \brief type of gradient
      typedef DifferentiableFunction::gradient_t gradient_t;
      /// \brief Size type
      typedef Function::size_type size_type;
      /// \brief Constraints type
      typedef problem_t::constraints_t constraints_t;
      /// \brief Constraint type
      typedef problem_t::constraint_t constraint_t;
      /// \brief Intervals type
      typedef problem_t::intervals_t intervals_t;
      /// \brief Interval type
      typedef problem_t::interval_t interval_t;

      /// \brief Solver state
      typedef SolverState<parent_t::problem_t> solverState_t;

      /// \brief RobOptim callback
      typedef parent_t::callback_t callback_t;

      /// \brief Constructor by problem
      explicit SolverNlp (const problem_t& problem);
      virtual ~SolverNlp ();
      /// \brief Solve the optimization problem
      virtual void solve ();

      /// \brief Return the number of variables.
      size_type n () const
      {
	return n_;
      }

      /// \brief Return the number of functions.
      size_type m () const
      {
	return m_;
      }

      /// \brief Get the optimization parameters.
      Function::argument_ref parameter ()
      {
	return x_;
      }

      /// \brief Get the optimization parameters.
      Function::const_argument_ref parameter () const
      {
	return x_;
      }

      /// \brief Set the callback called at each iteration.
      virtual void
      setIterationCallback (callback_t callback)
      {
        callback_ = callback;
      }

      /// \brief Get the callback called at each iteration.
      const callback_t& callback () const
      {
        return callback_;
      }

    public:
      static const int linearFunctionId = 0;
      static const int nonlinearFunctionId = 1;

    private:
      /// \brief Initialize solver parameters.
      void initializeParameters ();

    private:
      /// \brief Number of variables
      size_type n_;
      /// \brief Dimension of the cost function
      size_type m_;

      /// \brief Parameter of the function
      Function::argument_t x_;

      /// \brief State of the solver at each iteration
      solverState_t solverState_;

      /// \brief Intermediate callback (called at each end of iteration).
      callback_t callback_;

      /// \brief Map optimization result to result message
      std::map< ::nlopt::result, std::string> result_map_;

      /// \brief Map string to NLopt algorithm
      std::map<std::string, ::nlopt::algorithm> algo_map_;

      /// \brief Set of global algorithms that expect a local algorithm.
      std::set<std::string> global_algos_;

      /// \brief Epsilon
      double epsilon_;
    }; // class SolverNlp
  } // namespace nlopt
} // namespace roboptim
#endif // ROBOPTIM_CORE_PLUGIN_NLOPT_NLOPT_NLP_HH
