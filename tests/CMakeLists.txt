# Copyright (c) 2014 CNRS
# Authors: Benjamin Chretien
#
#
# This file is part of roboptim-core-plugin-eigen
# roboptim-core-plugin-eigen is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# roboptim-core-plugin-eigen is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Lesser Public License for more details.  You should have
# received a copy of the GNU Lesser General Public License along with
# roboptim-core-plugin-eigen  If not, see
# <http://www.gnu.org/licenses/>.


# Prepare test suite
SET(PLUGIN_PATH "${CMAKE_BINARY_DIR}/src")
INCLUDE(shared-tests/tests.cmake)

# Schittkowski problems that are expected to fail
SET(SCHITTKOWSKI_PROBLEMS_FAIL
    1 2 9
    13 12 16 19
    20 25 26
    33 34 36 39
    43 44 47 48 49
    51 55 56
    65 66
    81)

SET(SOLVER_NAME "nlopt")
SET(FUNCTION_TYPE ::roboptim::EigenMatrixDense)
SET(PROGRAM_SUFFIX "")
SET(COST_FUNCTION_TYPE ::roboptim::GenericDifferentiableFunction)
SET(CONSTRAINT_TYPE_1 ::roboptim::GenericLinearFunction)
SET(CONSTRAINT_TYPE_2 ::roboptim::GenericDifferentiableFunction)
BUILD_SCHITTKOWSKI_PROBLEMS()

# RobOptim test problems
SET(SOLVER_NAME "nlopt")
SET(FUNCTION_TYPE ::roboptim::EigenMatrixDense)
SET(PROGRAM_SUFFIX "")
SET(COST_FUNCTION_TYPE ::roboptim::GenericDifferentiableFunction)
BUILD_ROBOPTIM_PROBLEMS()
