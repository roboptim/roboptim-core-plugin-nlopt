# Copyright (c) 2013 CNRS
# Authors: Benjamin Chretien
#
#
# This file is part of roboptim-core-plugin-nlopt
# roboptim-core-plugin-nlopt is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# roboptim-core-plugin-nlopt is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Lesser Public License for more details.  You should have
# received a copy of the GNU Lesser General Public License along with
# roboptim-core-plugin-nlopt  If not, see
# <http://www.gnu.org/licenses/>.

# Define the directory where plug-ins will be installed.
ADD_DEFINITIONS(-DPLUGIN_PATH="${LIBRARY_OUTPUT_PATH}")

MACRO(NLOPT_PLUGIN NAME)
  ADD_LIBRARY(roboptim-core-plugin-${NAME} MODULE ${NAME}.cc ${HEADERS})
  PKG_CONFIG_USE_COMPILE_DEPENDENCY(roboptim-core-plugin-${NAME} roboptim-core)
  PKG_CONFIG_USE_DEPENDENCY(roboptim-core-plugin-${NAME} nlopt)
  SET_TARGET_PROPERTIES(roboptim-core-plugin-${NAME} PROPERTIES
    PREFIX ""
    SOVERSION 3.0.0)
  INSTALL(TARGETS roboptim-core-plugin-${NAME}
    DESTINATION ${ROBOPTIM_CORE_PLUGINDIR})
ENDMACRO()

# NLP solver
NLOPT_PLUGIN("nlopt")
