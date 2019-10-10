#
#  This file is a part of TiledArray.
#  Copyright (C) 2019  Virginia Tech
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  Edward Valeev
#  Department of Chemistry, Virginia Tech
#
#  SanitizeCUDAImplicitDirectories.cmake
#  Oct 7, 2019
#

#
# Filter out directories specific to the host compiler from CMAKE_CUDA_IMPLICIT_{INCLUDE,LINK}_DIRECTORIES
#

macro(sanitize_cuda_implicit_directories)
  foreach (_type INCLUDE LINK)
    set(_var CMAKE_CUDA_IMPLICIT_${_type}_DIRECTORIES)
    set(_sanitized_var )
    foreach (_component ${${_var}})
      if (NOT ${_component} MATCHES "/gcc/(.*/|)[0-9]\.[0-9]\.[0-9]")
        list(APPEND _sanitized_var ${_component})
      endif()
    endforeach()
    set(${_var} ${_sanitized_var})
  endforeach()
endmacro()
