/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef _KOKKOSKERNEL_CONTROLS_HPP
#define _KOKKOSKERNEL_CONTROLS_HPP
/// \file  KokkosKernels_Controls.hpp
/// \brief Mechanism to control internal behavior of kernels
/// \author Luc Berger-Vergiat (lberge@sandia.gov)

#include <unordered_map>
#include "KokkosKernels_config.h"
#include "KokkosKernels_tpl_handles_decl.hpp"

// TPLS headers
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include "cublas_v2.h"
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#include "cusparse.h"
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
#include "rocsparse.h"
#endif

namespace KokkosKernels {
namespace Experimental {

// Declaration of Controls class
class Controls {
 public:
  // Constructor
  Controls() = default;

  // set a new parameter
  void setParameter(const std::string& name, const std::string& value) {
    kernel_parameters[name] = value;
  }

  // check if a parameter is already set
  bool isParameter(const std::string& name) const {
    return kernel_parameters.end() != kernel_parameters.find(name);
  }

  /// \brief get the value associated with \c name, or \c default if not present
  ///
  /// \param name the name of the parameter to retrieve
  /// \param orUnset (default \c "" ) the value to return if \c name is not set
  std::string getParameter(const std::string& name,
                           const std::string& orUnset = "") const {
    auto search = kernel_parameters.find(name);
    if (kernel_parameters.end() == search) {
      std::cerr << "WARNING: Controls::getParameter for name \"" << name
                << "\" was unset" << std::endl;
      return orUnset;
    } else {
      return search->second;
    }
  }

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
  mutable cublasHandle_t cublasHandle = 0;

  cublasHandle_t getCublasHandle() const {
    if (cublasHandle == 0) {
      KokkosBlas::Impl::CudaBlasSingleton& s =
          KokkosBlas::Impl::CudaBlasSingleton::singleton();
      cublasHandle = s.handle;
    }
    return cublasHandle;
  }

  void setCublasHandle(const cublasHandle_t userCublasHandle) {
    cublasHandle = userCublasHandle;
  }
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
  mutable cusparseHandle_t cusparseHandle = 0;

  cusparseHandle_t getCusparseHandle() const {
    if (cusparseHandle == 0) {
      KokkosKernels::Impl::CusparseSingleton& s =
          KokkosKernels::Impl::CusparseSingleton::singleton();
      cusparseHandle = s.cusparseHandle;
    }
    return cusparseHandle;
  }

  void setCusparseHandle(const cusparseHandle_t userCusparseHandle) {
    cusparseHandle = userCusparseHandle;
  }
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
  mutable rocsparse_handle rocsparseHandle = 0;

  rocsparse_handle getRocsparseHandle() const {
    if (rocsparseHandle == 0) {
      KokkosKernels::Impl::RocsparseSingleton& s =
          KokkosKernels::Impl::RocsparseSingleton::singleton();
      rocsparseHandle = s.rocsparseHandle;
    }
    return rocsparseHandle;
  }

  void setRocsparseHandle(const rocsparse_handle userRocsparseHandle) {
    rocsparseHandle = userRocsparseHandle;
  }
#endif

 private:
  // storage for kernel parameters
  std::unordered_map<std::string, std::string> kernel_parameters;
};

}  // namespace Experimental
}  // namespace KokkosKernels

#endif  // _KOKKOSKERNEL_CONTROLS_HPP
