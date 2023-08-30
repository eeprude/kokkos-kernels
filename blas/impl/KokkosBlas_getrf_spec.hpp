//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOSBLAS_IMPL_GETRF_SPEC_HPP_
#define KOKKOSBLAS_IMPL_GETRF_SPEC_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>

// Include the actual functors
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include <KokkosBlas_getrf_impl.hpp>
#endif

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class AVT>
struct getrf_eti_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

//
// Macro for declaration of full specialization availability
// KokkosBlas::Impl::GETRF.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _INST macro below) across one or
// more .cpp files.
//
#define KOKKOSBLAS_GETRF_ETI_SPEC_AVAIL(SCALAR_TYPE, LAYOUT_TYPE,       \
                                       EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
  template <>                                                           \
  struct getrf_eti_spec_avail<                                          \
      Kokkos::View<SCALAR_TYPE **, LAYOUT_TYPE,                         \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,           \
    enum : bool { value = true };                                       \
  };

// Include the actual specialization declarations
#include <KokkosBlas_getrf_tpl_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosBlas_getrf_eti_spec_avail.hpp>

namespace KokkosBlas {
namespace Impl {

// Unification layer
/// \brief Implementation of KokkosBlas::getrf.

template <class AMatrix, class IPIVV,
          bool tpl_spec_avail = getrf_tpl_spec_avail<AMatrix>::value,
          bool eti_spec_avail = getrf_eti_spec_avail<AMatrix>::value>
struct GETRF {
  static void getrf(const AMatrix &A, const IPIVV &IPIV);
};

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
//! Full specialization of getrf for multi vectors.
// Unification layer
template <class AMatrix, class IPIVV>
struct GETRF<AMatrix, IPIVV, false, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  static void getrf(const AMatrix & /* A */, const IPIVV & /* IPIV */) {
    // NOTE: Might add the implementation of KokkosBlas::getrf later
    throw std::runtime_error(
        "No fallback implementation of GETRF (general LU factorization & solve) " // Aqui
        "exists. Enable BLAS and/or MAGMA TPL.");
  }
};

#endif
}  // namespace Impl
}  // namespace KokkosBlas

//
// Macro for declaration of full specialization of
// KokkosBlas::Impl::GETRF.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _DEF macro below) across one or
// more .cpp files.
//
#define KOKKOSBLAS_GETRF_ETI_SPEC_DECL(SCALAR_TYPE, LAYOUT_TYPE,       \
                                      EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
  extern template struct GETRF<                                        \
      Kokkos::View<SCALAR_TYPE **, LAYOUT_TYPE,                        \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
      Kokkos::View<int *, LAYOUT_TYPE,                                 \
                   Kokkos::Device<Kokkos::DefaultHostExecutionSpace,   \
                                  Kokkos::HostSpace>,                  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
      false, true>;

#define KOKKOSBLAS_GETRF_ETI_SPEC_INST(SCALAR_TYPE, LAYOUT_TYPE,       \
                                      EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
  template struct GETRF<                                               \
      Kokkos::View<SCALAR_TYPE **, LAYOUT_TYPE,                        \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
      Kokkos::View<int *, LAYOUT_TYPE,                                 \
                   Kokkos::Device<Kokkos::DefaultHostExecutionSpace,   \
                                  Kokkos::HostSpace>,                  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
      false, true>;

#include <KokkosBlas_getrf_tpl_spec_decl.hpp>

#endif  // KOKKOSBLAS_IMPL_GETRF_SPEC_HPP_
