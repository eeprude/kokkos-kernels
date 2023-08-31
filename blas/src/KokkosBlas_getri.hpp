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

// AquiEEP

/// \file KokkosBlas_getri.hpp
/// \brief Local dense linear solve
///
/// This file provides KokkosBlas::getri. This function performs a // Aqui
/// local (no MPI) dense linear solve on a system of linear equations
/// A * X = B where A is a general N-by-N matrix and X and B are N-by-NRHS
/// matrices.

#ifndef KOKKOSBLAS_GETRI_HPP_
#define KOKKOSBLAS_GETRI_HPP_

#include <type_traits>

#include "KokkosBlas_getri_spec.hpp"
#include "KokkosKernels_Error.hpp"

namespace KokkosBlas {

/// \brief Solve the dense linear equation system A*X = B. // Aqui
///
/// \tparam AMatrix Input matrix/Output LU, as a 2-D Kokkos::View.
/// \tparam BXMV Input (right-hand side)/Output (solution) (multi)vector, as a
/// 1-D or 2-D Kokkos::View. \tparam IPIVV Output pivot indices, as a 1-D
/// Kokkos::View
///
/// \param A [in,out] On entry, the N-by-N matrix to be solved. On exit, the
/// factors L and U from
///   the factorization A = P*L*U; the unit diagonal elements of L are not
///   stored.
/// \param B [in,out] On entry, the right hand side (multi)vector B. On exit,
/// the solution (multi)vector X. \param IPIV [out] On exit, the pivot indices
/// (for partial pivoting). If the View extents are zero and
///   its data pointer is NULL, pivoting is not used.
///
template <class AMatrix, class WORKV, class IPIVV>
void getri(const AMatrix& A, const IPIVV& IPIV, const WORKV& WORK) {
  // NOTE: Currently, KokkosBlas::getri only supports for MAGMA TPL and BLAS TPL.
  //       MAGMA TPL should be enabled to call the MAGMA GPU interface for
  //       device views BLAS TPL should be enabled to call the BLAS interface
  //       for host views

  static_assert(Kokkos::is_view<AMatrix>::value,
                "KokkosBlas::getri: A must be a Kokkos::View.");
  static_assert(Kokkos::is_view<IPIVV>::value,
                "KokkosBlas::getri: IPIV must be a Kokkos::View.");
  static_assert(Kokkos::is_view<WORKV>::value,
                "KokkosBlas::getri: WORK must be a Kokkos::View.");
  static_assert(static_cast<int>(AMatrix::rank) == 2,
                "KokkosBlas::getri: A must have rank 2.");
  static_assert(static_cast<int>(WORKV::rank) == 1,
                "KokkosBlas::getri: WORK must have rank 1.");
  static_assert(static_cast<int>(IPIVV::rank) == 1,
                "KokkosBlas::getri: IPIV must have rank 1.");

  // Check validity of pivot argument
  if ((IPIV.data()    != nullptr                          ) &&
      (IPIV.extent(0) == std::min(A.extent(0),A.extent(1)))) {
    // Ok
  }
  else {
    std::ostringstream os;
    os << "KokkosBlas::getrf(): invalid IPIV";
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  typedef Kokkos::View<
      typename AMatrix::non_const_value_type**, typename AMatrix::array_layout,
      typename AMatrix::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      AMatrix_Internal;
  typedef Kokkos::View<
      typename IPIVV::non_const_value_type*, typename IPIVV::array_layout,
      typename IPIVV::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      IPIVV_Internal;
  typedef Kokkos::View<typename WORKV::non_const_value_type*,
                       typename WORKV::array_layout, typename WORKV::device_type,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      WORKV_Internal;
  AMatrix_Internal A_i = A;
  IPIVV_Internal IPIV_i = IPIV;
  WORKV_Internal WORK_i = WORK;

  KokkosBlas::Impl::GETRI<AMatrix_Internal, WORKV_Internal, IPIVV_Internal>::getri(A_i, IPIV_i, WORK_i);
}

}  // namespace KokkosBlas

#endif  // KOKKOSBLAS_GETRI_HPP_
