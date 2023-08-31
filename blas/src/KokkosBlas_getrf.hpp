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

/// \file KokkosBlas_getrf.hpp
/// \brief Local dense linear solve
///
/// This file provides KokkosBlas::getrf. This function performs a // Aqui
/// local (no MPI) dense linear solve on a system of linear equations
/// A * X = B where A is a general N-by-N matrix and X and B are N-by-NRHS
/// matrices.

#ifndef KOKKOSBLAS_GETRF_HPP_
#define KOKKOSBLAS_GETRF_HPP_

#include <type_traits>

#include "KokkosBlas_getrf_spec.hpp"
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
template <class AMatrix, class IPIVV>
void getrf(const AMatrix& A, const IPIVV& IPIV) {
  // NOTE: Currently, KokkosBlas::getrf only supports for MAGMA TPL and BLAS TPL.
  //       MAGMA TPL should be enabled to call the MAGMA GPU interface for
  //       device views BLAS TPL should be enabled to call the BLAS interface
  //       for host views

  static_assert(Kokkos::is_view<AMatrix>::value,
                "KokkosBlas::getrf: A must be a Kokkos::View.");
  static_assert(Kokkos::is_view<IPIVV>::value,
                "KokkosBlas::getrf: IPIV must be a Kokkos::View.");
  static_assert(static_cast<int>(AMatrix::rank) == 2,
                "KokkosBlas::getrf: A must have rank 2.");
  static_assert(static_cast<int>(IPIVV::rank) == 1,
                "KokkosBlas::getrf: IPIV must have rank 1.");

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
  AMatrix_Internal A_i = A;
  IPIVV_Internal IPIV_i = IPIV;

  KokkosBlas::Impl::GETRF<AMatrix_Internal, IPIVV_Internal>::getrf(A_i, IPIV_i);
}

}  // namespace KokkosBlas

#endif  // KOKKOSBLAS_GETRF_HPP_
