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

#ifndef KOKKOSLAPACK_GEQRF_TPL_SPEC_DECL_HPP_
#define KOKKOSLAPACK_GEQRF_TPL_SPEC_DECL_HPP_

namespace KokkosLapack {
namespace Impl {
template <class AViewType, class TauViewType, class InfoViewType>
inline void geqrf_print_specialization() {
#ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
  printf("KokkosLapack::geqrf<> TPL MAGMA specialization for < %s , %s, %s >\n",
         typeid(AViewType).name(), typeid(TauViewType).name(),
         typeid(InfoViewType).name());
#else
#ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK
  printf(
      "KokkosLapack::geqrf<> TPL Lapack specialization for < %s , %s, %s >\n",
      typeid(AViewType).name(), typeid(TauViewType).name(),
      typeid(InfoViewType).name());
#endif
#endif
#endif
}
}  // namespace Impl
}  // namespace KokkosLapack

// Generic Host side LAPACK (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK
#include <KokkosLapack_Host_tpl.hpp>

namespace KokkosLapack {
namespace Impl {

template <class AViewType, class TauViewType, class InfoViewType>
void lapackGeqrfWrapper(const AViewType& A, const TauViewType& Tau,
                        const InfoViewType& Info) {
  using memory_space = typename AViewType::memory_space;
  using Scalar       = typename AViewType::non_const_value_type;
  using ALayout_t    = typename AViewType::array_layout;
  static_assert(std::is_same_v<ALayout_t, Kokkos::LayoutLeft>,
                "KokkosLapack - geqrf: A needs to have a Kokkos::LayoutLeft");
  const int m   = A.extent_int(0);
  const int n   = A.extent_int(1);
  const int lda = A.stride(1);

  int lwork = -1;
  Kokkos::View<Scalar*, memory_space> work("geqrf work buffer", 1);

  if constexpr (Kokkos::ArithTraits<Scalar>::is_complex) {
    using MagType = typename Kokkos::ArithTraits<Scalar>::mag_type;

    HostLapack<std::complex<MagType>>::geqrf(
        m, n, reinterpret_cast<std::complex<MagType>*>(A.data()), lda,
        reinterpret_cast<std::complex<MagType>*>(Tau.data()),
        reinterpret_cast<std::complex<MagType>*>(work.data()), lwork,
        Info.data());

    if (Info[0] < 0) return;

    lwork = static_cast<int>(work(0).real());

    work = Kokkos::View<Scalar*, memory_space>("geqrf work buffer", lwork);

    HostLapack<std::complex<MagType>>::geqrf(
        m, n, reinterpret_cast<std::complex<MagType>*>(A.data()), lda,
        reinterpret_cast<std::complex<MagType>*>(Tau.data()),
        reinterpret_cast<std::complex<MagType>*>(work.data()), lwork,
        Info.data());
  } else {
    HostLapack<Scalar>::geqrf(m, n, A.data(), lda, Tau.data(), work.data(),
                              lwork, Info.data());

    if (Info[0] < 0) return;

    lwork = static_cast<int>(work(0));

    work = Kokkos::View<Scalar*, memory_space>("geqrf work buffer", lwork);

    HostLapack<Scalar>::geqrf(m, n, A.data(), lda, Tau.data(), work.data(),
                              lwork, Info.data());
  }
}

#define KOKKOSLAPACK_GEQRF_LAPACK(SCALAR, LAYOUT, EXECSPACE, MEM_SPACE)        \
  template <>                                                                  \
  struct GEQRF<                                                                \
      EXECSPACE,                                                               \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      Kokkos::View<int*, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>,         \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      true,                                                                    \
      geqrf_eti_spec_avail<                                                    \
          EXECSPACE,                                                           \
          Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>, \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>,               \
          Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>,  \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>,               \
          Kokkos::View<int*, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>,     \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>>::value> {     \
    using AViewType =                                                          \
        Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>,   \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    using TauViewType =                                                        \
        Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>,    \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    using InfoViewType =                                                       \
        Kokkos::View<int*, LAYOUT, Kokkos::Device<EXECSPACE, MEM_SPACE>,       \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
                                                                               \
    static void geqrf(const EXECSPACE& /* space */, const AViewType& A,        \
                      const TauViewType& Tau, const InfoViewType& Info) {      \
      Kokkos::Profiling::pushRegion("KokkosLapack::geqrf[TPL_LAPACK," #SCALAR  \
                                    "]");                                      \
      geqrf_print_specialization<AViewType, TauViewType, InfoViewType>();      \
      lapackGeqrfWrapper(A, Tau, Info);                                        \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#if defined(KOKKOS_ENABLE_SERIAL)
KOKKOSLAPACK_GEQRF_LAPACK(float, Kokkos::LayoutLeft, Kokkos::Serial,
                          Kokkos::HostSpace)
KOKKOSLAPACK_GEQRF_LAPACK(double, Kokkos::LayoutLeft, Kokkos::Serial,
                          Kokkos::HostSpace)
KOKKOSLAPACK_GEQRF_LAPACK(Kokkos::complex<float>, Kokkos::LayoutLeft,
                          Kokkos::Serial, Kokkos::HostSpace)
KOKKOSLAPACK_GEQRF_LAPACK(Kokkos::complex<double>, Kokkos::LayoutLeft,
                          Kokkos::Serial, Kokkos::HostSpace)
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
KOKKOSLAPACK_GEQRF_LAPACK(float, Kokkos::LayoutLeft, Kokkos::OpenMP,
                          Kokkos::HostSpace)
KOKKOSLAPACK_GEQRF_LAPACK(double, Kokkos::LayoutLeft, Kokkos::OpenMP,
                          Kokkos::HostSpace)
KOKKOSLAPACK_GEQRF_LAPACK(Kokkos::complex<float>, Kokkos::LayoutLeft,
                          Kokkos::OpenMP, Kokkos::HostSpace)
KOKKOSLAPACK_GEQRF_LAPACK(Kokkos::complex<double>, Kokkos::LayoutLeft,
                          Kokkos::OpenMP, Kokkos::HostSpace)
#endif

#if defined(KOKKOS_ENABLE_THREADS)
KOKKOSLAPACK_GEQRF_LAPACK(float, Kokkos::LayoutLeft, Kokkos::Threads,
                          Kokkos::HostSpace)
KOKKOSLAPACK_GEQRF_LAPACK(double, Kokkos::LayoutLeft, Kokkos::Threads,
                          Kokkos::HostSpace)
KOKKOSLAPACK_GEQRF_LAPACK(Kokkos::complex<float>, Kokkos::LayoutLeft,
                          Kokkos::Threads, Kokkos::HostSpace)
KOKKOSLAPACK_GEQRF_LAPACK(Kokkos::complex<double>, Kokkos::LayoutLeft,
                          Kokkos::Threads, Kokkos::HostSpace)
#endif

}  // namespace Impl
}  // namespace KokkosLapack
#endif  // KOKKOSKERNELS_ENABLE_TPL_LAPACK

#if 0  // AquiEEP

// MAGMA
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include <KokkosLapack_magma.hpp>

namespace KokkosLapack {
namespace Impl {

template <class ExecSpace, class AViewType, class TauViewType>
void magmaGeqrfWrapper(const ExecSpace& space, const AViewType& A,
                      const TauViewType& Tau, const InfoViewType& Info) {
  using scalar_type = typename AViewType::non_const_value_type;

  Kokkos::Profiling::pushRegion("KokkosLapack::geqrf[TPL_MAGMA," +
                                Kokkos::ArithTraits<scalar_type>::name() + "]");
  geqrf_print_specialization<AViewType, TauViewType, InfoViewType>();

  magma_int_t N    = static_cast<magma_int_t>(A.extent(1));
  magma_int_t AST  = static_cast<magma_int_t>(A.stride(1));
  magma_int_t LDA  = (AST == 0) ? 1 : AST;
  magma_int_t BST  = static_cast<magma_int_t>(B.stride(1));
  magma_int_t LDB  = (BST == 0) ? 1 : BST;
  magma_int_t NRHS = static_cast<magma_int_t>(B.extent(1));

  KokkosLapack::Impl::MagmaSingleton& s =
      KokkosLapack::Impl::MagmaSingleton::singleton();
  magma_int_t info = 0;

  space.fence();
  if constexpr (std::is_same_v<scalar_type, float>) {
      magma_sgeqrf_nopiv_gpu(N, NRHS, reinterpret_cast<magmaFloat_ptr>(A.data()),
                            LDA, reinterpret_cast<magmaFloat_ptr>(B.data()),
                            LDB, &info);
  }

  if constexpr (std::is_same_v<scalar_type, double>) {
      magma_dgeqrf_nopiv_gpu(
          N, NRHS, reinterpret_cast<magmaDouble_ptr>(A.data()), LDA,
          reinterpret_cast<magmaDouble_ptr>(B.data()), LDB, &info);
  }

  if constexpr (std::is_same_v<scalar_type, Kokkos::complex<float>>) {
      magma_cgeqrf_nopiv_gpu(
          N, NRHS, reinterpret_cast<magmaFloatComplex_ptr>(A.data()), LDA,
          reinterpret_cast<magmaFloatComplex_ptr>(B.data()), LDB, &info);
  }

  if constexpr (std::is_same_v<scalar_type, Kokkos::complex<double>>) {
      magma_zgeqrf_nopiv_gpu(
          N, NRHS, reinterpret_cast<magmaDoubleComplex_ptr>(A.data()), LDA,
          reinterpret_cast<magmaDoubleComplex_ptr>(B.data()), LDB, &info);
  }
  ExecSpace().fence();
  Kokkos::Profiling::popRegion();
}

#define KOKKOSLAPACK_GEQRF_MAGMA(SCALAR, LAYOUT, MEM_SPACE)                    \
  template <>                                                                  \
  struct GEQRF<                                                                \
      Kokkos::Cuda,                                                            \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      true,                                                                    \
      geqrf_eti_spec_avail<                                                    \
          Kokkos::Cuda,                                                        \
          Kokkos::View<SCALAR**, LAYOUT,                                       \
                       Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,                \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>,               \
          Kokkos::View<SCALAR*, LAYOUT,                                        \
                       Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,                \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>>::value> {     \
    using AViewType = Kokkos::View<SCALAR**, LAYOUT,                           \
                                   Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,    \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;   \
    using TauViewType =                                                        \
        Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>, \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
                                                                               \
    static void geqrf(const Kokkos::Cuda& space, const AViewType& A,           \
                      const TauViewType& Tau, const InfoViewType& Info) {      \
      magmaGeqrfWrapper(space, A, Tau, Info);                                  \
    }                                                                          \
  };

KOKKOSLAPACK_GEQRF_MAGMA(float, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSLAPACK_GEQRF_MAGMA(double, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSLAPACK_GEQRF_MAGMA(Kokkos::complex<float>, Kokkos::LayoutLeft,
                        Kokkos::CudaSpace)
KOKKOSLAPACK_GEQRF_MAGMA(Kokkos::complex<double>, Kokkos::LayoutLeft,
                        Kokkos::CudaSpace)

}  // namespace Impl
}  // namespace KokkosLapack
#endif  // KOKKOSKERNELS_ENABLE_TPL_MAGMA

#endif  // AquiEEP

// CUSOLVER
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSOLVER
#include "KokkosLapack_cusolver.hpp"

namespace KokkosLapack {
namespace Impl {

template <class ExecutionSpace, class AViewType, class TauViewType,
          class InfoViewType>
void cusolverGeqrfWrapper(const ExecutionSpace& space, const AViewType& A,
                          const TauViewType& Tau, const InfoViewType& Info) {
  using memory_space = typename AViewType::memory_space;
  using Scalar       = typename AViewType::non_const_value_type;

  using ALayout_t = typename AViewType::array_layout;
  static_assert(
      std::is_same_v<ALayout_t, Kokkos::LayoutLeft>,
      "KokkosLapack - cusolver geqrf: A needs to have a Kokkos::LayoutLeft");
  const int m   = A.extent_int(0);
  const int n   = A.extent_int(1);
  const int lda = A.stride(1);
  int lwork     = 0;

  CudaLapackSingleton& s = CudaLapackSingleton::singleton();
  KOKKOS_CUSOLVER_SAFE_CALL_IMPL(
      cusolverDnSetStream(s.handle, space.cuda_stream()));
  if constexpr (std::is_same_v<Scalar, float>) {
    KOKKOS_CUSOLVER_SAFE_CALL_IMPL(
        cusolverDnSgeqrf_bufferSize(s.handle, m, n, A.data(), lda, &lwork));
    Kokkos::View<float*, memory_space> Workspace("cusolver sgeqrf workspace",
                                                 lwork);

    KOKKOS_CUSOLVER_SAFE_CALL_IMPL(
        cusolverDnSgeqrf(s.handle, m, n, A.data(), lda, Tau.data(),
                         Workspace.data(), lwork, Info.data()));
  }
  if constexpr (std::is_same_v<Scalar, double>) {
    KOKKOS_CUSOLVER_SAFE_CALL_IMPL(
        cusolverDnDgeqrf_bufferSize(s.handle, m, n, A.data(), lda, &lwork));
    Kokkos::View<double*, memory_space> Workspace("cusolver dgeqrf workspace",
                                                  lwork);

    KOKKOS_CUSOLVER_SAFE_CALL_IMPL(
        cusolverDnDgeqrf(s.handle, m, n, A.data(), lda, Tau.data(),
                         Workspace.data(), lwork, Info.data()));
  }
  if constexpr (std::is_same_v<Scalar, Kokkos::complex<float>>) {
    KOKKOS_CUSOLVER_SAFE_CALL_IMPL(cusolverDnCgeqrf_bufferSize(
        s.handle, m, n, reinterpret_cast<cuComplex*>(A.data()), lda, &lwork));
    Kokkos::View<cuComplex*, memory_space> Workspace(
        "cusolver cgeqrf workspace", lwork);

    KOKKOS_CUSOLVER_SAFE_CALL_IMPL(cusolverDnCgeqrf(
        s.handle, m, n, reinterpret_cast<cuComplex*>(A.data()), lda,
        reinterpret_cast<cuComplex*>(Tau.data()),
        reinterpret_cast<cuComplex*>(Workspace.data()), lwork, Info.data()));
  }
  if constexpr (std::is_same_v<Scalar, Kokkos::complex<double>>) {
    KOKKOS_CUSOLVER_SAFE_CALL_IMPL(cusolverDnZgeqrf_bufferSize(
        s.handle, m, n, reinterpret_cast<cuDoubleComplex*>(A.data()), lda,
        &lwork));
    Kokkos::View<cuDoubleComplex*, memory_space> Workspace(
        "cusolver zgeqrf workspace", lwork);

    KOKKOS_CUSOLVER_SAFE_CALL_IMPL(cusolverDnZgeqrf(
        s.handle, m, n, reinterpret_cast<cuDoubleComplex*>(A.data()), lda,
        reinterpret_cast<cuDoubleComplex*>(Tau.data()),
        reinterpret_cast<cuDoubleComplex*>(Workspace.data()), lwork,
        Info.data()));
  }
  KOKKOS_CUSOLVER_SAFE_CALL_IMPL(cusolverDnSetStream(s.handle, NULL));
}

#define KOKKOSLAPACK_GEQRF_CUSOLVER(SCALAR, LAYOUT, MEM_SPACE)                 \
  template <>                                                                  \
  struct GEQRF<                                                                \
      Kokkos::Cuda,                                                            \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      Kokkos::View<int*, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      true,                                                                    \
      geqrf_eti_spec_avail<                                                    \
          Kokkos::Cuda,                                                        \
          Kokkos::View<SCALAR**, LAYOUT,                                       \
                       Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,                \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>,               \
          Kokkos::View<SCALAR*, LAYOUT,                                        \
                       Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,                \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>,               \
          Kokkos::View<int*, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,  \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>>::value> {     \
    using AViewType = Kokkos::View<SCALAR**, LAYOUT,                           \
                                   Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,    \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;   \
    using TauViewType =                                                        \
        Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>, \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    using InfoViewType =                                                       \
        Kokkos::View<int*, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEM_SPACE>,    \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
                                                                               \
    static void geqrf(const Kokkos::Cuda& space, const AViewType& A,           \
                      const TauViewType& Tau, const InfoViewType& Info) {      \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosLapack::geqrf[TPL_CUSOLVER," #SCALAR "]");                    \
      geqrf_print_specialization<AViewType, TauViewType, InfoViewType>();      \
                                                                               \
      cusolverGeqrfWrapper(space, A, Tau, Info);                               \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

KOKKOSLAPACK_GEQRF_CUSOLVER(float, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSLAPACK_GEQRF_CUSOLVER(double, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSLAPACK_GEQRF_CUSOLVER(Kokkos::complex<float>, Kokkos::LayoutLeft,
                            Kokkos::CudaSpace)
KOKKOSLAPACK_GEQRF_CUSOLVER(Kokkos::complex<double>, Kokkos::LayoutLeft,
                            Kokkos::CudaSpace)

#if defined(KOKKOSKERNELS_INST_MEMSPACE_CUDAUVMSPACE)
KOKKOSLAPACK_GEQRF_CUSOLVER(float, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSLAPACK_GEQRF_CUSOLVER(double, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSLAPACK_GEQRF_CUSOLVER(Kokkos::complex<float>, Kokkos::LayoutLeft,
                            Kokkos::CudaUVMSpace)
KOKKOSLAPACK_GEQRF_CUSOLVER(Kokkos::complex<double>, Kokkos::LayoutLeft,
                            Kokkos::CudaUVMSpace)
#endif

}  // namespace Impl
}  // namespace KokkosLapack
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUSOLVER

// ROCSOLVER
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER
#include <KokkosBlas_tpl_spec.hpp>
#include <rocsolver/rocsolver.h>

namespace KokkosLapack {
namespace Impl {

  template <class ExecutionSpace, class AViewType, class TauViewType, class InfoViewType>
void rocsolverGeqrfWrapper(const ExecutionSpace& space, const AViewType& A,
                           const TauViewType& Tau, const InfoViewType& Info) {
  using Scalar = typename AViewType::non_const_value_type;

  using ALayout_t = typename AViewType::array_layout;
  static_assert(
      std::is_same_v<ALayout_t, Kokkos::LayoutLeft>,
      "KokkosLapack - rocsolver geqrf: A needs to have a Kokkos::LayoutLeft");
  const rocblas_int m   = static_cast<rocblas_int>(A.extent(0));
  const rocblas_int n   = static_cast<rocblas_int>(A.extent(1));
  const rocblas_int lda = static_cast<rocblas_int>(A.stride(1));

  KokkosBlas::Impl::RocBlasSingleton& s =
      KokkosBlas::Impl::RocBlasSingleton::singleton();
  KOKKOS_ROCBLAS_SAFE_CALL_IMPL(
      rocblas_set_stream(s.handle, space.hip_stream()));
  if constexpr (std::is_same_v<Scalar, float>) {
    KOKKOS_ROCBLAS_SAFE_CALL_IMPL(
        rocsolver_sgeqrf(s.handle, m, n, A.data(), lda, Tau.data()));
  }
  if constexpr (std::is_same_v<Scalar, double>) {
    KOKKOS_ROCBLAS_SAFE_CALL_IMPL(
        rocsolver_dgeqrf(s.handle, m, n, A.data(), lda, Tau.data()));
  }
  if constexpr (std::is_same_v<Scalar, Kokkos::complex<float>>) {
    KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocsolver_cgeqrf(
        s.handle, m, n, reinterpret_cast<rocblas_float_complex*>(A.data()), lda,
        reinterpret_cast<rocblas_float_complex*>(Tau.data())));
  }
  if constexpr (std::is_same_v<Scalar, Kokkos::complex<double>>) {
    KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocsolver_zgeqrf(
        s.handle, m, n, reinterpret_cast<rocblas_double_complex*>(A.data()),
        lda, reinterpret_cast<rocblas_double_complex*>(Tau.data())));
  }
  Info[0] = 0; // success
  KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_set_stream(s.handle, NULL));
}

#define KOKKOSLAPACK_GEQRF_ROCSOLVER(SCALAR, LAYOUT, MEM_SPACE)                \
  template <>                                                                  \
  struct GEQRF<                                                                \
      Kokkos::HIP,                                                             \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      Kokkos::View<int*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      true,                                                                    \
      geqrf_eti_spec_avail<                                                    \
          Kokkos::HIP,                                                         \
          Kokkos::View<SCALAR**, LAYOUT,                                       \
                       Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                 \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>,               \
          Kokkos::View<SCALAR*, LAYOUT,                                        \
                       Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                 \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>,               \
          Kokkos::View<int*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,   \
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>>::value> {     \
    using AViewType =                                                          \
        Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>, \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    using TauViewType =                                                        \
        Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,  \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
    using InfoViewType =                                                       \
        Kokkos::View<int*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,     \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
                                                                               \
    static void geqrf(const Kokkos::HIP& space, const AViewType& A,            \
                      const TauViewType& Tau, const InfoViewType& Info) {      \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosLapack::geqrf[TPL_ROCSOLVER," #SCALAR "]");                   \
      geqrf_print_specialization<AViewType, TauViewType, InfoViewType>();      \
                                                                               \
      rocsolverGeqrfWrapper(space, A, Tau, Info);                              \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

KOKKOSLAPACK_GEQRF_ROCSOLVER(float, Kokkos::LayoutLeft, Kokkos::HIPSpace)
KOKKOSLAPACK_GEQRF_ROCSOLVER(double, Kokkos::LayoutLeft, Kokkos::HIPSpace)
KOKKOSLAPACK_GEQRF_ROCSOLVER(Kokkos::complex<float>, Kokkos::LayoutLeft,
                             Kokkos::HIPSpace)
KOKKOSLAPACK_GEQRF_ROCSOLVER(Kokkos::complex<double>, Kokkos::LayoutLeft,
                             Kokkos::HIPSpace)

}  // namespace Impl
}  // namespace KokkosLapack
#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER

#endif
