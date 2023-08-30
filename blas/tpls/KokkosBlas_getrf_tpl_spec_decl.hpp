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

#ifndef KOKKOSBLAS_GETRF_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS_GETRF_TPL_SPEC_DECL_HPP_

namespace KokkosBlas {
namespace Impl {
template <class AViewType, class PViewType>
inline void getrf_print_specialization() {
#ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
  printf("KokkosBlas::getrf<> TPL MAGMA specialization for < %s , %s >\n",
         typeid(AViewType).name(), typeid(PViewType).name());
#else
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
  printf("KokkosBlas::getrf<> TPL Blas specialization for < %s , %s >\n",
         typeid(AViewType).name(), typeid(PViewType).name());
#endif
#endif
#endif
}
}  // namespace Impl
}  // namespace KokkosBlas

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#include <KokkosBlas_Host_tpl.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS_DGETRF_BLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                  \
  struct GETRF<                                                               \
      Kokkos::View<double**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<int*, LAYOUT,                                              \
                   Kokkos::Device<Kokkos::DefaultHostExecutionSpace,          \
                                  Kokkos::HostSpace>,                         \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef double SCALAR;                                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        int*, LAYOUT,                                                         \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getrf(const AViewType& A, const PViewType& IPIV) {            \
      Kokkos::Profiling::pushRegion("KokkosBlas::getrf[TPL_BLAS,double]");    \
      getrf_print_specialization<AViewType, PViewType>();                     \
                                                                              \
      const int M = static_cast<int>(A.extent(0));                            \
      const int N = static_cast<int>(A.extent(1));                            \
                                                                              \
      const bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
      const int  AST     = A_is_lr ? A.stride(0) : A.stride(1),               \
      const int  LDA     = (AST == 0) ? 1 : AST;                              \
                                                                              \
      int info = 0;                                                           \
                                                                              \
      HostBlas<double>::getrf(M, N, A.data(), LDA, IPIV.data(), info);        \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_SGETRF_BLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                  \
  struct GETRF<                                                               \
      Kokkos::View<float**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<int*, LAYOUT,                                              \
                   Kokkos::Device<Kokkos::DefaultHostExecutionSpace,          \
                                  Kokkos::HostSpace>,                         \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef float SCALAR;                                                     \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        int*, LAYOUT,                                                         \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getrf(const AViewType& A, const PViewType& IPIV) {            \
      Kokkos::Profiling::pushRegion("KokkosBlas::getrf[TPL_BLAS,float]");     \
      getrf_print_specialization<AViewType, PViewType>();                     \
                                                                              \
      const int M = static_cast<int>(A.extent(0));                            \
      const int N = static_cast<int>(A.extent(1));                            \
                                                                              \
      const bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
      const int  AST     = A_is_lr ? A.stride(0) : A.stride(1),               \
      const int  LDA     = (AST == 0) ? 1 : AST;                              \
                                                                              \
      int info = 0;                                                           \
                                                                              \
      HostBlas<float>::getrf(M, N, A.data(), LDA, IPIV.data(), info);         \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_ZGETRF_BLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                  \
  struct GETRF<Kokkos::View<Kokkos::complex<double>**, LAYOUT,                \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              Kokkos::View<int*, LAYOUT,                                      \
                           Kokkos::Device<Kokkos::DefaultHostExecutionSpace,  \
                                          Kokkos::HostSpace>,                 \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              true, ETI_SPEC_AVAIL> {                                         \
    typedef Kokkos::complex<double> SCALAR;                                   \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        int*, LAYOUT,                                                         \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getrf(const AViewType& A, const PViewType& IPIV) {            \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::getrf[TPL_BLAS,complex<double>]");                     \
      getrf_print_specialization<AViewType, PViewType>();                     \
                                                                              \
      const int M = static_cast<int>(A.extent(0));                            \
      const int N = static_cast<int>(A.extent(1));                            \
                                                                              \
      const bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
      const int  AST     = A_is_lr ? A.stride(0) : A.stride(1),               \
      const int  LDA     = (AST == 0) ? 1 : AST;                              \
                                                                              \
      int info = 0;                                                           \
                                                                              \
      HostBlas<std::complex<double> >::getrf(                                 \
          M, N, reinterpret_cast<std::complex<double>*>(A.data()), LDA,       \
          IPIV.data(), info);                                                 \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_CGETRF_BLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                  \
  struct GETRF<Kokkos::View<Kokkos::complex<float>**, LAYOUT,                 \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              Kokkos::View<int*, LAYOUT,                                      \
                           Kokkos::Device<Kokkos::DefaultHostExecutionSpace,  \
                                          Kokkos::HostSpace>,                 \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              true, ETI_SPEC_AVAIL> {                                         \
    typedef Kokkos::complex<float> SCALAR;                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        int*, LAYOUT,                                                         \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getrf(const AViewType& A, const PViewType& IPIV) {            \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::getrf[TPL_BLAS,complex<float>]");                      \
      getrf_print_specialization<AViewType, PViewType>();                     \
                                                                              \
      const int M = static_cast<int>(A.extent(0));                            \
      const int N = static_cast<int>(A.extent(1));                            \
                                                                              \
      const bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
      const int  AST     = A_is_lr ? A.stride(0) : A.stride(1),               \
      const int  LDA     = (AST == 0) ? 1 : AST;                              \
                                                                              \
      int info = 0;                                                           \
                                                                              \
      HostBlas<std::complex<float> >::getrf(                                  \
          M, N, reinterpret_cast<std::complex<float>*>(A.data()), LDA,        \
          IPIV.data(), info);                                                 \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

KOKKOSBLAS_DGETRF_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_DGETRF_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, false)

KOKKOSBLAS_SGETRF_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_SGETRF_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, false)

KOKKOSBLAS_ZGETRF_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_ZGETRF_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, false)

KOKKOSBLAS_CGETRF_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_CGETRF_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_BLAS

// MAGMA
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS_DGETRF_MAGMA(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)            \
  template <class ExecSpace>                                                  \
  struct GETRF<                                                               \
      Kokkos::View<double**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<magma_int_t*, LAYOUT,                                      \
                   Kokkos::Device<Kokkos::DefaultHostExecutionSpace,          \
                                  Kokkos::HostSpace>,                         \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef double SCALAR;                                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        magma_int_t*, LAYOUT,                                                 \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getrf(const AViewType& A, const PViewType& IPIV) {            \
      Kokkos::Profiling::pushRegion("KokkosBlas::getrf[TPL_MAGMA,double]");   \
      getrf_print_specialization<AViewType, PViewType>();                     \
                                                                              \
      magma_int_t M = static_cast<magma_int_t>(A.extent(0));                  \
      magma_int_t N = static_cast<magma_int_t>(A.extent(1));                  \
                                                                              \
      const bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
      magma_int_t AST    = A_is_lr ? A.stride(0) : A.stride(1),               \
      magma_int_t LDA    = (AST == 0) ? 1 : AST;                              \
                                                                              \
      KokkosBlas::Impl::MagmaSingleton& s =                                   \
          KokkosBlas::Impl::MagmaSingleton::singleton();                      \
      magma_int_t info = 0;                                                   \
                                                                              \
      magma_dgetrf_nopiv_gpu(                                                 \
          M, N, reinterpret_cast<magmaDouble_ptr>(A.data()), LDA, &info);     \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_SGETRF_MAGMA(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)            \
  template <class ExecSpace>                                                  \
  struct GETRF<                                                               \
      Kokkos::View<float**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<magma_int_t*, LAYOUT,                                      \
                   Kokkos::Device<Kokkos::DefaultHostExecutionSpace,          \
                                  Kokkos::HostSpace>,                         \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef float SCALAR;                                                     \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        magma_int_t*, LAYOUT,                                                 \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getrf(const AViewType& A, const PViewType& IPIV) {            \
      Kokkos::Profiling::pushRegion("KokkosBlas::getrf[TPL_MAGMA,float]");    \
      getrf_print_specialization<AViewType, PViewType>();                     \
                                                                              \
      magma_int_t M = static_cast<magma_int_t>(A.extent(0));                  \
      magma_int_t N = static_cast<magma_int_t>(A.extent(1));                  \
                                                                              \
      const bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
      magma_int_t AST    = A_is_lr ? A.stride(0) : A.stride(1),               \
      magma_int_t LDA    = (AST == 0) ? 1 : AST;                              \
                                                                              \
      KokkosBlas::Impl::MagmaSingleton& s =                                   \
          KokkosBlas::Impl::MagmaSingleton::singleton();                      \
      magma_int_t info = 0;                                                   \
                                                                              \
      magma_sgetrf_nopiv_gpu(                                                 \
          M, N, reinterpret_cast<magmaFloat_ptr>(A.data()), LDA, &info);      \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_ZGETRF_MAGMA(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)            \
  template <class ExecSpace>                                                  \
  struct GETRF<Kokkos::View<Kokkos::complex<double>**, LAYOUT,                \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              Kokkos::View<magma_int_t*, LAYOUT,                              \
                           Kokkos::Device<Kokkos::DefaultHostExecutionSpace,  \
                                          Kokkos::HostSpace>,                 \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              true, ETI_SPEC_AVAIL> {                                         \
    typedef Kokkos::complex<double> SCALAR;                                   \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        magma_int_t*, LAYOUT,                                                 \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getrf(const AViewType& A, const PViewType& IPIV) {            \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::getrf[TPL_MAGMA,complex<double>]");                    \
      getrf_print_specialization<AViewType, PViewType>();                     \
                                                                              \
      magma_int_t M = static_cast<magma_int_t>(A.extent(0));                  \
      magma_int_t N = static_cast<magma_int_t>(A.extent(1));                  \
                                                                              \
      const bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
      magma_int_t AST    = A_is_lr ? A.stride(0) : A.stride(1),               \
      magma_int_t LDA    = (AST == 0) ? 1 : AST;                              \
                                                                              \
      KokkosBlas::Impl::MagmaSingleton& s =                                   \
          KokkosBlas::Impl::MagmaSingleton::singleton();                      \
      magma_int_t info = 0;                                                   \
                                                                              \
      magma_zgetrf_nopiv_gpu(M, N,                                            \
          reinterpret_cast<magmaDoubleComplex_ptr>(A.data()), LDA, &info);    \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_CGETRF_MAGMA(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)            \
  template <class ExecSpace>                                                  \
  struct GETRF<Kokkos::View<Kokkos::complex<float>**, LAYOUT,                 \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              Kokkos::View<magma_int_t*, LAYOUT,                              \
                           Kokkos::Device<Kokkos::DefaultHostExecutionSpace,  \
                                          Kokkos::HostSpace>,                 \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              true, ETI_SPEC_AVAIL> {                                         \
    typedef Kokkos::complex<float> SCALAR;                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        magma_int_t*, LAYOUT,                                                 \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getrf(const AViewType& A, const PViewType& IPIV) {            \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::getrf[TPL_MAGMA,complex<float>]");                     \
      getrf_print_specialization<AViewType, PViewType>();                     \
                                                                              \
      magma_int_t M = static_cast<magma_int_t>(A.extent(0));                  \
      magma_int_t N = static_cast<magma_int_t>(A.extent(1));                  \
                                                                              \
      const bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
      magma_int_t AST    = A_is_lr ? A.stride(0) : A.stride(1),               \
      magma_int_t LDA    = (AST == 0) ? 1 : AST;                              \
                                                                              \
      KokkosBlas::Impl::MagmaSingleton& s =                                   \
          KokkosBlas::Impl::MagmaSingleton::singleton();                      \
      magma_int_t info = 0;                                                   \
                                                                              \
      magma_cgetrf_nopiv_gpu(                                                 \
          M, N, reinterpret_cast<magmaFloatComplex_ptr>(A.data()), &info);    \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

KOKKOSBLAS_DGETRF_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_DGETRF_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_SGETRF_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_SGETRF_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_ZGETRF_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_ZGETRF_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_CGETRF_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_CGETRF_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_MAGMA

#endif
