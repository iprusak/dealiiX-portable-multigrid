#ifndef nvtx_profiling_h
#define nvtx_profiling_h

/**
 * Thin wrapper around NVTX3. All macros collapse to no-ops unless NVTX is
 * available, so instrumentation can stay in the sources unconditionally.
 *
 * Enabled when compiling with nvcc/nvc++ and <nvtx3/nvToolsExt.h> is found.
 * Override with -DDEALIIX_FORCE_NVTX=1 (host builds) or
 * -DDEALIIX_DISABLE_NVTX (off everywhere). NVTX3 is header-only; no linking.
 *
 * Note: ranges are host-side. A plain range around an asynchronous Kokkos
 * launch measures the launch, not the kernel; use the _SYNC variants (they
 * fence before popping) when the range should span the device work.
 */

#if defined(DEALIIX_DISABLE_NVTX)
#  define DEALIIX_WITH_NVTX 0
#elif defined(DEALIIX_FORCE_NVTX) && DEALIIX_FORCE_NVTX
#  define DEALIIX_WITH_NVTX 1
#elif defined(__CUDACC__) || defined(__NVCOMPILER)
#  if defined(__has_include)
#    if __has_include(<nvtx3/nvToolsExt.h>)
#      define DEALIIX_WITH_NVTX 1
#    else
#      define DEALIIX_WITH_NVTX 0
#    endif
#  else
#    define DEALIIX_WITH_NVTX 1
#  endif
#else
#  define DEALIIX_WITH_NVTX 0
#endif


#if DEALIIX_WITH_NVTX

#  include <nvtx3/nvToolsExt.h>

#  include <Kokkos_Core.hpp>

#  include <cstdint>
#  include <map>
#  include <string>
#  include <utility>

#  if defined(__linux__)
#    include <sys/syscall.h>
#    include <unistd.h>
#  endif

namespace dealiiX::nvtx
{
  // ARGB palette; one hue per kind of operation so the Nsight timeline is
  // readable without reading the labels.
  namespace color
  {
    constexpr std::uint32_t level      = 0xff37474f; // blue grey
    constexpr std::uint32_t smoother   = 0xffff9800; // orange
    constexpr std::uint32_t matvec     = 0xff2196f3; // blue
    constexpr std::uint32_t restrict_  = 0xff4caf50; // green
    constexpr std::uint32_t prolongate = 0xff9c27b0; // purple
    constexpr std::uint32_t coarse     = 0xffe53935; // red
    constexpr std::uint32_t solver     = 0xff009688; // teal
    constexpr std::uint32_t comm       = 0xffffc107; // amber
    constexpr std::uint32_t setup      = 0xff795548; // brown
    constexpr std::uint32_t other      = 0xff607d8b; // grey
  } // namespace color

  inline nvtxDomainHandle_t
  domain()
  {
    static nvtxDomainHandle_t d = nvtxDomainCreateA("dealiiX");
    return d;
  }

  /**
   * Register @p name (suffixed with "[L<level>]" when @p level >= 0) once and
   * cache the handle, so the hot path neither formats nor allocates. Keyed on
   * the literal's address, which is why only string literals may be passed.
   */
  inline nvtxStringHandle_t
  registered_string(const char *name, const int level)
  {
    thread_local std::map<std::pair<const char *, int>, nvtxStringHandle_t> cache;

    const auto key = std::make_pair(name, level);
    auto       it  = cache.find(key);
    if (it == cache.end())
      {
        std::string text(name);
        if (level >= 0)
          text += " [L" + std::to_string(level) + "]";
        it = cache.emplace(key, nvtxDomainRegisterStringA(domain(), text.c_str())).first;
      }
    return it->second;
  }

  inline void
  push(const char *name, const std::uint32_t argb, const int level)
  {
    nvtxEventAttributes_t attr  = {};
    attr.version                = NVTX_VERSION;
    attr.size                   = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType              = NVTX_COLOR_ARGB;
    attr.color                  = argb;
    attr.messageType            = NVTX_MESSAGE_TYPE_REGISTERED;
    attr.message.registered     = registered_string(name, level);
    nvtxDomainRangePushEx(domain(), &attr);
  }

  inline void
  pop()
  {
    nvtxDomainRangePop(domain());
  }

  inline void
  mark(const char *name, const std::uint32_t argb, const int level)
  {
    nvtxEventAttributes_t attr = {};
    attr.version               = NVTX_VERSION;
    attr.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType             = NVTX_COLOR_ARGB;
    attr.color                 = argb;
    attr.messageType           = NVTX_MESSAGE_TYPE_REGISTERED;
    attr.message.registered    = registered_string(name, level);
    nvtxDomainMarkEx(domain(), &attr);
  }

  /// Label the calling thread, e.g. with the MPI rank.
  inline void
  name_thread(const std::string &name)
  {
#  if defined(__linux__)
    nvtxNameOsThreadA(static_cast<std::uint32_t>(::syscall(SYS_gettid)), name.c_str());
#  else
    (void)name;
#  endif
  }

  class ScopedRange
  {
  public:
    ScopedRange(const char         *name,
                const std::uint32_t argb  = color::other,
                const int           level = -1,
                const bool          sync  = false)
      : sync(sync)
    {
      push(name, argb, level);
    }

    ~ScopedRange()
    {
      if (sync)
        Kokkos::fence();
      pop();
    }

    ScopedRange(const ScopedRange &)            = delete;
    ScopedRange &operator=(const ScopedRange &) = delete;

  private:
    const bool sync;
  };
} // namespace dealiiX::nvtx

#  define DEALIIX_NVTX_JOIN_(a, b) a##b
#  define DEALIIX_NVTX_JOIN(a, b) DEALIIX_NVTX_JOIN_(a, b)
#  define DEALIIX_NVTX_VAR DEALIIX_NVTX_JOIN(dealiix_nvtx_range_, __LINE__)

/// Range over the enclosing scope. @p name must be a string literal.
#  define NVTX_RANGE(name, argb) \
    const ::dealiiX::nvtx::ScopedRange DEALIIX_NVTX_VAR(name, argb, -1, false)

/// As NVTX_RANGE, labelled with the multigrid level.
#  define NVTX_RANGE_LEVEL(name, argb, level) \
    const ::dealiiX::nvtx::ScopedRange DEALIIX_NVTX_VAR(name, argb, static_cast<int>(level), false)

/// Kokkos::fence() before closing, so the range spans the device work.
#  define NVTX_RANGE_SYNC(name, argb) \
    const ::dealiiX::nvtx::ScopedRange DEALIIX_NVTX_VAR(name, argb, -1, true)

#  define NVTX_RANGE_LEVEL_SYNC(name, argb, level) \
    const ::dealiiX::nvtx::ScopedRange DEALIIX_NVTX_VAR(name, argb, static_cast<int>(level), true)

/// Manual push/pop for ranges that do not match a scope.
#  define NVTX_PUSH(name, argb) ::dealiiX::nvtx::push(name, argb, -1)
#  define NVTX_PUSH_LEVEL(name, argb, level) \
    ::dealiiX::nvtx::push(name, argb, static_cast<int>(level))
#  define NVTX_POP() ::dealiiX::nvtx::pop()

/// Instantaneous event.
#  define NVTX_MARK(name, argb) ::dealiiX::nvtx::mark(name, argb, -1)
#  define NVTX_MARK_LEVEL(name, argb, level) \
    ::dealiiX::nvtx::mark(name, argb, static_cast<int>(level))

#  define NVTX_NAME_THREAD(name) ::dealiiX::nvtx::name_thread(name)

#else // DEALIIX_WITH_NVTX

#  include <cstdint>

namespace dealiiX::nvtx
{
  namespace color
  {
    constexpr std::uint32_t level      = 0u;
    constexpr std::uint32_t smoother   = 0u;
    constexpr std::uint32_t matvec     = 0u;
    constexpr std::uint32_t restrict_  = 0u;
    constexpr std::uint32_t prolongate = 0u;
    constexpr std::uint32_t coarse     = 0u;
    constexpr std::uint32_t solver     = 0u;
    constexpr std::uint32_t comm       = 0u;
    constexpr std::uint32_t setup      = 0u;
    constexpr std::uint32_t other      = 0u;
  } // namespace color

  /// Swallows the macro arguments so they do not look unused when NVTX is off.
  template <typename... Ts>
  constexpr void
  unused(const Ts &...)
  {}
} // namespace dealiiX::nvtx

#  define NVTX_RANGE(name, argb) ::dealiiX::nvtx::unused(name, argb)
#  define NVTX_RANGE_LEVEL(name, argb, level) ::dealiiX::nvtx::unused(name, argb, level)
#  define NVTX_RANGE_SYNC(name, argb) ::dealiiX::nvtx::unused(name, argb)
#  define NVTX_RANGE_LEVEL_SYNC(name, argb, level) ::dealiiX::nvtx::unused(name, argb, level)
#  define NVTX_PUSH(name, argb) ::dealiiX::nvtx::unused(name, argb)
#  define NVTX_PUSH_LEVEL(name, argb, level) ::dealiiX::nvtx::unused(name, argb, level)
#  define NVTX_POP() ((void)0)
#  define NVTX_MARK(name, argb) ::dealiiX::nvtx::unused(name, argb)
#  define NVTX_MARK_LEVEL(name, argb, level) ::dealiiX::nvtx::unused(name, argb, level)
#  define NVTX_NAME_THREAD(name) ::dealiiX::nvtx::unused(name)

#endif // DEALIIX_WITH_NVTX

#endif
