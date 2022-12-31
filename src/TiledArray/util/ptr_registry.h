/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2022  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Eduard Valeyev
 *  Department of Chemistry, Virginia Tech
 *
 *  util/ptr_registry.h
 *  December 31, 2022
 *
 */

#ifndef TILEDARRAY_UTIL_PTR_REGISTRY_H
#define TILEDARRAY_UTIL_PTR_REGISTRY_H

#include <iosfwd>
#include <map>
#include <string>
#include <unordered_map>

namespace TiledArray {

/// Registry of pointers

/// Stores {pointer,creation_context} pairs in a hash table (one table per
/// pointer "size"; hash tables are stored in the order of increasing size).
///
/// @details Useful for capturing graphs of pointers to detect issues
/// with smart pointer lifetimes (e.g., stray refs and/or cycles) and
/// tracing memory use
struct PtrRegistry {
  using ptr_container_type = std::unordered_map<void*, std::string>;
  using sized_ptr_container_type = std::map<std::size_t, ptr_container_type>;

  /// constructs empty registry
  PtrRegistry();
  PtrRegistry(const PtrRegistry&) = delete;
  PtrRegistry(PtrRegistry&&) = delete;
  PtrRegistry& operator=(const PtrRegistry&) = delete;
  PtrRegistry& operator=(PtrRegistry&&) = delete;

  /// constructs an empty registry configured to log insert/erase
  /// events to \p log
  /// \param log an ostream for logging insert/erase events
  PtrRegistry(std::ostream& log);

  ~PtrRegistry();

  /// sets the active logger to `*log`
  /// \param log pointer to an std::ostream to use for logging insert/erase
  /// events; if null, will do no logging
  PtrRegistry& log(std::ostream* log);

  /// @return pointer to the active logger; if null, no logging is performed
  std::ostream* log() const;

  /// @return reference to the size-ordered containers of sized pointers
  const sized_ptr_container_type& sized_ptrs() const;

  /// @return reference to the container of unsized (0-sized) pointers
  const ptr_container_type& unsized_ptrs() const;

  /// @return total number of pointers in the registry
  std::size_t size() const;

  /// inserts \p ptr associated with size \p sz to the registry
  /// \param ptr pointer to register
  /// \param sz size of the object pointed to by \p ptr
  /// \param context creation context; stored alongside the pointer
  /// \return `*this`
  PtrRegistry& insert(void* ptr, std::size_t sz,
                      const std::string& context = "");

  /// inserts \p ptr without associated size (i.e. `sz=0`) to the registry
  /// \param ptr pointer to register
  /// \param context creation context; stored alongside the pointer
  /// \return `*this`
  /// \note equivalent to `this->insert(ptr, 0, context);
  PtrRegistry& insert(void* ptr, const std::string& context = "");

  /// inserts \p ptr associated with size \p sz to the registry,
  /// appends backtrace of the caller to context separated by `:::::`
  /// \param ptr pointer to register
  /// \param sz size of the object pointed to by \p ptr
  /// \param context creation context; stored alongside the pointer
  /// \return `*this`
  PtrRegistry& insert_bt(void* ptr, std::size_t sz,
                         const std::string& context = "");

  /// inserts \p ptr without associated size (i.e. with size=0) to the registry,
  /// appends backtrace of the caller to context separated by `:::::`
  /// \param ptr pointer to register
  /// \param context creation context; stored alongside the pointer
  /// \return `*this`
  PtrRegistry& insert_bt(void* ptr, const std::string& context = "");

  /// erases \p ptr associated with size \p sz from the registry
  /// \param ptr pointer to erase
  /// \param sz size of the object pointed to by \p ptr
  /// \param context erasure context; if logging, will append this to the log
  /// \return `*this`
  PtrRegistry& erase(void* ptr, std::size_t sz,
                     const std::string& context = "");

  /// erases \p ptr without associated size (i.e., `sz=0`) from the registry
  /// \param ptr pointer to erase
  /// \param context erasure context; if logging, will append this to the log
  /// \return `*this`
  /// \note equivalent to `this->erase(ptr, 0, context);
  PtrRegistry& erase(void* ptr, const std::string& context = "");

  /// erases \p ptr associated with size \p sz from the registry
  /// \details introduced for symmetry with insert_bt() )
  /// \param ptr pointer to erase
  /// \param sz size of the object pointed to by \p ptr
  /// \param context erasure context; if logging, will append this and the
  /// backtrace of the caller to the log \return `*this`
  PtrRegistry& erase_bt(void* ptr, std::size_t sz,
                        const std::string& context = "");

  /// erases \p ptr without associated size (i.e. with `sz=0`) from the registry
  /// \details introduced for symmetry with insert_bt() )
  /// \param ptr pointer to erase
  /// \param sz size of the object pointed to by \p ptr
  /// \param context erasure context; if logging, will append this and the
  /// backtrace of the caller to the log \return `*this` \note equivalent to
  /// `this->erase_bt(ptr, 0, context);
  PtrRegistry& erase_bt(void* ptr, const std::string& context = "");

 private:
  std::ostream* log_ = nullptr;
  sized_ptr_container_type ptrs_;
  mutable ptr_container_type* unsized_ptrs_ = nullptr;  // &(ptrs_[0])
  std::mutex mtx_;

  /// inserts \p ptr associated with size \p sz to the registry,
  /// \param ptr pointer to register
  /// \param sz size of the object pointed to by \p ptr
  /// \param context string context to attach to the pointer
  /// \param backtrace if true, appends backtrace of the caller to \p context
  /// separated by `:::::`
  void insert(void* ptr, std::size_t sz, const std::string& context,
              bool backtrace);

  /// erases \p ptr associated with size \p sz from the registry
  /// \details introduced for symmetry with insert_bt() )
  /// \param ptr pointer to erase
  /// \param sz size of the object pointed to by \p ptr
  /// \param context erasure context; if logging, will append this to the log
  /// \param backtrace if true and logging, appends the backtrace of caller to
  /// the log
  void erase(void* ptr, std::size_t sz, const std::string& context,
             bool backtrace);
};

}  // namespace TiledArray

#endif  // TILEDARRAY_UTIL_PTR_REGISTRY_H
