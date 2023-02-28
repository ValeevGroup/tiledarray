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
 *  util/ptr_registry.cpp
 *  December 31, 2022
 *
 */

#include <TiledArray/fwd.h>

#include "TiledArray/util/ptr_registry.h"

#include "TiledArray/host/env.h"
#include "TiledArray/util/backtrace.h"

namespace TiledArray {

namespace detail {
void remove_linebreaks(std::string& str) {
  auto it = str.begin();
  while (it != str.end()) {
    if (*it == '\n')
      it = str.erase(it);
    else
      ++it;
  }
}
}  // namespace detail

PtrRegistry::PtrRegistry() = default;

PtrRegistry::PtrRegistry(std::ostream& os) : log_(&os) {}

PtrRegistry::~PtrRegistry() = default;

PtrRegistry& PtrRegistry::log(std::ostream* os_ptr) {
  log_ = os_ptr;
  return *this;
}

std::ostream* PtrRegistry::log() const { return log_; }

PtrRegistry& PtrRegistry::log_only(bool tf) {
  log_only_ = tf;
  return *this;
}

bool PtrRegistry::log_only() const { return log_only_; }

PtrRegistry& PtrRegistry::thread_local_logging(bool tf) {
  thread_local_logging_ = tf;
  return *this;
}

bool PtrRegistry::thread_local_logging() const { return thread_local_logging_; }

PtrRegistry& PtrRegistry::thread_local_log_filename_prefix(
    const std::string& pfx) {
  thread_local_log_filename_prefix_ = pfx;
  return *this;
}

PtrRegistry& PtrRegistry::append_backtrace(bool bt) {
  append_backtrace_ = bt;
  return *this;
}

bool PtrRegistry::append_backtrace() const { return append_backtrace_; }

const PtrRegistry::sized_ptr_container_type& PtrRegistry::sized_ptrs() const {
  return ptrs_;
}
const PtrRegistry::ptr_container_type& PtrRegistry::unsized_ptrs() const {
  if (unsized_ptrs_ == nullptr) {
    unsized_ptrs_ = &(const_cast<sized_ptr_container_type&>(ptrs_)[0]);
  }
  return *unsized_ptrs_;
}

std::size_t PtrRegistry::size() const {
  return std::accumulate(ptrs_.begin(), ptrs_.end(), 0,
                         [](const auto& total_size, const auto& sz_ptrs) {
                           auto&& [sz, ptrs] = sz_ptrs;
                           return total_size + ptrs.size();
                         });
}

void PtrRegistry::insert(void* ptr, std::size_t sz, const std::string& context,
                         bool backtrace) {
  auto* log = thread_local_logging_ ? thread_local_log() : log_;

  // early exit
  if (log_only_ && !log) return;

  std::string creation_context = context;
  if (backtrace) {
    detail::Backtrace bt;
    auto bt_str = bt.str(0);
    detail::remove_linebreaks(bt_str);
    creation_context += ":::::" + bt_str;
  }
  if (log) {
    *log << "PtrRegistry::insert():::::" << ptr << ":::::" << creation_context
         << std::endl
#ifdef TA_TENSOR_MEM_PROFILE
         << "  TA::Tensor allocator status {"
         << "hw="
         << hostEnv::instance()->host_allocator_getActualHighWatermark() << ","
         << "cur=" << hostEnv::instance()->host_allocator().getCurrentSize()
         << ","
         << "act=" << hostEnv::instance()->host_allocator().getActualSize()
         << "}"
         << " bytes" << std::endl
#endif  // TA_TENSOR_MEM_PROFILE
        ;
  }

  // track unless log_only_=true
  if (!log_only_) {
    std::scoped_lock lock(this->mtx_);
    auto& sz_ptrs = ptrs_[sz];
    TA_ASSERT(sz_ptrs.find(ptr) == sz_ptrs.end());
    sz_ptrs.emplace(ptr, std::move(creation_context));
  }
}

void PtrRegistry::erase(void* ptr, std::size_t sz, const std::string& context,
                        bool backtrace) {
  auto* log = thread_local_logging_ ? thread_local_log() : log_;

  // early exit
  if (log_only_ && !log) return;

  if (log) {
    std::string erasure_context = context;
    if (backtrace) {
      detail::Backtrace bt;
      auto bt_str = bt.str(0);
      detail::remove_linebreaks(bt_str);
      erasure_context += ":::::" + bt_str;
    }
    *log << "PtrRegistry::erase():::::" << ptr << ":::::" << erasure_context
         << std::endl
#ifdef TA_TENSOR_MEM_PROFILE
         << "  TA::Tensor allocator status {"
         << "hw="
         << hostEnv::instance()->host_allocator_getActualHighWatermark() << ","
         << "cur=" << hostEnv::instance()->host_allocator().getCurrentSize()
         << ","
         << "act=" << hostEnv::instance()->host_allocator().getActualSize()
         << "}"
         << " bytes" << std::endl
#endif  // TA_TENSOR_MEM_PROFILE
        ;
  }

  // track unless log_only=true
  if (!log_only_) {
    std::scoped_lock lock(this->mtx_);
    auto& sz_ptrs = ptrs_[sz];
    auto it = sz_ptrs.find(ptr);
    TA_ASSERT(it != sz_ptrs.end());
    sz_ptrs.erase(it);
  }
}

PtrRegistry& PtrRegistry::insert(void* ptr, std::size_t sz,
                                 const std::string& context) {
  this->insert(ptr, sz, context, /* backtrace = */ append_backtrace_);
  return *this;
}

PtrRegistry& PtrRegistry::insert(void* ptr, const std::string& context) {
  this->insert(ptr, /* sz = */ 0, context, /* backtrace = */ append_backtrace_);
  return *this;
}

PtrRegistry& PtrRegistry::insert_bt(void* ptr, std::size_t sz,
                                    const std::string& context) {
  this->insert(ptr, sz, context, /* backtrace = */ true);
  return *this;
}

PtrRegistry& PtrRegistry::insert_bt(void* ptr, const std::string& context) {
  this->insert(ptr, /* sz = */ 0, context, /* backtrace = */ true);
  return *this;
}

PtrRegistry& PtrRegistry::erase(void* ptr, std::size_t sz,
                                const std::string& context) {
  this->erase(ptr, sz, context, /* backtrace = */ append_backtrace_);
  return *this;
}

PtrRegistry& PtrRegistry::erase(void* ptr, const std::string& context) {
  this->erase(ptr, /* sz = */ 0, context, /* backtrace = */ append_backtrace_);
  return *this;
}

PtrRegistry& PtrRegistry::erase_bt(void* ptr, std::size_t sz,
                                   const std::string& context) {
  this->erase(ptr, sz, context, /* backtrace = */ true);
  return *this;
}

PtrRegistry& PtrRegistry::erase_bt(void* ptr, const std::string& context) {
  this->erase(ptr, /* sz = */ 0, context, /* backtrace = */ true);
  return *this;
}

std::ostream* PtrRegistry::thread_local_log() {
  static thread_local std::shared_ptr<std::ostream> thread_local_log_ =
      std::make_shared<std::ofstream>(
          thread_local_log_filename_prefix_ + ".thread_id=" +
          std::to_string(
              std::hash<std::thread::id>{}(std::this_thread::get_id())) +
          ".trace");
  return thread_local_log_.get();
}

}  // namespace TiledArray
