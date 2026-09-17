#pragma once

#include <utility>
#include <type_traits>
#include <atomic>
#include <cstddef>


namespace detail
{

struct control_block
{
  std::atomic<std::size_t> ref_count = 1;

  virtual void destory_object() noexcept = 0;
  virtual ~control_block() = default;
};

template <typename U>
struct pointer_control_block final: control_block
{
  explicit pointer_control_block(U* ptr)
    : ptr(ptr)
  {
  }

  void destory_object() noexcept override
  {
    delete ptr;
  }

  U* ptr;
};

}

template <typename T>
class my_shared_ptr
{
public:
  my_shared_ptr() noexcept = default;
  
  my_shared_ptr(std::nullptr_t) noexcept
  {
  }

  template <typename U>
  requires std::is_convertible_v<U*, T*>
  explicit my_shared_ptr(U* ptr)
    : ptr_(ptr), block_(nullptr)
  {
    static_assert(sizeof(U) > 0);

    try {
      block_ = new detail::pointer_control_block<U>(ptr);
    } catch (...) {
      delete ptr;
      throw;
    }
  }

  my_shared_ptr(const my_shared_ptr& other)
    : ptr_(other.ptr_), block_(other.block_)
  {
    if (block_) {
      block_->ref_count.fetch_add(1, std::memory_order_relaxed);
    }
  }
  my_shared_ptr& operator=(const my_shared_ptr& other)
  {
    my_shared_ptr tmp(other);
    this->swap(tmp);

    return *this;
  }
  
  my_shared_ptr(my_shared_ptr&& other) noexcept
    : ptr_(std::exchange(other.ptr_, nullptr)), block_(std::exchange(other.block_, nullptr))
  {
  }
  my_shared_ptr& operator=(my_shared_ptr&& other) noexcept
  {
    my_shared_ptr tmp(std::move(other));
    this->swap(tmp);
    
    return *this;
  }

  ~my_shared_ptr() noexcept
  {
    if (block_ && block_->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      block_->destory_object();
      delete block_;
    }
  }

  void swap(my_shared_ptr& other) noexcept
  {
    std::swap(ptr_, other.ptr_);
    std::swap(block_, other.block_);
  }

private:
  T* ptr_ = nullptr;
  detail::control_block* block_ = nullptr;
};