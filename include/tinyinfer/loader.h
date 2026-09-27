#pragma once

// Phase 2: safetensors loader.
//
// Format (parsed by hand):
//   [0:8)   little-endian uint64 = JSON header length N
//   [8:8+N) JSON header: { name: {dtype, shape, data_offsets}, ..., "__metadata__"? }
//   [8+N:)  raw data buffer; tensor bytes live at data_offsets relative to its start
//
// The whole file is memory-mapped (mmap, PROT_READ, MAP_PRIVATE), so large
// models load near-instantly and the OS pages in weights on demand.
// TensorInfo::data points directly into the mapping — no copies.

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinyinfer {

enum class DType {
    F32,
    F16,
    BF16,
    I32,
    I64,
    U8,
    BOOL,
    UNKNOWN,
};

size_t dtype_size(DType d);
const char* dtype_name(DType d);
DType dtype_from_string(const std::string& s);

struct TensorInfo {
    DType dtype = DType::UNKNOWN;
    std::vector<int64_t> shape;
    const void* data = nullptr; // into the mmap'd region; valid while loader is open
    size_t nbytes = 0;

    size_t numel() const;
    // Typed access, e.g. info.as<float>() for F32 tensors.
    template <typename T>
    const T* as() const { return static_cast<const T*>(data); }
};

class SafetensorsLoader {
public:
    SafetensorsLoader() = default;
    ~SafetensorsLoader();

    SafetensorsLoader(const SafetensorsLoader&) = delete;
    SafetensorsLoader& operator=(const SafetensorsLoader&) = delete;
    SafetensorsLoader(SafetensorsLoader&& other) noexcept;
    SafetensorsLoader& operator=(SafetensorsLoader&& other) noexcept;

    // Memory-map `path` and parse its header. Throws std::runtime_error on failure.
    void open(const std::string& path);
    void close();
    bool is_open() const { return map_ != nullptr; }

    bool has(const std::string& name) const;
    // Throws std::out_of_range if missing.
    const TensorInfo& get(const std::string& name) const;
    std::vector<std::string> names() const;
    size_t num_tensors() const { return tensors_.size(); }
    // File-level metadata (the "__metadata__" object), may be empty.
    const std::unordered_map<std::string, std::string>& metadata() const { return metadata_; }

private:
    int fd_ = -1;
    void* map_ = nullptr;
    size_t map_size_ = 0;
    std::unordered_map<std::string, TensorInfo> tensors_;
    std::unordered_map<std::string, std::string> metadata_;
};

} // namespace tinyinfer
