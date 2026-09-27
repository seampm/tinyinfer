// Phase 2: safetensors loader implementation.
//
// Parses the 8-byte little-endian header length + JSON header by hand and
// memory-maps the file so tensor data is paged in on demand — no bulk copy.

#include "tinyinfer/loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace tinyinfer {

size_t dtype_size(DType d) {
    switch (d) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::BF16: return 2;
        case DType::I32: return 4;
        case DType::I64: return 8;
        case DType::U8: return 1;
        case DType::BOOL: return 1;
        case DType::UNKNOWN: return 0;
    }
    return 0;
}

const char* dtype_name(DType d) {
    switch (d) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::BF16: return "BF16";
        case DType::I32: return "I32";
        case DType::I64: return "I64";
        case DType::U8: return "U8";
        case DType::BOOL: return "BOOL";
        case DType::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

DType dtype_from_string(const std::string& s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "I32") return DType::I32;
    if (s == "I64" || s == "I64 ") return DType::I64;
    if (s == "U8") return DType::U8;
    if (s == "BOOL") return DType::BOOL;
    return DType::UNKNOWN;
}

size_t TensorInfo::numel() const {
    if (shape.empty()) return 1; // scalar
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);
    return n;
}

SafetensorsLoader::~SafetensorsLoader() { close(); }

SafetensorsLoader::SafetensorsLoader(SafetensorsLoader&& other) noexcept
    : fd_(other.fd_), map_(other.map_), map_size_(other.map_size_),
      tensors_(std::move(other.tensors_)), metadata_(std::move(other.metadata_)) {
    other.fd_ = -1;
    other.map_ = nullptr;
    other.map_size_ = 0;
}

SafetensorsLoader& SafetensorsLoader::operator=(SafetensorsLoader&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        map_ = other.map_;
        map_size_ = other.map_size_;
        tensors_ = std::move(other.tensors_);
        metadata_ = std::move(other.metadata_);
        other.fd_ = -1;
        other.map_ = nullptr;
        other.map_size_ = 0;
    }
    return *this;
}

static uint64_t read_u64_le(const void* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    v = __builtin_bswap64(v);
#endif
    return v;
}

void SafetensorsLoader::open(const std::string& path) {
    close();

    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("safetensors: cannot open file: " + path);
    }

    struct stat st;
    if (::fstat(fd_, &st) != 0) {
        int e = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("safetensors: fstat failed: " + path + " errno=" + std::to_string(e));
    }
    if (static_cast<size_t>(st.st_size) < 8) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("safetensors: file too small to hold header: " + path);
    }
    map_size_ = static_cast<size_t>(st.st_size);

    map_ = ::mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_ == MAP_FAILED) {
        map_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("safetensors: mmap failed: " + path);
    }

    const char* base = static_cast<const char*>(map_);
    const uint64_t header_len = read_u64_le(base);
    if (8 + header_len > map_size_) {
        throw std::runtime_error("safetensors: header length exceeds file size: " + path);
    }

    nlohmann::json header;
    try {
        header = nlohmann::json::parse(base + 8, base + 8 + header_len);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("safetensors: bad JSON header: ") + e.what());
    }
    if (!header.is_object()) {
        throw std::runtime_error("safetensors: header is not a JSON object: " + path);
    }

    const char* data_base = base + 8 + header_len;
    const size_t data_size = map_size_ - 8 - static_cast<size_t>(header_len);

    for (auto it = header.begin(); it != header.end(); ++it) {
        const std::string& name = it.key();
        if (name == "__metadata__") {
            for (auto mit = it.value().begin(); mit != it.value().end(); ++mit) {
                if (mit.value().is_string()) metadata_[mit.key()] = mit.value().get<std::string>();
            }
            continue;
        }
        const auto& e = it.value();
        if (!e.is_object() || !e.contains("dtype") || !e.contains("shape") || !e.contains("data_offsets")) {
            throw std::runtime_error("safetensors: malformed entry for tensor '" + name + "'");
        }

        TensorInfo info;
        info.dtype = dtype_from_string(e["dtype"].get<std::string>());
        if (info.dtype == DType::UNKNOWN) {
            throw std::runtime_error("safetensors: unsupported dtype '" +
                                     e["dtype"].get<std::string>() + "' for tensor '" + name + "'");
        }
        for (const auto& d : e["shape"]) info.shape.push_back(d.get<int64_t>());

        const auto& off = e["data_offsets"];
        if (!off.is_array() || off.size() != 2) {
            throw std::runtime_error("safetensors: bad data_offsets for tensor '" + name + "'");
        }
        const uint64_t start = off[0].get<uint64_t>();
        const uint64_t end = off[1].get<uint64_t>();
        if (start > end || end > data_size) {
            throw std::runtime_error("safetensors: data_offsets out of range for tensor '" + name + "'");
        }
        const size_t expected = info.numel() * dtype_size(info.dtype);
        if (end - start != expected) {
            throw std::runtime_error("safetensors: byte range mismatch for tensor '" + name +
                                     "': expected " + std::to_string(expected) +
                                     ", got " + std::to_string(end - start));
        }
        info.data = data_base + start;
        info.nbytes = static_cast<size_t>(end - start);
        tensors_.emplace(name, std::move(info));
    }
}

void SafetensorsLoader::close() {
    tensors_.clear();
    metadata_.clear();
    if (map_ != nullptr) {
        ::munmap(map_, map_size_);
        map_ = nullptr;
        map_size_ = 0;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool SafetensorsLoader::has(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const TensorInfo& SafetensorsLoader::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::out_of_range("safetensors: no tensor '" + name + "'");
    return it->second;
}

std::vector<std::string> SafetensorsLoader::names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& kv : tensors_) out.push_back(kv.first);
    return out;
}

} // namespace tinyinfer
