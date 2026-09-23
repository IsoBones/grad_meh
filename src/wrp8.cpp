#include "wrp8.h"

#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Cursor {
    const uint8_t* buf;
    size_t         len;
    size_t         pos = 0;

    uint32_t u32() {
        if (pos + 4 > len) throw std::runtime_error("WVR8: unexpected end of data");
        uint32_t v;
        memcpy(&v, buf + pos, 4);
        pos += 4;
        return v;
    }

    float f32() {
        if (pos + 4 > len) throw std::runtime_error("WVR8: unexpected end of data");
        float v;
        memcpy(&v, buf + pos, 4);
        pos += 4;
        return v;
    }

    std::string str(size_t slen) {
        if (pos + slen > len) throw std::runtime_error("WVR8: unexpected end of data");
        std::string s(reinterpret_cast<const char*>(buf + pos), slen);
        pos += slen;
        return s;
    }

    size_t remaining() const { return pos < len ? len - pos : 0; }
};

} // namespace

// Detects the uncompressed 8WVR terrain format by its magic bytes. The format
// also has an LZSS-compressed variant (no "8WVR" magic); that is not handled
// here and continues to fall through to the OPRW code path unchanged.
bool isWvr8(const rust::Vec<uint8_t>& data) {
    return data.size() >= 4 && memcmp(data.data(), "8WVR", 4) == 0;
}

void populateFromWvr8(const rust::Vec<uint8_t>& data, arma_file_formats::cxx::OprwCxx& wrp) {
    Cursor c { data.data(), data.size() };

    // Skip magic "8WVR"
    c.pos = 4;

    // Header
    uint32_t tex_x   = c.u32();
    uint32_t tex_y   = c.u32();
    uint32_t ter_x   = c.u32();
    uint32_t ter_y   = c.u32();
    float    cellsz  = c.f32();

    wrp.layer_size_x   = tex_x;
    wrp.layer_size_y   = tex_y;
    wrp.map_size_x     = ter_x;
    wrp.map_size_y     = ter_y;
    wrp.layer_cell_size = cellsz;

    // Elevations: terrain_grid_size.x * terrain_grid_size.y floats
    size_t elev_count = static_cast<size_t>(ter_x) * ter_y;
    for (size_t i = 0; i < elev_count; i++)
        wrp.elevation.push_back(c.f32());

    // Material indices: texture_grid_size.x * texture_grid_size.y u16s (not used by C++ code, skip)
    size_t mat_idx_count = static_cast<size_t>(tex_x) * tex_y;
    c.pos += mat_idx_count * sizeof(uint16_t);

    // RvmatLayer — variable-length string encoding:
    //   for each of material_count materials: read length-prefixed strings until length == 0
    uint32_t material_count = c.u32();
    for (uint32_t m = 0; m < material_count; m++) {
        while (true) {
            uint32_t slen = c.u32();
            if (slen == 0) break;
            auto name = c.str(slen);
            arma_file_formats::cxx::TextureCxx tex{};
            // lossy UTF-8 conversion mirrors the reference parser's String::from_utf8_lossy
            tex.texture_filename = rust::String::lossy(name.data(), name.size());
            wrp.texures.push_back(std::move(tex));
        }
    }

    // Objects — read every entry until EOF. Each entry is:
    //   48-byte transform matrix, u32 object_id, u32 name_len, name_len name bytes
    struct RawObject {
        arma_file_formats::cxx::TransformMatrixCxx matrix;
        uint32_t                                   object_id;
        std::string                                p3d;
    };
    std::vector<RawObject> raws;

    while (c.remaining() >= 56) { // 48 (matrix) + 4 (id) + 4 (name_len) minimum
        arma_file_formats::cxx::TransformMatrixCxx tm{};
        tm._0.x = c.f32(); tm._0.y = c.f32(); tm._0.z = c.f32();
        tm._1.x = c.f32(); tm._1.y = c.f32(); tm._1.z = c.f32();
        tm._2.x = c.f32(); tm._2.y = c.f32(); tm._2.z = c.f32();
        tm._3.x = c.f32(); tm._3.y = c.f32(); tm._3.z = c.f32();

        uint32_t object_id = c.u32();
        uint32_t name_len  = c.u32();

        if (c.remaining() < name_len) break;
        auto p3d = c.str(name_len);

        raws.push_back({ std::move(tm), object_id, std::move(p3d) });
    }

    // The format always appends a dummy sentinel object; discard it BEFORE building
    // the model list so its (empty) path never becomes a model entry.
    if (!raws.empty())
        raws.pop_back();

    std::map<std::string, uint32_t> modelIndex;
    for (auto& raw : raws) {
        arma_file_formats::cxx::ObjectCxx obj{};
        obj.object_id       = raw.object_id;
        obj.transform_matrx = std::move(raw.matrix);
        obj.shape_params    = 0;

        auto it = modelIndex.find(raw.p3d);
        if (it == modelIndex.end()) {
            uint32_t idx = static_cast<uint32_t>(modelIndex.size());
            modelIndex[raw.p3d] = idx;
            wrp.models.push_back(rust::String::lossy(raw.p3d.data(), raw.p3d.size()));
            obj.model_index = idx;
        } else {
            obj.model_index = it->second;
        }

        wrp.objects.push_back(std::move(obj));
    }
}
