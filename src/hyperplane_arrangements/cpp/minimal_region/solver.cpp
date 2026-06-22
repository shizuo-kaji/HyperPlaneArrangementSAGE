#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

inline void hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
}

inline int64_t checked_int128_to_int64(__int128 value, const char* context) {
    if (value < static_cast<__int128>(std::numeric_limits<int64_t>::min()) ||
        value > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
        throw std::overflow_error(std::string("int64 overflow in ") + context);
    }
    return static_cast<int64_t>(value);
}

struct Rational {
    int64_t num = 0;
    int64_t den = 1;

    Rational() = default;
    explicit Rational(int64_t n) : num(n), den(1) {}
    Rational(int64_t n, int64_t d) { set(n, d); }

    void set(int64_t n, int64_t d) {
        if (d == 0) {
            throw std::runtime_error("Rational denominator cannot be zero");
        }
        if (d < 0) {
            n = -n;
            d = -d;
        }
        if (n == 0) {
            num = 0;
            den = 1;
            return;
        }
        int64_t g = std::gcd(std::llabs(n), d);
        num = n / g;
        den = d / g;
    }

    std::string to_string() const {
        if (den == 1) {
            return std::to_string(num);
        }
        return std::to_string(num) + "/" + std::to_string(den);
    }
};

inline bool operator==(const Rational& lhs, const Rational& rhs) {
    return lhs.num == rhs.num && lhs.den == rhs.den;
}

inline bool operator!=(const Rational& lhs, const Rational& rhs) {
    return !(lhs == rhs);
}

inline bool operator<(const Rational& lhs, const Rational& rhs) {
    return static_cast<__int128>(lhs.num) * rhs.den < static_cast<__int128>(rhs.num) * lhs.den;
}

struct RationalHash {
    std::size_t operator()(const Rational& r) const {
        std::size_t seed = 0;
        hash_combine(seed, std::hash<int64_t>{}(r.num));
        hash_combine(seed, std::hash<int64_t>{}(r.den));
        return seed;
    }
};

inline Rational mul_int(const Rational& r, int64_t k) {
    if (k == 0 || r.num == 0) {
        return Rational(0);
    }
    int64_t g = std::gcd(std::llabs(k), r.den);
    int64_t k2 = k / g;
    int64_t den = r.den / g;
    __int128 num = static_cast<__int128>(r.num) * k2;
    return Rational(checked_int128_to_int64(num, "mul_int numerator"), den);
}

inline Rational div_int(const Rational& r, int64_t k) {
    if (k == 0) {
        throw std::runtime_error("division by zero");
    }
    if (r.num == 0) {
        return Rational(0);
    }
    int64_t kk = k;
    int64_t num = r.num;
    if (kk < 0) {
        kk = -kk;
        num = -num;
    }
    int64_t g = std::gcd(std::llabs(num), kk);
    num /= g;
    kk /= g;
    __int128 den = static_cast<__int128>(r.den) * kk;
    return Rational(num, checked_int128_to_int64(den, "div_int denominator"));
}

inline Rational operator+(const Rational& lhs, const Rational& rhs) {
    int64_t g = std::gcd(lhs.den, rhs.den);
    int64_t lhs_mul = rhs.den / g;
    int64_t rhs_mul = lhs.den / g;
    __int128 n = static_cast<__int128>(lhs.num) * lhs_mul + static_cast<__int128>(rhs.num) * rhs_mul;
    __int128 d = static_cast<__int128>(lhs.den) * lhs_mul;
    return Rational(
        checked_int128_to_int64(n, "operator+ numerator"),
        checked_int128_to_int64(d, "operator+ denominator")
    );
}

inline Rational operator-(const Rational& lhs, const Rational& rhs) {
    int64_t g = std::gcd(lhs.den, rhs.den);
    int64_t lhs_mul = rhs.den / g;
    int64_t rhs_mul = lhs.den / g;
    __int128 n = static_cast<__int128>(lhs.num) * lhs_mul - static_cast<__int128>(rhs.num) * rhs_mul;
    __int128 d = static_cast<__int128>(lhs.den) * lhs_mul;
    return Rational(
        checked_int128_to_int64(n, "operator- numerator"),
        checked_int128_to_int64(d, "operator- denominator")
    );
}

struct Point {
    Rational x;
    Rational y;
};

inline bool operator==(const Point& lhs, const Point& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

struct PointLess {
    bool operator()(const Point& lhs, const Point& rhs) const {
        if (lhs.x != rhs.x) {
            return lhs.x < rhs.x;
        }
        return lhs.y < rhs.y;
    }
};

struct PointHash {
    std::size_t operator()(const Point& p) const {
        std::size_t seed = 0;
        hash_combine(seed, RationalHash{}(p.x));
        hash_combine(seed, RationalHash{}(p.y));
        return seed;
    }
};

using Normal = std::pair<int64_t, int64_t>;

struct NormalHash {
    std::size_t operator()(const Normal& n) const {
        std::size_t seed = 0;
        hash_combine(seed, std::hash<int64_t>{}(n.first));
        hash_combine(seed, std::hash<int64_t>{}(n.second));
        return seed;
    }
};

inline bool contains_offset(const std::vector<Rational>& offsets, const Rational& c) {
    return std::binary_search(offsets.begin(), offsets.end(), c);
}

inline void insert_offset(std::vector<Rational>& offsets, const Rational& c) {
    auto it = std::lower_bound(offsets.begin(), offsets.end(), c);
    if (it == offsets.end() || *it != c) {
        offsets.insert(it, c);
    }
}

inline bool contains_point(const std::vector<Point>& points, const Point& p) {
    return std::binary_search(points.begin(), points.end(), p, PointLess{});
}

inline std::vector<Point> add_point(const std::vector<Point>& base, const Point& p) {
    std::vector<Point> out = base;
    auto it = std::lower_bound(out.begin(), out.end(), p, PointLess{});
    if (it == out.end() || !(*it == p)) {
        out.insert(it, p);
    }
    return out;
}

inline void sort_and_unique_points(std::vector<Point>& points) {
    std::sort(points.begin(), points.end(), PointLess{});
    points.erase(std::unique(points.begin(), points.end()), points.end());
}

inline std::vector<Point> union_points(const std::vector<Point>& base, const std::vector<Point>& extras) {
    if (extras.empty()) {
        return base;
    }
    std::vector<Point> rhs = extras;
    sort_and_unique_points(rhs);
    std::vector<Point> out;
    out.reserve(base.size() + rhs.size());
    std::set_union(
        base.begin(),
        base.end(),
        rhs.begin(),
        rhs.end(),
        std::back_inserter(out),
        PointLess{}
    );
    return out;
}

class JsonValue {
public:
    enum class Type {
        kNull,
        kBool,
        kInteger,
        kString,
        kArray,
        kObject
    };

    Type type = Type::kNull;
    bool bool_value = false;
    int64_t int_value = 0;
    std::string string_value;
    std::vector<JsonValue> array_value;
    std::vector<std::pair<std::string, JsonValue>> object_value;

    const JsonValue& require_key(const std::string& key) const {
        if (type != Type::kObject) {
            throw std::runtime_error("JSON value is not an object");
        }
        for (const auto& kv : object_value) {
            if (kv.first == key) {
                return kv.second;
            }
        }
        throw std::runtime_error("Missing required key: " + key);
    }

    const JsonValue* find_key(const std::string& key) const {
        if (type != Type::kObject) {
            throw std::runtime_error("JSON value is not an object");
        }
        for (const auto& kv : object_value) {
            if (kv.first == key) {
                return &kv.second;
            }
        }
        return nullptr;
    }

    const std::vector<JsonValue>& as_array(const std::string& label) const {
        if (type != Type::kArray) {
            throw std::runtime_error(label + " must be an array");
        }
        return array_value;
    }

    int64_t as_int(const std::string& label) const {
        if (type != Type::kInteger) {
            throw std::runtime_error(label + " must be an integer");
        }
        return int_value;
    }

    bool as_bool(const std::string& label) const {
        if (type != Type::kBool) {
            throw std::runtime_error(label + " must be a boolean");
        }
        return bool_value;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text) {}

    JsonValue parse() {
        skip_ws();
        JsonValue root = parse_value();
        skip_ws();
        if (pos_ != text_.size()) {
            parse_error("Unexpected trailing characters");
        }
        return root;
    }

private:
    const std::string& text_;
    std::size_t pos_ = 0;

    [[noreturn]] void parse_error(const std::string& msg) const {
        throw std::runtime_error(msg + " at offset " + std::to_string(pos_));
    }

    void skip_ws() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
    }

    bool match_literal(const char* lit) {
        std::size_t i = 0;
        while (lit[i] != '\0') {
            if (pos_ + i >= text_.size() || text_[pos_ + i] != lit[i]) {
                return false;
            }
            ++i;
        }
        pos_ += i;
        return true;
    }

    JsonValue parse_value() {
        if (pos_ >= text_.size()) {
            parse_error("Unexpected EOF");
        }
        const char c = text_[pos_];
        if (c == '{') {
            return parse_object();
        }
        if (c == '[') {
            return parse_array();
        }
        if (c == '"') {
            JsonValue v;
            v.type = JsonValue::Type::kString;
            v.string_value = parse_string();
            return v;
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            JsonValue v;
            v.type = JsonValue::Type::kInteger;
            v.int_value = parse_integer();
            return v;
        }
        if (match_literal("true")) {
            JsonValue v;
            v.type = JsonValue::Type::kBool;
            v.bool_value = true;
            return v;
        }
        if (match_literal("false")) {
            JsonValue v;
            v.type = JsonValue::Type::kBool;
            v.bool_value = false;
            return v;
        }
        if (match_literal("null")) {
            JsonValue v;
            v.type = JsonValue::Type::kNull;
            return v;
        }
        parse_error("Invalid JSON token");
    }

    JsonValue parse_object() {
        JsonValue v;
        v.type = JsonValue::Type::kObject;
        ++pos_;  // consume '{'
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == '}') {
            ++pos_;
            return v;
        }
        while (true) {
            skip_ws();
            if (pos_ >= text_.size() || text_[pos_] != '"') {
                parse_error("Object key must be a string");
            }
            std::string key = parse_string();
            skip_ws();
            if (pos_ >= text_.size() || text_[pos_] != ':') {
                parse_error("Expected ':' after object key");
            }
            ++pos_;
            skip_ws();
            JsonValue value = parse_value();
            bool updated = false;
            for (auto& kv : v.object_value) {
                if (kv.first == key) {
                    kv.second = std::move(value);
                    updated = true;
                    break;
                }
            }
            if (!updated) {
                v.object_value.push_back(std::make_pair(std::move(key), std::move(value)));
            }
            skip_ws();
            if (pos_ >= text_.size()) {
                parse_error("Unexpected EOF in object");
            }
            if (text_[pos_] == '}') {
                ++pos_;
                break;
            }
            if (text_[pos_] != ',') {
                parse_error("Expected ',' in object");
            }
            ++pos_;
        }
        return v;
    }

    JsonValue parse_array() {
        JsonValue v;
        v.type = JsonValue::Type::kArray;
        ++pos_;  // consume '['
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == ']') {
            ++pos_;
            return v;
        }
        while (true) {
            skip_ws();
            v.array_value.push_back(parse_value());
            skip_ws();
            if (pos_ >= text_.size()) {
                parse_error("Unexpected EOF in array");
            }
            if (text_[pos_] == ']') {
                ++pos_;
                break;
            }
            if (text_[pos_] != ',') {
                parse_error("Expected ',' in array");
            }
            ++pos_;
        }
        return v;
    }

    std::string parse_string() {
        if (text_[pos_] != '"') {
            parse_error("String must start with '\"'");
        }
        ++pos_;  // consume '"'
        std::string out;
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') {
                return out;
            }
            if (c == '\\') {
                if (pos_ >= text_.size()) {
                    parse_error("Unexpected EOF in string escape");
                }
                const char esc = text_[pos_++];
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default:
                        parse_error("Unsupported string escape");
                }
            } else {
                out.push_back(c);
            }
        }
        parse_error("Unexpected EOF in string");
    }

    int64_t parse_integer() {
        const std::size_t start = pos_;
        if (text_[pos_] == '-') {
            ++pos_;
        }
        if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            parse_error("Invalid number");
        }
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == '.' || text_[pos_] == 'e' || text_[pos_] == 'E')) {
            parse_error("Only integer numbers are supported");
        }
        std::string token = text_.substr(start, pos_ - start);
        try {
            std::size_t consumed = 0;
            long long value = std::stoll(token, &consumed, 10);
            if (consumed != token.size()) {
                parse_error("Invalid integer");
            }
            return static_cast<int64_t>(value);
        } catch (const std::exception&) {
            parse_error("Integer out of range");
        }
    }
};

// 最終的な解（互換性のための1つ目の配置 + 全ての最小配置）
struct SolutionConfig {
    std::vector<std::vector<Rational>> lines;
    std::vector<Point> seeds;
};

struct Solution {
    int regions = 0;
    std::vector<Normal> normals;
    std::vector<std::vector<Rational>> lines_by_dir;
    std::vector<Point> seed_points;
    std::vector<SolutionConfig> configs; // 追加: 全ての最小配置
};

// 探索中に保持するベストな記録
struct BestRecord {
    int regions = 0;
    std::vector<SolutionConfig> configs;
};

class IncrementalJsonWriter {
public:
    IncrementalJsonWriter(const std::string& path, const std::vector<Normal>& normals, bool return_all_minimal) : out_(path, std::ios::out | std::ios::trunc), return_all_minimal_(return_all_minimal) {
        if (!out_) {
            throw std::runtime_error("Failed to open output file");
        }
        out_ << "{\n";
        out_ << "  \"normals\": [";
        for (std::size_t i = 0; i < normals.size(); ++i) {
            if (i > 0) {
                out_ << ", ";
            }
            out_ << "[" << normals[i].first << ", " << normals[i].second << "]";
        }
        out_ << "],\n";
        out_ << "  \"results\": [\n";
        footer_pos_ = out_.tellp();
        out_ << "  ]\n}\n";
        out_.flush();
    }

    void append_result(const std::vector<int>& counts, const BestRecord& record) {
        out_.seekp(footer_pos_);
        if (!out_) {
            throw std::runtime_error("Failed to seek while streaming JSON output");
        }
        if (has_results_) {
            out_ << ",\n";
        }
        write_result_object(out_, counts, record);
        out_ << "\n";
        footer_pos_ = out_.tellp();
        out_ << "  ]\n}\n";
        out_.flush();
        has_results_ = true;
    }

private:
    static void write_json_string(std::ostream& os, const std::string& s) {
        os << '"';
        for (char c : s) {
            switch (c) {
                case '\\': os << "\\\\"; break;
                case '"': os << "\\\""; break;
                case '\n': os << "\\n"; break;
                case '\r': os << "\\r"; break;
                case '\t': os << "\\t"; break;
                default: os << c; break;
            }
        }
        os << '"';
    }

    static bool is_origin(const Point& p) {
        return p.x.num == 0 && p.y.num == 0;
    }

    void write_result_object(std::ostream& os, const std::vector<int>& counts, const BestRecord& record) {
        os << "    {\n";
        os << "      \"counts\": [";
        for (std::size_t j = 0; j < counts.size(); ++j) {
            if (j > 0) {
                os << ", ";
            }
            os << counts[j];
        }
        os << "],\n";
        os << "      \"regions\": " << record.regions << ",\n";

        const auto& first_config = record.configs.front();
        os << "      \"seed_points\": [";
        bool first_seed = true;
        for (const Point& seed : first_config.seeds) {
            if (seed.x.num == 0 && seed.y.num == 0) { continue; }
            if (!first_seed) { os << ", "; }
            os << "[";
            write_json_string(os, seed.x.to_string());
            os << ", ";
            write_json_string(os, seed.y.to_string());
            os << "]";
            first_seed = false;
        }
        os << "],\n";

        os << "      \"lines_by_dir\": [";
        for (std::size_t d = 0; d < first_config.lines.size(); ++d) {
            if (d > 0) { os << ", "; }
            os << "[";
            for (std::size_t k = 0; k < first_config.lines[d].size(); ++k) {
                if (k > 0) { os << ", "; }
                write_json_string(os, first_config.lines[d][k].to_string());
            }
            os << "]";
        }
        os << "]";

        if (return_all_minimal_) {
            os << ",\n      \"configs\": [\n";
            for (std::size_t c = 0; c < record.configs.size(); ++c) {
                const auto& config = record.configs[c];
                os << "        {\n";
                os << "          \"seed_points\": [";
                first_seed = true;
                for (const Point& seed : config.seeds) {
                    if (seed.x.num == 0 && seed.y.num == 0) { continue; }
                    if (!first_seed) { os << ", "; }
                    os << "[";
                    write_json_string(os, seed.x.to_string());
                    os << ", ";
                    write_json_string(os, seed.y.to_string());
                    os << "]";
                    first_seed = false;
                }
                os << "],\n";
                os << "          \"lines_by_dir\": [";
                for (std::size_t d = 0; d < config.lines.size(); ++d) {
                    if (d > 0) { os << ", "; }
                    os << "[";
                    for (std::size_t k = 0; k < config.lines[d].size(); ++k) {
                        if (k > 0) { os << ", "; }
                        write_json_string(os, config.lines[d][k].to_string());
                    }
                    os << "]";
                }
                os << "]\n";
                os << "        }";
                if (c + 1 < record.configs.size()) {
                    os << ",";
                }
                os << "\n";
            }
            os << "      ]\n";
        } else {
            os << "\n";
        }
        os << "    }";
    }

    std::ofstream out_;
    std::streampos footer_pos_{};
    bool has_results_ = false;
    bool return_all_minimal_ = false;
};

struct VectorIntHash {
    std::size_t operator()(const std::vector<int>& v) const {
        std::size_t seed = v.size();
        for (int x : v) {
            hash_combine(seed, std::hash<int>{}(x));
        }
        return seed;
    }
};

struct DotKey {
    int dir = 0;
    Point p;
};

inline bool operator==(const DotKey& lhs, const DotKey& rhs) {
    return lhs.dir == rhs.dir && lhs.p == rhs.p;
}

struct DotKeyHash {
    std::size_t operator()(const DotKey& key) const {
        std::size_t seed = 0;
        hash_combine(seed, std::hash<int>{}(key.dir));
        hash_combine(seed, PointHash{}(key.p));
        return seed;
    }
};

struct InterKey {
    int i = 0;
    Rational c;
    int j = 0;
    Rational c2;
};

inline bool operator==(const InterKey& lhs, const InterKey& rhs) {
    return lhs.i == rhs.i && lhs.j == rhs.j && lhs.c == rhs.c && lhs.c2 == rhs.c2;
}

struct InterKeyHash {
    std::size_t operator()(const InterKey& key) const {
        std::size_t seed = 0;
        hash_combine(seed, std::hash<int>{}(key.i));
        hash_combine(seed, RationalHash{}(key.c));
        hash_combine(seed, std::hash<int>{}(key.j));
        hash_combine(seed, RationalHash{}(key.c2));
        return seed;
    }
};

struct MoveKey {
    int dir = 0;
    Rational c;
};

inline bool operator==(const MoveKey& lhs, const MoveKey& rhs) {
    return lhs.dir == rhs.dir && lhs.c == rhs.c;
}

struct MoveKeyHash {
    std::size_t operator()(const MoveKey& key) const {
        std::size_t seed = 0;
        hash_combine(seed, std::hash<int>{}(key.dir));
        hash_combine(seed, RationalHash{}(key.c));
        return seed;
    }
};

struct MoveData {
    int delta = 0;
    std::vector<Point> new_points;
};

struct StateKey {
    std::vector<std::vector<Rational>> lines;
    std::vector<Point> seeds;
};

inline bool operator==(const StateKey& lhs, const StateKey& rhs) {
    return lhs.lines == rhs.lines && lhs.seeds == rhs.seeds;
}

struct StateKeyHash {
    std::size_t operator()(const StateKey& key) const {
        std::size_t seed = key.lines.size();
        for (const auto& line : key.lines) {
            hash_combine(seed, line.size());
            for (const auto& c : line) {
                hash_combine(seed, RationalHash{}(c));
            }
        }
        hash_combine(seed, key.seeds.size());
        for (const auto& p : key.seeds) {
            hash_combine(seed, PointHash{}(p));
        }
        return seed;
    }
};

// ---- Compact state keys for deduplication ----------------------------------
// A search state is uniquely identified by (lines, seeds). We serialize it into
// a flat, length-prefixed vector<int64_t> ("key words"); distinct states always
// produce distinct key words. From the key words we derive either:
//   * a 128-bit fingerprint (default "hash" dedup) -- tiny and fast, with a
//     per-pair collision probability around 2^-128 (negligible in practice), or
//   * the exact key words themselves ("exact" dedup) -- bit-exact but heavier.
struct Fingerprint {
    uint64_t a = 0;
    uint64_t b = 0;
};

inline bool operator==(const Fingerprint& lhs, const Fingerprint& rhs) {
    return lhs.a == rhs.a && lhs.b == rhs.b;
}

struct FingerprintHash {
    std::size_t operator()(const Fingerprint& f) const {
        return static_cast<std::size_t>(f.a ^ (f.b * 0x9e3779b97f4a7c15ULL));
    }
};

struct KeyWordsHash {
    std::size_t operator()(const std::vector<int64_t>& v) const {
        uint64_t h = 1469598103934665603ULL ^ static_cast<uint64_t>(v.size());
        for (int64_t w : v) {
            uint64_t x = static_cast<uint64_t>(w);
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccdULL;
            x ^= x >> 33;
            h ^= x;
            h *= 0x100000001b3ULL;
        }
        h ^= h >> 29;
        return static_cast<std::size_t>(h);
    }
};

// Two reasonably independent 64-bit mixes over the same key words -> 128-bit id.
inline Fingerprint fingerprint_words(const std::vector<int64_t>& v) {
    uint64_t h1 = 0x9e3779b97f4a7c15ULL ^ static_cast<uint64_t>(v.size());
    uint64_t h2 = 0xc2b2ae3d27d4eb4fULL + static_cast<uint64_t>(v.size()) * 0x100000001b3ULL;
    for (int64_t w : v) {
        const uint64_t x = static_cast<uint64_t>(w);
        uint64_t y = x;
        y ^= y >> 33;
        y *= 0xff51afd7ed558ccdULL;
        y ^= y >> 33;
        h1 ^= y;
        h1 *= 0x100000001b3ULL;
        h1 ^= h1 >> 29;
        uint64_t z = x;
        z ^= z >> 29;
        z *= 0xc2b2ae3d27d4eb4fULL;
        z ^= z >> 32;
        h2 += z;
        h2 *= 0x9e3779b97f4a7c15ULL;
        h2 ^= h2 >> 31;
    }
    h1 ^= h1 >> 33;
    h1 *= 0xc4ceb9fe1a85ec53ULL;
    h1 ^= h1 >> 33;
    h2 ^= h2 >> 31;
    h2 *= 0xbf58476d1ce4e5b9ULL;
    h2 ^= h2 >> 33;
    return Fingerprint{h1, h2};
}

class FinalizedResultCollector {
public:
    using BestMap = std::unordered_map<std::vector<int>, BestRecord, VectorIntHash>;

    FinalizedResultCollector(
        std::vector<std::vector<int>> task_start_counts,
        BestMap initial_best,
        IncrementalJsonWriter* writer,
        bool return_all_minimal
    )
        : task_start_counts_(std::move(task_start_counts)),
          task_finished_(task_start_counts_.size(), false),
          merged_(std::move(initial_best)),
          writer_(writer),
          return_all_minimal_(return_all_minimal) {
        if (writer_ != nullptr) {
            for (const auto& entry : merged_) {
                ensure_tracked_locked(entry.first);
            }
            flush_ready_locked();
        }
    }

    void complete_task(std::size_t task_index, BestMap partial) {
        std::lock_guard<std::mutex> lock(mu_);
        if (task_index >= task_finished_.size()) {
            throw std::runtime_error("Task index out of range in FinalizedResultCollector");
        }
        if (task_finished_[task_index]) {
            throw std::runtime_error("Duplicate task completion in FinalizedResultCollector");
        }

        task_finished_[task_index] = true;
        if (writer_ != nullptr) {
            for (auto& [counts, tracker] : tracked_) {
                if (!tracker.finalized && tracker.remaining_tasks > 0 && can_task_reach_count(task_start_counts_[task_index], counts)) {
                    --tracker.remaining_tasks;
                }
            }
        }

        for (auto& [counts, record] : partial) {
            auto it = merged_.find(counts);
            if (it == merged_.end() || record.regions < it->second.regions) {
                merged_[counts] = std::move(record);
            } else if (return_all_minimal_ && record.regions == it->second.regions) {
                for (auto& config : record.configs) {
                    bool dup = false;
                    for (const auto& existing : it->second.configs) {
                        if (config.lines == existing.lines && config.seeds == existing.seeds) {
                            dup = true; break;
                        }
                    }
                    if (!dup) {
                        it->second.configs.push_back(std::move(config));
                    }
                }
            }
            if (writer_ != nullptr) {
                ensure_tracked_locked(counts);
            }
        }

        if (writer_ != nullptr) {
            flush_ready_locked();
        }
    }

    BestMap take_merged() {
        std::lock_guard<std::mutex> lock(mu_);
        return std::move(merged_);
    }

private:
    struct TrackState {
        std::size_t remaining_tasks = 0;
        bool finalized = false;
    };

    static bool can_task_reach_count(const std::vector<int>& start_counts, const std::vector<int>& target_counts) {
        if (start_counts.size() != target_counts.size()) {
            return false;
        }
        for (std::size_t i = 0; i < start_counts.size(); ++i) {
            if (start_counts[i] > target_counts[i]) {
                return false;
            }
        }
        return true;
    }

    void ensure_tracked_locked(const std::vector<int>& counts) {
        if (tracked_.find(counts) != tracked_.end()) {
            return;
        }
        TrackState state;
        for (std::size_t i = 0; i < task_start_counts_.size(); ++i) {
            if (!task_finished_[i] && can_task_reach_count(task_start_counts_[i], counts)) {
                ++state.remaining_tasks;
            }
        }
        tracked_.emplace(counts, state);
    }

    void flush_ready_locked() {
        std::vector<std::vector<int>> ready;
        ready.reserve(tracked_.size());
        for (const auto& [counts, tracker] : tracked_) {
            if (!tracker.finalized && tracker.remaining_tasks == 0) {
                ready.push_back(counts);
            }
        }
        std::sort(ready.begin(), ready.end());
        for (const auto& counts : ready) {
            auto it = tracked_.find(counts);
            if (it == tracked_.end() || it->second.finalized) {
                continue;
            }
            writer_->append_result(counts, merged_.at(counts));
            it->second.finalized = true;
        }
    }

    std::vector<std::vector<int>> task_start_counts_;
    std::vector<bool> task_finished_;
    BestMap merged_;
    IncrementalJsonWriter* writer_ = nullptr;
    std::unordered_map<std::vector<int>, TrackState, VectorIntHash> tracked_;
    std::mutex mu_;
    bool return_all_minimal_ = false;
};

// Thread-shared dedup set used to coordinate workers near the frontier.
// Only one of the two backing stores is populated, chosen by `hash_mode`.
class SharedVisited {
public:
    SharedVisited(bool hash_mode, std::size_t shard_count)
        : fp_shards_(hash_mode ? std::max<std::size_t>(1, shard_count) : std::size_t{0}),
          ex_shards_(hash_mode ? std::size_t{0} : std::max<std::size_t>(1, shard_count)) {}

    bool try_mark(const Fingerprint& fp) {
        std::size_t h = FingerprintHash{}(fp);
        FpShard& shard = fp_shards_[h % fp_shards_.size()];
        std::lock_guard<std::mutex> lock(shard.mu);
        return shard.states.insert(fp).second;
    }

    bool try_mark(const std::vector<int64_t>& key) {
        std::size_t h = KeyWordsHash{}(key);
        ExShard& shard = ex_shards_[h % ex_shards_.size()];
        std::lock_guard<std::mutex> lock(shard.mu);
        return shard.states.insert(key).second;
    }

private:
    struct FpShard {
        std::mutex mu;
        std::unordered_set<Fingerprint, FingerprintHash> states;
    };
    struct ExShard {
        std::mutex mu;
        std::unordered_set<std::vector<int64_t>, KeyWordsHash> states;
    };

    std::vector<FpShard> fp_shards_;
    std::vector<ExShard> ex_shards_;
};

inline bool stderr_is_tty() {
#ifdef _WIN32
    return _isatty(_fileno(stderr)) != 0;
#else
    return ::isatty(STDERR_FILENO) != 0;
#endif
}

// Per-worker progress accumulator. Each frontier subtree is worth one "unit"
// (= `subtree_ticks` ticks). Within a subtree the search distributes the unit
// fractionally by node weight (a node with k children passes weight/k to each),
// so the reported fraction advances smoothly instead of jumping per subtree.
// Updates are batched to limit atomic contention, and every finished subtree is
// snapped to contribute exactly one unit so the total converges to 100%.
struct ProgressReporter {
    std::atomic<uint64_t>* ticks = nullptr;  // shared sink (null => disabled)
    uint64_t subtree_ticks = 0;
    uint64_t flushed = 0;
    double pending = 0.0;

    void begin_subtree() {
        flushed = 0;
        pending = 0.0;
    }

    void add(double weight) {
        if (ticks == nullptr) {
            return;
        }
        pending += weight * static_cast<double>(subtree_ticks);
        if (pending >= 4096.0) {
            const uint64_t inc = static_cast<uint64_t>(pending);
            ticks->fetch_add(inc, std::memory_order_relaxed);
            flushed += inc;
            pending -= static_cast<double>(inc);
        }
    }

    void end_subtree() {
        if (ticks == nullptr) {
            return;
        }
        if (flushed < subtree_ticks) {
            ticks->fetch_add(subtree_ticks - flushed, std::memory_order_relaxed);
        }
        flushed = 0;
        pending = 0.0;
    }
};

// Renders an estimate of how much of the search tree has been explored. The
// fraction is interpolated from the shared tick counter fed by ProgressReporter
// instances, and refreshed from a dedicated thread so worker threads never block
// on rendering.
class ProgressMonitor {
public:
    ProgressMonitor(std::string label, std::size_t subtree_count, uint64_t subtree_ticks, bool enabled)
        : label_(std::move(label)),
          subtree_ticks_(std::max<uint64_t>(1, subtree_ticks)),
          subtree_count_(subtree_count),
          total_ticks_(std::max<uint64_t>(1, static_cast<uint64_t>(subtree_count) * std::max<uint64_t>(1, subtree_ticks))),
          enabled_(enabled),
          interactive_(enabled && stderr_is_tty()),
          start_(Clock::now()) {}

    std::atomic<uint64_t>* ticks_ptr() { return &ticks_; }
    uint64_t subtree_ticks() const { return subtree_ticks_; }

    void start() {
        if (!enabled_) {
            return;
        }
        render(0);
        thread_ = std::thread([this]() { loop(); });
    }

    void stop() {
        if (!enabled_) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
        render(total_ticks_, true);
    }

private:
    using Clock = std::chrono::steady_clock;

    void loop() {
        std::unique_lock<std::mutex> lock(mu_);
        const auto interval = interactive_ ? std::chrono::milliseconds(200) : std::chrono::milliseconds(2000);
        while (!stop_) {
            cv_.wait_for(lock, interval, [this]() { return stop_; });
            if (stop_) {
                break;
            }
            const uint64_t done = ticks_.load(std::memory_order_relaxed);
            lock.unlock();
            render(done);
            lock.lock();
        }
    }

    static std::string format_time(Clock::duration d) {
        auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(d).count();
        if (total_seconds < 0) {
            total_seconds = 0;
        }
        const auto hours = total_seconds / 3600;
        const auto minutes = (total_seconds / 60) % 60;
        const auto seconds = total_seconds % 60;
        std::ostringstream oss;
        if (hours > 0) {
            oss << hours << ':'
                << std::setw(2) << std::setfill('0') << minutes << ':'
                << std::setw(2) << std::setfill('0') << seconds;
        } else {
            oss << minutes << ':'
                << std::setw(2) << std::setfill('0') << seconds;
        }
        return oss.str();
    }

    void render(uint64_t done, bool finished = false) {
        static constexpr std::size_t kBarWidth = 30;
        if (done > total_ticks_) {
            done = total_ticks_;
        }
        const double ratio = static_cast<double>(done) / static_cast<double>(total_ticks_);
        std::size_t filled = static_cast<std::size_t>(ratio * static_cast<double>(kBarWidth));
        if (filled > kBarWidth) {
            filled = kBarWidth;
        }

        const auto elapsed = Clock::now() - start_;
        std::ostringstream oss;
        oss << "[progress] " << label_ << " ["
            << std::string(filled, '#')
            << std::string(kBarWidth - filled, '-')
            << "] " << std::fixed << std::setprecision(1) << (ratio * 100.0) << '%';
        if (subtree_count_ > 0) {
            const double approx = static_cast<double>(done) / static_cast<double>(subtree_ticks_);
            oss << " (~" << static_cast<uint64_t>(approx + 0.5) << '/' << subtree_count_ << " subtrees)";
        }
        oss << " elapsed " << format_time(elapsed);
        if (!finished && ratio > 1e-4) {
            const double elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
            const double eta_s = elapsed_s * (1.0 - ratio) / ratio;
            oss << " eta " << format_time(std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(eta_s)));
        }
        const std::string line = oss.str();

        if (interactive_) {
            std::cerr << '\r' << line;
            if (last_line_size_ > line.size()) {
                std::cerr << std::string(last_line_size_ - line.size(), ' ');
            }
            std::cerr << std::flush;
            last_line_size_ = line.size();
            if (finished) {
                std::cerr << '\n';
                last_line_size_ = 0;
            }
        } else {
            std::cerr << line << '\n';
        }
    }

    std::string label_;
    uint64_t subtree_ticks_;
    std::size_t subtree_count_;
    uint64_t total_ticks_;
    bool enabled_;
    bool interactive_;
    Clock::time_point start_;
    std::atomic<uint64_t> ticks_{0};
    bool stop_ = false;  // guarded by mu_
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::size_t last_line_size_ = 0;
};

inline Normal normalize_normal(int64_t nx, int64_t ny) {
    if (nx == 0 && ny == 0) {
        throw std::runtime_error("Normal (0,0) is invalid");
    }
    int64_t g = std::gcd(std::llabs(nx), std::llabs(ny));
    nx /= g;
    ny /= g;
    if (nx < 0 || (nx == 0 && ny < 0)) {
        nx = -nx;
        ny = -ny;
    }
    return {nx, ny};
}

inline std::pair<std::vector<Normal>, std::vector<int>> merge_normals(
    const std::vector<Normal>& normals,
    const std::vector<int>& max_counts
) {
    if (normals.size() != max_counts.size()) {
        throw std::runtime_error("normals and max_counts must have same length");
    }
    std::vector<Normal> merged_normals;
    std::vector<int> merged_max;
    std::unordered_map<Normal, std::size_t, NormalHash> index_by_normal;
    for (std::size_t i = 0; i < normals.size(); ++i) {
        Normal nn = normalize_normal(normals[i].first, normals[i].second);
        auto it = index_by_normal.find(nn);
        if (it == index_by_normal.end()) {
            std::size_t idx = merged_normals.size();
            merged_normals.push_back(nn);
            merged_max.push_back(max_counts[i]);
            index_by_normal.emplace(nn, idx);
        } else {
            merged_max[it->second] += max_counts[i];
        }
    }
    return {merged_normals, merged_max};
}

class GreedyCutAllSolver {
private:
    using BestMap = std::unordered_map<std::vector<int>, BestRecord, VectorIntHash>;

    struct PairCoeff {
        int64_t a1 = 0;
        int64_t b1 = 0;
        int64_t a2 = 0;
        int64_t b2 = 0;
        int64_t det = 0;
        bool valid = false;
    };

    struct Move {
        int dir = 0;
        Rational c;
        int delta = 0;
        std::vector<Point> new_points;
    };

    struct SearchState {
        std::vector<std::vector<Rational>> lines;
        std::vector<Point> points;
        std::vector<Point> seeds;
        int regions = 1;
        int depth = 0;
    };

public:
    GreedyCutAllSolver(const std::vector<Normal>& normals, const std::vector<int>& max_counts, bool return_all_minimal = false, bool use_hash_dedup = true)
        : return_all_minimal_(return_all_minimal), use_hash_dedup_(use_hash_dedup) {
        auto merged = merge_normals(normals, max_counts);
        normals_ = std::move(merged.first);
        max_counts_ = std::move(merged.second);
        m_ = static_cast<int>(normals_.size());

        bool has_horizontal_normal = false;
        for (const auto& n : normals_) {
            if (n.first == 0 && n.second == 1) {
                has_horizontal_normal = true;
                break;
            }
        }
        if (!has_horizontal_normal) {
            throw std::runtime_error("normals must include (0, 1)");
        }

        pair_coeffs_.assign(m_, std::vector<PairCoeff>(m_));
        for (int i = 0; i < m_; ++i) {
            const auto [a1, b1] = normals_[i];
            for (int j = i + 1; j < m_; ++j) {
                const auto [a2, b2] = normals_[j];
                PairCoeff coeff;
                coeff.a1 = a1;
                coeff.b1 = b1;
                coeff.a2 = a2;
                coeff.b2 = b2;
                coeff.det = a1 * b2 - a2 * b1;
                coeff.valid = true;
                pair_coeffs_[i][j] = coeff;
            }
        }
    }

    std::vector<std::pair<std::vector<int>, Solution>> solve_all() {
        reset_search_context();
        dfs(initial_state(), 1.0);
        return build_solutions(best_for_counts_);
    }

    const std::vector<Normal>& merged_normals() const {
        return normals_;
    }

    std::vector<std::pair<std::vector<int>, Solution>> solve_partitioned(int split_depth, bool show_progress, IncrementalJsonWriter* writer) {
        const int effective_split_depth = choose_frontier_split_depth(1, split_depth);
        if (show_progress) {
            std::cerr << "[progress] building frontier: split_depth=" << effective_split_depth << ", mode=serial\n";
        }

        BestMap prefix_best;
        std::vector<SearchState> frontier = build_frontier_states(effective_split_depth, prefix_best);
        std::vector<std::vector<int>> frontier_start_counts;
        frontier_start_counts.reserve(frontier.size());
        for (const SearchState& state : frontier) {
            frontier_start_counts.push_back(counts_from_lines(state.lines));
        }
        FinalizedResultCollector collector(std::move(frontier_start_counts), std::move(prefix_best), writer, return_all_minimal_);
        if (frontier.empty()) {
            if (show_progress) {
                std::cerr << "[progress] search completed during frontier expansion\n";
            }
            return build_solutions(collector.take_merged());
        }

        if (show_progress) {
            std::cerr << "[progress] processing " << frontier.size() << " frontier subtrees\n";
        }
        ProgressMonitor monitor("serial search", frontier.size(), kSubtreeTicks, show_progress);
        if (show_progress) {
            set_progress_sink(monitor.ticks_ptr(), monitor.subtree_ticks());
        }
        monitor.start();
        SharedVisited shared_visited(use_hash_dedup_, 256);
        const int shared_depth_limit = std::numeric_limits<int>::max();
        for (std::size_t idx = 0; idx < frontier.size(); ++idx) {
            BestMap partial = solve_subtree(frontier[idx], &shared_visited, shared_depth_limit);
            collector.complete_task(idx, std::move(partial));
        }
        monitor.stop();
        set_progress_sink(nullptr, 0);
        return build_solutions(collector.take_merged());
    }

    std::vector<std::pair<std::vector<int>, Solution>> solve_all_parallel(
        int num_threads,
        int split_depth,
        bool show_progress,
        IncrementalJsonWriter* writer
    ) {
        if (num_threads <= 1) {
            return (show_progress || writer != nullptr) ? solve_partitioned(split_depth, show_progress, writer) : solve_all();
        }
        split_depth = choose_frontier_split_depth(num_threads, split_depth);
        const bool debug_parallel = []() {
            const char* env = std::getenv("SOLVER_DEBUG_PARALLEL");
            return env != nullptr && env[0] != '\0' && std::string(env) != "0";
        }();
        if (show_progress) {
            std::cerr << "[progress] building frontier: split_depth=" << split_depth
                      << ", mode=parallel, requested_threads=" << num_threads << "\n";
        }

        BestMap prefix_best;
        std::vector<SearchState> frontier = build_frontier_states(split_depth, prefix_best);
        std::vector<std::vector<int>> frontier_start_counts;
        frontier_start_counts.reserve(frontier.size());
        for (const SearchState& state : frontier) {
            frontier_start_counts.push_back(counts_from_lines(state.lines));
        }
        FinalizedResultCollector collector(std::move(frontier_start_counts), std::move(prefix_best), writer, return_all_minimal_);
        if (frontier.empty()) {
            if (show_progress) {
                std::cerr << "[progress] search completed during frontier expansion\n";
            }
            if (debug_parallel) {
                std::cerr << "[parallel] frontier=0, returning prefix results\n";
            }
            return build_solutions(collector.take_merged());
        }

        int workers = std::min(num_threads, static_cast<int>(frontier.size()));
        if (show_progress) {
            std::cerr << "[progress] processing " << frontier.size()
                      << " frontier subtrees with " << workers << " workers\n";
        }
        ProgressMonitor monitor("parallel search", frontier.size(), kSubtreeTicks, show_progress);
        monitor.start();
        std::atomic<std::size_t> next_index{0};
        SharedVisited shared_visited(use_hash_dedup_, static_cast<std::size_t>(workers) * 256);
        const int shared_depth_limit = split_depth + 2;
        std::vector<std::size_t> task_counts(static_cast<std::size_t>(workers), 0);
        if (debug_parallel) {
            std::cerr << "[parallel] requested_threads=" << num_threads
                      << ", workers=" << workers
                      << ", frontier=" << frontier.size()
                      << ", split_depth=" << split_depth
                      << ", shared_depth_limit=" << shared_depth_limit << "\n";
        }
        std::vector<std::thread> threads;
        threads.reserve(workers);

        for (int w = 0; w < workers; ++w) {
            threads.emplace_back([&, w]() {
                GreedyCutAllSolver worker_solver(normals_, max_counts_, return_all_minimal_, use_hash_dedup_);
                if (show_progress) {
                    worker_solver.set_progress_sink(monitor.ticks_ptr(), monitor.subtree_ticks());
                }
                while (true) {
                    std::size_t idx = next_index.fetch_add(1);
                    if (idx >= frontier.size()) {
                        break;
                    }
                    ++task_counts[static_cast<std::size_t>(w)];
                    BestMap partial = worker_solver.solve_subtree(frontier[idx], &shared_visited, shared_depth_limit);
                    collector.complete_task(idx, std::move(partial));
                }
            });
        }
        for (std::thread& t : threads) {
            t.join();
        }
        monitor.stop();

        if (debug_parallel) {
            std::cerr << "[parallel] task_counts=[";
            for (std::size_t i = 0; i < task_counts.size(); ++i) {
                if (i > 0) {
                    std::cerr << ",";
                }
                std::cerr << task_counts[i];
            }
            std::cerr << "]\n";
        }
        return build_solutions(collector.take_merged());
    }

private:
    // Resolution of the per-subtree progress unit (ticks). Higher = smoother.
    static constexpr uint64_t kSubtreeTicks = 1u << 16;

    std::vector<Normal> normals_;
    std::vector<int> max_counts_;
    int m_ = 0;

    BestMap best_for_counts_;
    std::unordered_set<Fingerprint, FingerprintHash> visited_fp_;
    std::unordered_set<std::vector<int64_t>, KeyWordsHash> visited_exact_;
    std::vector<int64_t> key_scratch_;  // reused serialization buffer (hash mode)
    SharedVisited* shared_visited_ = nullptr;
    int shared_depth_limit_ = std::numeric_limits<int>::max();
    bool return_all_minimal_ = false;
    bool use_hash_dedup_ = true;
    ProgressReporter progress_;

    std::unordered_map<DotKey, Rational, DotKeyHash> dot_cache_;
    std::unordered_map<InterKey, std::optional<Point>, InterKeyHash> intersection_cache_;
    std::vector<std::vector<PairCoeff>> pair_coeffs_;

    void record_into_map(
        BestMap& target,
        const std::vector<int>& counts,
        int regions,
        const std::vector<std::vector<Rational>>& lines,
        const std::vector<Point>& seeds
    ) {
        auto it = target.find(counts);
        if (it == target.end() || regions < it->second.regions) {
            target[counts] = BestRecord{regions, {{lines, seeds}}};
        } else if (return_all_minimal_ && regions == it->second.regions) {
            bool dup = false;
            for (const auto& config : it->second.configs) {
                if (config.lines == lines && config.seeds == seeds) {
                    dup = true; break;
                }
            }
            if (!dup) {
                it->second.configs.push_back({lines, seeds});
            }
        }
    }

    void merge_best_maps(BestMap& target, const BestMap& src) {
        for (const auto& [counts, record] : src) {
            auto it = target.find(counts);
            if (it == target.end() || record.regions < it->second.regions) {
                target[counts] = record;
            } else if (return_all_minimal_ && record.regions == it->second.regions) {
                for (const auto& config : record.configs) {
                    bool dup = false;
                    for (const auto& existing : it->second.configs) {
                        if (config.lines == existing.lines && config.seeds == existing.seeds) {
                            dup = true; break;
                        }
                    }
                    if (!dup) {
                        it->second.configs.push_back(config);
                    }
                }
            }
        }
    }

    static Point origin() {
        return Point{Rational(0), Rational(0)};
    }

    SearchState initial_state() const {
        SearchState s;
        s.lines.assign(m_, std::vector<Rational>{});
        s.points = {origin()};
        s.seeds = {origin()};
        s.regions = 1;
        s.depth = 0;
        return s;
    }

    void reset_search_context() {
        best_for_counts_.clear();
        visited_fp_.clear();
        visited_exact_.clear();
        shared_visited_ = nullptr;
        shared_depth_limit_ = std::numeric_limits<int>::max();
        dot_cache_.clear();
        intersection_cache_.clear();
    }

    void serialize_state(const SearchState& state, std::vector<int64_t>& buf) const {
        buf.clear();
        for (int i = 0; i < m_; ++i) {
            const auto& offsets = state.lines[i];
            buf.push_back(static_cast<int64_t>(offsets.size()));
            for (const Rational& r : offsets) {
                buf.push_back(r.num);
                buf.push_back(r.den);
            }
        }
        buf.push_back(static_cast<int64_t>(state.seeds.size()));
        for (const Point& p : state.seeds) {
            buf.push_back(p.x.num);
            buf.push_back(p.x.den);
            buf.push_back(p.y.num);
            buf.push_back(p.y.den);
        }
    }

    void set_progress_sink(std::atomic<uint64_t>* ticks, uint64_t subtree_ticks) {
        progress_.ticks = ticks;
        progress_.subtree_ticks = subtree_ticks;
    }

    static int choose_frontier_split_depth(int num_threads, int split_depth) {
        if (split_depth >= 1) {
            return split_depth;
        }
        if (num_threads <= 1) {
            return 4;
        }
        int depth = 2;
        int t = 1;
        while (t < num_threads) {
            depth += 2;
            t <<= 1;
        }
        return depth;
    }

    std::vector<int> counts_from_lines(const std::vector<std::vector<Rational>>& lines) const {
        std::vector<int> counts(m_);
        for (int i = 0; i < m_; ++i) {
            counts[i] = static_cast<int>(lines[i].size());
        }
        return counts;
    }

    std::vector<std::pair<std::vector<int>, Solution>> build_solutions(const BestMap& best_map) const {
        std::vector<std::pair<std::vector<int>, Solution>> result;
        result.reserve(best_map.size());
        for (const auto& [counts, record] : best_map) {
            Solution sol;
            sol.regions = record.regions;
            sol.normals = normals_;
            const auto& first_config = record.configs.front();
            sol.lines_by_dir = first_config.lines;
            for (const auto& p : first_config.seeds) {
                if (!(p == origin())) {
                    sol.seed_points.push_back(p);
                }
            }
            if (return_all_minimal_) {
                sol.configs = record.configs;
            }
            result.push_back({counts, std::move(sol)});
        }
        std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });
        return result;
    }

    BestMap solve_subtree(const SearchState& start, SharedVisited* shared_visited, int shared_depth_limit) {
        reset_search_context();
        shared_visited_ = shared_visited;
        shared_depth_limit_ = shared_depth_limit;
        progress_.begin_subtree();
        dfs(start, 1.0);
        progress_.end_subtree();
        shared_visited_ = nullptr;
        shared_depth_limit_ = std::numeric_limits<int>::max();
        return std::move(best_for_counts_);
    }

    std::vector<SearchState> build_frontier_states(int split_depth, BestMap& prefix_best) {
        reset_search_context();
        std::vector<SearchState> frontier = {initial_state()};

        for (int depth = 0; depth < split_depth; ++depth) {
            if (frontier.empty()) {
                break;
            }

            std::vector<SearchState> next;
            std::unordered_map<StateKey, std::size_t, StateKeyHash> index_by_state;

            for (const SearchState& state : frontier) {
                std::vector<int> counts = counts_from_lines(state.lines);
                record_into_map(prefix_best, counts, state.regions, state.lines, state.seeds);

                std::vector<SearchState> children = generate_children(state, counts);
                for (SearchState& child : children) {
                    StateKey child_key{child.lines, child.seeds};
                    auto it = index_by_state.find(child_key);
                    if (it == index_by_state.end()) {
                        std::size_t idx = next.size();
                        next.push_back(std::move(child));
                        index_by_state.emplace(StateKey{next[idx].lines, next[idx].seeds}, idx);
                    } else if (child.regions < next[it->second].regions) {
                        next[it->second] = std::move(child);
                    }
                }
            }
            frontier = std::move(next);
        }

        return frontier;
    }

    // Enumerate and order the line-adding moves available from `state`. An empty
    // result means no line can be added (caller should try a seed point).
    std::vector<Move> compute_moves(const SearchState& state, const std::vector<int>& counts) {
        std::vector<Move> moves;
        std::vector<int> active_dirs;
        for (int i = 0; i < m_; ++i) {
            if (counts[i] < max_counts_[i]) {
                active_dirs.push_back(i);
            }
        }
        if (active_dirs.empty()) {
            return moves;
        }

        // 高速化：動的メモリ確保(unordered_map等)を減らすため、vectorとsort/uniqueを利用
        std::vector<int> other_lines;
        for (int i : active_dirs) {
            const auto& existing_offsets = state.lines[i];
            std::vector<Rational> candidate_offsets;
            candidate_offsets.reserve(state.points.size());

            for (const Point& p : state.points) {
                Rational c = dot_at(i, p);
                if (!contains_offset(existing_offsets, c)) {
                    candidate_offsets.push_back(c);
                }
            }
            if (candidate_offsets.empty()) {
                continue;
            }

            // 重複排除
            std::sort(candidate_offsets.begin(), candidate_offsets.end());
            candidate_offsets.erase(std::unique(candidate_offsets.begin(), candidate_offsets.end()), candidate_offsets.end());

            other_lines.clear();
            for (int j = 0; j < m_; ++j) {
                if (j != i && !state.lines[j].empty()) {
                    other_lines.push_back(j);
                }
            }

            for (const Rational& c : candidate_offsets) {
                std::vector<Point> new_points;
                for (int j : other_lines) {
                    for (const Rational& c2 : state.lines[j]) {
                        auto pt = intersection_at(i, c, j, c2);
                        if (pt.has_value()) {
                            new_points.push_back(*pt);
                        }
                    }
                }
                // std::unordered_setを避け、vectorのsort & uniqueを利用
                sort_and_unique_points(new_points);
                moves.push_back(Move{i, c, static_cast<int>(new_points.size()) + 1, std::move(new_points)});
            }
        }
        std::sort(moves.begin(), moves.end(), [](const Move& lhs, const Move& rhs) {
            if (lhs.delta != rhs.delta) {
                return lhs.delta < rhs.delta;
            }
            if (lhs.dir != rhs.dir) {
                return lhs.dir < rhs.dir;
            }
            return lhs.c < rhs.c;
        });
        return moves;
    }

    SearchState make_move_child(const SearchState& state, const Move& move) {
        SearchState child;
        child.lines = state.lines;
        insert_offset(child.lines[move.dir], move.c);
        child.points = union_points(state.points, move.new_points);
        child.seeds = state.seeds;
        child.regions = state.regions + move.delta;
        child.depth = state.depth + 1;
        return child;
    }

    std::optional<SearchState> make_seed_child(const SearchState& state, const std::vector<int>& counts) {
        auto seed = next_seed_point(state.lines, counts, state.seeds);
        if (!seed.has_value() || contains_point(state.points, *seed)) {
            return std::nullopt;
        }
        SearchState seeded;
        seeded.lines = state.lines;
        seeded.points = add_point(state.points, *seed);
        seeded.seeds = add_point(state.seeds, *seed);
        seeded.regions = state.regions;
        seeded.depth = state.depth + 1;
        return seeded;
    }

    std::vector<SearchState> generate_children(const SearchState& state, const std::vector<int>& counts) {
        std::vector<Move> moves = compute_moves(state, counts);
        if (moves.empty()) {
            auto seeded = make_seed_child(state, counts);
            if (seeded.has_value()) {
                return {std::move(*seeded)};
            }
            return {};
        }
        std::vector<SearchState> children;
        children.reserve(moves.size());
        for (const Move& move : moves) {
            children.push_back(make_move_child(state, move));
        }
        return children;
    }

    Rational dot_at(int dir, const Point& p) {
        DotKey key{dir, p};
        auto it = dot_cache_.find(key);
        if (it != dot_cache_.end()) {
            return it->second;
        }
        const auto [a, b] = normals_[dir];
        Rational value = mul_int(p.x, a) + mul_int(p.y, b);
        dot_cache_.emplace(std::move(key), value);
        return value;
    }

    std::optional<Point> intersection_at(int i, const Rational& c, int j, const Rational& c2) {
        int ii = i;
        int jj = j;
        Rational cc = c;
        Rational cc2 = c2;
        if (ii > jj) {
            std::swap(ii, jj);
            std::swap(cc, cc2);
        }
        InterKey key{ii, cc, jj, cc2};
        auto it = intersection_cache_.find(key);
        if (it != intersection_cache_.end()) {
            return it->second;
        }
        const PairCoeff& coeff = pair_coeffs_[ii][jj];
        if (!coeff.valid || coeff.det == 0) {
            intersection_cache_[key] = std::nullopt;
            return std::nullopt;
        }
        Rational x_num = mul_int(cc, coeff.b2) - mul_int(cc2, coeff.b1);
        Rational y_num = mul_int(cc2, coeff.a1) - mul_int(cc, coeff.a2);
        Point pt{div_int(x_num, coeff.det), div_int(y_num, coeff.det)};
        intersection_cache_[key] = pt;
        return pt;
    }

    std::optional<Point> next_seed_point(
        const std::vector<std::vector<Rational>>& lines,
        const std::vector<int>& counts,
        const std::vector<Point>& seeds
    ) {
        std::vector<int> active_dirs;
        for (int i = 0; i < m_; ++i) {
            if (counts[i] < max_counts_[i]) {
                active_dirs.push_back(i);
            }
        }
        if (active_dirs.empty()) {
            return std::nullopt;
        }

        std::unordered_set<Point, PointHash> used_seeds(seeds.begin(), seeds.end());
        std::unordered_set<Point, PointHash> intersections;
        for (int i = 0; i < m_; ++i) {
            if (lines[i].empty()) {
                continue;
            }
            for (int j = i + 1; j < m_; ++j) {
                if (lines[j].empty()) {
                    continue;
                }
                for (const Rational& c : lines[i]) {
                    for (const Rational& c2 : lines[j]) {
                        std::optional<Point> pt = intersection_at(i, c, j, c2);
                        if (pt.has_value()) {
                            intersections.insert(*pt);
                        }
                    }
                }
            }
        }

        std::vector<std::unordered_set<Rational, RationalHash>> blocked_offsets(m_);
        for (int i : active_dirs) {
            auto& blocked = blocked_offsets[i];
            for (const auto& p : intersections) {
                blocked.insert(dot_at(i, p));
            }
        }

        auto candidate_ok = [&](const Point& candidate) -> bool {
            if (used_seeds.find(candidate) != used_seeds.end()) {
                return false;
            }
            for (int i = 0; i < m_; ++i) {
                if (contains_offset(lines[i], dot_at(i, candidate))) {
                    return false;
                }
            }
            for (int i : active_dirs) {
                if (blocked_offsets[i].find(dot_at(i, candidate)) != blocked_offsets[i].end()) {
                    return false;
                }
            }
            return true;
        };

        for (int radius = 0;; ++radius) {
            if (radius == 0) {
                Point c{Rational(0), Rational(0)};
                if (candidate_ok(c)) {
                    return c;
                }
                continue;
            }
            for (int x = -radius; x <= radius; ++x) {
                Point bottom{Rational(x), Rational(-radius)};
                if (candidate_ok(bottom)) {
                    return bottom;
                }
                Point top{Rational(x), Rational(radius)};
                if (candidate_ok(top)) {
                    return top;
                }
            }
            for (int y = -radius + 1; y <= radius - 1; ++y) {
                Point left{Rational(-radius), Rational(y)};
                if (candidate_ok(left)) {
                    return left;
                }
                Point right{Rational(radius), Rational(y)};
                if (candidate_ok(right)) {
                    return right;
                }
            }
        }
    }

    // `weight` is this node's share of the enclosing subtree (root == 1.0); it is
    // split evenly across children and credited to the progress sink at leaves
    // and pruned (already-visited) nodes. See ProgressReporter.
    void dfs(const SearchState& state, double weight) {
        // Deduplicate identical (lines, seeds) states reached via different paths.
        if (use_hash_dedup_) {
            serialize_state(state, key_scratch_);
            const Fingerprint fp = fingerprint_words(key_scratch_);
            if (!visited_fp_.insert(fp).second) {
                progress_.add(weight);
                return;
            }
            if (shared_visited_ != nullptr && state.depth <= shared_depth_limit_ &&
                !shared_visited_->try_mark(fp)) {
                progress_.add(weight);
                return;
            }
        } else {
            std::vector<int64_t> buf;
            serialize_state(state, buf);
            auto [it, inserted] = visited_exact_.insert(std::move(buf));
            if (!inserted) {
                progress_.add(weight);
                return;
            }
            if (shared_visited_ != nullptr && state.depth <= shared_depth_limit_ &&
                !shared_visited_->try_mark(*it)) {
                progress_.add(weight);
                return;
            }
        }

        std::vector<int> counts = counts_from_lines(state.lines);
        record_into_map(best_for_counts_, counts, state.regions, state.lines, state.seeds);

        // Generate children lazily (one at a time) so we never hold the full set
        // of sibling states simultaneously -- a large peak-memory reduction.
        std::vector<Move> moves = compute_moves(state, counts);
        if (moves.empty()) {
            std::optional<SearchState> seeded = make_seed_child(state, counts);
            if (!seeded.has_value()) {
                progress_.add(weight);
                return;
            }
            dfs(*seeded, weight);
            return;
        }

        const double child_weight = weight / static_cast<double>(moves.size());
        for (const Move& move : moves) {
            dfs(make_move_child(state, move), child_weight);
        }
    }
};

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open input file: " + path);
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return text;
}

void print_json_string(std::ostream& os, const std::string& s) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default: os << c; break;
        }
    }
    os << '"';
}

std::vector<Normal> parse_normals(const JsonValue& root) {
    const auto& arr = root.require_key("normals").as_array("normals");
    std::vector<Normal> normals;
    normals.reserve(arr.size());
    for (std::size_t i = 0; i < arr.size(); ++i) {
        const auto& pair_val = arr[i].as_array("normals[" + std::to_string(i) + "]");
        if (pair_val.size() != 2) {
            throw std::runtime_error("Each normal must have exactly 2 integers");
        }
        int64_t nx = pair_val[0].as_int("normals[" + std::to_string(i) + "][0]");
        int64_t ny = pair_val[1].as_int("normals[" + std::to_string(i) + "][1]");
        normals.emplace_back(nx, ny);
    }
    return normals;
}

std::vector<int> parse_max_counts(const JsonValue& root) {
    const auto& arr = root.require_key("max_counts").as_array("max_counts");
    std::vector<int> out;
    out.reserve(arr.size());
    for (std::size_t i = 0; i < arr.size(); ++i) {
        int64_t v = arr[i].as_int("max_counts[" + std::to_string(i) + "]");
        if (v < 0 || v > std::numeric_limits<int>::max()) {
            throw std::runtime_error("max_counts must be in [0, INT_MAX]");
        }
        out.push_back(static_cast<int>(v));
    }
    return out;
}

int parse_optional_positive_int(const JsonValue& root, const std::string& key, int default_value) {
    const JsonValue* v = root.find_key(key);
    if (v == nullptr) {
        return default_value;
    }
    int64_t raw = v->as_int(key);
    if (raw < 1 || raw > std::numeric_limits<int>::max()) {
        throw std::runtime_error(key + " must be in [1, INT_MAX]");
    }
    return static_cast<int>(raw);
}

int parse_optional_nonnegative_int(const JsonValue& root, const std::string& key, int default_value) {
    const JsonValue* v = root.find_key(key);
    if (v == nullptr) {
        return default_value;
    }
    int64_t raw = v->as_int(key);
    if (raw < 0 || raw > std::numeric_limits<int>::max()) {
        throw std::runtime_error(key + " must be in [0, INT_MAX]");
    }
    return static_cast<int>(raw);
}

bool parse_optional_bool(const JsonValue& root, const std::string& key, bool default_value) {
    const JsonValue* v = root.find_key(key);
    if (v == nullptr) {
        return default_value;
    }
    return v->as_bool(key);
}

std::string parse_optional_string(const JsonValue& root, const std::string& key, const std::string& default_value) {
    const JsonValue* v = root.find_key(key);
    if (v == nullptr) {
        return default_value;
    }
    if (v->type != JsonValue::Type::kString) {
        throw std::runtime_error(key + " must be a string");
    }
    return v->string_value;
}

void print_results_json(
    std::ostream& os,
    const std::vector<std::pair<std::vector<int>, Solution>>& results,
    const std::vector<Normal>& merged_normals
) {
    os << "{\n";

    os << "  \"normals\": [";
    for (std::size_t i = 0; i < merged_normals.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << "[" << merged_normals[i].first << ", " << merged_normals[i].second << "]";
    }
    os << "],\n";

    os << "  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& counts = results[i].first;
        const auto& sol = results[i].second;
        os << "    {\n";
        os << "      \"counts\": [";
        for (std::size_t j = 0; j < counts.size(); ++j) {
            if (j > 0) {
                os << ", ";
            }
            os << counts[j];
        }
        os << "],\n";
        os << "      \"regions\": " << sol.regions << ",\n";

        os << "      \"seed_points\": [";
        for (std::size_t j = 0; j < sol.seed_points.size(); ++j) {
            if (j > 0) {
                os << ", ";
            }
            os << "[";
            print_json_string(os, sol.seed_points[j].x.to_string());
            os << ", ";
            print_json_string(os, sol.seed_points[j].y.to_string());
            os << "]";
        }
        os << "],\n";

        os << "      \"lines_by_dir\": [";
        for (std::size_t d = 0; d < sol.lines_by_dir.size(); ++d) {
            if (d > 0) { os << ", "; }
            os << "[";
            for (std::size_t k = 0; k < sol.lines_by_dir[d].size(); ++k) {
                if (k > 0) { os << ", "; }
                print_json_string(os, sol.lines_by_dir[d][k].to_string());
            }
            os << "]";
        }
        os << "]";

        if (!sol.configs.empty()) {
            os << ",\n      \"configs\": [\n";
            for (std::size_t c = 0; c < sol.configs.size(); ++c) {
                const auto& config = sol.configs[c];
                os << "        {\n";
                os << "          \"seed_points\": [";
                bool first_seed = true;
                for (const Point& seed : config.seeds) {
                    if (seed.x.num == 0 && seed.y.num == 0) { continue; }
                    if (!first_seed) { os << ", "; }
                    os << "[";
                    print_json_string(os, seed.x.to_string());
                    os << ", ";
                    print_json_string(os, seed.y.to_string());
                    os << "]";
                    first_seed = false;
                }
                os << "],\n";
                os << "          \"lines_by_dir\": [";
                for (std::size_t d = 0; d < config.lines.size(); ++d) {
                    if (d > 0) { os << ", "; }
                    os << "[";
                    for (std::size_t k = 0; k < config.lines[d].size(); ++k) {
                        if (k > 0) { os << ", "; }
                        print_json_string(os, config.lines[d][k].to_string());
                    }
                    os << "]";
                }
                os << "]\n";
                os << "        }";
                if (c + 1 < sol.configs.size()) { os << ","; }
                os << "\n";
            }
            os << "      ]\n";
        } else {
            os << "\n";
        }
        os << "    }";
        if (i + 1 < results.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ]\n";
    os << "}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc != 2 && argc != 3) {
            std::cerr << "Usage: " << argv[0] << " <input.json> [output.json]\n";
            std::cerr << "Optional JSON keys: threads (>=1), split_depth (>=0; 0=auto), progress (true/false),\n";
            std::cerr << "                   dedup (\"hash\"=default, fast/low-memory; \"exact\"=bit-exact)\n";
            return 1;
        }

        const std::string input_path = argv[1];
        const std::string text = read_file(input_path);

        JsonParser parser(text);
        JsonValue root = parser.parse();
        if (root.type != JsonValue::Type::kObject) {
            throw std::runtime_error("Input JSON root must be an object");
        }

        std::vector<Normal> normals = parse_normals(root);
        std::vector<int> max_counts = parse_max_counts(root);
        if (normals.size() != max_counts.size()) {
            throw std::runtime_error("normals and max_counts must have same length");
        }

        int threads = parse_optional_positive_int(root, "threads", 1);
        int split_depth = parse_optional_nonnegative_int(root, "split_depth", 0);
        bool progress = parse_optional_bool(root, "progress", stderr_is_tty());
        bool return_all_minimal = parse_optional_bool(root, "return_all_minimal", false);
        std::string dedup = parse_optional_string(root, "dedup", "hash");
        bool use_hash_dedup;
        if (dedup == "hash") {
            use_hash_dedup = true;
        } else if (dedup == "exact") {
            use_hash_dedup = false;
        } else {
            throw std::runtime_error("dedup must be \"hash\" or \"exact\"");
        }

        GreedyCutAllSolver solver(normals, max_counts, return_all_minimal, use_hash_dedup);
        std::optional<IncrementalJsonWriter> incremental_writer;
        if (argc == 3) {
            incremental_writer.emplace(argv[2], solver.merged_normals(), return_all_minimal);
        }

        auto results = (threads > 1)
            ? solver.solve_all_parallel(threads, split_depth, progress, incremental_writer ? &*incremental_writer : nullptr)
            : ((progress || incremental_writer)
                ? solver.solve_partitioned(split_depth, progress, incremental_writer ? &*incremental_writer : nullptr)
                : solver.solve_all());

        std::vector<Normal> merged_normals;
        if (!results.empty()) {
            merged_normals = results.front().second.normals;
        }

        if (argc == 3) {
            std::ofstream out(argv[2]);
            if (!out) {
                throw std::runtime_error("Failed to open output file");
            }
            print_results_json(out, results, merged_normals);
        } else {
            print_results_json(std::cout, results, merged_normals);
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
