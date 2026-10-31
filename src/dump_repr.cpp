#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "shi/shi_pla.hpp"

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    std::vector<uint64_t> keys;
    for (uint64_t i = 0; i < 4096; ++i) keys.push_back(i * i + 17);
    shi::SHIIndex idx(keys, 16, 0x5A175EEDULL, 64, shi::PLAMode::OptimalHull, 4);
    idx.erase(keys[100]);
    auto bytes = idx.serialize_bytes();
    std::ofstream out(argv[1], std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    std::cout << "bytes " << bytes.size() << std::endl;
    return out ? 0 : 1;
}
