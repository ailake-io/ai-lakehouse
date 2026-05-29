// SPDX-License-Identifier: MIT OR Apache-2.0
// CPU search example.
//
// Build:
//   cmake -B build && cmake --build build
//   ./build/ailake_search -w /data/warehouse -t default.docs -d 1536 -k 10
//
// With CUDA:
//   cmake -B build -DAILAKE_CUDA=ON && cmake --build build

#include <ailake/ailake.hpp>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

int main(int argc, char** argv) {
    std::string warehouse = ".";
    std::string table_arg = "default.table";
    int         dim       = 64;
    int         top_k     = 10;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-w" && i+1 < argc) warehouse = argv[++i];
        else if (a == "-t" && i+1 < argc) table_arg = argv[++i];
        else if (a == "-d" && i+1 < argc) dim = std::stoi(argv[++i]);
        else if (a == "-k" && i+1 < argc) top_k = std::stoi(argv[++i]);
    }

    // Parse "namespace.table"
    auto dot = table_arg.find('.');
    std::string ns  = (dot == std::string::npos) ? "default" : table_arg.substr(0, dot);
    std::string tbl = (dot == std::string::npos) ? table_arg : table_arg.substr(dot + 1);

    // Random query vector
    std::vector<float> query(dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : query) v = dist(rng);

    ailake::HadoopCatalog catalog(warehouse);

    try {
        auto info = catalog.load_table(ns, tbl);
        std::cout << "table:   " << info.table << "\n"
                  << "col:     " << info.vector_column << " dim=" << info.vector_dim
                  << " metric=" << info.vector_metric << "\n";
        if (info.snapshot_id) std::cout << "snapshot:" << *info.snapshot_id << "\n";
        std::cout << "\n";

        ailake::SearchOptions opts;
        opts.top_k = top_k;
        auto results = ailake::search(catalog, ns, tbl, query.data(), dim, opts);

        std::printf("%-6s %-12s %s\n", "rank", "distance", "file_path");
        for (size_t i = 0; i < results.size(); ++i)
            std::printf("%-6zu %-12.6f %s (row_id=%llu)\n",
                i+1, results[i].distance, results[i].file_path.c_str(),
                (unsigned long long)results[i].row_id);

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
