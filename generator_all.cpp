/*
 * Generate some example dense matrices for verification tests.
 *
 * Faculty of Mathematics, Informatics and Mechanics.
 * University of Warsaw, Warsaw, Poland.
 * 
 * Krzysztof Rzadca
 * LGPL, 2023
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "densematgen.h"


int main() {
    const int seeds[] = { 42, 442, 4242, 44242 };
    const int sizes[] = { 10, 64, 512 };
    for (int size_i = 0; size_i < sizeof(sizes)/sizeof(sizes[0]); size_i++) {
        for (int seed_i = 0; seed_i < sizeof(seeds)/sizeof(seeds[0]); seed_i++) {
            std::ostringstream filename_s;
            filename_s << "dense_" << std::setw(5) << std::setfill('0')
                       << sizes[size_i] << "_" << std::setw(5)
                       << std::setfill('0') << seeds[seed_i];
            std::ofstream mat_stream;
            mat_stream.open(filename_s.str());
            mat_stream << sizes[size_i] << " " << sizes[size_i] << std::endl;
            for (int r = 0; r < sizes[size_i]; r++) {
                for (int c = 0; c < sizes[size_i]; c++) {
                    const double entry = generate_double(seeds[seed_i], r, c);
                    mat_stream << entry << " ";
                }
                mat_stream << std::endl;
            }
            mat_stream.close();
        }
    }
    return 0;
}

