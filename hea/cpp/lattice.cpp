#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

constexpr int BCC_NUM_NEIGHBORS_SHELL1 = 8;
constexpr int BCC_NUM_NEIGHBORS_SHELL2 = 6;
constexpr int BCC_NUM_NEIGHBORS_SHELL3 = 12;
constexpr int BCC_TOTAL_NEIGHBORS_PER_ATOM = BCC_NUM_NEIGHBORS_SHELL1 + BCC_NUM_NEIGHBORS_SHELL2 + BCC_NUM_NEIGHBORS_SHELL3;

constexpr int FCC_NUM_NEIGHBORS_SHELL1 = 12;
constexpr int FCC_NUM_NEIGHBORS_SHELL2 = 6;
constexpr int FCC_NUM_NEIGHBORS_SHELL3 = 12;
constexpr int FCC_TOTAL_NEIGHBORS_PER_ATOM = FCC_NUM_NEIGHBORS_SHELL1 + FCC_NUM_NEIGHBORS_SHELL2 + FCC_NUM_NEIGHBORS_SHELL3;

class Lattice {
public:
    Lattice(int cellDim, double lattCon, const std::string& lattTyp)
        : cellDim(cellDim), lattCon(lattCon), lattTyp(lattTyp) {
        if (lattTyp == "FCC") {
            coords = calcSupercellFCC(cellDim, lattCon);
        } else if (lattTyp == "BCC") {
            coords = calcSupercellBCC(cellDim, lattCon);
        } else {
            throw std::invalid_argument("Invalid lattice type: " + lattTyp);
        }
    }

    std::vector<std::array<double, 3>> calcSupercellBCC(int cellDim, double lattCon) {
        int natms = cellDim * cellDim * cellDim * 2;
        std::vector<std::array<double, 3>> coords(natms);
        int id = 0;
        for (int ix = 0; ix < cellDim; ++ix) {
            for (int iy = 0; iy < cellDim; ++iy) {
                for (int iz = 0; iz < cellDim; ++iz) {
                    coords[id++] = {(ix + 0.0) * lattCon, (iy + 0.0) * lattCon, (iz + 0.0) * lattCon};
                    coords[id++] = {(ix + 0.5) * lattCon, (iy + 0.5) * lattCon, (iz + 0.5) * lattCon};
                }
            }
        }
        return coords;
    }

    std::vector<std::array<double, 3>> calcSupercellFCC(int cellDim, double lattCon) {
        int natms = cellDim * cellDim * cellDim * 4;
        std::vector<std::array<double, 3>> coords(natms);
        int id = 0;
        for (int ix = 0; ix < cellDim; ++ix) {
            for (int iy = 0; iy < cellDim; ++iy) {
                for (int iz = 0; iz < cellDim; ++iz) {
                    coords[id++] = {(ix + 0.0) * lattCon, (iy + 0.0) * lattCon, (iz + 0.0) * lattCon};
                    coords[id++] = {(ix + 0.5) * lattCon, (iy + 0.5) * lattCon, (iz + 0.0) * lattCon};
                    coords[id++] = {(ix + 0.5) * lattCon, (iy + 0.0) * lattCon, (iz + 0.5) * lattCon};
                    coords[id++] = {(ix + 0.0) * lattCon, (iy + 0.5) * lattCon, (iz + 0.5) * lattCon};
                }
            }
        }
        return coords;
    }

    std::vector<std::array<double, 3>> getCoords() const {
        return coords;
    }

private:
    int cellDim;
    double lattCon;
    std::string lattTyp;
    std::vector<std::array<double, 3>> coords;
};


