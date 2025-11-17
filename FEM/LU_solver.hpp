#include <array>
#include <numeric>
#include <utility>
#include <cmath>
#include <cassert>
#include <iostream>



template <size_t N>
constexpr std::pair<std::array<size_t, N>, std::array<std::array<double, N>, N>> LUDecomp(const std::array<std::array<double, N>, N>& A) {

    std::array<size_t, N> perm;
    std::iota(perm.begin(), perm.end(), 0);

    std::array<std::array<double, N>, N> LU = A;

    for (size_t k = 0; k < N - 1; ++k) {

        size_t pivot_idx = k;
        for (size_t i = k + 1; i < N; ++i) {
            if (std::abs(LU[i][k]) > std::abs(LU[pivot_idx][k])) {
                pivot_idx = i;
            }
        }

        // check if matrix is singular
        assert(std::abs(LU[pivot_idx][k]) >= 1e-6);

        std::swap(perm[k], perm[pivot_idx]);
        std::swap(LU[k], LU[pivot_idx]);

        for (size_t i = k + 1; i < N; ++i) {
            double m = LU[i][k] / LU[k][k];
            LU[i][k] = m;

            for (size_t j = k + 1; j < N; ++j) {
                LU[i][j] -= m * LU[k][j];
            }
        }

    }

    return { perm, LU };

}

template <size_t N>
constexpr std::array<double, N> solveLU(const std::array<std::array<double, N>, N>& A, const std::array<double, N>& b, const std::array<size_t, N>& perm) {

    std::array<double, N> y{0.0f}, x{0.0f};

    // Ly = b
    for (size_t i = 0; i < N; ++i) {
        double sum = 0.0f;
        for (size_t j = 0; j < i; ++j) {
            sum += A[i][j] * y[j];
        }

        y[i] = b[perm[i]] - sum;
    }

    // Ux = y
    for (size_t i = N; i > 0; --i) {
        double sum = 0.0f;
        for (size_t j = i; j < N; ++j) {
            sum += A[i - 1][j] * x[j];
        }

        x[i - 1] = (y[i - 1] - sum) / A[i - 1][i - 1];
    }

    return x;
}





std::vector<double> LUDecomp(std::vector<std::vector<double>>& a) {
	size_t n = a.size();

	// we need to keep track of the permutations so we can later apply the same permutation to the vector b in Ax = b
	std::vector<double> permutation(n);
	for (size_t i = 0; i < n; ++i) {
		permutation[i] = i;
	}

	for (size_t k = 0; k < n - 1; ++k) {

		size_t pivotIndex = k;
		for (size_t i = k + 1; i < n; ++i) {
			if (std::abs(a[i][k]) > std::abs(a[pivotIndex][k])) {
				pivotIndex = i;
			}
		}

		if (std::abs(a[pivotIndex][k]) < 1e-10) {
			std::cout << "matrix is singular.\n";
			return std::vector<double>{};
		}

		std::swap(permutation[k], permutation[pivotIndex]);
		std::swap(a[k], a[pivotIndex]);

		for (size_t i = k + 1; i < n; ++i) {
			double m = a[i][k] / a[k][k];
			a[i][k] = m;

			for (size_t j = k + 1; j < n; ++j) {
				a[i][j] -= m * a[k][j];
			}
		}
	}

	return permutation;
}

std::vector<double> solveLU(const std::vector<std::vector<double>>& a, const std::vector<double>& b, const std::vector<double>& permutation) {
	size_t n = a.size();
	std::vector<double> y(n), x(n);

	// Ly = b
	for (size_t i = 0; i < n; ++i) {

		double sum = 0;
		for (size_t j = 0; j < i; ++j) {
			sum += a[i][j] * y[j];
		}

		y[i] = b[permutation[i]] - sum;
	}

	// Ux = y
	for (int i = n - 1; i >= 0; --i) {

		double sum = 0;
		for (size_t j = i + 1; j < n; ++j) {
			sum += a[i][j] * x[j];
		}

		x[i] = (y[i] - sum) / a[i][i];
	}

	return x;
}