
#pragma once

#include <vector>
#include <cassert>
#include <iostream>
#include <utility>

#include "CSR.hpp"





double operator * (const std::vector<double>& v1, const std::vector<double>& v2) {
	size_t n = v1.size();

	double res = 0.0;
	for (size_t i = 0; i < n; ++i) {
		res += v1[i] * v2[i];
	}

	return res;
}




std::vector<double> operator * (const CSR& m, const std::vector<double>& v) {

	std::vector<double> res(m.rows, 0.0);
	for (size_t i = 0; i < m.rows; ++i) {

		for (size_t j = 0; j < m.getRowSize(i); ++j) {
			res[i] += m.getRowNthElement(i, j) * v[m.getRowNthColumn(i, j)];
		}

	}

	return res;
}




double updateResidual(const CSR& a, const std::vector<double>& b, const std::vector<double>& x, std::vector<double>& r) {
	size_t n = x.size();
	double maxR = -1.0;

	for (size_t i = 0; i < n; ++i) {
		r[i] = b[i];
		for (size_t j = 0; j < a.getRowSize(i); ++j) {
			r[i] -= a.getRowNthElement(i, j) * x[a.getRowNthColumn(i, j)];
		}
		maxR = std::max(maxR, std::abs(r[i]));
	}

	return maxR;
}






// a must be symmetric and positive definite
std::pair<CSR, CSR> incompleteCholeskyDecomp(const CSR& a) {

	CSR a_transposed = transpose(a);

    size_t n = a.rows;
    CSR g{n, n};

	// store the transpose as well so we can take advantage of sparsity better when solving G^Tx = y
    CSR gT{n, n};

    for (size_t k = 0; k < n; ++k) {

    	double sum = 0.0;

    	for (size_t j = 0; j < g.getRowSize(k); ++j) {
    		if (g.getRowNthColumn(k, j) >= k) break;
    		sum += std::pow(g.getRowNthElement(k, j), 2);
    	}

    	double r = a.getElement(k, k) - sum;

    	// r cannot be negative
    	assert(r > 0.0);

    	double g_kk = std::sqrt(r); // a[k][k] must be nonzero so I don't have to check
    	g.setElement(k, k, g_kk);
    	gT.setElement(k, k, g_kk);

    	// use the transpose here so we don't have to check in every line or the matrix
    	for (size_t i = a_transposed.getGreaterColIdx(k, k); i < a_transposed.getRowSize(k); ++i) {

    		size_t ith_col_idx = a_transposed.getRowNthColumn(k, i);

    		sum = 0.0;

    		// god this is terrible but I couldn't think of a better way to do it
    		size_t row_size_k = g.getRowSize(k);
    		size_t row_size_i = g.getRowSize(ith_col_idx);

    		size_t j_k = 0;
    		size_t j_i = 0;
    		while (j_k < row_size_k && j_i < row_size_i) {

    			size_t column_k = g.getRowNthColumn(k, j_k);
    			size_t column_i = g.getRowNthColumn(ith_col_idx, j_i);

    			if (column_i < column_k) j_i++;
    			else if (column_i > column_k) j_k++;
    			else sum += g.getRowNthElement(ith_col_idx, j_i++) * g.getRowNthElement(k, j_k++);
    		}

    		double new_elem = (a_transposed.getRowNthElement(k, i) - sum) / g_kk;
			g.setElement(ith_col_idx, k, new_elem);
			gT.setElement(k, ith_col_idx, new_elem);
    	}
    }

    return { g, gT };
}


std::vector<double> solveLowerUpperTriangular(const CSR& lower, const CSR& upper, const std::vector<double>& b) {
	size_t n = b.size();
	std::vector<double> y(n);
	std::vector<double> x(n);

	// Ly = b
	for (size_t i = 0; i < n; ++i) {
		double sum = 0.0;

		for (size_t j = 0; j < lower.getRowSize(i); ++j) {
			size_t j_th_column = lower.getRowNthColumn(i, j);
    		if (j_th_column >= i) break;
    		sum += lower.getRowNthElement(i, j) * y[j_th_column];
    	}

    	y[i] = (b[i] - sum) / lower.getElement(i, i);
	}

	// Ux = y
	for (size_t i = n; i > 0; --i) {
		double sum = 0.0;

		for (size_t j = upper.getGreaterColIdx(i - 1, i - 1); j < upper.getRowSize(i - 1); ++j) {
			sum += upper.getRowNthElement(i - 1, j) * x[upper.getRowNthColumn(i - 1, j)];
		}

		x[i - 1] = (y[i - 1] - sum) / upper.getElement(i - 1, i - 1);
	}

	return x;
}





bool ICCG(const CSR& a, const std::vector<double>& b, std::vector<double>& x, int maxIter = 1000, double tol = 1e-12) {
	size_t n = x.size();

	auto [m, mT] = incompleteCholeskyDecomp(a);

	std::vector<double> r(n, 0.0);
	double maxR = updateResidual(a, b, x, r);

	std::vector<double> z = solveLowerUpperTriangular(m, mT, r);
	std::vector<double> p = z;

	double rz = r * z;

	int iter = 0;
	while (maxR > tol && iter++ < maxIter) {

		std::vector<double> Ap = a * p;

		double alpha = rz / (p * Ap);

		maxR = 0.0;
		for (size_t i = 0; i < n; ++i) {
			x[i] += alpha * p[i];
			r[i] -= alpha * Ap[i];
			maxR = std::max(maxR, std::abs(r[i]));
		}

		z = solveLowerUpperTriangular(m, mT, r);

		double rz_next = r * z;
		double beta = rz_next / rz;
		rz = rz_next;

		for (size_t i = 0; i < n; ++i) {
			p[i] = z[i] + beta * p[i];
		}
	}

	if (iter >= maxIter) {
		std::cout << "ICCG failed to converge in " << maxIter << " iterations.\n";
		std::cout << "Residual norm: " << updateResidual(a, b, x, r) << "\n";
		return false;
	} else {
		std::cout << "ICCG converged in " << iter << " iterations.\n";
		std::cout << "Residual norm: " << updateResidual(a, b, x, r) << "\n";
		return true;
	}

	return iter < maxIter;
}