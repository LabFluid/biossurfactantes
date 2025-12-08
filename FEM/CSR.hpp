
#pragma once

#include <vector>
#include <algorithm>

struct CSR {

	size_t rows, cols, nnz;
	std::vector<std::vector<double>> values;
	std::vector<std::vector<size_t>> col_idx;

	// m is expected to have at least one row and all its columns should have the same size
	CSR(const std::vector<std::vector<double>>& m, double tol = 0.0) : rows(m.size()), cols(m[0].size()), nnz(0) {

		values.resize(rows);
		col_idx.resize(rows);

		for (size_t i = 0; i < rows; ++i) {

			for (size_t j = 0; j < cols; ++j) {
				// this ensures small enough elements can be considered zeros
				if (std::abs(m[i][j]) <= tol) continue;

				values[i].push_back(m[i][j]);
				col_idx[i].push_back(j);
				nnz++;
			}
		}

	}

	CSR(size_t m, size_t n) : rows(m), cols(n), nnz(0) {
		values.resize(rows);
		col_idx.resize(rows);
	}

	// returns number of non-zero entries
	size_t getNNZ() const {
		return nnz;
	}

	double getDensity() const {
		return (double) getNNZ() / (rows * cols);
	}

	double getSparsity() const {
		return 1.0 - getDensity();
	}

	inline size_t getRowSize(size_t row) const {
		return col_idx[row].size();
	}

	// get index of n-th column with non-zero entry at given row
	inline size_t getRowNthColumn(size_t row, size_t n) const {
		return col_idx[row][n];
	}

	// get value of n-th non-zero entry at given row
	inline double getRowNthElement(size_t row, size_t n) const {
		return values[row][n];
	}

	inline double getElement(size_t row, size_t col) const {
		size_t idx = std::distance(col_idx[row].begin(), std::lower_bound(col_idx[row].begin(), col_idx[row].end(), col));

		// there is an element in this position. Return it
		if (col_idx[row].size() > 0 && col_idx[row][idx] == col) {
			return values[row][idx];
		}

		return 0.0;
	}

	inline size_t getGreaterEqualColIdx(size_t row, size_t col) const {
		return std::distance(col_idx[row].begin(), std::lower_bound(col_idx[row].begin(), col_idx[row].end(), col));
	}
	inline size_t getGreaterColIdx(size_t row, size_t col) const {
		return std::distance(col_idx[row].begin(), std::upper_bound(col_idx[row].begin(), col_idx[row].end(), col));
	}

	
	void setElement(size_t row, size_t col, double elem) {

		// insert element but keep col_idx sorted
		size_t idx = std::distance(col_idx[row].begin(), std::lower_bound(col_idx[row].begin(), col_idx[row].end(), col));

		// there already exists an element in this position. Just update its value
		if (col_idx[row].size() > 0 && col_idx[row][idx] == col) {
			values[row][idx] = elem;
			return;
		}

		nnz++;

		values[row].insert(values[row].begin() + idx, elem);
		col_idx[row].insert(col_idx[row].begin() + idx, col);
	}

	// same thing as setElement, but if the element is already non-zero, add both together (instead of replacing)
	void addElement(size_t row, size_t col, double elem) {

		// insert element but keep col_idx sorted
		size_t idx = std::distance(col_idx[row].begin(), std::lower_bound(col_idx[row].begin(), col_idx[row].end(), col));

		// there already exists an element in this position. Just update its value
		if (col_idx[row].size() > 0 && col_idx[row][idx] == col) {
			values[row][idx] += elem;
			return;
		}

		nnz++;

		values[row].insert(values[row].begin() + idx, elem);
		col_idx[row].insert(col_idx[row].begin() + idx, col);
	}

	void updateElement(size_t row, size_t col, double val) {
		size_t idx = std::distance(col_idx[row].begin(), std::lower_bound(col_idx[row].begin(), col_idx[row].end(), col));
		values[row][idx] = val;
	}

};




CSR transpose(const CSR& a) {
	size_t n = a.rows;
	size_t m = a.cols;

	CSR transposed{m, n};
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < a.getRowSize(i); ++j) {
			double elem = a.getRowNthElement(i, j);
			size_t jth_col_idx = a.getRowNthColumn(i, j);
			transposed.values[jth_col_idx].push_back(elem);
			transposed.col_idx[jth_col_idx].push_back(i);
		}
	}
	transposed.nnz = a.nnz;

	return transposed;
}