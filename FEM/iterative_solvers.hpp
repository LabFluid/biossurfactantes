#include <vector>

#include "CSR.hpp"


typedef std::vector<double> Vec;
typedef std::vector<std::vector<double>> Mat;

template <typename T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& v) {

	os << "(";
	for (size_t i = 0; i < v.size(); ++i) {
		os << v[i];

		if (i + 1 < v.size()) {
			os << ", ";
		}
	}
	os << ")";

	return os;
}


Vec operator * (const Mat& m, const Vec& v) {
	size_t n = v.size();

	Vec res(n, 0.0);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			res[i] += m[i][j] * v[j];
		}
	}

	return res;
}






double calculateResidualNorm(const Mat& a, const Vec& b, const Vec& x) {
	size_t n = x.size();
	double maxR = -1.0;

	for (size_t i = 0; i < n; ++i) {
		double s = b[i];
		for (size_t j = 0; j < n; ++j) {
			s -= a[i][j] * x[j];
		}
		maxR = std::max(maxR, std::abs(s));
	}

	return maxR;
}

double updateResidual(const Mat& a, const Vec& b, const Vec& x, Vec& r) {
	size_t n = x.size();
	double maxR = -1.0;

	for (size_t i = 0; i < n; ++i) {
		r[i] = b[i];
		for (size_t j = 0; j < n; ++j) {
			r[i] -= a[i][j] * x[j];
		}
		maxR = std::max(maxR, std::abs(r[i]));
	}

	return maxR;
}


double calculateResidualNorm(const CSR& a, const Vec& b, const Vec& x) {
	size_t n = x.size();
	double maxR = -1.0;

	for (size_t i = 0; i < n; ++i) {
		double s = b[i];

		for (size_t j = 0; j < a.getRowSize(i); ++j) {
			s -= a.getRowNthElement(i, j) * x[a.getRowNthColumn(i, j)];
		}

		maxR = std::max(maxR, std::abs(s));
	}

	return maxR;
}




struct IterativeSolver {

	virtual bool solve(const Mat& a, const Vec& b, Vec& x, int maxIter = 1000, double tol = 1e-4) = 0;
	virtual bool solve(const CSR& a, const Vec& b, Vec& x, int maxIter = 1000, double tol = 1e-4) = 0;

};


struct Jacobi : IterativeSolver {

	bool solve(const Mat& a, const Vec& b, Vec& x, int maxIter, double tol) override {

		size_t n = x.size();
		int iter = 0;

		while (calculateResidualNorm(a, b, x) > tol && iter++ < maxIter) {

			Vec lastX = x;

			for (size_t i = 0; i < n; ++i) {
				x[i] = b[i];

				for (size_t j = 0; j < i; ++j) {
					x[i] -= a[i][j] * lastX[j];
				}
				for (size_t j = i + 1; j < n; ++j) {
					x[i] -= a[i][j] * lastX[j];
				}

				x[i] /= a[i][i];
			}
		}

		return iter < maxIter;
	}

	bool solve(const CSR& a, const Vec& b, Vec& x, int maxIter, double tol) override {

		size_t n = x.size();
		int iter = 0;

		while (calculateResidualNorm(a, b, x) > tol && iter++ < maxIter) {

			Vec lastX = x;

			for (size_t i = 0; i < n; ++i) {
				x[i] = b[i];

				double a_ii = 0.0;

				for (size_t j = 0; j < a.getRowSize(i); ++j) {

					if (a.getRowNthColumn(i, j) == i) {
						a_ii = a.getRowNthElement(i, j);
						continue;
					}

					x[i] -= a.getRowNthElement(i, j) * lastX[a.getRowNthColumn(i, j)];
				}

				x[i] /= a_ii;
			}
		}

		return iter < maxIter;
	}
};



struct GaussSeidel : IterativeSolver {

	bool solve(const Mat& a, const Vec& b, Vec& x, int maxIter, double tol) override {

		size_t n = x.size();
		int iter = 0;

		while (calculateResidualNorm(a, b, x) > tol && iter++ < maxIter) {

			for (size_t i = 0; i < n; ++i) {
				x[i] = b[i];

				for (size_t j = 0; j < i; ++j) {
					x[i] -= a[i][j] * x[j];
				}
				for (size_t j = i + 1; j < n; ++j) {
					x[i] -= a[i][j] * x[j];
				}

				x[i] /= a[i][i];
			}
		}

		return iter < maxIter;
	}

	bool solve(const CSR& a, const Vec& b, Vec& x, int maxIter, double tol) override {

		size_t n = x.size();
		int iter = 0;

		while (calculateResidualNorm(a, b, x) > tol && iter++ < maxIter) {

			for (size_t i = 0; i < n; ++i) {
				x[i] = b[i];

				double a_ii = 0.0;

				for (size_t j = 0; j < a.getRowSize(i); ++j) {

					if (a.getRowNthColumn(i, j) == i) {
						a_ii = a.getRowNthElement(i, j);
						continue;
					}

					x[i] -= a.getRowNthElement(i, j) * x[a.getRowNthColumn(i, j)];
				}

				x[i] /= a_ii;
			}
		}

		return iter < maxIter;
	}
};



struct SOR : IterativeSolver {

	double omega;
	SOR(double omega = 1.4) : omega(omega) {}

	bool solve(const Mat& a, const Vec& b, Vec& x, int maxIter, double tol) override {

		size_t n = x.size();
		int iter = 0;

		while (calculateResidualNorm(a, b, x) > tol && iter++ < maxIter) {

			for (size_t i = 0; i < n; ++i) {
				
				double s = b[i];

				for (size_t j = 0; j < i; ++j) {
					s -= a[i][j] * x[j];
				}
				for (size_t j = i + 1; j < n; ++j) {
					s -= a[i][j] * x[j];
				}

				s /= a[i][i];

				x[i] = omega * s + (1.0 - omega) * x[i];
			}
		}

		return iter < maxIter;
	}

	bool solve(const CSR& a, const Vec& b, Vec& x, int maxIter, double tol) override {

		size_t n = x.size();
		int iter = 0;

		while (calculateResidualNorm(a, b, x) > tol && iter++ < maxIter) {

			for (size_t i = 0; i < n; ++i) {

				double s = b[i];
				double a_ii = 0.0;

				for (size_t j = 0; j < a.getRowSize(i); ++j) {

					if (a.getRowNthColumn(i, j) == i) {
						a_ii = a.getRowNthElement(i, j);
						continue;
					}

					s -= a.getRowNthElement(i, j) * x[a.getRowNthColumn(i, j)];
				}

				s /= a_ii;
				x[i] = omega * s + (1.0 - omega) * x[i];

			}
		}

		return iter < maxIter;
	}
};


struct Gradient : IterativeSolver {

	bool solve(const Mat& a, const Vec& b, Vec& x, int maxIter, double tol) override {

		size_t n = x.size();

		Vec r(n, 0.0);
		int iter = 0;

		while (updateResidual(a, b, x, r) > tol && iter++ < maxIter) {

			double s = (r * r) / (r * (a * r));

			for (size_t i = 0; i < n; ++i) {
				x[i] += s * r[i];
			}
		}

		return iter < maxIter;
	}

	bool solve(const CSR& a, const Vec& b, Vec& x, int maxIter, double tol) override {

		size_t n = x.size();

		Vec r(n, 0.0);
		int iter = 0;

		while (updateResidual(a, b, x, r) > tol && iter++ < maxIter) {

			double s = (r * r) / (r * (a * r));

			for (size_t i = 0; i < n; ++i) {
				x[i] += s * r[i];
			}
		}

		return iter < maxIter;
	}
};



struct ConjugateGradient : IterativeSolver {

	bool solve(const Mat& a, const Vec& b, Vec& x, int maxIter, double tol) override {

		size_t n = x.size();

		Vec r(n, 0.0);
		double maxR = updateResidual(a, b, x, r);
		Vec p = r;

		int iter = 0;

		while (maxR > tol && iter++ < maxIter) {

			Vec Ap = a * p;
			double beta = (r * p) / (p * Ap);

			for (size_t i = 0; i < n; ++i) {
				x[i] += beta * p[i];
			}

			maxR = updateResidual(a, b, x, r);

			double alpha = -(r * Ap) / (p * Ap);

			for (size_t i = 0; i < n; ++i) {
				p[i] = r[i] + alpha * p[i];
			}
		}

		return iter < maxIter;
	}

	bool solve(const CSR& a, const Vec& b, Vec& x, int maxIter, double tol) override {

		size_t n = x.size();

		Vec r(n, 0.0);
		double maxR = updateResidual(a, b, x, r);
		Vec p = r;

		int iter = 0;

		while (maxR > tol && iter++ < maxIter) {

			Vec Ap = a * p;
			double beta = (r * p) / (p * Ap);

			for (size_t i = 0; i < n; ++i) {
				x[i] += beta * p[i];
			}

			maxR = updateResidual(a, b, x, r);

			double alpha = -(r * Ap) / (p * Ap);

			for (size_t i = 0; i < n; ++i) {
				p[i] = r[i] + alpha * p[i];
			}
		}

		if (iter >= maxIter) {
			std::cout << "Conjugate gradient failed to converge in " << maxIter << " iterations.\n";
			std::cout << "Residual norm: " << calculateResidualNorm(a, b, x) << "\n";
			return false;
		} else {
			std::cout << "Conjugate gradient converged in " << iter << " iterations.\n";
			std::cout << "Residual norm: " << calculateResidualNorm(a, b, x) << "\n";
			return true;
		}

		return iter < maxIter;
	}
};
