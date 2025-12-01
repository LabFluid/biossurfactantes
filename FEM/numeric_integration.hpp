#include <iostream>
#include <array>
#include <cmath>
#include <tuple>
#include <utility>


template <size_t N>
constexpr double NewtonPolynomial(const std::array<double, N>& coeffs, double x, int max_iter = 10, double eps = 1e-6) {

	if constexpr (N <= 1) return 0.0f;

	while (max_iter-- > 0) {
		double b = coeffs[N - 1];
		double c = b;

		for (size_t i = N - 2; i > 0; --i) {
			b = b * x + coeffs[i];
			c = c * x + b;
		}

		b = b * x + coeffs[0];
		x -= b / c;

		if (std::abs(b) <= eps) break;
	}

	return x;

}


template <size_t N>
constexpr double evaluatePolynomialDerivative(const std::array<double, N>& coeffs, double x) {

	if constexpr (N <= 1) return 0.0f;

	double v = coeffs[N - 1] * (static_cast<double>(N) - 1.0f);

	for (size_t i = N - 2; i > 0; --i) {
		v = v * x + coeffs[i] * static_cast<double>(i);
	}

	return v;
}





// uses the recurrent relation (n+1) * P_n+1 = (2n + 1)x * P_n - n * P_n-1, with P_0 = 1 and P_1 = x
template <size_t N>
constexpr std::array<double, N + 1> LegendreCoefficients() {
	std::array<double, N + 1> P_last{0.0f}, P_now{0.0f}, P_next{0.0f};

    // P_0 = 1 and P_1 = x
    P_last[0] = 1.0f;
	P_now[1] = 1.0f;

    // since we'll already start at n = 1
    if constexpr (N == 0) {
        return P_last;
    }

	for (size_t i = 1; i < N; ++i) {

		double n = static_cast<double>(i);

		double term1 = -n / (n + 1.0f);
		double term2 = (2.0f * n + 1.0f) / (n + 1.0f);

		P_next[0] = term1 * P_last[0];
		for (size_t j = 1; j <= i; ++j) {
			P_next[j] = term2 * P_now[j - 1] + term1 * P_last[j];
		}
		P_next[i + 1] = term2 * P_now[i];

		std::swap(P_last, P_now);
		std::swap(P_now, P_next);
	}

	return P_now;
}


// can't find where I got this formula from but it works great
template <size_t N>
constexpr std::array<double, N> estimateLegendreRoots() {
	double term1 = 3.14159265359f / (static_cast<double>(N) + 0.5f);
	double term2 = 1.0f - 1.0f / (8.0f * static_cast<double>(N * N)) + 1.0f / (8.0f * static_cast<double>(N * N * N));

	std::array<double, N> roots{0.0f};
	for (size_t i = 0; i < N / 2; ++i) {
		double theta = (static_cast<double>(N - i) - 0.25f) * term1;

		// std::cos will only be constexpr in c++26 (but GCC already handles it)
		// since this is an approximation after all it wouldn't be that difficult to make a constexpr cos for before c++26
		roots[i] = term2 * std::cos(theta);
		roots[N - i - 1] = -roots[i];
	}

	return roots;
}


template <size_t N>
constexpr std::pair<std::array<double, N + 1>, std::array<double, N>> getLegendreCoefficientsAndRoots() {
	auto coeffs = LegendreCoefficients<N>();
	auto roots = estimateLegendreRoots<N>();

	for (size_t i = 0; i < N / 2; ++i) {
		roots[i] = NewtonPolynomial(coeffs, roots[i]);
		roots[N - i - 1] = -roots[i];
	}

	return { coeffs, roots };
}


template <size_t N>
constexpr std::pair<std::array<double, N>, std::array<double, N>> GaussLegendrePointsAndWeights() {

	auto [coeffs, roots] = getLegendreCoefficientsAndRoots<N>();
	std::array<double, N> weights{0.0f};

	for (size_t i = 0; i <= N / 2; ++i) {
		double derivative = evaluatePolynomialDerivative(coeffs, roots[i]);

		weights[i] = 2.0f / ((1.0f - roots[i] * roots[i]) * derivative * derivative);
		weights[N - i - 1] = weights[i];
	}

	return { roots, weights };

}



template <size_t N, typename FuncType>
double GaussLegendreQuadrature(const FuncType& f, double a, double b) {
	static constexpr auto XWs = GaussLegendrePointsAndWeights<N>();
	auto Xs = XWs.first;
	auto Ws = XWs.second;

	double I = 0.0f;
	for (size_t i = 0; i < N; ++i) {
		I += Ws[i] * f(0.5 * (a + b + Xs[i] * (a - b)));
	}

	return I * (b - a) * 0.5;
}

