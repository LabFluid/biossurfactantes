#include <vector>
#include <array>
#include <iostream>
#include <numbers>
#include <omp.h>

#include "numeric_integration.hpp"
#include "CSR.hpp"
#include "ICCG.hpp"
#include "iterative_solvers.hpp"

#include "graph.h"

struct Point2
{
	double x, y;

	Point2(double x, double y) : x(x), y(y)
	{
	}
};

double get_det(const std::array<std::array<double, 2>, 2> &mat)
{
	return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
}

std::array<double, 2> inv_base_transform(double x, double y, const Point2 &p1, const Point2 &p2, const Point2 &p3, const Point2 &p4)
{
	return {
		(x - p1.x) / (p3.x - p1.x) * 2.0 - 1.0,
		(y - p1.y) / (p3.y - p1.y) * 2.0 - 1.0};
}

// função que leva da base (xi, eta) -> (x, y) (por elemento!)
std::array<double, 2> base_transform(double xi, double eta, const Point2 &p1, const Point2 &p2, const Point2 &p3, const Point2 &p4)
{

	// accidentaly implemented considering (xi, eta) in the domain [0, 1]x[0, 1], so just transform them here so I don't have to redo this function
	xi = (xi + 1.0) / 2.0;
	eta = (eta + 1.0) / 2.0;

	return {
		(1.0 - xi) * (1.0 - eta) * p1.x + xi * (1.0 - eta) * p2.x + xi * eta * p3.x + (1.0 - xi) * eta * p4.x,
		(1.0 - xi) * (1.0 - eta) * p1.y + xi * (1.0 - eta) * p2.y + xi * eta * p3.y + (1.0 - xi) * eta * p4.y};
}

std::array<std::array<double, 2>, 2> transformation_jacobian(double xi, double eta, const Point2 &p1, const Point2 &p2, const Point2 &p3, const Point2 &p4)
{

	// same thing
	xi = (xi + 1.0) / 2.0;
	eta = (eta + 1.0) / 2.0;

	// | dx/dxi   dx/deta |
	// | dy/dxi   dy/deta |

	return std::array<std::array<double, 2>, 2>({std::array<double, 2>({0.5 * (-(1.0 - eta) * p1.x + (1.0 - eta) * p2.x + eta * p3.x - eta * p4.x), 0.5 * (-(1.0 - xi) * p1.x - xi * p2.x + xi * p3.x + (1.0 - xi) * p4.x)}),
												 std::array<double, 2>({0.5 * (-(1.0 - eta) * p1.y + (1.0 - eta) * p2.y + eta * p3.y - eta * p4.y), 0.5 * (-(1.0 - xi) * p1.y - xi * p2.y + xi * p3.y + (1.0 - xi) * p4.y)})});
}

// shape functions: Q1 Lagrange over [-1, 1]x[-1, 1]
double N(int i, double xi, double eta)
{
	if (i == 1)
	{
		return 0.25 * (1.0 - xi) * (1.0 - eta);
	}
	else if (i == 2)
	{
		return 0.25 * (1.0 + xi) * (1.0 - eta);
	}
	else if (i == 3)
	{
		return 0.25 * (1.0 + xi) * (1.0 + eta);
	}
	else if (i == 4)
	{
		return 0.25 * (1.0 - xi) * (1.0 + eta);
	}

	return 0.0;
}

std::array<double, 2> N_grad(int i, double xi, double eta)
{
	if (i == 1)
	{
		return {-0.25 * (1.0 - eta), -0.25 * (1.0 - xi)};
	}
	else if (i == 2)
	{
		return {0.25 * (1.0 - eta), -0.25 * (1.0 + xi)};
	}
	else if (i == 3)
	{
		return {0.25 * (1.0 + eta), 0.25 * (1.0 + xi)};
	}
	else if (i == 4)
	{
		return {-0.25 * (1.0 + eta), 0.25 * (1.0 - xi)};
	}

	return {0.0, 0.0};
}

std::array<double, 2> N_grad_physical(int i, double xi, double eta, const Point2 &p1, const Point2 &p2, const Point2 &p3, const Point2 &p4)
{
	auto [dNdxi, dNdeta] = N_grad(i, xi, eta);

	auto jac = transformation_jacobian(xi, eta, p1, p2, p3, p4);
	double det = get_det(jac);

	return {
		(jac[1][1] * dNdxi - jac[0][1] * dNdeta) / det,
		(-jac[1][0] * dNdxi + jac[0][0] * dNdeta) / det};
}

double integrate2D(const auto &f)
{

	auto int_f_fixed_y = [&](double y)
	{
		auto f_y = [&](double x)
		{ return f(x, y); };
		return GaussLegendreQuadrature<5>(f_y, -1.0, 1.0);
	};

	return GaussLegendreQuadrature<5>(int_f_fixed_y, -1.0, 1.0);
}

bool is_Point2_in_rect(const Point2 &p, const Point2 &bottom_left, const Point2 &top_right, double tol = 1e-14)
{
	return (p.x >= bottom_left.x - tol) &&
		   (p.x <= top_right.x + tol) &&
		   (p.y >= bottom_left.y - tol) &&
		   (p.y <= top_right.y + tol);
}

double evaluate_sol(double x, double y, const auto &coeffs, const auto &rectangles, const auto &nodes)
{
	for (size_t i = 0; i < rectangles.size(); ++i)
	{
		if (!is_Point2_in_rect(Point2{x, y}, nodes[rectangles[i][0]], nodes[rectangles[i][2]]))
			continue;

		auto [idx1, idx2, idx3, idx4] = rectangles[i];
		auto [xi, eta] = inv_base_transform(x, y, nodes[idx1], nodes[idx2], nodes[idx3], nodes[idx4]);

		return coeffs[idx1] * N(1, xi, eta) + coeffs[idx2] * N(2, xi, eta) + coeffs[idx3] * N(3, xi, eta) + coeffs[idx4] * N(4, xi, eta);
	}

	return 0.0;
}

int main(int argc, char *argv[])
{

	// quantas vezes dividir o espaço em cada eixo
	int gridX = argv[1] ? std::atoi(argv[1]) : 25;
	int gridY = gridX;

	double minX = 0.0, maxX = 1.0;
	double minY = 0.0, maxY = 1.0;

	// cria todos os pontos
	std::vector<Point2> nodes{};
	nodes.reserve((gridX + 1) * (gridY + 1));
	for (int i = 0; i <= gridY; ++i)
	{

		double y = minY + (maxY - minY) / static_cast<double>(gridY) * static_cast<double>(i);

		for (int j = 0; j <= gridX; ++j)
		{
			double x = minX + (maxX - minX) / static_cast<double>(gridX) * static_cast<double>(j);
			nodes.emplace_back(x, y);
		}
	}

	// cria todos os elementos e faz uma lista ponto -> elementos
	std::vector<std::array<int, 4>> rectangles{};
	rectangles.reserve(gridX * gridY);
	for (int i = 0; i < gridY; ++i)
	{
		for (int j = 0; j < gridX; ++j)
		{

			// sentido anti-horário (começando do canto inferior esquerdo)
			rectangles.push_back({(i + 0) * (gridX + 1) + j + 0,
								  (i + 0) * (gridX + 1) + j + 1,
								  (i + 1) * (gridX + 1) + j + 1,
								  (i + 1) * (gridX + 1) + j + 0});
		}
	}

	// should return `true` if I would call apply_dirichlet on the node with given idx
	auto is_boundary_node = [gridX, gridY](size_t idx)
	{
		return (idx % (gridX + 1) == 0) ||	   // primeira coluna
			   (idx % (gridX + 1) == gridX) || // última coluna
			   (idx <= gridX) ||			   // primeira fileira
			   (idx >= gridY * (gridX + 1));   // última fileira
	};

	// std::vector<std::vector<double>> K(nodes.size(), std::vector<double>(nodes.size(), 0.0));
	CSR K{nodes.size(), nodes.size()};
	std::vector<double> F(nodes.size(), 0.0);

	std::cout << "Montando o sistema\n";

	/*	for (size_t i = 0; i < rectangles.size(); ++i) {
			auto rect = rectangles[i];

			Point2 &p1 = nodes[rect[0]];
			Point2 &p2 = nodes[rect[1]];
			Point2 &p3 = nodes[rect[2]];
			Point2 &p4 = nodes[rect[3]];

			for (int a = 1; a <= 4; ++a) {

				F[rect[a - 1]] += integrate2D([&](double xi, double eta)
											  {

					// pega as coordenadas no domínio físico
					auto [x, y] = base_transform(xi, eta, p1, p2, p3, p4);

					double val = 2.0 * std::numbers::pi * std::numbers::pi * sin(std::numbers::pi * x) * sin(std::numbers::pi * y); // f(x,y)
					auto J = transformation_jacobian(xi, eta, p1, p2, p3, p4);

					return N(a, xi, eta) * val * get_det(J); });

				for (int b = 1; b <= 4; ++b) {

					K.addElement(rect[a - 1], rect[b - 1], integrate2D([&](double xi, double eta) {
				//	K[rect[a - 1]][rect[b - 1]] += integrate2D([&](double xi, double eta) {

						auto grad_a = N_grad_physical(a, xi, eta, p1, p2, p3, p4);
						auto grad_b = N_grad_physical(b, xi, eta, p1, p2, p3, p4);

						auto jac = transformation_jacobian(xi, eta, p1, p2, p3, p4);

						return (grad_a[0] * grad_b[0] + grad_a[1] * grad_b[1]) * get_det(jac);
					}));
				}
			}
		}

		auto apply_dirichlet = [&](int node_idx) {
			for (size_t j = 0; j < nodes.size(); ++j)
			{
				if (K.getElement(j, node_idx) != 0.0) K.updateElement(j, node_idx, 0.0);
				if (K.getElement(node_idx, j) != 0.0) K.updateElement(node_idx, j, 0.0);
			//	K[j][node_idx] = 0.0;
			//	K[node_idx][j] = 0.0;
			}

			K.setElement(node_idx, node_idx, 1.0);
		//	K[node_idx][node_idx] = 1.0;
			F[node_idx] = 0.0;
		};

		for (int i = 0; i <= gridY; ++i)
		{

			// primeira coluna
			apply_dirichlet(i * (gridX + 1) + 0);

			// última coluna
			apply_dirichlet(i * (gridX + 1) + gridX);
		}

		for (int j = 0; j <= gridX; ++j)
		{

			// primeira fileira
			apply_dirichlet(0 * (gridX + 1) + j);

			// última fileira
			apply_dirichlet(gridY * (gridX + 1) + j);
		}*/

#pragma omp parallel
	{

		CSR K_thread{nodes.size(), nodes.size()};
		std::vector<double> F_thread(nodes.size(), 0.0);

#pragma omp for nowait
		for (size_t i = 0; i < rectangles.size(); ++i)
		{

			auto rect = rectangles[i];

			Point2 &p1 = nodes[rect[0]];
			Point2 &p2 = nodes[rect[1]];
			Point2 &p3 = nodes[rect[2]];
			Point2 &p4 = nodes[rect[3]];

			for (int a = 1; a <= 4; ++a)
			{

				if (is_boundary_node(rect[a - 1]))
				{
					// set diagonal to 1 and the rest or the row (including F_local) is already 0
					K_thread.addElement(rect[a - 1], rect[a - 1], 1.0);
					continue;
				}

				F_thread[rect[a - 1]] += integrate2D([&](double xi, double eta)
													 {

		    		// pega as coordenadas no domínio físico
					auto [x, y] = base_transform(xi, eta, p1, p2, p3, p4);
					
					double val = 2.0 * std::numbers::pi * std::numbers::pi * sin(std::numbers::pi * x) * sin(std::numbers::pi * y); // f(x,y)
					auto J = transformation_jacobian(xi, eta, p1, p2, p3, p4);

					return N(a, xi, eta) * val * get_det(J); });

				for (int b = 1; b <= 4; ++b)
				{

					if (is_boundary_node(rect[b - 1]))
					{
						// leave column as 0
						continue;
					}

					K_thread.addElement(rect[a - 1], rect[b - 1], integrate2D([&](double xi, double eta)
																			  {

		                auto grad_a = N_grad_physical(a, xi, eta, p1, p2, p3, p4);
		                auto grad_b = N_grad_physical(b, xi, eta, p1, p2, p3, p4);

		                auto jac = transformation_jacobian(xi, eta, p1, p2, p3, p4);

		                return (grad_a[0] * grad_b[0] + grad_a[1] * grad_b[1]) * get_det(jac); }));
				}
			}
		}

#pragma omp critical
		{
			for (size_t i = 0; i < nodes.size(); ++i)
			{
				if (is_boundary_node(i))
				{
					// set diagonal to 1 and the rest or the row (including F_local) is already 0
					K.setElement(i, i, 1.0);
					continue;
				}
				F[i] += F_thread[i];

				for (size_t j = 0; j < K_thread.getRowSize(i); ++j)
				{
					if (is_boundary_node(K_thread.getRowNthColumn(i, j)))
						continue;
					K.addElement(i, K_thread.getRowNthColumn(i, j), K_thread.getRowNthElement(i, j));
				}
			}
		}
	}

	std::cout << "Resolvendo o sistema\n";

	ConjugateGradient solver{};

	std::vector<double> coeffs(nodes.size(), 0.0);
	solver.solve(K, F, coeffs, 1000, 0.0);
	// ICCG(K, F, coeffs, 1000, 0.0);

	std::cout << "Calculando o erro\n";

	auto exact_sol = [](double x, double y)
	{
		return std::sin(std::numbers::pi * x) * std::sin(std::numbers::pi * y);
	};

	double L2 = 0.0;

	for (size_t i = 0; i < rectangles.size(); ++i)
	{
		auto rect = rectangles[i];

		Point2 &p1 = nodes[rect[0]];
		Point2 &p2 = nodes[rect[1]];
		Point2 &p3 = nodes[rect[2]];
		Point2 &p4 = nodes[rect[3]];

		double local_integral = integrate2D([&](double xi, double eta)
											{
												auto [x, y] = base_transform(xi, eta, p1, p2, p3, p4);

												// solução exata
												double u_exact = exact_sol(x, y);

												double uh = 0.0;
												uh += coeffs[rect[0]] * N(1, xi, eta);
												uh += coeffs[rect[1]] * N(2, xi, eta);
												uh += coeffs[rect[2]] * N(3, xi, eta);
												uh += coeffs[rect[3]] * N(4, xi, eta);

												double diff = uh - u_exact;

												auto J = transformation_jacobian(xi, eta, p1, p2, p3, p4);

												return diff * diff * get_det(J); });

		L2 += local_integral;
	}

	std::cout << "L2 error = " << std::sqrt(L2) << "\n";

	Graph3D graph(1000, 800);

	auto func = [&](double x, double y)
	{
		return evaluate_sol(x, y, coeffs, rectangles, nodes);
	};

	graph.setFunction(func);
	graph.waitFinish();

	return 0;
}
