#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Gaussian function
double gaus_pdf(double x, double b, double c) {
    return std::exp(-0.5 * std::pow((x - b) / c, 2)) / (c * std::sqrt(2 * M_PI));
}

// Quad Gaussian PDF for z-direction
double quad_gaus_pdf(double z, double b1, double c1, double a2, double b2, double c2,
                     double a3, double b3, double c3, double a4, double b4, double c4) {
    double gaus1 = gaus_pdf(z, b1, c1);
    double gaus2 = a2 * gaus_pdf(z, b2, c2);
    double gaus3 = a3 * gaus_pdf(z, b3, c3);
    double gaus4 = a4 * gaus_pdf(z, b4, c4);
    return (gaus1 + gaus2 + gaus3 + gaus4) / (1 + a2 + a3 + a4);
}

// Function to calculate modified sigma
double calculate_sigma(double sigma, double z, double beta_star, double angle_x, double angle_y) {
    if (beta_star != 0) {
        double distance_to_IP = z * std::sqrt(1 + std::tan(angle_x) * std::tan(angle_x) + std::tan(angle_y) * std::tan(angle_y));
        return sigma * std::sqrt(1 + (distance_to_IP * distance_to_IP) / (beta_star * 1e4 * beta_star * 1e4));
    }
    return sigma;
}

// Density calculation
py::array_t<double> density(
    py::array_t<double> x, py::array_t<double> y, py::array_t<double> z,
    double r_x, double r_y, double r_z, double sigma_x, double sigma_y,
    double angle_x, double angle_y, double beta_star_x, double beta_star_y,
    double beta_star_shift_x, double beta_star_shift_y,
    // Parameters for quad_gaus_pdf
    double b1, double c1, double a2, double b2, double c2,
    double a3, double b3, double c3, double a4, double b4, double c4
) {
    // Get the buffer info for the inputs
    auto buf_x = x.unchecked<3>();
    auto buf_y = y.unchecked<3>();
    auto buf_z = z.unchecked<3>();

    // Create an array to store the result, using the same shape as the input arrays
    std::vector<py::ssize_t> shape = { buf_x.shape(0), buf_x.shape(1), buf_x.shape(2) };
    py::array_t<double> result(shape);
    auto buf_result = result.mutable_unchecked<3>();

    // Precompute the cosine and sine of the rotation angles
    double cos_angle_xz = std::cos(angle_x);
    double sin_angle_xz = std::sin(angle_x);
    double cos_angle_yz = std::cos(angle_y);
    double sin_angle_yz = std::sin(angle_y);

    // Iterate over the grid points
    for (py::ssize_t i = 0; i < buf_x.shape(0); i++) {
        for (py::ssize_t j = 0; j < buf_x.shape(1); j++) {
            for (py::ssize_t k = 0; k < buf_x.shape(2); k++) {
                // Calculate the relative position vector
                double x_rel = buf_x(i, j, k) - r_x;
                double y_rel = buf_y(i, j, k) - r_y;
                double z_rel = buf_z(i, j, k) - r_z;

                // Apply rotation for the xz plane (rotation around the y-axis)
                double x_rot = x_rel * cos_angle_xz - z_rel * sin_angle_xz;
                double z_rot_xz = x_rel * sin_angle_xz + z_rel * cos_angle_xz;

                // Apply rotation for the yz plane (rotation around the x-axis)
                double y_rot = y_rel * cos_angle_yz - z_rot_xz * sin_angle_yz;
                double z_rot_yz = y_rel * sin_angle_yz + z_rot_xz * cos_angle_yz;

                // Compute the modified sigmas
                double sigma_x_mod = calculate_sigma(sigma_x, buf_z(i, j, k) - beta_star_shift_x, beta_star_x, angle_x, angle_y);
                double sigma_y_mod = calculate_sigma(sigma_y, buf_z(i, j, k) - beta_star_shift_y, beta_star_y, angle_x, angle_y);

                // Calculate the density using the quad Gaussian in the z-direction
                double z_density = quad_gaus_pdf(z_rot_yz, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4);

                // Calculate the exponent for the Gaussian distribution in x and y directions
                double exponent = -0.5 * (x_rot * x_rot / (sigma_x_mod * sigma_x_mod) +
                                          y_rot * y_rot / (sigma_y_mod * sigma_y_mod));

                // Compute the overall density and store it in the result array
                buf_result(i, j, k) = std::exp(exponent) * z_density / (std::pow(2 * M_PI, 1.0) * sigma_x_mod * sigma_y_mod);
            }
        }
    }

    return result;
}

// Generalized N-Gaussian PDF
double n_gaus_pdf(double z, const std::vector<std::array<double, 3>>& gaussians) {
    if (gaussians.empty()) return 0.0;

    double total = 0.0;
    double norm = 0.0;

    for (const auto& arr : gaussians) {
        double a = arr[0], b = arr[1], c = arr[2];
        total += a * gaus_pdf(z, b, c);
        norm += a;
    }

    return total / norm;
}

// Density calculation with arbitrary number of Gaussians
py::array_t<double> density_n_gaussians(
    py::array_t<double> x, py::array_t<double> y, py::array_t<double> z,
    double r_x, double r_y, double r_z, double sigma_x, double sigma_y,
    double angle_x, double angle_y, double beta_star_x, double beta_star_y,
    double beta_star_shift_x, double beta_star_shift_y,
    std::vector<std::array<double, 3>> gaussians_z  // <-- changed from tuple to array
) {
    auto buf_x = x.unchecked<3>();
    auto buf_y = y.unchecked<3>();
    auto buf_z = z.unchecked<3>();

    std::vector<py::ssize_t> shape = { buf_x.shape(0), buf_x.shape(1), buf_x.shape(2) };
    py::array_t<double> result(shape);
    auto buf_result = result.mutable_unchecked<3>();

    double cos_angle_xz = std::cos(angle_x);
    double sin_angle_xz = std::sin(angle_x);
    double cos_angle_yz = std::cos(angle_y);
    double sin_angle_yz = std::sin(angle_y);

    for (py::ssize_t i = 0; i < buf_x.shape(0); i++) {
        for (py::ssize_t j = 0; j < buf_x.shape(1); j++) {
            for (py::ssize_t k = 0; k < buf_x.shape(2); k++) {
                double x_rel = buf_x(i, j, k) - r_x;
                double y_rel = buf_y(i, j, k) - r_y;
                double z_rel = buf_z(i, j, k) - r_z;

                double x_rot = x_rel * cos_angle_xz - z_rel * sin_angle_xz;
                double z_rot_xz = x_rel * sin_angle_xz + z_rel * cos_angle_xz;
                double y_rot = y_rel * cos_angle_yz - z_rot_xz * sin_angle_yz;
                double z_rot_yz = y_rel * sin_angle_yz + z_rot_xz * cos_angle_yz;

                double sigma_x_mod = calculate_sigma(sigma_x, buf_z(i, j, k) - beta_star_shift_x, beta_star_x, angle_x, angle_y);
                double sigma_y_mod = calculate_sigma(sigma_y, buf_z(i, j, k) - beta_star_shift_y, beta_star_y, angle_x, angle_y);

                double z_density = n_gaus_pdf(z_rot_yz, gaussians_z);

                double exponent = -0.5 * (x_rot * x_rot / (sigma_x_mod * sigma_x_mod) +
                                          y_rot * y_rot / (sigma_y_mod * sigma_y_mod));

                buf_result(i, j, k) = std::exp(exponent) * z_density / (std::pow(2 * M_PI, 1.0) * sigma_x_mod * sigma_y_mod);
            }
        }
    }

    return result;
}


// Linearly interpolated PDF using (z_vals, pdf_vals), assuming z_vals is equally spaced
double interp_pdf(double z, const std::vector<double>& z_vals, const std::vector<double>& pdf_vals) {
    if (z_vals.empty() || pdf_vals.empty() || z_vals.size() != pdf_vals.size())
        return 0.0;

    double z_min = z_vals.front();
    double z_max = z_vals.back();

    // Out-of-bounds
    if (z <= z_min || z >= z_max) return 0.0;

    double dz = z_vals[1] - z_vals[0];  // assumes at least two points and uniform spacing
    size_t idx = static_cast<size_t>((z - z_min) / dz);

    if (idx >= z_vals.size() - 1)
        return 0.0;  // safety check in case of numerical precision issues

    double z0 = z_vals[idx], z1 = z_vals[idx + 1];
    double p0 = pdf_vals[idx], p1 = pdf_vals[idx + 1];

    // Linear interpolation
    return p0 + (p1 - p0) * (z - z0) / (z1 - z0);
}


py::array_t<double> density_interpolated_pdf(
    py::array_t<double> x, py::array_t<double> y, py::array_t<double> z,
    double r_x, double r_y, double r_z, double sigma_x, double sigma_y,
    double angle_x, double angle_y, double beta_star_x, double beta_star_y,
    double beta_star_shift_x, double beta_star_shift_y,
    std::vector<double> z_vals, std::vector<double> pdf_vals
) {
    auto buf_x = x.unchecked<3>();
    auto buf_y = y.unchecked<3>();
    auto buf_z = z.unchecked<3>();

    std::vector<py::ssize_t> shape = { buf_x.shape(0), buf_x.shape(1), buf_x.shape(2) };
    py::array_t<double> result(shape);
    auto buf_result = result.mutable_unchecked<3>();

    double cos_angle_xz = std::cos(angle_x);
    double sin_angle_xz = std::sin(angle_x);
    double cos_angle_yz = std::cos(angle_y);
    double sin_angle_yz = std::sin(angle_y);

    const double two_pi = 2.0 * M_PI;  // Speed up slightly by avoiding repeated calculation
    const double norm_factor = 1.0 / two_pi;

    for (py::ssize_t i = 0; i < buf_x.shape(0); i++) {
        for (py::ssize_t j = 0; j < buf_x.shape(1); j++) {
            for (py::ssize_t k = 0; k < buf_x.shape(2); k++) {
                double x_rel = buf_x(i, j, k) - r_x;
                double y_rel = buf_y(i, j, k) - r_y;
                double z_rel = buf_z(i, j, k) - r_z;

                double x_rot = x_rel * cos_angle_xz - z_rel * sin_angle_xz;
                double z_rot_xz = x_rel * sin_angle_xz + z_rel * cos_angle_xz;
                double y_rot = y_rel * cos_angle_yz - z_rot_xz * sin_angle_yz;
                double z_rot_yz = y_rel * sin_angle_yz + z_rot_xz * cos_angle_yz;

                double sigma_x_mod = calculate_sigma(sigma_x, buf_z(i, j, k) - beta_star_shift_x, beta_star_x, angle_x, angle_y);
                double sigma_y_mod = calculate_sigma(sigma_y, buf_z(i, j, k) - beta_star_shift_y, beta_star_y, angle_x, angle_y);

                double z_density = interp_pdf(z_rot_yz, z_vals, pdf_vals);

                double exponent = -0.5 * (x_rot * x_rot / (sigma_x_mod * sigma_x_mod) +
                                          y_rot * y_rot / (sigma_y_mod * sigma_y_mod));

                buf_result(i, j, k) = std::exp(exponent) * z_density * norm_factor / (sigma_x_mod * sigma_y_mod);
            }
        }
    }

    return result;
}


PYBIND11_MODULE(_bunch_density_cpp, m) {
    m.def("density", &density, "Calculate the density of the bunch at given points in the lab frame");
    m.def("density_n_gaussians", &density_n_gaussians, "Calculate density with arbitrary number of Gaussians in the z-direction");
    m.def("density_interpolated_pdf", &density_interpolated_pdf, "Calculate density using interpolated PDF in z");
}