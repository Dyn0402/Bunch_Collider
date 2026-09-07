#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 23 10:57 2024
Created in PyCharm
Created as sphenix_polarimetry/vernier_z_vertex_fitting

@author: Dylan Neff, dn277127
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize, minimize_scalar
from scipy.optimize import curve_fit as cf
from scipy.interpolate import interp1d
import pandas as pd
from datetime import datetime, timedelta
# import tensorflow as tf

try:
    import winsound
    sound_on = True
except ImportError:
    print('No winsound module, no sound will play.')
    sound_on = False

import uproot
import awkward as ak
import vector

from BunchCollider import BunchCollider
from Measure import Measure


def main():
    vernier_scan_date = 'Aug12'
    # vernier_scan_date = 'Jul11'
    orientation = 'Horizontal'
    # orientation = 'Vertical'
    base_path = '/local/home/dn277127/Bureau/vernier_scan/'
    # base_path = '/home/dylan/Desktop/vernier_scan/'
    # base_path = 'C:/Users/Dylan/Desktop/vernier_scan/'
    dist_root_file_name = f'vernier_scan_{vernier_scan_date}_mbd_vertex_z_distributions.root'
    z_vertex_root_path = f'{base_path}vertex_data/{dist_root_file_name}'
    cad_measurement_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_combined.dat'
    # pdf_out_path = f'{base_path}/Analysis/sim_vs_mbd_cad_params_{vernier_scan_date}.pdf'
    pdf_out_path = f'{base_path}/Analysis/{orientation.lower()}/simple_bw_fitting_{vernier_scan_date}.pdf'
    longitudinal_fit_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_COLOR_longitudinal_fit.dat'
    # z_vertex_root_path = f'C:/Users/Dylan/Desktop/vernier_scan/vertex_data/{dist_root_file_name}'
    # fit_head_on(z_vertex_root_path)
    # fit_head_on_manual(z_vertex_root_path)
    # plot_head_on(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path)
    # head_on_metric_sensitivity(base_path, z_vertex_root_path, longitudinal_fit_path)
    # plot_peripheral(z_vertex_root_path, longitudinal_fit_path)
    # fit_peripheral(z_vertex_root_path)
    # fit_peripheral_scipy(z_vertex_root_path, longitudinal_fit_path)
    # peripheral_metric_test(z_vertex_root_path)
    # peripheral_metric_sensitivity(base_path, z_vertex_root_path)
    # check_head_on_dependences(z_vertex_root_path)
    plot_head_on_and_peripheral(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path)
    # fit_angles_head_on_and_peripheral(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path)
    # plot_all_z_vertex_hists(z_vertex_root_path)
    # sim_cad_params(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, pdf_out_path)
    # sim_fit_cad_params(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, pdf_out_path)
    # perform_and_compare_vernier_scan(z_vertex_root_path, longitudinal_fit_path)
    # calc_vernier_scan_bw_residuals(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, pdf_out_path,
    #                                orientation)
    # avg_over_bunch_test(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path)
    # fit_peripheral_with_neural_net(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path)

    print('donzo')


def fit_peripheral_with_neural_net(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path):
    """
    Use neural network to fit the peripheral z-vertex distributions.
    """
    cad_data = read_cad_measurement_file(cad_measurement_path)
    cw_rates = get_cw_rates(cad_data)

    orientation = 'Horizontal'
    peripheral_scan_step = 6

    pe_step_cad_data = cad_data[(cad_data['orientation'] == orientation) & (cad_data['step'] == peripheral_scan_step)].iloc[0]

    model = tf.keras.models.load_model(f'/local/home/dn277127/Bureau/vernier_scan/training_data/'
                                       f'simple_par_training_set_1/neural_network_model.keras')

    # Important parameters
    bw_nom = 155
    beta_star_nom = 85.
    mbd_online_resolution = 2.0  # cm MBD resolution on trigger level
    bkg = 0.4e-17  # Background level
    gaus_eff_width = None
    pe_blue_angle_x, pe_yellow_angle_x = -pe_step_cad_data['bh8_avg'] / 1e3, -pe_step_cad_data['yh8_avg'] / 1e3  # mrad to rad
    pe_blue_angle_y, pe_yellow_angle_y = 0.0, 0.0

    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, first_dist=False, norms=cw_rates, abs_norm=True)
    pe_hist_data = [hist for hist in z_vertex_hists if hist['scan_axis'] == orientation and hist['scan_step'] == peripheral_scan_step][0]

    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(pe_blue_angle_x, pe_blue_angle_y, pe_yellow_angle_x, pe_yellow_angle_y)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gaus_eff_width)
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)
    collider_sim.set_amplitude(1.2048100105307071e+27)

    offset = pe_step_cad_data['offset_set_val'] * 1e3  # mm to um
    if orientation == 'Horizontal':
        collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
    elif orientation == 'Vertical':
        collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

    collider_sim.run_sim_parallel()
    fit_shift(collider_sim, pe_hist_data['counts'], pe_hist_data['centers'])
    pe_zs_shifted, pe_z_dist_shifted = collider_sim.get_z_density_dist()

    fig_pe, ax_pe = plt.subplots()
    ax_pe.bar(pe_hist_data['centers'], pe_hist_data['counts'],
                width=pe_hist_data['centers'][1] - pe_hist_data['centers'][0], label='MBD Vertex')
    ax_pe.plot(pe_zs_shifted, pe_z_dist_shifted, color='red', alpha=1.0, label='Simulation Shifted')

    ax_pe.set_title(f'Peripheral bw {bw_nom:.0f} {orientation} Scan Step {peripheral_scan_step}')
    ax_pe.set_xlabel('z Vertex Position (cm)')
    ax_pe.legend(loc='upper right')
    fig_pe.tight_layout()

    plt.show()



def avg_over_bunch_test(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path):
    """
    Test averaging over the bunch length in the longitudinal fit.
    """
    cad_data = read_cad_measurement_file(cad_measurement_path)
    cw_rates = get_cw_rates(cad_data)
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, first_dist=False, norms=cw_rates, abs_norm=True)

    orientation = 'Vertical'
    scan_step = 1
    hist_data = [hist for hist in z_vertex_hists if hist['scan_axis'] == orientation and hist['scan_step'] == scan_step][0]
    cad_step_data = cad_data[(cad_data['orientation'] == orientation) & (cad_data['step'] == scan_step)].iloc[0]

    longitudinal_fit_dir = '/'.join(longitudinal_fit_path.split('/')[:-1]) + '/Individual_Longitudinal_Bunch_Profile_Fits/'
    fit_path = f'{longitudinal_fit_dir}VernierScan_Aug12_COLOR_bunch_BUNCHNUM_longitudinal_fit.dat'

    blue_fit_path = fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = fit_path.replace('_COLOR_', '_yellow_')

    # Important parameters
    bw_nom = 155
    beta_star_nom = 85.
    mbd_online_resolution = 2.0  # cm MBD resolution on trigger level
    bkg = 0.4e-17  # Background level
    gaus_eff_width = None
    offsets = None
    blue_angle_x, yellow_angle_x = -cad_step_data['bh8_avg'] / 1e3, -cad_step_data['yh8_avg'] / 1e3  # mrad to rad
    blue_angle_y, yellow_angle_y = 0.0, 0.0

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gaus_eff_width)
    collider_sim.set_amplitude(1.2048100105307071e+27)

    z_dists = []
    for bunch_num in range(1, 110):
        print(f'Fitting Bunch {bunch_num}')
        blue_fit_path_i = blue_fit_path.replace('BUNCHNUM', str(bunch_num))
        yellow_fit_path_i = yellow_fit_path.replace('BUNCHNUM', str(bunch_num))
        collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path_i, yellow_fit_path_i)
        # collider_sim.bunch1.set_delay(0.02 * ((bunch_num % 2) - 0.5) * 2)
        blue_angle_xi, blue_angle_yi = np.random.normal(blue_angle_x, 0.05e-3), np.random.normal(blue_angle_y, 0.05e-3)
        yellow_angle_xi, yellow_angle_yi = np.random.normal(yellow_angle_x, 0.05e-3), np.random.normal(yellow_angle_y, 0.05e-3)
        collider_sim.set_bunch_crossing(blue_angle_xi, blue_angle_yi, yellow_angle_xi, yellow_angle_yi)
        blue_offset_x, blue_offset_y = np.random.normal(0., 10), np.random.normal(0., 10)
        yellow_offset_x, yellow_offset_y = np.random.normal(0., 10), np.random.normal(0., 10)
        collider_sim.set_bunch_offsets(np.array([blue_offset_x, blue_offset_y]), np.array([yellow_offset_x, yellow_offset_y]))

        collider_sim.run_sim_parallel()
        zs, z_dist = collider_sim.get_z_density_dist()
        z_dists.append(z_dist)

    z_dist_avg = np.mean(z_dists, axis=0)

    fig, ax = plt.subplots()
    ax.bar(hist_data['centers'], hist_data['counts'], width=hist_data['centers'][1] - hist_data['centers'][0], label='MBD Vertex')
    for z_dist in z_dists:
        ax.plot(zs, z_dist, color='gray', alpha=0.5, zorder=1)
    ax.plot(zs, z_dist_avg, color='r', label='Average', zorder=2)
    ax.set_title(f'bw {bw_nom:.0f} {orientation} Scan Step {scan_step}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.legend(loc='upper right')
    fig.tight_layout()

    plt.show()


def plot_head_on_and_peripheral(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path):
    cad_data = read_cad_measurement_file(cad_measurement_path)
    cw_rates = get_cw_rates(cad_data)

    orientation = 'Horizontal'
    # orientation = 'Vertical'
    head_on_scan_step = 1
    # peripheral_scan_step = 12
    peripheral_scan_step = 5

    ho_step_cad_data = cad_data[(cad_data['orientation'] == orientation) & (cad_data['step'] == head_on_scan_step)].iloc[0]
    pe_step_cad_data = cad_data[(cad_data['orientation'] == orientation) & (cad_data['step'] == peripheral_scan_step)].iloc[0]

    # Important parameters
    bw_nom = 162
    beta_star_nom = 85.
    mbd_online_resolution = 2.0  # cm MBD resolution on trigger level
    bkg = 0.4e-17 * 0  # Background level
    gaus_eff_width = 500
    ho_offsets = None
    pe_offsets = None
    ho_blue_angle_x, ho_yellow_angle_x = -ho_step_cad_data['bh8_avg'] / 1e3, -ho_step_cad_data['yh8_avg'] / 1e3  # mrad to rad
    ho_blue_angle_y, ho_yellow_angle_y = 0.0, 0.0
    pe_blue_angle_x, pe_yellow_angle_x = -pe_step_cad_data['bh8_avg'] / 1e3, -pe_step_cad_data['yh8_avg'] / 1e3  # mrad to rad
    # pe_blue_angle_x, pe_yellow_angle_x = -pe_step_cad_data['bh8_avg'] / 1e3, -0.063e-3  # mrad to rad
    pe_blue_angle_y, pe_yellow_angle_y = 0.0, 0.0
    # pe_blue_angle_y, pe_yellow_angle_y = -0.1e-3, +0.1e-3

    new_bw_x, new_bw_y = 164.0, 159.5
    # new_beta_star = 100.
    # new_beta_star = [97., 82., 88., 95.]
    new_beta_star = np.array([97., 82., 88., 95.]) * 1.1
    print(f'New Beta Star: {new_beta_star}')
    new_mbd_res = 2.0
    # new_bkg = 0.4e-17
    new_bkg = 0.4e-16 * 0
    new_additional_length_scaling = 1.0
    new_gaus_eff_width = 500  # cm
    new_ho_offsets = [[0.0, 0.0], [0.0, 0.0]]
    # new_pe_offsets = [[-900.0, 0.0], [0.0, 0.0]]
    new_pe_offsets = None
    new_ho_blue_angle_x, new_ho_yellow_angle_x = -ho_step_cad_data['bh8_avg'] / 1e3, -ho_step_cad_data['yh8_avg'] / 1e3
    new_ho_blue_angle_y, new_ho_yellow_angle_y = 0.0, 0.0
    # new_pe_blue_angle_x, new_pe_yellow_angle_x = -pe_step_cad_data['bh8_avg'] / 1e3, -0.09e-3
    new_pe_blue_angle_x, new_pe_yellow_angle_x = -pe_step_cad_data['bh8_avg'] / 1e3, -pe_step_cad_data['yh8_avg'] / 1e3
    new_pe_yellow_angle_x += 0.05e-3
    # new_pe_blue_angle_x, new_pe_yellow_angle_x = -pe_step_cad_data['bh8_avg'] / 1e3, -0.09e-3
    # new_pe_blue_angle_x, new_pe_yellow_angle_x = -pe_step_cad_data['bh8_avg'] / 1e3, -0.07e-3
    new_pe_blue_angle_y, new_pe_yellow_angle_y = 0.0e-3, -0.01e-3

    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, first_dist=False, norms=cw_rates, abs_norm=True)
    ho_hist_data = [hist for hist in z_vertex_hists if hist['scan_axis'] == orientation and hist['scan_step'] == head_on_scan_step][0]

    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(ho_blue_angle_x, ho_blue_angle_y, ho_yellow_angle_x, ho_yellow_angle_y)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gaus_eff_width)
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    blue_bunch_len_scaling = ho_step_cad_data['blue_bunch_length']
    yellow_bunch_len_scaling = ho_step_cad_data['yellow_bunch_length']
    collider_sim.set_longitudinal_fit_scaling(blue_bunch_len_scaling, yellow_bunch_len_scaling)

    if ho_offsets is not None:
        collider_sim.set_bunch_offsets(ho_offsets[0], ho_offsets[1])
    else:
        offset = ho_step_cad_data['offset_set_val'] * 1e3  # mm to um
        if orientation == 'Horizontal':
            collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
        elif orientation == 'Vertical':
            collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

    collider_sim.run_sim_parallel()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    scale = max(ho_hist_data['counts']) / max(z_dist_og)
    z_max_sim_og = zs_og[np.argmax(z_dist_og)]
    z_max_hist_og = ho_hist_data['centers'][np.argmax(ho_hist_data['counts'])]
    print(f'z_max_sim: {z_max_sim_og}, z_max_hist: {z_max_hist_og}')
    shift = z_max_sim_og - z_max_hist_og  # cm
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, ho_hist_data['counts'], ho_hist_data['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    print(f'scale: {scale}, shift: {shift}')

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    ho_zs_og, ho_z_dist_og = collider_sim.get_z_density_dist()

    # Calculate area under mbd distribution and simulation distribution
    mbd_area = np.sum(ho_hist_data['counts'])
    sim_interp = interp1d(ho_zs_og, ho_z_dist_og)
    sim_area = np.sum(sim_interp(ho_hist_data['centers']))
    print(f'MBD Area: {mbd_area}, Sim Area: {sim_area}')
    print(f'Area Ratio: {mbd_area / sim_area}')

    # Set and run new head on collider sim
    collider_sim.set_bunch_sigmas(np.array([new_bw_x, new_bw_y]), np.array([new_bw_x, new_bw_y]))
    # collider_sim.set_bunch_beta_stars(new_beta_star, new_beta_star)
    collider_sim.set_bunch_beta_stars(*new_beta_star)
    collider_sim.set_gaus_smearing_sigma(new_mbd_res)
    collider_sim.set_bkg(new_bkg)
    collider_sim.set_bunch_crossing(new_ho_blue_angle_x, new_ho_blue_angle_y, new_ho_yellow_angle_x, new_ho_yellow_angle_y)
    collider_sim.set_gaus_z_efficiency_width(new_gaus_eff_width)
    collider_sim.set_longitudinal_fit_scaling(blue_bunch_len_scaling * new_additional_length_scaling,
                                              yellow_bunch_len_scaling * new_additional_length_scaling)

    if new_ho_offsets is not None:
        collider_sim.set_bunch_offsets(new_ho_offsets[0], new_ho_offsets[1])

    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)

    collider_sim.run_sim_parallel()
    zs_new, z_dist_new = collider_sim.get_z_density_dist()

    scale = max(ho_hist_data['counts']) / max(z_dist_new)
    z_max_sim_new = zs_new[np.argmax(z_dist_new)]
    z_max_hist_new = ho_hist_data['centers'][np.argmax(ho_hist_data['counts'])]
    print(f'z_max_sim: {z_max_sim_new}, z_max_hist: {z_max_hist_new}')
    shift = z_max_sim_new - z_max_hist_new  # cm
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                     args=(collider_sim, scale, shift, ho_hist_data['counts'], ho_hist_data['centers']),
                     bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    ho_zs_new, ho_z_dist_new = collider_sim.get_z_density_dist()

    collider_params = collider_sim.get_param_string()

    fig_ho, ax_ho = plt.subplots()
    ax_ho.bar(ho_hist_data['centers'], ho_hist_data['counts'],
                width=ho_hist_data['centers'][1] - ho_hist_data['centers'][0], label='MBD Vertex')
    ax_ho.plot(ho_zs_og, ho_z_dist_og, color='gray', alpha=0.8, label='Simulation Guess')
    ax_ho.plot(ho_zs_new, ho_z_dist_new, color='red', label='Simulation Fit')
    ax_ho.annotate(collider_params, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, va='top', ha='left',
                   bbox=dict(facecolor='white', alpha=0.8))
    ax_ho.set_title(f'Head On bw {bw_nom:.0f} {orientation} Scan Step {head_on_scan_step}')
    ax_ho.set_xlabel('z Vertex Position (cm)')
    ax_ho.legend(loc='upper right')
    fig_ho.tight_layout()

    # # Plot x and y density distributions
    # x_vals, x_dist = collider_sim.get_x_density_dist()
    # fig_x_dist, ax_x_dist = plt.subplots()
    # ax_x_dist.plot(x_vals, x_dist, label='x Density')
    # ax_x_dist.set_title('x Density Distribution')
    # ax_x_dist.set_xlabel('x Position (um)')
    # ax_x_dist.axhline(0, color='black', linewidth=0.5)
    # ax_x_dist.set_ylim(bottom=0)
    # fig_x_dist.tight_layout()
    #
    # y_vals, y_dist = collider_sim.get_y_density_dist()
    # fig_y_dist, ax_y_dist = plt.subplots()
    # ax_y_dist.plot(y_vals, y_dist, label='y Density')
    # ax_y_dist.set_title('y Density Distribution')
    # ax_y_dist.set_xlabel('y Position (um)')
    # ax_y_dist.axhline(0, color='black', linewidth=0.5)
    # ax_y_dist.set_ylim(bottom=0)
    # fig_y_dist.tight_layout()

    pe_hist_data = [hist for hist in z_vertex_hists if hist['scan_axis'] == orientation and hist['scan_step'] == peripheral_scan_step][0]

    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(pe_blue_angle_x, pe_blue_angle_y, pe_yellow_angle_x, pe_yellow_angle_y)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gaus_eff_width)

    blue_bunch_len_scaling = pe_step_cad_data['blue_bunch_length']
    yellow_bunch_len_scaling = pe_step_cad_data['yellow_bunch_length']
    collider_sim.set_longitudinal_fit_scaling(blue_bunch_len_scaling, yellow_bunch_len_scaling)

    if pe_offsets is not None:
        collider_sim.set_bunch_offsets(pe_offsets[0], pe_offsets[1])
    else:
        offset = pe_step_cad_data['offset_set_val'] * 1e3
        if orientation == 'Horizontal':
            collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
        elif orientation == 'Vertical':
            collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

    collider_sim.run_sim_parallel()
    fit_shift(collider_sim, pe_hist_data['counts'], pe_hist_data['centers'])
    pe_zs_og, pe_z_dist_og = collider_sim.get_z_density_dist()

    # Set and run new peripheral collider sim
    collider_sim.set_bunch_sigmas(np.array([new_bw_x, new_bw_y]), np.array([new_bw_x, new_bw_y]))
    # collider_sim.set_bunch_beta_stars(new_beta_star, new_beta_star)
    collider_sim.set_bunch_beta_stars(*new_beta_star)
    collider_sim.set_gaus_smearing_sigma(new_mbd_res)
    collider_sim.set_bkg(new_bkg)
    collider_sim.set_bunch_crossing(new_pe_blue_angle_x, new_pe_blue_angle_y, new_pe_yellow_angle_x, new_pe_yellow_angle_y)
    collider_sim.set_gaus_z_efficiency_width(new_gaus_eff_width)

    if new_pe_offsets is not None:
        collider_sim.set_bunch_offsets(new_pe_offsets[0], new_pe_offsets[1])

    collider_sim.run_sim_parallel()
    fit_shift(collider_sim, pe_hist_data['counts'], pe_hist_data['centers'])
    pe_zs_new, pe_z_dist_new = collider_sim.get_z_density_dist()

    pe_params = collider_sim.get_param_string()

    fig_pe, ax_pe = plt.subplots()
    ax_pe.bar(pe_hist_data['centers'], pe_hist_data['counts'],
                width=pe_hist_data['centers'][1] - pe_hist_data['centers'][0], label='MBD Vertex')
    ax_pe.plot(pe_zs_og, pe_z_dist_og, color='gray', alpha=0.8, label='Simulation Guess')
    ax_pe.plot(pe_zs_new, pe_z_dist_new, color='red', label='Simulation Fit')
    ax_pe.annotate(pe_params, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, va='top', ha='left',
                   bbox=dict(facecolor='white', alpha=0.8))
    ax_pe.set_title(f'Peripheral bw {bw_nom:.0f} {orientation} Scan Step {peripheral_scan_step}')
    ax_pe.set_xlabel('z Vertex Position (cm)')
    ax_pe.legend(loc='upper right')
    fig_pe.tight_layout()

    plt.show()


def fit_angles_head_on_and_peripheral(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path):
    cad_data = read_cad_measurement_file(cad_measurement_path)
    cw_rates = get_cw_rates(cad_data)

    # orientation = 'Horizontal'
    orientation = 'Vertical'
    head_on_scan_step = 1
    # peripheral_scan_step = 12
    peripheral_scan_step = 5

    ho_step_cad_data = cad_data[(cad_data['orientation'] == orientation) & (cad_data['step'] == head_on_scan_step)].iloc[0]
    pe_step_cad_data = cad_data[(cad_data['orientation'] == orientation) & (cad_data['step'] == peripheral_scan_step)].iloc[0]

    # Important parameters
    bw_nom = 151
    beta_star_nom = 85.
    mbd_online_resolution = 2.0  # cm MBD resolution on trigger level
    bkg = 0.4e-17  # Background level
    gaus_eff_width = None
    ho_offsets = None
    pe_offsets = None
    ho_blue_angle_x, ho_yellow_angle_x = -ho_step_cad_data['bh8_avg'] / 1e3, -ho_step_cad_data['yh8_avg'] / 1e3  # mrad to rad
    ho_blue_angle_y, ho_yellow_angle_y = 0.0, 0.0
    pe_blue_angle_x, pe_yellow_angle_x = -pe_step_cad_data['bh8_avg'] / 1e3, -pe_step_cad_data['yh8_avg'] / 1e3  # mrad to rad
    pe_blue_angle_y, pe_yellow_angle_y = 0.0, 0.0

    new_bw_x, new_bw_y = 164.0, 159.5
    new_beta_star = 100.
    new_mbd_res = 2.0
    new_bkg = 0.4e-16 * 0
    new_additional_length_scaling = 1.0
    new_gaus_eff_width = 500  # cm
    new_ho_offsets = [[0.0, 0.0], [0.0, 0.0]]
    # new_pe_offsets = [[-900.0, 0.0], [0.0, 0.0]]
    new_pe_offsets = None
    new_ho_blue_angle_x, new_ho_yellow_angle_x = -ho_step_cad_data['bh8_avg'] / 1e3, -ho_step_cad_data['yh8_avg'] / 1e3
    new_ho_blue_angle_y, new_ho_yellow_angle_y = 0.0, 0.0
    # new_pe_blue_angle_x, new_pe_yellow_angle_x = -pe_step_cad_data['bh8_avg'] / 1e3, -0.07e-3
    new_pe_blue_angle_x, new_pe_yellow_angle_x = -pe_step_cad_data['bh8_avg'] / 1e3, -pe_step_cad_data['yh8_avg'] / 1e3
    # new_pe_blue_angle_x, new_pe_yellow_angle_x = -pe_step_cad_data['bh8_avg'] / 1e3, -0.09e-3
    # new_pe_blue_angle_x, new_pe_yellow_angle_x = -pe_step_cad_data['bh8_avg'] / 1e3, -0.07e-3
    new_pe_blue_angle_y, new_pe_yellow_angle_y = -0.4e-3, +0.4e-3

    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, first_dist=False, norms=cw_rates, abs_norm=True)
    ho_hist_data = [hist for hist in z_vertex_hists if hist['scan_axis'] == orientation and hist['scan_step'] == head_on_scan_step][0]

    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(ho_blue_angle_x, ho_blue_angle_y, ho_yellow_angle_x, ho_yellow_angle_y)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gaus_eff_width)
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    blue_bunch_len_scaling = ho_step_cad_data['blue_bunch_length']
    yellow_bunch_len_scaling = ho_step_cad_data['yellow_bunch_length']
    collider_sim.set_longitudinal_fit_scaling(blue_bunch_len_scaling, yellow_bunch_len_scaling)

    if ho_offsets is not None:
        collider_sim.set_bunch_offsets(ho_offsets[0], ho_offsets[1])
    else:
        offset = ho_step_cad_data['offset_set_val'] * 1e3  # mm to um
        if orientation == 'Horizontal':
            collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
        elif orientation == 'Vertical':
            collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

    collider_sim.run_sim_parallel()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    scale = max(ho_hist_data['counts']) / max(z_dist_og)
    z_max_sim_og = zs_og[np.argmax(z_dist_og)]
    z_max_hist_og = ho_hist_data['centers'][np.argmax(ho_hist_data['counts'])]
    print(f'z_max_sim: {z_max_sim_og}, z_max_hist: {z_max_hist_og}')
    shift = z_max_sim_og - z_max_hist_og  # cm
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, ho_hist_data['counts'], ho_hist_data['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    print(f'scale: {scale}, shift: {shift}')

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    ho_zs_og, ho_z_dist_og = collider_sim.get_z_density_dist()

    # Calculate area under mbd distribution and simulation distribution
    mbd_area = np.sum(ho_hist_data['counts'])
    sim_interp = interp1d(ho_zs_og, ho_z_dist_og)
    sim_area = np.sum(sim_interp(ho_hist_data['centers']))
    print(f'MBD Area: {mbd_area}, Sim Area: {sim_area}')
    print(f'Area Ratio: {mbd_area / sim_area}')

    # Set and run new head on collider sim
    collider_sim.set_bunch_sigmas(np.array([new_bw_x, new_bw_y]), np.array([new_bw_x, new_bw_y]))
    collider_sim.set_bunch_beta_stars(new_beta_star, new_beta_star)
    collider_sim.set_gaus_smearing_sigma(new_mbd_res)
    collider_sim.set_bkg(new_bkg)
    collider_sim.set_bunch_crossing(new_ho_blue_angle_x, new_ho_blue_angle_y, new_ho_yellow_angle_x, new_ho_yellow_angle_y)
    collider_sim.set_gaus_z_efficiency_width(new_gaus_eff_width)
    collider_sim.set_longitudinal_fit_scaling(blue_bunch_len_scaling * new_additional_length_scaling,
                                              yellow_bunch_len_scaling * new_additional_length_scaling)

    if new_ho_offsets is not None:
        collider_sim.set_bunch_offsets(new_ho_offsets[0], new_ho_offsets[1])

    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)

    collider_sim.run_sim_parallel()
    zs_new, z_dist_new = collider_sim.get_z_density_dist()

    scale = max(ho_hist_data['counts']) / max(z_dist_new)
    z_max_sim_new = zs_new[np.argmax(z_dist_new)]
    z_max_hist_new = ho_hist_data['centers'][np.argmax(ho_hist_data['counts'])]
    print(f'z_max_sim: {z_max_sim_new}, z_max_hist: {z_max_hist_new}')
    shift = z_max_sim_new - z_max_hist_new  # cm
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                     args=(collider_sim, scale, shift, ho_hist_data['counts'], ho_hist_data['centers']),
                     bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    ho_zs_new, ho_z_dist_new = collider_sim.get_z_density_dist()

    collider_params = collider_sim.get_param_string()

    fig_ho, ax_ho = plt.subplots()
    ax_ho.bar(ho_hist_data['centers'], ho_hist_data['counts'],
                width=ho_hist_data['centers'][1] - ho_hist_data['centers'][0], label='MBD Vertex')
    ax_ho.plot(ho_zs_og, ho_z_dist_og, color='gray', alpha=0.8, label='Simulation Guess')
    ax_ho.plot(ho_zs_new, ho_z_dist_new, color='red', label='Simulation Fit')
    ax_ho.annotate(collider_params, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, va='top', ha='left',
                   bbox=dict(facecolor='white', alpha=0.8))
    ax_ho.set_title(f'Head On bw {bw_nom:.0f} {orientation} Scan Step {head_on_scan_step}')
    ax_ho.set_xlabel('z Vertex Position (cm)')
    ax_ho.legend(loc='upper right')
    fig_ho.tight_layout()

    # # Plot x and y density distributions
    # x_vals, x_dist = collider_sim.get_x_density_dist()
    # fig_x_dist, ax_x_dist = plt.subplots()
    # ax_x_dist.plot(x_vals, x_dist, label='x Density')
    # ax_x_dist.set_title('x Density Distribution')
    # ax_x_dist.set_xlabel('x Position (um)')
    # ax_x_dist.axhline(0, color='black', linewidth=0.5)
    # ax_x_dist.set_ylim(bottom=0)
    # fig_x_dist.tight_layout()
    #
    # y_vals, y_dist = collider_sim.get_y_density_dist()
    # fig_y_dist, ax_y_dist = plt.subplots()
    # ax_y_dist.plot(y_vals, y_dist, label='y Density')
    # ax_y_dist.set_title('y Density Distribution')
    # ax_y_dist.set_xlabel('y Position (um)')
    # ax_y_dist.axhline(0, color='black', linewidth=0.5)
    # ax_y_dist.set_ylim(bottom=0)
    # fig_y_dist.tight_layout()

    pe_hist_data = [hist for hist in z_vertex_hists if hist['scan_axis'] == orientation and hist['scan_step'] == peripheral_scan_step][0]

    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(pe_blue_angle_x, pe_blue_angle_y, pe_yellow_angle_x, pe_yellow_angle_y)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gaus_eff_width)

    blue_bunch_len_scaling = pe_step_cad_data['blue_bunch_length']
    yellow_bunch_len_scaling = pe_step_cad_data['yellow_bunch_length']
    collider_sim.set_longitudinal_fit_scaling(blue_bunch_len_scaling, yellow_bunch_len_scaling)

    if pe_offsets is not None:
        collider_sim.set_bunch_offsets(pe_offsets[0], pe_offsets[1])
    else:
        offset = pe_step_cad_data['offset_set_val'] * 1e3
        if orientation == 'Horizontal':
            collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
        elif orientation == 'Vertical':
            collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

    collider_sim.run_sim_parallel()
    fit_shift(collider_sim, pe_hist_data['counts'], pe_hist_data['centers'])
    pe_zs_og, pe_z_dist_og = collider_sim.get_z_density_dist()

    # Use minimizer
    def objective_function_angle_y(angle_y, angle_yellow_x):
        """Objective function for a fixed angle_yellow_x, minimizing over angle_y."""
        # Set and run new peripheral collider sim
        collider_sim.set_bunch_sigmas(np.array([new_bw_x, new_bw_y]), np.array([new_bw_x, new_bw_y]))
        collider_sim.set_bunch_beta_stars(new_beta_star, new_beta_star)
        collider_sim.set_gaus_smearing_sigma(new_mbd_res)
        collider_sim.set_bkg(new_bkg)
        collider_sim.set_bunch_crossing(new_pe_blue_angle_x, -angle_y, angle_yellow_x, angle_y)
        collider_sim.set_gaus_z_efficiency_width(new_gaus_eff_width)

        if new_pe_offsets is not None:
            collider_sim.set_bunch_offsets(new_pe_offsets[0], new_pe_offsets[1])

        collider_sim.run_sim_parallel()
        fit_shift(collider_sim, pe_hist_data['counts'], pe_hist_data['centers'])
        pe_zs_new, pe_z_dist_new = collider_sim.get_z_density_dist()
        sim_interp_vals = interp1d(pe_zs_new, pe_z_dist_new)(pe_hist_data['centers'])
        residual = np.sqrt(np.sum((pe_hist_data['counts'] - sim_interp_vals) ** 2)) / np.mean(pe_hist_data['counts'])
        return residual

    def objective_function_angle_yellow_x(angle_yellow_x):
        """Objective function for angle_yellow_x, minimizing over angle_y."""
        bounds_y = (0.0, 0.4e-3)
        result_y = minimize_scalar(objective_function_angle_y, bounds=bounds_y, args=(angle_yellow_x,),
                                   method='bounded')
        print(f'Angle Yellow X: {angle_yellow_x}, Optimal Angle Y: {result_y.x}, Residual: {result_y.fun}')
        return result_y.fun

    def minimize_residual():
        """Minimizes residual over angle_yellow_x and angle_y."""
        bounds_yellow_x = (-2 * abs(new_pe_yellow_angle_x), 2 * abs(new_pe_yellow_angle_x))
        result_x = minimize_scalar(objective_function_angle_yellow_x, bounds=bounds_yellow_x, method='bounded')

        best_angle_yellow_x = result_x.x
        min_residual = result_x.fun

        # Perform a final minimization for angle_y with the optimized angle_yellow_x
        bounds_y = (0.0, 0.4e-3)
        result_y = minimize_scalar(objective_function_angle_y, bounds=bounds_y, args=(best_angle_yellow_x,),
                                   method='bounded')
        best_angle_y = result_y.x

        print(f"Optimized Angle Yellow X: {best_angle_yellow_x}")
        print(f"Optimized Angle Y: {best_angle_y}")
        print(f"Minimum Residual: {min_residual}")

        # Plot the final result
        collider_sim.set_bunch_crossing(new_pe_blue_angle_x, -best_angle_y, best_angle_yellow_x, best_angle_y)
        collider_sim.run_sim_parallel()
        fit_shift(collider_sim, pe_hist_data['counts'], pe_hist_data['centers'])
        pe_zs_new, pe_z_dist_new = collider_sim.get_z_density_dist()

        fig_pe, ax_pe = plt.subplots()
        ax_pe.bar(pe_hist_data['centers'], pe_hist_data['counts'],
                  width=pe_hist_data['centers'][1] - pe_hist_data['centers'][0], label='MBD Vertex')
        ax_pe.plot(pe_zs_new, pe_z_dist_new, label='Optimized Fit', color='red')
        ax_pe.legend()
        plt.show()
        return best_angle_y, best_angle_yellow_x

    # Run the minimization
    best_angle_y, best_angle_yellow_x = minimize_residual()

    # Run for a range of gaussian widths. Plot sims and residuals
    gaus_eff_widths = [300, 400, 500, 600, 700, 800, 900, 1000]

    fig_pe, ax_pe = plt.subplots()
    ax_pe.bar(pe_hist_data['centers'], pe_hist_data['counts'],
              width=pe_hist_data['centers'][1] - pe_hist_data['centers'][0], label='MBD Vertex')
    min_gaus_eff_width, min_reid = None, None
    residuals = []
    for gaus_eff_width in gaus_eff_widths:
        # Set and run new peripheral collider sim
        collider_sim.set_bunch_sigmas(np.array([new_bw_x, new_bw_y]), np.array([new_bw_x, new_bw_y]))
        collider_sim.set_bunch_beta_stars(new_beta_star, new_beta_star)
        collider_sim.set_gaus_smearing_sigma(new_mbd_res)
        collider_sim.set_bkg(new_bkg)
        collider_sim.set_bunch_crossing(new_pe_blue_angle_x, -best_angle_y, best_angle_yellow_x, best_angle_y)
        collider_sim.set_gaus_z_efficiency_width(gaus_eff_width)

        if new_pe_offsets is not None:
            collider_sim.set_bunch_offsets(new_pe_offsets[0], new_pe_offsets[1])

        collider_sim.run_sim_parallel()
        fit_shift(collider_sim, pe_hist_data['counts'], pe_hist_data['centers'])
        pe_zs_new, pe_z_dist_new = collider_sim.get_z_density_dist()
        sim_interp_vals = interp1d(pe_zs_new, pe_z_dist_new)(pe_hist_data['centers'])
        residual = np.sqrt(np.sum((pe_hist_data['counts'] - sim_interp_vals)**2) / np.mean(pe_hist_data['counts']))
        ax_pe.plot(pe_zs_new, pe_z_dist_new, label=f'{gaus_eff_width}: {residual:.2e}')
        if min_reid is None or residual < min_reid:
            min_reid = residual
            min_gaus_eff_width = gaus_eff_width
        residuals.append(residual)

    ax_pe.set_title(f'Peripheral bw {bw_nom:.0f} {orientation} Scan Step {peripheral_scan_step}')
    ax_pe.set_xlabel('z Vertex Position (cm)')
    ax_pe.legend()
    fig_pe.tight_layout()

    fig_resids, ax_resids = plt.subplots()
    ax_resids.plot(gaus_eff_widths, residuals)
    ax_resids.set_title(f'Peripheral bw {bw_nom:.0f} {orientation} Scan Step {peripheral_scan_step}')
    ax_resids.set_xlabel('Gaussian Efficiency Width (cm)')
    ax_resids.set_ylabel('Sum Residual')
    fig_resids.tight_layout()

    plt.show(block=True)

    # Here
    angles_yellow_x = np.linspace(-new_pe_yellow_angle_x, 2 * new_pe_yellow_angle_x, 10)
    angles_y = [0.4e-3, 0.35e-3, 0.3e-3, 0.25e-3, 0.2e-3, 0.15e-3, 0.1e-3, 0.05e-3, 0.0]

    fig_pe, ax_pe = plt.subplots()
    ax_pe.bar(pe_hist_data['centers'], pe_hist_data['counts'],
              width=pe_hist_data['centers'][1] - pe_hist_data['centers'][0], label='MBD Vertex')
    ax_pe.plot(pe_zs_og, pe_z_dist_og, color='gray', alpha=0.8, label='Simulation Guess')
    fig_resids, ax_resids = plt.subplots()
    min_angles, min_reid, min_zs, min_dist, min_interp_vals = None, None, None, None, None
    for angle_yellow_x in angles_yellow_x:
        new_resids = []
        print(f'Angle Yellow X: {angle_yellow_x}')
        for angle_y in angles_y:
            # Set and run new peripheral collider sim
            print(f'Angle: {angle_y}')
            collider_sim.set_bunch_sigmas(np.array([new_bw_x, new_bw_y]), np.array([new_bw_x, new_bw_y]))
            collider_sim.set_bunch_beta_stars(new_beta_star, new_beta_star)
            collider_sim.set_gaus_smearing_sigma(new_mbd_res)
            collider_sim.set_bkg(new_bkg)
            collider_sim.set_bunch_crossing(new_pe_blue_angle_x, -angle_y, angle_yellow_x, angle_y)
            collider_sim.set_gaus_z_efficiency_width(new_gaus_eff_width)

            if new_pe_offsets is not None:
                collider_sim.set_bunch_offsets(new_pe_offsets[0], new_pe_offsets[1])

            collider_sim.run_sim_parallel()
            fit_shift(collider_sim, pe_hist_data['counts'], pe_hist_data['centers'])
            pe_zs_new, pe_z_dist_new = collider_sim.get_z_density_dist()
            sim_interp_vals = interp1d(pe_zs_new, pe_z_dist_new)(pe_hist_data['centers'])
            residual = np.sqrt(np.sum((pe_hist_data['counts'] - sim_interp_vals)**2)) / np.mean(pe_hist_data['counts'])
            ax_pe.plot(pe_zs_new, pe_z_dist_new, label=f'{angle_y:.2e}: {residual:.2e}')
            new_resids.append(residual)
            if min_reid is None or residual < min_reid:
                min_reid = residual
                min_angles = [angle_yellow_x, angle_y]
                min_zs, min_dist = pe_zs_new, pe_z_dist_new
                min_interp_vals = sim_interp_vals
        ax_resids.plot(angles_y, new_resids, label=f'{angle_yellow_x:.2e}')

    ax_pe.set_title(f'Peripheral bw {bw_nom:.0f} {orientation} Scan Step {peripheral_scan_step}')
    ax_pe.set_xlabel('z Vertex Position (cm)')
    ax_pe.legend()
    fig_pe.tight_layout()

    ax_resids.set_title(f'Peripheral bw {bw_nom:.0f} {orientation} Scan Step {peripheral_scan_step}')
    ax_resids.set_xlabel('Angle Y (rad)')
    ax_resids.set_ylabel('Sum Residual')
    ax_resids.legend()
    fig_resids.tight_layout()

    # Rerun with the min residuals angles and plot just this
    collider_sim.set_bunch_sigmas(np.array([new_bw_x, new_bw_y]), np.array([new_bw_x, new_bw_y]))
    collider_sim.set_bunch_beta_stars(new_beta_star, new_beta_star)
    collider_sim.set_gaus_smearing_sigma(new_mbd_res)
    collider_sim.set_bkg(new_bkg)
    collider_sim.set_bunch_crossing(new_pe_blue_angle_x, -min_angles[1], min_angles[0], min_angles[1])
    collider_sim.set_gaus_z_efficiency_width(new_gaus_eff_width)

    if new_pe_offsets is not None:
        collider_sim.set_bunch_offsets(new_pe_offsets[0], new_pe_offsets[1])

    collider_sim.run_sim_parallel()
    fit_shift(collider_sim, pe_hist_data['counts'], pe_hist_data['centers'])
    pe_zs_new, pe_z_dist_new = collider_sim.get_z_density_dist()

    fig_pe_min, ax_pe_min = plt.subplots()
    ax_pe_min.bar(pe_hist_data['centers'], pe_hist_data['counts'],
                    width=pe_hist_data['centers'][1] - pe_hist_data['centers'][0], label='MBD Vertex')
    ax_pe_min.plot(pe_zs_new, pe_z_dist_new, color='red', label='Simulation Fit')
    ax_pe_min.plot(min_zs, min_dist, color='green', ls='--', label='Min Residual Fit')
    ax_pe_min.plot(pe_hist_data['centers'], min_interp_vals, color='purple', ls=':', label='Min Residual Interp')
    ax_pe_min.set_title(f'Peripheral bw {bw_nom:.0f} {orientation} Scan Step {peripheral_scan_step}')
    ax_pe_min.set_xlabel('z Vertex Position (cm)')
    ax_pe_min.legend()
    fig_pe_min.tight_layout()

    plt.show()


def calc_vernier_scan_bw_residuals(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, pdf_out_path,
                                   orientation):
    """
    Run simulation with a range of beam widths that are equal in the x and y directions.
    For each mbd z_vertex distribution, fit the simulation z-distribution varying only the crossing angles.
    Calculate the residuals for each step after fitting the crossing angles.
    Plot the mbd vs sim z-vertex distributions for each step (original guess and best fit).
    Plot the residuals values vs the step transverse offset for each beam width.
    Plot the sum of the residuals values vs the beam width.
    """

    # Calc residual also of just the sum of the counts. Plot all residuals vs step on same plot for all bw.
    # Plot sum residual vs bw. Plot lumi sums vs step for each bw compared to CW rates.

    # beam_widths = np.arange(135, 170, 1)
    # beam_widths = np.arange(145, 170, 5)
    # beam_widths = np.arange(145, 185, 1)
    beam_widths = np.arange(145, 160, 5)
    # orientation = 'Vertical'
    # orientation = 'Horizontal'
    cad_data = read_cad_measurement_file(cad_measurement_path)
    cw_rates = get_cw_rates(cad_data)
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, first_dist=False, norms=cw_rates, abs_norm=True)

    # Important parameters
    beta_star_nom = 85.
    mbd_resolution = 2.0  # cm MBD resolution
    # mbd_resolution = 20.0 # cm MBD resolution
    bkg = 0.4e-17  # Background level
    # n_points_xy, n_points_z, n_points_t = 31, 51, 30
    n_points_xy, n_points_z, n_points_t = 61, 151, 61

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(n_points_xy, n_points_xy, n_points_z, n_points_t)
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)
    collider_sim.set_bkg(bkg)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    z_vertex_hists_orient = [hist for hist in z_vertex_hists if hist['scan_axis'] == orientation]

    residuals_sums, sum_residual_sums, all_residual_sums, all_sum_residual_sums = [], [], [], []
    all_total_rates = []
    start_time = datetime.now()
    n_fits = 0
    for bw in beam_widths:
        # Fit the first step, which is head on
        collider_sim = fit_sim_to_mbd_step(collider_sim, z_vertex_hists_orient[0], cad_data, bw, True)
        n_fits += 1

        step_nums, residuals_steps, sum_residual_steps, blue_opt_angles, yellow_opt_angles = [], [], [], [], []
        total_rates_steps, z_shifts = [], []
        for hist_data in z_vertex_hists_orient:
            # if hist_data['scan_step'] not in [6, 12]:
            #     continue
            total_hists = (len(z_vertex_hists_orient) + 1) * len(beam_widths)
            fit_rate = n_fits / (datetime.now() - start_time).total_seconds()
            est_end_time = datetime.now() + timedelta(seconds=(total_hists - n_fits) / fit_rate)
            print(f'Fitting Step {hist_data["scan_step"]} of {len(z_vertex_hists_orient)} for bw {bw:.0f}, '
                  f'Estimated End Time: {est_end_time.strftime("%H:%M:%S")}')

            fig_step_z_dists, ax_step_z_dists = plt.subplots()
            ax_step_z_dists.bar(hist_data['centers'], hist_data['counts'],
                                width=hist_data['centers'][1] - hist_data['centers'][0], label='MBD Vertex')

            # zs, z_dist = collider_sim.get_z_density_dist()
            # ax_step_z_dists.plot(zs, z_dist, color='gray', alpha=0.5, label='Simulation Guess')

            collider_sim = fit_sim_to_mbd_step(collider_sim, hist_data, cad_data, bw)

            zs_opt, z_dist_opt = collider_sim.get_z_density_dist()
            ax_step_z_dists.plot(zs_opt, z_dist_opt, color='r', label='Simulation Fit')

            sim_interp_vals = interp1d(zs_opt, z_dist_opt)(hist_data['centers'])
            residual = np.sum(((hist_data['counts'] - sim_interp_vals) / np.mean(hist_data['counts'])) ** 2)
            sum_residual = ((np.sum(hist_data['counts']) - np.sum(sim_interp_vals)) / np.mean(hist_data['counts'])) ** 2
            z_shifts.append(collider_sim.z_shift / 1e4)  # um to cm

            # Run with +- 10 micron offset
            scan_orientation = hist_data['scan_axis']
            step_cad_data = cad_data[(cad_data['orientation'] == scan_orientation) &
                                     (cad_data['step'] == hist_data['scan_step'])].iloc[0]

            offset_low = step_cad_data['offset_set_val'] * 1e3 - 10  # mm to um
            if scan_orientation == 'Horizontal':
                collider_sim.set_bunch_offsets(np.array([offset_low, 0.]), np.array([0., 0.]))
            elif scan_orientation == 'Vertical':
                collider_sim.set_bunch_offsets(np.array([0., offset_low]), np.array([0., 0.]))

            collider_sim.run_sim_parallel()
            zs_low, z_dist_low = collider_sim.get_z_density_dist()
            ax_step_z_dists.plot(zs_low, z_dist_low, color='orange', ls='--', alpha=0.5, label='Simulation -10um Offset')

            offset_high = step_cad_data['offset_set_val'] * 1e3 + 10  # mm to um
            if scan_orientation == 'Horizontal':
                collider_sim.set_bunch_offsets(np.array([offset_high, 0.]), np.array([0., 0.]))
            elif scan_orientation == 'Vertical':
                collider_sim.set_bunch_offsets(np.array([0., offset_high]), np.array([0., 0.]))

            collider_sim.run_sim_parallel()
            zs_high, z_dist_high = collider_sim.get_z_density_dist()
            ax_step_z_dists.plot(zs_high, z_dist_high, color='r', ls='--', alpha=0.5, label='Simulation +10um Offset')

            step_nums.append(hist_data['scan_step'])
            residuals_steps.append(residual)
            sum_residual_steps.append(sum_residual)
            total_rates_steps.append(np.sum(sim_interp_vals))
            if orientation == 'Horizontal':
                blue_opt_angles.append(collider_sim.bunch1.angle_x)
                yellow_opt_angles.append(collider_sim.bunch2.angle_x)
            elif orientation == 'Vertical':
                blue_opt_angles.append(collider_sim.bunch1.angle_y)
                yellow_opt_angles.append(collider_sim.bunch2.angle_y)

            ax_step_z_dists.set_title(f'bw {bw:.0f} {hist_data["scan_axis"]} Scan Step {hist_data["scan_step"]}')
            ax_step_z_dists.set_xlabel('z Vertex Position (cm)')
            ax_step_z_dists.legend()
            fig_step_z_dists.tight_layout()

            n_fits += 1

        residuals_sums.append(np.sum(residuals_steps))
        sum_residual_sums.append(np.sum(sum_residual_steps))
        all_residual_sums.append(residuals_steps)
        all_sum_residual_sums.append(sum_residual_steps)
        all_total_rates.append(total_rates_steps)

        fig_resid_vs_step, ax_resid_vs_step = plt.subplots()
        ax_resid_vs_step_sum = ax_resid_vs_step.twinx()
        ax_resid_vs_step.plot(step_nums, residuals_steps, marker='o', label='z Residuals')
        ax_resid_vs_step_sum.plot(step_nums, sum_residual_steps, marker='o', c='g', label='Total Residuals')
        ax_resid_vs_step.set_title(f'bw {bw:.0f} Residuals vs Step')
        ax_resid_vs_step.set_xlabel('Step')
        ax_resid_vs_step.set_ylabel('Residual')
        ax_resid_vs_step.legend(loc='upper left')
        ax_resid_vs_step_sum.legend(loc='upper right')
        ax_resid_vs_step.set_ylim(bottom=0)
        ax_resid_vs_step_sum.set_ylim(bottom=0)
        fig_resid_vs_step.tight_layout()

        fig_angles_vs_step, ax_angles_vs_step = plt.subplots()
        ax_angles_vs_step.plot(step_nums, np.array(blue_opt_angles) * 1e3, marker='o', color='blue', label='Blue')
        ax_angles_vs_step.plot(step_nums, np.array(yellow_opt_angles) * 1e3, marker='o', color='orange', label='Yellow')
        ax_angles_vs_step.set_title(f'bw {bw:.0f} Angles vs Step')
        ax_angles_vs_step.set_xlabel('Step')
        ax_angles_vs_step.set_ylabel('Angle (mrad)')
        ax_angles_vs_step.legend()
        fig_angles_vs_step.tight_layout()

        fig_z_shift_vs_step, ax_z_shift_vs_step = plt.subplots()
        ax_z_shift_vs_step.plot(step_nums, z_shifts, marker='o', color='blue')
        ax_z_shift_vs_step.set_title(f'bw {bw:.0f} Z Shift vs Step')
        ax_z_shift_vs_step.set_xlabel('Step')
        ax_z_shift_vs_step.set_ylabel('Z Shift (cm)')
        fig_z_shift_vs_step.tight_layout()

    fig_resid_vs_bw, ax_resid_vs_bw = plt.subplots()
    # Twin axis for sum residuals
    ax_resid_vs_bw_sum = ax_resid_vs_bw.twinx()
    ax_resid_vs_bw.plot(beam_widths, residuals_sums, marker='o', label='z Residuals')
    ax_resid_vs_bw_sum.plot(beam_widths, sum_residual_sums, marker='o', c='g', label='Integral Residuals')
    ax_resid_vs_bw.set_title('Sum of Residuals vs Beam Width')
    ax_resid_vs_bw.set_xlabel('Beam Width (microns)')
    ax_resid_vs_bw.set_ylabel('Sum of Residuals Over Steps')
    ax_resid_vs_bw.legend(loc='upper left')
    ax_resid_vs_bw_sum.legend(loc='upper right')
    ax_resid_vs_bw.set_ylim(bottom=0)
    ax_resid_vs_bw_sum.set_ylim(bottom=0)
    fig_resid_vs_bw.tight_layout()

    fig_all_z_resid_vs_step, ax_all_z_resid_vs_step = plt.subplots()
    for bw_i, residuals_steps in enumerate(all_residual_sums):
        ax_all_z_resid_vs_step.plot(step_nums, residuals_steps, marker='o', alpha=0.5,
                                    label=f'bw {beam_widths[bw_i]:.0f}')
    ax_all_z_resid_vs_step.set_title('All z Residuals vs Step')
    ax_all_z_resid_vs_step.set_xlabel('Step')
    ax_all_z_resid_vs_step.set_ylabel('Residual')
    if len(beam_widths) < 10:
        ax_all_z_resid_vs_step.legend()
    ax_all_z_resid_vs_step.set_ylim(bottom=0)
    fig_all_z_resid_vs_step.tight_layout()

    fig_all_sum_resid_vs_step, ax_all_sum_resid_vs_step = plt.subplots()
    for bw_i, sum_residual_steps in enumerate(all_sum_residual_sums):
        ax_all_sum_resid_vs_step.plot(step_nums, sum_residual_steps, marker='o', alpha=0.5,
                                      label=f'bw {beam_widths[bw_i]:.0f}')
    ax_all_sum_resid_vs_step.set_title('All Integral Residuals vs Step')
    ax_all_sum_resid_vs_step.set_xlabel('Step')
    ax_all_sum_resid_vs_step.set_ylabel('Residual')
    if len(beam_widths) < 10:
        ax_all_sum_resid_vs_step.legend()
    ax_all_sum_resid_vs_step.set_ylim(bottom=0)
    fig_all_sum_resid_vs_step.tight_layout()

    offsets, total_rates = [], []
    for hist_data in z_vertex_hists_orient:
        step_cad_data = cad_data[(cad_data['orientation'] == hist_data['scan_axis']) &
                                 (cad_data['step'] == int(hist_data['scan_step']))]
        offsets.append(step_cad_data['offset_set_val'].values[0])
        total_rates.append(hist_data['counts'].sum())

    offsets_sim = offsets.copy()

    # Sort rates and offsets together by offset
    offsets, total_rates = zip(*sorted(zip(offsets, total_rates)))
    fig_rate_vs_offset, ax_rate_vs_offset = plt.subplots()
    ax_rate_vs_offset.plot(offsets, total_rates, marker='o', alpha=0.5, label='Total Rate CW', zorder=10)
    for bw_i, total_rates_sim in enumerate(all_total_rates):
        total_rates_sim = [total_rates_sim[i] for i in np.argsort(offsets_sim)]
        ax_rate_vs_offset.plot(offsets, total_rates_sim, label=f'Sim bw {beam_widths[bw_i]:.0f}', alpha=0.5, zorder=1)
    ax_rate_vs_offset.set_ylim(bottom=0)
    ax_rate_vs_offset.set_title('Total Rate vs Offset')
    ax_rate_vs_offset.set_xlabel('Offset (um)')
    ax_rate_vs_offset.set_ylabel('Total Rate')
    if len(all_total_rates) < 10:
        ax_rate_vs_offset.legend()
    fig_rate_vs_offset.tight_layout()

    with PdfPages(pdf_out_path) as pdf:
        for fig_num in plt.get_fignums():
            pdf.savefig(plt.figure(fig_num))
            plt.close(fig_num)

    # if sound_on:
    #     winsound.Beep(1000, 3000)


def fit_sim_to_mbd_step(collider_sim, hist_data, cad_data, bw, fit_amp_shift_flag=False):
    """
    Fit simulation to MBD data for a single step.
    """
    collider_sim.set_bunch_sigmas(np.array([bw, bw]), np.array([bw, bw]))

    angle_per_step = {
        'Horizontal': {
            1: 0.07,
            2: 0.08,
            3: 0.17,
            4: 0.15,
            5: 0.11,
            6: 0.08,
            7: 0.06,
            8: 0.04,
            9: 0.03,
            10: -0.11,
            11: -0.06,
            12: -0.02
        },
        'Vertical': {}
    }

    scan_orientation = hist_data['scan_axis']
    step_cad_data = cad_data[(cad_data['orientation'] == scan_orientation) & (cad_data['step'] == hist_data['scan_step'])].iloc[0]
    print(f'\nOrientation: {hist_data["scan_axis"]}, Step: {hist_data["scan_step"]}')

    yellow_bunch_len_scaling = step_cad_data['yellow_bunch_length']
    blue_bunch_len_scaling = step_cad_data['blue_bunch_length']
    collider_sim.set_longitudinal_fit_scaling(blue_bunch_len_scaling, yellow_bunch_len_scaling)

    offset = step_cad_data['offset_set_val'] * 1e3  # mm to um
    if scan_orientation == 'Horizontal':
        collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
    elif scan_orientation == 'Vertical':
        collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

    blue_angle, yellow_angle = 0, 0
    if scan_orientation == 'Horizontal':
        blue_angle, yellow_angle = -step_cad_data['bh8_avg'] / 1e3, -step_cad_data['yh8_avg'] / 1e3  # mrad to rad
        collider_sim.set_bunch_crossing(blue_angle, 0, yellow_angle, 0)
    elif scan_orientation == 'Vertical':
        collider_sim.set_bunch_crossing(0, blue_angle, 0, yellow_angle)

    print(f'Offset: {offset}, blue_len_scale: {blue_bunch_len_scaling}, yellow_len_scale: {yellow_bunch_len_scaling}')
    print(f'Blue Angle: {blue_angle * 1e3:.3f} mrad, Yellow Angle: {yellow_angle * 1e3:.3f} mrad')

    if fit_amp_shift_flag:
        collider_sim.set_amplitude(1.0)
        collider_sim.set_z_shift(0.0)

    collider_sim.run_sim_parallel()

    if fit_amp_shift_flag:
        fit_amp_shift(collider_sim, hist_data['counts'], hist_data['centers'])

    # yellow_angles = np.linspace(-1.0, 2.0, 21) * yellow_angle
    # residuals = []
    # for yellow_angle_i in yellow_angles:
    #     collider_sim.set_bunch_crossing(blue_angle, 0.0, yellow_angle_i, 0.0)
    #     collider_sim.run_sim_parallel()
    #
    #     zs, z_dist = collider_sim.get_z_density_dist()
    #     sim_interp_vals = interp1d(zs, z_dist)(hist_data['centers'])
    #     residual = np.sum(((hist_data['counts'] - sim_interp_vals) / np.sum(hist_data['counts'])) ** 2)
    #     # sim_metrics = get_dist_metrics(zs, z_dist)
    #     # hist_metrics = get_dist_metrics(hist_data['centers'], hist_data['counts'])
    #     # residual = (sim_metrics[0].val - hist_metrics[0].val)**2
    #     residuals.append(residual)
    #     fig_trial, ax_trial = plt.subplots()
    #     ax_trial.bar(hist_data['centers'], hist_data['counts'], width=hist_data['centers'][1] - hist_data['centers'][0], label='MBD Vertex')
    #     ax_trial.plot(zs, z_dist, color='red', alpha=0.5, label='Simulation')
    #     # ax_trial.plot(hist_data['centers'], sim_interp_vals, color='green', alpha=0.5, label='Interpolated Simulation')
    #     ax_trial.set_title(f'Yellow Angle {yellow_angle_i * 1e3:.3f} Residual {residual:.3f}')
    #     ax_trial.set_xlabel('z Vertex Position (cm)')
    #     ax_trial.legend()
    #     fig_trial.tight_layout()
    #
    # min_y_angle = yellow_angles[np.argmin(residuals)]
    #
    # fig, ax = plt.subplots()
    # ax.plot(yellow_angles * 1e3, residuals, marker='o')
    # ax.scatter(min_y_angle * 1e3, np.min(residuals), color='red', marker='x')
    # ax.set_title(f'Yellow Angle Residuals')
    # ax.set_xlabel('Yellow Angle (mrad)')
    # ax.set_ylabel('Residual')
    # fig.tight_layout()
    #
    # collider_sim.set_bunch_crossing(blue_angle, 0.0, min_y_angle, 0.0)

    # if sound_on:
    #     winsound.Beep(1000, 3000)
    # plt.show()

    if hist_data['scan_step'] in angle_per_step[scan_orientation]:
        yellow_angle_opt = angle_per_step[scan_orientation][hist_data['scan_step']] / 1e3
    else:
        res = minimize(fit_beam_pars1, np.array([1.0]),
                       args=(collider_sim, blue_angle, yellow_angle, hist_data['counts'], hist_data['centers'], scan_orientation),
                       bounds=((-4.0, 5.0),))
        yellow_angle_opt = res.x[0] * yellow_angle
    if scan_orientation == 'Horizontal':
        collider_sim.set_bunch_crossing(blue_angle, 0., yellow_angle_opt, 0.)
    elif scan_orientation == 'Vertical':
        collider_sim.set_bunch_crossing(0., blue_angle, 0., yellow_angle_opt)

    # res = minimize(fit_beam_pars2, np.array([1.0, 1.0]),  # Figure out angle orientation!!!
    #                args=(collider_sim, blue_angle, yellow_angle, hist_data['counts'], hist_data['centers']))  #,
    #                # bounds=((-1.5, 3.0), (-1.5, 3.0)))
    # angle1_x, angle2_x = res.x[0] * blue_angle, res.x[1] * yellow_angle
    # print(f'Optimized Angles: {angle1_x * 1e3:.3f} mrad, {angle2_x * 1e3:.3f} mrad')
    # collider_sim.set_bunch_crossing(angle1_x, 0., angle2_x, 0.)

    collider_sim.run_sim_parallel()  # Run simulation with optimized angles
    fit_shift(collider_sim, hist_data['counts'], hist_data['centers'])

    return collider_sim


def vary_angle(collider_sim, hist_data, cad_data, bw, fit_amp_shift_flag=False):
    """
    Fit simulation to MBD data for a single step.
    """
    collider_sim.set_bunch_sigmas(np.array([bw, bw]), np.array([bw, bw]))

    scan_orientation = hist_data['scan_axis']
    step_cad_data = cad_data[(cad_data['orientation'] == scan_orientation) & (cad_data['step'] == hist_data['scan_step'])].iloc[0]
    print(f'\nOrientation: {hist_data["scan_axis"]}, Step: {hist_data["scan_step"]}')

    yellow_bunch_len_scaling = step_cad_data['yellow_bunch_length']
    blue_bunch_len_scaling = step_cad_data['blue_bunch_length']
    # collider_sim.set_longitudinal_fit_scaling(blue_bunch_len_scaling, yellow_bunch_len_scaling)

    offset = step_cad_data['offset_set_val'] * 1e3  # mm to um
    if scan_orientation == 'Horizontal':
        collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
    elif scan_orientation == 'Vertical':
        collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

    blue_angle, yellow_angle = -step_cad_data['bh8_avg'] / 1e3, -step_cad_data['yh8_avg'] / 1e3  # mrad to rad
    collider_sim.set_bunch_crossing(blue_angle, 0, yellow_angle, 0)


    collider_sim.run_sim_parallel()

    angle1_y, angle2_y = res.x[0] * blue_angle, res.x[1] * yellow_angle
    collider_sim.set_bunch_crossing(0., angle1_y, 0., angle2_y)

    # res = minimize(fit_beam_pars1, np.array([1.0]),
    #                args=(collider_sim, blue_angle, yellow_angle, hist_data['counts'], hist_data['centers']),
    #                bounds=((0.0, 4.0),))
    # angle1_y = res.x[0] * blue_angle
    # collider_sim.set_bunch_crossing(0., angle1_y, 0., yellow_angle)

    return collider_sim


def perform_and_compare_vernier_scan(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path):
    """
    Perform a full vernier scan analysis and compare to MBD data.
    """
    cad_data = read_cad_measurement_file(cad_measurement_path)

    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, first_dist=False)

    # Important parameters
    bw_nom = 170
    beta_star_nom = 85.
    mbd_online_resolution = 5.0  # cm MBD resolution on trigger level

    # Will be overwritten by CAD values
    y_offset_nom = 0.
    angle_y_blue, angle_y_yellow = -0.05e-3, -0.18e-3

    collider_sim = BunchCollider()

    collider_sim.set_bunch_rs(np.array([0., y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(0, angle_y_blue, 0, angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    collider_sim.set_bkg(0.2e-17)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)


def sim_fit_cad_params(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, pdf_out_path):
    """
    Run simulation with CAD measurements and fit to MBD distributions.
    """
    cad_data = read_cad_measurement_file(cad_measurement_path)
    print(cad_data)

    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, first_dist=False)

    # Important parameters
    bw_nom = 135
    beta_star_nom = 85.
    mbd_online_resolution = 5.0  # cm MBD resolution on trigger level

    # Will be overwritten by CAD values
    y_offset_nom = +750.
    angle_nom = +0.14e-3

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(0, angle_nom, 0, 0)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    # Sort z_vertex_hists by scan axis and step, horizontal first
    z_vertex_hists = sorted(z_vertex_hists, key=lambda x: (x['scan_axis'], int(x['scan_step'])))

    for hist_data in z_vertex_hists:
        scan_orientation = hist_data['scan_axis']
        step_cad_data = cad_data[cad_data['orientation'] == scan_orientation].iloc[int(hist_data['scan_step'])]
        print(f'\nOrientation: {hist_data["scan_axis"]}, Step: {hist_data["scan_step"]}')
        print(step_cad_data)

        # Blue from left to right, yellow from right to left
        # Negative angle moves blue bunch from positive value to negative value, yellow from negative to positive
        # Offset blue bunch, fix yellow bunch at (0, 0)
        # Angle in x axis --> horizontal, negative angle from cad goes from negative to positive, flip of my convention
        # Horizontal scan in x, vertical scan in y
        xing_uncert = 50.  # microradians
        offset = step_cad_data['offset_set_val'] * 1e3  # mm to um
        blue_angle, yellow_angle = -step_cad_data['bh8_avg'] / 1e3, -step_cad_data['yh8_avg'] / 1e3  # mrad to rad
        # blue_angle_min, blue_angle_max = -step_cad_data['bh8_min'] / 1e3, -step_cad_data['bh8_max'] / 1e3
        blue_angle_min, blue_angle_max = blue_angle - xing_uncert * 1e-6, blue_angle + xing_uncert * 1e-6
        # yellow_angle_min, yellow_angle_max = -step_cad_data['yh8_min'] / 1e3, -step_cad_data['yh8_max'] / 1e3
        yellow_angle_min, yellow_angle_max = yellow_angle - xing_uncert * 1e-6, yellow_angle + xing_uncert * 1e-6
        blue_bunch_len, yellow_bunch_len = step_cad_data['blue_bunch_length'], step_cad_data['yellow_bunch_length']
        blue_bunch_len, yellow_bunch_len = blue_bunch_len * 1e6, yellow_bunch_len * 1e6  # m to microns

        print(f'Offset: {offset}, Blue Angle: {blue_angle}, Yellow Angle: {yellow_angle}, Blue Bunch Length: {blue_bunch_len}, Yellow Bunch Length: {yellow_bunch_len}')

        if scan_orientation == 'Horizontal':
            collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
        elif scan_orientation == 'Vertical':
            collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

        collider_sim.set_bunch_crossing(blue_angle, 0, yellow_angle, 0)
        blue_bunch_sigma = np.array([bw_nom, bw_nom, blue_bunch_len])
        yellow_bunch_sigma = np.array([bw_nom, bw_nom, yellow_bunch_len])
        collider_sim.set_bunch_sigmas(blue_bunch_sigma, yellow_bunch_sigma)

        collider_sim.set_amplitude(1.0)
        collider_sim.set_z_shift(0.0)
        collider_sim.run_sim_parallel()

        fit_amp_shift(collider_sim, hist_data['counts'], hist_data['centers'])
        zs, z_dist = collider_sim.get_z_density_dist()

        resid = np.sum((hist_data['counts'] - interp1d(zs, z_dist)(hist_data['centers'])) ** 2)
        print(f'Residual: {resid}')

        res = minimize(fit_beam_pars2, np.array([1.0, 1.0]),
                       args=(collider_sim, blue_angle, yellow_angle, hist_data['counts'], hist_data['centers']),
                       bounds=((0.0, 4.0), (0.0, 4.0)))
        print(res)
        angle1_y, angle2_y = res.x[0] * blue_angle, res.x[1] * yellow_angle
        # angle1_x, angle2_x = res.x[2] * blue_angle + 0., res.x[3] * yellow_angle + 0.

        # collider_sim.set_bunch_crossing(angle1_x, angle1_y, angle2_x, angle2_y)
        collider_sim.set_bunch_crossing(0., angle1_y, 0., angle2_y)
        collider_sim.set_amplitude(1.0)
        collider_sim.set_z_shift(0.0)
        collider_sim.run_sim_parallel()

        fit_amp_shift(collider_sim, hist_data['counts'], hist_data['centers'])
        zs_opt, z_dist_opt = collider_sim.get_z_density_dist()
        collider_param = collider_sim.get_param_string()

        fig, ax = plt.subplots(figsize=(8, 7))
        bin_width = hist_data['centers'][1] - hist_data['centers'][0]
        ax.bar(hist_data['centers'], hist_data['counts'], width=bin_width, label='MBD Vertex')
        ax.plot(zs, z_dist, color='g', alpha=0.5, ls='--', label='Simulation CAD Angles')
        ax.plot(zs_opt, z_dist_opt, color='r', label='Simulation Fit Angles')
        ax.set_xlim(-399, 399)
        ax.set_title(f'{hist_data["scan_axis"]} Scan Step {hist_data["scan_step"]} | {offset} um')
        ax.set_xlabel('z Vertex Position (cm)')
        ax.annotate(f'{collider_param}', (0.01, 0.99), xycoords='axes fraction', verticalalignment='top')
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(f"{pdf_out_path.replace('.pdf', f'_{scan_orientation}_{offset}.png')}", format='png')
        # plt.show()

    with PdfPages(pdf_out_path) as pdf:
        for fig_num in plt.get_fignums():
            # plt.savefig(plt.figure(fig_num), format='png')
            pdf.savefig(plt.figure(fig_num))
            plt.close(fig_num)


def sim_cad_params(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, pdf_out_path):
    """
    Run simulation with CAD measurements and compare to MBD distributions.
    """
    cad_data = read_cad_measurement_file(cad_measurement_path)
    print(cad_data)

    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, first_dist=False)

    # Important parameters
    bw_nom = 135
    beta_star_nom = 85.
    mbd_online_resolution = 5.0  # cm MBD resolution on trigger level

    # Will be overwritten by CAD values
    y_offset_nom = +750.
    bl1_nom = 130.e4
    bl2_nom = 117.e4
    angle_nom = +0.14e-3

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom, bl1_nom]), np.array([bw_nom, bw_nom, bl2_nom]))
    collider_sim.set_bunch_crossing(0, angle_nom, 0, 0)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    # Sort z_vertex_hists by scan axis and step, horizontal first
    z_vertex_hists = sorted(z_vertex_hists, key=lambda x: (x['scan_axis'], int(x['scan_step'])))

    for hist_data in z_vertex_hists:
        scan_orientation = hist_data['scan_axis']
        step_cad_data = cad_data[cad_data['orientation'] == scan_orientation].iloc[int(hist_data['scan_step'])]
        print(f'\nOrientation: {hist_data["scan_axis"]}, Step: {hist_data["scan_step"]}')
        print(step_cad_data)

        # Blue from left to right, yellow from right to left
        # Negative angle moves blue bunch from positive value to negative value, yellow from negative to positive
        # Offset blue bunch, fix yellow bunch at (0, 0)
        # Angle in x axis --> horizontal, negative angle from cad goes from negative to positive, flip of my convention
        # Horizontal scan in x, vertical scan in y
        xing_uncert = 50.  # microradians
        offset = step_cad_data['offset_set_val'] * 1e3  # mm to um
        blue_angle, yellow_angle = -step_cad_data['bh8_avg'] / 1e3, -step_cad_data['yh8_avg'] / 1e3  # mrad to rad
        # blue_angle_min, blue_angle_max = -step_cad_data['bh8_min'] / 1e3, -step_cad_data['bh8_max'] / 1e3
        blue_angle_min, blue_angle_max = blue_angle - xing_uncert * 1e-6, blue_angle + xing_uncert * 1e-6
        # yellow_angle_min, yellow_angle_max = -step_cad_data['yh8_min'] / 1e3, -step_cad_data['yh8_max'] / 1e3
        yellow_angle_min, yellow_angle_max = yellow_angle - xing_uncert * 1e-6, yellow_angle + xing_uncert * 1e-6
        blue_bunch_len, yellow_bunch_len = step_cad_data['blue_bunch_length'], step_cad_data['yellow_bunch_length']
        blue_bunch_len, yellow_bunch_len = blue_bunch_len * 1e6, yellow_bunch_len * 1e6  # m to microns

        print(f'Offset: {offset}, Blue Angle: {blue_angle}, Yellow Angle: {yellow_angle}, Blue Bunch Length: {blue_bunch_len}, Yellow Bunch Length: {yellow_bunch_len}')

        if scan_orientation == 'Horizontal':
            collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
        elif scan_orientation == 'Vertical':
            collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

        collider_sim.set_bunch_crossing(blue_angle, 0, yellow_angle, 0)
        blue_bunch_sigma = np.array([bw_nom, bw_nom, blue_bunch_len])
        yellow_bunch_sigma = np.array([bw_nom, bw_nom, yellow_bunch_len])
        collider_sim.set_bunch_sigmas(blue_bunch_sigma, yellow_bunch_sigma)

        collider_sim.set_amplitude(1.0)
        collider_sim.set_z_shift(0.0)
        collider_sim.run_sim_parallel()

        zs, z_dist = collider_sim.get_z_density_dist()
        scale = max(hist_data['counts']) / max(z_dist)
        z_max_sim = zs[np.argmax(z_dist)]
        z_max_hist = hist_data['centers'][np.argmax(hist_data['counts'])]
        shift = z_max_sim - z_max_hist  # microns

        res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                       args=(collider_sim, scale, shift, hist_data['counts'], hist_data['centers']),
                       bounds=((0.0, 2.0), (-10e4, 10e4)))
        scale = res.x[0] * scale
        shift = res.x[1] + shift

        collider_sim.set_amplitude(scale)
        collider_sim.set_z_shift(shift)
        zs, z_dist = collider_sim.get_z_density_dist()

        collider_param = collider_sim.get_param_string()

        resid = np.sum((hist_data['counts'] - interp1d(zs, z_dist)(hist_data['centers'])) ** 2)

        min_z_dist, max_z_dist = get_min_max_angles(collider_sim, hist_data, hist_data['centers'],
                                                    blue_angle_min, blue_angle_max, yellow_angle_min, yellow_angle_max)

        fig, ax = plt.subplots(figsize=(8, 7))
        bin_width = hist_data['centers'][1] - hist_data['centers'][0]
        ax.bar(hist_data['centers'], hist_data['counts'], width=bin_width, label='MBD Vertex')
        ax.plot(zs, z_dist, color='r', label='Simulation')
        ax.fill_between(hist_data['centers'], min_z_dist, max_z_dist, color='r', alpha=0.3)
        ax.set_xlim(-399, 399)
        ax.set_title(f'{hist_data["scan_axis"]} Scan Step {hist_data["scan_step"]} | {offset} um')
        ax.set_xlabel('z Vertex Position (cm)')
        ax.annotate(f'{collider_param}', (0.01, 0.99), xycoords='axes fraction', verticalalignment='top')
        ax.legend(loc='upper right')
        fig.tight_layout()
        # plt.show()

    with PdfPages(pdf_out_path) as pdf:
        for fig_num in plt.get_fignums():
            # plt.savefig(plt.figure(fig_num), format='png')
            pdf.savefig(plt.figure(fig_num))
            plt.close(fig_num)


def get_min_max_angles(collider_sim, hist_data, def_zs,
                       min_blue_x_angle, max_blue_x_angle, min_yellow_x_angle, max_yellow_x_angle):
    """
    Run combinations of blue and yellow crossing angles and get z_vertex distributions for each. From these create
    interp_1Ds. Finally, combine these to find the minimum and maximum z_vertex distributions.
    """
    z_dists = []
    for blue_angle in [min_blue_x_angle, max_blue_x_angle]:
        for yellow_angle in [min_yellow_x_angle, max_yellow_x_angle]:
            collider_sim.set_bunch_crossing(blue_angle, 0, yellow_angle, 0)
            collider_sim.set_amplitude(1.0)
            collider_sim.set_z_shift(0.0)
            collider_sim.run_sim_parallel()

            zs, z_dist = collider_sim.get_z_density_dist()
            scale = max(hist_data['counts']) / max(z_dist)
            z_max_sim = zs[np.argmax(z_dist)]
            z_max_hist = hist_data['centers'][np.argmax(hist_data['counts'])]
            shift = z_max_sim - z_max_hist  # microns

            res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                           args=(collider_sim, scale, shift, hist_data['counts'], hist_data['centers']),
                           bounds=((0.0, 2.0), (-10e4, 10e4)))
            scale = res.x[0] * scale
            shift = res.x[1] + shift

            collider_sim.set_amplitude(scale)
            collider_sim.set_z_shift(shift)
            zs, z_dist = collider_sim.get_z_density_dist()
            z_dists.append(interp1d(zs, z_dist)(def_zs))

    min_z_dist = np.min(z_dists, axis=0)
    max_z_dist = np.max(z_dists, axis=0)
    return min_z_dist, max_z_dist



def read_cad_measurement_file(cad_measurement_path):
    """
    Read CAD measurement file and return pandas data frame. First row is header, first column is index.
    """
    cad_data = pd.read_csv(cad_measurement_path, sep='\t', header=0, index_col=0)
    return cad_data


def check_head_on_dependences(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path)
    hist_0 = z_vertex_hists[0]

    # collider_sim = BunchCollider()
    # x_sim = np.array([150., 1.1e6, 85.])
    # bunch_widths = np.linspace(100, 200, 30)
    # resids = []
    # for bunch_width in bunch_widths:
    #     x_sim[0] = bunch_width
    #     res = fit_sim_pars_to_vertex(x_sim, hist_0['centers'], hist_0['counts'], collider_sim)
    #     resids.append(res)
    #
    # fig, ax = plt.subplots()
    # ax.plot(bunch_widths, resids)
    # ax.set_xlabel('Bunch Width (microns)')
    # ax.set_ylabel('Residual')

    collider_sim = BunchCollider()
    x_sim = np.array([150., 110, 85.])
    bunch_lengths = np.linspace(50, 200, 30)
    resids = []
    for bunch_length in bunch_lengths:
        x_sim[1] = bunch_length
        res = fit_sim_pars_to_vertex(x_sim, hist_0['centers'], hist_0['counts'], collider_sim, True)
        resids.append(res)

    fig, ax = plt.subplots()
    ax.plot(bunch_lengths / 1e6, resids)
    ax.set_xlabel('Bunch Length (m)')
    ax.set_ylabel('Residual')

    plt.show()


def fit_head_on(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path)
    hist_0 = z_vertex_hists[0]

    bin_width = hist_0['centers'][1] - hist_0['centers'][0]

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(85., 85.)
    collider_sim.set_bunch_sigmas(np.array([135., 135., 130.e4]), np.array([135., 135., 117.e4]))
    collider_sim.set_bunch_crossing(-0.13e-3, +0.0e-3)
    collider_sim.run_sim()
    zs, z_dist = collider_sim.get_z_density_dist()
    collider_param_str = collider_sim.get_param_string()

    # x0_sim = np.array([150., 110, 85.])
    # bounds = ((0, None), (0, None), (0, None))
    # x0_sim = np.array([117., 130., 85., 85.])
    # bounds = ((0, None), (0, None), (0, None), (0, None))
    x0_sim = np.array([117., 130.])
    bounds = ((1, None), (1, None))
    res = minimize(fit_sim_pars_to_vertex, x0_sim, args=(hist_0['centers'], hist_0['counts'], collider_sim, True),
                   bounds=bounds)
    print(res)

    x0_sim = np.array([*res.x, 85., 85.])
    bounds = ((0, None), (0, None), (0, None), (0, None))
    res = minimize(fit_sim_pars_to_vertex, x0_sim, args=(hist_0['centers'], hist_0['counts'], collider_sim, True),
                   bounds=bounds)
    print(res)

    collider_sim.run_sim()
    zs_opt, z_dist_opt = collider_sim.get_z_density_dist()
    collider_param_str_opt = collider_sim.get_param_string()

    scale = max(hist_0['counts']) / max(z_dist)
    scale_opt = max(hist_0['counts'] / max(z_dist_opt))

    fig, ax = plt.subplots()
    ax.bar(hist_0['centers'], hist_0['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs, z_dist * scale, color='r', label='Simulation')
    ax.plot(zs_opt, z_dist_opt * scale_opt, color='g', label='Simulation Optimized')
    # ax.plot(hist_0['centers'], gaus(hist_0['centers'], *x0), color='gray', label='Guess')
    # ax.plot(hist_0['centers'], gaus(hist_0['centers'], *res.x), color='green', label='Fit')
    ax.set_title(f'{hist_0["scan_axis"]} Scan Step {hist_0["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param_str}', (0.02, 0.75), xycoords='axes fraction',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.bar(hist_0['centers'], hist_0['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs_opt, z_dist_opt * scale_opt, color='r', label='Simulation Optimized')
    # ax.plot(hist_0['centers'], gaus(hist_0['centers'], *x0), color='gray', label='Guess')
    # ax.plot(hist_0['centers'], gaus(hist_0['centers'], *res.x), color='green', label='Fit')
    ax.set_title(f'{hist_0["scan_axis"]} Scan Step {hist_0["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param_str_opt}', (0.02, 0.75), xycoords='axes fraction',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend()
    fig.tight_layout()

    plt.show()


def fit_head_on_manual(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path)
    hist_0 = z_vertex_hists[0]

    bin_width = hist_0['centers'][1] - hist_0['centers'][0]

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(85., 85.)
    collider_sim.set_bunch_sigmas(np.array([135., 135., 130.e4]), np.array([135., 135., 117.e4]))
    collider_sim.set_bunch_crossing(-0.13e-3, +0.0e-3)
    collider_sim.run_sim()
    zs, z_dist = collider_sim.get_z_density_dist()
    collider_param_str = collider_sim.get_param_string()

    scale = max(hist_0['counts']) / max(z_dist)

    fig, ax = plt.subplots()
    ax.bar(hist_0['centers'], hist_0['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs, z_dist * scale, color='r', label='Simulation')
    ax.set_title(f'{hist_0["scan_axis"]} Scan Step {hist_0["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param_str}', (0.02, 0.75), xycoords='axes fraction',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend()
    fig.tight_layout()

    plt.show()


def plot_head_on(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path):
    cad_data = read_cad_measurement_file(cad_measurement_path)
    cw_rates = get_cw_rates(cad_data)

    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False, norms=cw_rates, abs_norm=True)

    # step_to_use = 'vertical_0'
    step_to_use = 'vertical_6'

    all_params_dict = {
        'vertical_0': {
            'hist_num': 0,
        },
        'vertical_6': {
            'hist_num': 6,
        },
    }
    params = all_params_dict[step_to_use]

    hist = z_vertex_hists[params['hist_num']]

    scan_orientation = hist['scan_axis']
    step_cad_data = cad_data[(cad_data['orientation'] == scan_orientation) &
                             (cad_data['step'] == hist['scan_step'])].iloc[0]

    bin_width = hist['centers'][1] - hist['centers'][0]

    # Important parameters
    bw_nom = 160
    beta_star_nom = 85.
    mbd_resolution = 2.0  # cm MBD resolution
    bkg = 0.4e-17  # Background level
    n_points_xy, n_points_z, n_points_t = 61, 151, 61

    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')

    collider_sim = BunchCollider()
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_grid_size(n_points_xy, n_points_xy, n_points_z, n_points_t)
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)
    collider_sim.set_bkg(bkg)
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    yellow_bunch_len_scaling = step_cad_data['yellow_bunch_length']
    blue_bunch_len_scaling = step_cad_data['blue_bunch_length']
    collider_sim.set_longitudinal_fit_scaling(blue_bunch_len_scaling, yellow_bunch_len_scaling)

    offset = step_cad_data['offset_set_val'] * 1e3  # mm to um
    if scan_orientation == 'Horizontal':
        collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
    elif scan_orientation == 'Vertical':
        collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

    blue_angle, yellow_angle = -step_cad_data['bh8_avg'] / 1e3, -step_cad_data['yh8_avg'] / 1e3  # mrad to rad
    collider_sim.set_bunch_crossing(blue_angle, 0, yellow_angle, 0)

    # collider_sim = BunchCollider()
    # collider_sim.set_bunch_rs(np.array([x_offset_nom, y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    # collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    # collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    # collider_sim.set_bunch_crossing(angle_x_blue, angle_y_blue, angle_x_yellow, angle_y_yellow)
    # collider_sim.set_gaus_smearing_sigma(mbd_online_resolution_nom)
    # collider_sim.set_gaus_z_efficiency_width(z_eff_width)
    # collider_sim.set_longitudinal_fit_scaling(blue_length_scaling, yellow_length_scaling)
    # blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    # yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    # collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)


    collider_sim.run_sim_parallel()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()
    # collider_param_str_og = collider_sim.get_param_string()

    scale = max(hist['counts']) / max(z_dist_og)
    z_max_sim_og = zs_og[np.argmax(z_dist_og)]
    z_max_hist_og = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim_og}, z_max_hist: {z_max_hist_og}')
    shift = z_max_sim_og - z_max_hist_og  # cm
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]), args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    new_y_offset = 0.
    new_x_offset = 0.
    new_bw = 160
    new_bkg = 0.4e-17
    beta_star = 110
    new_angle_y_blue, new_angle_y_yellow = collider_sim.bunch1.angle_y, collider_sim.bunch2.angle_y
    # new_angle_y_blue, new_angle_y_yellow = -0.0e-3, -0.0e-3
    # new_angle_y_blue, new_angle_y_yellow = -0.07e-3, -0.114e-3
    # new_angle_x_blue, new_angle_x_yellow = 0.05e-3, -0.02e-3
    # new_angle_x_blue, new_angle_x_yellow = -0.05e-3, 0.05e-3
    # new_angle_x_blue, new_angle_x_yellow = -0.0e-3, 0.0e-3
    new_angle_x_blue, new_angle_x_yellow = collider_sim.bunch1.angle_x, collider_sim.bunch2.angle_x
    # new_mbd_online_resolution_nom = None  # cm MBD resolution on trigger level
    new_mbd_online_resolution_nom = 5.0  # cm MBD resolution on trigger level
    new_z_eff_width = 500.  # cm

    collider_sim.set_bunch_rs(np.array([new_x_offset, new_y_offset, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star, beta_star)
    collider_sim.set_bunch_sigmas(np.array([new_bw, new_bw]), np.array([new_bw, new_bw]))
    collider_sim.set_bunch_crossing(new_angle_x_blue, new_angle_y_blue, new_angle_x_yellow, new_angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(new_mbd_online_resolution_nom)
    collider_sim.set_gaus_z_efficiency_width(new_z_eff_width)
    collider_sim.set_bkg(new_bkg)
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    blue_bunch_len_scaling = step_cad_data['blue_bunch_length']
    yellow_bunch_len_scaling = step_cad_data['yellow_bunch_length']
    collider_sim.set_longitudinal_fit_scaling(blue_bunch_len_scaling, yellow_bunch_len_scaling)

    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)

    collider_sim.run_sim_parallel()
    zs, z_dist = collider_sim.get_z_density_dist()
    # collider_param_str_og = collider_sim.get_param_string()

    scale = max(hist['counts']) / max(z_dist)
    z_max_sim = zs[np.argmax(z_dist)]
    z_max_hist = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim}, z_max_hist: {z_max_hist}')
    shift = z_max_sim - z_max_hist  # cm
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift
    print(f'scale: {scale}, shift: {shift}')

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs, z_dist = collider_sim.get_z_density_dist()

    residual = np.sum((hist['counts'] - interp1d(zs, z_dist)(hist['centers']))**2)
    print(f'Opt_shift_residual: {residual}')

    collider_param = collider_sim.get_param_string()

    fig, ax = plt.subplots()
    ax.bar(hist['centers'], hist['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs_og, z_dist_og, color='gray', ls='--', alpha=0.6, label='Simulation Guess')
    ax.plot(zs, z_dist, color='r', label='Simulation Fit')
    ax.set_xlim(-399, 399)
    ax.set_title(f'{hist["scan_axis"]} Scan Step {hist["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param}', (0.01, 0.99), xycoords='axes fraction', verticalalignment='top',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper right')
    fig.tight_layout()

    plt.show()


def plot_peripheral(z_vertex_root_path, longitudinal_fit_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[-1]
    rate = 8730.4249
    # time_correction_head_on = 0.23213476851091536
    # time_correction = np.sum(hist['counts']) / rate
    # print(f'Time / scale_factor * correction factor: {time_correction}')

    time_per_step = 45  # seconds nominal time to spend per step
    # Adjust hist['counts'] to match rate for time_per_step
    hist['counts'] = hist['counts'] * rate / np.sum(hist['counts']) * time_per_step

    bin_width = hist['centers'][1] - hist['centers'][0]

    # Important parameters
    bw_nom = 180
    beta_star_nom = 85.
    mbd_online_resolution_nom = 5.0  # cm MBD resolution on trigger level
    z_eff_width = 500.  # cm
    y_offset_nom = -930.
    x_offset_nom = 0.
    yellow_length_scaling, blue_length_scaling = 1.00043315069711, 1.00030348127217
    angle_y_blue, angle_y_yellow = -0.07e-3, -0.114e-3
    angle_x_blue, angle_x_yellow = 0., 0.
    # fixed_scale = None
    # fixed_scale = 9.681147189701276e+28
    fixed_scale = 6.0464621817851715e+28

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([x_offset_nom, y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(angle_x_blue, angle_y_blue, angle_x_yellow, angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution_nom)
    collider_sim.set_gaus_z_efficiency_width(z_eff_width)
    collider_sim.set_longitudinal_fit_scaling(blue_length_scaling, yellow_length_scaling)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    collider_sim.run_sim_parallel()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()
    # collider_param_str_og = collider_sim.get_param_string()

    scale = max(hist['counts']) / max(z_dist_og)
    z_max_sim_og = zs_og[np.argmax(z_dist_og)]
    z_max_hist_og = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim_og}, z_max_hist: {z_max_hist_og}')
    shift = z_max_sim_og - z_max_hist_og  # microns
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]), args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    if fixed_scale is not None:
        scale = fixed_scale
    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    new_y_offset = -930
    new_x_offset = -100
    new_bw = 160.0
    new_beta_star = 85.0
    new_angle_y_blue, new_angle_y_yellow = -0.05e-3, -0.18e-3
    new_angle_x_blue, new_angle_x_yellow = 0.05e-3, -0.02e-3
    # new_angle_x_blue, new_angle_x_yellow = 0.0e-3, -0.0e-3
    new_mbd_online_resolution_nom = 5.0  # cm MBD resolution on trigger level
    new_z_eff_width = 500.  # cm

    collider_sim.set_bunch_rs(np.array([new_x_offset, new_y_offset, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(new_beta_star, new_beta_star)
    collider_sim.set_bunch_sigmas(np.array([new_bw, new_bw]), np.array([new_bw, new_bw]))
    collider_sim.set_bunch_crossing(new_angle_x_blue, new_angle_y_blue, new_angle_x_yellow, new_angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(new_mbd_online_resolution_nom)
    collider_sim.set_gaus_z_efficiency_width(new_z_eff_width)
    collider_sim.set_bkg(0.2e-17)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)

    collider_sim.run_sim_parallel()
    zs, z_dist = collider_sim.get_z_density_dist()
    # collider_param_str_og = collider_sim.get_param_string()

    scale = max(hist['counts']) / max(z_dist)
    z_max_sim = zs[np.argmax(z_dist)]
    z_max_hist = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim}, z_max_hist: {z_max_hist}')
    shift = z_max_sim - z_max_hist  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift
    print(f'scale: {scale}, shift: {shift}')

    if fixed_scale is not None:
        scale = fixed_scale
    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs, z_dist = collider_sim.get_z_density_dist()

    residual = np.sum((hist['counts'] - interp1d(zs, z_dist)(hist['centers']))**2)
    print(f'Opt_shift_residual: {residual}')

    collider_param = collider_sim.get_param_string()

    fig, ax = plt.subplots()
    ax.bar(hist['centers'], hist['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs_og, z_dist_og, color='gray', ls='--', alpha=0.6, label='Simulation Guess')
    ax.plot(zs, z_dist, color='r', label='Simulation Fit')
    ax.set_xlim(-399, 399)
    ax.set_title(f'{hist["scan_axis"]} Scan Step {hist["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param}', (0.01, 0.99), xycoords='axes fraction', verticalalignment='top',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper right')
    fig.tight_layout()

    plt.show()


def fit_peripheral(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[-1]

    bin_width = hist['centers'][1] - hist['centers'][0]

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., +750., -6.e6]), np.array([0., 0., +6.e6]))
    # collider_sim.set_bunch_rs(np.array([0., -1000., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(85., 85.)
    collider_sim.set_bunch_sigmas(np.array([135., 135., 130.e4]), np.array([135., 135., 117.e4]))
    # collider_sim.set_bunch_sigmas(np.array([150., 150., 130.e4]), np.array([150., 150., 130.e4]))
    # collider_sim.set_bunch_crossing(-0.08e-3 / 2, +0.107e-3 / 2)
    # collider_sim.set_bunch_crossing(-0.07e-3 / 2, +0.114e-3 / 2)
    # collider_sim.set_bunch_crossing(-0.05e-3, 0)
    # collider_sim.set_bunch_crossing(-0.028e-3, 0)
    collider_sim.set_bunch_crossing(0, -0.2e-3, 0, +0.0e-3)
    # z_shift = -5  # cm Distance to shift center of collisions

    collider_sim.run_sim()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()
    # collider_param_str_og = collider_sim.get_param_string()

    scale = max(hist['counts']) / max(z_dist_og)
    z_max_sim_og = zs_og[np.argmax(z_dist_og)]
    z_max_hist_og = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim_og}, z_max_hist: {z_max_hist_og}')
    shift = z_max_sim_og - z_max_hist_og  # microns
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]), args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    # Calculate residual
    residual = np.sum((hist['counts'] - interp1d(zs_og, z_dist_og)(hist['centers']))**2)
    print(f'Guess Residual: {residual}')

    angles_y = np.linspace(-0.2e-3, -0.1e-3, 50)
    resids = []
    for angle_y in angles_y:
        collider_sim.set_bunch_crossing(0, angle_y, 0, +0.0e-3)
        collider_sim.set_amplitude(1.0)
        collider_sim.set_z_shift(0.0)
        collider_sim.run_sim()

        zs, z_dist = collider_sim.get_z_density_dist()
        scale = max(hist['counts']) / max(z_dist)
        z_max_sim = zs[np.argmax(z_dist)]
        z_max_hist = hist['centers'][np.argmax(hist['counts'])]
        shift = z_max_sim - z_max_hist  # microns

        res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                       args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                       bounds=((0.0, 2.0), (-10e4, 10e4)))
        scale = res.x[0] * scale
        shift = res.x[1] + shift

        collider_sim.set_amplitude(scale)
        collider_sim.set_z_shift(shift)
        zs, z_dist = collider_sim.get_z_density_dist()

        resid = np.sum((hist['counts'] - interp1d(zs, z_dist)(hist['centers']))**2)
        resids.append(resid)
        print(f'angle_y: {angle_y}, residual: {resid}')

        # fig, ax = plt.subplots()
        # ax.bar(hist['centers'], hist['counts'], width=bin_width, label='MBD Vertex')
        # ax.plot(zs_og, z_dist_og, color='gray', ls='--', alpha=0.6, label='Simulation Guess')
        # ax.plot(zs, z_dist, color='r', label='Simulation Fit')
        # ax.set_title(f'Angle Y: {angle_y * 1e3:.4f} mrad')
        # ax.set_xlabel('z Vertex Position (cm)')
        # # ax.annotate(f'{collider_param}', (0.02, 0.75), xycoords='axes fraction',
        # #             bbox=dict(facecolor='wheat', alpha=0.5))
        # ax.annotate(f'Residual: {resid:.2e}', (0.02, 0.75), xycoords='axes fraction',
        #             bbox=dict(facecolor='wheat', alpha=0.5))
        # ax.legend()
        # fig.tight_layout()

    fig_resids, ax_resids = plt.subplots()
    ax_resids.plot(angles_y, resids)
    ax_resids.set_xlabel('Crossing Angle (rad)')
    ax_resids.set_ylabel('Residual')

    # Get best angle
    best_angle = angles_y[np.argmin(resids)]

    collider_sim.set_bunch_crossing(0, best_angle, 0, +0.0e-3)
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)
    collider_sim.run_sim()
    zs, z_dist = collider_sim.get_z_density_dist()
    scale = max(hist['counts']) / max(z_dist)
    z_max_sim = zs[np.argmax(z_dist)]
    z_max_hist = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim}, z_max_hist: {z_max_hist}')
    shift = z_max_sim - z_max_hist  # microns

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs, z_dist = collider_sim.get_z_density_dist()

    collider_param = collider_sim.get_param_string()

    fig, ax = plt.subplots()
    ax.bar(hist['centers'], hist['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs_og, z_dist_og, color='gray', ls='--', alpha=0.6, label='Simulation Guess')
    ax.plot(zs, z_dist, color='r', label='Simulation Fit')
    ax.set_title(f'{hist["scan_axis"]} Scan Step {hist["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param}', (0.02, 0.75), xycoords='axes fraction',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend()
    fig.tight_layout()

    plt.show()


def fit_peripheral_scipy(z_vertex_root_path, longitudinal_fit_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[-1]

    bin_width = hist['centers'][1] - hist['centers'][0]

    # Important parameters
    bw_nom = 160
    beta_star_nom = 85.
    mbd_online_resolution = 5.0  # cm MBD resolution on trigger level
    y_offset_nom = -930.
    angle_y_blue, angle_y_yellow = -0.05e-3, -0.18e-3

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(0, angle_y_blue, 0, angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    collider_sim.set_bkg(0.2e-17)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    collider_sim.run_sim_parallel()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    scale = max(hist['counts']) / max(z_dist_og)
    z_max_sim_og = zs_og[np.argmax(z_dist_og)]
    z_max_hist_og = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim_og}, z_max_hist: {z_max_hist_og}')
    shift = z_max_sim_og - z_max_hist_og  # cm
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]), args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    # Calculate residual
    residual = np.sum((hist['counts'] - interp1d(zs_og, z_dist_og)(hist['centers']))**2)
    print(f'Guess Residual: {residual}')

    res = minimize(fit_beam_pars3, np.array([1.0, 1.0, 1.0]),
                   args=(collider_sim, angle_y_blue, angle_y_yellow, bw_nom,
                         hist['counts'], hist['centers']),
                   bounds=((0.1, 2.0), (0.1, 2.0), (0.1, 2.0)))
    print(res)
    angle_y_blue, angle_y_yellow = res.x[0] * angle_y_blue, res.x[1] * angle_y_yellow
    bw = res.x[2] * bw_nom

    collider_sim.set_bunch_crossing(0, angle_y_blue, 0, angle_y_yellow)
    collider_sim.set_bunch_sigmas(np.array([bw, bw]), np.array([bw, bw]))
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)
    collider_sim.run_sim()
    zs, z_dist = collider_sim.get_z_density_dist()
    scale = max(hist['counts']) / max(z_dist)
    z_max_sim = zs[np.argmax(z_dist)]
    z_max_hist = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim}, z_max_hist: {z_max_hist}')
    shift = z_max_sim - z_max_hist # microns

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs, z_dist = collider_sim.get_z_density_dist()

    collider_param = collider_sim.get_param_string()

    fig, ax = plt.subplots()
    ax.bar(hist['centers'], hist['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs_og, z_dist_og, color='gray', ls='--', alpha=0.6, label='Simulation Guess')
    ax.plot(zs, z_dist, color='r', label='Simulation Fit')
    ax.set_title(f'{hist["scan_axis"]} Scan Step {hist["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.set_xlim(-349, 349)
    ax.annotate(f'{collider_param}', (0.02, 0.75), xycoords='axes fraction',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper right')
    fig.tight_layout()

    plt.show()


def peripheral_metric_test(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[-1]
    # hist = z_vertex_hists[0]

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., +750., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(85., 85.)
    collider_sim.set_bunch_sigmas(np.array([135., 135., 130.e4]), np.array([135., 135., 117.e4]))
    collider_sim.set_bunch_crossing(0, -0.14e-3, 0, +0.0e-3)

    collider_sim.run_sim()
    zs_sim, z_dist_sim = collider_sim.get_z_density_dist()
    # fit_amp_shift(collider_sim, hist['counts'], hist['centers'])

    print('\nData Metrics:')
    metrics_data = get_dist_metrics(hist['centers'], hist['counts'], True)
    print('\nSimulation Metrics:')
    metrics_sim = get_dist_metrics(zs_sim, z_dist_sim, True)

    plt.show()


def peripheral_metric_sensitivity(base_path, z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[-1]
    metrics_data = get_dist_metrics(hist['centers'], hist['counts'])

    out_dir = f'{base_path}Analysis/metric_sensitivities/'

    y_offset_nom = +750.
    bw_nom = 135
    bl1_nom = 130.e4
    bl2_nom = 117.e4
    angle_nom = -0.14e-3
    beta_star_nom = 85.

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom, bl1_nom]), np.array([bw_nom, bw_nom, bl2_nom]))
    collider_sim.set_bunch_crossing(0, angle_nom, 0, 0)

    points_per_var = 30

    angles = np.linspace(-0.2e-3, -0.1e-3, points_per_var)
    angle_metrics = []
    for angle in angles:
        collider_sim.set_bunch_crossing(0, angle, 0, 0)
        collider_sim.run_sim()
        zs, z_dist = collider_sim.get_z_density_dist()
        angle_metrics_i = get_dist_metrics(zs, z_dist)
        print(f'Angle: {angle}, Metrics: {angle_metrics_i}')
        angle_metrics.append(angle_metrics_i)
    collider_sim.set_bunch_crossing(0, angle_nom, 0, 0)

    plot_metric_sensitivities(angles * 1e3, np.array(angle_metrics), metrics_data, angle_nom * 1e3, 'Crossing Angle (mrad)')

    beam_widths = np.linspace(100, 200, points_per_var)
    beam_width_metrics = []
    for beam_width in beam_widths:
        bunch1_sigma, bunch2_sigma = collider_sim.get_beam_sigmas()
        bunch1_sigma = np.array([beam_width, beam_width, bunch1_sigma[2]])
        bunch2_sigma = np.array([beam_width, beam_width, bunch2_sigma[2]])
        collider_sim.set_bunch_sigmas(bunch1_sigma, bunch2_sigma)
        collider_sim.run_sim()
        zs, z_dist = collider_sim.get_z_density_dist()
        beam_width_metrics_i = get_dist_metrics(zs, z_dist)
        print(f'Beam width: {beam_width}, Metrics: {beam_width_metrics_i}')
        beam_width_metrics.append(beam_width_metrics_i)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom, bl1_nom]), np.array([bw_nom, bw_nom, bl2_nom]))

    plot_metric_sensitivities(beam_widths, np.array(beam_width_metrics), metrics_data, bw_nom, 'Beam Width (microns)')

    beta_stars = np.linspace(80, 90, points_per_var)
    beta_star_metrics = []
    for beta_star in beta_stars:
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
        collider_sim.run_sim()
        zs, z_dist = collider_sim.get_z_density_dist()
        beta_star_metrics_i = get_dist_metrics(zs, z_dist)
        print(f'Beta Star: {beta_star}, Metrics: {beta_star_metrics_i}')
        beta_star_metrics.append(beta_star_metrics_i)
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)

    plot_metric_sensitivities(beta_stars, np.array(beta_star_metrics), metrics_data, beta_star_nom, 'Beta Star (cm)')

    y_offsets = np.linspace(700, 800, points_per_var)
    y_offset_metrics = []
    for y_offset in y_offsets:
        collider_sim.set_bunch_offsets(np.array([0., y_offset]), np.array([0., 0.]))
        collider_sim.run_sim()
        zs, z_dist = collider_sim.get_z_density_dist()
        y_offset_metrics_i = get_dist_metrics(zs, z_dist)
        print(f'Y Offset: {y_offset}, Metrics: {y_offset_metrics_i}')
        y_offset_metrics.append(y_offset_metrics_i)
    collider_sim.set_bunch_offsets(np.array([0., y_offset_nom]), np.array([0., 0.]))

    plot_metric_sensitivities(y_offsets, np.array(y_offset_metrics), metrics_data, y_offset_nom, 'Y Offset (cm)')

    bl1s = np.linspace(120.e4, 140.e4, points_per_var)
    bl1_metrics = []
    for bl1 in bl1s:
        bunch1_sigma, bunch2_sigma = collider_sim.get_beam_sigmas()
        bunch1_sigma = np.array([bw_nom, bw_nom, bl1])
        collider_sim.set_bunch_sigmas(bunch1_sigma, bunch2_sigma)
        collider_sim.run_sim()
        zs, z_dist = collider_sim.get_z_density_dist()
        bl1_metrics_i = get_dist_metrics(zs, z_dist)
        print(f'BL1: {bl1}, Metrics: {bl1_metrics_i}')
        bl1_metrics.append(bl1_metrics_i)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom, bl1_nom]), np.array([bw_nom, bw_nom, bl2_nom]))

    plot_metric_sensitivities(bl1s / 1e4, np.array(bl1_metrics), metrics_data, bl1_nom / 1e4, 'Bunch1 Length (cm)')

    for fig in plt.get_fignums():
        fig_title = plt.figure(fig).canvas.manager.get_window_title()
        plt.figure(fig).savefig(f'{out_dir}{fig_title}.png')

    plt.show()


def head_on_metric_sensitivity(base_path, z_vertex_root_path, longitudinal_fit_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[0]
    metrics_data = get_head_on_dist_width(hist['centers'], hist['counts'])

    out_dir = f'{base_path}Analysis/metric_sensitivities_head_on/'
    if not os.path.exists(out_dir):  # Make out_dir if it doesn't exist
        os.makedirs(out_dir)

    # Important parameters
    bw_nom = 180
    beta_star_nom = 85.
    mbd_online_resolution_nom = 5.0  # cm MBD resolution on trigger level
    y_offset_nom = -0.
    x_offset_nom = 0.
    yellow_length_scaling, blue_length_scaling = 0.991593955543314, 0.993863022403956
    angle_y_blue, angle_y_yellow = -0.07e-3, -0.114e-3
    angle_x_blue, angle_x_yellow = 0., 0.

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([x_offset_nom, y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(angle_x_blue, angle_y_blue, angle_x_yellow, angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution_nom)
    collider_sim.set_longitudinal_fit_scaling(blue_length_scaling, yellow_length_scaling)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    points_per_var = 30

    angles = np.linspace(-0.2e-3, -0.0e-3, points_per_var)
    angle_metrics = []
    for angle in angles:
        collider_sim.set_bunch_crossing(0, angle, 0, 0)
        collider_sim.run_sim_parallel()
        zs, z_dist = collider_sim.get_z_density_dist()
        angle_metrics_i = get_head_on_dist_width(zs, z_dist)
        print(f'Angle: {angle}, Metrics: {angle_metrics_i}')
        angle_metrics.append(angle_metrics_i)
    collider_sim.set_bunch_crossing(0, angle_y_blue, 0, 0)

    plot_metric_sensitivities(angles * 1e3, np.array(angle_metrics), metrics_data, angle_y_blue * 1e3, 'Crossing Angle (mrad)')

    beam_widths = np.linspace(100, 200, points_per_var)
    beam_width_metrics = []
    for beam_width in beam_widths:
        bunch1_sigma = np.array([beam_width, beam_width])
        bunch2_sigma = np.array([beam_width, beam_width])
        collider_sim.set_bunch_sigmas(bunch1_sigma, bunch2_sigma)
        collider_sim.run_sim_parallel()
        zs, z_dist = collider_sim.get_z_density_dist()
        beam_width_metrics_i = get_head_on_dist_width(zs, z_dist)
        print(f'Beam width: {beam_width}, Metrics: {beam_width_metrics_i}')
        beam_width_metrics.append(beam_width_metrics_i)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))

    plot_metric_sensitivities(beam_widths, np.array(beam_width_metrics), metrics_data, bw_nom, 'Beam Width (microns)')

    beta_stars = np.linspace(80, 90, points_per_var)
    beta_star_metrics = []
    for beta_star in beta_stars:
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
        collider_sim.run_sim_parallel()
        zs, z_dist = collider_sim.get_z_density_dist()
        beta_star_metrics_i = get_head_on_dist_width(zs, z_dist)
        print(f'Beta Star: {beta_star}, Metrics: {beta_star_metrics_i}')
        beta_star_metrics.append(beta_star_metrics_i)
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)

    plot_metric_sensitivities(beta_stars, np.array(beta_star_metrics), metrics_data, beta_star_nom, 'Beta Star (cm)')

    y_offsets = np.linspace(-100, 100, points_per_var)
    y_offset_metrics = []
    for y_offset in y_offsets:
        collider_sim.set_bunch_offsets(np.array([0., y_offset]), np.array([0., 0.]))
        collider_sim.run_sim_parallel()
        zs, z_dist = collider_sim.get_z_density_dist()
        y_offset_metrics_i = get_head_on_dist_width(zs, z_dist)
        print(f'Y Offset: {y_offset}, Metrics: {y_offset_metrics_i}')
        y_offset_metrics.append(y_offset_metrics_i)
    collider_sim.set_bunch_offsets(np.array([0., y_offset_nom]), np.array([0., 0.]))

    plot_metric_sensitivities(y_offsets, np.array(y_offset_metrics), metrics_data, y_offset_nom, 'Y Offset (cm)')

    # bl1s = np.linspace(120.e4, 140.e4, points_per_var)
    # bl1_metrics = []
    # for bl1 in bl1s:
    #     bunch1_sigma, bunch2_sigma = collider_sim.get_beam_sigmas()
    #     bunch1_sigma = np.array([bw_nom, bw_nom, bl1])
    #     collider_sim.set_bunch_sigmas(bunch1_sigma, bunch2_sigma)
    #     collider_sim.run_sim()
    #     zs, z_dist = collider_sim.get_z_density_dist()
    #     bl1_metrics_i = get_dist_metrics(zs, z_dist)
    #     print(f'BL1: {bl1}, Metrics: {bl1_metrics_i}')
    #     bl1_metrics.append(bl1_metrics_i)
    # collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom, bl1_nom]), np.array([bw_nom, bw_nom, bl2_nom]))
    #
    # plot_metric_sensitivities(bl1s / 1e4, np.array(bl1_metrics), metrics_data, bl1_nom / 1e4, 'Bunch1 Length (cm)')

    for fig in plt.get_fignums():
        fig_title = plt.figure(fig).canvas.manager.get_window_title()
        plt.figure(fig).savefig(f'{out_dir}{fig_title}.png')

    plt.show()


def plot_metric_sensitivities(variable_vals, metrics, metrics_data, var_nom, variable_xlabel=None):
    var_name = variable_xlabel.split('(')[0].strip()
    fig_height_ratio, ax_hr = plt.subplots()
    ax_hr.errorbar(variable_vals, [m.val for m in metrics[:, 0]], yerr=[m.err for m in metrics[:, 0]],
                   label='Height Ratio', ls='none', marker='o')
    ax_hr.set_xlabel(variable_xlabel)
    ax_hr.set_ylabel('Height Ratio')
    ax_hr.axhline(metrics_data[0].val, color='green', ls='--', label='Data')
    ax_hr.axhspan(metrics_data[0].val - metrics_data[0].err, metrics_data[0].val + metrics_data[0].err,
                    color='green', alpha=0.3)
    ax_hr.axvline(var_nom, color='blue', ls='--', label='Nominal')
    ax_hr2 = ax_hr.twinx()
    ax_hr2.plot([], [])
    ax_hr2.set_ylabel('Height Ratio % Diff')
    ax_hr2.set_ylim((np.array(ax_hr.get_ylim()) - metrics_data[0].val) / metrics_data[0].val * 100)
    ax_hr2.yaxis.set_major_formatter(PercentFormatter())
    ax_hr.legend()
    ax_hr.set_title(f'Height Ratio Sensitivity to {var_name}')
    fig_height_ratio.canvas.manager.set_window_title(f'Height Ratio Sensitivity to {var_name}')
    fig_height_ratio.tight_layout()

    if metrics.shape[1] < 3:
        return

    fig_peak_sep, ax_ps = plt.subplots()
    ax_ps.errorbar(variable_vals, [m.val for m in metrics[:, 1]], yerr=[m.err for m in metrics[:, 1]],
                   label='Peak Separation', ls='none', marker='o')
    ax_ps.set_xlabel(variable_xlabel)
    ax_ps.set_ylabel('Peak Separation (cm)')
    ax_ps.axhline(metrics_data[1].val, color='green', ls='--', label='Data')
    ax_ps.axhspan(metrics_data[1].val - metrics_data[1].err, metrics_data[1].val + metrics_data[1].err,
                    color='green', alpha=0.3)
    ax_ps.axvline(var_nom, color='blue', ls='--', label='Nominal')
    ax_ps2 = ax_ps.twinx()
    ax_ps2.plot([], [])
    ax_ps2.set_ylabel('Peak Separation % Diff')
    ax_ps2.set_ylim((np.array(ax_ps.get_ylim()) - metrics_data[1].val) / metrics_data[1].val * 100)
    ax_ps2.yaxis.set_major_formatter(PercentFormatter())
    ax_ps.legend()
    ax_ps.set_title(f'Peak Separation Sensitivity to {var_name}')
    fig_peak_sep.canvas.manager.set_window_title(f'Peak Separation Sensitivity to {var_name}')
    fig_peak_sep.tight_layout()

    fig_main_peak_width, ax_mpw = plt.subplots()
    ax_mpw.errorbar(variable_vals, [m.val for m in metrics[:, 2]], yerr=[m.err for m in metrics[:, 2]],
                    label='Main Peak Width', ls='none', marker='o')
    ax_mpw.set_xlabel(variable_xlabel)
    ax_mpw.set_ylabel('Main Peak Width (cm)')
    ax_mpw.axhline(metrics_data[2].val, color='green', ls='--', label='Data')
    ax_mpw.axhspan(metrics_data[2].val - metrics_data[2].err, metrics_data[2].val + metrics_data[2].err,
                    color='green', alpha=0.3)
    ax_mpw.axvline(var_nom, color='blue', ls='--', label='Nominal')
    ax_mpw2 = ax_mpw.twinx()
    ax_mpw2.plot([], [])
    ax_mpw2.set_ylabel('Main Peak Width % Diff')
    ax_mpw2.set_ylim((np.array(ax_mpw.get_ylim()) - metrics_data[2].val) / metrics_data[2].val * 100)
    ax_mpw2.yaxis.set_major_formatter(PercentFormatter())
    ax_mpw.legend()
    ax_mpw.set_title(f'Main Peak Width Sensitivity to {var_name}')
    fig_main_peak_width.canvas.manager.set_window_title(f'Main Peak Width Sensitivity to {var_name}')
    fig_main_peak_width.tight_layout()


def get_dist_metrics(zs, z_dist, plot=False):
    """
    Get metrics characterizing the given z-vertex distribution.
    Height ratio: Ratio of the heights of the two peaks.
    Peak separation: Distance between the peaks.
    Main peak width: Width of the main peak.
    :param zs: cm z-vertex positions
    :param z_dist: z-vertex distribution
    :param plot: True to plot the distribution and fit
    :return: height_ratio, peak_separation, main_peak_width
    """
    bin_width = zs[1] - zs[0]

    max_hist = np.max(z_dist)
    z_max_hist = zs[np.argmax(z_dist)]

    # If z_max_hist > 0 find max on negative side, else find max on positive side
    if z_max_hist > 0:
        second_counts = z_dist[zs < 0]
        second_zs = zs[zs < 0]
    else:
        second_counts = z_dist[zs > 0]
        second_zs = zs[zs > 0]

    max_second = np.max(second_counts)
    z_max_second = second_zs[np.argmax(second_counts)]

    sigma_est = 50

    p0 = [max_hist, z_max_hist, sigma_est, max_second, z_max_second, sigma_est]

    popt, pcov = cf(double_gaus_bkg, zs, z_dist, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    opt_measures = [Measure(p, e) for p, e in zip(popt, perr)]

    height_ratio = opt_measures[0] / opt_measures[3]
    peak_separation = abs(opt_measures[1] - opt_measures[4])
    main_peak_width = opt_measures[2]

    if plot:
        fig, ax = plt.subplots()
        ax.bar(zs, z_dist, width=bin_width, label='MBD Vertex')
        ax.plot(zs, double_gaus_bkg(zs, *p0), color='gray', ls='--', alpha=0.6, label='Guess')
        ax.plot(zs, double_gaus_bkg(zs, *popt), color='r', label='Fit')
        ax.set_xlabel('z Vertex Position (cm)')
        text_str = (f'Height Ratio: {height_ratio}\nPeak Separation: {peak_separation} cm\n'
                    f'Main Peak Width: {main_peak_width} cm')
        print(f'p0: {p0}')
        opt_str = ', '.join([str(measure) for measure in opt_measures])
        print(f'Optimized: {opt_str}')
        print(text_str)
        ax.annotate(text_str, (0.02, 0.75), xycoords='axes fraction', bbox=dict(facecolor='wheat', alpha=0.5))
        fig.tight_layout()

    return height_ratio, peak_separation, main_peak_width


def get_head_on_dist_width(zs, z_dist, plot=False):
    """
    Get width of the main peak of the given z-vertex distribution.
    :param zs: cm z-vertex positions
    :param z_dist: z-vertex distribution
    :param plot: True to plot the distribution and fit
    :return: main_peak_width
    """
    bin_width = zs[1] - zs[0]

    max_hist = np.max(z_dist)
    z_max_hist = zs[np.argmax(z_dist)]

    sigma_est = 50

    p0 = [max_hist, z_max_hist, sigma_est]

    popt, pcov = cf(gaus_bkg, zs, z_dist, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    opt_measures = [Measure(p, e) for p, e in zip(popt, perr)]

    main_peak_width = opt_measures[2]

    if plot:
        fig, ax = plt.subplots()
        ax.bar(zs, z_dist, width=bin_width, label='MBD Vertex')
        ax.plot(zs, gaus_bkg(zs, *p0), color='gray', ls='--', alpha=0.6, label='Guess')
        ax.plot(zs, gaus_bkg(zs, *popt), color='r', label='Fit')
        ax.set_xlabel('z Vertex Position (cm)')
        text_str = f'Main Peak Width: {main_peak_width} cm'
        print(f'p0: {p0}')
        opt_str = ', '.join([str(measure) for measure in opt_measures])
        print(f'Optimized: {opt_str}')
        print(text_str)
        ax.annotate(text_str, (0.02, 0.75), xycoords='axes fraction', bbox=dict(facecolor='wheat', alpha=0.5))
        fig.tight_layout()

    return (main_peak_width,)


def get_mbd_z_dists(z_vertex_dist_root_path, first_dist=True, norms=None, abs_norm=False):
    vector.register_awkward()

    z_vertex_hists = []
    with (uproot.open(z_vertex_dist_root_path) as file):
        print(file.keys())
        for key in file.keys():
            hist = file[key]
            scan_axis = key.split('_')[1]
            scan_step = int(key.split('_')[-1].split(';')[0]) + 1
            z_vertex_hists.append({
                'scan_axis': scan_axis,
                'scan_step': scan_step,
                'centers': hist.axis().centers(),
                'counts': hist.counts(),
                'count_errs': hist.errors()
            })
            if norms is not None:
                norm = norms[scan_axis][scan_step] if not abs_norm else np.sum(z_vertex_hists[-1]['counts']) / norms[scan_axis][scan_step]
                z_vertex_hists[-1]['counts'] /= norm
                z_vertex_hists[-1]['count_errs'] /= norm
            if first_dist:
                break
    return z_vertex_hists


def plot_all_z_vertex_hists(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    for hist in z_vertex_hists:
        fig, ax = plt.subplots()
        bin_width = hist['centers'][1] - hist['centers'][0]
        ax.bar(hist['centers'], hist['counts'], width=bin_width)
        ax.set_title(f'{hist["scan_axis"]} Scan Step {hist["scan_step"]}')
        fig.tight_layout()
    plt.show()


def fit_sim_pars_to_vertex(x, zs, z_dist, collider_sim, scale_fit=False):
    if len(x) == 2:
        length_1, length_2 = x
        length_1 *= 1e4
        length_2 *= 1e4
        bunch_sigmas_1, bunch_sigmas_2 = collider_sim.get_beam_sigmas()
        bunch_sigmas_1[2] = length_1
        bunch_sigmas_2[2] = length_2
        collider_sim.set_bunch_sigmas(bunch_sigmas_1, bunch_sigmas_2)
    elif len(x) == 3:
        width, length, beta_star = x
        length *= 1e4
        bunch_sigmas = np.array([width, width, length])
        collider_sim.set_bunch_sigmas(bunch_sigmas, bunch_sigmas)
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
    elif len(x) == 4:
        length_1, length_2, beta_star_1, beta_star_2 = x
        length_1 *= 1e4
        length_2 *= 1e4
        bunch_sigmas_1 = np.array([150, 150, length_1])
        bunch_sigmas_2 = np.array([150, 150, length_2])
        collider_sim.set_bunch_sigmas(bunch_sigmas_1, bunch_sigmas_2)
        collider_sim.set_bunch_beta_stars(beta_star_1, beta_star_2)
    elif len(x) == 6:
        width_1, width_2, length_1, length_2, beta_star_1, beta_star_2 = x
        length_1 *= 1e4
        length_2 *= 1e4
        bunch_sigmas_1 = np.array([width_1, width_1, length_1])
        bunch_sigmas_2 = np.array([width_2, width_2, length_2])
        collider_sim.set_bunch_sigmas(bunch_sigmas_1, bunch_sigmas_2)
        collider_sim.set_bunch_beta_stars(beta_star_1, beta_star_2)

    collider_sim.run_sim()
    # collider_sim.run_sim_parallel()
    sim_zs, sim_z_dist = collider_sim.get_z_density_dist()
    if scale_fit:
        scale_0 = max(z_dist) / max(sim_z_dist)
        res = minimize(fit_amp, np.array([scale_0]), args=(sim_zs, sim_z_dist, z_dist, zs))
        scale = res.x
    else:
        scale = max(z_dist) / max(sim_z_dist)
    sim_interp = interp1d(sim_zs / 1e4, sim_z_dist * scale)
    residual = np.sum((z_dist - sim_interp(zs))**2)
    print(f'{x}: {residual}')
    return residual


def fit_beam_pars(x, collider_sim, angle1_y_0, offset1_y_0, beam_width_0, beta_star_0, beam1_length_0, beam2_length_0,
                  z_dist_data, zs_data):
    """
    Fit beam parameters
    :param x:
    :param collider_sim:
    :param angle1_y_0:
    :param offset1_y_0:
    # :param offset1_x0:
    :param beam_width_0:
    :param beta_star_0:
    :param beam1_length_0:
    :param beam2_length_0:
    :param z_dist_data:
    :param zs_data:
    :return:
    """
    angle_y = x[0] * angle1_y_0
    offset1_y = x[1] * offset1_y_0
    # offset1_x = x[2] * offset1_x0
    beam_width = x[2] * beam_width_0
    beta_star = x[3] * beta_star_0
    beam1_length = x[4] * beam1_length_0
    beam2_length = x[5] * beam2_length_0
    collider_sim.set_bunch_crossing(0, angle_y, 0, +0.0)
    bunch1_r = collider_sim.bunch1_r_original
    bunch2_r = collider_sim.bunch2_r_original
    # bunch1_r[1], bunch1_r[0] = offset1_y, offset1_x
    bunch1_r[1] = offset1_y
    collider_sim.set_bunch_rs(bunch1_r, bunch2_r)
    # bunch1_sigmas, bunch2_sigmas = collider_sim.get_beam_sigmas()
    bunch1_sigmas = np.array([beam_width, beam_width, beam1_length])
    bunch2_sigmas = np.array([beam_width, beam_width, beam2_length])
    collider_sim.set_bunch_sigmas(bunch1_sigmas, bunch2_sigmas)
    collider_sim.set_bunch_beta_stars(beta_star, beta_star)
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)
    collider_sim.run_sim()

    fit_amp_shift(collider_sim, z_dist_data, zs_data)
    zs, z_dist = collider_sim.get_z_density_dist()
    residual = np.sum((z_dist_data - interp1d(zs, z_dist)(zs_data)) ** 2)
    print(f'{x}: {residual:.2e}')
    if np.isnan(residual):
        print(zs)
        print(z_dist)
        print(f'angle_y_0: {angle1_y_0}, offset1_y_0: {offset1_y_0}, beam_width_0: {beam_width_0}, beta_star_0: {beta_star_0}, beam1_length_0: {beam1_length_0}, beam2_length_0: {beam2_length_0}')
        print(f'angle_y: {angle_y}, offset1_y: {offset1_y}, beam_width: {beam_width}, beta_star: {beta_star}, beam1_length: {beam1_length}, beam2_length: {beam2_length}')

    return residual


def fit_beam_pars1(x, collider_sim, angle1_x_0,
                   angle2_x_0, z_dist_data, zs_data, scan_orientation='Horizontal'):
    """
    Fit beam parameters
    :param x:
    :param collider_sim:
    :param angle1_x_0:  Horizontal angle of the first beam
    :param angle2_x_0:
    :param z_dist_data:
    :param zs_data:
    :param scan_orientation: 'Horizontal' or 'Vertical'
    :return:
    """
    angle2_x = x[0] * angle2_x_0
    if scan_orientation == 'Horizontal':
        collider_sim.set_bunch_crossing(angle1_x_0, 0.0, angle2_x, 0.0)
    elif scan_orientation == 'Vertical':
        collider_sim.set_bunch_crossing(0.0, angle1_x_0, 0.0, angle2_x)
    collider_sim.run_sim_parallel()

    fit_shift(collider_sim, z_dist_data, zs_data)
    zs, z_dist = collider_sim.get_z_density_dist()
    sim_interp_vals = interp1d(zs, z_dist)(zs_data)
    residual = np.sum(((z_dist_data - sim_interp_vals) / np.mean(z_dist_data)) ** 2)
    print(f'{x}: {residual:.2e}')
    if np.isnan(residual):
        print(zs)
        print(z_dist)
        # print(f'angle1_y_0: {angle1_y_0}, angle1_x_0: {angle1_x_0}, angle2_y_0: {angle2_y_0}, angle2_x_0: {angle2_x_0}')

    return residual


def fit_beam_pars2(x, collider_sim, angle1_x_0, angle2_x_0,
                   z_dist_data, zs_data):
    """
    Fit beam parameters
    :param x:
    :param collider_sim:
    :param angle1_x_0:
    :param angle2_x_0:
    :param z_dist_data:
    :param zs_data:
    :return:
    """
    angle1_x = x[0] * angle1_x_0
    # angle1_x = x[1] * angle1_y_0 + angle1_x_0
    angle2_x = x[1] * angle2_x_0
    # angle2_x = x[3] * angle2_y_0 + angle2_x_0
    collider_sim.set_bunch_crossing(angle1_x, 0.0, angle2_x, 0.0)
    # collider_sim.set_amplitude(1.0)
    # collider_sim.set_z_shift(0.0)
    collider_sim.run_sim_parallel()

    # fit_amp_shift(collider_sim, z_dist_data, zs_data)
    zs, z_dist = collider_sim.get_z_density_dist()
    sim_interp_vals = interp1d(zs, z_dist)(zs_data)
    residual = np.sum(((z_dist_data - sim_interp_vals) / np.mean(z_dist_data)) ** 2)
    print(f'{x}: {residual:.2e}')
    if np.isnan(residual):
        print(zs)
        print(z_dist)
        # print(f'angle1_y_0: {angle1_y_0}, angle1_x_0: {angle1_x_0}, angle2_y_0: {angle2_y_0}, angle2_x_0: {angle2_x_0}')

    return residual


def fit_beam_pars3(x, collider_sim, angle1_y_0, angle2_y_0, bw_0,
                  z_dist_data, zs_data):
    """
    Fit beam parameters
    :param x:
    :param collider_sim:
    :param angle1_y_0:
    :param angle2_y_0:
    :param bw_0:
    :param z_dist_data:
    :param zs_data:
    :return:
    """
    angle1_y = x[0] * angle1_y_0
    # angle1_x = x[1] * angle1_y_0 + angle1_x_0
    angle2_y = x[1] * angle2_y_0
    # angle2_x = x[3] * angle2_y_0 + angle2_x_0
    bw = x[2] * bw_0
    collider_sim.set_bunch_crossing(0.0, angle1_y, 0.0, angle2_y)
    collider_sim.set_bunch_sigmas(np.array([bw, bw]), np.array([bw, bw]))
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)
    collider_sim.run_sim_parallel()

    fit_amp_shift(collider_sim, z_dist_data, zs_data)
    zs, z_dist = collider_sim.get_z_density_dist()
    residual = np.sum((z_dist_data - interp1d(zs, z_dist)(zs_data)) ** 2)
    print(f'{x}: {residual:.2e}')
    if np.isnan(residual):
        print(zs)
        print(z_dist)
        # print(f'angle1_y_0: {angle1_y_0}, angle1_x_0: {angle1_x_0}, angle2_y_0: {angle2_y_0}, angle2_x_0: {angle2_x_0}')

    return residual


def fit_beam_pars4(x, collider_sim, angle1_y_0, angle2_y_0, bw_0,
                  z_dist_data, zs_data):
    """
    Fit beam parameters
    :param x:
    :param collider_sim:
    :param angle1_y_0:
    :param angle2_y_0:
    :param bw_0:
    :param z_dist_data:
    :param zs_data:
    :return:
    """
    angle1_y = x[0] * angle1_y_0
    # angle1_x = x[1] * angle1_y_0 + angle1_x_0
    angle2_y = x[1] * angle2_y_0
    # angle2_x = x[3] * angle2_y_0 + angle2_x_0
    bw = x[2] * bw_0
    collider_sim.set_bunch_crossing(0.0, angle1_y, 0.0, angle2_y)
    collider_sim.set_bunch_sigmas(np.array([bw, bw]), np.array([bw, bw]))
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)
    collider_sim.run_sim_parallel()

    # fit_amp_shift(collider_sim, z_dist_data, zs_data)
    zs, z_dist = collider_sim.get_z_density_dist()
    residual = get_dist_metric_residuals(zs, z_dist, z_dist_data, zs_data)
    print(f'{x}: {residual}')
    # if np.isnan(residual):
    #     print(zs)
    #     print(z_dist)
    #     # print(f'angle1_y_0: {angle1_y_0}, angle1_x_0: {angle1_x_0}, angle2_y_0: {angle2_y_0}, angle2_x_0: {angle2_x_0}')

    return residual.val


def get_dist_metric_residuals(zs, z_dist, z_dist_data, zs_data):
    """
    Get residual between simulated and measured distribution metrics
    """
    metrics_data = get_dist_metrics(zs_data, z_dist_data)
    metrics_sim = get_dist_metrics(zs, z_dist)
    residuals = [((m_val - s_val) / m_val)**2 for m_val, s_val in zip(metrics_data, metrics_sim)]
    print(f'Residuals: {residuals}')
    residual = np.mean(residuals)
    return residual


def fit_amp_shift(collider_sim, z_dist_data, zs_data):
    zs, z_dist = collider_sim.get_z_density_dist()
    scale = max(z_dist_data) / max(z_dist)
    z_max_sim = zs[np.argmax(z_dist)]
    z_max_hist = zs_data[np.argmax(z_dist_data)]
    shift = z_max_sim - z_max_hist  # microns

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, z_dist_data, zs_data),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)


def fit_shift(collider_sim, z_dist_data, zs_data):
    sim_zs, sim_z_dist = collider_sim.get_z_density_dist()
    sim_interp = interp1d(sim_zs, sim_z_dist)
    upper_bound = min(zs_data) - min(sim_zs) - 0.1
    lower_bound = max(zs_data) - max(sim_zs) + 0.1
    res = minimize(shift_residual, np.array([collider_sim.z_shift]),
                   args=(sim_interp, z_dist_data, zs_data),
                   bounds=((lower_bound, upper_bound),))
    collider_sim.set_z_shift(collider_sim.z_shift - res.x[0] * 1e4)


def amp_shift_residual(x, collider_sim, scale_0, shift_0, z_dist_data, zs_data):
    collider_sim.set_amplitude(x[0] * scale_0)
    collider_sim.set_z_shift(x[1] + shift_0)
    sim_zs, sim_z_dist = collider_sim.get_z_density_dist()
    sim_interp = interp1d(sim_zs, sim_z_dist)
    return np.sum((z_dist_data - sim_interp(zs_data)) ** 2)


def shift_residual(x, sim_interp, z_dist_data, zs_data):
    return np.sum((z_dist_data - sim_interp(zs_data - x)) ** 2)


def get_mbd_step_times(cad_data):
    """
    Use start and end time for each step to get the time each step took
    :param cad_data:
    :return:
    """
    mbd_step_times = {'Horizontal': {}, 'Vertical': {}}
    # Iterate over pandas dataframe rows
    for index, row in cad_data.iterrows():
        orientation = row['orientation']
        step = row['step']
        duration = row['duration']
        if orientation == 'Horizontal':
            mbd_step_times['Horizontal'][step] = duration
        elif orientation == 'Vertical':
            mbd_step_times['Vertical'][step] = duration
    return mbd_step_times


def get_cw_rates(cad_data):
    """
    Get the rate of the CW signal for each step
    :param cad_data:
    :return:
    """
    cw_rates = {'Horizontal': {}, 'Vertical': {}}
    # Iterate over pandas dataframe rows
    for index, row in cad_data.iterrows():
        orientation = row['orientation']
        step = row['step']
        rate = row['cw_rate']
        if orientation == 'Horizontal':
            cw_rates['Horizontal'][step] = rate
        elif orientation == 'Vertical':
            cw_rates['Vertical'][step] = rate

    return cw_rates


def gaus(x, a, x0, sigma):
    return a * np.exp(-0.5 * (x - x0) ** 2 / sigma ** 2)


def double_gaus_bkg(x, a1, x01, sigma1, a2, x02, sigma2):
    return gaus(x, a1, x01, sigma1) + gaus(x, a2, x02, sigma2)


def gaus_bkg(x, a, x0, sigma):
    return gaus(x, a, x0, sigma)


if __name__ == '__main__':
    main()
