# mike.laverick@auckland.ac.nz
# Specular point related functions
import math
import cmath
import numpy as np
import pyproj
from scipy import constants
from scipy.interpolate import interpn, interp2d
from scipy.signal import convolve2d
import geopy.distance as geo_dist
import time
import pymap3d as pm
from timeit import default_timer as timer
from load_files import get_local_dem, get_map_value
from cal_functions import db2power

# define WGS84
wgs84 = pyproj.Geod(ellps="WGS84")
abc = np.array([wgs84.a, wgs84.a, wgs84.b])

# define projections
ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
lla = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")

# define Fibonacci sequence here, once
k_fib = range(60)
term1_fib = np.ones(60) * ((1 + np.sqrt(5)) / 2)
term1_fib = term1_fib ** (np.array(k_fib) + 1)
term2_fib = np.ones(60) * ((1 - np.sqrt(5)) / 2)
term2_fib = term2_fib ** (np.array(k_fib) + 1)
fib_seq = (term1_fib - term2_fib) / np.sqrt(5)

# define chip length
# L1 GPS cps
chip_rate = 1.023e6
# chip length
l_chip = constants.c / chip_rate
# constant grid matrix 11*11
num_grid = 11


def nadir(m_ecef):
    """% This function, based on the WGS84 model, computes the ECEF coordinate
    of the nadir point (n) of a in-space point (m) on a WGS84 model"""
    # calculate nadir point. latitude degrees
    thetaD = math.asin(m_ecef[2] / np.linalg.norm(m_ecef, 2))
    cost2 = math.cos(thetaD) * math.cos(thetaD)
    # lat-dependent Earth radius
    r = wgs84.a * np.sqrt((1 - wgs84.es) / (1 - wgs84.es * cost2))
    # return nadir of m on WGS84
    return r * m_ecef / np.linalg.norm(m_ecef, 2)


def pdis(Tx, Rx, Sx):
    """% This function computes the distance from Tx to Sx to Rx
    % based on their coordinates given in ECEF"""
    return np.linalg.norm(Sx - Tx, 2) + np.linalg.norm(Rx - Sx, 2)


def ite(tx_pos_xyz, rx_pos_xyz):
    """% This function iteratively solve the positions of specular points
    % based on the WGS84 model
    % Inputs:
    % 1) Tx and Rx coordiates in ECEF XYZ
    % Ouputputs
    % 1) Sx coordinate in ECEF XYZ
    """
    s_t2r = np.linalg.norm(rx_pos_xyz - tx_pos_xyz, 2)

    # determine iteration
    N = next(x[0] for x in enumerate(fib_seq) if x[1] > s_t2r)

    # first iteration parameters
    a = rx_pos_xyz
    b = tx_pos_xyz

    for k in range(int(N) - 1):
        term1 = fib_seq[N - k - 1] / fib_seq[N - k + 1]
        term2 = fib_seq[N - k] / fib_seq[N - k + 1]
        m_lambda = a + term1 * (b - a)
        m_mu = a + term2 * (b - a)

        # nadir points
        s_lambda = nadir(m_lambda)
        s_mu = nadir(m_mu)
        # propagation distance
        f_lambda = np.linalg.norm(s_lambda - tx_pos_xyz, 2) + np.linalg.norm(
            rx_pos_xyz - s_lambda, 2
        )
        f_mu = np.linalg.norm(s_mu - tx_pos_xyz, 2) + np.linalg.norm(
            rx_pos_xyz - s_mu, 2
        )

        # f_lambda = pdis(tx_pos_xyz, rx_pos_xyz, s_lambda)
        # f_mu = pdis(tx_pos_xyz, rx_pos_xyz, s_mu)

        if f_lambda > f_mu:
            a = m_lambda
            b = b
        else:
            a = a
            b = m_mu
    # TODO: stlightly different from matlab
    return nadir(m_lambda)


def coarsetune(tx_pos_xyz, rx_pos_xyz):
    """% this function computes the SP on a pure WGS84 datum based on
    % Inputs:
    % 1) tx_pos_xyz: ECEF coordinate of the TX
    % 2) rx_pos_xyz: ECEF coordinate of the RX
    % Outputs:
    % 1) SP_xyz, SP_lla: ECEF and LLA coordinate of a SP on a pure WGS84 datum"""

    # find coarse SP using Fibonacci sequence
    # TODO: different from matlab from here, but the code looks fine
    SP_xyz_coarse = ite(tx_pos_xyz, rx_pos_xyz)
    SP_lla_coarse = pyproj.transform(ecef, lla, *SP_xyz_coarse, radians=False)
    # longitude adjustment
    if SP_lla_coarse[0] < 0:
        SP_lla_coarse[0] += 360
    elif SP_lla_coarse[0] > 360:
        SP_lla_coarse[0] -= 360
    # change order to lat, lon, alt
    SP_lla_coarse = SP_lla_coarse[1], SP_lla_coarse[0], SP_lla_coarse[2]
    return SP_xyz_coarse, SP_lla_coarse


def los_status(tx_pos_xyz, rx_pos_xyz):
    """% This function determines if the RT vector has intersections
    % with the WGS84 ellipsoid (LOS existence)
    % input: tx and rx locations in ECEF frame
    % output: flag to indicate if LOS exists between tx and rx"""

    # rx for NGRx, tx for satellite, given in ECEF-XYZ, pos vectors
    T_ecef = np.divide(tx_pos_xyz, abc)
    R_ecef = np.divide(rx_pos_xyz, abc)

    # unit vector of RT
    RT_unit = (T_ecef - R_ecef) / np.linalg.norm((T_ecef - R_ecef), 2)

    # determine if LOS exists (flag = 1)
    A = np.linalg.norm(RT_unit, 2) * np.linalg.norm(RT_unit, 2)
    B = 2 * np.dot(R_ecef, RT_unit)
    C = np.linalg.norm(R_ecef, 2) * np.linalg.norm(R_ecef, 2) - 1

    t1 = (B * B) - 4 * A * C
    if t1 < 0:
        return True
    t2 = (B * -1) + np.sqrt(t1) / (2 * A)
    t3 = (B * -1) - np.sqrt(t1) / (2 * A)
    if (t2 < 0) and (t3 < 0):
        return True
    return False


def finetune(tx_xyz, rx_xyz, sx_lla, L, model):
    """% This code fine-tunes the coordinate of the initial SP based on the DTU10
    % datum thorugh a number of iterative steps."""
    # find the pixel location
    # in Python sx_lla is (lon, lat, alt) not (lat, lon, alt)
    min_lat, max_lat = (sx_lla[0] - L / 2, sx_lla[0] + L / 2)
    min_lon, max_lon = (sx_lla[1] - L / 2, sx_lla[1] + L / 2)

    lat_bin = np.linspace(min_lat, max_lat, num_grid)
    lon_bin = np.linspace(min_lon, max_lon, num_grid)

    # Vectorise the 11*11 nested loop
    lat_bin_v = np.repeat(lat_bin, 11)
    lon_bin_v = np.tile(lon_bin, 11)
    ele = interpn(
        points=(model["lon"], model["lat"]),
        values=model["ele"],
        xi=(lon_bin_v, lat_bin_v),
        method="linear",
    )
    p_x, p_y, p_z = pyproj.transform(
        lla, ecef, *[lon_bin_v, lat_bin_v, ele], radians=False
    )
    p_xyz = np.array((p_x, p_y, p_z))
    p_xyz_t = p_xyz - tx_xyz.reshape(-1, 1)
    p_xyz_r = np.repeat(rx_xyz.reshape(-1, 1), len(p_x), axis=1) - p_xyz
    delay_chip = np.linalg.norm(p_xyz_t, 2, axis=0) + np.linalg.norm(p_xyz_r, 2, axis=0)
    ele = ele.reshape(11, -1)
    delay_chip = (delay_chip / l_chip).reshape(11, -1)

    # index of the pixel with minimal reflection path
    min_delay = np.min(delay_chip)
    m_i, n_i = np.where(delay_chip == (np.min(delay_chip)))

    # unpack arrays with [0] else they keep nesting
    sx_temp = [lat_bin[m_i][0], lon_bin[n_i][0], ele[m_i, n_i][0]]
    # TODO we calculate geodesic distance between points in metres - replaces m_lldist.m
    # this is between Matlab idx = 5,6 so extra "-1" due to python 0-indexing (Mat5,6 -> Py4,5)
    NN = int((num_grid - 1) / 2) - 1
    res = geo_dist.geodesic(
        (lat_bin[NN], lon_bin[NN]), (lat_bin[NN + 1], lon_bin[NN + 1])
    ).m
    return res, min_delay, sx_temp


def finetune_ocean(tx_pos_xyz, rx_pos_xyz, sp_lla_coarse, model, L, res_grid):
    """% This function fine tunes the SP coordiantes using a DTU10 datum
    % Inputs:
    % 1) TX and 2) RX coordinates in the form of ECEF-XYZ and
    % 3) SP coordinate in the form of LLA
    % 4) model: earth model - currently DTU10
    % 5) L: inital searching area in deg
    % 6) res_grid: targeted resolution of each grid when quitting the iteration
    % in metres
    % Output: return
    % 1) fine-tuned SP coordiantes in ECEF-XYZ, and
    % 2) local incidence angle"""

    # derive SP on the ocean surface
    res = 1000
    sp_temp = sp_lla_coarse
    while res > res_grid:
        res, _, sp_temp = finetune(
            tx_pos_xyz, rx_pos_xyz, sp_temp, L, model
        )
        # parameters for the next iteration - new searching area, new SP coordinate
        L = L * 2.0 / 11.0
    sp_temp1 = [sp_temp[1], sp_temp[0], sp_temp[2]]
    sx_xyz = pyproj.transform(lla, ecef, *sp_temp1, radians=False)
    return sx_xyz, sp_temp


def angles(local_dem, tx_pos_xyz, rx_pos_xyz):
    """% This function computes the local incidence and reflection angles of
    % the middle pixel in a 3 by 3 DEM pixel matrix
    % Inputs:
    % 1) lat,lon, and ele matrices of the 3*3 pixel matrix
    % 2) Tx and Rx coordinates ECEF(x,y,z)
    % Outputs:
    % 1) theta_i, phi_i: local incidence angle along elevation and azimuth
    % angles in degree
    % 2) theta_s, phi_s: local scattering (reflection) angles along elevation
    % and azimuth angles in degree"""

    # origin of the local enu frame
    s0 = [local_dem["lat"][1], local_dem["lon"][1], local_dem["ele"][1, 1]]

    # convert tx and rx to local ENU centred at s0
    ts = np.array([0, 0, 0]) - np.array(
        pm.ecef2enu(*tx_pos_xyz, *s0, deg=True)
    )  # default = wgs84
    sr = pm.ecef2enu(*rx_pos_xyz, *s0, deg=True)  # -[0,0,0]  == same...

    # convert s1-s4 to the same local ENU
    s1 = np.array(
        pm.geodetic2enu(
            local_dem["lat"][0], local_dem["lon"][1], local_dem["ele"][0, 1], *s0
        )
    )  # north
    s2 = np.array(
        pm.geodetic2enu(
            local_dem["lat"][2], local_dem["lon"][1], local_dem["ele"][2, 1], *s0
        )
    )  # south
    s3 = np.array(
        pm.geodetic2enu(
            local_dem["lat"][1], local_dem["lon"][2], local_dem["ele"][1, 0], *s0
        )
    )  # east
    s4 = np.array(
        pm.geodetic2enu(
            local_dem["lat"][1], local_dem["lon"][0], local_dem["ele"][1, 2], *s0
        )
    )  # west

    # local unit North, East and Up vectors
    unit_e = (s3 - s4) / np.linalg.norm(s3 - s4, 2)
    unit_n = (s1 - s2) / np.linalg.norm(s1 - s2, 2)
    unit_u = np.cross(unit_e, unit_n)

    p_1e, p_1n, p_1u = np.dot(ts, unit_e), np.dot(ts, unit_n), np.dot(ts, unit_u)
    p_2e, p_2n, p_2u = np.dot(sr, unit_e), np.dot(sr, unit_n), np.dot(sr, unit_u)

    term1, term2 = p_1e * p_1e + p_1n * p_1n, p_2e * p_2e + p_2n * p_2n
    theta_i = np.rad2deg(np.arctan(math.sqrt(term1) / abs(p_1u)))
    theta_s = np.rad2deg(np.arctan(math.sqrt(term2) / p_2u))
    phi_i = np.rad2deg(np.arctan(p_1n / p_1e))
    phi_s = np.rad2deg(np.arctan(p_2n / p_2e))
    return theta_i, theta_s, phi_i, phi_s



def sp_solver(tx_pos_xyz, rx_pos_xyz, dem, dtu10, dist_to_coast_nz):
    """% SP solver derives the coordinate(s) of the specular reflection (sx)
    % SP solver also reports the local incidence angle and the distance to coast in km where the SP occur
    % All variables are reported in ECEF
    % Inputs:
    % 1) tx and rx positions
    % 2) DEM models: dtu10, NZSRTM30, and land-ocean mask
    % Outputs:
    % 1) sx_pos_xyz: sx positions in ECEF
    % 2) in_angle_deg: local incidence angle at the specular reflection
    % 3) distance to coast in kilometer
    % 4) LOS flag"""

    # check if LOS exists
    LOS_flag = los_status(tx_pos_xyz, rx_pos_xyz)

    if not LOS_flag:
        # no sx if no LOS between rx and tx
        return [np.nan, np.nan, np.nan], np.nan, np.nan, np.nan, LOS_flag

    # derive SP coordinate on WGS84 and DTU10
    # TODO: result is slightly not the same with matlab code
    sx_xyz_coarse, sx_lla_coarse = coarsetune(tx_pos_xyz, rx_pos_xyz)

    # initial searching region in degrees
    L_ocean_deg = 1.0
    # converge criteria 0.01 meter
    res_ocean_meter = 0.01

    # derive local angles
    # TODO: result is slightly different from matlab code
    sx_pos_xyz, sx_pos_lla = finetune_ocean(
        tx_pos_xyz, rx_pos_xyz, sx_lla_coarse, dtu10, L_ocean_deg, res_ocean_meter
    )
    # sx_pos_xyz = pyproj.transform(lla, ecef, *sx_pos_lla, radians=False)
    # replaces get_map_value function
    # TODO Q: the resualt is not the same as get_map_value
    # dist = interpn(
    #     points=(dist_to_coast_nz["lon"], dist_to_coast_nz["lat"]),
    #     values=dist_to_coast_nz["ele"],
    #     xi=(sx_pos_lla[0], sx_pos_lla[1]),
    #     method="linear",
    # )
    dist = get_map_value(sx_pos_lla[0], sx_pos_lla[1], dist_to_coast_nz)

    local_dem = get_local_dem(sx_pos_lla, dem, dtu10, dist)
    theta_i, theta_s, phi_i, phi_s = angles(local_dem, tx_pos_xyz, rx_pos_xyz)

    if dist > 0:
        # local height of the SP = local_dem["ele"][1,1]
        # projection to local dem
        # sx_pos_xyz += (sx_pos_xyz / np.linalg.norm(sx_pos_xyz, 2)) * local_dem["ele"][
        #     1, 1
        # ]
        local_height = local_dem['ele']
        local_height = local_height[1, 1]       # local height of the SP

        # projection to local dem
        term1 = np.array(sx_xyz_coarse) / np.linalg.norm(sx_xyz_coarse)
        term2 = term1.dot(local_height)
        sx_pos_xyz = np.array(sx_xyz_coarse) + term2

    v_tsx = tx_pos_xyz - sx_pos_xyz
    unit_tsx = v_tsx / np.linalg.norm(v_tsx, 2)
    unit_sx = sx_pos_xyz / np.linalg.norm(sx_pos_xyz, 2)
    inc_angle_deg = np.rad2deg(np.arccos(np.dot(unit_tsx, unit_sx)))

    d_phi1 = np.sin(np.deg2rad(phi_s - phi_i + 180)) / np.cos(
        np.deg2rad(phi_s - phi_i + 180)
    )
    d_phi = np.rad2deg(np.arctan(d_phi1))
    d_snell_deg = abs(theta_i - theta_s) + abs(d_phi)

    return sx_pos_xyz, inc_angle_deg, d_snell_deg, dist, LOS_flag


def ecef2orf(P, V, S_ecef):
    """
    this function computes the elevation (theta) and azimuth (phi) angle of a point
    in the object's orbit reference frame (orf)
    Input (all vectors are row vectors):
    1) P & V: object's ECEF position (P) and velocity (V) vectors
    2) S_ecef: ECEF coordinate of the point to be computed (S_ecef)
    Output:
    1) theta_orf & phi_orf: polar and azimuth angles of S in SV's orf in degree
    2) S_orf: coordinate of S in orf S_orf
    """
    P = P.T
    V = V.T
    S_ecef = np.array(S_ecef).T
    u_ecef = S_ecef - P  # vector from P to S

    theta_e = 7.2921158553e-5  # earth rotation rate, rad/s
    W_e = np.array([0, 0, theta_e]).T  # earth rotation vector
    Vi = V + np.cross(W_e, P)  # SC ECEF inertial velocity vector

    # define orbit reference frame - unit vectors
    y_orf = np.cross(-1 * P, Vi) / np.linalg.norm(np.cross(-1 * P, Vi))
    z_orf = -1 * P / np.linalg.norm(P)
    x_orf = np.cross(y_orf, z_orf)

    # transformation matrix
    T_orf = np.array([x_orf.T, y_orf.T, z_orf.T])
    S_orf = np.dot(T_orf, u_ecef)

    # elevation and azimuth angles
    theta_orf = np.rad2deg(np.arccos(S_orf[2] / (np.linalg.norm(S_orf))))
    phi_orf = math.degrees(math.atan2(S_orf[1], S_orf[0]))

    if phi_orf < 0:
        phi_orf = 360 + phi_orf

    return theta_orf, phi_orf


def deg2rad(degrees):
    radians = degrees * math.pi / 180
    return radians


def ecef2brf(P, V, S_ecef, SC_att):
    """
    this function computes the elevation (theta) and azimuth (phi) angle of a
    ecef vector in the objects's body reference frame (brf)
    Input:
    1) P, V: object's ecef position vector
    2) SC_att: object's attitude (Euler angle) in the sequence of
    roll, pitch, yaw, in degrees
    3) S_ecef: ecef coordinate of the point to be computed
    Output:
    1) theta_brf: elevation angle of S in the SC's brf in degree
    2) phi_brf: azimuth angle of S in the SC's brf in degree
    """
    P = P.T
    V = V.T
    S_ecef = np.array(S_ecef).T

    phi = deg2rad(SC_att[0])  # roll
    theta = deg2rad(SC_att[1])  # pitch
    psi = deg2rad(SC_att[2])  # yaw

    # TODO: the result is different from the matlab code, since the input is slightly different
    u_ecef = S_ecef - P  # vector from P to S

    # define heading frame - unit vectors
    y_hrf = np.cross(-1 * P, V) / np.linalg.norm(np.cross(-1 * P, V))
    z_hrf = -1 * P / np.linalg.norm(-1 * P)
    x_hrf = np.cross(y_hrf, z_hrf)

    T_hrf = np.array([x_hrf.T, y_hrf.T, z_hrf.T])

    # S in hrf  TODO: the result is different from the matlab code since u_ecef is slightly different
    S_hrf = np.dot(T_hrf, u_ecef)

    # construct aircraft's attitude matrix
    Rx_phi = np.array([[1, 0, 0],
                       [0, math.cos(phi), math.sin(phi)],
                       [0, -1 * math.sin(phi), math.cos(phi)]])

    Ry_theta = np.array([[math.cos(theta), 0, -1 * math.sin(theta)],
                         [0, 1, 0],
                         [math.sin(theta), 0, math.cos(theta)]])

    Rz_psi = np.array([[math.cos(psi), math.sin(psi), 0],
                      [-1 * math.sin(psi), math.cos(psi), 0],
                      [0, 0, 1]])

    R = Ry_theta.dot(Rx_phi).dot(Rz_psi)  # transformation matrix

    S_brf = np.dot(R, S_hrf.T)

    # TODO: the result slightly different from the matlab code
    theta_brf = np.rad2deg(np.arccos(S_brf[2] / (np.linalg.norm(S_brf))))
    phi_brf = math.degrees(math.atan2(S_brf[1], S_brf[0]))

    if phi_brf < 0:
        phi_brf = 360 + phi_brf

    return theta_brf, phi_brf


def cart2sph(x, y, z):
    xy = x**2 + y**2
    r = math.sqrt(xy + z**2)
    theta = math.atan2(z, math.sqrt(xy))
    phi = math.atan2(y, x)
    return phi, theta, r   # for consistency with MATLAB


def ecef2enuf(P, S_ecef):
    """
    this function computes the elevation (theta) and azimuth (phi) angle of a point
    in the object's ENU frame (enuf)
    input:
    1) P: object's ECEF position vector
    2) S_ecef: ECEF coordinate of the point to be computed
    output:
    1) theta_enuf & phi_enuf: elevation and azimuth angles of S in enuf in degree
    """
    # P = [-4593021.50000000,	608280.500000000,	-4370184.50000000]
    # S_ecef = [-4590047.30433596,	610685.547457113,	-4371634.83935421]

    lon, lat, alt = pyproj.transform(ecef, lla, *P, radians=False)
    # TODO: The difference is gradually magnified.
    S_east, S_north, S_up = pm.ecef2enu(*S_ecef, lat, lon, alt, deg=True)
    phi_enuf, theta_enuf1, _ = cart2sph(S_east, S_north, S_up)

    phi_enuf = np.rad2deg(phi_enuf)
    theta_enuf1 = np.rad2deg(theta_enuf1)

    theta_enuf = 90 - theta_enuf1

    return theta_enuf, phi_enuf


def sp_related(tx, rx, sx_pos_xyz, SV_eirp_LUT):
    """
    this function computes the sp-related variables, including angles in
    various coordinate frames, ranges, EIRP, nadir antenna gain etc
    Inputs:
    1) tx, rx: tx and rx structures
    2) sx_pos_xyz: sx ECEF position vector
    3) SV_PRN_LUT,SV_eirp_LUT: look-up table between SV number and PRN
    Outputs:
    1) sp_angle_body: sp angle in body frame, az and theta
    2) sp_angle_enu: sp angle in ENU frame, az and theta
    3) theta_gps: GPS off boresight angle
    4) range: tx to sx range, and rx to sx range
    5) gps_rad: EIRP, tx power
    """
    # sparse structres
    tx_pos_xyz = tx['tx_pos_xyz']
    tx_vel_xyz = tx['tx_vel_xyz']
    sv_num = tx['sv_num']

    rx_pos_xyz = rx['rx_pos_xyz']
    rx_vel_xyz = rx['rx_vel_xyz']
    rx_att = rx['rx_attitude']

    # compute angles
    theta_gps, _ = ecef2orf(tx_pos_xyz, tx_vel_xyz, sx_pos_xyz)

    sp_theta_body, sp_az_body = ecef2brf(rx_pos_xyz, rx_vel_xyz, sx_pos_xyz, rx_att)
    sp_theta_enu, sp_az_enu = ecef2enuf(rx_pos_xyz, sx_pos_xyz)

    sp_angle_body = [sp_theta_body, sp_az_body]
    sp_angle_enu = [sp_theta_enu, sp_az_enu]

    # compute ranges
    R_tsx = np.linalg.norm(sx_pos_xyz - tx_pos_xyz)        # range from tx to sx
    R_rsx = np.linalg.norm(sx_pos_xyz - rx_pos_xyz)        # range from rx to sx

    range = [R_tsx, R_rsx]

    # 0-based index
    # compute gps radiation properties
    j = SV_eirp_LUT[:, 0] == sv_num             # index of SV number in eirp LUT

    gps_pow_dbw = SV_eirp_LUT[j, 2]            # gps power in dBw

    # coefficients to compute gps antenna gain
    a = SV_eirp_LUT[j, 3]
    b = SV_eirp_LUT[j, 4]
    c = SV_eirp_LUT[j, 5]
    d = SV_eirp_LUT[j, 6]
    e = SV_eirp_LUT[j, 7]
    f = SV_eirp_LUT[j, 8]

    # fitting antenna gain
    gps_gain_dbi = a * theta_gps ** 5 + b * theta_gps ** 4 + c * theta_gps ** 3 + d * theta_gps ** 2 + e * theta_gps + f

    # compute static gps eirp
    stat_eirp_dbw = gps_pow_dbw + gps_gain_dbi   # static eirp in dbw
    stat_eirp_watt = 10 ** (stat_eirp_dbw / 10)     # static eirp in linear watts

    gps_rad = [gps_pow_dbw[0], gps_gain_dbi[0], stat_eirp_watt[0]]

    # compute angles in nadir antenna frame and rx gain
    sp_theta_ant = sp_theta_body
    sp_az_ant = sp_az_body + 180

    if sp_az_ant > 360:
        sp_az_ant = sp_az_ant - 360

    sp_angle_ant = [sp_theta_ant, sp_az_ant]

    return sp_angle_body, sp_angle_enu, sp_angle_ant, theta_gps, range, gps_rad


def get_sx_rx_gain(sp_angle_ant, nadir_pattern):
    """
    define azimuth and elevation angle in the antenna frame

    Parameters
    ----------
    sp_angle_ant
    nadir_pattern

    Returns
    -------
    """
    res = 0.1  # resolution in degrees
    az_deg = np.arange(0, 360, res)
    el_deg = np.arange(120, 0, -1 * res)

    lhcp_gain_pattern = nadir_pattern['LHCP']
    rhcp_gain_pattern = nadir_pattern['RHCP']

    sp_theta_ant = sp_angle_ant[0]
    sp_az_ant = sp_angle_ant[1]

    az_index = np.argmin(np.abs(sp_az_ant - az_deg))
    el_index = np.argmin(np.abs(sp_theta_ant - el_deg))

    lhcp_gain_dbi = lhcp_gain_pattern[el_index, az_index]
    rhcp_gain_dbi = rhcp_gain_pattern[el_index, az_index]

    sx_rx_gain = [lhcp_gain_dbi, rhcp_gain_dbi]

    return sx_rx_gain


def get_amb_fun(dtau_s, dfreq_Hz, tau_c, Ti):
    """
    this function computes the ambiguity function
    inputs
    1) tau_s: delay in seconds
    2) freq_Hz: Doppler in Hz
    3) tau_c: chipping period in second, 1/chip_rate
    4) Ti: coherent integration time in seconds
    output
    1) chi: ambiguity function, product of Lambda and S
    """
    det = tau_c * (1 + tau_c / Ti)   # discriminant for computing Lambda

    Lambda = np.full_like(dtau_s, np.nan)

    #compute Lambda - delay
    Lambda[np.abs(dtau_s) <= det] = (1 - np.abs(dtau_s) / tau_c)[np.abs(dtau_s) <= det]
    Lambda[np.abs(dtau_s) > det] = (-tau_c / Ti)

    # compute S - Doppler
    S1 = math.pi * dfreq_Hz * Ti

    S = np.full_like(S1, np.nan, dtype=complex)

    S[S1 == 0] = 1

    term1 = np.sin(S1[S1 != 0]) / S1[S1 !=0]
    term2 = np.exp(-1j * S1[S1 != 0])
    S[S1 != 0] = term1 * term2

    # compute complex chi
    chi = Lambda * S
    return chi


def get_chi2(num_delay_bins, num_doppler_bins):
    chip_rate = 1.023e6
    tau_c = 1 / chip_rate
    T_coh = 1 / 1000

    delay_res = 0.25
    doppler_res = 500

    delay_center_bin = 20  # 0-based index
    doppler_center_bin = 2  # 0-based index

    # chi = np.zeros([num_delay_bins, num_doppler_bins])

    def ix_func(i, j):
        dtau = (i - delay_center_bin) * delay_res * tau_c
        dfreq = (j - doppler_center_bin) * doppler_res
        # compute complex AF value at each delay-doppler bin
        return get_amb_fun(dtau, dfreq, tau_c, T_coh)

    chi = np.fromfunction(ix_func, (num_delay_bins, num_doppler_bins))  # 10 times faster than for loop

    chi_mag = np.abs(chi)           # magnitude
    chi2 = np.square(chi_mag)       # chi_square

    return chi2


def meter2chips(x):
    """
    this function converts from meters to chips
    input: x - distance in meters
    output: y - distance in chips
    """
    # define constants
    c = 299792458  # light speed metre per second
    chip_rate = 1.023e6  # L1 GPS chip-per-second, code modulation frequency
    tau_c = 1 / chip_rate  # C/A code chiping period
    l_chip = c * tau_c  # chip length
    y = x / l_chip
    return y



def delay_correction(delay_chips_in, P):
    # this function correct the input code phase to a value between 0 and
    # a defined value P, P = 1023 for GPS L1 and P = 4092 for GAL E1
    temp = delay_chips_in

    if temp < 0:
        while temp < 0:
            temp = temp + P
    elif temp > 1023:
        while temp > 1023:
            temp = temp - P

    delay_chips_out = temp

    return delay_chips_out


def deldop(tx_pos_xyz, rx_pos_xyz, tx_vel_xyz, rx_vel_xyz, p_xyz):
    """
    # This function computes absolute delay and doppler values for a given
    # pixel whose coordinate is <lat,lon,ele>
    # The ECEF position and velocity vectors of tx and rx are also required
    # Inputs:
    # 1) tx_xyz, rx_xyz: ecef position of tx, rx
    # 2) tx_vel, rx_vel: ecef velocity of tx, rx
    # 3) p_xyz of the pixel under computation
    # Outputs:
    # 1) delay_chips: delay measured in chips
    # 2) doppler_Hz: doppler measured in Hz
    # 3) add_delay_chips: additional delay measured in chips
    """
    # common parameters
    c = 299792458  # light speed metre per second
    fc = 1575.42e6  # L1 carrier frequency in Hz
    _lambda = c / fc  # wavelength

    V_tp = tx_pos_xyz - p_xyz
    R_tp = np.linalg.norm(V_tp)
    V_tp_unit = V_tp / R_tp
    V_rp = rx_pos_xyz - p_xyz
    R_rp = np.linalg.norm(V_rp)
    V_rp_unit = V_rp / R_rp
    V_tr = tx_pos_xyz - rx_pos_xyz
    R_tr = np.linalg.norm(V_tr)

    delay = R_tp + R_rp
    delay_chips = meter2chips(delay)
    add_delay_chips = meter2chips(R_tp + R_rp - R_tr)

    # absolute Doppler frequency in Hz
    term1 = np.dot(tx_vel_xyz, V_tp_unit)
    term2 = np.dot(rx_vel_xyz, V_rp_unit)

    # TODO: slightly different from the matlab version from V_rp_unit
    doppler_hz = -1 * (term1 + term2) / _lambda  # Doppler in Hz

    return delay_chips, doppler_hz, add_delay_chips


def get_specular_bin(tx, rx, sx, ddm):
    """
    this function derives the
    1) precise SP bin location in the DDM - 0-indexed
    2) confidence flag for the computed SP and also
    3) zenith code phase directly tracked by the NGRx
    """
    c = 299792458

    tx_pos_xyz = tx['tx_pos_xyz']
    tx_vel_xyz = tx['tx_vel_xyz']

    rx_pos_xyz = rx['rx_pos_xyz']
    rx_vel_xyz = rx['rx_vel_xyz']
    rx_clk_drift = rx['rx_clk_drift']

    sx_pos_xyz = sx['sx_pos_xyz']
    sx_d_snell = sx['sx_d_snell']
    dist_to_coast = sx['dist_to_coast']

    raw_counts = ddm['raw_counts']
    delay_resolution = ddm['delay_resolution']
    delay_center_chips = ddm['delay_center_chips']
    delay_center_bin = ddm['delay_center_bin']

    doppler_resolution = ddm['doppler_resolution']
    doppler_center_hz = ddm['doppler_center_hz']
    doppler_center_bin = ddm['doppler_center_bin']

    add_range_to_sp = ddm['add_range_to_sp']
    snr_db = ddm['snr_db']

    # derive zenith code phase
    add_range_to_sp_chips = meter2chips(add_range_to_sp)
    zenith_code_phase1 = delay_center_chips + add_range_to_sp_chips
    zenith_code_phase = delay_correction(zenith_code_phase1, 1023)

    # derive precise SP bin location
    _, pixel_doppler_hz, pixel_add_range_to_sp_chips = deldop(tx_pos_xyz, rx_pos_xyz, tx_vel_xyz, rx_vel_xyz, sx_pos_xyz)

    delay_error = add_range_to_sp_chips - pixel_add_range_to_sp_chips
    sp_delay_row = delay_center_bin + delay_error / delay_resolution    # cygnss is using "-"

    doppler_clk = rx_clk_drift / c
    pixel_doppler_hz = pixel_doppler_hz + doppler_clk

    doppler_error = doppler_center_hz - pixel_doppler_hz  # slightly different from the matlab version < 0.1 / e2
    sp_dopp_col = doppler_center_bin - doppler_error / doppler_resolution

    sp_delay_error = delay_error
    sp_dopp_error = doppler_error

    # derive confidence flag
    if dist_to_coast < 0:
        confidence_flag = 3   # confident on the ocean surface
    else:
        delay_max_bin, doppler_max_bin = np.unravel_index(raw_counts.argmax(), raw_counts.shape)

        # delay_max = delay_center_chips + (delay_max_bin - delay_center_bin - 1) * delay_resolution  # diff < 1 / e-2
        delay_max = delay_center_chips + (delay_max_bin - delay_center_bin) * delay_resolution  # diff < 1 / e-2
        delay_sp = zenith_code_phase1 - pixel_add_range_to_sp_chips
        delay_diff = abs(delay_sp - delay_max)

        # doppler_max = doppler_center_hz + (doppler_max_bin - doppler_center_bin - 1) * doppler_resolution
        doppler_max = doppler_center_hz + (doppler_max_bin - doppler_center_bin) * doppler_resolution
        doppler_diff = abs(pixel_doppler_hz - doppler_max)

        delay_doppler_snell = (delay_diff < 2.5) and (doppler_diff < 200) and (sx_d_snell < 2)

        if snr_db >= 2 and not delay_doppler_snell:
            confidence_flag = 0
        elif snr_db < 2 and not delay_doppler_snell:
            confidence_flag = 1
        elif snr_db < 2 and delay_doppler_snell:
            confidence_flag = 2
        elif snr_db >= 2 and delay_doppler_snell:
            confidence_flag = 3
        else:
            confidence_flag = np.nan

    specular_bin = [sp_delay_row, sp_dopp_col, sp_delay_error, sp_dopp_error]

    return specular_bin, zenith_code_phase, confidence_flag


def get_ddm_Aeff(tx, rx, sx, local_dem, phy_ele_size, chi2):
    """
    this function computes the effective scattering area at the given surface
    Inputs:
    1) tx, rx: tx and rx structures
    2) sx_pos_xyz: ecef position of specular points
    3) ddm: ddm structure
    4) local_dem: local region centred at sx
    5) T_coh: coherent integration duration
    Output:
    1) A_eff: effective scattering area
    2) sp_delay_bin,sp_doppler_bin: floating specular bin
    """
    delay_res = 0.25
    doppler_res = 500

    # sparse structures
    tx_pos_xyz = tx['tx_pos_xyz']
    tx_vel_xyz = tx['tx_vel_xyz']

    rx_pos_xyz = rx['rx_pos_xyz']
    rx_vel_xyz = rx['rx_vel_xyz']

    sx_pos_xyz = sx['sx_pos_xyz']
    # ecef2ella
    lon, lat, alt = pyproj.transform(ecef, lla, *sx_pos_xyz, radians=False)
    sx_pos_lla = [lat, lon, alt]

    sx_delay_bin = sx['sx_delay_bin'] + 1  # need to fix all 0-indexed bin to 1-indexed
    sx_doppler_bin = sx['sx_doppler_bin'] + 1

    # sparse local_dem structure
    local_lat = local_dem['lat']
    local_lon = local_dem['lon']
    local_ele = local_dem['ele']

    num_grids = len(local_lat)

    # get coarsen local_dem
    sample_rate = 30

    lat_coarse = local_lat[::sample_rate]
    lon_coarse = local_lon[::sample_rate]
    ele_coarse = local_ele[::sample_rate, ::sample_rate]

    num_grids_coarse = len(lat_coarse)

    # get delay-doppler map over the surface
    delay_coarse = np.zeros((num_grids_coarse, num_grids_coarse))
    doppler_coarse = np.zeros((num_grids_coarse, num_grids_coarse))

    delay_chips_sx, doppler_Hz_sx, _ = deldop(tx_pos_xyz, rx_pos_xyz, tx_vel_xyz, rx_vel_xyz, sx_pos_xyz)  #

    for m in range(num_grids_coarse):  # 0.4s
        for n in range(num_grids_coarse):
            # ecef2ella
            p_pos_lla1 = [lon_coarse[n], lat_coarse[m], ele_coarse[m, n]]
            p_pos_xyz1 = pyproj.transform(lla, ecef, *p_pos_lla1, radians=False)

            delay_p1, doppler_p1, _ = deldop(tx_pos_xyz, rx_pos_xyz, tx_vel_xyz, rx_vel_xyz, p_pos_xyz1)

            delay_coarse[m, n] = delay_p1 - delay_chips_sx
            doppler_coarse[m, n] = doppler_p1 - doppler_Hz_sx  # diff < 1 / e3

    # interpolate to 30-m resolution, TODO: slighly different from matlab, diff < 0.1 / e2
    # xx, yy = np.meshgrid(local_lon, local_lat)
    # points_out = np.array((xx.ravel(), yy.ravel())).T
    # delay_chips = interpn(points=(lon_coarse, lat_coarse),
    #                       values=delay_coarse,
    #                       xi=points_out,
    #                       method='cubic').reshape(len(local_lon), len(local_lat))
    # doppler_Hz = interpn(points=(lon_coarse, lat_coarse),
    #                      values=doppler_coarse,
    #                      xi=points_out,
    #                      method='cubic').reshape(len(local_lon), len(local_lat))
    # 141.76s
    interp_delay_chips = interp2d(lon_coarse, lat_coarse, delay_coarse, kind='cubic')
    delay_chips = interp_delay_chips(local_lon, local_lat)
    delay_chips = np.flipud(delay_chips)
    interp_doppler_Hz = interp2d(lon_coarse, lat_coarse, doppler_coarse, kind='cubic')
    doppler_Hz = interp_doppler_Hz(local_lon, local_lat)
    doppler_Hz = np.flipud(doppler_Hz)
    # 0.01s

    # get physical size
    sx_pos_lat = sx_pos_lla[0]
    idx_lat = np.argmin(np.abs(phy_ele_size[:, 0] - sx_pos_lat))
    dA = phy_ele_size[int(idx_lat - np.floor(num_grids / 2)): int(idx_lat + np.floor(num_grids / 2) + 1), 1]  # 0-based index

    dA = np.tile(dA, (num_grids, 1))

    # construct physical size DDM
    A_phy = np.zeros((5, 40))

    # t0 = timer()
    # bin to physical size DDM
    for m in range(num_grids):  # 0.9s -> 0.6s by using continue
        for n in range(num_grids):
            delay_bin_idx1 = int(np.floor(-1 * delay_chips[n, m] / delay_res + sx_delay_bin) - 1)  # 0-based index
            if delay_bin_idx1 < 0 or delay_bin_idx1 > 39:
                continue
            doppler_bin_idx1 = int(np.floor(doppler_Hz[n, m] / doppler_res + sx_doppler_bin) - 1)  # 0-based index
            if doppler_bin_idx1 < 0 or doppler_bin_idx1 > 4:
                continue

            # 0-based index
            # if (delay_bin_idx1 >= 0) and (delay_bin_idx1 <= 39) and (doppler_bin_idx1 >= 0) and (doppler_bin_idx1 <= 4):
            temp = A_phy[doppler_bin_idx1, delay_bin_idx1]
            temp = temp + dA[n, m]
            A_phy[doppler_bin_idx1, delay_bin_idx1] = temp


    # print('t0 --- ', timer() - t0)

    # it = np.nditer(delay_chips, flags=['multi_index'])  #1.2s
    # for _ in it:
    #     i, j = it.multi_index
    #     delay_bin_idx1 = int(np.floor(-1 * delay_chips[j, i] / delay_res + sx_delay_bin) - 1)  # 0-based index
    #     doppler_bin_idx1 = int(np.floor(doppler_Hz[j, i] / doppler_res + sx_doppler_bin) - 1)  # 0-based index

    #     # 0-based index
    #     if (delay_bin_idx1 >= 0) and (delay_bin_idx1 <= 39) and (doppler_bin_idx1 >= 0) and (doppler_bin_idx1 <= 4):
    #         temp = A_phy[doppler_bin_idx1, delay_bin_idx1]
    #         temp = temp + dA[j, i]
    #         A_phy[doppler_bin_idx1, delay_bin_idx1] = temp

    # convolution to A_eff
    A_eff1 = convolve2d(A_phy, chi2.T)
    A_eff = A_eff1[2:7, 19: 59]  # cut suitable size for A_eff, 0-based index
    A_eff_all = A_eff1

    return A_eff, A_eff_all


def ddm_brcs(power_analog, eirp_watt, rx_gain_db_i, TSx, RSx):
    """
    this function computes bistatic radar cross section (BRCS) according to
    bistatic radar equation based on the inputs as below
    inputs:
    1) power_analog: L1a product in watts
    2) eirp_watt, rx_gain_db_i: gps eirp in watts and rx antenna gain in dBi
    3) TSx, RSx: Tx to Sx and Rx to Sx ranges
    outputs:
    1) brcs: bistatic RCS
    """
    # define constants
    c = 299792458  # light speed, m/s
    f = 1575.42e6  # GPS L1 band, Hz
    _lambda = c / f  # wavelength, m
    _lambda2 = _lambda * _lambda

    rx_gain = db2power(rx_gain_db_i)  # linear rx gain

    term1 = eirp_watt * rx_gain

    term2_1 = TSx * RSx
    term2 = 1 / (term2_1 * term2_1)

    power_factor = _lambda2 * term1 * term2 / pow(4 * math.pi, 3)

    brcs = power_analog / power_factor

    return brcs


def get_ddm_nbrcs2(brcs, A_eff, sx_bin, flag):
    """
    this function computes two versions of NBRCS
    floating SP bin location is not considered
    TES and LES are not included in this version
    """
    sx_delay_bin = math.floor(sx_bin[0]) if not np.isnan(sx_bin[0]) else 0  # 0-based index
    sx_doppler_bin = math.floor(sx_bin[1]) if not np.isnan(sx_bin[1]) else 0  # 0-based index

    # case 1: small scattering area
    # case 2: large scattering area

    nbrcs = np.nan
    nbrcs_scatter = np.nan

    if flag == 1:
        if 1 < sx_delay_bin <= 39 and 0 < sx_doppler_bin < 4:
            sx_doppler_bin_range = list(range(sx_doppler_bin - 1, sx_doppler_bin + 2))
            sx_delay_bin_range = list(range(sx_delay_bin - 2, sx_delay_bin + 1))

            brcs_ddma = brcs[sx_delay_bin_range, :][:, sx_doppler_bin_range]
            A_eff_ddma = A_eff[sx_delay_bin_range, :][:, sx_doppler_bin_range]

            brcs_total = np.sum(brcs_ddma)
            A_eff_total = np.sum(A_eff_ddma)

            nbrcs = brcs_total / A_eff_total
            nbrcs_scatter = A_eff_total
        else:
            nbrcs = np.nan
            nbrcs_scatter = np.nan

    if flag == 2:
        if 29 < sx_delay_bin <= 39 and 0 < sx_doppler_bin < 4:
            sx_doppler_bin_range = list(range(sx_doppler_bin - 1, sx_doppler_bin + 2))
            sx_delay_bin_range = list(range(sx_delay_bin - 29, sx_delay_bin + 1))

            brcs_ddma = brcs[sx_delay_bin_range, :][:, sx_doppler_bin_range]
            A_eff_ddma = A_eff[sx_delay_bin_range, :][:, sx_doppler_bin_range]

            brcs_total = np.sum(brcs_ddma)
            A_eff_total = np.sum(A_eff_ddma)

            nbrcs = brcs_total / A_eff_total
            nbrcs_scatter = A_eff_total

        elif 0 < sx_delay_bin <= 29 and 0 < sx_doppler_bin < 4:
            sx_doppler_bin_range = list(range(sx_doppler_bin - 1, sx_doppler_bin + 2))
            sx_delay_bin_range = list(range(0, sx_delay_bin + 1))

            brcs_ddma = brcs[sx_delay_bin_range, :][:, sx_doppler_bin_range]
            A_eff_ddma = A_eff[sx_delay_bin_range, :][:, sx_doppler_bin_range]

            brcs_total = np.sum(brcs_ddma)
            A_eff_total = np.sum(A_eff_ddma)

            nbrcs = brcs_total / A_eff_total
            nbrcs_scatter = A_eff_total

    return nbrcs, nbrcs_scatter


def ddm_refl(power_analog, eirp_watt, rx_gain_db_i, R_tsx, R_rsx):
    """
    this function computes the land reflectivity
    inputs
    1)power_analog: L1a product, DDM power in watt
    2)eirp_watt: transmitter eirp in watt
    3)rx_gain_db_i: receiver antenna gain in the direction of SP, in dBi
    4)R_tsx, R_rsx: tx to sp range and rx to sp range, in meters
    outputs
    1)reflectivity
    2)reflectivity peak
    """
    # define constants
    c = 299792458  # speed of light, meter per second
    freq = 1575.42e6  # GPS L1 operating frequency, Hz
    _lambda = c / freq  # wavelength, meter
    _lambda2 = _lambda * _lambda

    sp_rx_gain_pow = db2power(rx_gain_db_i)  # convert antenna gain to linear form

    range = R_tsx + R_rsx

    term1 = np.power(4 * math.pi * range, 2)
    term2 = eirp_watt * sp_rx_gain_pow * _lambda2
    term3 = term1 / term2

    reflectivity = power_analog * term3
    reflectivity_peak = np.amax(reflectivity)

    return reflectivity, reflectivity_peak

def get_fresnel(tx_pos_xyz,rx_pos_xyz,sx_pos_xyz,dist_to_coast,inc_angle,ddm_ant):
    """
    this function derives Fresnel dimensions based on the Tx, Rx and Sx positions.
    Fresnel dimension is computed only the DDM is classified as coherent reflection.
    """
    # define constants
    eps_ocean = 74.62 + 51.92j  # complex permittivity of ocean
    fc = 1575.42e6  # operating frequency
    c = 299792458  # speed of light
    _lambda = c / fc  # wavelength

    # compute dimensions
    R_tsp = np.linalg.norm(np.array(tx_pos_xyz) - np.array(sx_pos_xyz))
    R_rsp = np.linalg.norm(np.array(rx_pos_xyz) - np.array(sx_pos_xyz))

    term1 = R_tsp * R_rsp
    term2 = R_tsp + R_rsp

    # semi axis
    a = math.sqrt(_lambda * term1 / term2)  # major semi
    b = a / math.cos(math.radians(inc_angle))  # minor semi

    # compute orientation relative to North
    lon, lat, alt = pyproj.transform(ecef, lla, *sx_pos_xyz, radians=False)
    sx_lla = [lat, lon, alt]

    tx_e, tx_n, _ = pm.ecef2enu(*tx_pos_xyz, *sx_lla, deg=True)
    rx_e, rx_n, _ = pm.ecef2enu(*rx_pos_xyz, *sx_lla, deg=True)

    tx_en = np.array([tx_e, tx_n])
    rx_en = np.array([rx_e, rx_n])

    vector_tr = rx_en - tx_en
    unit_north = [0, 1]

    term3 = np.dot(vector_tr, unit_north)
    term4 = np.linalg.norm(vector_tr) * np.linalg.norm(unit_north)

    theta = math.degrees(math.acos(term3 / term4))

    fresnel_axis = [2 * a, 2 * b]
    fresnel_orientation = theta

    # fresenel coefficient only compute for ocean SPs
    fresnel_coeff = np.nan

    if dist_to_coast <= 0:

        sint = math.sin(math.radians(inc_angle))
        cost = math.cos(math.radians(inc_angle))

        temp1 = cmath.sqrt(eps_ocean - sint * sint)

        R_vv = (eps_ocean * cost - temp1) / (eps_ocean * cost + temp1)
        R_hh = (cost - temp1) / (cost + temp1)

        R_rl = (R_vv - R_hh) / 2
        R_rr = (R_vv + R_hh) / 2

        if ddm_ant == 1:
            fresnel_coeff = abs(R_rl) * abs(R_rl)

        elif ddm_ant == 2:
            fresnel_coeff = abs(R_rr) * abs(R_rr)

    return fresnel_coeff, fresnel_axis, fresnel_orientation


def coh_det(raw_counts, snr_db):
    """
    this function computes the coherency of an input raw-count ddm
    Inputs
    1)raw ddm measured in counts
    2)SNR measured in decibels
    Outputs
    1)coherency ratio (CR)
    2)coherency state (CS)
    """
    peak_counts = np.amax(raw_counts)
    delay_peak, dopp_peak = np.unravel_index(raw_counts.argmax(), raw_counts.shape)

    # thermal noise exclusion
    # TODO: the threshold may need to be redefined
    if not np.isnan(snr_db):
        thre_coeff = 1.055 * math.exp(-0.193 * snr_db)
        thre = thre_coeff * peak_counts  # noise exclusion threshold

        raw_counts[raw_counts < thre] = 0

    # deterimine DDMA range
    delay_range = list(range(delay_peak - 1, delay_peak + 2))
    delay_min = min(delay_range)
    delay_max = max(delay_range)
    dopp_range = list(range(dopp_peak - 1, dopp_peak + 2))
    dopp_min = min(dopp_range)
    dopp_max = max(dopp_range)

    # determine if DDMA is within DDM, refine if needed
    if delay_min < 1:
        delay_range = [0, 1, 2]
    elif delay_max > 38:
        delay_range = [37, 38, 39]

    if dopp_min < 1:
        dopp_range = [0, 1, 2]
    elif dopp_max > 3:
        dopp_range = [2, 3, 4]

    C_in = np.sum(raw_counts[delay_range, :][:, dopp_range])  # summation of DDMA
    C_out = np.sum(raw_counts) - C_in  # summation of DDM excluding DDMA

    CR = C_in / C_out  # coherency ratio

    if CR >= 2:
        CS = 1
    else:  # CR < 2
        CS = 0

    return CR, CS


