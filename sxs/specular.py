# mike.laverick@auckland.ac.nz
# Specular point related functions
import math
import numpy as np
import pyproj
from scipy import constants
from scipy.interpolate import interpn
import geopy.distance as geo_dist
import time
import pymap3d as pm

from load_files import get_local_dem, get_map_value

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


def get_chi2(num_delay_bins, num_doppler_bins):
    pass
    # return chi2


def get_specular_bin(tx, rx, sx, ddm):
    pass
    # return specular_bin, zenith_code_phase, confidence_flag





