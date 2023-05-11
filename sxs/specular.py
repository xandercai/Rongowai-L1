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

from load_files import get_local_dem

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
        else:
            b = m_mu

    return nadir(m_lambda)


def coarsetune(tx_pos_xyz, rx_pos_xyz):
    """% this function computes the SP on a pure WGS84 datum based on
    % Inputs:
    % 1) tx_pos_xyz: ECEF coordinate of the TX
    % 2) rx_pos_xyz: ECEF coordinate of the RX
    % Outputs:
    % 1) SP_xyz, SP_lla: ECEF and LLA coordinate of a SP on a pure WGS84 datum"""

    # find coarse SP using Fibonacci sequence
    SP_xyz_coarse = ite(tx_pos_xyz, rx_pos_xyz)
    SP_lla_coarse = pyproj.transform(ecef, lla, *SP_xyz_coarse, radians=False)
    # longitude adjustment
    if SP_lla_coarse[0] < 0:
        SP_lla_coarse[0] += 360
    elif SP_lla_coarse[0] > 360:
        SP_lla_coarse[0] -= 360
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
    min_lat, max_lat = (sx_lla[1] - L / 2, sx_lla[1] + L / 2)
    min_lon, max_lon = (sx_lla[0] - L / 2, sx_lla[0] + L / 2)

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
    sx_temp = [lon_bin[n_i][0], lat_bin[m_i][0], ele[m_i, n_i][0]]
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
    while res > res_grid:
        res, _, sp_lla_coarse = finetune(
            tx_pos_xyz, rx_pos_xyz, sp_lla_coarse, L, model
        )
        # parameters for the next iteration - new searching area, new SP coordinate
        L = L * 2.0 / 11.0
    return sp_lla_coarse


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
    _, sx_lla_coarse = coarsetune(tx_pos_xyz, rx_pos_xyz)

    # initial searching region in degrees
    L_ocean_deg = 1.0
    # converge criteria 0.01 meter
    res_ocean_meter = 0.01

    # derive local angles
    sx_pos_lla = finetune_ocean(
        tx_pos_xyz, rx_pos_xyz, sx_lla_coarse, dtu10, L_ocean_deg, res_ocean_meter
    )
    sx_pos_xyz = pyproj.transform(lla, ecef, *sx_pos_lla, radians=False)
    # replaces get_map_value function
    dist = interpn(
        points=(dist_to_coast_nz["lon"], dist_to_coast_nz["lat"]),
        values=dist_to_coast_nz["ele"],
        xi=(sx_pos_lla[0], sx_pos_lla[1]),
        method="linear",
    )

    local_dem = get_local_dem(sx_pos_lla, dem, dtu10, dist)
    theta_i, theta_s, phi_i, phi_s = angles(local_dem, tx_pos_xyz, rx_pos_xyz)

    if dist > 0:
        # local height of the SP = local_dem["ele"][1,1]
        # projection to local dem
        sx_pos_xyz += (sx_pos_xyz / np.linalg.norm(sx_pos_xyz, 2)) * local_dem["ele"][
            1, 1
        ]

    v_tsx = tx_pos_xyz - sx_pos_xyz
    unit_tsx = v_tsx / np.linalg.norm(v_tsx, 2)
    unit_sx = sx_pos_xyz / np.linalg.norm(sx_pos_xyz, 2)
    inc_angle_deg = np.rad2deg(np.arccos(np.dot(unit_tsx, unit_sx)))

    d_phi1 = np.sin(np.deg2rad(phi_s - phi_i + 180)) / np.cos(
        np.deg2rad(phi_s - phi_i + 180)
    )
    d_phi = np.rad2deg(np.arctan(d_phi1))
    d_snell_deg = abs(theta_i - theta_s) + abs(d_phi)

    return sx_pos_lla, sx_pos_xyz, inc_angle_deg, d_snell_deg, dist, LOS_flag
