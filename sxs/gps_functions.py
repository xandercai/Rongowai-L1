from astropy.time import Time as astro_time
from ctypes import CDLL, c_double, c_uint, c_char_p, byref
import numpy as np
from pathlib import Path
from scipy import constants

# load C++ file for orbit calculations
c_path = Path().absolute().joinpath(Path("./sxs/lib/"))
GPS_GetSVInfo_filename = Path("GPS_GetSVInfo.so")
GPS_GetSVInfo = CDLL(str(c_path.joinpath(GPS_GetSVInfo_filename)))
# specify C++ variable for len(8) array of doubles
double_array_8 = c_double * 8


def gps2utc(gpsweek, gpsseconds):
    """GPS time to unix timestamp.

    Parameters
    ----------
    gpsweek : numpy.array(int)
        GPS week number, i.e. 1866.
    gpsseconds : numpy.array(int)
        Number of seconds since the beginning of week.

    Returns
    -------
    numpy.array(numpy.float64)
        Unix timestamp (UTC time).
    """
    secs_in_week = 604800
    secs = gpsweek * secs_in_week + gpsseconds

    t_gps = astro_time(secs, format="gps")
    t_utc = astro_time(t_gps, format="iso", scale="utc")

    return t_utc.unix


def utc2gps(timestamp):
    """unix timestamp to GPS.

    Parameters
    ----------
    numpy.array(numpy.float64)
        Unix timestamp (UTC time).

    Returns
    -------
    gpsweek : numpy.array(int)
        GPS week number, i.e. 1866.
    gpsseconds : numpy.array(int)
        Number of seconds since the beginning of week.
    """
    secs_in_week = 604800
    t_utc = astro_time(timestamp, format="unix", scale="utc")
    t_gps = astro_time(t_utc, format="gps")
    gpsweek, gpsseconds = divmod(t_gps.value, secs_in_week)
    return gpsweek, gpsseconds


def satellite_orbits(
    J_2,
    gps_week,
    gps_tow,
    transmitter_id,
    SV_PRN_LUT,
    orbit_file,
    tx_pos_x,
    tx_pos_y,
    tx_pos_z,
    tx_vel_x,
    tx_vel_y,
    tx_vel_z,
    tx_clk_bias,
    prn_code,
    sv_num,
    track_id,
    trans_id_unique,
    end=0,
    start=0,
):
    """Calculate satellite orbital information using C++ function.

    Parameters
    ----------
    J_2 : int
        Half number of NGRX_channels (J) to iterate over
    gps_week : np.array(int)
        Array of GPS week number, i.e. 1866.
    gps_tow : np.array(int)
        Array of number of seconds since the beginning of week.
    transmitter_id : np.array(int)
        Array of GPS transmitter IDs.
    SV_PRN_LUT : numpy.array()
        Numpy array containing mappings of transmitter ID to satellite vehicle number
    orbit_file : pathlib.Path
        Path of orbit file to use for calculations
    tx_pos_x : numpy.array()
        Empty array to receive x-component position of satellites
    tx_pos_y : numpy.array()
        Empty array to receive y-component position of satellites
    tx_pos_z : numpy.array()
        Empty array to receive z-component position of satellites
    tx_vel_x : numpy.array()
        Empty array to receive x-component velocity of satellites
    tx_vel_y : numpy.array()
        Empty array to receive y-component velocity of satellites
    tx_vel_z : numpy.array()
        Empty array to receive z-component velocity of satellites
    prn_code : numpy.array()
       Empty array to receive PRN of satellites
    sv_num : numpy.array()
       Empty array to receive Satellite vehicle number of satellites
    track_id : numpy.array()
       Empty array to receive indexes used to track satellite across NGRx channels
    trans_id_unique : numpy.array()
       Array containing the mapping for unique transponder IDs

    Optional parameters
    ----------
    end : int
        Last index of gps_tow to iterate up to. Default = 0
    start : int
        First index of gps_tow to iterate from. Default = 0
    """
    # Default range of seconds to iterate over
    iter_range = range(len(gps_tow))
    # truncate at end first, then at start for idx consistency
    if end:
        iter_range = iter_range[:end]
    if start:
        iter_range = iter_range[start:]

    # iterate over seconds of flight
    for sec in iter_range:
        # ngrx_channel=j, 20 channels, 10 satellites
        # iterate over the 10 possible satellites
        for ngrx_channel in range(J_2):
            prn1 = transmitter_id[sec][ngrx_channel]
            # check if satellite is being tracked in channel
            if not prn1:
                continue
            # determine satellite designation for C++ code, initialise result array
            sv_num1 = SV_PRN_LUT[np.where(SV_PRN_LUT == prn1)[0][0]][1]
            sat_pos = double_array_8()
            # call C++ function to calculate satellite info
            GPS_GetSVInfo.main(
                c_uint(prn1),
                c_uint(int(gps_week[sec])),
                c_double(gps_tow[sec]),
                byref(sat_pos),
                c_char_p(bytes(str(orbit_file), "utf-8")),
            )
            # assign C++ values to corresponding array indexes,
            # duplicating for left/right mirroring in DDMs
            posx, posy, posz, bias, velx, vely, velz, _ = sat_pos
            tx_pos_x[sec][ngrx_channel] = tx_pos_x[sec][ngrx_channel + J_2] = posx
            tx_pos_y[sec][ngrx_channel] = tx_pos_y[sec][ngrx_channel + J_2] = posy
            tx_pos_z[sec][ngrx_channel] = tx_pos_z[sec][ngrx_channel + J_2] = posz
            tx_vel_x[sec][ngrx_channel] = tx_vel_x[sec][ngrx_channel + J_2] = velx
            tx_vel_y[sec][ngrx_channel] = tx_vel_y[sec][ngrx_channel + J_2] = vely
            tx_vel_z[sec][ngrx_channel] = tx_vel_z[sec][ngrx_channel + J_2] = velz
            tx_clk_bias[sec][ngrx_channel] = tx_clk_bias[sec][ngrx_channel + J_2] = (
                bias * constants.c
            )
            prn_code[sec][ngrx_channel] = prn_code[sec][ngrx_channel + J_2] = prn1
            sv_num[sec][ngrx_channel] = sv_num[sec][ngrx_channel + J_2] = sv_num1
            track_id[sec][ngrx_channel] = track_id[sec][ngrx_channel + J_2] = (
                np.where(trans_id_unique == prn1)[0][0] + 1
            )
