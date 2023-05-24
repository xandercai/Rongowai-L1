# mike.laverick@auckland.ac.nz
# L1_main_L0.py
# Initial draft L1 script
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import netCDF4 as nc
import numpy
import numpy as np
import rasterio
from scipy.interpolate import interpn
import pyproj
from datetime import datetime
from PIL import Image
from timeit import default_timer as timer

from cal_functions import ddm_calibration, db2power, power2db, get_quality_flag
from gps_functions import gps2utc, utc2gps, satellite_orbits
from load_files import (
    load_netcdf,
    load_antenna_pattern,
    interp_ddm,
    get_orbit_file,
    load_dat_file_grid,
    get_surf_type2,
    get_local_dem_new,
)
from specular import (
    sp_solver,
    sp_related,
    get_sx_rx_gain,
    get_chi2,
    get_specular_bin,
    get_ddm_Aeff,
    ddm_brcs,
    get_ddm_nbrcs2,
    ddm_refl,
    get_fresnel,
    coh_det,
    meter2chips,
    delay_correction,
)

# Required to load the land cover mask file
Image.MAX_IMAGE_PIXELS = None


### ---------------------- Prelaunch 1: Load L0 data

# specify input L0 netcdf file
raw_data_path = Path().absolute().joinpath(Path("./dat/raw/"))
L0_filename = Path("20230404-065056_NZTU-NZWN.nc")
L0_dataset = nc.Dataset(raw_data_path.joinpath(L0_filename))


# load in rx-related variables
# PVT GPS week and sec
pvt_gps_week = load_netcdf(L0_dataset["/science/GPS_week_of_SC_attitude"])
pvt_gps_sec = load_netcdf(L0_dataset["/science/GPS_second_of_SC_attitude"])
# rx positions in ECEF, metres
rx_pos_x_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_position_x_ecef_m"])
rx_pos_y_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_position_y_ecef_m"])
rx_pos_z_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_position_z_ecef_m"])
# rx velocity in ECEF, m/s
rx_vel_x_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_velocity_x_ecef_mps"])
rx_vel_y_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_velocity_y_ecef_mps"])
rx_vel_z_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_velocity_z_ecef_mps"])
# rx attitude, deg | TODO this is actually radians and will be updated
rx_pitch_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_attitude_pitch_deg"])
rx_roll_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_attitude_roll_deg"])
rx_yaw_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_attitude_yaw_deg"])
# rx clock bias and drifts
rx_clk_bias_m_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_clock_bias_m"])
rx_clk_drift_mps_pvt = load_netcdf(L0_dataset["/geometry/receiver/rx_clock_drift_mps"])

# TODO: Some processing required here to fix leading/trailing/sporadic "zero" values?

# load in ddm-related variables
# tx ID/satellite PRN
transmitter_id = load_netcdf(L0_dataset["/science/ddm/transmitter_id"])
# raw counts and ddm parameters
first_scale_factor = load_netcdf(L0_dataset["/science/ddm/first_scale_factor"])
# raw counts, uncalibrated
raw_counts = load_netcdf(L0_dataset["/science/ddm/counts"])
zenith_i2q2 = load_netcdf(L0_dataset["/science/ddm/zenith_i2_plus_q2"])
rf_source = load_netcdf(L0_dataset["/science/ddm/RF_source"])
# binning standard deviation
std_dev_rf1 = load_netcdf(L0_dataset["/science/ddm/RF1_zenith_RHCP_std_dev"])
std_dev_rf2 = load_netcdf(L0_dataset["/science/ddm/RF2_nadir_LHCP_std_dev"])
std_dev_rf3 = load_netcdf(L0_dataset["/science/ddm/RF3_nadir_RHCP_std_dev"])

delay_bin_res = load_netcdf(L0_dataset['/science/ddm/delay_bin_res_narrow'])        # delay bin resolution
doppler_bin_res = load_netcdf(L0_dataset['/science/ddm/doppler_bin_res_narrow'])    # doppler bin resolution

# absolute ddm center delay and doppler
delay_center_chips = load_netcdf(L0_dataset["/science/ddm/center_delay_bin_code_phase"])
doppler_center_hz = load_netcdf(L0_dataset["/science/ddm/center_doppler_bin_frequency"])

# coherent duration and noncoherent integration
coherent_duration = (
    # xcai load_netcdf(L0_dataset["/science/ddm/L1_E1_coherent_duration"]) / 1000
    load_netcdf(L0_dataset["/science/ddm/L1_E1_coherent_duration"])
)
non_coherent_integrations = (
    # xcai load_netcdf(L0_dataset["/science/ddm/L1_E1_non_coherent_integrations"]) / 1000
    load_netcdf(L0_dataset["/science/ddm/L1_E1_non_coherent_integrations"])
)

# NGRx estimate additional delay path
add_range_to_sp_pvt = load_netcdf(L0_dataset["/science/ddm/additional_range_to_SP"])

# antenna temperatures and engineering timestamp
eng_timestamp = load_netcdf(L0_dataset["/eng/packet_creation_time"])
zenith_ant_temp_eng = load_netcdf(L0_dataset["/eng/zenith_ant_temp"])
nadir_ant_temp_eng = load_netcdf(L0_dataset["/eng/nadir_ant_temp"])


### ---------------------- Prelaunch 1.5: Filter valid timestampes


# rx-related variables

pvt_gps_week = pvt_gps_week.compressed()
pvt_gps_sec = pvt_gps_sec.compressed()

rx_pos_x_pvt = rx_pos_x_pvt.compressed()
rx_pos_y_pvt = rx_pos_y_pvt.compressed()
rx_pos_z_pvt = rx_pos_z_pvt.compressed()

rx_vel_x_pvt = rx_vel_x_pvt.compressed()
rx_vel_y_pvt = rx_vel_y_pvt.compressed()
rx_vel_z_pvt = rx_vel_z_pvt.compressed()

rx_roll_pvt = rx_roll_pvt.compressed()
rx_pitch_pvt = rx_pitch_pvt.compressed()
rx_yaw_pvt = rx_yaw_pvt.compressed()

rx_clk_bias_m_pvt = rx_clk_bias_m_pvt.compressed()
rx_clk_drift_mps_pvt = rx_clk_drift_mps_pvt.compressed()

# identify and compensate the value equal to 0 (randomly happens)
assert not (pvt_gps_week == 0).any(), "pvt_gps_week contains 0, need to compensate."

# ddm-related variables
transmitter_id = np.ma.compress_rows(np.ma.masked_invalid(transmitter_id))

first_scale_factor = np.ma.compress_rows(np.ma.masked_invalid(first_scale_factor))
raw_counts = raw_counts[~raw_counts[:, 0, 0, 0].mask, :, :, :]
zenith_i2q2 = np.ma.compress_rows(np.ma.masked_invalid(zenith_i2q2))

rf_source = np.ma.compress_rows(np.ma.masked_invalid(rf_source))

std_dev_rf1 = std_dev_rf1.compressed()
std_dev_rf2 = std_dev_rf2.compressed()
std_dev_rf3 = std_dev_rf3.compressed()

# absolute ddm center delay and doppler
delay_center_chips = np.ma.compress_rows(np.ma.masked_invalid(delay_center_chips))
doppler_center_hz = np.ma.compress_rows(np.ma.masked_invalid(doppler_center_hz))

# coherent duration and noncoherent integration
coherent_duration = coherent_duration.compressed() / 1000  # convert to seconds
non_coherent_integrations = non_coherent_integrations.compressed() / 1000


# NGRx estimate additional delay path
add_range_to_sp_pvt = np.ma.compress_rows(np.ma.masked_invalid(add_range_to_sp_pvt))

# the below is to process when ddm-related and rx-related variables do not
# have the same length, which happens for some of the L0 products
assert pvt_gps_week.shape[0] == transmitter_id.shape[0], "pvt_gps_week and transmitter_id do not have the same length."
#
# TODO: Additional processing if ddm- and rx- related varaibles aren't the same length
#

# temperatures from engineering data
eng_timestamp = eng_timestamp.compressed()
nadir_ant_temp_eng = nadir_ant_temp_eng.compressed()
zenith_ant_temp_eng = zenith_ant_temp_eng.compressed()


### ---------------------- Prelaunch 2 - define external data paths and filenames

# load L1a calibration tables
L1a_path = Path().absolute().joinpath(Path(r'./dat/L1a_cal/'))
L1a_cal_ddm_counts_db = np.loadtxt(L1a_path.joinpath(r'L1A_cal_ddm_counts_dB.dat'))
L1a_cal_ddm_power_dbm = np.loadtxt(L1a_path.joinpath(r'L1A_cal_ddm_power_dBm.dat'))


# load SRTM_30 DEM
dem_path = Path().absolute().joinpath(Path("./dat/dem/"))
dem_filename = Path("nzsrtm_30_v1.tif")
dem = rasterio.open(dem_path.joinpath(dem_filename))
dem = {
    "ele": dem.read(1),
    "lat": np.linspace(dem.bounds.top, dem.bounds.bottom, dem.height),
    "lon": np.linspace(dem.bounds.left, dem.bounds.right, dem.width),
}


# load DTU10 model
dtu_path = Path().absolute().joinpath(Path("./dat/dtu/"))
dtu_filename = Path("dtu10_v1.dat")
dtu10 = load_dat_file_grid(dtu_path.joinpath(dtu_filename))

# load ocean/land (distance to coast) mask
landmask_path = Path().absolute().joinpath(Path("./dat/cst/"))
landmask_filename = Path("dist_to_coast_nz_v1.dat")
landmask_nz = load_dat_file_grid(landmask_path.joinpath(landmask_filename))

# load landcover mask
lcv_path = Path().absolute().joinpath(Path("./dat/lcv/"))
lcv_filename = Path("lcv.png")
lcv_mask = Image.open(lcv_path.joinpath(lcv_filename))

# process inland water mask
pek_path = Path().absolute().joinpath(Path("./dat/pek/"))

water_mask = {}
for path in [
    "160E_40S",
    "170E_30S",
    "170E_40S",
]:
    water_mask[path] = {}
    pek_file = rasterio.open(pek_path.joinpath("occurrence_" + path + ".tif"))
    water_mask[path]["lon_min"] = pek_file._transform[0]
    water_mask[path]["res_deg"] = pek_file._transform[1]
    water_mask[path]["lat_max"] = pek_file._transform[3]
    water_mask[path]["data"] = pek_file.read(1)

# load PRN-SV and SV-EIRP(static) LUT
gps_path = Path().absolute().joinpath(Path("./dat/gps/"))
SV_PRN_filename = Path("PRN_SV_LUT_v1.dat")
SV_eirp_filename = Path("GPS_SV_EIRP_Params_v7.dat")
SV_PRN_LUT = np.loadtxt(gps_path.joinpath(SV_PRN_filename), usecols=(0, 1))
SV_eirp_LUT = np.loadtxt(gps_path.joinpath(SV_eirp_filename))

# load and process nadir NGRx-GNSS antenna patterns
rng_path = Path().absolute().joinpath(Path("./dat/rng/"))
LHCP_L_filename = Path("GNSS_LHCP_L_gain_db_i_v1.dat")
LHCP_R_filename = Path("GNSS_LHCP_R_gain_db_i_v1.dat")
RHCP_L_filename = Path("GNSS_RHCP_L_gain_db_i_v1.dat")
RHCP_R_filename = Path("GNSS_RHCP_R_gain_db_i_v1.dat")
LHCP_pattern = {
    "LHCP": load_antenna_pattern(rng_path.joinpath(LHCP_L_filename)),
    "RHCP": load_antenna_pattern(rng_path.joinpath(LHCP_R_filename)),
}
RHCP_pattern = {
    "LHCP": load_antenna_pattern(rng_path.joinpath(RHCP_L_filename)),
    "RHCP": load_antenna_pattern(rng_path.joinpath(RHCP_R_filename)),
}

# load physical scattering area LUT
phy_ele_filename = Path("phy_ele_size.dat")  # same path as DEM
phy_ele_size = np.loadtxt(dem_path.joinpath(phy_ele_filename))


### ---------------------- Part 1: General processing
# This part derives global constants, timestamps, and all the other
# parameters at ddm timestamps


# make array (ddm_pvt_bias) of non_coherent_integrations divided by 2
ddm_pvt_bias = non_coherent_integrations / 2
# make array (pvt_utc) of gps to unix time (see above)
pvt_utc = np.array(
    [gps2utc(week, pvt_gps_sec[i]) for i, week in enumerate(pvt_gps_week)]
)
# make array (ddm_utc) of ddm_pvt_bias + pvt_utc
ddm_utc = pvt_utc + ddm_pvt_bias
# make arrays (gps_week, gps_tow) of ddm_utc to gps week/sec (inc. 1/2*integration time)
gps_week, gps_tow = utc2gps(ddm_utc)

# interpolate rx positions onto new time grid
rx_pos_x = interp_ddm(pvt_utc, rx_pos_x_pvt, ddm_utc)
rx_pos_y = interp_ddm(pvt_utc, rx_pos_y_pvt, ddm_utc)
rx_pos_z = interp_ddm(pvt_utc, rx_pos_z_pvt, ddm_utc)
rx_pos_xyz = [rx_pos_x, rx_pos_y, rx_pos_z]
# interpolate rx velocities onto new time grid
rx_vel_x = interp_ddm(pvt_utc, rx_vel_x_pvt, ddm_utc)
rx_vel_y = interp_ddm(pvt_utc, rx_vel_y_pvt, ddm_utc)
rx_vel_z = interp_ddm(pvt_utc, rx_vel_z_pvt, ddm_utc)
rx_vel_xyz = [rx_vel_x, rx_vel_y, rx_vel_z]
# interpolate rx roll/pitch/yaw onto new time grid
rx_roll = interp_ddm(pvt_utc, rx_roll_pvt, ddm_utc)
rx_pitch = interp_ddm(pvt_utc, rx_pitch_pvt, ddm_utc)
rx_yaw = interp_ddm(pvt_utc, rx_yaw_pvt, ddm_utc)
rx_attitude = [rx_roll, rx_pitch, rx_yaw]
# interpolate bias+drift onto new time grid
rx_clk_bias_m = interp_ddm(pvt_utc, rx_clk_bias_m_pvt, ddm_utc)
rx_clk_drift_mps = interp_ddm(pvt_utc, rx_clk_drift_mps_pvt, ddm_utc)
rx_clk = [rx_clk_bias_m, rx_clk_drift_mps]

# define maximum NGRx signal capacity, and half
J = 20
J_2 = int(J / 2)

# interpolate "additional_range_to_SP" to new time grid
add_range_to_sp = np.full([*add_range_to_sp_pvt.shape], np.nan)
for ngrx_channel in range(J):
    add_range_to_sp[:, ngrx_channel] = interp_ddm(
        pvt_utc, add_range_to_sp_pvt[:, ngrx_channel], ddm_utc
    )
# interpolate temperatures onto new time grid
ant_temp_zenith = interp_ddm(eng_timestamp, zenith_ant_temp_eng, ddm_utc)
ant_temp_nadir = interp_ddm(eng_timestamp, nadir_ant_temp_eng, ddm_utc)


# ecef2lla Matlab function
# define projections and transform
# TODO function is depreciated,see following url
# https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1
ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
lla = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
# ecef2ella
lon, lat, alt = pyproj.transform(ecef, lla, *rx_pos_xyz, radians=False)
rx_pos_lla = [lat, lon, alt]

# determine specular point "over land" flag from landmask
# replaces get_map_value function
status_flags_one_hz = interpn(
    points=(landmask_nz["lon"], landmask_nz["lat"]),
    values=landmask_nz["ele"],
    xi=(lon, lat),
    method="linear",
)
status_flags_one_hz[status_flags_one_hz > 0] = 5
status_flags_one_hz[status_flags_one_hz <= 0] = 4

# write global variables

L1_postCal = {}

time_coverage_start_obj = datetime.utcfromtimestamp(ddm_utc[0])
L1_postCal['time_coverage_start'] = time_coverage_start_obj.strftime("%Y-%m-%d %H:%M:%S")
time_coverage_end_obj = datetime.utcfromtimestamp(ddm_utc[-1])
L1_postCal['time_coverage_end'] = time_coverage_end_obj.strftime("%d-%m-%Y %H:%M:%S")
L1_postCal['time_coverage_resolution'] = ddm_utc[1] - ddm_utc[0]

# time coverage
hours, remainder = divmod((ddm_utc[-1] - ddm_utc[0] + 1), 3600)
minutes, seconds = divmod(remainder, 60)
L1_postCal['time_coverage_duration'] = f"P0DT{int(hours)}H{int(minutes)}M{int(seconds)}S"

L1_postCal['aircraft_reg'] = 'ZK-NFA'             # default value
L1_postCal['ddm_source'] = 2                      # 1 = GPS signal simulator, 2 = aircraft
L1_postCal['ddm_time_type_selector'] = 1          # 1 = middle of DDM sampling period
L1_postCal['delay_resolution'] = 0.25             # unit in chips
L1_postCal['dopp_resolution'] = 500               # unit in Hz
L1_postCal['dem_source'] = 'SRTM30'

# write algorithm and LUT versions
L1_postCal['l1_algorithm_version'] = '1.1'
L1_postCal['l1_data_version'] = '1'
L1_postCal['l1a_sig_LUT_version'] = '1'
L1_postCal['l1a_noise_LUT_version'] = '1'
L1_postCal['ngrx_port_mapping_version'] = '1'
L1_postCal['nadir_ant_data_version'] = '1'
L1_postCal['zenith_ant_data_version'] = '1'
L1_postCal['prn_sv_maps_version'] = '1'
L1_postCal['gps_eirp_param_version'] = '7'
L1_postCal['land_mask_version'] = '1'
L1_postCal['surface_type_version'] = '1'
L1_postCal['mean_sea_surface_version'] = '1'
L1_postCal['per_bin_ant_version'] = '1'

# write timestamps and ac-related variables
L1_postCal['pvt_timestamp_gps_week'] = pvt_gps_week
L1_postCal['pvt_timestamp_gps_sec'] = pvt_gps_sec
L1_postCal['pvt_timestamp_utc'] = pvt_utc

L1_postCal['ddm_timestamp_gps_week'] = gps_week
L1_postCal['ddm_timestamp_gps_sec'] = gps_tow
L1_postCal['ddm_timestamp_utc'] = ddm_utc

L1_postCal['ddm_pvt_bias'] = ddm_pvt_bias

# 0-indexed sample and DDM
L1_postCal['sample'] = np.arange(0, len(pvt_gps_sec))
L1_postCal['ddm'] = np.arange(0, J)

L1_postCal['sp_fsw_delay'] = delay_center_chips
L1_postCal['sp_ngrx_dopp'] = doppler_center_hz

L1_postCal['add_range_to_sp'] = add_range_to_sp
L1_postCal['add_range_to_sp_pvt'] = add_range_to_sp_pvt

L1_postCal['ac_lat'] = rx_pos_lla[0]
L1_postCal['ac_lon'] = rx_pos_lla[1]
L1_postCal['ac_alt'] = rx_pos_lla[2]

L1_postCal['ac_pos_x_pvt'] = rx_pos_x_pvt
L1_postCal['ac_pos_y_pvt'] = rx_pos_y_pvt
L1_postCal['ac_pos_z_pvt'] = rx_pos_z_pvt

L1_postCal['ac_pos_x'] = rx_pos_x
L1_postCal['ac_pos_y'] = rx_pos_y
L1_postCal['ac_pos_z'] = rx_pos_z

L1_postCal['ac_vel_x_pvt'] = rx_vel_x_pvt
L1_postCal['ac_vel_y_pvt'] = rx_vel_y_pvt
L1_postCal['ac_vel_z_pvt'] = rx_vel_z_pvt

L1_postCal['ac_vel_x'] = rx_vel_x
L1_postCal['ac_vel_y'] = rx_vel_y
L1_postCal['ac_vel_z'] = rx_vel_z

L1_postCal['ac_roll_pvt'] = rx_roll_pvt
L1_postCal['ac_pitch_pvt'] = rx_pitch_pvt
L1_postCal['ac_yaw_pvt'] = rx_yaw_pvt

L1_postCal['ac_roll'] = rx_attitude[0]
L1_postCal['ac_pitch'] = rx_attitude[1]
L1_postCal['ac_yaw'] = rx_attitude[2]

L1_postCal['rx_clk_bias_pvt'] = rx_clk_bias_m_pvt
L1_postCal['rx_clk_drift_pvt'] = rx_clk_drift_mps_pvt

L1_postCal['rx_clk_bias'] = rx_clk_bias_m
L1_postCal['rx_clk_drift'] = rx_clk_drift_mps

L1_postCal['ant_temp_nadir'] = ant_temp_nadir
L1_postCal['ant_temp_zenith'] = ant_temp_zenith

L1_postCal['status_flags_one_hz'] = status_flags_one_hz

# part 1 ends

### ---------------------- Part 2: Derive TX related variables
# This part derives TX positions and velocities, maps between PRN and SVN,
# and gets track ID
# This part is to deal with the new SP3 naming policy, TODO for old SP3 naming policy (Nov 2022 and before)

# determine unique satellite transponder IDs
trans_id_unique = np.unique(transmitter_id)
trans_id_unique = trans_id_unique[trans_id_unique > 0]

# create data arrays for C++ code to populate
tx_pos_x = np.full([*transmitter_id.shape], np.nan)
tx_pos_y = np.full([*transmitter_id.shape], np.nan)
tx_pos_z = np.full([*transmitter_id.shape], np.nan)
tx_vel_x = np.full([*transmitter_id.shape], np.nan)
tx_vel_y = np.full([*transmitter_id.shape], np.nan)
tx_vel_z = np.full([*transmitter_id.shape], np.nan)
tx_clk_bias = np.full([*transmitter_id.shape], np.nan)
prn_code = np.full([*transmitter_id.shape], np.nan)
sv_num = np.full([*transmitter_id.shape], np.nan)
track_id = np.full([*transmitter_id.shape], np.nan)
orbit_bundle = [
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
]

# determine whether flight spans a UTC day
if time_coverage_start_obj.day == time_coverage_end_obj.day:
    # determine single orbit file of that day
    orbit_file1 = get_orbit_file(
        gps_week,
        gps_tow,
        time_coverage_start_obj,
        time_coverage_end_obj,
    )
    # calculate satellite orbits, data assigned to orbit_bundle arrays
    satellite_orbits(
        J_2, gps_week, gps_tow, transmitter_id, SV_PRN_LUT, orbit_file1, *orbit_bundle
    )
else:
    # find idx of day change in timestamps
    # np.diff does "arr_new[i] = arr[i+1] - arr[i]" thus +1 to find changed idx
    change_idx = np.where(np.diff(np.floor(gps_tow / 86400)) > 0)[0][0] + 1
    # determine day_N and day_N+1 orbit files to use
    orbit_file1, orbit_file2 = get_orbit_file(
        gps_week,
        gps_tow,
        time_coverage_start_obj,
        time_coverage_end_obj,
        change_idx=change_idx,
    )
    # calculate first chunk of specular points using 1st orbit file
    # data assigned to orbit_bundle arrays
    satellite_orbits(
        J_2,
        gps_week,
        gps_tow,
        transmitter_id,
        SV_PRN_LUT,
        orbit_file1,
        *orbit_bundle,
        end=change_idx,
    )
    # calculate last chunk of specular points using 2nd orbit file
    # data assigned to orbit_bundle arrays
    satellite_orbits(
        J_2,
        gps_week,
        gps_tow,
        transmitter_id,
        SV_PRN_LUT,
        orbit_file2,
        *orbit_bundle,
        start=change_idx,
    )

# write TX variables
L1_postCal['tx_pos_x'] = tx_pos_x
L1_postCal['tx_pos_y'] = tx_pos_y
L1_postCal['tx_pos_z'] = tx_pos_z
L1_postCal['tx_vel_x'] = tx_vel_x
L1_postCal['tx_vel_y'] = tx_vel_y
L1_postCal['tx_vel_z'] = tx_vel_z
L1_postCal['tx_clk_bias'] = tx_clk_bias
L1_postCal['prn_code'] = prn_code
L1_postCal['sv_num'] = sv_num
L1_postCal['track_id'] = track_id

### ----------------------  Part 3: L1a calibration
# this part converts from raw counts to signal power in watts and complete
# L1a calibration


# create data arrays to hold DDM power/count arrays
# initialise variables for L1a results
ddm_power_counts = np.full([*raw_counts.shape], np.nan)
power_analog = np.full([*raw_counts.shape], np.nan)

noise_floor_counts = np.full([*transmitter_id.shape], np.nan)
noise_floor = np.full([*transmitter_id.shape], np.nan)
snr_db = np.full([*transmitter_id.shape], np.nan)

peak_ddm_counts = np.full([*transmitter_id.shape], np.nan)
peak_ddm_watts = np.full([*transmitter_id.shape], np.nan)
peak_delay_bin = np.full([*transmitter_id.shape], np.nan)

ddm_noise_counts = np.full([*transmitter_id.shape], np.nan)
ddm_noise_watts = np.full([*transmitter_id.shape], np.nan)

ddm_ant = np.full([*transmitter_id.shape], np.nan)
inst_gain = np.full([*transmitter_id.shape], np.nan)

# invoke calibration function which populates above arrays
ddm_calibration(
    std_dev_rf1,
    std_dev_rf2,
    std_dev_rf3,
    J,
    prn_code,
    raw_counts,
    rf_source,
    first_scale_factor,
    ddm_power_counts,
    power_analog,
    ddm_ant,
    ddm_noise_counts,
    ddm_noise_watts,
    peak_ddm_counts,
    peak_ddm_watts,
    peak_delay_bin,
    noise_floor_counts,
    noise_floor,
    inst_gain,
    snr_db
)

# save outputs to L1 structure
L1_postCal['raw_counts'] = ddm_power_counts
L1_postCal['l1a_power_ddm'] = power_analog
L1_postCal['zenith_sig_i2q2'] = zenith_i2q2  # read from file

L1_postCal['ddm_noise_floor'] = noise_floor
L1_postCal['ddm_snr'] = snr_db

L1_postCal['inst_gain'] = inst_gain
L1_postCal['ddm_ant'] = ddm_ant  # 0-based




#   # --------------------- Part 4A: SP solver and geometries
# initialise variables
# initialise a huge amount of empty arrays
sx_pos_x = np.full([*transmitter_id.shape], np.nan)
sx_pos_y = np.full([*transmitter_id.shape], np.nan)
sx_pos_z = np.full([*transmitter_id.shape], np.nan)

sx_lat = np.full([*transmitter_id.shape], np.nan)
sx_lon = np.full([*transmitter_id.shape], np.nan)
sx_alt = np.full([*transmitter_id.shape], np.nan)

sx_vel_x = np.full([*transmitter_id.shape], np.nan)
sx_vel_y = np.full([*transmitter_id.shape], np.nan)
sx_vel_z = np.full([*transmitter_id.shape], np.nan)

sx_inc_angle = np.full([*transmitter_id.shape], np.nan)
sx_d_snell_angle = np.full([*transmitter_id.shape], np.nan)
dist_to_coast_km = np.full([*transmitter_id.shape], np.nan)
surface_type = np.full([*transmitter_id.shape], np.nan)

LOS_flag = np.full([*transmitter_id.shape], np.nan)

tx_to_sp_range = np.full([*transmitter_id.shape], np.nan)
rx_to_sp_range = np.full([*transmitter_id.shape], np.nan)

gps_boresight = np.full([*transmitter_id.shape], np.nan)

sx_theta_body = np.full([*transmitter_id.shape], np.nan)
sx_az_body = np.full([*transmitter_id.shape], np.nan)

sx_theta_enu = np.full([*transmitter_id.shape], np.nan)
sx_az_enu = np.full([*transmitter_id.shape], np.nan)

gps_tx_power_db_w = np.full([*transmitter_id.shape], np.nan)
gps_ant_gain_db_i = np.full([*transmitter_id.shape], np.nan)
static_gps_eirp = np.full([*transmitter_id.shape], np.nan)

sx_rx_gain = np.full([*transmitter_id.shape], np.nan)

sx_rx_gain_copol = np.full([*transmitter_id.shape], np.nan)
sx_rx_gain_xpol = np.full([*transmitter_id.shape], np.nan)


# iterate over each second of flight
for sec in range(len(transmitter_id)):
    print(f'******** start processing part 4A {sec} second data ********')
    # retrieve rx positions, velocities and attitdues
    # bundle up craft pos/vel/attitude data into per sec, and rx1
    rx_pos_xyz1 = np.array([rx_pos_x[sec], rx_pos_y[sec], rx_pos_z[sec]])
    rx_vel_xyz1 = np.array([rx_vel_x[sec], rx_vel_y[sec], rx_vel_z[sec]])
    rx_attitude1 = np.array([rx_roll[sec], rx_pitch[sec], rx_yaw[sec]])
    rx1 = {
        "rx_pos_xyz": rx_pos_xyz1,
        "rx_vel_xyz": rx_vel_xyz1,
        "rx_attitude": rx_attitude1,
    }

    # variables are solved only for LHCP channels
    # RHCP channels share the same vales except RX gain solved for each channel
    for ngrx_channel in range(J_2):
        # retrieve tx positions and velocities
        # bundle up satellite position and velocity data into per sec, and tx1
        tx_pos_xyz1 = np.array(
            [
                tx_pos_x[sec][ngrx_channel],
                tx_pos_y[sec][ngrx_channel],
                tx_pos_z[sec][ngrx_channel],
            ]
        )
        tx_vel_xyz1 = np.array(
            [
                tx_vel_x[sec][ngrx_channel],
                tx_vel_y[sec][ngrx_channel],
                tx_vel_z[sec][ngrx_channel],
            ]
        )

        trans_id1 = prn_code[sec][ngrx_channel]
        sv_num1 = sv_num[sec][ngrx_channel]

        ddm_ant1 = ddm_ant[sec][ngrx_channel]

        tx1 = {"tx_pos_xyz": tx_pos_xyz1, "tx_vel_xyz": tx_vel_xyz1, "sv_num": sv_num1}

        # only process these with valid TX positions
        # TODO is checking only pos_x enough? it could be.
        if not np.isnan(tx_pos_x[sec][ngrx_channel]):

            # Part 4.1: SP solver
            # derive SP positions, angle of incidence and distance to coast
            # returning sx_pos_lla1 in Py version to avoid needless coord conversions
            (
                sx_pos_xyz1,
                inc_angle_deg1,
                d_snell_deg1,
                dist_to_coast_km1,
                LOS_flag1,
            ) = sp_solver(tx_pos_xyz1, rx_pos_xyz1, dem, dtu10, landmask_nz)

            lon, lat, alt = pyproj.transform(ecef, lla, *sx_pos_xyz1, radians=False)
            sx_pos_lla1 = [lat, lon, alt]

            LOS_flag[sec][ngrx_channel] = int(LOS_flag1)

            # only process samples with valid sx positions, i.e., LOS = True
            if LOS_flag1:
                # <lon,lat,alt> of the specular reflection
                # algorithm version 1.11
                surface_type1 = get_surf_type2(
                    sx_pos_lla1, landmask_nz, lcv_mask, water_mask
                )

                # only process samples with valid sx positions, i.e., LOS = 1
                # derive sx velocity
                # time step in second
                dt = 1
                tx_pos_xyz_dt = tx_pos_xyz1 + tx_vel_xyz1  # dt* no point if this is 1s
                rx_pos_xyz_dt = rx_pos_xyz1 + rx_vel_xyz1
                (
                    sx_pos_xyz_dt,
                    _,
                    _,
                    _,
                    _,
                ) = sp_solver(tx_pos_xyz_dt, rx_pos_xyz_dt, dem, dtu10, landmask_nz)

                # TODO: here get large difference from matlab version
                sx_vel_xyz1 = np.array(sx_pos_xyz_dt) - np.array(sx_pos_xyz1)

                # save sx values to variables
                sx_pos_x[sec][ngrx_channel] = sx_pos_xyz1[0]
                sx_pos_y[sec][ngrx_channel] = sx_pos_xyz1[1]
                sx_pos_z[sec][ngrx_channel] = sx_pos_xyz1[2]

                # TODO: sx_alt is different from the matlab version, need to debug
                sx_lat[sec][ngrx_channel] = sx_pos_lla1[0]
                sx_lon[sec][ngrx_channel] = sx_pos_lla1[1]
                sx_alt[sec][ngrx_channel] = sx_pos_lla1[2]

                # TODO: vel is different from the matlab version but the process is the same
                sx_vel_x[sec][ngrx_channel] = sx_vel_xyz1[0]
                sx_vel_y[sec][ngrx_channel] = sx_vel_xyz1[1]
                sx_vel_z[sec][ngrx_channel] = sx_vel_xyz1[2]
                surface_type[sec][ngrx_channel] = surface_type1

                sx_inc_angle[sec][ngrx_channel] = inc_angle_deg1
                sx_d_snell_angle[sec][ngrx_channel] = d_snell_deg1
                dist_to_coast_km[sec][ngrx_channel] = dist_to_coast_km1

                # Part 4.2: SP-related variables - 1
                # this part derives tx/rx gains, ranges and other related variables
                # derive SP related geo-parameters, including angles in various frames, ranges and antenna gain/GPS EIRP
                (
                    sx_angle_body1,
                    sx_angle_enu1,
                    sx_angle_ant1,
                    theta_gps1,
                    ranges1,
                    gps_rad1
                ) = sp_related(
                    tx1, rx1, sx_pos_xyz1, SV_eirp_LUT
                )

                # get values for deriving BRCS and reflectivity
                R_tsx1 = ranges1[0]
                R_rsx1 = ranges1[1]
                gps_eirp_watt1 = gps_rad1[2]

                # get active antenna gain for LHCP and RHCP channels
                sx_rx_gain_LHCP1 = get_sx_rx_gain(sx_angle_ant1, LHCP_pattern)
                sx_rx_gain_RHCP1 = get_sx_rx_gain(sx_angle_ant1, RHCP_pattern)

                # save to variables
                sx_theta_body[sec, ngrx_channel] = sx_angle_body1[0]
                sx_az_body[sec, ngrx_channel] = sx_angle_body1[1]

                sx_theta_enu[sec, ngrx_channel] = sx_angle_enu1[0]
                sx_az_enu[sec, ngrx_channel] = sx_angle_enu1[1]

                gps_boresight[sec, ngrx_channel] = theta_gps1

                tx_to_sp_range[sec, ngrx_channel] = ranges1[0]
                rx_to_sp_range[sec, ngrx_channel] = ranges1[1]

                gps_tx_power_db_w[sec, ngrx_channel] = gps_rad1[0]
                gps_ant_gain_db_i[sec, ngrx_channel] = gps_rad1[1]
                static_gps_eirp[sec, ngrx_channel] = gps_rad1[2]

                sx_rx_gain[sec, ngrx_channel] = sx_rx_gain_LHCP1[0]  # LHCP channel rx gain
                sx_rx_gain[sec, ngrx_channel + J_2] = sx_rx_gain_RHCP1[1]  # RHCP channel rx gain


# expand to RHCP channels
sx_pos_x[:, J_2:J] = sx_pos_x[:, 0:J_2]
sx_pos_y[:, J_2:J] = sx_pos_y[:, 0:J_2]
sx_pos_z[:, J_2:J] = sx_pos_z[:, 0:J_2]

sx_lat[:, J_2:J] = sx_lat[:, 0:J_2]
sx_lon[:, J_2:J] = sx_lon[:, 0:J_2]
sx_alt[:, J_2:J] = sx_alt[:, 0:J_2]

sx_vel_x[:, J_2:J] = sx_vel_x[:, 0:J_2]
sx_vel_y[:, J_2:J] = sx_vel_y[:, 0:J_2]
sx_vel_z[:, J_2:J] = sx_vel_z[:, 0:J_2]

surface_type[:, J_2:J] = surface_type[:, 0:J_2]
dist_to_coast_km[:, J_2:J] = dist_to_coast_km[:, 0:J_2]
LOS_flag[:, J_2:J] = LOS_flag[:, 0:J_2]

rx_to_sp_range[:, J_2:J] = rx_to_sp_range[:, 0:J_2]
tx_to_sp_range[:, J_2:J] = tx_to_sp_range[:, 0:J_2]

sx_inc_angle[:, J_2:J] = sx_inc_angle[:, 0:J_2]
sx_d_snell_angle[:, J_2:J] = sx_d_snell_angle[:, 0:J_2]

sx_theta_body[:, J_2:J] = sx_theta_body[:, 0:J_2]
sx_az_body[:, J_2:J] = sx_az_body[:, 0:J_2]

sx_theta_enu[:, J_2:J] = sx_theta_enu[:, 0:J_2]
sx_az_enu[:, J_2:J] = sx_az_enu[:, 0:J_2]

gps_boresight[:, J_2:J] = gps_boresight[:, 0:J_2]

static_gps_eirp[:, J_2:J] = static_gps_eirp[:, 0:J_2]

gps_tx_power_db_w[:, J_2:J] = gps_tx_power_db_w[:, 0:J_2]
gps_ant_gain_db_i[:, J_2:J] = gps_ant_gain_db_i[:, 0:J_2]

# save variables
L1_postCal['sp_pos_x'] = sx_pos_x                       # checked value diff < 1 / e5
L1_postCal['sp_pos_y'] = sx_pos_y                       # checked value diff < 1 / e5
L1_postCal['sp_pos_z'] = sx_pos_z                       # checked value diff < 1 / e6

L1_postCal['sp_lat'] = sx_lat                           # checked ok
L1_postCal['sp_lon'] = sx_lon                           # checked ok
L1_postCal['sp_alt'] = sx_alt                           # checked ok

L1_postCal['sp_vel_x'] = sx_vel_x                       # checked value diff < 10
L1_postCal['sp_vel_y'] = sx_vel_y                       # checked value diff < 10
L1_postCal['sp_vel_z'] = sx_vel_z                       # checked value diff < 10

L1_postCal['sp_surface_type'] = surface_type            # checked ok
L1_postCal['sp_dist_to_coast_km'] = dist_to_coast_km    # checked ok
L1_postCal['LOS_flag'] = LOS_flag                       # checked ok

L1_postCal['rx_to_sp_range'] = rx_to_sp_range           # checked value diff < 1 / e2
L1_postCal['tx_to_sp_range'] = tx_to_sp_range           # checked value diff < 1 / e7

L1_postCal['sp_inc_angle'] = sx_inc_angle               # checked ok
L1_postCal['sp_d_snell_angle'] = sx_d_snell_angle       # checked ok

L1_postCal['sp_theta_body'] = sx_theta_body             # checked value diff < 0.1
L1_postCal['sp_az_body'] = sx_az_body                   # checked value diff < 0.01
L1_postCal['sp_theta_enu'] = sx_theta_enu               # checked value diff < 0.1 / e2
L1_postCal['sp_az_enu'] = sx_az_enu                     # checked ok

L1_postCal['sp_rx_gain'] = sx_rx_gain                   # checked ok

L1_postCal['gps_off_boresight_angle_deg'] = gps_boresight  # checked ok

L1_postCal['static_gps_eirp'] = static_gps_eirp         # checked ok
L1_postCal['gps_tx_power_db_w'] = gps_tx_power_db_w     # checked ok
L1_postCal['gps_ant_gain_db_i'] = gps_ant_gain_db_i     # checked ok

# np.save('debug.npy', L1_postCal)

##############
# L1_postCal_loaded = np.load('debug.npy', allow_pickle=True).item()

# def dic_to_keys_values(dic):
#     keys, values = list(dic.keys()), list(dic.values())
#     return keys, values
#
#
# def numpy_assert_almost_dict_values(dict1, dict2):
#     keys1, values1 = dic_to_keys_values(dict1)
#     keys2, values2 = dic_to_keys_values(dict2)
#     np.testing.assert_equal(keys1, keys2)
#     np.testing.assert_equal(values1, values2)
#
#
# numpy_assert_almost_dict_values(L1_postCal, L1_postCal_loaded)
##############

# L1_postCal = np.load('debug.npy', allow_pickle=True).item()
#
# sx_pos_x = L1_postCal['sp_pos_x']
# sx_pos_y = L1_postCal['sp_pos_y']
# sx_pos_z = L1_postCal['sp_pos_z']
#
# sx_lat = L1_postCal['sp_lat']
# sx_lon = L1_postCal['sp_lon']
# sx_alt = L1_postCal['sp_alt']
#
# sx_vel_x = L1_postCal['sp_vel_x']
# sx_vel_y = L1_postCal['sp_vel_y']
# sx_vel_z = L1_postCal['sp_vel_z']
#
# surface_type = L1_postCal['sp_surface_type']
# dist_to_coast_km = L1_postCal['sp_dist_to_coast_km']
# LOS_flag = L1_postCal['LOS_flag']
#
# rx_to_sp_range = L1_postCal['rx_to_sp_range']
# tx_to_sp_range = L1_postCal['tx_to_sp_range']
#
# sx_inc_angle = L1_postCal['sp_inc_angle']
# sx_d_snell_angle = L1_postCal['sp_d_snell_angle']
#
# sx_theta_body = L1_postCal['sp_theta_body']
# sx_az_body = L1_postCal['sp_az_body']
# sx_theta_enu = L1_postCal['sp_theta_enu']
# sx_az_enu = L1_postCal['sp_az_enu']
#
# sx_rx_gain = L1_postCal['sp_rx_gain']
#
# gps_boresight = L1_postCal['gps_off_boresight_angle_deg']
#
# static_gps_eirp = L1_postCal['static_gps_eirp']
# gps_tx_power_db_w = L1_postCal['gps_tx_power_db_w']
# gps_ant_gain_db_i = L1_postCal['gps_ant_gain_db_i']

##############

# -------------------- Part 4B: BRCS/NBRCS, reflectivity, coherent status and fresnel zone
# initialise variables
brcs_ddm_peak_bin_delay_row = np.full([*transmitter_id.shape], np.nan)
brcs_ddm_peak_bin_dopp_col = np.full([*transmitter_id.shape], np.nan)

brcs_ddm_sp_bin_delay_row = np.full([*transmitter_id.shape], np.nan)
brcs_ddm_sp_bin_dopp_col = np.full([*transmitter_id.shape], np.nan)

sp_delay_error = np.full([*transmitter_id.shape], np.nan)
sp_dopp_error = np.full([*transmitter_id.shape], np.nan)

confidence_flag = np.full([*transmitter_id.shape], np.nan)

zenith_code_phase = np.full([*transmitter_id.shape], np.nan)

brcs = np.full([*transmitter_id.shape, 40, 5], np.nan)
A_eff = np.full([*transmitter_id.shape, 40, 5], np.nan)
A_eff_all = np.full([*transmitter_id.shape, 79, 9], np.nan)  # debug only

norm_refl_waveform = np.full([*transmitter_id.shape, 40, 1], np.nan)

nbrcs_scatter_area_v1 = np.full([*transmitter_id.shape], np.nan)
ddm_nbrcs_v1 = np.full([*transmitter_id.shape], np.nan)

nbrcs_scatter_area_v2 = np.full([*transmitter_id.shape], np.nan)
ddm_nbrcs_v2 = np.full([*transmitter_id.shape], np.nan)

surface_reflectivity = np.full([*transmitter_id.shape, 40, 5], np.nan)
surface_reflectivity_peak = np.full([*transmitter_id.shape], np.nan)

fresnel_coeff = np.full([*transmitter_id.shape], np.nan)
fresnel_minor = np.full([*transmitter_id.shape], np.nan)
fresnel_major = np.full([*transmitter_id.shape], np.nan)
fresnel_orientation = np.full([*transmitter_id.shape], np.nan)

coherency_ratio = np.full([*transmitter_id.shape], np.nan)
coherency_state = np.full([*transmitter_id.shape], np.nan)

# derive amb-function (chi2) to be used in computing A_eff
chi2 = get_chi2(40, 5)  # 0-based

# derive floating SP bin location and effective scattering area A_eff
for sec in range(len(transmitter_id)):
    t0 = timer()

    # retrieve rx positions and velocities
    rx_pos_xyz1 = np.array([rx_pos_x[sec], rx_pos_y[sec], rx_pos_z[sec]])
    rx_vel_xyz1 = np.array([rx_vel_x[sec], rx_vel_y[sec], rx_vel_z[sec]])
    rx_clk_drift1 = rx_clk_drift_mps[sec]
    rx1 = {
        "rx_pos_xyz": rx_pos_xyz1,
        "rx_vel_xyz": rx_vel_xyz1,
        "rx_clk_drift": rx_clk_drift1,
    }

    for ngrx_channel in range(J_2):
        # retrieve tx positions and velocities
        tx_pos_xyz1 = np.array(
            [
                tx_pos_x[sec][ngrx_channel],
                tx_pos_y[sec][ngrx_channel],
                tx_pos_z[sec][ngrx_channel],
            ]
        )
        tx_vel_xyz1 = np.array(
            [
                tx_vel_x[sec][ngrx_channel],
                tx_vel_y[sec][ngrx_channel],
                tx_vel_z[sec][ngrx_channel],
            ]
        )

        tx1 = {'tx_pos_xyz': tx_pos_xyz1, 'tx_vel_xyz': tx_vel_xyz1}

        # retrieve sx-related parameters
        sx_pos_xyz1 = np.array(
            [
                sx_pos_x[sec][ngrx_channel],
                sx_pos_y[sec][ngrx_channel],
                sx_pos_z[sec][ngrx_channel],
            ]
        )
        lon, lat, alt = pyproj.transform(ecef, lla, *sx_pos_xyz1, radians=False)
        sx_pos_lla1 = [lat, lon, alt]

        sx_inc_angle1 = sx_inc_angle[sec][ngrx_channel]
        sx_d_snell_deg1 = sx_d_snell_angle[sec][ngrx_channel]
        dist_to_coast1 = dist_to_coast_km[sec][ngrx_channel]

        sx1 = {'sx_pos_xyz': sx_pos_xyz1,
               'sx_d_snell': sx_d_snell_deg1,
               'dist_to_coast': dist_to_coast1}

        # retrieve ddm-related variables
        raw_counts1 = raw_counts[sec, ngrx_channel, :, :]
        add_range_to_sp1 = add_range_to_sp[sec][ngrx_channel]
        snr_db1 = snr_db[sec][ngrx_channel]

        delay_center_chips1 = delay_center_chips[sec][ngrx_channel]

        doppler_center_hz1 = doppler_center_hz[sec][ngrx_channel]

        T_coh1 = coherent_duration[sec]
        ddm1 = {'raw_counts': raw_counts1,
                'add_range_to_sp': add_range_to_sp1,
                'snr_db': snr_db1,
                'delay_center_chips': delay_center_chips1,
                'doppler_center_hz': doppler_center_hz1,
                'T_coh': T_coh1,
                'delay_resolution': 0.25,
                'num_delay_bins': 40,
                'delay_center_bin': 20,
                'doppler_resolution': 500,
                'num_doppler_bins': 5,
                'doppler_center_bin': 2}

        if (not np.isnan(sx_pos_x[sec][ngrx_channel])) and (np.count_nonzero(raw_counts1) > 0):
            # Part 4.3: SP-related variables - 2
            # this part derives confidence and floating bin locations of SP
            peak_delay_bin1, peak_doppler_bin1 = np.unravel_index(raw_counts1.argmax(), raw_counts1.shape)

            specular_bin1, zenith_code_phase1, confidence_flag1 = get_specular_bin(tx1, rx1, sx1, ddm1)

            sx1['sx_delay_bin'] = specular_bin1[0]
            sx1['sx_doppler_bin'] = specular_bin1[1]

            # Part 4.4a: Effective scattering area
            L = 18030
            grid_res = 30   # L may need to be updated in the future

            local_dem1 = get_local_dem_new(sx_pos_lla1, L, grid_res, dem, dtu10, dist_to_coast1)

            A_eff1, A_eff_all1 = get_ddm_Aeff(tx1, rx1, sx1, local_dem1, phy_ele_size, chi2)

            # save to variables
            brcs_ddm_peak_bin_delay_row[sec][ngrx_channel] = peak_delay_bin1   # 0-based indces
            brcs_ddm_peak_bin_dopp_col[sec][ngrx_channel] = peak_doppler_bin1

            brcs_ddm_sp_bin_delay_row[sec][ngrx_channel] = specular_bin1[0]
            brcs_ddm_sp_bin_dopp_col[sec][ngrx_channel] = specular_bin1[1]
            sp_delay_error[sec][ngrx_channel] = specular_bin1[2]
            sp_dopp_error[sec][ngrx_channel] = specular_bin1[3]

            zenith_code_phase[sec][ngrx_channel] = zenith_code_phase1

            confidence_flag[sec][ngrx_channel] = confidence_flag1

            A_eff[sec, ngrx_channel, :, :] = A_eff1.T
            A_eff_all[sec, ngrx_channel, :, :] = A_eff_all1.T
    print(f'******** finish processing part 4B {sec} second data with {timer() - t0}********')

# extend to RHCP channels
brcs_ddm_peak_bin_delay_row[:, J_2:J] = brcs_ddm_peak_bin_delay_row[:, 0:J_2]
brcs_ddm_peak_bin_dopp_col[:, J_2:J] = brcs_ddm_peak_bin_dopp_col[:, 0:J_2]

brcs_ddm_sp_bin_delay_row[:, J_2:J] = brcs_ddm_sp_bin_delay_row[:, 0:J_2]
brcs_ddm_sp_bin_dopp_col[:, J_2:J] = brcs_ddm_sp_bin_dopp_col[:, 0:J_2]
sp_delay_error[:, J_2:J] = sp_delay_error[:, 0:J_2]
sp_dopp_error[:, J_2:J] = sp_dopp_error[:, 0:J_2]

zenith_code_phase[:, J_2:J] = zenith_code_phase[:, 0:J_2]

confidence_flag[:, J_2:J] = confidence_flag[:, 0:J_2]

A_eff[:, J_2:J, :, :] = A_eff[:, 0:J_2, :, :]
A_eff_all[:, J_2:J, :, :] = A_eff_all[:, 0:J_2, :, :]

# save variables
L1_postCal['brcs_ddm_peak_bin_delay_row'] = brcs_ddm_peak_bin_delay_row     # checked ok
L1_postCal['brcs_ddm_peak_bin_dopp_col'] = brcs_ddm_peak_bin_dopp_col       # checked ok

L1_postCal['brcs_ddm_sp_bin_delay_row'] = brcs_ddm_sp_bin_delay_row         # checked diff 0.001
L1_postCal['brcs_ddm_sp_bin_dopp_col'] = brcs_ddm_sp_bin_dopp_col           # checked diff 0.001

L1_postCal['sp_delay_error'] = sp_delay_error                               # checked diff 1 e-4
L1_postCal['sp_dopp_error'] = sp_dopp_error                                 # checked diff 1 / e2
L1_postCal['sp_ngrx_delay_correction'] = sp_delay_error
L1_postCal['sp_ngrx_dopp_correction'] = sp_dopp_error

L1_postCal['zenith_code_phase'] = zenith_code_phase                         # checked ok

L1_postCal['confidence_flag'] = confidence_flag                             # checked ok

L1_postCal['eff_scatter'] = A_eff
L1_postCal['A_eff_all'] = A_eff_all

######################################################################
# np.save('debug1.npy', L1_postCal)

######################################################################
# L1_postCal = np.load('debug1.npy', allow_pickle=True).item()
#
# sx_pos_x = L1_postCal['sp_pos_x']
# sx_pos_y = L1_postCal['sp_pos_y']
# sx_pos_z = L1_postCal['sp_pos_z']
#
# sx_lat = L1_postCal['sp_lat']
# sx_lon = L1_postCal['sp_lon']
# sx_alt = L1_postCal['sp_alt']
#
# sx_vel_x = L1_postCal['sp_vel_x']
# sx_vel_y = L1_postCal['sp_vel_y']
# sx_vel_z = L1_postCal['sp_vel_z']
#
# sx_inc_angle = L1_postCal['sp_inc_angle']
# sx_d_snell_angle = L1_postCal['sp_d_snell_angle']
#
# surface_type = L1_postCal['sp_surface_type']
# dist_to_coast_km = L1_postCal['sp_dist_to_coast_km']
# LOS_flag = L1_postCal['LOS_flag']
#
# static_gps_eirp = L1_postCal['static_gps_eirp']
# gps_tx_power_db_w = L1_postCal['gps_tx_power_db_w']
# gps_ant_gain_db_i = L1_postCal['gps_ant_gain_db_i']
#
# sx_rx_gain = L1_postCal['sp_rx_gain']
#
# rx_to_sp_range = L1_postCal['rx_to_sp_range']
# tx_to_sp_range = L1_postCal['tx_to_sp_range']
#
# A_eff = L1_postCal['eff_scatter']
#
# brcs_ddm_sp_bin_delay_row = L1_postCal['brcs_ddm_sp_bin_delay_row']
# brcs_ddm_sp_bin_dopp_col = L1_postCal['brcs_ddm_sp_bin_dopp_col']
# ######################################################################


# derive brcs, nbrcs, and other parameters
for sec in range(len(transmitter_id)):
    t0 = timer()
    for ngrx_channel in range(J):

        # variables for deriving BRCS and reflectivity
        tx_pos_xyz1 = [tx_pos_x[sec][ngrx_channel], tx_pos_y[sec][ngrx_channel], tx_pos_z[sec][ngrx_channel]]
        rx_pos_xyz1 = [rx_pos_x[sec], rx_pos_y[sec], rx_pos_z[sec]]
        sx_pos_xyz1 = [sx_pos_x[sec][ngrx_channel], sx_pos_y[sec][ngrx_channel], sx_pos_z[sec][ngrx_channel]]

        inc_angle1 = sx_inc_angle[sec][ngrx_channel]
        dist_to_coast1 = dist_to_coast_km[sec][ngrx_channel]

        eirp_watt1 = static_gps_eirp[sec][ngrx_channel]
        rx_gain_db_i1 = sx_rx_gain[sec][ngrx_channel]
        TSx1 = tx_to_sp_range[sec][ngrx_channel]
        RSx1 = rx_to_sp_range[sec][ngrx_channel]

        ddm_ant1 = ddm_ant[sec][ngrx_channel]

        # retrieve ddm-related variables
        raw_counts1 = ddm_power_counts[sec, ngrx_channel, :, :]
        snr_db1 = snr_db[sec][ngrx_channel]

        power_analog1 = power_analog[sec, ngrx_channel, :, :]  # L1a calibrated power watts

        if (not np.isnan(ddm_ant1)) and (not np.isnan(sx_pos_x[sec][ngrx_channel])) and (np.count_nonzero(raw_counts1) > 0):

            # compensate cable loss
            cable_loss_db = 0.0

            if ddm_ant1 == 1:
                cable_loss_db = 0.6600         # LHCP cable loss

            if ddm_ant1 == 2:
                cable_loss_db = 0.5840         # RHCP cable loss

            cable_loss = db2power(cable_loss_db)
            power_analog_cable_loss1 = power_analog1 * cable_loss

            # Part 4.4b: brcs, nbrcs, LES and TES
            # power factor is e-20  which amplifies the difference of power_analog_cable_loss1 between matlab and python
            brcs1 = ddm_brcs(power_analog_cable_loss1, eirp_watt1, rx_gain_db_i1, TSx1, RSx1)

            A_eff1 = A_eff[sec, ngrx_channel, :, :]
            sx_bin1 = np.zeros(2)
            sx_bin1[0] = brcs_ddm_sp_bin_delay_row[sec][ngrx_channel]
            sx_bin1[1] = brcs_ddm_sp_bin_dopp_col[sec][ngrx_channel]

            # the below computes two versions of NBRCS
            # version 1: smaller area, version 2: larger area
            nbrcs_v1_1, nbrcs_scatter_v1_1 = get_ddm_nbrcs2(brcs1, A_eff1, sx_bin1, 1)
            nbrcs_v2_1, nbrcs_scatter_v2_1 = get_ddm_nbrcs2(brcs1, A_eff1, sx_bin1, 2)

            # Part 4.5: reflectivity and peak reflectivity
            refl1, refl_peak1 = ddm_refl(power_analog_cable_loss1, eirp_watt1, rx_gain_db_i1, TSx1, RSx1)

            # Part 4.6: Fresnel coefficient and dimensions
            fresnel_coeff1, fresnel_axis1, fresnel_orientation1 = get_fresnel(tx_pos_xyz1, rx_pos_xyz1, sx_pos_xyz1,
                                                                              dist_to_coast1, inc_angle1,ddm_ant1)

            # Part 4.7: coherent status
            CR1, CS1 = coh_det(raw_counts1, snr_db1)

            # normalised reflected waveform
            refl_waveform1 = refl1.sum(axis=1)
            norm_refl_waveform1 = refl_waveform1 / refl_peak1

            # save to variables
            brcs[sec, ngrx_channel] = brcs1

            nbrcs_scatter_area_v1[sec][ngrx_channel] = nbrcs_scatter_v1_1
            ddm_nbrcs_v1[sec][ngrx_channel] = nbrcs_v1_1

            nbrcs_scatter_area_v2[sec][ngrx_channel] = nbrcs_scatter_v2_1
            ddm_nbrcs_v2[sec][ngrx_channel] = nbrcs_v2_1

            surface_reflectivity[sec][ngrx_channel] = refl1
            surface_reflectivity_peak[sec][ngrx_channel] = refl_peak1

            fresnel_coeff[sec][ngrx_channel] = fresnel_coeff1
            fresnel_major[sec][ngrx_channel] = fresnel_axis1[0]
            fresnel_minor[sec][ngrx_channel] = fresnel_axis1[1]
            fresnel_orientation[sec][ngrx_channel] = fresnel_orientation1

            coherency_ratio[sec][ngrx_channel] = CR1
            coherency_state[sec][ngrx_channel] = CS1

            norm_refl_waveform[sec][ngrx_channel][:, 0] = norm_refl_waveform1
    print(f'******** finish processing part 4B-2 {sec} second data with {timer() - t0}********')

L1_postCal['brcs'] = brcs

L1_postCal['nbrcs_scatter_area_v1'] = nbrcs_scatter_area_v1             # checked diff 1000 / e6
L1_postCal['ddm_nbrcs_v1'] = ddm_nbrcs_v1                               # checked diff 0.1

L1_postCal['nbrcs_scatter_area_v2'] = nbrcs_scatter_area_v2             # checked diff 100000 / e8
L1_postCal['ddm_nbrcs_v2'] = ddm_nbrcs_v2                               # checked diff 0.1

L1_postCal['surface_reflectivity'] = surface_reflectivity
L1_postCal['surface_reflectivity_peak'] = surface_reflectivity_peak     # checked diff 0.01

L1_postCal['fresnel_coeff'] = fresnel_coeff                             # checked ok
L1_postCal['fresnel_major'] = fresnel_major                             # checked diff 0.01
L1_postCal['fresnel_minor'] = fresnel_minor                             # checked diff 0.1 / e2
L1_postCal['fresnel_orientation'] = fresnel_orientation

L1_postCal['coherency_ratio'] = coherency_ratio                         # checked ok
L1_postCal['coherency_state'] = coherency_state                         # checked ok

L1_postCal['norm_refl_waveform'] = norm_refl_waveform
#
# ######################################################################
# np.save('debug2.npy', L1_postCal)
#
# ######################################################################

#  L1_postCal = np.load('debug2.npy', allow_pickle=True).item()
#
#  sx_pos_x = L1_postCal['sp_pos_x']
#  sx_pos_y = L1_postCal['sp_pos_y']
#  sx_pos_z = L1_postCal['sp_pos_z']
#
#  sx_lat = L1_postCal['sp_lat']
#  sx_lon = L1_postCal['sp_lon']
#  sx_alt = L1_postCal['sp_alt']
#
#  sx_vel_x = L1_postCal['sp_vel_x']
#  sx_vel_y = L1_postCal['sp_vel_y']
#  sx_vel_z = L1_postCal['sp_vel_z']
#
#  sx_inc_angle = L1_postCal['sp_inc_angle']
#  sx_d_snell_angle = L1_postCal['sp_d_snell_angle']
#
#  surface_type = L1_postCal['sp_surface_type']
#  dist_to_coast_km = L1_postCal['sp_dist_to_coast_km']
#  LOS_flag = L1_postCal['LOS_flag']
#
#  static_gps_eirp = L1_postCal['static_gps_eirp']
#  gps_tx_power_db_w = L1_postCal['gps_tx_power_db_w']
#  gps_ant_gain_db_i = L1_postCal['gps_ant_gain_db_i']
#
#  sx_rx_gain = L1_postCal['sp_rx_gain']
#
#  rx_to_sp_range = L1_postCal['rx_to_sp_range']
#  tx_to_sp_range = L1_postCal['tx_to_sp_range']
#
#  brcs_ddm_sp_bin_delay_row = L1_postCal['brcs_ddm_sp_bin_delay_row']
#  brcs_ddm_sp_bin_dopp_col = L1_postCal['brcs_ddm_sp_bin_dopp_col']
# ######################################################################

# Cross Pol

tx_pos_x = np.full([*transmitter_id.shape], np.nan)
nbrcs_cross_pol_v1 = np.full([*transmitter_id.shape], np.nan)
nbrcs_cross_pol_v2 = np.full([*transmitter_id.shape], np.nan)

for sec in range(len(transmitter_id)):
    for ngrx_channel in range(J_2):

        nbrcs_LHCP_v1 = ddm_nbrcs_v1[sec][ngrx_channel]
        nbrcs_RHCP_v1 = ddm_nbrcs_v1[sec][ngrx_channel + J_2]
        
        nbrcs_LHCP_v2 = ddm_nbrcs_v2[sec][ngrx_channel]
        nbrcs_RHCP_v2 = ddm_nbrcs_v2[sec][ngrx_channel + J_2]

        CP1 = nbrcs_LHCP_v1 / nbrcs_RHCP_v1
        CP_db1 = power2db(CP1)

        CP2 = nbrcs_LHCP_v2 / nbrcs_RHCP_v2
        CP_db2 = power2db(CP2)

        nbrcs_cross_pol_v1[sec][ngrx_channel] = CP_db1
        nbrcs_cross_pol_v2[sec][ngrx_channel] = CP_db2


nbrcs_cross_pol_v1[:, 10:19] = -1 * nbrcs_cross_pol_v1[:, 0:9]
nbrcs_cross_pol_v2[:, 10:19] = -1 * nbrcs_cross_pol_v2[:, 0:9]

L1_postCal['nbrcs_cross_pol_v1'] = nbrcs_cross_pol_v1
L1_postCal['nbrcs_cross_pol_v2'] = nbrcs_cross_pol_v2
L1_postCal['lna_noise_figure'] = np.full([*transmitter_id.shape], 3)     # LNA noise figure is 3 dB according to the specification

# Quality Flags

quality_flags1 = np.full([*transmitter_id.shape], np.nan)

for sec in range(len(transmitter_id)):
    for ngrx_channel in range(J):
        quality_flag1_1 = np.full([23, 1], 0)

        # flag 1, 2 and 22  0-based indexing
        rx_roll1 = rx_roll[sec]
        rx_pitch1 = rx_pitch[sec]
        rx_yaw1 = rx_yaw[sec]
    
        if (rx_roll1 >= 29) or (rx_pitch1 >= 9) or (rx_yaw1 >= 4):  # 0-based indexing
            quality_flag1_1[2] = 1
        else:
            quality_flag1_1[1] = 1

        if rx_roll1 > 1:
            quality_flag1_1[22] = 1

        # flag 3   0-based indexing
        quality_flag1_1[3] = 0

        # flag 4 and 5
        trans_id1 = transmitter_id[sec][ngrx_channel]
        if trans_id1 == 0:
            quality_flag1_1[4] = 1

        if trans_id1 == 28:
            quality_flag1_1[5] = 1

        # flag 6 and 9
        snr_db1 = snr_db[sec][ngrx_channel]

        if sec > 0:  # 0-based indexing
            snr_db2 = snr_db[sec-1][ngrx_channel]
            diff1 = (db2power(snr_db1) - db2power(snr_db2)) / db2power(snr_db1)
            diff2 = snr_db1 - snr_db2

            if abs(diff1) > 0.1:
                quality_flag1_1[6] = 1

            if abs(diff2) > 0.24:
                quality_flag1_1[9] = 1

        # flag 7 and 8
        dist_to_coast1 = dist_to_coast_km[sec][ngrx_channel]

        if dist_to_coast1 > 0:
            quality_flag1_1[7] = 1

        if dist_to_coast1 > -25:
            quality_flag1_1[8] = 1

        # flag 10
        ant_temp1 = ant_temp_nadir[sec]
        if sec > 0:
            ant_temp2 = ant_temp_nadir[sec - 1]
            rate = (ant_temp2 - ant_temp1) * 60

            if rate > 1:
                quality_flag1_1[10] = 1

        # flag 11
        zenith_code_phase1 = zenith_code_phase[sec][ngrx_channel]
        signal_code_phase1 = delay_correction(meter2chips(add_range_to_sp[sec][ngrx_channel]), 1023)
        diff1 = zenith_code_phase1 - signal_code_phase1
        if diff1 >= 10:
            quality_flag1_1[11] = 1

        # flag 14 and 15
        sp_delay_row = brcs_ddm_sp_bin_delay_row[sec][ngrx_channel] if not np.isnan(brcs_ddm_sp_bin_delay_row[sec][ngrx_channel]) else 0
        sp_dopp_col = brcs_ddm_sp_bin_dopp_col[sec][ngrx_channel] if not np.isnan(brcs_ddm_sp_bin_dopp_col[sec][ngrx_channel]) else 0

        if (sp_delay_row < 15) or (sp_delay_row > 35):
            quality_flag1_1[14] = 1

        if (sp_dopp_col < 2) or (sp_dopp_col > 4):
            quality_flag1_1[15] = 1

        # flag 16
        if ((math.floor(sp_delay_row) < 38) and (math.floor(sp_delay_row) > 0) and
            (math.floor(sp_dopp_col) < 5) and (math.floor(sp_dopp_col) > 1)):
            sp_dopp_col_range = list(range(math.floor(sp_dopp_col) - 1, math.floor(sp_dopp_col) + 2))
            sp_delay_raw_range = list(range(math.floor(sp_delay_row), math.floor(sp_dopp_col) + 4))  # TODO: sp_dopp_col, again?
            brcs_ddma = brcs[sp_delay_raw_range, :][:, sp_dopp_col_range]
            det = brcs_ddma[brcs_ddma < 0]
            if len(det) > 0:
                quality_flag1_1[16] = 1

        # flag 17
        tx_pos_x1 = tx_pos_x[sec][ngrx_channel]
        prn_code1 = prn_code[sec][ngrx_channel]
        if (tx_pos_x1 == 0) and (not np.isnan(prn_code1)):
            quality_flag1_1[17] = 1

        # flag 18
        sx_pos_x1 = sx_pos_x[sec][ngrx_channel]
        if np.isnan(sx_pos_x1) and (not np.isnan(prn_code1)):
            quality_flag1_1[18] = 1

        # flag 19
        rx_gain1 = sx_rx_gain[sec][ngrx_channel]
        if np.isnan(rx_gain1) and (not np.isnan(prn_code1)):
            quality_flag1_1[19] = 1

        quality_flag1_1[20] = 1

        # flag 21
        rx_alt = rx_pos_lla[2][sec]
        if rx_alt > 15000:
            quality_flag1_1[21] = 1

        # flag 1
        if (quality_flag1_1[2]  == 1 or
            quality_flag1_1[3]  == 1 or
            quality_flag1_1[4]  == 1 or
            quality_flag1_1[5]  == 1 or
            quality_flag1_1[6]  == 1 or
            quality_flag1_1[9]  == 1 or
            quality_flag1_1[10] == 1 or
            quality_flag1_1[11] == 1 or
            quality_flag1_1[12] == 1 or
            quality_flag1_1[13] == 1 or
            quality_flag1_1[14] == 1 or
            quality_flag1_1[15] == 1 or
            quality_flag1_1[16] == 1 or
            quality_flag1_1[17] == 1 or
            quality_flag1_1[19] == 1 or
            quality_flag1_1[21] == 1 or
            quality_flag1_1[22] == 1):
        
            quality_flag1_1[0] = 1

        quality_flags1[sec][ngrx_channel] = get_quality_flag(quality_flag1_1)
        

L1_postCal['quality_flags1'] = quality_flags1

np.save('debug3.npy', L1_postCal)

# packet to netCDF

# netCDF_name = './out/sample1.nc'
# ncfile = nc.Dataset(netCDF_name, 'w', format='NETCDF4')
# for k, v in L1_postCal.items():
#     setattr(ncfile, k, v)
# ncfile.close()  # ValueError: multi-dimensional array attributes not supported

# L1 calibration ends
