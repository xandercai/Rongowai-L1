import numpy as np
import math
from pathlib import Path
from scipy.interpolate import interp1d


# load L1a calibration tables
L1a_path = Path().absolute().joinpath(Path("./dat/L1a_cal/"))
L1a_cal_ddm_counts_db_filename = Path("L1A_cal_ddm_counts_dB.dat")
L1a_cal_ddm_power_dbm_filename = Path("L1A_cal_ddm_power_dBm.dat")
L1a_cal_ddm_counts_db = np.loadtxt(L1a_path.joinpath(L1a_cal_ddm_counts_db_filename))
L1a_cal_ddm_power_dbm = np.loadtxt(L1a_path.joinpath(L1a_cal_ddm_power_dbm_filename))

# offset delay rows to derive noise floor
offset = 4
# map rf_source to ANZ_port
ANZ_port = {0: 0, 4: 1, 8: 2}
binning_thres_db = [50.5, 49.6, 50.4]

# create the interpolation functions for the 3 ports
L1a_cal_1dinterp = {}
for i in range(3):
    L1a_cal_1dinterp[i] = interp1d(
        L1a_cal_ddm_counts_db[i, :],
        L1a_cal_ddm_power_dbm[i, :],
        kind="cubic",
        fill_value="extrapolate",
    )


def get_ANZ_port(rf_source):
    """Returns the ANZ port for the given rf_source

    Parameters
    ----------
    rf_source : int
        Radio frequency source value (RF1/RF2/RF3)

    Returns
    -------
    int
        ANZ port value
    """
    if rf_source == 0:
        # zenith
        # anz_port = 1
        anz_port = 0
    elif rf_source == 4:
        # nadir LHCP
        # anz_port = 2
        anz_port = 1
    elif rf_source == 8:
        # nadir RHCP
        # anz_port = 3
        anz_port = 2
    else:
        assert False, f"Invalid rf_source value {rf_source}"
    return anz_port


def power2db(power):
    return 10 * math.log10(power)

def db2power(db):
    return np.power(10, db/10)

def L1a_counts2watts(ddm_counts, ANZ_port, ddm_counts_cal_db, ddm_power_cal_dbm, std_dev):
    """Converts raw DDM counts to DDM power in watts

    Parameters
    ----------
    ddm_counts : np.array()
        Scaled counts of a given DDM
    std_dev : List[float,float,float]
        List of binning standard deviation values per sec per RF1/RF2/RF3
    rf_source : int
        Radio frequency source value (RF1/RF2/RF3) to determine std to use

    Returns
    -------
    ddm_power_dbm : numpy.array(numpy.float64)
        Returns DDM as calibrated power in Watts
    """
    binning_thres_db = [50.5, 49.6, 50.4]
    # cable_loss_db = [1.8051, 0.6600, 0.5840]

    # select approiate calibration constants based on the input ANZ port channel
    ddm_counts_db_ch = ddm_counts_cal_db[ANZ_port]
    ddm_power_dbm_ch = ddm_power_cal_dbm[ANZ_port]

    std_dev_ch = std_dev[ANZ_port]

    binning_thres_db_ch = binning_thres_db[ANZ_port]
    # cable_loss_db_ch = cable_loss_db[ANZ_port]

    # convert to dB scale
    ddm_counts_db = 10 * np.log10(ddm_counts)
    std_dev_db_ch = 20 * np.log10(std_dev_ch)

    # evaluate ddm power in dBm
    # Scipy doesn't like masked arrays, so undo here and reply after
    ddm_power_dbm = L1a_cal_1dinterp[ANZ_port](np.ma.getdata(ddm_counts_db))
    ddm_power_dbm = (
        ddm_power_dbm + std_dev_db_ch - binning_thres_db[ANZ_port]  # + cable_loss_db_ch
    )
    # ? reapply mask to array to hide nonsense interp.
    # ? ddm_power_dbm = np.ma.masked_where(np.ma.getmask(ddm_counts_db), ddm_power_dbm)
    # convert to watts (TODO - why 30?)
    return 10 ** ((ddm_power_dbm - 30) / 10)


def ddm_calibration(
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
):
    """Calibrates raw DDMs into power DDMs in Watts

    Parameters
    ----------
    std_dev_rf1 : numpy.array()
        Binning standard deviation of RF1 channel
    std_dev_rf2 : numpy.array()
        Binning standard deviation of RF2 channel
    std_dev_rf3 : numpy.array()
        Binning standard deviation of RF3 channel
    J : int
        Number of NGRX_channels to iterate over
    prn_code : numpy.array()
        Array of PRNs of satellites
    raw_counts : 4D numpy.array()
        Array of DDMs per ngrx_channel (J) per second of flight
    rf_source : numpy.array()
        Array of RF sources (RF1/RF2/RF3) per ngrx_channel per second of flight
    first_scale_factor : numpy.array()
        Scale factor to calibrate DDM raw counts
    ddm_power_counts : numpy.array()
        Empty 4D array to receive calibrated counts of DDMs
    power_analog : numpy.array()
        Empty 4D array to recieve calibrated powers of DDMs in Watts
    ddm_ant : numpy.array()
        Empty array to receive ANZ_port of each DDM
    ddm_noise_counts : numpy.array()
        Empty array to receive estimate of noise per DDM
    ddm_noise_watts : numpy.array()
        Empty array to receive estimate of noise per DDM in Watts
    peak_ddm_counts : numpy.array()
        Empty array to receive maximum counts per calibrated DDM
    peak_ddm_watts : numpy.array()
        Empty array to receive maximum Power per calibrated DDM in Watts
    peak_delay_bin : numpy.array()
        Empty array to receive delay_bin of peak count value per calibrated DDM
    noise_floor_counts : numpy.array()
        Empty array to receive noise floor per calibrated DDM
    noise_floor : numpy.array()
        Empty array to receive noise floor per calibrated DDM in Watts
    inst_gain : numpy.array()
        Empty array to receive instantaneous gain per calibrated DDM
    snr_db : numpy.array()
        Empty array to receive SNR per calibrated DDM in dB
    """
    # TODO - what to do about partial DDMs?
    # derive signal power
    # iterate over seconds of flight
    for sec in range(len(std_dev_rf1)):
        # retrieve noise standard deviation in counts for all three channels
        # bundle std_X[i] values for ease
        std_dev1 = [std_dev_rf1[sec], std_dev_rf2[sec], std_dev_rf3[sec]]
        # iterate over the 20 NGRX_channels
        for ngrx_channel in range(J):
            # assign local variables for PRN and DDM counts
            prn_code1 = prn_code[sec, ngrx_channel]
            rf_source1 = rf_source[sec, ngrx_channel]
            first_scale_factor1 = first_scale_factor[sec, ngrx_channel]
            raw_counts1 = raw_counts[sec, ngrx_channel, :, :]
            # solve only when presenting a valid PRN and DDM counts
            if (not np.isnan(prn_code1)) and (raw_counts1[0, 0] != raw_counts1[20, 2]) and (raw_counts1[1, 1] != 0):
                # scale raw counts and convert from counts to watts
                ANZ_port1 = get_ANZ_port(rf_source1)
                ddm_power_counts1 = raw_counts1 * first_scale_factor1

                # perform L1a calibration from Counts to Watts
                ddm_power_watts1 = L1a_counts2watts(ddm_power_counts1, ANZ_port1, L1a_cal_ddm_counts_db, L1a_cal_ddm_power_dbm, std_dev1)

                # noise floor in counts for each DDM
                # determine noise counts from offset value
                ddm_noise_counts1 = np.mean(ddm_power_counts1[-offset - 1 :, :])
                ddm_noise_watts1 = L1a_counts2watts(ddm_noise_counts1, ANZ_port1, L1a_cal_ddm_counts_db, L1a_cal_ddm_power_dbm, std_dev1)

                # peak ddm location
                # find peak counts/watts/delay from DDM data
                peak_ddm_counts1 = np.max(ddm_power_counts1)
                # 0-based index
                peak_delay_bin1 = np.where(ddm_power_counts1 == peak_ddm_counts1)[0][0]

                peak_ddm_watts1 = np.max(ddm_power_watts1)

                # save variables
                ddm_power_counts[sec][ngrx_channel] = ddm_power_counts1
                power_analog[sec][ngrx_channel] = ddm_power_watts1
                # 0-based index
                ddm_ant[sec][ngrx_channel] = ANZ_port1
                ddm_noise_counts[sec][ngrx_channel] = ddm_noise_counts1
                ddm_noise_watts[sec][ngrx_channel] = ddm_noise_watts1

                peak_ddm_counts[sec][ngrx_channel] = peak_ddm_counts1
                peak_ddm_watts[sec][ngrx_channel] = peak_ddm_watts1
                # this is 0-based
                peak_delay_bin[sec][ngrx_channel] = peak_delay_bin1

    # derive noise floor, SNR and instrument gain
    for sec in range(len(std_dev_rf1)):
        noise_counts_LHCP1 = ddm_noise_counts[sec, 0:int(J/2)]
        noise_watts_LHCP1 = ddm_noise_watts[sec, 0:int(J/2)]
        peak_delay_bin_LHCP1 = peak_delay_bin[sec, 0:int(J/2)]

        noise_counts_RHCP1 = ddm_noise_counts[sec, int(J/2):J]
        noise_watts_RHCP1 = ddm_noise_watts[sec, int(J/2):J]
        peak_delay_bin_RHCP1 = peak_delay_bin[sec, int(J/2):J]

        # TODO Q: why is 31?
        noise_index_LHCP = np.where(np.logical_and(peak_delay_bin_LHCP1 < 31, peak_delay_bin_LHCP1 > 0))[0]
        noise_index_RHCP = np.where(np.logical_and(peak_delay_bin_RHCP1 < 31, peak_delay_bin_RHCP1 > 0))[0]

        if len(noise_index_LHCP) > 0:
            avg_noise_counts_LHCP1 = np.mean(noise_counts_LHCP1[noise_index_LHCP])
            avg_noise_watts_LHCP1 = np.mean(noise_watts_LHCP1[noise_index_LHCP])

            avg_noise_counts_RHCP1 = np.mean(noise_counts_RHCP1[noise_index_RHCP])
            avg_noise_watts_RHCP1 = np.mean(noise_watts_RHCP1[noise_index_RHCP])

        elif (len(noise_index_LHCP) == 0) and (np.any(~np.isnan(noise_counts_LHCP1))):
            # TODO Q: the index operation is strange, j (ngrx_channel) is out of loops.
            avg_noise_counts_LHCP1 = noise_floor_counts[sec - 1, ngrx_channel]
            avg_noise_watts_LHCP1 = noise_floor[sec - 1, ngrx_channel]

            avg_noise_counts_RHCP1 = noise_floor_counts[sec - 1, ngrx_channel + int(J/2)]
            avg_noise_watts_RHCP1 = noise_floor[sec - 1, ngrx_channel + int(J/2)]

        for ngrx_channel in range(int(J/2)):
            peak_power_counts_LHCP1 = peak_ddm_counts[sec, ngrx_channel]
            peak_signal_watts_LHCP1 = peak_ddm_watts[sec, ngrx_channel]

            peak_power_counts_RHCP1 = peak_ddm_counts[sec, ngrx_channel + int(J/2)]
            peak_signal_watts_RHCP1 = peak_ddm_watts[sec, ngrx_channel + int(J/2)]

            if not math.isnan(peak_power_counts_LHCP1):
                # derive SNR
                snr_LHCP1 = peak_signal_watts_LHCP1 / avg_noise_watts_LHCP1
                snr_db_LHCP1 = power2db(snr_LHCP1)
                snr_RHCP1 = peak_signal_watts_RHCP1 / avg_noise_watts_RHCP1
                snr_db_RHCP1 = power2db(snr_RHCP1)

                # derive instrument gain
                peak_signal_counts_LHCP1 = peak_power_counts_LHCP1 - avg_noise_counts_LHCP1
                inst_gain_LHCP1 = peak_signal_counts_LHCP1 / peak_signal_watts_LHCP1

                peak_signal_counts_RHCP1 = peak_power_counts_RHCP1 - avg_noise_counts_RHCP1
                inst_gain_RHCP1 = peak_signal_counts_RHCP1 / peak_signal_watts_RHCP1

                # save variables
                noise_floor_counts[sec, ngrx_channel] = avg_noise_counts_LHCP1
                noise_floor_counts[sec, ngrx_channel + int(J/2)] = avg_noise_counts_RHCP1

                noise_floor[sec, ngrx_channel] = avg_noise_watts_LHCP1
                noise_floor[sec, ngrx_channel + int(J/2)] = avg_noise_watts_RHCP1

                inst_gain[sec, ngrx_channel] = inst_gain_LHCP1
                inst_gain[sec, ngrx_channel + int(J/2)] = inst_gain_RHCP1

                snr_db[sec, ngrx_channel] = snr_db_LHCP1
                snr_db[sec, ngrx_channel + int(J/2)] = snr_db_RHCP1

def get_quality_flag(quality_flag1):
    quality_flag = (2 ** 22 * quality_flag1[0]  +
                    2 ** 21 * quality_flag1[1]  +
                    2 ** 20 * quality_flag1[2]  +
                    2 ** 19 * quality_flag1[3]  +
                    2 ** 18 * quality_flag1[4]  +
                    2 ** 17 * quality_flag1[5]  +
                    2 ** 16 * quality_flag1[6]  +
                    2 ** 15 * quality_flag1[7]  +
                    2 ** 14 * quality_flag1[8]  +
                    2 ** 13 * quality_flag1[9]  +
                    2 ** 12 * quality_flag1[10] +
                    2 ** 11 * quality_flag1[11] +
                    2 ** 10 * quality_flag1[12] +
                    2 ** 9  * quality_flag1[13] +
                    2 ** 8  * quality_flag1[14] +
                    2 ** 7  * quality_flag1[15] +
                    2 ** 6  * quality_flag1[16] +
                    2 ** 5  * quality_flag1[17] +
                    2 ** 4  * quality_flag1[18] +
                    2 ** 3  * quality_flag1[19] +
                    2 ** 2  * quality_flag1[20] +
                    2 ** 1  * quality_flag1[21] +
                    2 ** 0  * quality_flag1[22])
    return quality_flag
