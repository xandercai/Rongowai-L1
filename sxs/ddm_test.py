import netCDF4 as nc
from load_files import (
    load_netcdf,
    load_antenna_pattern,
    interp_ddm,
    get_orbit_file,
    load_dat_file_grid,
)
from pathlib import Path


raw_data_path = Path().absolute().joinpath(Path("./out/"))
L0_filename = Path("20221103-121416_NZNV-NZCH_L1.nc")
# L0_filenames = glob.glob("*.nc")

L0_dataset = nc.Dataset(raw_data_path.joinpath(L0_filename))

raw_counts = L0_dataset["/l1a_power_ddm"]  # raw counts, uncalibrated
for i in range(20):
    print(raw_counts[0, i, :, :])
    print("####### " + str(i))
