import math
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from datetime import timedelta

import numpy as np
import orekit_jpype

orekit_jpype.initVM()
from astropy import units as u
from astropy.coordinates import ITRS, GCRS, CartesianRepresentation, CartesianDifferential
from astropy.time import Time
from eof.download import download_eofs
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.hipparchus.ode.nonstiff import ClassicalRungeKuttaIntegrator
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit, OrbitType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import IERSConventions
from org.orekit.utils import PVCoordinates, Constants
from tqdm import tqdm
import orekit_jpype.pyhelpers

# --- CONFIGURATION ---
MISSION = 'S1A'
START_DATE = datetime(2023, 5, 1)
DAYS_TO_FETCH = 2
MIN_PROPAGATION_HOURS = 2
MAX_PROPAGATION_HOURS = 24
OUTPUT_FILE = "data/state_vectors.txt"

# Tell Orekit where the physics data zip file is
data_file = os.path.join(os.getcwd(), 'orekit-data-main.zip')
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Please download {data_file} and place it in this directory.")
orekit_jpype.pyhelpers.setup_orekit_data(filenames=data_file)


def fetch_and_convert_to_eci(start_date, days):
    """Fetches Sentinel EOFs and converts ECEF to ECI (GCRS)."""
    dates_to_fetch = [start_date + timedelta(days=i) for i in range(days)]
    print(f"\n[1/3] Fetching {days} days of precise orbit files...")

    # Fix applied: multiplying the MISSION string to match dates list length
    downloaded_files = download_eofs(dates_to_fetch, missions=[MISSION] * len(dates_to_fetch), save_dir='.')

    all_osvs = []
    for eof_file_path in downloaded_files:
        tree = ET.parse(eof_file_path)
        root = tree.getroot()
        all_osvs.extend(root.findall('.//OSV'))

    eci_state_vectors = []

    print(f"\n[2/3] Parsing {len(all_osvs)} state vectors and converting ECEF to ECI...")
    for osv in tqdm(all_osvs, desc="Astropy Conversion", unit="vectors"):
        utc_str = osv.find('UTC').text.replace('UTC=', '')

        x, y, z = float(osv.find('X').text), float(osv.find('Y').text), float(osv.find('Z').text)
        vx, vy, vz = float(osv.find('VX').text), float(osv.find('VY').text), float(osv.find('VZ').text)

        t = Time(utc_str, format='isot', scale='utc')
        itrs_pos = CartesianRepresentation(x=x * u.m, y=y * u.m, z=z * u.m)
        itrs_vel = CartesianDifferential(d_x=vx * u.m / u.s, d_y=vy * u.m / u.s, d_z=vz * u.m / u.s, unit=u.m / u.s)

        itrs_coord = ITRS(itrs_pos.with_differentials(itrs_vel), obstime=t)
        gcrs_coord = itrs_coord.transform_to(GCRS(obstime=t))

        eci_state_vectors.append({
            'datetime': t.to_datetime(),
            'pos': np.array([gcrs_coord.cartesian.x.value,
                             gcrs_coord.cartesian.y.value,
                             gcrs_coord.cartesian.z.value]),
            'vel': np.array([gcrs_coord.cartesian.differentials['s'].d_x.to(u.m / u.s).value,
                             gcrs_coord.cartesian.differentials['s'].d_y.to(u.m / u.s).value,
                             gcrs_coord.cartesian.differentials['s'].d_z.to(u.m / u.s).value])
        })

    eci_state_vectors.sort(key=lambda x: x['datetime'])
    return eci_state_vectors


# Force Model Imports


def calculate_orekit_error(dt_epoch: datetime,
                           target_datetime: datetime,
                           epoch_pos_eci: list,
                           epoch_vel_eci: list,
                           truth_pos_eci: list,
                           mu: float = Constants.WGS84_EARTH_MU) -> float:
    """
    Propagates an initial state to a target datetime and calculates the norm
    of the difference vector against a truth position. Forces are configured
    internally via commenting/uncommenting
    """

    # Setup Time and Dates
    utc = TimeScalesFactory.getUTC()

    def to_absolute_date(dt):
        return AbsoluteDate(dt.year, dt.month, dt.day,
                            dt.hour, dt.minute,
                            float(dt.second) + dt.microsecond / 1e6,
                            utc)

    initial_date = to_absolute_date(dt_epoch)

    final_date = to_absolute_date(target_datetime)

    # Setup Frames and Initial State
    eci_frame = FramesFactory.getGCRF()

    pos_vec = Vector3D(float(epoch_pos_eci[0]), float(epoch_pos_eci[1]), float(epoch_pos_eci[2]))
    vel_vec = Vector3D(float(epoch_vel_eci[0]), float(epoch_vel_eci[1]), float(epoch_vel_eci[2]))
    initial_pv = PVCoordinates(pos_vec, vel_vec)

    initial_orbit = CartesianOrbit(initial_pv, eci_frame, initial_date, mu)
    initial_state = SpacecraftState(initial_orbit)

    # Setup Integrator
    fixed_step_size = 30.0
    integrator = ClassicalRungeKuttaIntegrator(fixed_step_size)

    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(initial_orbit.getType())
    # 4. Setup Propagator
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(initial_state)

    # =========================================================================
    # 5. CONFIGURE FORCES HERE
    # =========================================================================

    # --- A. Simple 2-Body Gravity ---
    # propagator.addForceModel(NewtonianAttraction(mu))

    # --- B. Complex Gravity Field (e.g., J2 to J70) ---
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    gravity_provider = GravityFieldFactory.getNormalizedProvider(70, 70)  # Degree 70, Order 70
    propagator.addForceModel(HolmesFeatherstoneAttractionModel(itrf, gravity_provider))

    # --- C. Third-Body Perturbations ---
    # propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getSun()))
    # propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMoon()))

    # Propagate to Target Date
    final_state = propagator.propagate(final_date)
    final_pos_vec = final_state.getPVCoordinates().getPosition()

    # Calculate Error Norm
    truth_x, truth_y, truth_z = truth_pos_eci

    dx = final_pos_vec.getX() - truth_x
    dy = final_pos_vec.getY() - truth_y
    dz = final_pos_vec.getZ() - truth_z

    error_norm = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    return error_norm


def generate_test_cases():
    sv_list = fetch_and_convert_to_eci(START_DATE, DAYS_TO_FETCH)
    if not sv_list:
        return

    step_size = 60 * 60  # skipping epochs to keep file size sane
    epochs_to_test = list(range(0, len(sv_list), step_size))

    print(f"\n[3/3] Generating {len(epochs_to_test)} test cases with Orekit baseline...")
    results = []

    for i in tqdm(epochs_to_test, desc="Propagating Orbits", unit="case"):
        epoch_sv = sv_list[i]
        dt_epoch = epoch_sv['datetime']

        for j in range(i + 1, len(sv_list), step_size):
            target_sv = sv_list[j]
            time_from_epoch = target_sv['datetime'] - dt_epoch
            hours_from_epoch = time_from_epoch.total_seconds() / 3600.0

            if MIN_PROPAGATION_HOURS <= hours_from_epoch <= MAX_PROPAGATION_HOURS:
                baseline_err = calculate_orekit_error(
                    dt_epoch, target_sv['datetime'],
                    epoch_sv['pos'], epoch_sv['vel'], target_sv['pos']
                )

                results.append((
                    epoch_sv['pos'][0], epoch_sv['pos'][1], epoch_sv['pos'][2],
                    epoch_sv['vel'][0], epoch_sv['vel'][1], epoch_sv['vel'][2],
                    dt_epoch.year, dt_epoch.month, dt_epoch.day,
                    dt_epoch.hour, dt_epoch.minute, dt_epoch.second,
                    time_from_epoch.total_seconds(),
                    target_sv['pos'][0], target_sv['pos'][1], target_sv['pos'][2],
                    baseline_err
                ))

    # Write to a flat text file, space-separated, one case per line
    with open(OUTPUT_FILE, 'w') as f:
        for res in results:
            line = " ".join(str(item) for item in res)
            f.write(f"{line}\n")

    print(f"\nSuccess! Wrote {len(results)} test cases to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_test_cases()
