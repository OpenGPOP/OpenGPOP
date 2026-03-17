# INFO
# This script reads real data from space-track.org for given NORAD Catalog ID,
# start date and time span in days. Then it makes test sets with this data.
# Each test data is written in one line in following format:
# <epoch x> <epoch y> <epoch z> <epoch dx> <epoch dy> <epoch dz> <year> <month>
# <day> <minute> <second> <delta seconds> <second x> <second y> <second z>
# <sgp4 error>
# where delta seconds is time from epoch to second position, sgp4 error is the
# error of prediction calculated by sgp4 model from skyfield.
#
# Good parameters for testing:
# --id 37849 --start 2019-05-05 --span 2.

import argparse
import requests
import os
import sys
import json
from datetime import datetime, timedelta
from skyfield.api import EarthSatellite, load, utc
from dotenv import load_dotenv
import math

# Load credentials from .env
load_dotenv()
USERNAME = os.getenv('SPACETRACK_USERNAME')
PASSWORD = os.getenv('SPACETRACK_PASSWORD')


def calculate_distance_between(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2)


def fail(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def sat_from_json(entry):
    tle1 = entry['TLE_LINE1']
    tle2 = entry['TLE_LINE2']

    sat = EarthSatellite(tle1, tle2, entry.get(
        'OBJECT_NAME', 'Unnamed'), ts)
    return sat


# Parse CLI arguments
parser = argparse.ArgumentParser(
    description="Download and convert TLEs to state vectors.")
parser.add_argument('--id', type=int, required=True,
                    help='NORAD Catalog ID of the satellite')
parser.add_argument('--start', type=str, default=datetime.utcnow().strftime(
    '%Y-%m-%d'), help='Start date (YYYY-MM-DD)')
parser.add_argument('--span', type=int, default=1, help='Time span in days')
args = parser.parse_args()

# Validate date
try:
    START_DATE = datetime.strptime(args.start, "%Y-%m-%d")
except ValueError:
    fail("Invalid date format. Use YYYY-MM-DD.")

NORAD_ID = args.id
DAYS = args.span
END_DATE = START_DATE + timedelta(days=DAYS)

# Check for .env credentials
if not USERNAME or not PASSWORD:
    fail("Missing SPACETRACK_USERNAME or SPACETRACK_PASSWORD in .env")

# Build URLs
login_url = 'https://www.space-track.org/ajaxauth/login'
query_url = (
    'https://www.space-track.org/basicspacedata/query/class/omm'
    f'/NORAD_CAT_ID/{NORAD_ID}'
    f'/EPOCH/{START_DATE.strftime("%Y-%m-%d")
    }--{END_DATE.strftime("%Y-%m-%d")}'
    '/orderby/EPOCH asc/format/json'
)

# Authenticate
print("[INFO] Authenticating with Space-Track...")
session = requests.Session()
login_resp = session.post(
    login_url, data={'identity': USERNAME, 'password': PASSWORD})
if login_resp.status_code != 200:
    fail(f"Login failed with status {
    login_resp.status_code}: {login_resp.text}")

# Get data
print("[INFO] Querying TLE data...")
response = session.get(query_url)
if response.status_code != 200:
    fail(f"TLE data query failed with status {
    response.status_code}: {response.text}")

try:
    data = response.json()
except json.JSONDecodeError as e:
    fail(f"Failed to parse JSON: {
    str(e)}\nRaw response:\n{response.text[:500]}")

if not isinstance(data, list):
    fail(f"Expected list of TLE entries, got: {type(data)}")
if len(data) == 0:
    fail("No TLE data returned for the specified date range.")

print(f"[INFO] Received {len(data)} TLE entries.")

# Init Skyfield
ts = load.timescale()
results = []

for i, entry in enumerate(data):
    if i == 0:
        continue
    try:
        # Calculate epoch data
        epoch_sat = sat_from_json(data[i - 1])
        epoch_str = data[i - 1]['EPOCH']
        epoch_dt = datetime.fromisoformat(epoch_str).replace(tzinfo=utc)
        epoch_t = ts.utc(epoch_dt)
        epoch_geocentric = epoch_sat.at(epoch_t)
        epoch_position = epoch_geocentric.position.m
        epoch_velocity = epoch_geocentric.velocity.m_per_s

        # Calculate second time point
        after_time_str = entry['EPOCH']
        sat = sat_from_json(entry)
        dt = datetime.fromisoformat(after_time_str).replace(tzinfo=utc)
        t = ts.utc(dt)
        geocentric = sat.at(t)
        dt = t.utc_datetime()

        # Calculate real position
        after_pos = geocentric.position.m

        # Predict with sgp4 (earth satellites use sgp4 in .at() method)
        predicted_geocentric = epoch_sat.at(t)
        predicted_position = predicted_geocentric.position.m

        sgp4_error = calculate_distance_between(predicted_position, after_pos)

        time_from_epoch = dt - epoch_dt
        results.append((
            epoch_position[0], epoch_position[1], epoch_position[2],
            epoch_velocity[0], epoch_velocity[1], epoch_velocity[2],
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute, dt.second,
            time_from_epoch.total_seconds(),
            after_pos[0], after_pos[1], after_pos[2],
            sgp4_error
        ))

    except Exception as e:
        print(f"[WARNING] Failed to process TLE entry #{
        i}: {e}", file=sys.stderr)
        continue

if not results:
    fail("No valid state vectors generated. Exiting.")

print("[INFO] Writing state vectors and time deltas to file...")
try:
    with open("data/state_vectors.txt", "w") as f:
        for i, r in enumerate(results):
            line = ' '.join(map(str, r)) + '\n'
            f.write(line)
except Exception as e:
    fail(f"Failed to write output file: {str(e)}")

print("[SUCCESS] File written: state_vectors.txt")
