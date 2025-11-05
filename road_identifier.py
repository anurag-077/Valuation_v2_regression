# road_identifier.py
import requests
import logging
from typing import List, Dict, Optional, Tuple
from shapely.geometry import Point, LineString
from shapely.ops import transform
import pyproj
import numpy as np

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
SEARCH_RADIUS_M = 200                     # look for roads inside this radius
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]

# Your custom A-B-C-D classification (same as in your app)
CATEGORY_BY_OSM = {
    "tertiary": ("A", "Category A – Minor Road (<12 m)", 2.5),
    "residential": ("A", "Category A – Minor Road (<12 m)", 2.5),
    "unclassified": ("A", "Category A – Minor Road (<12 m)", 2.5),
    "service": ("A", "Category A – Minor Road (<12 m)", 2.5),
    "track": ("A", "Category A – Minor Road (<12 m)", 2.5),
    "path": ("A", "Category A – Minor Road (<12 m)", 2.5),
    "living_street": ("A", "Category A – Minor Road (<12 m)", 2.5),
    "pedestrian": ("A", "Category A – Minor Road (<12 m)", 2.5),
    "road": ("A", "Category A – Minor Road (<12 m)", 2.5),

    "secondary": ("B", "Category B – Local Main Road (12–18 m)", 5.0),
    "secondary_link": ("B", "Category B – Local Main Road (12–18 m)", 5.0),

    "primary": ("C", "Category C – Major / Sub-Arterial (18–30 m)", 7.5),
    "primary_link": ("C", "Category C – Major / Sub-Arterial (18–30 m)", 7.5),
    "trunk_link": ("C", "Category C – Major / Sub-Arterial (18–30 m)", 7.5),

    "trunk": ("D", "Category D – Arterial / Highway (30–75 m)", 10.0),
    "motorway": ("D", "Category D – Arterial / Highway (30–75 m)", 10.0),
    "motorway_link": ("D", "Category D – Arterial / Highway (30–75 m)", 10.0),
}

# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def overpass_query(lat: float, lon: float, radius_m: int = SEARCH_RADIUS_M) -> dict:
    """Ask Overpass for every way with a highway tag inside the radius."""
    highway_filter = "|".join(CATEGORY_BY_OSM.keys())
    query = f"""
    [out:json][timeout:30];
    way(around:{radius_m},{lat},{lon})["highway"~"^({highway_filter})$"];
    out geom;
    """
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(endpoint, data={"data": query}, timeout=35)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logging.warning(f"Overpass {endpoint} failed: {e}")
    raise RuntimeError("All Overpass endpoints failed.")

def local_metric_transformer(lat: float, lon: float):
    """Create a transformer that converts WGS-84 to a local metre-based CRS."""
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    try:
        src = pyproj.CRS.from_epsg(4326)
        dst = pyproj.CRS.from_epsg(epsg)
    except Exception:
        src = pyproj.CRS.from_epsg(4326)
        dst = pyproj.CRS.from_epsg(3395)          # fallback World Mercator
    return pyproj.Transformer.from_crs(src, dst, always_xy=True).transform

def linestring_from_way(way: dict) -> Optional[LineString]:
    """Convert Overpass geometry list to a Shapely LineString."""
    geom = way.get("geometry")
    if not geom or len(geom) < 2:
        return None
    coords = [(pt["lon"], pt["lat"]) for pt in geom]
    return LineString(coords)

def category_from_highway(highway_tag: str) -> Tuple[Optional[str], Optional[str], float]:
    """Return (code, label, boost_pct) for a given OSM highway tag."""
    return CATEGORY_BY_OSM.get(highway_tag, (None, None, 0.0))

# ----------------------------------------------------------------------
# MAIN PUBLIC FUNCTION
# ----------------------------------------------------------------------
def identify_road(lat: float, lon: float) -> Tuple[List[Dict], Dict]:
    """
    Return:
        1. List of **all** roads inside the search radius (sorted by distance)
        2. **Nearest / widest** road (the one you will use for valuation)

    Each dict contains:
        - name          : str
        - highway       : OSM tag
        - category      : 'A'|'B'|'C'|'D'
        - category_label: full description
        - distance_m    : float
        - boost_pct     : valuation boost %
        - geometry      : List[Tuple[float,float]]
    """
    data = overpass_query(lat, lon, SEARCH_RADIUS_M)
    ways = [el for el in data.get("elements", []) if el.get("type") == "way"]
    if not ways:
        logging.info("No roads found.")
        return [], {
            "name": None, "highway": None, "category": None,
            "category_label": None, "distance_m": None, "boost_pct": 0.0,
            "geometry": None
        }

    # Local metre projection
    to_metres = local_metric_transformer(lat, lon)
    subject_ll = Point(lon, lat)
    subject_m = transform(to_metres, subject_ll)

    roads = []
    for w in ways:
        line_ll = linestring_from_way(w)
        if line_ll is None:
            continue

        line_m = transform(to_metres, line_ll)
        dist_m = subject_m.distance(line_m)
        if dist_m > SEARCH_RADIUS_M:
            continue

        tags = w.get("tags", {})
        highway = tags.get("highway")
        code, label, boost = category_from_highway(highway)
        if code is None:
            continue

        roads.append({
            "name": tags.get("name", "Unnamed"),
            "highway": highway,
            "category": code,
            "category_label": label,
            "distance_m": float(dist_m),
            "boost_pct": boost,
            "geometry": list(line_ll.coords)
        })

    if not roads:
        return [], {
            "name": None, "highway": None, "category": None,
            "category_label": None, "distance_m": None, "boost_pct": 0.0,
            "geometry": None
        }

    # 1. Sort by **valuation impact first**, then by distance
    roads.sort(key=lambda r: (-r["boost_pct"], r["distance_m"]))

    # 2. The *nearest widest* road (first after the sort)
    nearest_widest = roads[0]

    # 3. Return **all** roads sorted by distance (for tables / maps)
    all_sorted = sorted(roads, key=lambda r: r["distance_m"])

    return all_sorted, nearest_widest

# ----------------------------------------------------------------------
# QUICK DEMO (run with `python road_identifier.py`)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example coordinates – replace with any lat,lon you like
    LAT, LON = 18.5530, 73.7589

    all_roads, best = identify_road(LAT, LON)

    print("\n=== ALL ROADS (sorted by distance) ===")
    for r in all_roads:
        print(f"{r['name']} | {r['category']} | {r['distance_m']:.1f}m | {r['highway']}")

    print("\n=== BEST ROAD FOR VALUATION ===")
    print(f"Name          : {best['name']}")
    print(f"Category      : {best['category']} ({best['category_label']})")
    print(f"Distance      : {best['distance_m']:.1f} m")
    print(f"Boost %       : {best['boost_pct']}")
    print(f"OSM highway   : {best['highway']}")