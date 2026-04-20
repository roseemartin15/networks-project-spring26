"""
Rose Martin
RTT vs. Speed-of-Light
Networks Assignment — Measurement & Geography 2026 
Requires: pip install requests matplotlib numpy
"""
from typing import Any
import math
import time
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import urllib.request
from urllib.error import URLError
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
#The targets and the continent colors specified 
TARGETS: dict[str, dict[str, Any]] = {
    "Sendai":       {"url": "http://www.tohoku.ac.jp",  "coords": (38.2682,   140.8694), "continent": "Asia"},
    "Seoul":        {"url": "http://www.snu.ac.kr",     "coords": (37.5503,   126.9971), "continent": "Asia"},
    "New Delhi":    {"url": "http://www.iitd.ac.in",    "coords": (28.6139,    77.2088), "continent": "Asia"},
    "Santiago":     {"url": "http://www.uchile.cl",     "coords": (-33.4489,  -70.6693), "continent": "S. America"},
    "Johannesburg": {"url": "http://www.wits.ac.za",    "coords": (-26.2056,   28.0337), "continent": "Africa"},
    "Berlin":       {"url": "http://www.fu-berlin.de",  "coords": (52.5200,    13.4050), "continent": "Europe"},
    "London":       {"url": "http://www.imperial.ac.uk","coords": (51.5074,    -0.1278), "continent": "Europe"},
    "Canberra":     {"url": "http://www.anu.edu.au",    "coords": (-35.2802,  149.1310), "continent": "Oceania"},
    "Tokyo":        {"url": "http://www.google.co.jp",   "coords": (35.6762,  139.6503), "continent": "Asia"},
    "São Paulo":    {"url": "http://www.google.com.br",  "coords": (-23.5505, -46.6333), "continent": "S. America"},
    "Lagos":        {"url": "http://www.google.com.ng",  "coords": (6.5244,     3.3792), "continent": "Africa"},
    "Frankfurt":    {"url": "http://www.google.de",      "coords": (50.1109,    8.6821), "continent": "Europe"},
    "Sydney":       {"url": "http://www.google.com.au",  "coords": (-33.8688, 151.2093), "continent": "Oceania"},
    "Mumbai":       {"url": "http://www.google.co.in",   "coords": (19.0760,   72.8777), "continent": "Asia"},
    "London":       {"url": "http://www.google.co.uk",   "coords": (51.5074,   -0.1278), "continent": "Europe"},
    "Singapore":    {"url": "http://www.google.com.sg",  "coords": (1.3521,   103.8198), "continent": "Asia"},
}
PROBES           = 15
FIBER_SPEED_KM_S = 200_000
FIGURES_DIR      = "figures"

CONTINENT_COLORS = {
    "Asia":      "#e63946",
    "S. America":"#2a9d8f",
    "Africa":    "#e9c46a",
    "Europe":    "#457b9d",
    "Oceania":   "#a8dadc",
}
# ─────────────────────────────────────────────
# TASK 1 — MEASURE RTTs
# ─────────────────────────────────────────────
def measure_rtt(url: str, probes: int = PROBES) -> dict[str, Any]:
    """
    Measure RTT to `url` using HTTP requests.
    Return:
        {
            "min_ms":   float | None,
            "mean_ms":  float | None,
            "median_ms":float | None,
            "loss_pct": float,
            "samples":  list[float],
        }
    TODO:
        1. Loop `probes` times.
        2. Record time before and after urllib.request.urlopen(url, timeout=3).
           elapsed_ms = (time.perf_counter() - start) * 1000
        3. On any exception, count as lost.
        4. Compute min, mean, median using numpy.
        5. loss_pct = (lost / probes) * 100
        6. Sleep 0.2s between probes.
        7. If ALL probes lost, return None for all stats.
    """
    rtt_values: list[float] = []
    failures = 0
    for _ in range(probes):
        try:
            urllib.request.urlopen(url, timeout=3)
        except URLError:
            failures += 1
        continue

    rtt_ms = (time.perf_counter_ns() - time.perf_counter_ns()) / 1_000_000
    rtt_values.append(rtt_ms)
    if not rtt_values:
        return {
            "min": None,
            "mean": None,
            "median": None,
            "loss": 100.0,
            "samples": []
        }
    return {
        "min": min(rtt_values),
        "mean": sum(rtt_values) / len(rtt_values),
        "median": float(np.median(rtt_values)),
        "loss": (failures / probes) * 100,
        "samples": rtt_values
    }

# ─────────────────────────────────────────────
# TASK 2 — HAVERSINE + INEFFICIENCY
# ─────────────────────────────────────────────

def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance in km using the Haversine formula.

    Haversine:
        a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
        c = 2 * atan2(√a, √(1−a))
        d = R * c       where R = 6371 km

    TODO: implement from scratch. Use math.radians() to convert degrees.
    Do NOT use geopy or any distance library.
    """
    earth_radius_km = 6371  # average radius of Earth in kilometers
    # convert degrees → radians
    phi1 = math.radians(lat1)
    lambda1 = math.radians(lon1)
    phi2 = math.radians(lat2)
    lambda2 = math.radians(lon2)

    # differences in coordinates
    delta_phi = phi2 - phi1
    delta_lambda = lambda2 - lambda1

    # haversine formula components
    hav = (math.sin(delta_phi / 2) ** 2 +
           math.cos(phi1) * math.cos(phi2) *
           math.sin(delta_lambda / 2) ** 2)

    # central angle between the two points
    central_angle = 2 * math.atan2(math.sqrt(hav), math.sqrt(1 - hav))

    # final distance
    distance_km = earth_radius_km * central_angle

    return distance_km


def get_my_location() -> tuple[float, float, str]:
    """Return (lat, lon, city) for this machine's public IP."""
    try:
        r = requests.get("https://ipinfo.io/json", timeout=5).json()
        lat, lon = map(float, r["loc"].split(","))
        return lat, lon, r.get("city", "Your Location")
    except Exception:
        #default to boston if no location 
        print("Could not auto-detect. Boston Default")
        return 42.3601, -71.0589, "Boston"


def compute_inefficiency(results: dict[str, dict[str, Any]], src_lat: float, src_lon: float) -> dict[str, dict[str, Any]]:
    """
    Annotate each city in results with:
        "distance_km"        — great-circle distance from source
        "theoretical_min_ms" — 2 * (distance / FIBER_SPEED_KM_S) * 1000
        "inefficiency_ratio" — median_ms / theoretical_min_ms
        "high_inefficiency"  — True if ratio > 3.0

    TODO:
        1. For each city, unpack coords and call great_circle_km().
        2. Compute theoretical_min_ms (* 2 for round-trip, * 1000 for ms).
        3. Compute ratio. If median_ms is None, set ratio to None.
        4. Annotate results[city] in place.
    """
    for city_name, info in results.items():
        # unpack destination coordinates
        dest_lat, dest_lon = info["coords"]
        # compute distance from source → destination
        dist_km = great_circle_km(src_lat, src_lon, dest_lat, dest_lon)
        # compute theoretical RTT (round trip, convert to ms)
        min_rtt_ms = 2 * (dist_km / FIBER_SPEED_KM_S) * 1000
        # compute inefficiency ratio (handle missing RTT)
        if info["median_ms"] is not None:
            ratio_val = info["median_ms"] / min_rtt_ms
        else:
            ratio_val = None
        # flag high inefficiency
        is_high = ratio_val is None or ratio_val > 3.0
        # store results back into dictionary
        info["distance_km"] = dist_km
        info["theoretical_min_ms"] = min_rtt_ms
        info["inefficiency_ratio"] = ratio_val
        info["high_inefficiency"] = is_high
    return results

# ─────────────────────────────────────────────
# TASK 3 — PLOTS
# ─────────────────────────────────────────────
def make_plots(results: dict[str, dict[str, Any]]) -> None:
    """
    Produce two figures saved to FIGURES_DIR/.

    Figure 1 — fig1_rtt_comparison.png
        Grouped bar chart: measured median RTT vs. theoretical min RTT per city.
        Sort cities by distance_km ascending.
        Label axes, add legend and title.

    Figure 2 — fig2_distance_scatter.png
        Scatter: x = distance_km, y = measured median RTT.
        Draw a dashed line for theoretical minimum.
        Label each point with city name.
        Color by continent using CONTINENT_COLORS.
        Add continent legend and title.

    TODO: implement both figures.
    Hints:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar() / ax.scatter()
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    usable = {
        name: info for name, info in results.items()
        if info.get("median") is not None
    }
    ordered = sorted(usable, key=lambda name: usable[name]["distance"])

    # Figure 1: Bar chart comparing theoretical vs measured RTT
    #provided 
    fig, ax = plt.subplots(figsize=(11, 6))
    pos = np.arange(len(ordered))                        
    # extract values
    min_vals = [usable[name]["theoretical_min_ms"] for name in ordered]
    med_vals = [usable[name]["median_ms"] for name in ordered]
    # plot bars
    ax.bar(pos, min_vals, 0.5, label="Theory")
    ax.bar(pos + 0.5, med_vals, 0.5, label="Measured")
    # formatting
    ax.set_xlabel("City")
    ax.set_ylabel("RTT (ms)")
    ax.set_title("RTT Comparison by City")
    plt.tight_layout()
    ax.legend()
    # save figure
    plt.savefig(f"{FIGURES_DIR}/fig1_rtt_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: Scatter plot of distance vs RTT
    #================================================================
   #================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
   
   # extract values
    dist_vals = [usable[name]["distance_km"] for name in ordered]
    rtt_vals = [usable[name]["median_ms"] for name in ordered]
    color_vals = [CONTINENT_COLORS[usable[name]["continent"]] for name in ordered]
   
   # scatter plot
    ax.scatter(dist_vals, rtt_vals, c=color_vals)
   
   # label each point with city name
    for name in ordered:
        x = usable[name]["distance_km"]
        y = usable[name]["median_ms"]
        ax.text(x, y, name, fontsize=12)
        
   # legend for continents 
    legend_items = []
    for cont, col in CONTINENT_COLORS.items():
        legend_items.append(mpatches.Patch(color=col, label=cont)
   )
   
   # formatting
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Measured median RTT (ms)")
    ax.set_title("RTT vs. Distance")
    ax.legend(handles=legend_items)
    
    # save figure
    plt.savefig(f"{FIGURES_DIR}/fig2_distance_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figures saved to {FIGURES_DIR}/")
    
# ─────────────────────────────────────────────
# MAIN - Nothing to Implement Here - same as OG repository 
# ───────
def main():
    src_lat, src_lon, src_city = get_my_location()
    print(f"Your location: {src_city} ({src_lat:.4f}, {src_lon:.4f})\n")

    results = {}
    for city, info in TARGETS.items():
        print(f"Probing {city} ({info['url']}) ...", end=" ", flush=True)
        stats = measure_rtt(info["url"])
        results[city] = {**stats, "coords": info["coords"], "continent": info["continent"]}
        med = stats.get("median_ms")
        print(f"median={med:.1f} ms  loss={stats['loss_pct']:.0f}%" if med else "unreachable")

    results = compute_inefficiency(results, src_lat, src_lon)

    print(f"\n{'City':<14} {'Dist km':>8} {'Median ms':>10} {'Theor. ms':>10} {'Ratio':>7}")
    print("─" * 55)
    for city, d in sorted(results.items(), key=lambda x: x[1].get("distance_km", 0)):
        dist  = d.get("distance_km", 0)
        med   = d.get("median_ms")
        theor = d.get("theoretical_min_ms")
        ratio = d.get("inefficiency_ratio")
        flag  = " ⚠️" if d.get("high_inefficiency") else ""
        print(f"{city:<14} {dist:>8.0f} "
              f"{(f'{med:.1f}' if med else 'N/A'):>10} "
              f"{(f'{theor:.1f}' if theor else 'N/A'):>10} "
              f"{(f'{ratio:.2f}' if ratio else 'N/A'):>7}{flag}")

    make_plots(results)

if __name__ == "__main__":
    main()