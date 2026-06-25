import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point

# Site coordinates
site_lat = 40.46542
site_lon = -79.960757

# Create GeoDataFrame
site = gpd.GeoDataFrame(
    {"name": ["ASCENT Pittsburgh"]},
    geometry=[Point(site_lon, site_lat)],
    crs="EPSG:4326"
)

# Convert to Web Mercator for basemap
site_web = site.to_crs(epsg=3857)

# Make plot
fig, ax = plt.subplots(figsize=(6, 6))

# Set map extent around site
buffer_m = 6000  # 6 km around site
x, y = site_web.geometry.x.iloc[0], site_web.geometry.y.iloc[0]
ax.set_xlim(x - buffer_m, x + buffer_m)
ax.set_ylim(y - buffer_m, y + buffer_m)

# Add basemap
ctx.add_basemap(
    ax,
    source=ctx.providers.OpenStreetMap.Mapnik,
    attribution=False
)

# Plot site marker
site_web.plot(ax=ax, color="red", markersize=80, zorder=5)

# Label site
ax.annotate(
    "ASCENT Pittsburgh site",
    xy=(x, y),
    xytext=(x + 600, y + 600),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", linewidth=1)
)

# Clean up axes
ax.set_axis_off()

# Add attribution manually
fig.text(
    0.01, 0.01,
    "Basemap data © OpenStreetMap contributors",
    fontsize=7
)

plt.tight_layout()

plt.savefig("ASCENT_Pittsburgh_site_map.png", dpi=600, bbox_inches="tight")
plt.savefig("ASCENT_Pittsburgh_site_map.pdf", bbox_inches="tight")
plt.show()