import numpy as np
import xarray as xr
import rioxarray as rio
from datetime import datetime, timezone


def _underwater_vars(results: dict) -> dict:
    """Build underwater data_vars entries when depth attenuation was applied."""
    return {
        "underwater_solar_direct": (
            ("location", "time", "depth", "wavelength"),
            results["underwater_solar_direct"],
            {"units": "W m-2 nm-1"},
        ),
        "underwater_solar_diffuse": (
            ("location", "time", "depth", "wavelength"),
            results["underwater_solar_diffuse"],
            {"units": "W m-2 nm-1"},
        ),
        "underwater_lunar_direct": (
            ("location", "time", "depth", "wavelength"),
            results["underwater_lunar_direct"],
            {"units": "W m-2 nm-1"},
        ),
        "underwater_lunar_diffuse": (
            ("location", "time", "depth", "wavelength"),
            results["underwater_lunar_diffuse"],
            {"units": "W m-2 nm-1"},
        ),
        "underwater_spectral_par": (
            ("location", "time", "depth", "wavelength"),
            results["underwater_spectral_par"],
            {"units": "µmol photons m-2 s-1"},
        ),
        "underwater_broadband_par": (
            ("location", "time", "depth"),
            results["underwater_broadband_par"],
            {"units": "µmol photons m-2 s-1"},
        ),
        "underwater_broadband_irradiance": (
            ("location", "time", "depth"),
            results["underwater_broadband_irradiance"],
            {"units": "W m-2"},
        ),
    }


def _below_surface_vars(results: dict) -> dict:
    """Build the below-surface data_vars entries when transmission was applied."""
    return {
        "below_solar_direct": (
            ("location", "time", "wavelength"),
            results["below_solar_direct"],
            {"units": "W m-2 nm-1"},
        ),
        "below_solar_diffuse": (
            ("location", "time", "wavelength"),
            results["below_solar_diffuse"],
            {"units": "W m-2 nm-1"},
        ),
        "below_lunar_direct": (
            ("location", "time", "wavelength"),
            results["below_lunar_direct"],
            {"units": "W m-2 nm-1"},
        ),
        "below_lunar_diffuse": (
            ("location", "time", "wavelength"),
            results["below_lunar_diffuse"],
            {"units": "W m-2 nm-1"},
        ),
        "below_spectral_par": (
            ("location", "time", "wavelength"),
            results["below_spectral_par"],
            {"units": "µmol photons m-2 s-1"},
        ),
        "below_broadband_par": (
            ("location", "time"),
            results["below_broadband_par"],
            {"units": "µmol photons m-2 s-1"},
        ),
        "below_broadband_irradiance": (
            ("location", "time"),
            results["below_broadband_irradiance"],
            {"units": "W m-2"},
        ),
    }


def save_results_to_netcdf(results: dict, filename: str):
    """
    Save model output dictionary to a NetCDF file.

    Dimensions
    ----------
    location : spatial points (lat/lon pairs)
    time     : model timestamps
    wavelength : spectral bands (nm)

    Parameters
    ----------
    results : dict
        Output dictionary returned by run_model().
    filename : str
        Path to output NetCDF file.
    """

    # ----------------------------
    # Extract coordinates
    # ----------------------------
    wavelengths = results["wavelengths_nm"]

    times_unix = results["timestamps"]
    times_dt = np.array([
        datetime.fromtimestamp(t, tz=timezone.utc)
        for t in times_unix
    ], dtype="datetime64[ns]")

    latitudes = np.array([loc.latitude for loc in results["locations"]])
    latitudes = np.unique(latitudes)  # Ensure unique latitudes for the grid
    longitudes = np.array([loc.longitude for loc in results["locations"]])
    longitudes = np.unique(longitudes)  # Ensure unique longitudes for the grid
    
    print(f"Latitudes: {latitudes}")
    print(f"Longitudes: {longitudes}")
    
    print(f"Solar direct shape: {results['solar_direct'].shape}")
    print(results["solar_direct"])
    
    #for each dataset reshape so first dimension is split into latitude and longitude dimensions
    n_latitudes = len(latitudes)
    n_longitudes = len(longitudes)
    
    solar_direct_reshaped = results["solar_direct"].reshape((n_latitudes, n_longitudes, len(times_unix), len(wavelengths)))
    solar_diffuse_reshaped = results["solar_diffuse"].reshape((n_latitudes, n_longitudes, len(times_unix), len(wavelengths)))
    lunar_direct_reshaped = results["lunar_direct"].reshape((n_latitudes, n_longitudes, len(times_unix), len(wavelengths)))
    lunar_diffuse_reshaped = results["lunar_diffuse"].reshape((n_latitudes, n_longitudes, len(times_unix), len(wavelengths)))
    spectral_par_reshaped = results["spectral_par"].reshape((n_latitudes, n_longitudes, len(times_unix), len(wavelengths)))
    broadband_par_reshaped = results["broadband_par"].reshape((n_latitudes, n_longitudes, len(times_unix)))
    broadband_irradiance_reshaped = results["broadband_irradiance"].reshape((n_latitudes, n_longitudes, len(times_unix)))
    solar_zenith_reshaped = results["solar_zenith"].reshape((n_latitudes, n_longitudes, len(times_unix)))
    lunar_zenith_reshaped = results["lunar_zenith"].reshape((n_latitudes, n_longitudes, len(times_unix)))
    cloud_cover_reshaped = results["cloud_cover"].reshape((n_latitudes, n_longitudes, len(times_unix)))
    
    print(f"Reshaped solar direct shape: {solar_direct_reshaped.shape}")
    print(solar_direct_reshaped)

    # ----------------------------
    # Build xarray Dataset
    # ----------------------------
    ds = xr.Dataset(

        data_vars=dict(

            # --- Solar ---
            solar_direct=(
                ("latitude", "longitude", "time", "wavelength"),
                solar_direct_reshaped,
                {"units": "W m-2 nm-1"}
            ),
            solar_diffuse=(
                ("latitude", "longitude", "time", "wavelength"),
                solar_diffuse_reshaped,
                {"units": "W m-2 nm-1"}
            ),

            # --- Lunar ---
            lunar_direct=(
                ("latitude", "longitude", "time", "wavelength"),
                lunar_direct_reshaped,
                {"units": "W m-2 nm-1"}
            ),
            lunar_diffuse=(
                ("latitude", "longitude", "time", "wavelength"),
                lunar_diffuse_reshaped,
                {"units": "W m-2 nm-1"}
            ),

            # --- PAR ---
            spectral_par=(
                ("latitude", "longitude", "time", "wavelength"),
                spectral_par_reshaped,
                {"units": "µmol photons m-2 s-1"}
            ),
            broadband_par=(
                ("latitude", "longitude", "time"),
                broadband_par_reshaped,
                {"units": "µmol photons m-2 s-1"}
            ),

            # --- Broadband irradiance ---
            broadband_irradiance=(
                ("latitude", "longitude", "time"),
                broadband_irradiance_reshaped,
                {"units": "W m-2"}
            ),

            # --- Geometry ---
            solar_zenith=(
                ("latitude", "longitude", "time"),
                solar_zenith_reshaped,
                {"units": "degrees"}
            ),
            lunar_zenith=(
                ("latitude", "longitude", "time"),
                lunar_zenith_reshaped,
                {"units": "degrees"}
            ),

            # --- Cloud cover ---
            cloud_cover=(
                ("latitude", "longitude", "time"),
                cloud_cover_reshaped,
                {"units": "fraction", "valid_range": [0, 1]}
            ),

            **(_below_surface_vars(results) if "below_solar_direct" in results else {}),
        **(_underwater_vars(results)    if "underwater_solar_direct" in results else {}),
        ),

        coords=dict(
            wavelength=("wavelength", wavelengths, {"units": "nm"}),
            time=("time", times_dt),
            latitude=("latitude", latitudes, {"units": "degrees_north"}),
            longitude=("longitude", longitudes, {"units": "degrees_east"}),
            **({"depth": ("depth", results["depth_steps"], {"units": "m",
                          "positive": "down"})}
               if "depth_steps" in results else {}),
        ),

        attrs=dict(
            title="Solar–Lunar Spectral Irradiance Model Output",
            created=datetime.utcnow().isoformat() + "Z",
            time_reference="UTC",
            conventions="CF-1.8",
        )
    )
    
    ds.rio.write_crs("EPSG:4326", inplace=True)  # Set CRS for geospatial data

    
    ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)  # Set spatial dimensions
    # ----------------------------
    # Save to NetCDF
    # ----------------------------
    ds.to_netcdf(filename)
    
    #ds.broadband_par.isel(time=0).rio.to_raster(filename.replace(".nc", ".tif"))  # Save first time slice as GeoTIFF
    print(f"Saved NetCDF file: {filename}")