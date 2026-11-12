//! Geodesy helpers: ellipsoidal HEALPix (authalic latitude) and unit vectors.
//!
//! HEALPix cells are defined on the authalic sphere. With `Ellipsoid::Wgs84`
//! the geodetic latitude of every input (bathymetry, coastline, domain) is
//! converted to the authalic latitude before hashing, and cell centres are
//! converted back to geodetic latitude for WAVEWATCH III. Longitudes are
//! unchanged. With `Ellipsoid::Sphere` both latitudes coincide.

use serde::{Deserialize, Serialize};

/// Earth model.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Ellipsoid {
    /// sphere of radius 6371 km; latitudes are not converted
    Sphere,
    /// WGS84 ellipsoid; latitudes are converted to and from authalic latitudes
    Wgs84,
}

/// Latitude conversions and radius of the HEALPix sphere for an [`Ellipsoid`].
#[derive(Clone, Copy, Debug)]
pub struct Earth {
    /// the earth model
    pub kind: Ellipsoid,
    /// first eccentricity squared
    e2: f64,
    /// radius of the sphere on which HEALPix lives (authalic radius), metres
    pub radius: f64,
}

impl Earth {
    /// Earth for `kind`; for WGS84 the radius is the authalic radius
    /// (6 371 007.18 m).
    pub fn new(kind: Ellipsoid) -> Self {
        match kind {
            Ellipsoid::Sphere => Earth { kind, e2: 0.0, radius: 6_371_000.0 },
            Ellipsoid::Wgs84 => {
                let a = 6_378_137.0_f64;
                let f: f64 = 1.0 / 298.257_223_563;
                let e2 = f * (2.0 - f);
                let e = e2.sqrt();
                let qp = 1.0 + (1.0 - e2) / (2.0 * e) * ((1.0 + e) / (1.0 - e)).ln();
                Earth { kind, e2, radius: a * (qp / 2.0).sqrt() }
            }
        }
    }

    fn q(&self, phi: f64) -> f64 {
        let e = self.e2.sqrt();
        let s = phi.sin();
        (1.0 - self.e2)
            * (s / (1.0 - self.e2 * s * s) - 1.0 / (2.0 * e) * ((1.0 - e * s) / (1.0 + e * s)).ln())
    }

    /// Geodetic latitude (rad) -> authalic latitude (rad), exact formula.
    pub fn to_authalic(&self, phi: f64) -> f64 {
        if self.e2 == 0.0 {
            return phi;
        }
        let qp = self.q(std::f64::consts::FRAC_PI_2);
        (self.q(phi) / qp).clamp(-1.0, 1.0).asin()
    }

    /// Authalic latitude (rad) -> geodetic latitude (rad), series to e^6 then
    /// two Newton iterations on the exact forward formula.
    pub fn to_geodetic(&self, beta: f64) -> f64 {
        if self.e2 == 0.0 {
            return beta;
        }
        let (e2, e4, e6) = (self.e2, self.e2 * self.e2, self.e2 * self.e2 * self.e2);
        let mut phi = beta
            + (e2 / 3.0 + 31.0 * e4 / 180.0 + 517.0 * e6 / 5040.0) * (2.0 * beta).sin()
            + (23.0 * e4 / 360.0 + 251.0 * e6 / 3780.0) * (4.0 * beta).sin()
            + (761.0 * e6 / 45360.0) * (6.0 * beta).sin();
        let qp = self.q(std::f64::consts::FRAC_PI_2);
        for _ in 0..2 {
            if phi.abs() >= std::f64::consts::FRAC_PI_2 - 1e-12 {
                break;
            }
            // d q / d phi = 2 (1-e2) cos(phi) / (1 - e2 sin^2 phi)^2
            let s = phi.sin();
            let dq = 2.0 * (1.0 - e2) * phi.cos() / (1.0 - e2 * s * s).powi(2);
            let f = self.q(phi) - qp * beta.sin();
            phi -= f / dq;
        }
        phi
    }
}

/// Unit vector of (`lon`, `lat`) in radians.
pub fn lonlat_to_vec(lon: f64, lat: f64) -> [f64; 3] {
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    [cl * co, cl * so, sl]
}

/// (lon, lat) in radians of a unit vector, lon in (-pi, pi].
pub fn vec_to_lonlat(v: &[f64; 3]) -> (f64, f64) {
    (v[1].atan2(v[0]), v[2].clamp(-1.0, 1.0).asin())
}

/// Dot product.
#[inline]
pub fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product.
#[inline]
pub fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Difference `a - b`.
#[inline]
pub fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Euclidean norm.
pub fn norm(a: &[f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// `a / |a|`.
pub fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = norm(&a);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// Great-circle angle between two unit vectors (rad), robust for small angles.
pub fn angle(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    norm(&cross(a, b)).atan2(dot(a, b))
}

/// Local (east, north) basis at unit vector p; east x north = p.
pub fn tangent_basis(p: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    let z = [0.0, 0.0, 1.0];
    let mut e = cross(&z, p);
    if norm(&e) < 1e-12 {
        e = [0.0, 1.0, 0.0];
    }
    let e = normalize(e);
    let n = cross(p, &e);
    (e, n)
}

/// HEALPix centre (authalic lon, lat in rad) -> (lon, lat) in degrees on the
/// ellipsoid (lon in (-180, 180]).
pub fn healpix_to_geodetic_deg(earth: &Earth, lon: f64, lat: f64) -> (f64, f64) {
    let mut lo = lon.to_degrees();
    if lo > 180.0 {
        lo -= 360.0;
    }
    (lo, earth.to_geodetic(lat).to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authalic_round_trip() {
        let e = Earth::new(Ellipsoid::Wgs84);
        for i in -89..=89 {
            let phi = (i as f64).to_radians();
            let b = e.to_authalic(phi);
            let back = e.to_geodetic(b);
            assert!((back - phi).abs() < 1e-12, "lat {i}: {}", back - phi);
        }
        // known value: authalic radius of WGS84
        assert!((e.radius - 6_371_007.1809).abs() < 1e-3);
        // max difference geodetic - authalic is ~0.128 deg near 45 deg
        let d = (45f64.to_radians() - e.to_authalic(45f64.to_radians())).to_degrees();
        assert!((d - 0.1283).abs() < 1e-3, "{d}");
    }
}
