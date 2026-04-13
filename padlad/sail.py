"""
padlad/sail.py
==============
SAIL volume scattering kernel and LIDF utilities.
Faithful translation of new_ks_wo2.m, ncampbell.m, volscatt.m
(Ameni Mkaouar / Wout Verhoef).
"""
import numpy as np

# 15-bin leaf angle table — litab from new_ks_wo2.m / simplex_tst.m
LITAB = np.array([5,15,25,35,45,55,64,72,77,79,81,83,85,87,89], dtype=float)
NA    = len(LITAB)

# Angle bin limits from ncampbell.m
TX1 = np.array([10,20,30,40,50,60,68,76,78,80,82,84,86,88,90], dtype=float)
TX2 = np.array([ 0,10,20,30,40,50,60,68,76,78,80,82,84,86,88], dtype=float)

# Leaf optical properties — hemlock values from new_ks_wo2.m
RHO = 0.0133
TAU = 0.0192


def volscatt(tts: float, tto: float, psi: float, ttl: float) -> tuple:
    """
    SAIL volume scattering phase function.
    Faithful translation of volscatt.m (Wout Verhoef, April 2001).

    Parameters
    ----------
    tts : solar/scanner zenith angle (degrees)
    tto : observer zenith angle (degrees) — 0 for TLS
    psi : azimuth angle (degrees) — 0 for TLS
    ttl : leaf inclination angle (degrees)

    Returns
    -------
    chi_s, chi_o, frho, ftau
    Note: frho and ftau are clipped to ≥ 0 as in MATLAB original.
    """
    rd     = np.pi / 180.0
    costs  = np.cos(rd * tts)
    costo  = np.cos(rd * tto)
    sints  = np.sin(rd * tts)
    sinto  = np.sin(rd * tto)
    cospsi = np.cos(rd * psi)
    psir   = rd * psi
    costl  = np.cos(rd * ttl)
    sintl  = np.sin(rd * ttl)

    cs = costl * costs
    co = costl * costo
    ss = sintl * sints
    so = sintl * sinto

    # betas and betao — transition angles
    cosbts = 5.0
    if abs(ss) > 1e-6:
        cosbts = -cs / ss
    cosbto = 5.0
    if abs(so) > 1e-6:
        cosbto = -co / so

    if abs(cosbts) < 1:
        bts = np.arccos(cosbts)
        ds  = ss
    else:
        bts = np.pi
        ds  = cs

    chi_s = 2.0/np.pi * ((bts - np.pi*0.5)*cs + np.sin(bts)*ss)

    if abs(cosbto) < 1:
        bto = np.arccos(cosbto)
        doo = so
    elif tto < 90:
        bto = np.pi
        doo = co
    else:
        bto = 0.0
        doo = -co

    chi_o = 2.0/np.pi * ((bto - np.pi*0.5)*co + np.sin(bto)*so)

    # auxiliary angles
    btran1 = abs(bts - bto)
    btran2 = np.pi - abs(bts + bto - np.pi)

    if psir <= btran1:
        bt1, bt2, bt3 = psir,    btran1, btran2
    elif psir <= btran2:
        bt1, bt2, bt3 = btran1,  psir,   btran2
    else:
        bt1, bt2, bt3 = btran1,  btran2, psir

    t1 = 2.0 * cs * co + ss * so * cospsi
    t2 = 0.0
    if bt2 > 0:
        # MATLAB: t2 = sin(bt2)*(2*ds*doo + ss*so*cos(bt1)*cos(bt3))
        t2 = np.sin(bt2) * (2.0*ds*doo + ss*so*np.cos(bt1)*np.cos(bt3))

    denom = 2.0 * np.pi**2
    frho  = ((np.pi - bt2)*t1 + t2) / denom
    ftau  =          (-bt2 *t1 + t2) / denom

    # clip — matches MATLAB: if frho<0: frho=0; if ftau<0: ftau=0
    frho = max(frho, 0.0)
    ftau = max(ftau, 0.0)

    return chi_s, chi_o, frho, ftau


def new_ks_wo(tts: float, lidf: np.ndarray) -> tuple:
    """
    Extinction coefficient ks and scattering w.
    Faithful translation of new_ks_wo2.m (Ameni Mkaouar).
    Uses the estimated LIDF vector directly (not a parametric model).

    Parameters
    ----------
    tts  : view/scanner zenith angle (degrees)
    lidf : (15,) normalised LIDF — output of constrain_lidf()

    Returns
    -------
    ks : extinction coefficient  (G = ks * cos(tts))
    w  : bidirectional scattering
    """
    rd     = np.pi / 180.0
    cts    = np.cos(rd * tts)
    ctscto = cts * 1.0          # tto = 0 → cto = 1

    ks = sob = sof = 0.0
    for i in range(NA):
        ttl  = LITAB[i]
        chi_s, chi_o, frho, ftau = volscatt(tts, 0.0, 0.0, ttl)
        ksli  = chi_s / (cts + 1e-12)
        sobli = frho  * np.pi / (ctscto + 1e-12)
        sofli = ftau  * np.pi / (ctscto + 1e-12)
        ks  += ksli  * lidf[i]
        sob += sobli * lidf[i]
        sof += sofli * lidf[i]

    w = sob * RHO + sof * TAU
    return ks, w


def ncampbell(ala: float) -> np.ndarray:
    """
    Campbell (1986) ellipsoidal LIDF for average leaf angle ala (degrees).
    Returns normalised 15-bin array on LITAB angles.
    Faithful translation of ncampbell.m.
    """
    tl1    = np.radians(TX1)
    tl2    = np.radians(TX2)
    excent = np.exp(-1.6184e-5*ala**3 + 2.1145e-3*ala**2
                    - 1.2390e-1*ala + 3.2491)
    freq   = np.zeros(NA)

    for i in range(NA):
        x1 = excent / np.sqrt(1.0 + excent**2 * np.tan(tl1[i])**2)
        x2 = excent / np.sqrt(1.0 + excent**2 * np.tan(tl2[i])**2)

        if abs(excent - 1.0) < 1e-9:
            freq[i] = abs(np.cos(tl1[i]) - np.cos(tl2[i]))
        else:
            alpha  = excent / np.sqrt(abs(1.0 - excent**2))
            alpha2 = alpha**2
            if excent > 1:
                alpx1   = np.sqrt(alpha2 + x1**2)
                alpx2   = np.sqrt(alpha2 + x2**2)
                dum     = x1*alpx1 + alpha2*np.log(x1 + alpx1)
                freq[i] = abs(dum - (x2*alpx2 + alpha2*np.log(x2 + alpx2)))
            else:
                almx1   = np.sqrt(max(alpha2 - x1**2, 0.0))
                almx2   = np.sqrt(max(alpha2 - x2**2, 0.0))
                dum     = x1*almx1 + alpha2*np.arcsin(np.clip(x1/alpha, -1, 1))
                freq[i] = abs(dum - (x2*almx2
                              + alpha2*np.arcsin(np.clip(x2/alpha, -1, 1))))

    total = freq.sum()
    return freq / total if total > 0 else np.ones(NA) / NA


def constrain_lidf(raw: np.ndarray) -> np.ndarray:
    """
    Enforce LIDF constraints from simplex_tst.m:
      C = raw + 1/na  →  clip(0,1)  →  normalise
    Returns the normalised LIDF (not raw form).
    """
    C = raw + 1.0 / NA
    C = np.clip(C, 0.0, 1.0)
    s = C.sum()
    return C / s if s > 1e-15 else np.ones(NA) / NA


def constrain_raw(raw: np.ndarray) -> np.ndarray:
    """Apply constraint and return raw form (for SCE-UA candidates)."""
    lidf = constrain_lidf(raw)
    return lidf - 1.0 / NA
