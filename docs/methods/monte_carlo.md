# Monte Carlo Method

*(part of [Methodology](../METHODOLOGY.md))*

A real X-ray tube's output is a spectrum, not a single photon energy, and material attenuation
$\mu(E)$ is energy-dependent. This method approximates the resulting beam-hardening and scatter
effects with a deliberately simplified model — polychromatic ray-marching at fixed step intervals,
plus a Gaussian-blur scatter kernel — rather than tracing individual photon histories
stochastically (hence not being true Monte Carlo transport, despite the name).

## Algorithm

1. **Geometry.** Build a world-space source point and a grid of detector-pixel positions from the
   configured source-to-detector distance, source-to-axis distance, isocenter, view direction, and
   up vector.
2. **Spectrum discretisation.** Split the modeled X-ray tube spectrum (80 kVp tungsten anode,
   aluminum + copper filtration) into a small number of discrete energy bins, each assigned a
   normalised weight $w_e$ (Eq. 1).
3. **Per-bin ray march.** For each energy bin $e$ and each ray, step along the ray at $N$ fixed
   intervals; at each step, look up the local density and blend it into an effective attenuation
   coefficient for that bin (Eq. 2), then accumulate that bin's line integral (Eq. 3) and convert
   it to a transmission (Eq. 4).
4. **Polychromatic combination.** Weight-sum every bin's transmission by its spectral weight to get
   one polychromatic primary transmission (Eq. 5) — this is what produces beam hardening.
5. **Scatter approximation.** Apply a Gaussian blur to the attenuated primary signal to approximate
   Compton scatter (Eq. 6), then convert the combined signal to attenuation units (Eq. 7).
6. **Optional noise.** Re-sample the deterministic image from a Poisson distribution to add
   photon-counting (shot) noise (Eq. 8).

<figure>
<img src="../../assets/results/method_crops/monte_carlo.png" alt="Monte Carlo rendered chest DRR, AP view" width="360">
<figcaption>Actual Monte Carlo output on the same CT/geometry as the Plastimatch reference above. The softer contrast and diffuse haze relative to Plastimatch — most visible around the lung fields — is the polychromatic beam-hardening (Eq 5) and Gaussian-blur scatter term (Eq 6) this method adds on top of the base line integral; see <code>METHODOLOGY.md</code> for why that's an intentional physical difference, not lower quality.</figcaption>
</figure>

<figure>
<svg viewBox="0 0 600 300" width="100%" style="max-width:600px;font-family:-apple-system,Segoe UI,Roboto,sans-serif" role="img" aria-labelledby="mcov-title mcov-desc">
<title id="mcov-title">One sample's density, traced through the attenuation blend</title>
<desc id="mcov-desc">Effective attenuation is a quadratic blend of tissue and bone attenuation, weighted by local density; the curve at right shows that blend is nonlinear, not a straight interpolation.</desc>
<rect x="20" y="20" width="260" height="260" fill="none" stroke="#93c5fd" stroke-width="1.5"></rect>
<image href="../../assets/results/method_crops/monte_carlo_small.png" x="20" y="20" width="260" height="260"></image>
<circle cx="150" cy="165" r="7" fill="none" stroke="#facc15" stroke-width="3"></circle>
<path d="M157,165 C230,165 250,150 300,150" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4 3" fill="none"></path>
<rect x="300" y="20" width="280" height="260" rx="8" fill="#fafbfc" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="440" y="40" text-anchor="middle" font-size="11.5" fill="#1e3a5f" font-weight="600">density → attenuation, at 50 keV</text>
<path d="M320,200 H560" stroke="#64748b" stroke-width="1.5" fill="none"></path>
<path d="M320,200 V60" stroke="#64748b" stroke-width="1.5" fill="none"></path>
<path d="M320,200 L344,196 L368,189 L392,181 L416,170 L440,157 L464,142 L488,125 L512,105 L536,84 L560,60"
      stroke="#2b6cb0" stroke-width="2.5" fill="none"></path>
<circle cx="476" cy="133" r="4.5" fill="#d97706"></circle>
<path d="M476,133 V200" stroke="#d97706" stroke-width="1" stroke-dasharray="3 2"></path>
<text x="476" y="220" text-anchor="middle" font-size="9.5" fill="#8a6d1f">ρₖ=0.65</text>
<text x="486" y="130" font-size="10" fill="#8a6d1f">μ_eff≈0.046</text>
<text x="320" y="214" font-size="10" fill="#64748b">air (ρ=0)</text>
<text x="560" y="214" text-anchor="end" font-size="10" fill="#64748b">bone (ρ=1)</text>
<text x="440" y="234" text-anchor="middle" font-size="10" fill="#64748b">ρₖ (local density)</text>
<text x="312" y="55" text-anchor="end" font-size="10" fill="#64748b">μ_eff</text>
<text x="440" y="256" text-anchor="middle" font-size="10.5" fill="#334155">μ_eff = ρₖ·(ρₖ·μ_bone + (1-ρₖ)·μ_tissue)</text>
</svg>
<figcaption>Eq 2's density blend is quadratic, not linear — this curve plots it at the 50 keV bin (<code>μ_tissue≈0.023</code>, <code>μ_bone≈0.097</code> mm⁻¹, from the NIST XCOM tables in <code>monte_carlo.py</code>). A sample at <code>ρₖ=0.65</code> doesn't get 65% of the way from tissue to bone attenuation — the outer <code>ρₖ·(…)</code> factor pulls the curve down for medium densities, so partial-density voxels (edges, partial-volume effects) attenuate less than a linear blend would suggest.</figcaption>
</figure>

<figure>
<svg viewBox="0 0 380 230" width="100%" style="max-width:380px;font-family:-apple-system,Segoe UI,Roboto,sans-serif" role="img" aria-labelledby="mcex-title mcex-desc">
<title id="mcex-title">Polychromatic spectrum discretisation and beam hardening</title>
<desc id="mcex-desc">The modeled 80kVp X-ray tube spectrum is split into discrete energy bins, each with its own weight; each bin is ray-marched separately at its own attenuation rate, so low-energy bins are depleted faster with depth than high-energy ones.</desc>
<defs>
<marker id="mcex-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#64748b"></path>
</marker>
</defs>
<rect x="150" y="14" width="14" height="10" fill="#2b6cb0"></rect>
<text x="168" y="23" font-size="9.5" fill="#334155">entry spectrum</text>
<path d="M260,19 H274" stroke="#d97706" stroke-width="2" stroke-dasharray="4 2" fill="none"></path>
<text x="278" y="23" font-size="9.5" fill="#334155">after ~10cm tissue</text>
<path d="M20,180 H365" stroke="#64748b" stroke-width="1.5" fill="none" marker-end="url(#mcex-arrow)"></path>
<path d="M20,180 V50" stroke="#64748b" stroke-width="1.5" fill="none"></path>
<rect x="25" y="175" width="22" height="5" fill="#2b6cb0"></rect>
<rect x="51" y="158" width="22" height="22" fill="#2b6cb0"></rect>
<rect x="77" y="131" width="22" height="49" fill="#2b6cb0"></rect>
<rect x="103" y="98" width="22" height="82" fill="#2b6cb0"></rect>
<rect x="129" y="72" width="22" height="108" fill="#2b6cb0"></rect>
<rect x="155" y="60" width="22" height="120" fill="#2b6cb0"></rect>
<rect x="181" y="60" width="22" height="120" fill="#2b6cb0"></rect>
<rect x="207" y="70" width="22" height="110" fill="#2b6cb0"></rect>
<rect x="233" y="74" width="22" height="106" fill="#2b6cb0"></rect>
<rect x="259" y="116" width="22" height="64" fill="#2b6cb0"></rect>
<rect x="285" y="122" width="22" height="58" fill="#2b6cb0"></rect>
<rect x="311" y="132" width="22" height="48" fill="#2b6cb0"></rect>
<path d="M25,179 H47" stroke="#d97706" stroke-width="2" stroke-dasharray="4 2"></path>
<path d="M51,175 H73" stroke="#d97706" stroke-width="2" stroke-dasharray="4 2"></path>
<path d="M77,167 H99" stroke="#d97706" stroke-width="2" stroke-dasharray="4 2"></path>
<path d="M103,153 H125" stroke="#d97706" stroke-width="2" stroke-dasharray="4 2"></path>
<path d="M129,138 H151" stroke="#d97706" stroke-width="2" stroke-dasharray="4 2"></path>
<path d="M155,127 H177" stroke="#d97706" stroke-width="2" stroke-dasharray="4 2"></path>
<path d="M181,119 H203" stroke="#d97706" stroke-width="2" stroke-dasharray="4 2"></path>
<path d="M207,118 H229" stroke="#d97706" stroke-width="2" stroke-dasharray="4 2"></path>
<path d="M233,114 H255" stroke="#d97706" stroke-width="2" stroke-dasharray="4 2"></path>
<path d="M259,136 H281" stroke="#d97706" stroke-width="2" stroke-dasharray="4 2"></path>
<path d="M285,137 H307" stroke="#d97706" stroke-width="2" stroke-dasharray="4 2"></path>
<path d="M311,142 H333" stroke="#d97706" stroke-width="2" stroke-dasharray="4 2"></path>
<text x="36" y="196" text-anchor="middle" font-size="10" fill="#64748b">20 keV</text>
<text x="322" y="196" text-anchor="middle" font-size="10" fill="#64748b">80 keV</text>
<text x="192" y="210" text-anchor="middle" font-size="11" fill="#334155">energy bins e</text>
<text x="16" y="55" text-anchor="end" font-size="11" fill="#334155">wₑ</text>
</svg>
<figcaption>The modeled tube spectrum (Eq 1), discretised: each bar is one energy bin's weight <code>wₑ</code> at skin entry. The dashed amber lines mark, illustratively, how much of each bin survives ~10cm of tissue — the low-energy (left) bins are cut down much further than the high-energy (right) ones, so the surviving spectrum is skewed toward higher energies at depth. Every bin is still ray-marched independently at its own <code>μ</code> (Eq 2-4) and only combined afterward (Eq 5); this per-bin depletion, not the entry spectrum's shape alone, is what beam hardening actually is.</figcaption>
</figure>

## Mathematical Formulation

**(1) Spectrum weights.** The modeled tube spectrum is represented by weights $w_e$ for each energy
bin $e$, derived from the Kramers–Bohlin bremsstrahlung model plus characteristic lines and
filtration, normalised so $\sum_e w_e = 1$.

**(2) Per-sample effective attenuation.** At sample point $k$ along a ray (parametric offset
$t_k=(k+0.5)/N$), with local density $\rho_k\in[0,1]$:

$$
\mu_{\text{eff},k} = \rho_k\cdot\Big(\rho_k\,\mu_{\text{bone},e} + (1-\rho_k)\,\mu_{\text{tissue},e}\Big)
$$

The outer $\rho_k\cdot(\cdots)$ factor — beyond the density/(1−density) blend itself — guarantees
$\rho_k=0$ (air) contributes exactly zero attenuation; without it, a density of exactly 0 would
still pick up $\mu_{\text{tissue},e}$.

**(3) Per-bin line integral (Riemann sum).**

$$
\int \mu_e\,dl \approx \left(\sum_k \mu_{\text{eff},k}\right)\cdot\frac{\lVert\mathbf{ray}\rVert}{N}
$$

**(4) Per-bin transmission.**

$$
T_e = \exp\left(-\int \mu_e\,dl\right)
$$

**(5) Polychromatic combination (beam hardening).**

$$
T_{\text{primary}} = \sum_e w_e\,T_e = \sum_e w_e\exp\left(-\int \mu_e\,dl\right)
$$

Unlike a monochromatic model, this is a weighted sum of exponentials, not a single exponential of
one averaged $\mu$ — each bin attenuates at its own rate, and low-energy bins are removed faster
than high-energy ones as depth increases, which is exactly the beam-hardening effect.

**(6) Scatter kernel.**

$$
S = \text{GaussianBlur}\big((1-T_{\text{primary}})\cdot f_{\text{scatter}},\ \sigma\big)
$$

**(7) Attenuation conversion.**

$$
\text{image} = -\log\big(\text{clip}(T_{\text{primary}}+S,\ \epsilon,\ 1)\big)
$$

matching the attenuation convention (bone bright, air dark) shared by every method.

**(8) Poisson shot noise (optional).**

$$
\text{counts}\sim\text{Poisson}(\text{image}\cdot M), \qquad \text{image}_{\text{noisy}} = \frac{\text{counts}}{M}
$$

for $M$ open-beam photons.
