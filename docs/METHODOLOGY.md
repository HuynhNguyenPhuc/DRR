# Methodology

This suite compares five ways of generating a Digitally Reconstructed Radiograph (DRR) from a CT
volume. All five are trying to answer the same physical question — how much of an X-ray beam
survives the trip from source to detector, along each ray — but they answer it with different
amounts of approximation. This doc is about that physics and math: what each method actually
computes, and where its assumptions differ from the others. It isn't about the code.

## The physics every method is approximating

An X-ray image is a map of transmission: for each detector pixel, how much of the beam that left
the source along the ray to that pixel made it through the body. For a beam of a single photon
energy, that fraction follows Beer-Lambert's law:

**(1)**

$$
T = \frac{I}{I_0} = \exp\left(-\int \mu(l) \, dl\right)
$$

where $\mu$ is the linear attenuation coefficient of tissue at position $l$ along the ray, and the
integral runs the full length of the ray through the volume. Radiographs are usually displayed as
attenuation rather than transmission:

**(2)**

$$
-\log(T) = \int \mu \, dl
$$

since that convention makes dense tissue like bone come out bright and air come out dark, matching
how a real film or detector is read. Every method covered in this document computes some version of
Eq. (1)/(2)'s integral — the differences below are all in *how* that one integral gets evaluated.

Every method here is computing some version of that integral. They differ along three axes:

- **How the integral is evaluated.** Exactly, by working out precisely how far the ray travels
  through each voxel it crosses — or approximately, by sampling the volume at a finite number of
  points along the ray and summing.
- **Monochromatic or polychromatic.** A real X-ray tube doesn't emit one photon energy, it emits a
  spectrum. A monochromatic model picks one effective energy and treats $\mu$ as constant per
  material; a polychromatic one integrates over the whole spectrum, which changes the result in a
  way a single-energy model can't reproduce (see beam hardening, below).
- **Whether scatter is modeled.** Some photons don't travel straight from source to detector — they
  get deflected (mostly by Compton scattering) and still arrive, just from an angle. That adds a
  diffuse haze on top of the sharp "primary" image and lowers contrast. Modeling it means adding a
  second signal on top of Beer-Lambert; ignoring it means treating every photon that interacts as
  simply absorbed.

<figure>
<svg viewBox="0 0 700 150" width="100%" style="max-width:700px;font-family:-apple-system,Segoe UI,Roboto,sans-serif" role="img" aria-labelledby="meth-fig-title meth-fig-desc">
<title id="meth-fig-title">Shared Beer-Lambert pipeline</title>
<desc id="meth-fig-desc">Every method computes the same four-stage pipeline from attenuation coefficient to final image; they differ only in how the middle two stages are evaluated.</desc>
<defs>
<marker id="meth-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"></path>
</marker>
</defs>
<rect x="10" y="40" width="140" height="60" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="80" y="65" text-anchor="middle" font-size="14" fill="#1e3a5f" font-weight="600">μ(l)</text>
<text x="80" y="83" text-anchor="middle" font-size="10.5" fill="#5b7897">tissue attenuation</text>
<path d="M150,70 H188" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#meth-arrow)"></path>
<rect x="190" y="40" width="150" height="60" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="265" y="65" text-anchor="middle" font-size="14" fill="#1e3a5f" font-weight="600">∫ μ dl</text>
<text x="265" y="83" text-anchor="middle" font-size="10.5" fill="#5b7897">line integral along ray</text>
<path d="M340,70 H378" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#meth-arrow)"></path>
<rect x="380" y="40" width="160" height="60" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="460" y="63" text-anchor="middle" font-size="13" fill="#1e3a5f" font-weight="600">T = exp(-∫μ dl)</text>
<text x="460" y="81" text-anchor="middle" font-size="10.5" fill="#5b7897">transmission — Eq 1</text>
<path d="M540,70 H578" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#meth-arrow)"></path>
<rect x="580" y="40" width="110" height="60" rx="8" fill="#f0fdf4" stroke="#4ade80" stroke-width="1.5"></rect>
<text x="635" y="63" text-anchor="middle" font-size="13" fill="#14532d" font-weight="600">-log T</text>
<text x="635" y="81" text-anchor="middle" font-size="10.5" fill="#3f6c4f">image — Eq 2</text>
</svg>
<figcaption>The pipeline every method in this document evaluates. All five compute the same four stages — they differ only in <em>how</em> <code>μ(l)</code> is looked up (mono- vs. polychromatic, with or without scatter) and <em>how</em> <code>∫μ dl</code> is evaluated (exact chord lengths vs. fixed-step sampling); see the table below for exactly how each of the five differs at those two steps.</figcaption>
</figure>

## Shared imaging geometry

Every renderer takes the same `GeometryConfig` (`src/drr/renderers/config.py`) and has to convert
it into its own coordinate convention — this is the shared physical setup underneath all five
methods, before any per-method ray-integration choices come in.

<figure>
<svg viewBox="0 0 620 260" width="100%" style="max-width:620px;font-family:-apple-system,Segoe UI,Roboto,sans-serif" role="img" aria-labelledby="geo-title geo-desc">
<title id="geo-title">Cone-beam source/isocenter/detector geometry</title>
<desc id="geo-desc">A source sits SAD=850mm from the isocenter along the view direction, with the detector plane a further 170mm beyond it, for a total source-to-detector distance of 1020mm.</desc>
<defs>
<marker id="geo-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#334155"></path>
</marker>
<marker id="geo-dim" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"></path>
</marker>
</defs>
<path d="M42,90 L42,110 L60,100 Z" fill="#1e3a5f"></path>
<text x="51" y="78" text-anchor="middle" font-size="11" fill="#334155">source</text>
<rect x="440" y="65" width="70" height="70" fill="#eff6ff" fill-opacity="0.7" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="475" y="55" text-anchor="middle" font-size="11" fill="#334155">isocenter (0,0,0)</text>
<circle cx="475" cy="100" r="3" fill="#1e3a5f"></circle>
<path d="M475,60 V30" stroke="#d97706" stroke-width="2" fill="none" marker-end="url(#geo-arrow)"></path>
<text x="475" y="22" text-anchor="middle" font-size="10.5" fill="#8a6d1f" font-weight="600">up = +Z</text>
<path d="M560,40 V160" stroke="#334155" stroke-width="1.5"></path>
<text x="560" y="30" text-anchor="middle" font-size="11" fill="#334155">detector plane</text>
<path d="M60,100 H552" stroke="#2b6cb0" stroke-width="2" fill="none" marker-end="url(#geo-arrow)"></path>
<text x="300" y="92" text-anchor="middle" font-size="10.5" fill="#1e3a5f" font-weight="600">view_dir = (0,1,0)</text>
<path d="M50,180 V190 M475,180 V190" stroke="#94a3b8" stroke-width="1"></path>
<path d="M52,185 H473" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-start="url(#geo-dim)" marker-end="url(#geo-dim)"></path>
<text x="262" y="202" text-anchor="middle" font-size="11" fill="#334155" font-weight="600">SAD = 850 mm</text>
<path d="M50,215 V225 M560,215 V225" stroke="#94a3b8" stroke-width="1"></path>
<path d="M52,220 H558" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-start="url(#geo-dim)" marker-end="url(#geo-dim)"></path>
<text x="305" y="237" text-anchor="middle" font-size="11" fill="#334155" font-weight="600">SDD = 1020 mm</text>
<text x="517" y="118" text-anchor="middle" font-size="9.5" fill="#64748b">170mm</text>
</svg>
<figcaption>The default <code>DEFAULT_GEOMETRY</code>: source-to-axis distance (SAD) 850mm, source-to-detector distance (SDD) 1020mm — so the detector sits 170mm beyond the isocenter, not at it. Default view direction is <code>+Y</code> (AP projection) with <code>+Z</code> as up. Every renderer places its own camera/ray-source this same 850mm from the isocenter and its detector this same 1020mm away; getting that translation right per renderer is what <code>CLAUDE.md</code> calls "the central hard problem in this codebase."</figcaption>
</figure>

Two gotchas worth calling out precisely, since they're easy to get backwards:

- **The detector is not at the isocenter.** SDD ≠ SAD (1020mm vs. 850mm) — a renderer that places its detector plane at the isocenter instead of 170mm beyond it will still produce a plausible-looking image (just at the wrong magnification/field of view), which is why this bug class is hard to catch by eye.
- **Plastimatch's `-n` flag wants the detector *normal*, not the view direction** — `generate_plastimatch_drr` passes `-n f"{-geo.view_dir_x} {-geo.view_dir_y} {-geo.view_dir_z}"`, the negation of `view_dir`, matching the note in `CLAUDE.md`.

## The five methods

Each method's full write-up — including its formulas — lives in its own file under
[`methods/`](methods):

1. [Plastimatch](methods/plastimatch.md) — exact, monochromatic, no scatter (ground truth)
2. [Monte Carlo](methods/monte_carlo.md) (secondary reference) — polychromatic ray-marching with a
   scatter approximation
3. [DiffDRR](methods/diffdrr.md) — the same exact integration as Plastimatch, made differentiable
4. [DVR](methods/dvr.md) — the graphics volume-rendering equation, repurposed for attenuation
5. [DeepDRR](methods/deepdrr.md) — exact ray tracing through segmented materials, polychromatic but
   scatter-free in this benchmark (the installed library's scatter simulation is deprecated)

## How they compare

| Method | Ray integration | Spectrum | Scatter |
|---|---|---|---|
| Plastimatch | exact (Siddon) | monochromatic | none |
| DiffDRR (Siddon) | exact (Siddon) | monochromatic | none |
| DiffDRR (Trilinear) | sampled (trilinear interpolation) | monochromatic | none |
| DVR | sampled (Riemann sum, graphics volume-rendering equation) | monochromatic | none |
| Monte Carlo | sampled (fixed-step march) | polychromatic (80 kVp, multi-bin spectrum) | Gaussian-blur heuristic |
| DeepDRR | exact ray tracing through segmented materials | polychromatic (90 kVp, library default) | none (deprecated in the installed library) |

Plastimatch and DiffDRR's Siddon mode solve exactly the same equation, which is why they agree so
closely with each other. DVR and DiffDRR's Trilinear mode both introduce a small sampling
approximation on top of the same underlying integral. Monte Carlo departs further, on purpose —
it's modeling beam hardening and scatter that the exact, scatter-free renderers don't attempt at
all, so a bigger difference from Plastimatch there reflects a different physical model, not lower
quality. DeepDRR also models beam hardening (via its own, differently-configured polychromatic
spectrum) but — in this benchmark, unlike in DeepDRR's own published pipeline — adds no scatter on
top, since the installed `deepdrr` package has deprecated that feature entirely; see
[DeepDRR](methods/deepdrr.md) for the full explanation.

![Rendered output side-by-side against Plastimatch ground truth](../assets/results/quality_comparison.png)

*All five renderers on the same volume/geometry — visually confirms the ray-integration/scatter
differences described above rather than a coordinate-system misalignment.*

<figure>
<svg viewBox="0 0 560 230" width="100%" style="max-width:560px;font-family:-apple-system,Segoe UI,Roboto,sans-serif" role="img" aria-labelledby="ssim-title ssim-desc">
<title id="ssim-title">Measured SSIM vs. Plastimatch ground truth</title>
<desc id="ssim-desc">DVR and both DiffDRR modes score above 0.91 SSIM against the exact-integration ground truth; Monte Carlo and DeepDRR score lower by design, since they model physics the exact renderers don't attempt.</desc>
<path d="M160,15 V195" stroke="#334155" stroke-width="1"></path>
<path d="M160,195 H530" stroke="#334155" stroke-width="1"></path>
<path d="M160,15 V195 M250,190 V195 M340,190 V195 M430,190 V195 M520,190 V195" stroke="#94a3b8" stroke-width="1"></path>
<text x="160" y="210" text-anchor="middle" font-size="9.5" fill="#64748b">0.80</text>
<text x="340" y="210" text-anchor="middle" font-size="9.5" fill="#64748b">0.90</text>
<text x="520" y="210" text-anchor="middle" font-size="9.5" fill="#64748b">1.00</text>
<text x="345" y="223" text-anchor="middle" font-size="10.5" fill="#334155">SSIM vs. Plastimatch (GT) — axis starts at 0.80</text>
<text x="155" y="34" text-anchor="end" font-size="11" fill="#1e3a5f">DVR</text>
<rect x="160" y="20" width="244" height="20" fill="#2b6cb0"></rect>
<text x="410" y="34" font-size="10.5" fill="#334155" font-weight="600">0.9356</text>
<text x="155" y="70" text-anchor="end" font-size="11" fill="#1e3a5f">DiffDRR (Trilinear)</text>
<rect x="160" y="56" width="229" height="20" fill="#2b6cb0"></rect>
<text x="395" y="70" font-size="10.5" fill="#334155" font-weight="600">0.9272</text>
<text x="155" y="106" text-anchor="end" font-size="11" fill="#1e3a5f">DiffDRR (Siddon)</text>
<rect x="160" y="92" width="200" height="20" fill="#2b6cb0"></rect>
<text x="366" y="106" font-size="10.5" fill="#334155" font-weight="600">0.9113</text>
<text x="155" y="142" text-anchor="end" font-size="11" fill="#1e3a5f">Monte Carlo</text>
<rect x="160" y="128" width="100" height="20" fill="#d97706"></rect>
<text x="266" y="142" font-size="10.5" fill="#334155" font-weight="600">0.8558</text>
<text x="155" y="178" text-anchor="end" font-size="11" fill="#1e3a5f">DeepDRR</text>
<rect x="160" y="164" width="78" height="20" fill="#d97706"></rect>
<text x="244" y="178" font-size="10.5" fill="#334155" font-weight="600">0.8432</text>
</svg>
<figcaption>Real measured numbers from this benchmark's <code>run_quality</code> (200×200, AP view), not illustrative. Blue = exact/near-exact ray integrators (DVR, both DiffDRR modes) clustering above 0.91 SSIM; amber = the two methods deliberately modeling beam hardening/scatter that Plastimatch's ground truth doesn't have, scoring lower by design rather than by defect. Full RMSE/PSNR numbers for all five are in <code>README.md</code> §"Sample results".</figcaption>
</figure>
