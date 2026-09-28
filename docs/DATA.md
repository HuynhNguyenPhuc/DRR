# Data Loading (`src/drr/data.py`)

*(part of [Documentation Router](INDEX.md))*

Every renderer needs the same two things before it can render anything: a CT density volume as a
`(1, 1, D, H, W)` tensor normalised to `[0, 1]`, and a voxel spacing in mm. `load_ct_volume()` is
the single entry point that produces both, falling back through three sources so the benchmark
never hard-fails for lack of data.

## Algorithm

1. **Source 1 — user NIfTI.** If a path is given, load it with `nibabel`, min-max normalise to
   `[0, 1]`, and average any anisotropic per-axis spacing into one isotropic value (Eq. 1) —
   downstream code assumes a single spacing.
2. **Source 2 — DiffDRR's example CT.** If no path was given (or loading it failed), fall back to
   DiffDRR's bundled example chest CT via `load_example_ct()`. This is the only source that
   returns a `torchio.Subject` directly, since it's the one DiffDRR already knows how to render.
3. **Source 3 — synthetic phantom.** If both of the above are unavailable (missing dependency, bad
   path, etc.), generate a spherical two-density phantom analytically (Eq. 2). This source cannot
   fail, which is what guarantees the pipeline always produces *something* to render.
4. **Resample.** Trilinearly interpolate whichever volume was produced to `target_size³` voxels,
   rescaling `voxel_spacing` so the real-world physical size stays constant (Eq. 3) — resampling
   changes voxel count, not the volume's actual size in mm.
5. **Clamp.** Clip the final tensor to `[0, 1]`, since density outside that range isn't physical.

<figure>
<svg viewBox="0 0 460 590" width="100%" style="max-width:420px;font-family:-apple-system,Segoe UI,Roboto,sans-serif" role="img" aria-labelledby="data-fig-title data-fig-desc">
<title id="data-fig-title">Three-source fallback chain</title>
<desc id="data-fig-desc">load_ct_volume tries a user NIfTI, then DiffDRR's example CT, then a synthetic phantom that always succeeds, before resampling and clamping to produce the final tuple.</desc>
<defs>
<marker id="data-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"></path>
</marker>
</defs>
<rect x="20" y="16" width="230" height="60" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="135" y="41" text-anchor="middle" font-size="13" fill="#1e3a5f" font-weight="600">Source 1</text>
<text x="135" y="58" text-anchor="middle" font-size="10.5" fill="#5b7897">user NIfTI (nibabel)</text>
<path d="M135,76 V150" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#data-arrow)"></path>
<text x="145" y="116" font-size="10" fill="#94a3b8">missing / fails</text>
<path d="M250,46 H310 V435 H252" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#data-arrow)"></path>
<text x="315" y="42" font-size="10" fill="#94a3b8">ok</text>
<rect x="20" y="152" width="230" height="60" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="135" y="177" text-anchor="middle" font-size="13" fill="#1e3a5f" font-weight="600">Source 2</text>
<text x="135" y="194" text-anchor="middle" font-size="10.5" fill="#5b7897">DiffDRR example CT</text>
<path d="M135,212 V286" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#data-arrow)"></path>
<text x="145" y="252" font-size="10" fill="#94a3b8">fails</text>
<path d="M250,182 H280 V471 H252" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#data-arrow)"></path>
<text x="285" y="178" font-size="10" fill="#94a3b8">ok</text>
<rect x="20" y="288" width="230" height="60" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="135" y="313" text-anchor="middle" font-size="13" fill="#1e3a5f" font-weight="600">Source 3</text>
<text x="135" y="330" text-anchor="middle" font-size="10.5" fill="#5b7897">synthetic phantom (always ok)</text>
<path d="M135,348 V421" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#data-arrow)"></path>
<rect x="20" y="423" width="230" height="60" rx="8" fill="#fefce8" stroke="#eab308" stroke-width="1.5"></rect>
<text x="135" y="447" text-anchor="middle" font-size="12" fill="#713f12" font-weight="600">resample to target³</text>
<text x="135" y="464" text-anchor="middle" font-size="10.5" fill="#8a6d1f">rescale spacing, clamp [0,1]</text>
<path d="M135,483 V520" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#data-arrow)"></path>
<rect x="20" y="522" width="230" height="60" rx="8" fill="#f0fdf4" stroke="#4ade80" stroke-width="1.5"></rect>
<text x="135" y="546" text-anchor="middle" font-size="11.5" fill="#14532d" font-weight="600">(volume_tensor, subject,</text>
<text x="135" y="563" text-anchor="middle" font-size="11.5" fill="#14532d" font-weight="600">voxel_spacing)</text>
</svg>
<figcaption>Sequential fallback: each source is only tried if the previous one is unavailable or fails; Source 3 is guaranteed to succeed, so the function never returns without a volume. Only Source 2 populates <code>subject</code> directly — Sources 1 and 3 leave it <code>None</code>, which is why non-DiffDRR-sourced volumes need <code>make_diffdrr_subject()</code> below.</figcaption>
</figure>

Only the DiffDRR-example path returns a real `subject`; the other two return `subject=None`, so
non-DiffDRR-sourced volumes need a `torchio.Subject` built by hand — that's what
`make_diffdrr_subject()` is for.

## `make_diffdrr_subject()`

DiffDRR's own subject-building helpers (`canonicalize()`, the affine/reorient setup) don't work
standalone on an arbitrary tensor — they expect a subject that's already been through DiffDRR's
own loader. `make_diffdrr_subject()` reconstructs what they need manually:

1. Build a world-space affine that centers the volume at the origin (Eq. 4).
2. Wrap the tensor in two `torchio.ScalarImage`s (`volume` and `density` — `DRR.__init__` reads
   `subject.density.data`, and since the tensor is already normalised `[0, 1]`, density and volume
   are the same array).
3. Set `mask=None` and `fiducials=None` explicitly as attributes (not constructor kwargs — a
   `torchio.Subject` can silently drop non-`Image` kwargs depending on the installed version), and
   attach the AP reorientation matrix DiffDRR expects, rotating the C-arm 90° about the x-axis so
   the detector faces the AP direction (Eq. 5).
4. Run the result through `canonicalize()` before returning it.

## Mathematical Formulation

**(1) Isotropic spacing approximation.** For anisotropic per-axis spacing $(s_x, s_y, s_z)$:

$$
s_{\text{iso}} = \frac{s_x + s_y + s_z}{3}
$$

logged as a warning whenever $\max(s_x,s_y,s_z)/\min(s_x,s_y,s_z) > 1.05$.

**(2) Synthetic phantom.** With normalised radius $r = \sqrt{x^2+y^2+z^2}$ over
$(x,y,z) \in [-1,1]^3$:

$$
\rho(x,y,z) = 0.3\cdot\mathbb{1}[r<0.40] + 0.7\cdot\mathbb{1}[r<0.15]
$$

giving a dense inner sphere (0.7 + 0.3 = 1.0 where both indicators fire) inside a lower-density
outer shell — a two-material phantom with a clean, known geometry for sanity-checking orientation.

**(3) Spacing rescale after resampling.** Physical size must stay constant across a change in
voxel count $D \to D'$:

$$
s' = \frac{s\cdot D}{D'}
$$

**(4) Centring affine.** For voxel spacing $s$ and volume side length $D$ (voxels), the affine
maps voxel index $i$ to world-space mm coordinate $s\cdot i - \tfrac{sD}{2}$, so index $D/2$ lands
at the world origin.

**(5) AP reorientation.** A 90° rotation about the x-axis:

$$
R_{\text{AP}} = \begin{bmatrix} 1&0&0\\0&0&-1\\0&1&0\end{bmatrix}
$$
