# DVR

*(part of [Methodology](../METHODOLOGY.md))*

This method starts from a different equation entirely: the volume-rendering equation used in
computer graphics for translucent media like smoke or clouds, where every point along a ray has
both an opacity and a colour, accumulated front-to-back. There's no meaningful "emitted colour" for
a CT voxel, so density is reused for both terms — and the resulting accumulation turns out to be
mathematically equivalent to the same Beer-Lambert integral every other method computes, just
arrived at from a different starting equation and evaluated as a finite Riemann sum rather than an
exact integral.

## Algorithm

1. **Unit conversion.** Scale physical mm units into a normalized device coordinate (NDC) cube
   $[-1,1]^3$ (Eq. 1).
2. **Camera basis.** Build the same orthonormal view/up basis used elsewhere (Eq. 1 in
   [DiffDRR](diffdrr.md)), oriented for this coordinate convention.
3. **Camera placement.** Solve for the source position and translation in NDC space (Eq. 2).
4. **Volume rendering.** Accumulate opacity and colour front-to-back along each ray using the
   volume-rendering equation (Eq. 3), reusing density for both terms (Eq. 4).
5. **Reversed accumulation.** Raymarch back-to-front (reversed from the typical graphics order) so
   absorbing structures like bone come out bright, then reduce to the final per-ray opacity
   (Eq. 5) — the value used as the rendered pixel.

<figure>
<img src="../../assets/results/method_crops/dvr.png" alt="DVR rendered chest DRR, AP view" width="360">
<figcaption>Actual DVR output on the same CT/geometry as the Plastimatch reference. Despite starting from a completely different equation (the graphics volume-rendering equation, Eq 3, rather than Beer-Lambert directly), the reversed-accumulation trick (Eq 5) converges to a visually near-identical result once enough samples are taken along each ray.</figcaption>
</figure>

<figure>
<svg viewBox="0 0 600 300" width="100%" style="max-width:600px;font-family:-apple-system,Segoe UI,Roboto,sans-serif" role="img" aria-labelledby="dvrov-title dvrov-desc">
<title id="dvrov-title">One pixel's ray, shown passing through the volume in perspective</title>
<desc id="dvrov-desc">A camera casts a ray through a 3D volume to a detector plane; samples along that one highlighted ray accumulate opacity as they go, becoming the circled pixel.</desc>
<rect x="20" y="20" width="260" height="260" fill="none" stroke="#93c5fd" stroke-width="1.5"></rect>
<image href="../../assets/results/method_crops/dvr_small.png" x="20" y="20" width="260" height="260"></image>
<circle cx="150" cy="165" r="7" fill="none" stroke="#facc15" stroke-width="3"></circle>
<path d="M157,165 C230,165 250,150 300,150" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4 3" fill="none"></path>
<rect x="300" y="20" width="280" height="260" rx="8" fill="#fafbfc" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="440" y="40" text-anchor="middle" font-size="11.5" fill="#1e3a5f" font-weight="600">this pixel's ray, in perspective</text>
<path d="M430,88 L530,88 L530,198 L430,198 Z" fill="#eff6ff" fill-opacity="0.5" stroke="#93c5fd" stroke-width="1.5"></path>
<path d="M400,110 L500,110 L500,220 L400,220 Z" fill="#eff6ff" fill-opacity="0.85" stroke="#93c5fd" stroke-width="1.5"></path>
<path d="M400,110 L430,88 M500,110 L530,88 M400,220 L430,198 M500,220 L530,198" stroke="#93c5fd" stroke-width="1.5"></path>
<text x="465" y="78" text-anchor="middle" font-size="10.5" fill="#5b7897">volume</text>
<path d="M312,155 L312,175 L328,165 Z" fill="#1e3a5f"></path>
<text x="305" y="145" text-anchor="end" font-size="10" fill="#64748b">camera</text>
<path d="M570,105 V225" stroke="#334155" stroke-width="1.5"></path>
<text x="567" y="102" text-anchor="end" font-size="10" fill="#64748b">detector</text>
<path d="M328,172 L570,190" stroke="#94a3b8" stroke-width="1.5" fill="none"></path>
<path d="M328,165 L570,165" stroke="#94a3b8" stroke-width="1.5" fill="none"></path>
<circle cx="570" cy="190" r="3" fill="#94a3b8"></circle>
<circle cx="570" cy="165" r="3" fill="#94a3b8"></circle>
<path d="M328,158 L570,140" stroke="#d97706" stroke-width="2" fill="none"></path>
<circle cx="376" cy="154" r="4" fill="#d97706" fill-opacity="0.2"></circle>
<circle cx="425" cy="151" r="4" fill="#d97706" fill-opacity="0.4"></circle>
<circle cx="473" cy="147" r="4" fill="#d97706" fill-opacity="0.6"></circle>
<circle cx="522" cy="144" r="4" fill="#d97706" fill-opacity="0.8"></circle>
<circle cx="570" cy="140" r="4.5" fill="#d97706"></circle>
<text x="570" y="130" text-anchor="middle" font-size="10" fill="#8a6d1f" font-weight="600">this pixel</text>
</svg>
<figcaption>The circled pixel in the real DVR render (left) is one camera ray (amber, right) passing through the volume to the detector — the four faint-to-solid dots along it are the same six-sample accumulation from the diagram above, just drawn along the actual 3D ray instead of unrolled into a row. Every other pixel is a different parallel ray through the same volume, landing at a different point on the detector plane.</figcaption>
</figure>

<figure>
<svg viewBox="0 0 360 165" width="100%" style="max-width:360px;font-family:-apple-system,Segoe UI,Roboto,sans-serif" role="img" aria-labelledby="dvrex-title dvrex-desc">
<title id="dvrex-title">Front-to-back opacity accumulation</title>
<desc id="dvrex-desc">Six samples along one ray, each with density reused as opacity; the remaining transmittance bar shrinks toward zero as the accumulated opacity approaches one.</desc>
<defs>
<marker id="dvrex-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"></path>
</marker>
</defs>
<path d="M10,50 H345" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#dvrex-arrow)"></path>
<text x="10" y="42" font-size="10.5" fill="#64748b">ray march, k = 1…N</text>
<rect x="20" y="60" width="46" height="46" fill="#2b6cb0" fill-opacity="0.15" stroke="#93c5fd"></rect>
<rect x="74" y="60" width="46" height="46" fill="#2b6cb0" fill-opacity="0.35" stroke="#93c5fd"></rect>
<rect x="128" y="60" width="46" height="46" fill="#2b6cb0" fill-opacity="0.60" stroke="#93c5fd"></rect>
<rect x="182" y="60" width="46" height="46" fill="#2b6cb0" fill-opacity="0.80" stroke="#93c5fd"></rect>
<rect x="236" y="60" width="46" height="46" fill="#2b6cb0" fill-opacity="0.50" stroke="#93c5fd"></rect>
<rect x="290" y="60" width="46" height="46" fill="#2b6cb0" fill-opacity="0.25" stroke="#93c5fd"></rect>
<text x="43" y="52" font-size="9.5" fill="#94a3b8">1</text>
<text x="97" y="52" font-size="9.5" fill="#94a3b8">2</text>
<text x="151" y="52" font-size="9.5" fill="#94a3b8">3</text>
<text x="205" y="52" font-size="9.5" fill="#94a3b8">4</text>
<text x="259" y="52" font-size="9.5" fill="#94a3b8">5</text>
<text x="313" y="52" font-size="9.5" fill="#94a3b8">6</text>
<text x="43" y="119" text-anchor="middle" font-size="10" fill="#1e3a5f" font-weight="600">0.15</text>
<text x="97" y="119" text-anchor="middle" font-size="10" fill="#1e3a5f" font-weight="600">0.45</text>
<text x="151" y="119" text-anchor="middle" font-size="10" fill="#1e3a5f" font-weight="600">0.78</text>
<text x="205" y="119" text-anchor="middle" font-size="10" fill="#1e3a5f" font-weight="600">0.96</text>
<text x="259" y="119" text-anchor="middle" font-size="10" fill="#1e3a5f" font-weight="600">0.98</text>
<text x="313" y="119" text-anchor="middle" font-size="10" fill="#1e3a5f" font-weight="600">0.98</text>
<text x="0" y="139" font-size="10.5" fill="#64748b">Tₖ</text>
<rect x="20" y="133" width="46" height="10" fill="#eef2f7" stroke="#cbd5e1" stroke-width="0.5"></rect>
<rect x="74" y="133" width="46" height="10" fill="#eef2f7" stroke="#cbd5e1" stroke-width="0.5"></rect>
<rect x="128" y="133" width="46" height="10" fill="#eef2f7" stroke="#cbd5e1" stroke-width="0.5"></rect>
<rect x="182" y="133" width="46" height="10" fill="#eef2f7" stroke="#cbd5e1" stroke-width="0.5"></rect>
<rect x="236" y="133" width="46" height="10" fill="#eef2f7" stroke="#cbd5e1" stroke-width="0.5"></rect>
<rect x="290" y="133" width="46" height="10" fill="#eef2f7" stroke="#cbd5e1" stroke-width="0.5"></rect>
<rect x="20" y="133" width="39" height="10" fill="#64748b"></rect>
<rect x="74" y="133" width="25" height="10" fill="#64748b"></rect>
<rect x="128" y="133" width="10" height="10" fill="#64748b"></rect>
<rect x="182" y="133" width="2" height="10" fill="#64748b"></rect>
<rect x="236" y="133" width="1" height="10" fill="#64748b"></rect>
<rect x="290" y="133" width="1" height="10" fill="#64748b"></rect>
</svg>
<figcaption>Illustrative: opacity <code>αₖ = ρₖ</code> (Eq 4) at each sample — darker fill means denser tissue. The number under each sample is the accumulated opacity so far (Eq 3, Eq 5); the bar below it is the remaining transmittance <code>Tₖ = Π(1-αⱼ)</code> — by sample 4 here, accumulated opacity has already reached 0.96, so almost none of the ray's original transmittance is left for samples 5-6 to contribute.</figcaption>
</figure>

## Mathematical Formulation

**(1) NDC scaling.** With physical volume side length $L$:

$$
\text{SDD}_{\text{ndc}} = \text{SDD}\cdot\frac{2}{L}, \qquad \text{SAD}_{\text{ndc}} = \text{SAD}\cdot\frac{2}{L}
$$

**(2) Camera placement.** Screen-space focal length in pixels, for a $W\times H$ image over
physical extent $L$:

$$
f_x=\frac{\text{SDD}}{L/W}, \qquad f_y=\frac{\text{SDD}}{L/H}
$$

Source position in NDC: $S = I_{\text{ndc}} - \text{SAD}_{\text{ndc}}\cdot\mathbf{d}$, giving
camera translation $T=-SR$ under a row-vector camera convention $X_{\text{cam}}=X_{\text{world}}R+T$.

**(3) Volume rendering equation.** With per-sample opacity $\alpha_k$, colour $c_k$, and
transmittance $T_k=\prod_{j<k}(1-\alpha_j)$:

$$
C = \sum_k T_k\,\alpha_k\,c_k
$$

**(4) Density reuse.** Since a CT voxel has no meaningful colour, $\alpha_k=c_k=\rho_k$ (the local
density).

**(5) Final opacity (Beer-Lambert equivalence).**

$$
\text{opacity} = 1-\prod_k(1-\rho_k)
$$

which is a Riemann-sum discretisation of $\exp(-\int\mu\,dl)$'s complement, computed via a reversed
running product of $(1-\rho_k)$ from the far end of the ray backwards (mathematically equivalent to
the direct product above). The number of sample points controls how well this approximates the true
integral: too few under-resolves fine structure; past some point, adding more barely changes the
result because the sum has already converged.
