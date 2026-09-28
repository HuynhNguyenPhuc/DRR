# DiffDRR

*(part of [Methodology](../METHODOLOGY.md))*

DiffDRR computes the identical exact, monochromatic, scatter-free Siddon line integral Plastimatch
does — same physics, same algorithm — but implemented so every operation from the camera's pose (a
rotation and a translation) through to the final pixel value is differentiable. This is what makes
gradient-based 2D/3D registration possible: recovering an unknown pose by repeatedly nudging it in
the direction that makes the rendered image look more like a target image.

## Algorithm

1. **Field of view.** Compute per-pixel spacing from the volume's physical size.
2. **Camera basis.** Build an orthonormal basis from the configured view direction $\mathbf{d}$
   and up vector $\mathbf{u}$ (Eq. 1).
3. **Pose construction.** Map that basis onto the renderer's base camera frame via a rotation
   matrix (Eq. 2), convert it to Euler angles, and solve for the translation that places the
   camera at the correct world position (Eq. 3).
4. **Exact integration.** Render using the same Siddon line integral as Plastimatch (Eq. 4) — and
   since every step is differentiable, the gradient of the rendered pixels with respect to the
   pose parameters can be computed directly (Eq. 5).
5. **Trilinear mode.** As an alternative, sample the volume at evenly spaced points and sum,
   trading exactness for simplicity (Eq. 6).

<figure>
<img src="../../assets/results/method_crops/diffdrr_siddon.png" alt="DiffDRR Siddon-mode rendered chest DRR, AP view" width="270">
<img src="../../assets/results/method_crops/diffdrr_trilinear.png" alt="DiffDRR trilinear-mode rendered chest DRR, AP view" width="270">
<figcaption>DiffDRR's two rendering modes on the same CT/geometry: Siddon (left, Eq 4) is pixel-for-pixel the same exact integral Plastimatch computes, made differentiable; Trilinear (right, Eq 6) trades that exactness for a simple sampled sum. Both are used only to produce the image — 2D/3D registration (Eq 5) works on the gradient of these pixels with respect to pose, not on a separate rendering.</figcaption>
</figure>

<figure>
<svg viewBox="0 0 600 300" width="100%" style="max-width:600px;font-family:-apple-system,Segoe UI,Roboto,sans-serif" role="img" aria-labelledby="ddov-title ddov-desc">
<title id="ddov-title">One pixel's gradient, flowing back to the pose</title>
<desc id="ddov-desc">Rendering runs forward from pose to pixel; because every step is differentiable, the same path runs backward from a pixel's loss gradient to a gradient with respect to pose, which is what registration actually optimizes.</desc>
<rect x="20" y="20" width="260" height="260" fill="none" stroke="#93c5fd" stroke-width="1.5"></rect>
<image href="../../assets/results/method_crops/diffdrr_siddon_small.png" x="20" y="20" width="260" height="260"></image>
<circle cx="150" cy="165" r="7" fill="none" stroke="#facc15" stroke-width="3"></circle>
<path d="M157,165 C230,165 250,150 300,150" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4 3" fill="none"></path>
<rect x="300" y="20" width="280" height="260" rx="8" fill="#fafbfc" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="440" y="40" text-anchor="middle" font-size="11.5" fill="#1e3a5f" font-weight="600">forward render, backward gradient</text>
<defs>
<marker id="ddov-fwd" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"></path>
</marker>
<marker id="ddov-bwd" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#d97706"></path>
</marker>
</defs>
<circle cx="350" cy="135" r="30" fill="none" stroke="#2b6cb0" stroke-width="1.5"></circle>
<path d="M338,115 A24,24 0 1,1 333,135" stroke="#2b6cb0" stroke-width="2" fill="none" marker-end="url(#ddov-fwd)"></path>
<path d="M332,135 H368" stroke="#2b6cb0" stroke-width="2" fill="none" marker-end="url(#ddov-fwd)"></path>
<text x="350" y="180" text-anchor="middle" font-size="12" fill="#1e3a5f" font-weight="600">pose (R, t)</text>
<rect x="470" y="95" width="80" height="80" fill="none" stroke="#93c5fd" stroke-width="1.5"></rect>
<path d="M510,95 V175 M470,135 H550" stroke="#93c5fd" stroke-width="1"></path>
<rect x="510" y="135" width="40" height="40" fill="#fefce8" stroke="#eab308" stroke-width="2"></rect>
<text x="510" y="87" text-anchor="middle" font-size="11" fill="#5b7897">rendered image</text>
<path d="M382,150 H468" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#ddov-fwd)"></path>
<text x="425" y="167" text-anchor="middle" font-size="10" fill="#64748b">render (Eq 4/6)</text>
<path d="M478,118 H384" stroke="#d97706" stroke-width="2.5" fill="none" marker-end="url(#ddov-bwd)"></path>
<text x="425" y="108" text-anchor="middle" font-size="10.5" fill="#8a6d1f" font-weight="600">∇ (Eq 5)</text>
</svg>
<figcaption>The circled pixel in the real render (left) sits at the end of the forward path (gray) from pose to pixel — but because every step is differentiable, that path also runs backward (amber): the loss gradient at this one pixel propagates all the way back to a gradient with respect to <code>(R, t)</code> (Eq 5), which is exactly what the registration loop below descends on.</figcaption>
</figure>

<figure>
<svg viewBox="0 0 380 180" width="100%" style="max-width:380px;font-family:-apple-system,Segoe UI,Roboto,sans-serif" role="img" aria-labelledby="ddex-title ddex-desc">
<title id="ddex-title">Exact vs. sampled integration along one ray</title>
<desc id="ddex-desc">The Siddon mode marks the ray at every irregular voxel-boundary crossing it actually hits; the Trilinear mode marks it at fixed, evenly spaced intervals regardless of where the voxel boundaries are.</desc>
<rect x="10" y="10" width="175" height="150" rx="8" fill="#fafbfc" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="97" y="32" text-anchor="middle" font-size="12" fill="#1e3a5f" font-weight="600">Siddon (exact)</text>
<path d="M25,95 H170" stroke="#2b6cb0" stroke-width="2" fill="none"></path>
<path d="M38,85 V105 M56,85 V105 M92,85 V105 M108,85 V105 M138,85 V105" stroke="#1e3a5f" stroke-width="2"></path>
<text x="97" y="128" text-anchor="middle" font-size="10.5" fill="#5b7897">irregular spacing = voxel</text>
<text x="97" y="142" text-anchor="middle" font-size="10.5" fill="#5b7897">boundaries, lᵢ varies</text>
<rect x="195" y="10" width="175" height="150" rx="8" fill="#fafbfc" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="282" y="32" text-anchor="middle" font-size="12" fill="#1e3a5f" font-weight="600">Trilinear (sampled)</text>
<path d="M210,95 H355" stroke="#2b6cb0" stroke-width="2" fill="none"></path>
<circle cx="220" cy="95" r="3.5" fill="#1e3a5f"></circle>
<circle cx="240" cy="95" r="3.5" fill="#1e3a5f"></circle>
<circle cx="260" cy="95" r="3.5" fill="#1e3a5f"></circle>
<circle cx="280" cy="95" r="3.5" fill="#1e3a5f"></circle>
<circle cx="300" cy="95" r="3.5" fill="#1e3a5f"></circle>
<circle cx="320" cy="95" r="3.5" fill="#1e3a5f"></circle>
<circle cx="340" cy="95" r="3.5" fill="#1e3a5f"></circle>
<text x="282" y="128" text-anchor="middle" font-size="10.5" fill="#5b7897">fixed spacing Δl,</text>
<text x="282" y="142" text-anchor="middle" font-size="10.5" fill="#5b7897">independent of voxel grid</text>
</svg>
<figcaption>Illustrative, not to scale: the same ray marked two ways. Siddon (Eq 4) stops only where the ray actually crosses a voxel boundary, so segment length <code>lᵢ</code> varies with the geometry — no discretisation error. Trilinear (Eq 6) instead samples at a fixed interval <code>Δl</code> chosen ahead of time, trading that exactness for simplicity.</figcaption>
</figure>

<figure>
<svg viewBox="0 0 420 230" width="100%" style="max-width:420px;font-family:-apple-system,Segoe UI,Roboto,sans-serif" role="img" aria-labelledby="ddreg-title ddreg-desc">
<title id="ddreg-title">The 2D/3D registration loop</title>
<desc id="ddreg-desc">Pose feeds rendering, rendering feeds a loss against the target image, the loss's gradient with respect to pose feeds a pose update, which closes the loop.</desc>
<defs>
<marker id="ddreg-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"></path>
</marker>
</defs>
<rect x="20" y="20" width="170" height="55" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="105" y="42" text-anchor="middle" font-size="12.5" fill="#1e3a5f" font-weight="600">pose</text>
<text x="105" y="60" text-anchor="middle" font-size="11" fill="#5b7897">(R, t)</text>
<path d="M190,47 H228" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#ddreg-arrow)"></path>
<rect x="230" y="20" width="170" height="55" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="315" y="42" text-anchor="middle" font-size="12.5" fill="#1e3a5f" font-weight="600">render</text>
<text x="315" y="60" text-anchor="middle" font-size="11" fill="#5b7897">Eq 4 or Eq 6</text>
<path d="M315,75 V146" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#ddreg-arrow)"></path>
<rect x="230" y="150" width="170" height="55" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"></rect>
<text x="315" y="172" text-anchor="middle" font-size="12.5" fill="#1e3a5f" font-weight="600">ZNCC loss</text>
<text x="315" y="190" text-anchor="middle" font-size="11" fill="#5b7897">vs. target image</text>
<path d="M228,177 H192" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#ddreg-arrow)"></path>
<rect x="20" y="150" width="170" height="55" rx="8" fill="#fefce8" stroke="#eab308" stroke-width="1.5"></rect>
<text x="105" y="172" text-anchor="middle" font-size="12.5" fill="#713f12" font-weight="600">∇ w.r.t. (R, t)</text>
<text x="105" y="190" text-anchor="middle" font-size="11" fill="#8a6d1f">Eq 5</text>
<path d="M105,150 V79" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#ddreg-arrow)"></path>
<text x="112" y="118" font-size="10" fill="#64748b">update pose</text>
</svg>
<figcaption>The registration loop this differentiability enables, with the actual numbers <code>run_optimization</code> uses: start 15° off the ground-truth pose, render at 150×150 (Eq 4/6), score against the target with negative ZNCC (Eq 5), back-propagate to <code>∇(R,t)</code>, and step with Adam (<code>lr=1e-2</code>) — repeating until the loss reaches -0.99 (ZNCC ≥ 0.99) or the iteration budget runs out. This is what <code>run_optimization</code> does for DVR and DiffDRR, the two differentiable renderers in this comparison; see <code>README.md</code> §"Sample results" for measured convergence iteration counts.</figcaption>
</figure>

## Mathematical Formulation

**(1) Orthonormal camera basis.** Given normalised view $\mathbf{d}$ and up $\mathbf{u}$:

$$
\mathbf{r} = \mathbf{d}\times\mathbf{u}, \qquad \mathbf{u} \leftarrow \mathbf{r}\times\mathbf{d}
$$

(the second step re-orthogonalises $\mathbf{u}$ against $\mathbf{d}$ and $\mathbf{r}$).

**(2) Rotation matrix.** The renderer's base camera looks down $[0,-1,0]$ with up $[0,0,1]$ and
right $[-1,0,0]$. Mapping those axes onto $\mathbf{d}$, $\mathbf{u}$, $\mathbf{r}$ respectively
gives a rotation matrix whose columns are

$$
R = \begin{bmatrix} -\mathbf{r} & -\mathbf{d} & \mathbf{u} \end{bmatrix}
$$

**(3) Translation.** With camera centre $= R\cdot\mathbf{t}$ in this convention, and the camera
required to sit at $\text{isocenter} - \text{SAD}\cdot\mathbf{d}$, solving
$\mathbf{t} = R^\top(\text{isocenter}-\text{SAD}\cdot\mathbf{d})$ gives

$$
t_x = -\mathbf{r}\cdot I_w, \qquad t_y = -\mathbf{d}\cdot I_w+\text{SAD}, \qquad t_z = \mathbf{u}\cdot I_w
$$

where $I_w$ is the isocenter in world coordinates.

**(4) Exact line integral — identical to Plastimatch.**

$$
\int \mu\,dl \approx \sum_i \mu_i\cdot l_i
$$

with exact per-voxel chord lengths $l_i$ (see [Plastimatch](plastimatch.md)).

**(5) Differentiability.** Since $l_i$ is a differentiable function of the ray's endpoints, which
are themselves a differentiable function of $(R,\mathbf{t})$, the pipeline supports
$\partial(\text{pixel})/\partial(R,\mathbf{t})$. `run_optimization` (`benchmarks.py`) uses this to
recover a 15°-perturbed pose: minimise the negative zero-normalised cross-correlation
(`utils/losses.py`'s `zncc_loss`) between rendered and target images —

$$
\mathcal{L}(R,\mathbf{t}) = -\,\frac{\sum_p \big(\hat{I}_p-\bar{\hat I}\big)\big(I_p-\bar I\big)}
{\sqrt{\sum_p\big(\hat{I}_p-\bar{\hat I}\big)^2\cdot\sum_p\big(I_p-\bar I\big)^2}}
$$

(rendered image $\hat I$, target $I$, both zero-meaned over pixels $p$) — with Adam
($\eta=10^{-2}$), until $\mathcal{L}\le-0.99$ (equivalently ZNCC $\ge 0.99$) or the iteration
budget runs out:

$$
(R,\mathbf{t}) \leftarrow \text{Adam-step}\big((R,\mathbf{t}),\ \nabla_{(R,\mathbf{t})}\mathcal{L}\big)
$$

**(6) Trilinear sampled integral.**

$$
\int \mu\,dl \approx \left(\sum_k \mu(\mathbf{p}_k)\right)\cdot\Delta l, \qquad \Delta l=\frac{\text{ray length}}{N}
$$

the same sampled-integral shortcut DVR uses (see [DVR](dvr.md)), without the
volume-rendering-equation detour.
