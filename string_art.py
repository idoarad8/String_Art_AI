"""
Improved String Art Generator (Radon-guided + residual-driven)
- Fixes Radon indexing/geometry (skimage radon output axes were used incorrectly)
- Scores lines using the *residual* (what's still missing), reducing saturation/over-dark scribbling
- Adds anti ping-pong (tabu: don't go back to previous nail)
- Uses adaptive intensity based on current fill
- Keeps optional GPU acceleration (PyTorch) for drawing + scoring
"""

import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from skimage.transform import radon
from scipy.ndimage import gaussian_filter

# Try to import PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


class StringArtGenerator:
    def __init__(
        self,
        image_path: str,
        num_nails: int = 300,
        image_size: int = 500,
        use_gpu: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            image_path: Path to input image
            num_nails: number of nails on circle
            image_size: resized square size
            use_gpu: enable GPU if torch+cuda available
            seed: optional RNG seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            if TORCH_AVAILABLE:
                torch.manual_seed(seed)

        self.num_nails = int(num_nails)
        self.image_size = int(image_size)

        # Setup GPU if available
        self.use_gpu = bool(use_gpu) and TORCH_AVAILABLE
        self.device = None
        if self.use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device.type == "cuda":
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("GPU requested but not available, using CPU")
                self.use_gpu = False
                self.device = None
        else:
            print("Using CPU (PyTorch not available or GPU disabled)")

        # Load and process image
        self.image = self._load_image(image_path)
        self.target_image = self._process_image(self.image)  # float32 [0..1], inverted

        # Nail positions
        self.nail_positions = self._calculate_nail_positions()

        # Working image canvas
        if self.use_gpu:
            self.working_image_gpu = torch.zeros(
                (self.image_size, self.image_size), dtype=torch.float32, device=self.device
            )
            self.working_image = None  # sync when needed
        else:
            self.working_image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            self.working_image_gpu = None

        # Residual (what still needs string)
        self.residual_cpu = None
        self.residual_gpu = None

        self.nail_sequence: List[int] = []

        # Precompute Radon transform
        self._precompute_radon_transform()

        # Precompute nails + target on GPU
        if self.use_gpu:
            self.nail_positions_gpu = torch.from_numpy(self.nail_positions).float().to(self.device)
            self.target_image_gpu = torch.from_numpy(self.target_image).float().to(self.device)
        else:
            self.nail_positions_gpu = None
            self.target_image_gpu = None

    def _load_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path)

    def _process_image(self, image: Image.Image) -> np.ndarray:
        # Convert to grayscale
        if image.mode != "L":
            image = image.convert("L")

        # Resize
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        # Blur slightly
        image = image.filter(ImageFilter.GaussianBlur(radius=1))

        # Normalize [0..1]
        img = np.array(image, dtype=np.float32) / 255.0

        # Invert so dark areas "want more string"
        img = 1.0 - img

        # Mild contrast curve
        img = np.power(img, 0.8)

        return img

    def _calculate_nail_positions(self) -> np.ndarray:
        center = self.image_size / 2.0
        radius = self.image_size * 0.45
        angles = np.linspace(0, 2 * np.pi, self.num_nails, endpoint=False)

        pos = np.zeros((self.num_nails, 2), dtype=np.float32)
        pos[:, 0] = center + radius * np.cos(angles)  # x
        pos[:, 1] = center + radius * np.sin(angles)  # y
        return pos

    # ----------------------------
    # Radon guidance (fixed)
    # ----------------------------
    def _precompute_radon_transform(self):
        print("Computing Radon transform...")

        num_angles = min(360, self.num_nails * 2)
        theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)

        # sinogram shape: (len_r, len_theta)
        sinogram = radon(self.target_image, theta=theta, circle=True)

        self.sinogram = sinogram.astype(np.float32)
        self.theta = theta.astype(np.float32)
        self.num_radon_angles = len(theta)

        # For mapping r: approximate r in [-R, +R]
        R = self.image_size / 2.0
        self.r_min = -R
        self.r_max = +R
        self.sinogram_len_r = sinogram.shape[0]

        # Normalize sinogram
        mn, mx = float(sinogram.min()), float(sinogram.max())
        if mx > mn:
            self.sinogram_normalized = ((sinogram - mn) / (mx - mn)).astype(np.float32)
        else:
            self.sinogram_normalized = np.zeros_like(sinogram, dtype=np.float32)

        print(f"  Radon computed: {sinogram.shape} (r_bins, angles)")

    def _get_line_radon_score(self, nail_a: int, nail_b: int) -> float:
        """
        Map a nail-to-nail line to (r, theta) in Radon space and sample sinogram.
        """
        pa = self.nail_positions[nail_a]
        pb = self.nail_positions[nail_b]

        dx = float(pb[0] - pa[0])
        dy = float(pb[1] - pa[1])
        L = (dx * dx + dy * dy) ** 0.5
        if L < 1e-6:
            return 0.0

        # unit direction along the line
        ux, uy = dx / L, dy / L

        # theta in Radon is normal angle (perpendicular to line direction)
        theta_line = (np.degrees(np.arctan2(uy, ux)) + 90.0) % 180.0

        # closest theta index
        theta_idx = int(np.argmin(np.abs(self.theta - theta_line)))

        # midpoint in centered coordinates
        c = self.image_size / 2.0
        xm = (pa[0] + pb[0]) * 0.5 - c
        ym = (pa[1] + pb[1]) * 0.5 - c

        # normal direction from theta
        th = np.radians(float(self.theta[theta_idx]))
        nx, ny = np.cos(th), np.sin(th)

        r = xm * nx + ym * ny

        # map r -> r_idx
        r_clamped = float(np.clip(r, self.r_min, self.r_max))
        r_idx = int(
            round((r_clamped - self.r_min) / (self.r_max - self.r_min) * (self.sinogram_len_r - 1))
        )
        r_idx = max(0, min(self.sinogram_len_r - 1, r_idx))

        return float(self.sinogram_normalized[r_idx, theta_idx])

    # ----------------------------
    # Drawing
    # ----------------------------
    def _draw_line(self, start_nail: int, end_nail: int, intensity: float = 1.0):
        if self.use_gpu:
            self._draw_line_gpu(start_nail, end_nail, intensity)
        else:
            self._draw_line_cpu(start_nail, end_nail, intensity)

    def _draw_line_cpu(self, start_nail: int, end_nail: int, intensity: float = 1.0):
        sp = self.nail_positions[start_nail]
        ep = self.nail_positions[end_nail]
        x0, y0 = float(sp[0]), float(sp[1])
        x1, y1 = float(ep[0]), float(ep[1])

        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            return

        steps = int(length) + 1
        steps = max(2, steps)

        for i in range(steps):
            t = i / (steps - 1)
            x = x0 + t * dx
            y = y0 + t * dy

            xf = int(np.floor(x))
            yf = int(np.floor(y))
            xc = xf + 1
            yc = yf + 1

            wx = x - xf
            wy = y - yf

            if 0 <= xf < self.image_size and 0 <= yf < self.image_size:
                self.working_image[yf, xf] += intensity * (1 - wx) * (1 - wy)
            if 0 <= xc < self.image_size and 0 <= yf < self.image_size:
                self.working_image[yf, xc] += intensity * wx * (1 - wy)
            if 0 <= xf < self.image_size and 0 <= yc < self.image_size:
                self.working_image[yc, xf] += intensity * (1 - wx) * wy
            if 0 <= xc < self.image_size and 0 <= yc < self.image_size:
                self.working_image[yc, xc] += intensity * wx * wy

    def _draw_line_gpu(self, start_nail: int, end_nail: int, intensity: float = 1.0):
        sp = self.nail_positions_gpu[start_nail]
        ep = self.nail_positions_gpu[end_nail]

        x0, y0 = sp[0], sp[1]
        x1, y1 = ep[0], ep[1]
        dx, dy = x1 - x0, y1 - y0
        length = torch.sqrt(dx * dx + dy * dy)
        if length.item() < 1e-6:
            return

        steps = int(length.item()) + 1
        steps = max(2, steps)

        t = torch.linspace(0, 1, steps, device=self.device)
        x = x0 + t * dx
        y = y0 + t * dy

        xf = torch.floor(x).long()
        yf = torch.floor(y).long()
        xc = xf + 1
        yc = yf + 1

        wx = x - xf.float()
        wy = y - yf.float()

        valid_ff = (xf >= 0) & (xf < self.image_size) & (yf >= 0) & (yf < self.image_size)
        valid_cf = (xc >= 0) & (xc < self.image_size) & (yf >= 0) & (yf < self.image_size)
        valid_fc = (xf >= 0) & (xf < self.image_size) & (yc >= 0) & (yc < self.image_size)
        valid_cc = (xc >= 0) & (xc < self.image_size) & (yc >= 0) & (yc < self.image_size)

        w_ff = intensity * (1 - wx) * (1 - wy)
        w_cf = intensity * wx * (1 - wy)
        w_fc = intensity * (1 - wx) * wy
        w_cc = intensity * wx * wy

        flat = self.working_image_gpu.view(-1)

        if valid_ff.any():
            idx = yf[valid_ff] * self.image_size + xf[valid_ff]
            flat.index_add_(0, idx, w_ff[valid_ff])
        if valid_cf.any():
            idx = yf[valid_cf] * self.image_size + xc[valid_cf]
            flat.index_add_(0, idx, w_cf[valid_cf])
        if valid_fc.any():
            idx = yc[valid_fc] * self.image_size + xf[valid_fc]
            flat.index_add_(0, idx, w_fc[valid_fc])
        if valid_cc.any():
            idx = yc[valid_cc] * self.image_size + xc[valid_cc]
            flat.index_add_(0, idx, w_cc[valid_cc])

    # ----------------------------
    # Residual update
    # ----------------------------
    def _refresh_residual(self):
        """
        Residual = clamp(target - clamp(working,0,1), 0, 1)
        (i.e., what's still missing)
        """
        if self.use_gpu:
            w = torch.clamp(self.working_image_gpu, 0.0, 1.0)
            self.residual_gpu = torch.clamp(self.target_image_gpu - w, 0.0, 1.0)
        else:
            w = np.clip(self.working_image, 0.0, 1.0)
            self.residual_cpu = np.clip(self.target_image - w, 0.0, 1.0)

    # ----------------------------
    # Scoring (residual + Radon)
    # ----------------------------
    def _calculate_line_benefit(self, current_nail: int, candidate_nail: int) -> float:
        if self.use_gpu:
            return self._calculate_line_benefit_gpu(current_nail, candidate_nail)
        return self._calculate_line_benefit_cpu(current_nail, candidate_nail)

    def _calculate_line_benefit_cpu(self, current_nail: int, candidate_nail: int) -> float:
        sp = self.nail_positions[current_nail]
        ep = self.nail_positions[candidate_nail]
        x0, y0 = float(sp[0]), float(sp[1])
        x1, y1 = float(ep[0]), float(ep[1])

        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            return -1e9

        steps = int(length) + 1
        steps = max(2, steps)

        t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        x = x0 + t * dx
        y = y0 + t * dy

        xi = np.round(x).astype(np.int32)
        yi = np.round(y).astype(np.int32)

        valid = (xi >= 0) & (xi < self.image_size) & (yi >= 0) & (yi < self.image_size)
        if not np.any(valid):
            return -1e9

        xv = xi[valid]
        yv = yi[valid]

        # residual already >= 0
        res_vals = self.residual_cpu[yv, xv]
        avg_benefit = float(res_vals.mean())

        radon_score = self._get_line_radon_score(current_nail, candidate_nail)

        # Blend (Radon weight a bit higher now that it's correct)
        combined = 0.80 * avg_benefit + 0.20 * radon_score

        # Small length penalty
        combined -= float(length) * 0.0003
        return combined

    def _calculate_line_benefit_gpu(self, current_nail: int, candidate_nail: int) -> float:
        sp = self.nail_positions_gpu[current_nail]
        ep = self.nail_positions_gpu[candidate_nail]
        x0, y0 = sp[0], sp[1]
        x1, y1 = ep[0], ep[1]

        dx, dy = x1 - x0, y1 - y0
        length = torch.sqrt(dx * dx + dy * dy)
        if length.item() < 1e-6:
            return -1e9

        steps = int(length.item()) + 1
        steps = max(2, steps)

        t = torch.linspace(0.0, 1.0, steps, device=self.device)
        x = x0 + t * dx
        y = y0 + t * dy

        xi = torch.round(x).long()
        yi = torch.round(y).long()

        valid = (xi >= 0) & (xi < self.image_size) & (yi >= 0) & (yi < self.image_size)
        if valid.sum().item() == 0:
            return -1e9

        xv = xi[valid]
        yv = yi[valid]

        res_vals = self.residual_gpu[yv, xv]
        avg_benefit = res_vals.mean().item()

        radon_score = self._get_line_radon_score(current_nail, candidate_nail)

        combined = 0.80 * avg_benefit + 0.20 * radon_score
        combined -= length.item() * 0.0003
        return float(combined)

    # ----------------------------
    # Main generation
    # ----------------------------
    def generate_string_art(
        self,
        num_iterations: int = 3000,
        min_distance: int = 10,
        max_distance: Optional[int] = None,
        residual_refresh_every: int = 25,
        candidates_early: int = 260,
        candidates_late: int = 160,
    ) -> List[int]:
        """
        Args:
            num_iterations: number of lines
            min_distance: avoid adjacent nails
            max_distance: None = half circle
            residual_refresh_every: recompute residual map every N iterations
            candidates_early / candidates_late: candidate sample sizes
        """
        if max_distance is None:
            max_distance = self.num_nails // 2

        # Reset canvas
        if self.use_gpu:
            self.working_image_gpu.zero_()
        else:
            self.working_image.fill(0.0)

        self.nail_sequence = []

        # Start at random nail
        current_nail = int(np.random.randint(0, self.num_nails))
        self.nail_sequence.append(current_nail)

        # Build initial residual
        self._refresh_residual()

        print(f"Generating string art: iterations={num_iterations}, nails={self.num_nails}")
        if self.use_gpu:
            print("  GPU enabled")
        print("  Using residual-driven scoring + Radon guidance")

        all_nails = np.arange(self.num_nails, dtype=np.int32)

        for it in range(num_iterations):
            if (it + 1) % 100 == 0:
                print(f"  Progress: {it + 1}/{num_iterations}")

            if it % residual_refresh_every == 0:
                self._refresh_residual()

            prev_nail = self.nail_sequence[-2] if len(self.nail_sequence) >= 2 else None

            # candidate count schedule
            frac = it / max(1, (num_iterations - 1))
            num_candidates = int(round(candidates_early * (1 - frac) + candidates_late * frac))
            num_candidates = min(num_candidates, self.num_nails)

            candidates = np.random.choice(all_nails, size=num_candidates, replace=False)

            # distance constraint
            d = np.abs(candidates - current_nail)
            d = np.minimum(d, self.num_nails - d)
            valid = (d >= min_distance) & (d <= max_distance)

            valid_candidates = candidates[valid]

            # tabu: avoid immediate backtrack
            if prev_nail is not None and valid_candidates.size > 0:
                valid_candidates = valid_candidates[valid_candidates != prev_nail]

            best_nail = None
            best_score = -1e9

            # Evaluate candidates (loop is fine; drawing is the expensive part and is GPU-accelerated)
            for cand in valid_candidates:
                score = self._calculate_line_benefit(current_nail, int(cand))
                if score > best_score:
                    best_score = score
                    best_nail = int(cand)

            # If no good candidate, relax slightly (still respect min_distance)
            if best_nail is None:
                relaxed = candidates[d >= min_distance]
                if prev_nail is not None and relaxed.size > 0:
                    relaxed = relaxed[relaxed != prev_nail]

                for cand in relaxed[: max(50, len(relaxed))]:
                    score = self._calculate_line_benefit(current_nail, int(cand))
                    if score > best_score:
                        best_score = score
                        best_nail = int(cand)

            # Adaptive intensity (prevents over-dark late stage)
            if self.use_gpu:
                fill = torch.mean(torch.clamp(self.working_image_gpu, 0.0, 1.0)).item()
            else:
                fill = float(np.mean(np.clip(self.working_image, 0.0, 1.0)))

            intensity = 0.35 * (1.0 - fill)
            intensity = float(np.clip(intensity, 0.08, 0.35))

            # Draw if beneficial enough, else move deterministically
            if best_nail is not None and best_score > 0.002:
                self._draw_line(current_nail, best_nail, intensity=intensity)
                self.nail_sequence.append(best_nail)
                current_nail = best_nail
            else:
                # fallback step
                current_nail = (current_nail + min_distance) % self.num_nails
                self.nail_sequence.append(current_nail)

        # sync GPU -> CPU if needed
        if self.use_gpu:
            self.working_image = self.working_image_gpu.detach().cpu().numpy()

        # post smoothing (tiny)
        self.working_image = gaussian_filter(self.working_image, sigma=0.5).astype(np.float32)

        print(f"Done. Nail sequence length: {len(self.nail_sequence)}")
        return self.nail_sequence

    # ----------------------------
    # Visualization / IO
    # ----------------------------
    def visualize(
        self,
        show_nails: bool = True,
        show_sequence: bool = True,
        figsize: Tuple[int, int] = (15, 5),
        show_radon: bool = False,
    ):
        num_plots = 4 if show_radon else 3
        fig, axes = plt.subplots(1, num_plots, figsize=(figsize[0] + (5 if show_radon else 0), figsize[1]))

        plot_idx = 0

        # Original (invert for display)
        original_display = 1.0 - self.target_image
        axes[plot_idx].imshow(original_display, cmap="gray")
        axes[plot_idx].set_title("Original Image")
        axes[plot_idx].axis("off")
        plot_idx += 1

        if self.use_gpu and self.working_image is None:
            self.working_image = self.working_image_gpu.detach().cpu().numpy()

        result = np.clip(self.working_image, 0, 1)
        result_display = 1.0 - result

        axes[plot_idx].imshow(result_display, cmap="gray")
        if show_nails:
            axes[plot_idx].plot(self.nail_positions[:, 0], self.nail_positions[:, 1], "ro", markersize=2, alpha=0.5)
        axes[plot_idx].set_title("String Art Result")
        axes[plot_idx].axis("off")
        plot_idx += 1

        if show_radon:
            axes[plot_idx].imshow(self.sinogram_normalized, aspect="auto", cmap="gray")
            axes[plot_idx].set_title("Radon (Sinogram)")
            axes[plot_idx].set_xlabel("Angle index")
            axes[plot_idx].set_ylabel("r index")
            plot_idx += 1

        if show_sequence and len(self.nail_sequence) > 1:
            axes[plot_idx].axis([0, self.image_size, 0, self.image_size])
            axes[plot_idx].set_aspect("equal")
            axes[plot_idx].invert_yaxis()

            axes[plot_idx].plot(self.nail_positions[:, 0], self.nail_positions[:, 1], "ko", markersize=3)

            max_lines = min(600, len(self.nail_sequence) - 1)
            for i in range(max_lines):
                a = self.nail_sequence[i]
                b = self.nail_sequence[i + 1]
                pa = self.nail_positions[a]
                pb = self.nail_positions[b]
                axes[plot_idx].plot([pa[0], pb[0]], [pa[1], pb[1]], "b-", alpha=0.25, linewidth=0.5)

            axes[plot_idx].set_title(f"String Path (first {max_lines} lines)")
            axes[plot_idx].axis("off")
        else:
            axes[plot_idx].imshow(result_display, cmap="gray")
            axes[plot_idx].set_title("String Art (Alt view)")
            axes[plot_idx].axis("off")

        plt.tight_layout()
        return fig

    def save_nail_sequence(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            for nail in self.nail_sequence:
                f.write(f"{int(nail)}\n")
        print(f"Saved nail sequence: {filename}")

    def load_nail_sequence(self, filename: str, intensity: float = 0.25):
        self.nail_sequence = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                self.nail_sequence.append(int(line.strip()))

        # Reset canvas
        if self.use_gpu:
            self.working_image_gpu.zero_()
        else:
            self.working_image.fill(0.0)

        for i in range(len(self.nail_sequence) - 1):
            self._draw_line(self.nail_sequence[i], self.nail_sequence[i + 1], intensity=float(intensity))

        if self.use_gpu:
            self.working_image = self.working_image_gpu.detach().cpu().numpy()

        self.working_image = gaussian_filter(self.working_image, sigma=0.5).astype(np.float32)
        print(f"Loaded nail sequence: {filename} ({len(self.nail_sequence)} nails)")


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Put your image path here
    image_path = "input.jpg"

    gen = StringArtGenerator(
        image_path=image_path,
        num_nails=300,
        image_size=500,
        use_gpu=True,
        seed=42,
    )

    gen.generate_string_art(
        num_iterations=3000,
        min_distance=10,
        max_distance=None,
        residual_refresh_every=25,
        candidates_early=260,
        candidates_late=160,
    )

    fig = gen.visualize(show_nails=True, show_sequence=True, show_radon=False)
    plt.show()

    gen.save_nail_sequence("nails.txt")
