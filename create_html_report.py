#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create an HTML report for easy visual comparison of v2 model results.
"""

import os
import glob

output_dir = "outputs/v2_comparison"

html_content = """<!DOCTYPE html>
<html>
<head>
    <title>v2 Model Comparison: NoisyLR vs Restored vs Ground Truth</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .summary { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .grid-container { display: flex; flex-wrap: wrap; gap: 20px; }
        .sample { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 600px; }
        .sample h3 { margin-top: 0; color: #444; }
        .grid-img { width: 100%; max-width: 800px; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        .metrics { display: flex; gap: 20px; margin: 10px 0; }
        .metric { background: #e8f4fd; padding: 8px 16px; border-radius: 4px; font-family: monospace; }
        .legend { display: flex; gap: 30px; margin: 10px 0; font-size: 14px; color: #666; }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .color-box { width: 20px; height: 20px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>v2 Model Comparison: Chained Denoise_v2 + SR_v2 with D4 TTA</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Models:</strong> denoise_v2_best_ema.pth (DenoiseUNet, n_feats=48, unet_blocks=5) + sr_v2_best.pth (EDSR, n_feats=48, n_blocks=16)</p>
        <p><strong>Pipeline:</strong> NoisyLR (128x128) -> Denoiser -> SR (256x256) with D4 test-time augmentation on each stage</p>
        <p><strong>Validation Split:</strong> 10% (seed=42), 320 images total</p>
    </div>

    <div class="legend">
        <div class="legend-item"><div class="color-box" style="background: linear-gradient(to right, #000, #fff);"></div> NoisyLR (upscaled 2x, range [-0.3, 2.0])</div>
        <div class="legend-item"><div class="color-box" style="background: linear-gradient(to right, #000, #fff);"></div> Restored (model output, clamped [0,1])</div>
        <div class="legend-item"><div class="color-box" style="background: linear-gradient(to right, #000, #fff);"></div> Ground Truth [0,1]</div>
        <div class="legend-item"><div class="color-box" style="background: linear-gradient(to right, #000, #f00);"></div> |Restored - GT| (0 to 0.5)</div>
    </div>

    <div class="grid-container">
"""

sample_data = [
    ("002384", 33.09, 0.8911),
    ("002538", 22.73, 0.2447),
    ("002176", 30.08, 0.9074),
    ("000897", 30.70, 0.9132),
    ("000214", 37.03, 0.9397),
    ("002380", 30.00, 0.8434),
    ("002714", 20.99, 0.2588),
    ("002576", 23.27, 0.6105),
    ("000102", 32.36, 0.8019),
    ("000192", 28.12, 0.9078),
]

for name, psnr, ssim in sample_data:
    grid_path = f"{name}_grid.png"
    html_content += f"""
        <div class="sample">
            <h3>Sample {name}</h3>
            <div class="metrics">
                <div class="metric">PSNR: {psnr:.2f} dB</div>
                <div class="metric">SSIM: {ssim:.4f}</div>
            </div>
            <img class="grid-img" src="{grid_path}" alt="Comparison grid for {name}">
        </div>
"""

html_content += """
    </div>

    <div class="summary" style="margin-top: 30px;">
        <h2>Grid Layout</h2>
        <p>Each grid shows four panels:</p>
        <ol>
            <li><strong>Top-Left:</strong> NoisyLR input (upscaled 2x with nearest neighbor for visual comparison)</li>
            <li><strong>Top-Right:</strong> Restored output from chained v2 model (DenoiseUNet -> EDSR with D4 TTA)</li>
            <li><strong>Bottom-Left:</strong> Ground Truth (clean, 256x256)</li>
            <li><strong>Bottom-Right:</strong> Absolute difference map |Restored - GT| (red = higher error)</li>
        </ol>
        <p>Mean PSNR: 28.84 dB | Mean SSIM: 0.7319 | Range: 20.99 - 37.03 dB</p>
    </div>
</body>
</html>
"""

with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"HTML report created: {output_dir}/report.html")
print(f"Open in browser: file://{os.path.abspath(output_dir)}/report.html")