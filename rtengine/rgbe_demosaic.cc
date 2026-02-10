////////////////////////////////////////////////////////////////
//
//  RGBE bilinear demosaic for Sony DSC-F828 4-color sensors
//
//  Ported from demosaic/src/demosaic.rs bilinear() to RawTherapee conventions.
//  The F828 uses a 4-color RGBE (Red, Green, Blue, Emerald) CFA instead of
//  standard Bayer RGGB. This performs bilinear interpolation over a 3x3
//  neighborhood, grouping by CFA channel, then applies a 3x4 color matrix
//  to convert the 4 interpolated channels to RGB output.
//
//  Copyright (c) 2025
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
////////////////////////////////////////////////////////////////

#include "rawimagesource.h"
#include "rt_math.h"

using namespace rtengine;

void RawImageSource::rgbe_demosaic()
{
    // Simple 4->3 channel mapping: Emerald contributes equally to G and B.
    // RawTherapee's color pipeline (camera profile / cam_xyz) handles the
    // proper camera-space to working-space conversion afterwards.
    // Layout: cam_rgb[output_rgb][input_rgbe].
    constexpr float cam_rgb[3][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},  // R_out = R
        {0.0f, 0.5f, 0.0f, 0.5f},  // G_out = 0.5*G + 0.5*E
        {0.0f, 0.0f, 0.5f, 0.5f}   // B_out = 0.5*B + 0.5*E
    };

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            // Clamp neighborhood bounds at image edges.
            const int r_top = std::max(row - 1, 0);
            const int r_bot = std::min(row + 1, H - 1);
            const int c_left = std::max(col - 1, 0);
            const int c_right = std::min(col + 1, W - 1);

            // Accumulate sums and counts for each of the 4 CFA channels.
            float sums[4] = {0.f, 0.f, 0.f, 0.f};
            int counts[4] = {0, 0, 0, 0};

            for (int r = r_top; r <= r_bot; ++r) {
                for (int c = c_left; c <= c_right; ++c) {
                    const unsigned ch = FC(r, c);
                    sums[ch] += rawData[r][c];
                    counts[ch]++;
                }
            }

            // Average each channel, then apply the color matrix.
            float avg[4];
            for (int ch = 0; ch < 4; ++ch) {
                avg[ch] = counts[ch] > 0 ? sums[ch] / counts[ch] : 0.f;
            }

            float r_out = 0.f, g_out = 0.f, b_out = 0.f;
            for (int ch = 0; ch < 4; ++ch) {
                r_out += avg[ch] * cam_rgb[0][ch];
                g_out += avg[ch] * cam_rgb[1][ch];
                b_out += avg[ch] * cam_rgb[2][ch];
            }

            red[row][col]   = std::max(r_out, 0.f);
            green[row][col] = std::max(g_out, 0.f);
            blue[row][col]  = std::max(b_out, 0.f);
        }
    }
}
