// 文档 https://github.com/hooke007/mpv_PlayKit/wiki/4_GLSL

/*

LICENSE:
  --- Paper ver.
  https://johncostella.com/magic/

*/


//!HOOK MAIN
//!BIND HOOKED
//!DESC [Magic_Kernel_Sharp2013]
//!WIDTH OUTPUT.w
//!HEIGHT OUTPUT.h
//!WHEN OUTPUT.w HOOKED.w = ! OUTPUT.h HOOKED.h = ! +

const float KERNEL_RADIUS = 2.5;
const int UPSCALE_RADIUS = 3;

float magic_kernel_sharp_2013(float x) {
	x = abs(x);
	if (x <= 0.5) return 17.0/16.0 - (7.0/4.0) * x * x;
	if (x <= 1.5) return 0.25 * (4.0 * x * x - 11.0 * x + 7.0);
	if (x <= KERNEL_RADIUS) return -0.125 * (x - 2.5) * (x - 2.5);
	return 0.0;
}

vec4 hook() {

	vec2 src_size = HOOKED_size;
	vec2 dst_size = target_size;

	vec2 ratio = src_size / dst_size;
	vec2 scale = max(ratio, vec2(1.0));
	// Kernel radius (fixed for upscale, scaled for downscale)
	ivec2 radius = ivec2(mix(vec2(UPSCALE_RADIUS), ceil(KERNEL_RADIUS * scale), greaterThan(ratio, vec2(1.0))));

	vec2 src_pos = HOOKED_pos * src_size - 0.5;
	ivec2 src_base = ivec2(floor(src_pos));
	vec2 frac_pos = src_pos - vec2(src_base);

	vec4 sum_color = vec4(0.0);
	float wsum = 0.0;

	for (int ky = -radius.y; ky <= radius.y; ky++) {
		int sy = src_base.y + ky;
		if (sy < 0 || sy >= int(src_size.y)) continue;

		float dy_dist = abs(frac_pos.y - float(ky)) / scale.y;
		if (dy_dist >= KERNEL_RADIUS) continue;
		float wy = magic_kernel_sharp_2013(dy_dist);

		for (int kx = -radius.x; kx <= radius.x; kx++) {
			int sx = src_base.x + kx;
			if (sx < 0 || sx >= int(src_size.x)) continue;

			float dx_dist = abs(frac_pos.x - float(kx)) / scale.x;
			if (dx_dist >= KERNEL_RADIUS) continue;
			float wx = magic_kernel_sharp_2013(dx_dist);

			float w = wx * wy;
			vec4 sample_color = texelFetch(HOOKED_raw, ivec2(sx, sy), 0);
			sample_color = linearize(sample_color);
			sum_color += sample_color * w;
			wsum += w;
		}
	}

	if (wsum > 0.0) {
		sum_color /= wsum;
	}

	vec4 orig = HOOKED_texOff(0);
	sum_color = delinearize(sum_color);
	sum_color.a = orig.a;
	return sum_color;

}

