#include <stdio.h>
#include <math.h>
#include <array>

float factorial(int n) {
	return n == 0 ? 1 : (n * factorial(n - 1));
}

#define NINTERP 5
#define XMAX 3.0

struct interp_coeff {
	float a, b, c, d, e, f;
	float operator()(float x) {
		return a + x * (b + x * (c + x * (d + x * (e + x * f))));
	}
};
/*
 float func(float x) {
 return erfc(x) + 2.0 / sqrt(M_PI) * x * exp(-x * x);
 }
 float dfunc(float x) {
 return -4.0 * x * x * exp(-x * x) / sqrt(M_PI);
 }

 float d2func(float x) {
 return 8.0 * x * exp(-x * x) * (x * x - 1.0) / sqrt(M_PI);
 }
 */

float func(float x) {
	return erfc(x);
}
float dfunc(float x) {
	return -2.0 * exp(-x * x) / sqrt(M_PI);
}

float d2func(float x) {
	return 4.0 * x * exp(-x * x) / sqrt(M_PI);
}

int main() {
	std::array<interp_coeff, NINTERP> co;
	for (int i = 0; i < NINTERP; i++) {
		float x1 = (float) (i) / NINTERP * XMAX;
		float x2 = (float) (i + 1) / NINTERP * XMAX;
		float dx = x2 - x1;
		float y1 = func(x1);
		float y2 = func(x2);
		float k1 = dfunc(x1) * dx;
		float k2 = dfunc(x2) * dx;
		float q1 = d2func(x1) * dx * dx;
		float q2 = d2func(x2) * dx * dx;
		co[i].a = y1;
		co[i].b = k1;
		co[i].c = q1 / 2.;
		co[i].d = -6 * k1 - 4 * k2 - (3 * q1) / 2. + q2 / 2. - 10 * y1 + 10 * y2;
		co[i].e = 8 * k1 + 7 * k2 + (3 * q1) / 2. - q2 + 15 * y1 - 15 * y2;
		co[i].f = -3 * k1 - 3 * k2 - q1 / 2. + q2 / 2. - 6 * y1 + 6 * y2;
	}
	const float dr = XMAX / NINTERP;
	float max_err = 0.0;
	float max_abs_err = 0.0;
	for (float r = 0; r < 2.5; r += .01) {
		int i = r / dr;
		float x = r / dr - i;
		float y = co[i](x);
		float ya = func(r);
		float err = std::abs(y - ya) / ya;
		float err_abs = std::abs(y - ya);
		max_err = std::max(max_err, err);
		max_abs_err = std::max(max_abs_err, err_abs);
		//	printf("%e %e %e %e\n", r, y, ya, err_abs);
	}
	fprintf(stderr, "%e %e\n", max_abs_err, max_err);
	printf("__device__ float function(float x) {\n");
	printf("\tconstexpr float dxinv = %e;\n", (double) NINTERP / XMAX);
	printf("\tconstexpr float a[] = {");
	for (int i = 0; i < NINTERP; i++) {
		printf("%.8e", co[i].a);
		if (i != NINTERP - 1) {
			printf(",");
		}
	}
	printf("};\n");
	printf("\tconstexpr float b[] = {");
	for (int i = 0; i < NINTERP; i++) {
		printf("%.8e", co[i].b);
		if (i != NINTERP - 1) {
			printf(",");
		}
	}
	printf("};\n");
	printf("\tconstexpr float c[] = {");
	for (int i = 0; i < NINTERP; i++) {
		printf("%.8e", co[i].c);
		if (i != NINTERP - 1) {
			printf(",");
		}
	}
	printf("};\n");
	printf("\tconstexpr float d[] = {");
	for (int i = 0; i < NINTERP; i++) {
		printf("%.8e", co[i].d);
		if (i != NINTERP - 1) {
			printf(",");
		}
	}
	printf("};\n");
	printf("\tconstexpr float e[] = {");
	for (int i = 0; i < NINTERP; i++) {
		printf("%.8e", co[i].e);
		if (i != NINTERP - 1) {
			printf(",");
		}
	}
	printf("};\n");
	printf("\tconstexpr float f[] = {");
	for (int i = 0; i < NINTERP; i++) {
		printf("%.8e", co[i].f);
		if (i != NINTERP - 1) {
			printf(",");
		}
	}
	printf("};\n");
	printf("\tx *= dxinv;\n");
	printf("\tconst int i = int(x);\n");
	printf("\tx -= float(i);\n");
	printf("\tfloat y = f[i];\n");
	printf("\ty = fmaf(x, y, e[i]);\n");
	printf("\ty = fmaf(x, y, d[i]);\n");
	printf("\ty = fmaf(x, y, c[i]);\n");
	printf("\ty = fmaf(x, y, b[i]);\n");
	printf("\ty = fmaf(x, y, a[i]);\n");
	printf("\treturn y;\n");
	printf("}\n");

}
