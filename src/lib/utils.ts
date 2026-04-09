// Some color / math code tweaked from functions found in this repo: https://github.com/mattdesl/tiny-artblocks

export type RGB = {
	r: number;
	g: number;
	b: number;
};

export type Oklab = {
	L: number;
	a: number;
	b: number;
};

export function get_median(arr: number[]) {
  let mid = Math.floor(arr.length / 2);
  let nums = [...arr].sort((a, b) => a - b);

  return arr.length % 2 !== 0 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
};

export function get_distance(color_a: Oklab, color_b: Oklab): number {
    let delta_L_sq = (color_a.L - color_b.L) * (color_a.L - color_b.L);
    let delta_a_sq = (color_a.a - color_b.a) * (color_a.a - color_b.a);
    let delta_b_sq = (color_a.b - color_b.b) * (color_a.b - color_b.b);

    return Math.sqrt(delta_L_sq + delta_a_sq + delta_b_sq);
}

export function clamp(value: number, min: number, max: number): number {
	return Math.max(Math.min(value, max), min);
}

// returns an object literal which is r, g and b integers from an RGB hex string:
export function hexToRGB(str: string): RGB {
	const hex = str.replace('#', '');
	// NOTE: This can be removed for brevity if you stick with 6-character codes:
	// if (hex.length === 3) {hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];}
	const num = Number.parseInt(hex, 16);

	return { r: num >> 16, g: (num >> 8) & 255, b: num & 255 };
}

// converts RGB integer values to hex:
export function rgbToHex({ r, g, b }: RGB): string {
	// NEXT LINE OPTIONAL, depending on your purposes:
	// const clampedR = clamp(r, 0, 255); const clampedG = clamp(g, 0, 255); const clampedB = clamp(b, 0, 255);
	return `#${(b | (g << 8) | (r << 16) | (1 << 24)).toString(16).slice(1)}`;
}

// correlary of first psuedocode block here (f_inv) : https://bottosson.github.io/posts/colorwrong/#what-can-we-do%3F ; "applying the inverse of the sRGB nonlinear transform function.." -- keeping the abbreviated syntax of arrow functions and ? : if/then, despite that they confuse and stretch my noob brain:
export const gammaToLinear = (c: number): number =>
	c >= 0.04045 ? Math.pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
// correlary of the first " : "..then switching back" :
export const linearToGamma = (c: number): number =>
	c >= 0.0031308 ? 1.055 * Math.pow(c, 1 / 2.4) - 0.055 : 12.92 * c;

// Lab coordinates (parameters):
// L = Lightness (0 (black) to ?? (diffuse white)
// a = green (at negative -- is there a lower bound?) to red (positive)
// b = blue (at negative) to yellow (at positive).
// You can manually construct an object literal to pass to this function this way:
// labVals = {L: 0.75, a: 0.7, b: 0.2};
// sRGBresultObjectLiteral = oklabToSRGB(labVals);
// You can also construct that as just {0.75, 0.7, 0.2}, and still pass it and it will work, apparently
// "..Oklab is represented as an object {L, a, b} where L is between 0 and 1 for normal SRGB colors. a and b have a less clearly defined range, but will normally be between -0.5 and +0.5. Neutral gray colors will have a and b at zero (or very close)." re: https://www.npmjs.com/package/oklab
// numbers updated from C++ example at https://bottosson.github.io/posts/oklab/ as updated 2021-01-25
// helpful references:
// https://observablehq.com/@sebastien/srgb-rgb-gamma
// Other references: https://matt77hias.github.io/blog/2018/07/01/linear-gamma-and-sRGB-color-spaces.html
// Takeaway: before manipulating color for sRGB (gamma-corrected or gamma compressed), convert it to linear RGB or a linear color space. Then do the manipulation, then convert it back to sRGB.
export function rgbToOklab({ r, g, b }: RGB): Oklab {
	// This is my undersanding: JavaScript canvas and many other virtual and literal devices use gamma-corrected (non-linear lightness) RGB, or sRGB. To convert sRGB values for manipulation in the Oklab color space, you must first convert them to linear RGB. Where Oklab interfaces with RGB it expects and returns linear RGB values. This next step converts (via a function) sRGB to linear RGB for Oklab to use:
	r = gammaToLinear(r / 255);
	g = gammaToLinear(g / 255);
	b = gammaToLinear(b / 255);
	// This is the Oklab math:
	let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
	let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
	let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
	// Math.crb (cube root) here is the equivalent of the C++ cbrtf function here: https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
	l = Math.cbrt(l);
	m = Math.cbrt(m);
	s = Math.cbrt(s);

	return {
		L: l * +0.2104542553 + m * +0.7936177850 + s * -0.0040720468,
		a: l * +1.9779984951 + m * -2.4285922050 + s * +0.4505937099,
		b: l * +0.0259040371 + m * +0.7827717662 + s * -0.8086757660
	};
}

export function oklabToSRGB({ L, a, b }: Oklab): RGB {
	let l = L + a * +0.3963377774 + b * +0.2158037573;
	let m = L + a * -0.1055613458 + b * -0.0638541728;
	let s = L + a * -0.0894841775 + b * -1.2914855480;
	// The ** operator here cubes; same as l_*l_*l_ in the C++ example:
	l = l ** 3;
	m = m ** 3;
	s = s ** 3;
	let r = l * +4.0767416621 + m * -3.3077115913 + s * +0.2309699292;
	let g = l * -1.2684380046 + m * +2.6097574011 + s * -0.3413193965;
	let bOut = l * -0.0041960863 + m * -0.7034186147 + s * +1.7076147010;
	// Convert linear RGB values returned from oklab math to sRGB for our use before returning them:
	r = 255 * linearToGamma(r);
	g = 255 * linearToGamma(g);
	bOut = 255 * linearToGamma(bOut);
	// OPTION: clamp r g and b values to the range 0-255; but if you use the values immediately to draw, JavaScript clamps them on use:
	r = clamp(r, 0, 255);
	g = clamp(g, 0, 255);
	bOut = clamp(bOut, 0, 255);
	// OPTION: round the values. May not be necessary if you use them immediately for rendering in JavaScript, as JavaScript (also) discards decimals on render:
	r = Math.round(r);
	g = Math.round(g);
	bOut = Math.round(bOut);

	return { r, g, b: bOut };
}

// A DEV TEST that determined min and max range for oklab L, a, and b as found via conversion from all possible RGB256 values (I want those ranges for the curiosity of generating random colors in Oklab space but converting the results to rgb):
// var counter = 0;
// var aMin = 0, aMax = 0, bMin = 0, bMax = 0, lMin = 50; lMax = -50;
// for (var r = 0; r < 256; r++) {
//   for (var g = 0; g < 256; g++) {
//     for (var b = 0; b < 256; b++) {
//       counter += 1;
//       var sRGBcolor = {r, g, b};
//       var labColor = rgbToOklab(sRGBcolor);
//       if (labColor.L < lMin) {lMin = labColor.L;}
//       if (labColor.L > lMax) {lMax = labColor.L;}
//       if (labColor.a < aMin) {aMin = labColor.a;}
//       if (labColor.a > aMax) {aMax = labColor.a;}
//       if (labColor.b < bMin) {bMin = labColor.b;}
//       if (labColor.b > bMax) {bMax = labColor.b;}
//     }
//   }
// }
// console.log("counter:", counter);
// console.log("lMin:", lMin, "lMax:", lMax, "aMin:", aMin, "aMax:", aMax, "bMin:", bMin, "bMax:", bMax);
// results without and with gammaToLinear -> linearToGamma and other "OPTIONAL" calculations:
// without extra calculations       with extra calculations
// lMin: 0                          0
// lMax: 254.99999832226274         0.9999999934735462
// aMin: -118.66066044645729        -0.23388757418790818
// aMax: 118.66066050838631         0.27621639742350523
// bMin: -108.13237868927439        -0.3115281476783751
// bMax: 108.1323881808137          0.19856975465179516

// POSSIBILITY FOR GRAYSCALE CALC; you can use the value returned from this with a = 0, b = 0 (for oklab) to convert RGB to gray:
// function linearrgbToOklabLightness({r, g, b}) {
//   const l = Math.cbrt(r * +0.4122214708 + g * +0.5363325363 + b * +0.0514459929);
//   const m = Math.cbrt(r * +0.2119034982 + g * +0.6806995451 + b * +0.1073969566);
//   const s = Math.cbrt(r * +0.0883024619 + g * +0.2817188376 + b * +0.6299787005);
//
//   return l * +0.2104542553 + m * +0.7936177850 + s * -0.0040720468;
// };

// returns an array which is a range of values at N intervals over range min-max (interpolations), inclusive of max:
export function getLerpRange(min: number, max: number, count: number): number[] {
	const dividend = (max - min) / (count - 1);
	const arr: number[] = [];
	let currentVal = min;

	for (let i = 0; i < count; i++) {
		arr.push(currentVal);
		currentVal += dividend;
	}

	return arr;
}

// returns an array of sRGB hex color codes, from interpolation in numberOfColors steps through Oklab space (and converted back to sRGB), including start and end value:
export function getRGBlerpRangeInOklabSpace(
	startRGBhexColor: string,
	endRGBhexColor: string,
	numberOfColors: number
): RGB[] {
	const interpolatedColors: RGB[] = [];
	const startLabVals = rgbToOklab(hexToRGB(startRGBhexColor));
	const endLabVals = rgbToOklab(hexToRGB(endRGBhexColor));
	const lValsArr = getLerpRange(startLabVals.L, endLabVals.L, numberOfColors);
	const aValsArr = getLerpRange(startLabVals.a, endLabVals.a, numberOfColors);
	const bValsArr = getLerpRange(startLabVals.b, endLabVals.b, numberOfColors);

	for (let i = 0; i < numberOfColors; i++) {
		// I could nest function calls here for brevity . . .
		const tmpOklabVals: Oklab = { L: lValsArr[i], a: aValsArr[i], b: bValsArr[i] };
		const tmpsRGBvals = oklabToSRGB(tmpOklabVals);
		// const hexVals = rgbToHex(tmpsRGBvals);
		interpolatedColors.push(tmpsRGBvals);
	}

	return interpolatedColors;
}