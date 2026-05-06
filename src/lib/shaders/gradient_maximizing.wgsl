/*
This shader culls the gaussian gradient texture to be just the maximums.

The purpose in TheVectorizer, is for this shader to be given the gaussian gradient
texture. For each pixel in the gradient texture, this shader looks at the neighboring
pixels along the gradient normal to see if that pixel is the maximum or not.

This basically just filters the texture to be only the edge pixels

TODO: update above description for when this shader actually does sub-pixel stuff
*/

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Fullscreen triangle
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );

    // UVs (can go outside 0..1; we clamp later)
    var uv = array<vec2f, 3>(
        vec2f(0.0, 1.0),
        vec2f(2.0, 1.0),
        vec2f(0.0, -1.0),
    );

    var out: VsOut;
    out.pos = vec4f(pos[vid], 0.0, 1.0);
    out.uv = uv[vid];
    return out;
}


/*
grad_tex: texture_2d<f32>
	x -> grad_mag (magnitude of gradient)
	y -> theta    (direction of gradient)
    z -> 0        (unused)
    w -> 0        (unused)

output:
    x -> edge flag
    y -> subpixel_offset (in the direction of the gradient)
    z -> 0               (unused)
    w -> 0               (unused)
*/
@group(0) @binding(0) var grad_tex: texture_2d<f32>;
@group(0) @binding(1) var grad_sampler: sampler;


// TODO: move this LOW value to an external variable passed through uniforms
const LOW: f32 = 0.1;


@fragment
fn cs_main(in: VsOut) -> @location(0) vec4f {
    // Clamp UV to avoid sampling outside (because our fullscreen triangle uses UV beyond 0..1)
    let uv = clamp(in.uv, vec2f(0.0), vec2f(1.0));

    let dims = textureDimensions(grad_tex);

    let texel = vec2u(
        min(u32(uv.x * f32(dims.x)), dims.x - 1u),
        min(u32(uv.y * f32(dims.y)), dims.y - 1u)
    );

    // TODO: might want to switch below to just use textureLoad
    // Sample with linear filtering to allow sub-pixel neighborhood checks.
    let grad_pixel = textureSampleLevel(grad_tex, grad_sampler, uv, 0.0);
    let grad_mag = grad_pixel.x;
    let theta = grad_pixel.y;

    // TODO: is there a way to cast the pixels as some struct, so that it's extra obvious that what the xyzw values are?

    // TODO: refine the node position to have the maximum not just be the center of the pixel (check neighbors to find actual maximum)

    if (grad_mag < LOW) {
        return vec4f(0, 0, 0, 0);
    }

    // --- Finding Edge Seed Pixels ---
    var greatest = true; // store whether neighbor of greater magnitude has been found (set to false if neighbor is found)
    // for each neighboring pixel
    for (var dx = -1; dx <= 1; dx ++) {
        for (var dy = -1; dy <= 1; dy ++) {
            // dont compare the pixel to itsself
            if (dx == 0 && dy == 0) { continue; }

            let neighbor_pix = textureLoad(grad_tex, clamp(vec2i(texel) + vec2i(dx, dy), vec2i(0), vec2i(dims) - vec2i(1)), 0);
            let neighbor_grad_mag = neighbor_pix.x;
            let neighbor_theta = neighbor_pix.y;

            // if the neighbor has a greater magnitude, then our pixel isn't the greatest
            if (grad_mag <= neighbor_grad_mag) {
                greatest = false;
            }
        }
    }

    // TODO: might be worth it to move subpixel shifting to the edge tracing steps to improve performance
    let subpixel_offset = get_subpixel_offset(grad_pixel, texel, dims);
    // let subpixel_offset = 0f;

    // TODO: rename greatest to prevent confusion with greatest self vs greatest neighbor
    // --- Marking Edge Seeds ---
    if (greatest) {
        // mark this pixel as part of an edge
        return vec4f(1, subpixel_offset, 0, 0);
    }

    // these pixels might later be part of an edge, so their magnitude and theta need to be stored
    return vec4f(0, subpixel_offset, 0, 0);
}

fn get_subpixel_offset(grad_pixel: vec4f, texel: vec2u, dims: vec2u) -> f32 {
	let neighbot_checks: u32 = 2; // number of neighbor checks in each direction (so value of 2 would mean checking 4 neighbors + 1 for the pixel itsself)
	let neighbor_check_distance: f32 = 1.5;
    let theta = grad_pixel.y;


    // ?NOTE: this commented out section is an older (maybe worse idk) method of calculating the subpixel offset
    // // pick the neighbor with the greatest gradient magnitude
    // var greatest_neighbor_pix = vec4f(0); // temporary, should be overwritten
    // var greatest_neighbor_flip = 0;        // temporary, should be overwritten
    // for (var flip = -1; flip <= 1; flip += 2) {
    //     let neighbor_offset = f32(flip) * vec2f(cos(theta), sin(theta)) * neighbor_check_distance;
    //     let neighbor_pos_uv = (vec2f(texel) + neighbor_offset) / vec2f(dims);

    //     let neighbor_pix = textureSampleLevel(grad_tex, grad_sampler, neighbor_pos_uv, 0.0);

    //     if (neighbor_pix.x > greatest_neighbor_pix.x) {
    //         greatest_neighbor_pix = neighbor_pix;
    //         greatest_neighbor_flip = flip;
    //     }
    // }

    // // calculating subpixel_offset based off of greatest gradient neighbor
    // let subpixel_offset = (greatest_neighbor_pix.x * f32(greatest_neighbor_flip) * neighbor_check_distance) / (grad_pixel.x + greatest_neighbor_pix.x);

    var weighted_offset_sum = 0f; // sum of pixel offsets in direction of gradient weighted by that pixel's gradient magnitude
    var weights_sum = 0f;         // sum of the gradient magnitudes (for normalizing)

    for (var mult: i32 = -1; mult <= 1; mult += 1) {
    	let offset_magnitude = f32(mult) * neighbor_check_distance;
        let neighbor_offset = vec2f(cos(theta), sin(theta)) * offset_magnitude;
        let neighbor_pos_uv = (vec2f(texel) + neighbor_offset) / vec2f(dims);

        let neighbor_pix = textureSampleLevel(grad_tex, grad_sampler, neighbor_pos_uv, 0.0);
        let neighbor_grad_mag = neighbor_pix.x;
        let neighbor_theta = neighbor_pix.y;

        // TODO: figure out how to weight it so that neighbors who's gradient alligns better are weighted higher than missaligned neighbors
        let weight = neighbor_grad_mag;

        weighted_offset_sum += offset_magnitude * weight;
        weights_sum += weight;
    }

    let subpixel_offset = weighted_offset_sum / weights_sum;

    // return 0;
    return subpixel_offset;
}
