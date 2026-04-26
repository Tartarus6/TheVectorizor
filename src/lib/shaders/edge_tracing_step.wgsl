/*
This shader culls the gaussian gradient texture to be just the maximums.

The purpose in TheVectorizer, is for this shader to be given the gaussian gradient
texture. For each pixel in the gradient texture, this shader looks at the neighboring
pixels along the gradient normal to see if that pixel is the maximum or not.

This basically just filters the texture to be only the edge pixels

TODO: update above description for when this shader actually does sub-pixel stuff
*/

/*
input_tex and output_tex:
    x -> edge flag
    y -> grad_mag
    z -> theta
    w -> subpixel offset (in the direction of the gradient)
*/
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var grad_sampler: sampler;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba16float, write>;


const PI = 3.1415926535;
const CANDIDATES_SIZE: u32 = 3u;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(input_tex);

    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let texel = gid.xy;
    let uv = (vec2f(texel) + vec2f(0.5, 0.5)) / vec2f(dims);

    var out_pixel = textureSampleLevel(input_tex, grad_sampler, uv, 0.0);
    let in_pixel = textureLoad(input_tex, clamp(vec2i(texel), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let edge_flag = in_pixel.x;
    let grad_mag = in_pixel.y;
    let theta = in_pixel.z;
    let subpixel_offset = in_pixel.w;

    // TODO: might be able to save these texture stores with some initial copying or something
    textureStore(output_tex, texel, in_pixel);

    if (edge_flag == 0f) {
        return;
    }

    // let section = u32((theta + (PI / 4.0) + (PI / 2.0)) / (2 * PI) * 4.0) % 4;
    let section = get_section(theta, 4);

    let dirs = array<vec2i, 4>(
        vec2i(1, 0),   // right
        vec2i(0, -1),  // up
        vec2i(-1, 0),  // left
        vec2i(0, 1)    // down
    );

    let dir = dirs[section];
    let perp = vec2i(-dir.y, dir.x);

    // ?NOTE: the error below is untrue, CANDIDATES_SIZE does properly work to define the array
    var candidates = array<vec2i, CANDIDATES_SIZE>(
        dir + perp,
        dir,
        dir - perp,
    );

    // check for followup edge pixel along theta forwards and also backwards
    for (var flip_mult = -1; flip_mult <= 1; flip_mult += 2) {
        var best_pix = vec4f(-1);  // default, should be overwritten
        var best_pos = vec2u(0);   // default, should be overwritten

        for (var i = 0u; i < CANDIDATES_SIZE; i = i + 1u) {
            // ?NOTE: the error below is untrue, CANDIDATES_SIZE does properly work to define the array
            let pos = vec2u(clamp(vec2i(texel) + (flip_mult * candidates[i]), vec2i(0), vec2i(dims) - vec2i(1)));
            let cand_pix = textureLoad(input_tex, pos, 0);
            
            // if candidate is already part of an edge, just pick that one immediately
            if (cand_pix.x != 0) {
                best_pix = cand_pix;
                best_pos = pos;
                break;
            }

            // if candidate is new best, update best
            if (cand_pix.y > best_pix.y) {
                best_pix = cand_pix;
                best_pos = pos;
            }
        }

        // write best candidate as an edge
        textureStore(output_tex, best_pos, vec4f(1, best_pix.yz, 1));
    }
}

fn get_section(theta: f32, section_count: u32) -> u32 {
    // change gradient direction into edge tangent direction
    let edge_direction = theta + (PI / 2f);

    // offset edge tangent direction so that each section will be centered around their primary direction
    // e.g. section 0 with `section_count` of 4 includes (-π/4, π/4)
    let offset_direction = edge_direction + (PI / f32(section_count));

    // apply a scaling factor so that rounding produces `section_count` sections each rotation
    let float_section = offset_direction * (f32(section_count) / (2f * PI));

    // TODO: prevent possible overflow if theta is negative (theta shouldnt ever be negative, but it'd be smart to fix it here)
    // truncate `float_section` to get integer, and modulus to keep within range of [0, `section_count`]
    let section = u32(float_section) % section_count;

    return section;
}
