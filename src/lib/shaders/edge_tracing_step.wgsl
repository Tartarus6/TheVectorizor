/*
This shader culls the gaussian gradient texture to be just the maximums.

The purpose in TheVectorizer, is for this shader to be given the gaussian gradient
texture. For each pixel in the gradient texture, this shader looks at the neighboring
pixels along the gradient normal to see if that pixel is the maximum or not.

This basically just filters the texture to be only the edge pixels

TODO: update above description to actually explain what subpixel stuff it does and how
*/

/*
grad_tex: texture_2d<f32>
	x -> grad_mag (magnitude of gradient)
	y -> theta    (direction of gradient)
    z -> 0        (unused)
    w -> 0        (unused)

in_edge_tex/out_edge_tex:
    x -> edge flag        (whether this pixel is part of an edge)
    y -> subpixel_offset  (in the direction of the gradient)
    z -> packed neighbors (0..63 value that indicates the 2 connected neighbor edges. note: value of 0 is not possible, so its safe to assume a value of 0 means it's unset)
    w -> power            (number of edge connections to pixel)
*/
@group(0) @binding(0) var grad_tex: texture_2d<f32>;
@group(0) @binding(1) var in_edge_tex: texture_2d<f32>;
@group(0) @binding(2) var out_edge_tex: texture_storage_2d<rgba16float, write>;


const PI = 3.1415926535;
const CANDIDATES_SIZE: u32 = 3u;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(grad_tex); // both textures have the same dimensions, so its arbitrary which is used

    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let texel = gid.xy;
    let uv = (vec2f(texel) + vec2f(0.5, 0.5)) / vec2f(dims);

    // loading data from input textures
    let grad_pix = textureLoad(grad_tex, clamp(vec2i(texel), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let in_edge_pix = textureLoad(in_edge_tex, clamp(vec2i(texel), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let edge_flag = in_edge_pix.x;
    let packed_connections = in_edge_pix.z;
    let grad_mag = grad_pix.x;
    let theta = grad_pix.y;

    // if pixel already has connections, then skip it
    // note: a packed connections value of 0 is not possible since that'd mean both chosen candidates are the same. so that can be used to ignore
    if (packed_connections != 0) {
    	return;
    }

    // TODO: might be able to save these texture stores with some initial copying or something
    // textureStore(out_edge_tex, texel, in_pixel);

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
        dir,
        dir + perp,
        dir - perp,
    );

    // (dx A, dy A, dx B, dy B)
    var candidate_offsets = vec4i(0); // offsets to both selected candidates

    // check for followup edge pixel along theta forwards and also backwards
    for (var flip_mult = -1; flip_mult <= 1; flip_mult += 2) {
    	var best_edge_pix = vec4f(-1); // default, should be overwritten
        var best_pix_grad_mag = -1f;   // default, should be overwritten
        var best_pos = vec2u(0);       // default, should be overwritten

        var found_edge_connection = false; // whether the chosen candidate was already marked as an edge

        for (var i = 0u; i < CANDIDATES_SIZE; i = i + 1u) {
            // ?NOTE: the error below is untrue, CANDIDATES_SIZE does properly work to define the array
            let pos = vec2u(clamp(vec2i(texel) + (flip_mult * candidates[i]), vec2i(0), vec2i(dims) - vec2i(1)));
            let cand_edge_pix = textureLoad(in_edge_tex, pos, 0);
            let cand_grad_mag = textureLoad(grad_tex, pos, 0).x;

            let cand_edge_flag = cand_edge_pix.x;

            // if candidate is already part of an edge, just pick that one immediately
            if (cand_edge_flag != 0) {
                best_edge_pix = cand_edge_pix;
                best_pix_grad_mag = cand_grad_mag;
                best_pos = pos;
                found_edge_connection = true; // mark that chosen candidate was already marked as an edge
                break;
            }

            // if candidate is new best, update best
            if (cand_grad_mag > best_pix_grad_mag) {
                best_edge_pix = cand_edge_pix;
               	best_pix_grad_mag = cand_grad_mag;
                best_pos = pos;
            }
        }

        // if chosen candidate wasn't already marked as an edge
        if !found_edge_connection {
	        // write best candidate as an edge
	        textureStore(out_edge_tex, best_pos, vec4f(1, best_edge_pix.yzw));
		}

		// TODO: probably should rewrite this to something more robust
		let candidate_offset: vec2i = vec2i(best_pos) - vec2i(texel);
		if (flip_mult < 0) {
			candidate_offsets.x = candidate_offset.x;
			candidate_offsets.y = candidate_offset.y;
		} else {
			candidate_offsets.z = candidate_offset.x;
			candidate_offsets.w = candidate_offset.y;
		}
    }

    // pack candidate offsets and save it into current pixel
    let packed: f32 = pack_neighbors(candidate_offsets);
    textureStore(out_edge_tex, texel, vec4f(in_edge_pix.xy, packed, in_edge_pix.w));
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

// Used for packing and unpacking neighbor offsets
// Direction order: E, NE, N, NW, W, SW, S, SE
const DIRS: array<vec2i, 8> = array<vec2i, 8>(
    vec2i(1, 0),   // 0: E
    vec2i(1, -1),  // 1: NE
    vec2i(0, -1),  // 2: N
    vec2i(-1, -1), // 3: NW
    vec2i(-1, 0),  // 4: W
    vec2i(-1, 1),  // 5: SW
    vec2i(0, 1),   // 6: S
    vec2i(1, 1)    // 7: SE
);

/*
Packs the position deltas of 2 neighbors into one float for storing in the texture

Note: the order of the 2 neighbors does not matter.

Input: (dx A, dy A, dx B, dy B)
Example:
	□ · ·
	· ■ ·   →   Input is (-1, -1, 1, 1)   →   Output is
	· · □

	· □ ·
	· ■ □   →   Input is (0, -1, 1, 0)   →   Output is
	· · ·
*/
// TODO: would be better if this could be switched to returning an unsigned int, would have to change the texture setup though
fn pack_neighbors(neighbors: vec4i) -> f32 {
	// get individual (dx, dy) offsets
	let offset_a: vec2i = vec2i(neighbors.xy);
	let offset_b: vec2i = vec2i(neighbors.zw);

	// turn (dx, dy) offsets into direction indeces (0..7)
	let dir_a: u32 = offset_to_direction(offset_a);
	let dir_b: u32 = offset_to_direction(offset_b);

	// pack the direction indeces together (0..7) → (0..63)
	let packed = dir_a + 8u * dir_b;

	return f32(packed);
}


fn unpack_neighbors(packed_neighbors: f32) -> vec4i {
	// convert to integer
	let packed = u32(packed_neighbors);

	// unpack 0..63 into two 0..7 direction indeces
	let dir_a: u32 = packed % 8u;
	let dir_b: u32 = packed / 8u;

	// convert direction indeces (0..7) into offsets
	let offset_a: vec2i = DIRS[dir_a];
	let offset_b: vec2i = DIRS[dir_b];

	return vec4i(offset_a, offset_b);
}

/*
Takes in an 8-connected offset and turns it into an 8-connected direction from 0 to 7

Input: (dx, dy)
	-1 <= dx, dy <= 1
	dx and dy can't both be 0
Output:
	Output follows this pattern

	3 2 1
	4 ■ 0
	5 6 7

Example:
	□ · ·
	· ■ ·   →   Input is (-1, -1)   →   Output is 3
	· · ·

	· · ·
	· ■ □   →   Input is (1, 0)   →   Output is 0
	· · ·
*/
fn offset_to_direction(offset: vec2i) -> u32{
	// TODO: validate input maybe?

    for (var i: u32 = 0; i < 8; i = i + 1) {
        if (all(offset == DIRS[i])) {
            return i;
        }
    }
    return 0u; // invalid inputs just default to 0
}
