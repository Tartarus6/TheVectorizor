/*
This shader culls the gaussian gradient texture to be just the maximums.

The purpose in TheVectorizer, is for this shader to be given the gaussian gradient
texture. For each pixel in the gradient texture, this shader looks at the neighboring
pixels along the gradient normal to see if that pixel is the maximum or not.

This basically just filters the texture to be only the edge pixels
*/

/*
grad_tex: texture_2d<f32>
	x → grad_mag        (magnitude of gradient)
	y → theta           (direction of gradient)
	z → subpixel_offset (in the direction of the gradient)
	w → 0               (unused)

in_edge_tex/out_edge_tex (rgba16uint):
	x → edge flag        (whether this pixel is part of an edge)
	y → packed neighbors (bitmask to say which of the 8 neighbor pixels are connected edge pixels)
	z → 0                (unused)
	w → 0                (unused)
*/
@group(0) @binding(0) var grad_tex: texture_2d<f32>;
@group(0) @binding(1) var in_edge_tex: texture_2d<u32>;
@group(0) @binding(2) var out_edge_tex: texture_storage_2d<rgba16uint, write>;


const PI = 3.1415926535;
const CANDIDATES_SIZE: u32 = 3u;
const MAX_NEIGHBORS: u32 = 8u; // maximum packed neighbors (one per 8-connected direction)

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
    let packed_connections = in_edge_pix.y;
    let grad_mag = grad_pix.x;
    let theta = grad_pix.y;

    // if pixel already has connections, then skip it
    // note: a packed connections value of 0 is not possible since that'd mean both chosen candidates are the same. so that can be used to ignore
    if (packed_connections != 0u) {
    	return;
    }

    // TODO: might be able to save these texture stores with some initial copying or something
    // textureStore(out_edge_tex, texel, in_pixel);

    if (edge_flag == 0u) {
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

    var neighbor_offsets = array<vec2i, MAX_NEIGHBORS>(
        vec2i(0), vec2i(0), vec2i(0), vec2i(0),
        vec2i(0), vec2i(0), vec2i(0), vec2i(0)
    );
    var neighbor_count: u32 = 0u; // number of selected neighbors

    // check for followup edge pixel along theta forwards and also backwards
    for (var flip_mult = -1; flip_mult <= 1; flip_mult += 2) {
    	var best_edge_pix = vec4u(0u); // default, should be overwritten
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
            if (cand_edge_flag != 0u) {
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
	        textureStore(out_edge_tex, best_pos, vec4u(1u, best_edge_pix.yzw));
		}

		// TODO: probably should rewrite this to something more robust
		let candidate_offset: vec2i = vec2i(best_pos) - vec2i(texel);
		if (neighbor_count < MAX_NEIGHBORS) {
			neighbor_offsets[neighbor_count] = candidate_offset;
			neighbor_count = neighbor_count + 1u;
		}
    }

    // pack candidate offsets and save it into current pixel
    let packed: u32 = pack_neighbors(neighbor_offsets, neighbor_count);
    textureStore(out_edge_tex, texel, vec4u(in_edge_pix.x, packed, in_edge_pix.zw));
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

// TODO: remove this code duplication if possible. maybe there's a way to shader this const and functions between shaders, idk.
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
Function that packs the position deltas of two neighbors into one uint for storing in the texture

Note: the order of the neighbors does not matter.

Input: (dx A, dy A, dx B, dy B)
Example:
	□ · ·
	· ■ ·   →   Input is [(-1, -1), (1, 1)]   →   Output is 2³ + 2⁷   →   10001000 (binary)   →   136 (decimal)
	· · □

	· □ ·
	· ■ □   →   Input is [(0, -1), (1, 0), (-1, 1)]   →   Output is 2² + 2⁰ + 2⁵   →   00100101 (binary)   →   37 (decimal)
	□ · ·

*/
fn pack_neighbors(neighbors: array<vec2i, MAX_NEIGHBORS>, neighbor_count: u32) -> u32 {
    // pack up to MAX_NEIGHBORS offsets into a direction bitmask
    var packed: u32 = 0u;
    let count = min(neighbor_count, MAX_NEIGHBORS);

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let dir: u32 = offset_to_direction(neighbors[i]);
        packed = packed | (1u << dir);
    }

    return packed;
}

struct UnpackedNeighbors {
    neighbors: array<vec2i, MAX_NEIGHBORS>,
    count: u32,
};

fn unpack_neighbors(packed_neighbors: u32) -> UnpackedNeighbors {
    var result = UnpackedNeighbors(
        array<vec2i, MAX_NEIGHBORS>(
            vec2i(0), vec2i(0), vec2i(0), vec2i(0),
            vec2i(0), vec2i(0), vec2i(0), vec2i(0)
        ),
        0u
    );

    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        if ((packed_neighbors & (1u << i)) != 0u) {
            if (result.count < MAX_NEIGHBORS) {
                result.neighbors[result.count] = DIRS[i];
                result.count = result.count + 1u;
            }
        }
    }

    return result;
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
