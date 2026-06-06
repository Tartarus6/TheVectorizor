/*
This shader runs after all of the edges have been traced.

This shader finds one-way connections between pixels and fixes them by making them two-way.

A pixel at some kind of intersection may have more than 2 neighbors, but the edge_tracing only
handles 2 neighbors per pixel. So this pass fixes that by finding unreciprocated neighbor connections
and adding a connection to the "hub" pixel that's in the intersection.
*/

/*
edge_tex (rgba16uint):
	x → edge flag        (whether this pixel is part of an edge)
	y → packed neighbors (bitmask to say which of the 8 neighbor pixels are connected edge pixels)
	z → 0                (unused)
	w → 0                (unused)
*/
@group(0) @binding(0) var edge_tex: texture_storage_2d<rgba16uint, read>;
@group(0) @binding(1) var edge_out: texture_storage_2d<rgba16uint, write>;

const MAX_NEIGHBORS: u32 = 8u; // maximum packed neighbors (one per 8-connected direction)

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(edge_tex);

    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let texel = gid.xy;


    // loading data from input textures
    let in_edge_pix = textureLoad(edge_tex, clamp(vec2i(texel), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)));
    let edge_flag = in_edge_pix.x;
    let packed_connections = in_edge_pix.y;

    var updated_packed: u32 = packed_connections; // variable to store new packed conenctions

    for (var dx: i32 = -1; dx <= 1; dx++) { // for dx
        for (var dy: i32 = -1; dy <= 1; dy++) { // for dy
        	// skip self pixel
            if (dx == 0 && dy == 0) {
                continue;
            }

            let offset = vec2i(dx, dy);
            let pos: vec2i = vec2i(texel) + offset;

            // ignore pixels outside the texture
            if (pos.x < 0 || pos.y < 0 || pos.x >= i32(dims.x) || pos.y >= i32(dims.y)) {
                continue;
            }

            let neighbor_pix = textureLoad(edge_tex, pos);

            let neighbor_neighbors = unpack_neighbors(neighbor_pix.y);
            var neighbor_points_to_this = false;

            // for this neighbor, look at each of its neighbors to see if self is one of them
            for (var i: u32 = 0u; i < neighbor_neighbors.count; i = i + 1u) {
                if (all(neighbor_neighbors.neighbors[i] * -1 == offset)) {
                    neighbor_points_to_this = true;
                    break;
                }
            }

            // if neighbor points to this pixel, add it to this pixel's neighbors
            if (neighbor_points_to_this) {
                updated_packed = updated_packed | (1u << offset_to_direction(offset));
            }
        }
    }

    textureStore(edge_out, texel, vec4u(in_edge_pix.x, updated_packed, in_edge_pix.zw));
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
