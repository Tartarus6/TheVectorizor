/*
This shader runs after all of the edges have been traced.

This shader counts how many edges point to it, which I'm calling that edge pixel's "power".

This number is used later to identify "hub" pixels in order to prevent gaps in the corners between shapes.
Read the big comment block in shaders.ts for a better idea on exactly how this number will be used and why.
*/

/*
edge_tex (rgba16uint):
    x → edge flag        (whether this pixel is part of an edge)
    y → 0                (unused)
    z → packed neighbors (0..63 value that indicates the 2 connected neighbor edges. note: value of 0 is not possible, so its safe to assume a value of 0 means it's unset)
    w → power            (number of edge connections to pixel)
*/
@group(0) @binding(0) var edge_tex: texture_storage_2d<rgba16uint, read>;
@group(0) @binding(1) var edge_out: texture_storage_2d<rgba16uint, write>;

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
    let packed_connections = in_edge_pix.z;

    var power: u32 = 0u; // counter for this pixel's power

    for (var dx: i32 = -1; dx <= 1; dx++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            let offset = vec2i(dx, dy);

            let pos: vec2i = vec2i(texel) + offset;

            // ignore pixels outside the texture
            if (pos.x < 0 || pos.y < 0 || pos.x >= i32(dims.x) || pos.y >= i32(dims.y)) {
                continue;
            }

            let neighbor_pix = textureLoad(edge_tex, pos);

            let neighbot_neighbors = unpack_neighbors(neighbor_pix.z);

            let neighbor_neighbor_a = vec2i(neighbot_neighbors.xy);
            let neighbor_neighbor_b = vec2i(neighbot_neighbors.zw);

            // if neighbor points to this pixel, add that to this pixel's power
            if (all(neighbor_neighbor_a * -1 == offset) || all(neighbor_neighbor_b * -1 == offset)) {
                power++;
            }
        }
    }

    textureStore(edge_out, texel, vec4u(in_edge_pix.xyz, power));
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
fn pack_neighbors(neighbors: vec4i) -> u32 {
	// get individual (dx, dy) offsets
	let offset_a: vec2i = vec2i(neighbors.xy);
	let offset_b: vec2i = vec2i(neighbors.zw);

	// turn (dx, dy) offsets into direction indeces (0..7)
	let dir_a: u32 = offset_to_direction(offset_a);
	let dir_b: u32 = offset_to_direction(offset_b);

	// pack the direction indeces together (0..7) → (0..63)
	let packed = dir_a + 8u * dir_b;

	return packed;
}


fn unpack_neighbors(packed_neighbors: u32) -> vec4i {
	// unpack 0..63 into two 0..7 direction indeces
	let dir_a: u32 = packed_neighbors % 8u;
	let dir_b: u32 = packed_neighbors / 8u;

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
