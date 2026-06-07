/*
 ? This code is mostly LLM-written, and might be worth a rewrite.
*/

/*
Face tracing init pass.

Builds a directed edge graph (one edge per pixel-direction) and chooses the next
counterclockwise edge around each vertex. Also writes an initial face id and a
color sample for each directed edge.

edge_tex (rgba16uint):
	x → edge flag        (whether this pixel is part of an edge)
	y → packed neighbors (bitmask to say which of the 8 neighbor pixels are connected edge pixels)
	z → edge_id          (unique edge id, corresponds to the starting index of the pixel's connections)
	w → 0                (unused)
color_tex (rgba8unorm or rgba16float):
    sampled to determine face color

Buffers:
    next_edge   → next directed edge index
    face_id     → minimum edge id for the face
    edge_color  → per-edge color sample (vec4f)
*/

struct EdgeData {
	next_edge_idx: u32, // index of actual next edge in the array
    jump_next_idx: u32, // index of next jump edge in the array
    face_id: u32,       // index of "face" index in this array
    pos_idx: u32,
    color: vec4f,       // average color
}

@group(0) @binding(0) var edge_tex: texture_2d<u32>;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> edge_data_out: array<EdgeData>;

const INVALID: u32 = 0xffffffffu;

// Direction order: E, NE, N, NW, W, SW, S, SE (counterclockwise)
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

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(edge_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let texel = gid.xy;
    let edge_pix = textureLoad(edge_tex, vec2i(texel), 0);
    let edge_flag = edge_pix.x;
    let packed = edge_pix.y;

    let pixel_index = texel.y * dims.x + texel.x;
    let base_edge_index = pixel_index * 8u;

    for (var dir: u32 = 0u; dir < 8u; dir = dir + 1u) {
    	// if pixel isn'y edge or connection doesn't exist, skip
        if (edge_flag == 0u || (packed & (1u << dir)) == 0u) { continue; }

        let my_sparse_idx = get_sparse_index(vec2i(texel), dir, packed);

        let dir_vec = DIRS[dir];
        let neighbor = vec2i(texel) + dir_vec;
        if (neighbor.x < 0 || neighbor.y < 0 || neighbor.x >= i32(dims.x) || neighbor.y >= i32(dims.y)) {
            continue;
        }

        let neighbor_pix = textureLoad(edge_tex, neighbor, 0);
        if (neighbor_pix.x == 0u) {
            continue;
        }

        let neighbor_packed = neighbor_pix.y;
        let incoming_dir = (dir + 4u) & 7u;

        var next_dir = INVALID;
        for (var step: u32 = 1u; step <= 8u; step = step + 1u) {
            let candidate = (incoming_dir + step) & 7u;
            if ((neighbor_packed & (1u << candidate)) != 0u) {
                next_dir = candidate;
                break;
            }
        }

        // if (next_dir == INVALID) { continue; }

        let next_sparse_idx = get_sparse_index(neighbor, next_dir, neighbor_packed);

        // Populate the sparse buffer
        edge_data_out[my_sparse_idx].next_edge_idx = next_sparse_idx;
        edge_data_out[my_sparse_idx].jump_next_idx = next_sparse_idx;
        // Face ID initialization (using sparse indices as IDs)
        edge_data_out[my_sparse_idx].face_id = my_sparse_idx;

        // position
        edge_data_out[my_sparse_idx].pos_idx = texel.x + dims.x * texel.y;

        // color
        let perp = vec2i(-dir_vec.y, dir_vec.x);
        let sample_pos = clamp(vec2i(texel) + perp * 2, vec2i(0, 0), vec2i(dims) - vec2i(1, 1));
        let color = textureLoad(color_tex, sample_pos, 0);

        edge_data_out[my_sparse_idx].color = color;

    }
}

// Helper to find which "slot" a specific direction occupies for a pixel
fn get_sparse_index(pixel_coords: vec2i, dir: u32, packed_mask: u32) -> u32 {
	let base = textureLoad(edge_tex, pixel_coords, 0).z;

	// The direction's slot is the base index + how many bits were set before it
	let mask_before = (1u << dir) - 1u;
	let offset = countOneBits(packed_mask & mask_before);
	return base + offset;
}
