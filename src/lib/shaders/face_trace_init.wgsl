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
	z → 0                (unused)
	w → 0                (unused)
color_tex (rgba8unorm or rgba16float):
    sampled to determine face color

Buffers:
    next_edge   → next directed edge index
    face_id     → minimum edge id for the face
    edge_color  → per-edge color sample (vec4f)
*/

@group(0) @binding(0) var edge_tex: texture_2d<u32>;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> next_edge: array<u32>;
@group(0) @binding(3) var<storage, read_write> face_id: array<u32>;
@group(0) @binding(4) var<storage, read_write> edge_color: array<vec4f>;

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
        let edge_index = base_edge_index + dir;

        if (edge_flag == 0u || (packed & (1u << dir)) == 0u) {
            next_edge[edge_index] = INVALID;
            face_id[edge_index] = INVALID;
            edge_color[edge_index] = vec4f(0.0, 0.0, 0.0, 0.0);
            continue;
        }

        let dir_vec = DIRS[dir];
        let neighbor = vec2i(texel) + dir_vec;
        if (neighbor.x < 0 || neighbor.y < 0 || neighbor.x >= i32(dims.x) || neighbor.y >= i32(dims.y)) {
            next_edge[edge_index] = INVALID;
            face_id[edge_index] = INVALID;
            edge_color[edge_index] = vec4f(0.0, 0.0, 0.0, 0.0);
            continue;
        }

        let neighbor_pix = textureLoad(edge_tex, neighbor, 0);
        if (neighbor_pix.x == 0u) {
            next_edge[edge_index] = INVALID;
            face_id[edge_index] = INVALID;
            edge_color[edge_index] = vec4f(0.0, 0.0, 0.0, 0.0);
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

        if (next_dir == INVALID) {
            next_edge[edge_index] = INVALID;
            face_id[edge_index] = INVALID;
            edge_color[edge_index] = vec4f(0.0, 0.0, 0.0, 0.0);
            continue;
        }

        let neighbor_index = u32(neighbor.y) * dims.x + u32(neighbor.x);
        let next_edge_index = neighbor_index * 8u + next_dir;

        next_edge[edge_index] = next_edge_index;
        face_id[edge_index] = min(edge_index, next_edge_index);

        let perp = vec2i(-dir_vec.y, dir_vec.x);
        let sample_pos = clamp(vec2i(texel) + perp * 2, vec2i(0, 0), vec2i(dims) - vec2i(1, 1));
        let color = textureLoad(color_tex, sample_pos, 0);
        edge_color[edge_index] = color;
    }
}
