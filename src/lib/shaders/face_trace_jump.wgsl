/*
 ? This code is mostly LLM-written, and might be worth a rewrite.
*/

/*
Pointer jumping pass for face tracing.

Each directed edge updates its next pointer and propagates the minimum face id
along the cycle.
*/

struct Params {
    edge_count: u32,
};

struct EdgeData {
	next_edge_idx: u32, // index of actual next edge in the array
    jump_next_idx: u32, // index of next jump edge in the array
    face_id: u32,       // index of "face" index in this array
    pos_idx: u32,
    color: vec4f,       // average color
}

/*
edge_tex (rgba16uint):
	x → edge flag        (whether this pixel is part of an edge)
	y → packed neighbors (bitmask to say which of the 8 neighbor pixels are connected edge pixels)
	z → edge_id          (unique edge id, corresponds to the starting index of the pixel's connections)
	w → 0                (unused)
*/

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> edge_data_in: array<EdgeData>;
@group(0) @binding(2) var<storage, read_write> edge_data_out: array<EdgeData>;

// const INVALID: u32 = 0xffffffffu;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let edge_index = gid.x;
    if (edge_index >= params.edge_count) {
        return;
    }

    let jump_next = edge_data_in[edge_index].jump_next_idx;
    let face_id = edge_data_in[edge_index].face_id;

    // if (next_edge == INVALID || next_edge >= params.edge_count) {
    //     edge_data_out[edge_index] = next_edge;
    //     edge_data_out[edge_index] = face_id;
    //     return;
    // }

    let jump_next_next = edge_data_in[jump_next].jump_next_idx;
    let face_next = edge_data_in[jump_next].face_id;

    edge_data_out[edge_index].jump_next_idx = jump_next_next;
    edge_data_out[edge_index].face_id = min(face_id, face_next);
}
