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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> next_edge_in: array<u32>;
@group(0) @binding(2) var<storage, read> face_id_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> next_edge_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> face_id_out: array<u32>;

const INVALID: u32 = 0xffffffffu;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let edge_index = gid.x;
    if (edge_index >= params.edge_count) {
        return;
    }

    let next_edge = next_edge_in[edge_index];
    let face_id = face_id_in[edge_index];

    if (next_edge == INVALID || next_edge >= params.edge_count) {
        next_edge_out[edge_index] = next_edge;
        face_id_out[edge_index] = face_id;
        return;
    }

    let next_next = next_edge_in[next_edge];
    let face_next = face_id_in[next_edge];

    next_edge_out[edge_index] = next_next;
    face_id_out[edge_index] = min(face_id, face_next);
}
