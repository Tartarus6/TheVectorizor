struct Uniforms {
    partial_sum_size: u32, /// each thread will be in charge of summing this many elements
    num_remaining_elements: u32,   /// how many elements exist in `in_partial_sums`
}


@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage,read> in_partial_sums: array<f32>;
@group(0) @binding(2) var<storage,read_write> out_partial_sums: array<f32>;


// TODO: switch references to `arrayLength(&out_partial_sums)` and `arrayLength(&in_partial_sums)` to use `total_elements`, since we don't resize the arrays between passes, we just leave the extra space
@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if id.x * uniforms.partial_sum_size >= uniforms.num_remaining_elements {return;}

    var sum: f32 = 0f;

    let start_in_index = id.x * uniforms.partial_sum_size;
    let end_in_index = min(start_in_index + uniforms.partial_sum_size, uniforms.num_remaining_elements);
    for (var in_index = start_in_index; in_index < end_in_index; in_index++) {
        sum += in_partial_sums[in_index];
    }

    out_partial_sums[id.x] = sum;
}
