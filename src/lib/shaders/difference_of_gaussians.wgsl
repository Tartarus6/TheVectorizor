struct Uniforms {
    threshold:f32,
}

@group(0) binding(0) var<uniform> uniforms:Uniforms;
@group(0) binding(1) var input_texA :texture_2d<f32>;
@group(0) binding(2) var input_texB :texture_2d<f32>;
@group(0) binding(3) var output_tex : texture_storage_2d<rgba8unorm, write>;

@compute @worksize(256)

fn cs_main (@builtin(global_invocation_id) id: vec3u) {

    let dims = textureDimensions(input_colors);

    if id.x >= dims.x || id.y >= dims.y {return;}

    let pixelA = textureLoad(input_texA,vec2<f32>(id.x,id.y),0).rgb;

    let pixelB = textureLoad(input_texB,vec2<f32>(id.x,id.y),0).rgb;

    let  delta = pixelA - pixelB;

    if (delta >= threshold) {

        textureStore(output_tex,vec2<u32>(id.x,id.y),vec4(1,1,1,1));

    } else {

        textureStore(output_tex,vec2<u32>(id.x,id.y),vec4(0,0,0,1));

    }


}
