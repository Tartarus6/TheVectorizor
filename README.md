# The Vectorizor

This is a webapp that can be used to turn simple raster (pixel) images into vector images.


## Goals/Limitations
The focus is on simple images. That means limited color palettes, no gradients, etc. Some examples of the targets would be simple logos, or the art from Slime Rancher 2.


## Stages (prospective)
1. quantize
    - do something like mean shift clustering to remove anti-aliasing artifacts
    - probably want to put colors in a perceptualy uniform color space like oklab
2. edge detect
    - somehow, needs to include directionality
3. polygonalization
    - identify the boundaries between colors, and make a polygon with shared points denoting the edges between the colors, prevents gaps and overlaps
4. simplification
    - make the poly lines into bezier curves to clean it
5. svg-ification
    - make it into an svg
