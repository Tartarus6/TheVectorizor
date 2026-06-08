# Updated Ideas

͟U͟p͟d͟a͟t͟e: I have solved the issue of finding the right neighbors. The shader passes now store the connections between
        the edge pixels. So reconstructing the path that the edge tracing steps took is easy. This solves the
        staircase issue.

The issue that remains is tracing the closed faces in the edge pixel graph, and getting the color for each face.

So the idea is to use *Pointer Jumping* to have a shader that combined edges into pairs, then pairs of pairs, etc.,
always picking the "counterclockwise" option.

The trick is that each directed edge corresponds to a closed face.
 ╭1─2─3─4─5─╮
A│  ↙↘  ↙↘  │
B│↓↗  ↓↗  ↖↓│
C│↓↑  ↓↑  ↑↓│
D│↑↘  ↙↑  ↙↑│
E│  ↖↗  ↖↗  │
 ╰──────────╯

The shader will have some data structure like the one shown below:
```wgsl
struct GraphData {
    next_edge : array<u32>,
    face_id   : array<u32>,
};
```
The first pass of the shader will run with one thread per edge connection. It will choose the next "counterclockwise"
connection, and write that as the `next_edge` for the thread's edge connection. It will then take the minimum "ID" of
the 2 edge connections (itsself and the one it just connected to), and write that as the face ID for the thread's edge
pixel.

Then the next pass will again have one thread per edge connection. It will look at the previously chosen `next_edge`
for the thread's edge connection, and then look at the `next_edge` for that edge connection. Then, it will write that
"next next" as the new next for the thread's edge connection, doubling the distance that's been checked. It will also
look at the face ID for the thread's edge connection as well as for the new next, and update the thread's face ID to be
the min of the two.

This will be repeated for some number of iterations until every closed shape has been explored (probably just choose
some arbitrary number of passes that should be enough doublings. something like 5-8, idk). By the end, each edge
connection will correspond to some face ID.

͟I͟s͟s͟u͟e͟ ͟1: Color
This one is pretty easy to solve.
When faces are traced, we need to figure out what color each should be filled in with.

For each edge connection that we look at in the face tracing passes, we can sample the color texture 90 degrees offset
from the direction of the connection at something like a distance of 2 pixels from the connection itsself (in order to
make sure we are getting the internal color, not just edge weirdness). Then each time we do a doubling, we can take the
average of the 2 samples, and write that as the edge's color

That strategy might not work, due to unpredictable comparisons. I think each connection on a face might end up with
slightly different average colors.
So instead, we can just do all of the face tracing, then do another set of passes afterwards. We can split the connectiosn
for each face into pairs, and average them. Then average the pairs, then pairs of pairs, etc. And this should allow us to
get a single average color.

Another idea would be to just use the thing mentioned before, with averaging the colors within the face tracing passes,
then just picking the color of the connection corresponding to the face ID as the color for the shape. If enough passes
happen, all of the connections should have approximately the same color. It should be close enough.


͟I͟s͟s͟u͟e͟ ͟2: Infinitely Large Faces
When certain edges are traced, they will produce an inverted shape with infinite area.
 ╭1─2─3─4─5─6─7─╮
A│              │
B│    ↙→←→←↘    │
C│  ↓↗      ↖↓  │
D│  ↓↑      ↑↓  │
E│  ↑↘      ↙↑  │
F│    ↖→←→←↗    │
G│              │
 ╰──────────────╯

When the graph above is traced, there will be 2 faces created. One face will be the center circle, and the other will
be an open shape that is everything except that circle.

This needs to be dealt with somehow. It might just work to check the area of the resulting directed shape, and if it's
infinite, then throw away the shape?


͟I͟s͟s͟u͟e͟ ͟3: Z-Ordering
With a shader-based face tracing, it will have to be decided which shapes are on top of each other.

This wasn't an issue with the CPU approach, since it iterated through pixels from top left corner to bottom right
corner. This strategy happens to always identify the shapes in the correct order to draw them.

We could do this on the CPU side, using the same method that worked before, but having already traced the faces. As
the CPU algorithm builds the final SVG, we can just have it do so from top left to bottom right. That way it will
(hopefully) correctly order the shapes.
