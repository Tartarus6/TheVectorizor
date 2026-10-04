# Adding and Using Icons

The CSS of this site is designed to handle lucide icons perfectly.

## Steps for Adding and Icon:

1. Find the icon you want on [lucide](https://lucide.dev/icons/)
2. Click the "Copy SVG" button
3. Create a `.svelte` file for your icon, call it whatever you want
4. Paste your copied SVG code into the `.svelte` file you just made

The SVG properties are automatically overwritten by the CSS on this site, making sure they automatically adhere with the theming.


## Using an Icon:

The CSS on this site sets the height and width of any lucid svg to 100%, so you have to make sure you put it in a container of the right size for how you want the icon. The icons stroke size is hard-set to match the theming on the site, so that's handled for you. The icons are colored based on `currentColor`, so simply set the text color in order to set the color of the icon.
