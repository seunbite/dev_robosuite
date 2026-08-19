# Paper HTML and Figure Style

Use this specification for every paper-facing storytelling HTML page and figure in this repository.

## Storytelling HTML

- Preserve the existing serif body typography: `Iowan Old Style`, `Palatino Linotype`, Palatino, Georgia, serif.
- Use a sans-serif system stack for findings, details, tables, captions, navigation, and toggles.
- Organize the page as a short argument. Each section contains, in order: heading, figure or table, caption, `Finding.` box, concise detail bullets, and optional toggles.
- Number figures and tables consecutively in reading order. Use `Figure N. Descriptive title.` and `Table N. Descriptive title.`
- A Finding is a bordered white rectangle with one direct empirical claim. Do not use rounded corners, shadows, gradients, or decorative cards.
- Detail bullets explain protocol, mechanism, exceptions, and limitations. They must not merely repeat the Finding.
- Put dense tables, input examples, protocols, and secondary figures behind square bordered `<details>` toggles.
- Tables use black headers with white text, black one-pixel borders, right-aligned numeric columns, and no decorative color fills.
- Use a white page, black text, black borders, and blue only as the single accent.

## Figure layout

- Follow the proportions of the token-cost Option A/Option B figures: approximately 9.8 × 5.2 inches for one panel. Scale multi-panel figures horizontally while preserving the same panel height.
- Export PNG at 220 dpi and SVG or PDF for vector use. Use a white background and tight bounding box.
- Center the title; use normal font weight. Recommended sizes: title 17 pt, axes 13 pt, tick labels 13 pt, legend 10.5–12 pt, annotations at least 10.5 pt.
- Use one y-axis title with units. Never use a logarithmic y-axis.
- Remove top and right spines. Keep remaining axes near 1 pt and use only light neutral horizontal guides when necessary.
- Legends have a white background, square corners, black border, and no shadow. Keep them inside unused plot space where possible.
- Avoid excess margins and empty space. Align panel titles, axes, and plot heights across multi-panel figures.

## Color and grayscale accessibility

- Permitted ink colors are black `#202020` (or `#000000`) and blue `#1F6FEB` only. White is the paper background, not a data color.
- Every series must remain distinguishable when printed in grayscale. Combine color with at least one redundant encoding: marker shape, solid/dashed/dotted line, hatch, cross-hatch, or outline.
- For bar charts, prefer solid black, white with black diagonal hatch, white with black cross-hatch, and solid blue. Use consistent black or blue outlines.
- For line charts, use distinct markers and line styles in addition to color. Never rely on hue alone.
- For heatmaps, use a sequential blue scale with black/white annotations selected for contrast. State explicitly whether higher or lower is better.

## Quantitative integrity

- Use a linear y-axis and show zero when bar magnitude is the comparison of interest.
- Do not combine independent metrics into an undocumented composite.
- Distinguish observed values from projections visually and in the caption. State all projection assumptions.
- Show sample counts or missingness when they differ across cells.
- Keep method and embodiment ordering identical across related figures and tables.
