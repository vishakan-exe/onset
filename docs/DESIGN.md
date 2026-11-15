# Design notes: the light table

The interface is written for people who will question the instrument, so its first job is to keep every claim next to
the evidence for it. Its second job is to not look like the dashboard every analysis tool grows into: a column of
rounded cards, a gradient header, a doughnut chart and an emoji. The look is borrowed from a place where small
differences are examined with care: **a photographer's light table**.

## The concept

| Element | Light-table counterpart | Used for |
|---|---|---|
| Page ground `--table` (pale acrylic, `#dde1dc`) | the lit glass | every page |
| **Plates** on film-black `--film` | contact sheets and negatives | faces, always grayscale crops; the thing being examined |
| **Chart paper** `--chart` with a red-ish grid | strip-chart recorder paper | anything that runs along time: the Time Lens, traces |
| **Red pen** `--pen` | the grease pencil on the sheet | what the *instrument claims*: detections, the spotting threshold, events |
| **Cyanotype blue** `--cyan` | blueprint ink | what the *model explains*: attributions, selection, links |
| Prussian ink `--ink`, graphite for secondary text | the writing on the margin | text and traces |
| A registration-mark kicker above each title | printers' crop marks | every station head |

Exactly two inks carry meaning, and the rule is kept everywhere: if it is red, the instrument is asserting something; if
it is blue, it is showing what an assertion rests on. Status colours (good, caution, poor) appear only for recording
quality and for pass/fail checks, never as decoration.

The grid on the chart paper carries information: a vertical line every 100 ms (a micro-expression is at most five heavy
squares wide) and, in the detection lane, the heavy horizontal line *is* the spotting threshold. Where something is
simulated or magnified, it is labelled on the object, not in a footnote.

## Type

* **Archivo** (variable width and weight) for the interface: wide, heavy display titles (`font-stretch` up to 125 %),
  condensed labels (down to 80 %), tabular figures for every number.
* **Source Serif 4** for explanation: ledes, captions, the reasoning. Prose is set at 17 px, 62 characters wide, because
  it is meant to be read, in the voice of a lab notebook ("The instrument found…", "This cannot tell…").
* Both are bundled (SIL OFL), so nothing is fetched from a network.

## The six stations

| Station | Design idea |
|---|---|
| **Capture** | a guided pre-flight (light, sharpness, frame rate, face position, measured by the same machinery that will analyse the clip), consent, a 3-2-1, a recording of up to 12 s with a stop control, a review with a contact strip and a frame-timing check, and a "developing tray" while it is analysed. Nothing is analysed live: the whole recording is looked at, so the quiet stretches can serve as the reference. |
| **Bench** | the report for one recording. Left: the plate player (stabilised crops with an SVG overlay of regions and displacement arrows). Right: the Time Lens and the findings log. Below: a chronophotograph strip (the recording, the magnified recording, and where the face moved), then the explanation (what moved, what it could mean, where the evidence sits) next to recording quality and limits. |
| **Cabinet** | 30 synthetic specimens. Watch first, mark what you saw, *then* reveal the truth, so the tool is easy to question. A stress lab re-films a specimen with a worse camera, noise, blur, compression or a weaker movement. Two look-alikes can be compared side by side. |
| **Cue Lab** | the critical module. Nine sections, from the position ("why a face cannot be read for lies") through base rates, effect size, small counts and a forking-paths demonstration to a segment study and a protocol checklist. The page, like the server, refuses to display a field shaped like a verdict. |
| **Evidence** | what has and has not been measured, as charts with a table twin for each (sensitivity per camera, false alarms and their causes, calibration, amplitude read-out, category sets, controls, learning-harness runs). |
| **Atlas** | the twelve movements, the 27 regions, the template for each and which pairs are hard to tell apart, read from the templates the analysis itself uses so the page cannot drift from the instrument. |

### The first look

The first time the app opens, a specimen is played at the size a person would see it: a frown lasting four tenths of a
second that moves the brows by less than a pixel. Then the instrument "develops" it, as a latent image comes up in a
tray: the chart paper emerges, the movement is found, magnified and laid out frame by frame, and the page ends on what
the instrument can and cannot say. It is the shortest honest demonstration of the whole project, and it needs no webcam.

## Charts

All charts share one set of primitives (`web/js/charts.js`): hairline grid, 2 px lines, markers with a surface ring, a
legend whenever there is more than one series, a crosshair tooltip that follows the keyboard as well as the pointer, and
**a table twin for every chart**. Charts are rebuilt at their real pixel width when their container resizes, so text is
always its real size. Series colours live on the marks; text is always drawn in text colours. The five categorical colours
were checked against the light surfaces for separation and contrast.

## Accessibility

* A skip link; visible focus rings (`--focus`) on every control; a logical tab order; keyboard control of the plate
  player, specimen marking, Time Lens and chart crosshairs.
* Every chart and drawing has a table twin or text equivalent; live regions announce recording state, analysis progress
  and the running count of marks.
* `prefers-reduced-motion` switches off animation (the "developing" effects become instant); `forced-colors` mode
  restores borders and the current-page outline.
* Text colours (ink, graphite, pen-text, cyan and the status colours) are at least 4.5 : 1 against the table, mat and
  chart-paper grounds they are used on; the lighter `--faint` is reserved for ticks and rules and is never used for text.
  Where red and blue separate two things, labels and positions repeat the distinction.
* Layout is responsive from a wide desk (spine navigation, two-column bench) to a phone (single column, controls reachable
  by thumb); all stations were rendered and read at both widths.

## Architecture

* Plain JavaScript ES modules, no framework and no build step: `main.js` (shell, hash router), `api.js` (all network
  access), `ui.js` and `util.js` (shared pieces), one module per station or visual component. About 5,000 lines of
  JavaScript and CSS in total, so the whole interface can be read in an afternoon.
* The same interface runs against the local server or, with `window.ONSET_STATIC`, against static files: the published
  *showcase* shows the specimen library and evidence without computing anything and hides the capture and upload paths.
* Fonts and icons are bundled; there is no analytics, no external request, no cookie.

## What was deliberately avoided

* A grid of equal rounded cards with soft shadows; gradients as decoration; a "score" in a doughnut; confidence shown as
  a single big percentage without its interval; a green tick for "passed".
* A "lie" or "emotion" gauge of any kind.
* Stock face imagery: the only faces are synthetic specimens (shown in colour in the Cabinet, as what they are: renders)
  and the user's own recording, which appears only as grayscale crops on a dark plate.
* Hiding the instrument's limits in a footer: each panel states what is measured, what is modelled and how sure it is,
  and the report names the recording's own limits next to the findings.
