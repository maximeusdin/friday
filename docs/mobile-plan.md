# Building the mobile site

Written 15 September 2026 against branch `v14-ui-overhaul`; built the same day
on `v15-mobile`. All four stages below are implemented. The plan is kept as the
record of what was decided and why, with a status note on each stage and a
section at the end on what has and has not been verified.

---

## 1. What already works on a phone

Verified at 375 × 812 during the redesign:

| Behaviour | Status |
|---|---|
| Sessions rail becomes an overlay drawer under 900px | working, with a tap-to-close scrim |
| Scope picker docks as a bottom sheet under 680px | working, full width |
| Help modal docks to the bottom, section rail becomes a tab strip | working |
| Header hides the strapline and nav labels under 680px | working |
| Touch pointers get 44px controls | working, via `pointer: coarse` in tokens.css |
| Safe-area padding under the composer | in place, needs `viewport-fit=cover` (already set) |
| Thread, composer and answers reflow | working |

So a phone user today can read an answer and ask a question. What they cannot
do comfortably is *research*: the viewer, search results and downloads are all
built for a mouse and a wide screen.

## 2. The real problems

These are the things that will not fix themselves with a media query.

**2.1 Three surfaces compete for one screen.**
Chat, Search and the document viewer are peers on desktop. On a phone only one
can exist at a time, and the transitions between them have to be navigation,
not layout. Today `activeTab` and `activeEvidence` are component state; on a
phone they need to behave like history entries so the back gesture works.

**2.2 The document viewer assumes a mouse.**
The toolbar carries back, title, page nav, zoom, find and an overflow menu in
one row. At 375px that is roughly twice the available width. Pinch-zoom does
not exist: zoom is a pair of buttons stepping through `ZOOM_LEVELS`. A scanned
page at 125% on a 375px screen is unreadable, and the fit-to-width case is the
only one that matters on a phone.

**2.3 Search results are a three-column grid.**
`.hit` is `2.25rem | 1fr | auto` with hover-revealed actions. Hover does not
exist on touch, so the remove control is unreachable.

**2.4 The scope tree is a desktop tree.**
Rows are 24px with a separate 18px expander next to a 16px checkbox. Three
different tap targets within 40px of each other.

**2.5 Bulk download is a table with checkboxes.**
`CollectionsDownloads` renders a `<table>` with a checkbox column. It needs to
become a list, or be declared desktop-only and say so.

**2.6 The composer is the only persistent input.**
On a phone the software keyboard takes half the screen. `100dvh` handles the
resize, but the thread must keep its scroll anchored to the newest turn as the
keyboard opens.

## 3. Build order

Four stages. Each ends somewhere shippable.

### Stage 1 — Navigation model (the foundation)

**Built** (`c7366c6`). `lib/useLayout.ts` decides phone / tablet / desktop by
width and touch by pointer plus user agent; the shell stamps `data-layout`,
`data-touch` and `data-view` on `<html>` and `.app`. Viewer and drawer push
history entries on touch layouts. Phones get a bottom tab bar.

Nothing else is worth doing until this exists.

- Introduce a single `view` state: `chat | search | viewer`, replacing the
  implicit pairing of `activeTab` and `activeEvidence`.
- Push a history entry when the view changes on narrow screens, so the phone
  back gesture pops the viewer back to the thread instead of leaving the site.
  `history.pushState` plus a `popstate` listener in `page.tsx`.
- Add a bottom tab bar under 680px: Chat | Search | Sessions. Sessions opens
  the existing drawer. This replaces the header hamburger on phones.
- Keep the desktop layout untouched: everything above is inside the
  `max-width: 680px` branch plus one piece of state.

*Deliverable: you can move between chat, search and a document with the back
gesture, and never lose your place.*

### Stage 2 — The document viewer

**Built** (`0c512f7`). Fit-to-width on touch layouts, pinch with a live CSS
preview and a sharp re-render on release, double-tap between fit and 2.2x,
zoom in the overflow menu as the non-gesture path, collapsible quote strip.

The hardest surface, and the one researchers spend the most time in.

- Collapse the toolbar to: back, page indicator, find, overflow. Zoom moves
  into pinch. Page navigation moves to the scroll itself, which already works
  (continuous scrolling landed in `d6ddc25`).
- Default to fit-to-width on phones: compute scale from the container width and
  `baseSize` rather than the `ZOOM_LEVELS` ladder. The ladder stays for desktop.
- Pinch-zoom via a `touch-action: pinch-zoom` container, or re-rendering the
  page at the pinched scale on gesture end (cheaper; the canvas is already
  re-rendered on zoom change).
- The quote strip becomes a collapsible bar: two lines by default, tap to
  expand. It currently takes a third of a phone screen.
- Keep `RENDER_WINDOW` at 2; memory matters more on a phone, not less.

*Deliverable: reading a cited page on a phone is as good as in any PDF app.*

### Stage 3 — Search and scope

**Built** (`22b09f3`). Single-column hits with visible remove controls,
scrolling chip and option rows, whole-row scope targets at 44px, Enter as
newline on touch keyboards.

- `.hit` becomes a single column: collection and page on one line, document
  name on the second, snippet below. Remove-and-restore moves from hover into
  a swipe action or a persistent overflow button.
- Scope picker rows get a 44px target, with the whole row toggling selection
  and the expander as a separate right-aligned control with its own 44px box.
- Search chips scroll horizontally in one row rather than wrapping to three.
- The syntax popover becomes a sheet (it already will, via `Popover`).

*Deliverable: a researcher can run and prune a search on a phone.*

### Stage 4 — The long tail

**Built** (`4390627`), except real-device testing, which needs hardware or a
simulator this Mac does not have (see below).

- Bulk download: replace the file table with a list of rows, each with a
  checkbox and size. If that proves fiddly, gate it behind "open on a computer
  to download in bulk" and keep single-file download working.
- Concordance: already a list, needs target sizing only.
- Evidence bullets and claims: check wrapping at 320px.
- Test on a real device, not only an emulated viewport. Safari on iOS differs
  on `dvh`, safe areas and scroll anchoring.

## 4. What has been verified, and what has not

Verified in an emulated Chromium tab at 375x812 (touch user agent, five touch
points), 768x1024, 320x568 and 1440x900:

- Layout selection, bottom bar, drawer, history entries and the back gesture.
- Fit-to-width, pinch (synthetic two-pointer events), double-tap, scroll
  anchoring after zoom, the collapsed quote strip, the 320px toolbar.
- Running a search, removing a hit, toggling scope from the sheet, opening a
  hit into the viewer, the downloads and concordance sheets.
- No horizontal overflow on any surface at 320px.

Not verified, because it needs a real device or an iOS simulator (this Mac
has only the Xcode command-line tools, and the simulator needs full Xcode):

- iOS Safari specifics: `100dvh` with the toolbars, the software keyboard
  resizing the visual viewport, safe-area insets on a notched phone, and
  whether `touch-action: pan-x pan-y` fully suppresses page pinch inside the
  viewer on iOS 16 and 17.
- Real pinch feel: gesture inertia and the moment of the sharp re-render.
- Android Chrome's URL bar collapsing during scroll.

Do those on a phone before calling this done. Everything in the list is a
known WebKit or Android behaviour with a known fix, not a design unknown.

## 5. Things that were decided along the way

1. **Is the phone for reading or for research?** If reading, Stage 1 and 2 are
   the whole job and Stage 3 can be a graceful degradation. If research, all
   four stages matter. This decision changes how much of Stage 3 and 4 is worth
   building.
2. **Tablet.** 768–1024px currently gets the desktop layout with a drawer. That
   is probably right, but it is untested.
3. **Offline/PWA.** Not covered here. If researchers want documents on a phone
   in an archive with no signal, that is its own project.

## 6. Notes for whoever extends it

- Every structural dimension is a token in `app/tokens.css`. Change the value,
  not the rule.
- The breakpoints in use are 1100px (narrower rail), 900px (drawer) and 680px
  (sheets, icon-only header). Add phone rules to the 680px block rather than
  inventing a fourth breakpoint.
- `Popover` already switches to a sheet at 680px via `data-sheet`. Any new
  floating surface should go through it rather than positioning itself.
- The viewer's page wrappers are measured from page 1's viewport
  (`baseSize`). Fit-to-width should reuse that rather than measuring separately.
- Do not reach for a UI framework for this. The primitives that exist (Popover,
  Modal, Menu, Toast, Icon) cover every pattern the phone layout needs.
