# Building the mobile site

Written 15 September 2026, against branch `v14-ui-overhaul`.

The desktop redesign was built so the phone layout would be a set of decisions,
not a rewrite. This is what is already in place, what is actually missing, and
the order to build it in.

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

- `.hit` becomes a single column: collection and page on one line, document
  name on the second, snippet below. Remove-and-restore moves from hover into
  a swipe action or a persistent overflow button.
- Scope picker rows get a 44px target, with the whole row toggling selection
  and the expander as a separate right-aligned control with its own 44px box.
- Search chips scroll horizontally in one row rather than wrapping to three.
- The syntax popover becomes a sheet (it already will, via `Popover`).

*Deliverable: a researcher can run and prune a search on a phone.*

### Stage 4 — The long tail

- Bulk download: replace the file table with a list of rows, each with a
  checkbox and size. If that proves fiddly, gate it behind "open on a computer
  to download in bulk" and keep single-file download working.
- Concordance: already a list, needs target sizing only.
- Evidence bullets and claims: check wrapping at 320px.
- Test on a real device, not only an emulated viewport. Safari on iOS differs
  on `dvh`, safe areas and scroll anchoring.

## 4. Things to decide before starting

1. **Is the phone for reading or for research?** If reading, Stage 1 and 2 are
   the whole job and Stage 3 can be a graceful degradation. If research, all
   four stages matter. This decision changes how much of Stage 3 and 4 is worth
   building.
2. **Tablet.** 768–1024px currently gets the desktop layout with a drawer. That
   is probably right, but it is untested.
3. **Offline/PWA.** Not covered here. If researchers want documents on a phone
   in an archive with no signal, that is its own project.

## 5. Notes for whoever builds it

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
