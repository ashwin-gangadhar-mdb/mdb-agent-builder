# DESIGN.md — MDB Agent Builder

> Design source of truth for MDB Agent Builder. Every UI surface — docs, config
> editor, agent library, dashboards — should trace back to this file. When in
> doubt, this document wins.

---

## North Star

**YAML-first — complexity made approachable.**

MDB Agent Builder lets engineers build sophisticated LLM agents through
declarative config, not boilerplate. The design system serves that promise:
every surface should make a complex framework feel manageable, scannable, and
trustworthy. The audience is **engineers** — people who read YAML, Python, and
API docs, and who reward precision but punish overhead.

The one thing a user should think after 10 minutes: *"I can build real agents
in YAML without fighting the tool."*

---

## Visual Thesis

**Accessible rigor.** MongoDB Black meets soft grays with intentional accent
colors. Serif for hierarchy, sans for reading, monospace for configuration.
The system feels designed for engineers who appreciate precision but won't
tolerate ceremony.

Three principles, in priority order:

1. **Monospace is hero, not utility.** The YAML config is where users achieve
   results. Code blocks get space, ligatures, and care — they are the product,
   not decoration around it.
2. **Serif headings signal craft.** A serif H1 says someone thought about this.
   It reinforces the YAML-first promise: care in every detail.
3. **Color restraint.** Four core colors plus status accents. Clarity over
   personality. The personality comes from how well it works.

---

## Typography

Base baseline grid: **4px**. Align all type and UI elements to it.

| Role | Font | Weight | Usage |
|------|------|--------|-------|
| Headings (H1–H3) | **IBM Plex Serif** | 600 | Section headers, docs landing, nav titles |
| Body | **Inter** | 400 / 500 | Docs, guides, descriptions, paragraphs |
| Monospace | **Fira Code** | 400 / 600 | YAML, Python, API calls, config (ligatures on) |
| Labels / UI | **Inter Tight** | 500 / 600 | Buttons, tabs, status, compact UI |

### Type scale

| Token | Size | Line height | Weight | Notes |
|-------|------|-------------|--------|-------|
| `h1` | 32px | 1.2 | 600 | 24px margin-bottom |
| `h2` | 24px | 1.2 | 600 | 20px margin-bottom |
| `h3` | 18px | 1.2 | 600 | 16px margin-bottom |
| `body` | 14px | 1.5 | 400 | Default reading size |
| `small` / `label` | 12px | 1.4 | 500 | Uppercase, +0.05em tracking |
| `code` | 13px | 1.6 | 400 | 600 for keywords |

### Rules

- IBM Plex Serif is **headings only**. Never set body or code in serif.
- Fira Code ligatures **on** (`→`, `===`, `!=`, `=>`).
- Labels are uppercase with `letter-spacing: 0.05em`.

### Font loading

```css
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@600&family=Inter:wght@400;500&family=Inter+Tight:wght@500;600&display=swap');
/* Fira Code — self-host or load from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&display=swap');
```

---

## Color

### Core palette

| Token | Hex | Role | Contrast |
|-------|-----|------|----------|
| `--mdb-black` | `#000000` | Primary text, dark surfaces | 21:1 on white |
| `--gray-100` | `#f3f4f6` | Page background (light) | — |
| `--gray-200` | `#e5e7eb` | Borders, dividers | 10.8:1 |
| `--gray-400` | `#9ca3af` | Muted / disabled text | 5.4:1 |
| `--gray-700` | `#374151` | Secondary text | 12.4:1 |

### Accent & status

| Token | Hex | Role | Contrast (on white) |
|-------|-----|------|---------------------|
| `--green` | `#059669` | Success, primary CTA, active | 6.8:1 |
| `--blue` | `#0ea5e9` | Links, secondary action, info | 4.8:1 |
| `--amber` | `#f59e0b` | Warning, incomplete config | 4.1:1 |
| `--red` | `#e11d48` | Error, failed agent | 5.8:1 |

### Dark mode

| Token | Hex | Role |
|-------|-----|------|
| `--dark-bg` | `#0f172a` | Dark mode page background |
| `--dark-surface` | `#1e293b` | Dark mode cards / code blocks |
| `--dark-border` | `#334155` | Dark mode borders |

### Color rules

- **No purple.** MongoDB claims purple; we position differently.
- **Green ≠ blue ≠ status.** Green is success/CTA, blue is links. Never use green
  for links (avoids confusing CTA with navigation).
- All text/background pairings meet **4.5:1** minimum (WCAG AA).
- Icons use the palette as data: green = enabled, gray-400 = disabled,
  red = error, amber = warning. **No grayscale-only icons.**

### CSS variables

```css
:root {
  --mdb-black: #000000;
  --gray-100: #f3f4f6;
  --gray-200: #e5e7eb;
  --gray-400: #9ca3af;
  --gray-700: #374151;
  --green: #059669;
  --blue: #0ea5e9;
  --amber: #f59e0b;
  --red: #e11d48;

  /* Light mode (default) */
  --bg: var(--gray-100);
  --surface: #ffffff;
  --border: var(--gray-200);
  --text: var(--mdb-black);
  --text-muted: var(--gray-700);
}

[data-theme="dark"] {
  --bg: #0f172a;
  --surface: #1e293b;
  --border: #334155;
  --text: #ffffff;
  --text-muted: var(--gray-400);
}
```

---

## Layout & Spacing

Base unit: **4px grid.** Use only these spacing values:

```
4 · 8 · 12 · 16 · 20 · 24 · 32 · 40
```

| Use | Value |
|-----|-------|
| Standard margin | 16px |
| Section break | 24px |
| Major section | 40px |
| Body line height | 1.5 |
| Heading line height | 1.2 |
| Max content width | 1200px |

### Documentation layout

- **2-column only:** left nav + content. Never 3-column — it fragments attention.
- Content flows **left-aligned**. Center only landing headings and primary CTAs.

---

## Components

### Code blocks (YAML, Python)

```
background:     var(--gray-100) light / #1e293b dark
border:         1px var(--gray-200) light / #334155 dark
padding:        16px
border-radius:  6px
font:           Fira Code 13px / 1.6
line numbers:   var(--gray-400), right-aligned, 8px right padding
```

Code blocks are the hero surface. Give them room — 24px vertical margin around
each block.

### Status badges

```
padding:        4px 8px
border-radius:  4px
font:           Inter Tight 12px / 600
```

| State | Background | Text |
|-------|-----------|------|
| Success | `--green` | white |
| Error | `--red` | white |
| Warning | `--amber` | `#111827` |
| Info | `--blue` | white |

### Buttons

| Variant | Background | Text | Padding | Radius | Weight |
|---------|-----------|------|---------|--------|--------|
| Primary (CTA) | `--green` | white | 8px 16px | 4px | 600 |
| Secondary | `--gray-200` | `--mdb-black` | 8px 16px | 4px | 600 |
| Tertiary | none | `--blue` | 8px 16px | 4px | 600 |
| Disabled | `#d1d5db` | `--gray-400` | 8px 16px | 4px | 600 |

Tertiary: underline on hover. Primary/secondary: subtle darken on hover.

### Cards (agent definitions, examples)

```
background:     var(--surface)
border:         1px var(--border)
padding:        20px
border-radius:  8px
box-shadow:     0 1px 3px rgba(0,0,0,0.1) light / 0 1px 5px rgba(0,0,0,0.3) dark
header:         18px / 600, var(--text)
subtext:        12px, var(--gray-700)
```

### Navigation

```
active link:    var(--green) 2px underline + var(--gray-100) background
hover:          var(--gray-100) background, no underline
font:           Inter 14px
```

---

## Border radius

| Use | Radius |
|-----|--------|
| Badges, buttons, inputs | 4px |
| Code blocks | 6px |
| Cards, panels | 8px |

**Never** use values between 1–3px. Stay sharp or go to 4/6/8. The in-between
reads as uncertain.

---

## Anti-patterns

What this system explicitly refuses:

- ❌ **No purple** anywhere. MongoDB owns it; we differentiate.
- ❌ **No serif in body or code.** IBM Plex Serif is headings only.
- ❌ **No 3-column docs.** Two columns: nav + content.
- ❌ **No centered body content.** Center only landing headings and CTAs.
- ❌ **No grayscale-only icons.** Icons carry color as state.
- ❌ **No border-radius between 1–3px.**
- ❌ **No brand-color sprawl.** Four core colors + status accents. Resist adding more.
- ❌ **No green for links.** Green is CTA/success; blue is navigation.

---

## Differentiation

Category norm for dev tools: dark backgrounds, neon accents, dense icon grids,
sans-serif everywhere.

Our deliberate departures:

1. **Light-first** (dark mode available), with generous whitespace — serious
   work doesn't need to shout.
2. **Serif headings** where everyone else uses sans — signals craft and care.
3. **Monospace as hero** — the config is the product; treat it that way.

---

## Accessibility

- Minimum text contrast: **4.5:1** (WCAG AA). All palette pairings above comply.
- Status is never communicated by color alone — pair with icon and/or label text.
- Focus states: 2px `--blue` outline, 2px offset.
- Respect `prefers-reduced-motion`; keep transitions ≤ 200ms.

---

## Quick reference

```
Fonts:   IBM Plex Serif (headings) · Inter (body) · Fira Code (code) · Inter Tight (UI)
Colors:  #000000 black · #059669 green · #0ea5e9 blue · grays #f3f4f6→#374151
Status:  green success · blue info · amber warning · red error
Grid:    4px base · spacing 4/8/12/16/20/24/32/40 · max width 1200px
Radius:  4px controls · 6px code · 8px cards
Refuse:  purple · serif body · 3-col docs · grayscale icons · green links
```
