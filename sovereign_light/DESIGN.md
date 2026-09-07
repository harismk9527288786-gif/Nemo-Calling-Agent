---
name: Sovereign Light
colors:
  surface: '#ffffff'
  surface-dim: '#f8fafc'
  surface-bright: '#faf8ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f3ff'
  surface-container: '#f1f5f9'
  surface-container-high: '#e2e7ff'
  surface-container-highest: '#dae2fd'
  on-surface: '#131b2e'
  on-surface-variant: '#434655'
  inverse-surface: '#283044'
  inverse-on-surface: '#eef0ff'
  outline: '#737686'
  outline-variant: '#c3c6d7'
  surface-tint: '#0053db'
  primary: '#004ac6'
  on-primary: '#ffffff'
  primary-container: '#2563eb'
  on-primary-container: '#eeefff'
  inverse-primary: '#b4c5ff'
  secondary: '#006c4a'
  on-secondary: '#ffffff'
  secondary-container: '#82f5c1'
  on-secondary-container: '#00714e'
  tertiary: '#973400'
  on-tertiary: '#ffffff'
  tertiary-container: '#c04400'
  on-tertiary-container: '#ffede7'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dbe1ff'
  primary-fixed-dim: '#b4c5ff'
  on-primary-fixed: '#00174b'
  on-primary-fixed-variant: '#003ea8'
  secondary-fixed: '#85f8c4'
  secondary-fixed-dim: '#68dba9'
  on-secondary-fixed: '#002114'
  on-secondary-fixed-variant: '#005137'
  tertiary-fixed: '#ffdbce'
  tertiary-fixed-dim: '#ffb599'
  on-tertiary-fixed: '#370e00'
  on-tertiary-fixed-variant: '#7f2b00'
  background: '#faf8ff'
  on-background: '#131b2e'
  surface-variant: '#dae2fd'
  border-subtle: '#e2e8f0'
  on-surface-muted: '#64748b'
  status-emerald: '#10b981'
typography:
  display-metrics:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '600'
    lineHeight: 44px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '500'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
  transcript-text:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 24px
  label-mono:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  container-padding: 24px
  stack-gap-lg: 32px
  stack-gap-md: 16px
  stack-gap-sm: 8px
  grid-gutter: 20px
  sidebar-width: 280px
---

## Brand & Style

The design system evolves into a high-clarity, professional light mode optimized for intense enterprise operations. The brand personality remains rooted in **sovereignty and precision**, but shifts its visual language from "command-center dark" to "executive-transparency light." It evokes a sense of absolute reliability, intellectual rigor, and high-stakes professionalism.

The chosen style is **Modern Corporate Minimalism**. It leverages a pristine white and off-white foundation to maximize legibility and reduce cognitive load. The aesthetic response is clinical and authoritative, removing all unnecessary visual noise to focus entirely on data integrity and system performance. It is designed for users who require high information density without the fatigue of high-contrast dark interfaces.

## Colors

The palette is optimized for a bright, professional environment, utilizing a sophisticated range of Slates and Charcoals to maintain visual hierarchy.

- **Main Canvas**: Use `#ffffff` for the primary background to ensure the cleanest possible workspace.
- **Surface Layering**: Use `#f8fafc` (Surface Dim) for sidebars or secondary regions, and `#f1f5f9` (Surface Container) for inset elements like code blocks or input fields.
- **Primary Action**: A vibrant Corporate Blue (`#2563eb`) provides a clear call to action and denotes active system states.
- **Status Accents**: Emerald Green (`#059669`) is used for positive system health, recalibrated for high visibility against light surfaces. 
- **Typography**: Primary text uses Deep Slate (`#0f172a`) to achieve maximum WCAG legibility, while secondary metadata uses a muted slate (`#64748b`).

## Typography

This system prioritizes rapid data scanning and technical precision. **Inter** provides the functional backbone for the UI, while **JetBrains Mono** distinguishes machine-level data.

- **Metrics**: High-value data points use `display-metrics` to command attention.
- **Transcripts**: Real-time dialogue uses `transcript-text`. The 24px line-height is critical for maintaining readability during fast-scrolling live calls.
- **System Tags**: Use `label-mono` for all non-human identifiers (IDs, Timestamps, Hash keys). This visual distinction helps users mentally separate content from metadata.

## Layout & Spacing

The layout follows a **Fixed-Fluid Hybrid** model designed for high-density dashboards.

- **Structure**: A fixed 280px sidebar manages navigation, while the main content area utilizes a 12-column fluid grid.
- **Rhythm**: Spacing is governed by a strict 8px/4px scale. Content density is maintained at `stack-gap-md` (16px) to ensure significant data volume is visible on the first fold.
- **Breakpoints**: 
  - **Desktop (1440px+)**: 12 columns, 24px margins.
  - **Tablet (768px-1024px)**: 6 columns, 16px margins, sidebar collapses to a 64px rail.
  - **Mobile**: Single column stack with horizontal overflow enabled for complex data tables.

## Elevation & Depth

Hierarchy in the light mode system is achieved through **Subtle Outlines** and **Ambient Shadows**. This prevents the UI from feeling washed out while maintaining a "sovereign" and grounded feel.

1. **Base Surface**: The main application background is flat.
2. **Card Elevation**: Containers use a 1px solid `border-subtle` (`#e2e8f0`). For active or highlighted content, apply a soft, highly diffused shadow: `0px 4px 12px rgba(15, 23, 42, 0.05)`.
3. **Floating Elements**: Modals and dropdowns use a more pronounced elevation: `0px 12px 32px rgba(15, 23, 42, 0.1)`.

Avoid heavy shadows or dark glows. Depth should feel like layers of premium paper stacked precisely on a white desk.

## Shapes

The shape language reflects the "ROUND_EIGHT" standard (8px/0.5rem base) to balance technical precision with modern approachability.

- **Standard Containers**: All dashboard cards and primary containers use `rounded-lg` (16px).
- **Interactive Elements**: Buttons, inputs, and selection chips use the base `rounded` (8px).
- **Status Elements**: System status indicators and "Live" tags use `rounded-xl` (full pill) to differentiate them from square-cornered data blocks.

## Components

### Buttons
- **Primary**: Solid `#2563eb` with white text.
- **Secondary**: Clear background with a 1px `#e2e8f0` border and `#0f172a` text.
- **Tertiary/Ghost**: No border, slate text, light gray background on hover.

### Inputs
- **Field Style**: Background set to `surface-container` (`#f1f5f9`) with no border in the default state to create an "etched" look.
- **Focus State**: Background shifts to white with a 2px solid primary blue border.

### Data Tables
- Row height: 52px for comfortable scanning.
- Alternating rows are not used; instead, use a 1px bottom border (`#e2e8f0`).
- Header cells: `#f8fafc` background with `label-mono` uppercase text.

### Status Pills
- Use a low-opacity version of the status color for the background (e.g., 10% Emerald) with a high-contrast version of that same color for the text to ensure accessibility on white surfaces.

### Cards
- Dashboard metrics cards should feature a `label-mono` header in muted slate and a `display-metrics` value in deep charcoal.