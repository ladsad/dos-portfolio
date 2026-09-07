# Modular Architecture & System Design Guide

[Back to Documentation Index](./INDEX.md) | [Feature Reference](./FEATURES.md) | [Development Guide](./DEVELOPMENT.md) | [Scripts Reference](./SCRIPTS.md)

---

## 1. System Overview

The **Portfolio Webpage** repository is an interactive retro-computing personal portfolio designed to present software engineering, distributed systems, machine learning, and data engineering projects. Built with **React 19** and bundled with **Vite 7**, the application emulates an authentic **Windows 98** and **MS-DOS** operating system environment directly in the browser.

### Key Architectural Pillars
- **Operating System Shell Emulation**: A window management subsystem (`WindowManager.jsx`) supporting dynamic window instantiation, mouse dragging via `react-draggable`, custom multidirectional resizing, window focus layering (z-index cascade), minimization, and task switching.
- **Dual-Mode Navigation**: Users can explore credentials and projects via graphical desktop icons, the Windows 98 Start Menu, or an interactive MS-DOS command interpreter (`Terminal.jsx`) with alias resolution, history navigation, and tab autocompletion.
- **Zero-Network Audio Synthesis**: 8-bit retro sound effects (clicks, beeps, floppy drive seeking, explosion noise, victory fanfares) are synthesized purely at runtime via the browser's native **Web Audio API** without external audio assets.
- **Unidirectional LaTeX Resume Synchronization**: ATS-compliant resume data originates in master LaTeX sources (`Resumes/Data_Resume.tex` and `Resumes/SDE_Resume.tex`), flowing into structured JSON models (`web-app/src/data/resumeData.js`), and rendered into a printable paper document (`ResumeViewer.jsx`) with exact `#004f90` color matching.

---

## 2. High-Level System Architecture Diagram

```
+---------------------------------------------------------------------------------------------------+
|                                        Browser Viewport                                           |
|                                                                                                   |
|  +---------------------------------------------------------------------------------------------+  |
|  | Layout.jsx (.dos-container [data-theme="retro|amber|matrix|cyberpunk|modern"])              |  |
|  | - CRT Scanlines Layer (.scanlines)                                                          |  |
|  | - Phosphor Flicker Layer (.crt-flicker)                                                     |  |
|  | - DOS Version Banner Header                                                                 |  |
|  |                                                                                             |  |
|  |  +---------------------------------------------------------------------------------------+  |  |
|  |  | WindowManager.jsx (Central OS Kernel & State Manager)                                 |  |  |
|  |  |                                                                                       |  |  |
|  |  |  +---------------------------+  +--------------------------------------------------+  |  |  |
|  |  |  | Desktop Shell Layer       |  | Floating Active Desktop Widgets                  |  |  |  |
|  |  |  | - DesktopIcons.jsx        |  | - RecruiterWidget.jsx (Fast CV / email copy)     |  |  |  |
|  |  |  | - AltTabSwitcher.jsx      |  | - LinkedInWidget.jsx (LINKEDIN.EXE)              |  |  |  |
|  |  |  +---------------------------+  | - TipWidget.jsx (Periodic helper)                |  |  |  |
|  |  |                                 +--------------------------------------------------+  |  |  |
|  |  |  +---------------------------------------------------------------------------------+  |  |  |
|  |  |  | Window.jsx (Wrapped in Draggable, Resizable Border Handles, Title Bar Controls) |  |  |  |
|  |  |  |                                                                                 |  |  |  |
|  |  |  |  [type='terminal']    --> Terminal.jsx (DOS prompt, 5-tier project resolver)    |  |  |  |
|  |  |  |  [type='project']     --> ProjectDetail.jsx (ReactMarkdown, modules checklist) |  |  |  |
|  |  |  |  [type='resume']      --> ResumeViewer.jsx (Printable ATS CV, LaTeX sync)       |  |  |  |
|  |  |  |  [type='minesweeper'] --> Minesweeper.jsx (Windows 98 winmine.exe recreation)   |  |  |  |
|  |  |  |  [type='settings']    --> DisplaySettings.jsx (5 themes, CRT filters, audio)    |  |  |  |
|  |  |  +---------------------------------------------------------------------------------+  |  |  |
|  |  |                                                                                       |  |  |
|  |  |  +---------------------------------------------------------------------------------+  |  |  |
|  |  |  | Taskbar.jsx (Fixed Bottom Bar: Start Button, Window Tabs, Audio Toggle, Clock)   |  |  |  |
|  |  |  | StartMenu.jsx (Programs [14 projects], Games, Documents, Settings, Shut Down)   |  |  |  |
|  |  |  +---------------------------------------------------------------------------------+  |  |  |
|  |  +---------------------------------------------------------------------------------------+  |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                                                                   |
|  +--------------------------------+               +--------------------------------------------+  |
|  | Runtime Lifecycle Overlays     |               | Web Audio Synthesizer                      |  |
|  | - SplashScreen.jsx (BIOS POST) |               | - soundEngine.js (Oscillators/Noise/Gains) |  |
|  | - BiosSetup.jsx (PhoenixBIOS)  |               +--------------------------------------------+  |
|  +--------------------------------+                                                               |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Technology Stack Breakdown

| Subsystem / Layer | Package / Technology | Version | Purpose & Architectural Rationale |
|---|---|---|---|
| **Core Framework** | `react` | `^19.2.0` | Modern React 19 functional architecture using hooks (`useState`, `useEffect`, `useCallback`, `useRef`). Zero class components. |
| **DOM Renderer** | `react-dom` | `^19.2.0` | React 19 client rendering via `createRoot` in `web-app/src/main.jsx`. |
| **Build & Dev Server** | `vite` | `^7.2.4` | Fast HMR, ES module compilation, Rollup bundling, raw file loading via `?raw` imports. |
| **Window Movement** | `react-draggable` | `^4.5.0` | Production drag physics bound to `.window-header` and constrained to `bounds="parent"`. |
| **Markdown Engine** | `react-markdown` | `^10.1.0` | AST-based markdown parsing for 14 project case studies in `ProjectDetail.jsx`. |
| **Vector Icons** | `lucide-react` | `^0.554.0` | Scalable modern SVG glyphs for window management controls (`X`, `Minus`, `Square`, `Copy`, `Lightbulb`, `Linkedin`). |
| **Audio Synthesis** | Native Web Audio API | Browser Standard | Real-time procedural sound generator (`soundEngine.js`). Zero external audio files or network latency. |
| **Typography** | Google Fonts | Web API | `VT323` for authentic DOS green/cyan monospace; `Inter` and `JetBrains Mono` for modern theme; `Times New Roman` for ATS resume. |
| **Test Runner** | Node.js Test Runner | Native (`node:test`) | Built-in test execution via `node --test test/*.test.mjs` with zero third-party dependencies. |
| **Deployment** | GitHub Pages (`gh-pages`) | `^6.1.1` | Static export pipeline using `dist/` with relative asset links (`base: './'`). |

---

## 4. Styling Architecture & Theme System

### 4.1 Monolithic Stylesheet Philosophy
The application styling is centralized in `web-app/src/index.css` (2,333 lines). Rather than fragmenting styles into CSS Modules or introducing CSS-in-JS runtime overhead, a single stylesheet was chosen to:
1. Guarantee instant stylesheet parsing and zero CSS runtime overhead.
2. Enable global theme switching via a single attribute (`data-theme`) on the root container.
3. Consistently emulate physical Windows 98 3D outset/inset border bevels across diverse window components.
4. Unify print stylesheet rules (`@media print`) alongside screen rules.

### 4.2 CSS Variables & Design Tokens
Global design tokens are defined in `:root` and overridden per theme:
- `--dos-blue`: `#0000AA` (MS-DOS default background)
- `--dos-gray`: `#C0C0C0` (Standard Windows 98 chassis button/panel gray)
- `--dos-dark-gray`: `#808080` (Shadow bevels and inactive window headers)
- `--dos-white`: `#FFFFFF` (Highlight bevels and high-contrast text)
- `--dos-cyan`: `#00FFFF` (DOS prompt highlight and links)
- `--dos-green`: `#00AA00` / `#00FF00` (Status badges and terminal indicators)
- `--dos-yellow`: `#FFFF55` (Terminal commands and emphasis)
- `--dos-black`: `#000000` (Terminal background and deep inset shadows)

### 4.3 Windows 98 3D Bevel Simulation
Authentic 3D physical surfaces are achieved through asymmetric two-tone borders:
- **Outset Bevel (Raised window, button, or menu)**:
  ```css
  border: 2px solid var(--dos-white);
  border-right-color: var(--dos-black);
  border-bottom-color: var(--dos-black);
  box-shadow: inset 1px 1px 0px #dfdfdf, inset -1px -1px 0px #808080;
  ```
- **Inset Bevel (Sunken text input, canvas, or pressed button)**:
  ```css
  border: 2px solid var(--dos-dark-gray);
  border-right-color: var(--dos-white);
  border-bottom-color: var(--dos-white);
  box-shadow: inset 1px 1px 0px #000000, inset -1px -1px 0px #dfdfdf;
  ```

### 4.4 CRT Monitor Overlays & Animations
- **Scanlines (`.scanlines`)**: Fixed overlay element with `pointer-events: none` rendering subtle horizontal lines via CSS repeating linear gradients (`linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%)`) combined with a radial vignette mimicking curved tube glass.
- **CRT Flicker (`.crt-flicker`)**: Keyframe animation (`@keyframes flicker`) oscillating container opacity between `0.98` and `1.0` at `0.15s infinite` frequency, emulating decaying cathode phosphor.
- **Window Open Animation**: Emulates low-frame-rate 1990s GUI acceleration using stepped animation:
  ```css
  @keyframes windowOpen {
    0% { transform: scale(0.7); opacity: 0; }
    50% { transform: scale(0.85); opacity: 0.8; }
    80% { transform: scale(1.04); opacity: 1; }
    100% { transform: scale(1.0); opacity: 1; }
  }
  .window-frame {
    animation: windowOpen 0.3s steps(3, end) both;
  }
  ```

### 4.5 Theme Definitions Matrix
The active theme is toggled via `data-theme` on `.dos-container` and persisted in `localStorage.getItem('displayMode')`:

| Theme ID | Name | Background | Primary Foreground | Accent / Prompt | Characteristic Aesthetic |
|---|---|---|---|---|---|
| `retro` | Classic Windows 98 | Teal `#008080` & Blue `#0000AA` | Silver `#C0C0C0` / White `#FFFFFF` | Cyan `#00FFFF` | Authentic Windows 98 desktop with MS-DOS prompt and classic beveled chrome |
| `amber` | Amber Phosphor | Deep Amber `#1a0f00` | Phosphor Amber `#ffb000` | Bright Orange `#ff8800` | Monochromatic DEC VT220 CRT terminal with `0 0 8px #ffb000` text glow |
| `matrix` | Matrix Green | Pitch Black-Green `#001100` | Phosphor Green `#00ff66` | Bright Green `#00cc44` | "Digital Rain" cyberpunk phosphor terminal |
| `cyberpunk` | Cyberpunk Neon | Midnight Violet `#180026` | Hot Pink / Magenta `#ff007f` | Electric Neon Cyan `#00f0ff` | 1980s synthwave neon hacker workstation |
| `modern` | Modern Slate ("CYBERTERM") | Slate Black `#0d0f14` with 40px grid | Muted Gray `#8b949e` | Neon Cyan `#58a6ff` | GitHub dark aesthetic, rounded cards (`border-radius: 10px`), macOS traffic light controls |

---

## 5. Component Hierarchy & Subsystem Map

```
index.html
└── web-app/src/main.jsx
    └── App.jsx (Top-level view controller & boot state machine)
        ├── [isLoading === true] SplashScreen.jsx (POST memory counter & hardware probe)
        ├── [showBios === true] BiosSetup.jsx (PhoenixBIOS interactive configuration)
        └── [Normal] Layout.jsx (.dos-container, .scanlines, .crt-flicker)
            └── WindowManager.jsx (Operating System Kernel)
                ├── DesktopIcons.jsx (8 desktop shortcuts, selection box, execution)
                ├── RecruiterWidget.jsx (Floating card: CV link, PDF trigger, email copy)
                ├── LinkedInWidget.jsx (LINKEDIN.EXE floating profile card)
                ├── TipWidget.jsx (Periodic "Did you know?" helper balloon)
                ├── AltTabSwitcher.jsx (Shift+Tab modal task switcher)
                ├── StartMenu.jsx (Programs [14 projects], Games, Documents, Settings, Shut Down)
                ├── Taskbar.jsx (Start button, dynamic window tabs, audio mute toggle, clock)
                └── Window.jsx (Wrapped in Draggable, header controls, resize handles)
                    ├── [type === 'terminal'] Terminal.jsx (MS-DOS command prompt & project router)
                    ├── [type === 'resume'] ResumeViewer.jsx (Printable ATS resume paper sheet)
                    ├── [type === 'project'] ProjectDetail.jsx (Markdown parser & HUD dashboard)
                    ├── [type === 'minesweeper'] Minesweeper.jsx (Windows 98 winmine recreation)
                    └── [type === 'settings'] DisplaySettings.jsx (Themes, CRT filters, audio toggle)
```

### Component Directory Mapping
- **`web-app/src/components/common/`**:
  - `Layout.jsx`: Top-level frame rendering CRT effects and DOS version banner.
  - `SplashScreen.jsx`: Award Modular BIOS POST sequence (RAM count 0 to 640 KB, drive detection).
  - `TipWidget.jsx`: Timed tooltip cycling 10 engineering tips every 50 seconds.
- **`web-app/src/components/os/`**:
  - `WindowManager.jsx`: Central operating system state manager.
  - `Window.jsx`: Window frame primitive handling drag, resize, maximize, minimize, close.
  - `DesktopIcons.jsx`: Left-aligned desktop shortcut icons.
  - `Taskbar.jsx`: Fixed bottom bar tracking running tasks and system tray.
  - `StartMenu.jsx`: Cascading Windows 98 application launcher.
  - `AltTabSwitcher.jsx`: Keyboard-driven task switching overlay.
- **`web-app/src/components/apps/`**:
  - `Terminal.jsx`: Interactive MS-DOS prompt and CLI navigator.
  - `ResumeViewer.jsx`: Printable paper document ATS resume viewer.
  - `ProjectDetail.jsx`: Markdown HUD dashboard for project case studies.
  - `Minesweeper.jsx`: Authentic Windows 98 winmine game.
  - `DisplaySettings.jsx`: System control panel for themes and audio.
  - `BiosSetup.jsx`: Full-screen PhoenixBIOS setup utility.
  - `RecruiterWidget.jsx`: Floating fast-path recruitment card.
  - `LinkedInWidget.jsx`: Floating LinkedIn connect card.

---

## 6. State Handling & Interaction Mechanics

### 6.1 Window Lifecycle & Z-Index Cascade
All active windows are tracked as plain JavaScript objects in `WindowManager.jsx`:

```javascript
{
  id: 'terminal' | 'resume' | 'minesweeper' | 'settings' | `project-${project.name}`,
  type: 'terminal' | 'resume' | 'minesweeper' | 'settings' | 'project',
  title: string,
  content?: object,              // Project metadata passed to ProjectDetail
  zIndex: number,                // Current stacking order layer
  minimized: boolean,            // Visibility in desktop viewport
  initialPosition: { x, y },     // Default screen coordinates
  initialSize: { width, height } // Viewport constraints
}
```

- **Spawning (`openWindow`)**: When an application or project is launched, `WindowManager` checks if the window already exists. If it exists and is minimized, it restores it and elevates its z-index. If new, it assigns `zIndex = nextZIndex++` and adds it to the window collection with a cascading offset of `50 + (windows.length * 20)`.
- **Z-Index Layering (`bringToFront`)**: Managed via a monotonically increasing integer state `nextZIndex` (initialized to `2`). Clicking on any window frame, taskbar tab, or switching via Alt+Tab sets `win.zIndex = nextZIndex`, increments `nextZIndex`, and marks `activeWindowId = win.id`.
- **Minimization**: Clicking the minimize control (`-`) sets `minimized: true`, triggers `playMinimize()` audio, and clears `activeWindowId`. Minimized windows remain mounted in state and visible on the taskbar, but are hidden from the desktop viewport.
- **Maximization**: Toggled via double-clicking `.window-header` or clicking the maximize control (`Square` / `Copy`). Maximized windows apply `position: fixed; top: 0; left: 0; width: 100%; height: calc(100vh - 40px)` (reserving space for the 40px taskbar) and disable dragging and resizing.
- **Closure Protection**: Terminal is permanent and cannot be destroyed (`if (id === 'terminal') return;`). Closing other windows filters them out of the `windows` array.

### 6.2 Alt+Tab Task Switcher
- Global key listener detects `Shift + Tab` combinations.
- Sets `isAltTabActive = true` and cycles selection pointer `altTabSelectedIndex = (prev + 1) % windows.length`.
- Renders screen-centered `AltTabSwitcher.jsx` modal displaying icons and titles of all open windows.
- Releasing `Shift` dismisses the modal, restores the selected window if minimized, and calls `bringToFront(selectedWindow.id)`.

### 6.3 Sound Toggle & Web Audio Synthesizer
- Audio state is managed in `web-app/src/utils/soundEngine.js` and persisted in `localStorage.getItem('retroSoundMuted')`.
- When unmuted, Web Audio synthesizes 8 distinct retro sound events on-the-fly:
  - `playClick`: Triangle wave (1200 Hz -> 400 Hz, 40 ms) for mouse clicks.
  - `playOpen`: Sine wave (440 Hz -> 880 Hz, 150 ms) for window launches.
  - `playMinimize`: Sine wave (700 Hz -> 250 Hz, 150 ms) for minimizing windows.
  - `playError`: Sawtooth wave (150 Hz -> 130 Hz, 280 ms) for invalid commands.
  - `playFloppySeek`: Bandpass-filtered white noise simulating mechanical head movement.
  - `playMinesweeperExplosion`: Lowpass-filtered white noise detonation rumble.
  - `playMinesweeperWin`: 4-note square wave victory arpeggio (C5, E5, G5, C6).
  - `playDing`: Sine wave (880 Hz -> 440 Hz, 500 ms) for periodic tip notifications.

### 6.4 Mobile Viewport Adaptations (`<= 768px`)
- Responsive CSS rules in `index.css` adapt the desktop environment to mobile screens:
  - Windows automatically expand to full width (`100%`) and height (`calc(100vh - 40px)`).
  - Mouse drag handles and resize borders are deactivated.
  - Floating widgets (`RecruiterWidget`, `LinkedInWidget`) are hidden to preserve screen real estate.
  - Desktop icons support single-tap touch execution via `onTouchEnd`.

---

## 7. LaTeX Resume Synchronization Pipeline

```
+-------------------------------------------------------------+
| Master LaTeX Resume Sources                                 |
| - Resumes/Data_Resume.tex (Data Eng & Applied AI, 9,442 B)  |
| - Resumes/SDE_Resume.tex (Systems & Distributed, 9,595 B)   |
+------------------------------+------------------------------+
                               |
                               | (Unidirectional Normalization)
                               v
+-------------------------------------------------------------+
| Canonical Data Model: web-app/src/data/resumeData.js        |
| - Contact Header (Verified LinkedIn, GitHub, Email, Phone)  |
| - Education (VIT Chennai, B.Tech CS AI/ML, CGPA: 8.93/10.0) |
| - Experience (Data Mavericks, Bidaal, eMudhra)              |
| - Projects (13 Structured Bullet Summaries)                 |
| - Technical Skills (Categorized ATS taxonomy)               |
| - Awards & Certifications (Snowflake SnowPro, Orchestrate)  |
+------------------------------+------------------------------+
                               |
                               | (Data Ingestion)
                               v
+-------------------------------------------------------------+
| UI Viewer: web-app/src/components/apps/ResumeViewer.jsx     |
| - Simulated Paper Sheet (.resume-paper)                     |
| - Times New Roman Serif Typography                          |
| - Exact Primary Color Match: #004f90 (RGB 0, 79, 144)       |
| - Top Toolbar: Print / Save as PDF Button                   |
+------------------------------+------------------------------+
                               |
                               | (window.print() Trigger)
                               v
+-------------------------------------------------------------+
| Print Engine (@media print in web-app/src/index.css)        |
| - Suppresses taskbar, start menu, scanlines, toolbars       |
| - Strips window chrome, 3D borders, drop shadows            |
| - Forces pure white page background and exact paper margins |
| - Produces 100% ATS-compliant printout or PDF export        |
+-------------------------------------------------------------+
```

### 7.1 Color & Typographic Parity
- The primary branding color defined in both LaTeX sources (`\definecolor{primaryColor}{RGB}{0, 79, 144}`) is replicated verbatim across `ResumeViewer.jsx` and `index.css` as `#004f90`.
- Dividers, section titles, and applicant header text share the identical serif typography (`Times New Roman`, `Times`, `serif`) matching LaTeX Computer Modern / Times Roman output.

### 7.2 Print Stylesheet Engine
When a user clicks "Print / Save as PDF", `ResumeViewer.jsx` invokes `window.print()`. The dedicated `@media print` section in `src/index.css` activates:
1. `.no-print` elements (desktop icons, taskbar, start menu, CRT scanlines, flicker overlay, window headers, close buttons) are set to `display: none !important`.
2. The `.resume-paper` container is stripped of box shadows, background borders, and fixed height constraints, rendering cleanly onto standard letter/A4 paper margins.
3. Typography colors are forced to black and `#004f90`, ensuring high-contrast, ATS-scanner-ready output.
