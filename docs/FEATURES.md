# Feature & Component Reference

[Back to Documentation Index](./INDEX.md) | [Architecture Guide](./ARCHITECTURE.md) | [Development Guide](./DEVELOPMENT.md) | [Scripts Reference](./SCRIPTS.md)

---

## 1. Executive Summary

This reference provides an exhaustive inventory of all applications, interactive UI components, desktop widgets, audio effects, visual filters, assets, and content customization points across the portfolio web application.

---

## 2. Portfolio Sections & Applications Catalog

The application features 8 specialized windowed/modal applications and lifecycle utilities:

### 2.1 MS-DOS Terminal (`Terminal.jsx`)
- **File**: `web-app/src/components/apps/Terminal.jsx`
- **Default Size & Position**: 600px × 400px, centered on the desktop viewport.
- **Window Title**: `COMMAND PROMPT`
- **Permanence Protection**: Cannot be closed; `WindowManager.jsx` explicitly intercepts `closeWindow('terminal')` to maintain persistent CLI access.
- **Core Capabilities**:
  - **Command History**: Traversed with `Up` and `Down` arrow keys; maintained in `cmdHistory` array.
  - **Tab Autocompletion**: Automatically completes known commands or project names when prefixed with `open ` (e.g. `open risk` + `Tab` -> `open RiskShield`).
  - **Clickable Hyperlinks**: Terminal output formats URLs and project references as clickable spans, allowing users to launch project windows directly from text listings.
  - **Procedural Sound Hooks**: Plays `playFloppySeek()` on project open, `playError()` on invalid commands.
- **Built-in Command Suite**:
  | Command | Aliases | Description / Output |
  |---|---|---|
  | `help` | `?` | Displays formatted manual of all available terminal commands. |
  | `about` | `bio`, `contact` | Prints candidate contact card (location, email, LinkedIn, GitHub). |
  | `education` | `edu`, `school` | Formats academic degrees, institutions, and coursework (VIT Chennai, CGPA 8.93). |
  | `experience` | `exp`, `work` | Chronological employment history with duty highlights (Data Mavericks, Bidaal, eMudhra). |
  | `projects` | `proj`, `ls`, `dir` | Lists all 14 portfolio projects categorized with numbers and domains. |
  | `open <target>` | `launch`, `run` | Launches dedicated GUI project window via 5-tier fuzzy resolver. |
  | `skills` | `tech`, `stack` | Categorized technical competencies across 6 core domains. |
  | `awards` | `honors`, `certs` | Verified certifications, honors, and competition finishes. |
  | `neofetch` | `sysinfo`, `info` | Displays ASCII logo banner with system and software stack specifications. |
  | `minesweeper` | `game`, `winmine` | Launches Windows 98 Minesweeper game window. |
  | `matrix` | — | Displays famous Matrix quote followed by green binary stream easter egg. |
  | `theme <name>` | `color` | Sets active theme (`retro`, `amber`, `matrix`, `cyberpunk`, `modern`). |
  | `clear` | `cls` | Purges terminal history buffer. |
- **5-Tier Project Name Resolver (`resolveProject`)**:
  When `open <name>` is executed, the query is resolved through the following priority cascade:
  1. *Direct Alias Match*: Exact key lookup in `PROJECT_ALIASES` (supports over 40 shortcuts: `risk`, `kestrel`, `pitwall`, `f1`, `confoundr`, `orchestrate`, `finflow`, `churn`, etc.).
  2. *Stripped Alias Match*: Compares non-alphanumeric sanitized strings.
  3. *Exact Project Match*: Case-insensitive match against `projects[i].name`.
  4. *Stripped Project Match*: Alphanumeric comparison against project titles.
  5. *Substring / Bidirectional Inclusion*: Matches if search query is a substring of the project title or vice versa.

---

### 2.2 Project Detail HUD Viewer (`ProjectDetail.jsx`)
- **File**: `web-app/src/components/apps/ProjectDetail.jsx`
- **Default Size & Position**: `60vw` × `60vh`, staggered with cascading screen offset `(50 + N*20, 50 + N*20)`.
- **Window Title**: Set dynamically to the selected project's full title (e.g. `RiskShield`, `Kestrel`).
- **Core Capabilities**:
  - **ASCII Header Banner**: Dynamic ASCII title block rendered in retro monospace typography.
  - **System Status Panel**: Green `[ONLINE]` telemetry badge, category taxonomy tag, and outbound GitHub repository link.
  - **One-Click Git Clone**: Interactive `[COPY CLONE CMD]` button that copies `git clone <link>` to clipboard with a 2-second visual confirmation state (`[COPIED!]`).
  - **Modules Checklist**: Renders project highlights as loaded architectural modules with `[OK]` verification indicators.
  - **Markdown Rendering Engine**: Evaluates authored technical case studies using `ReactMarkdown`, rendering headings, benchmark tables, code snippets, and hyperlinked citations with `target="_blank"`.

---

### 2.3 Retro ATS Resume Viewer (`ResumeViewer.jsx`)
- **File**: `web-app/src/components/apps/ResumeViewer.jsx`
- **Default Size & Position**: `50vw` × `80vh`, default coordinates `(40px, 40px)`.
- **Window Title**: `Resume.txt`
- **Core Capabilities**:
  - **Simulated Paper Document**: Renders inside a `.resume-paper` card against a neutral PDF viewer slate background (`#525659`).
  - **Typographic & Color Fidelity**: Uses serif typography (`Times New Roman`) and exact LaTeX primary branding color `#004f90` (RGB 0, 79, 144) for section headers and divider bars.
  - **Print & PDF Export Button**: Top toolbar button executing native `window.print()`.
  - **Clean Print Styling**: Triggered via `@media print` in `index.css`, hiding desktop frames, taskbars, scanlines, and buttons while preserving margins for clean 1-2 page export.
  - **Sections Rendered**: Header & Links, Education, Experience, Projects (13 bullet summaries), Technical Skills (6 ATS categories), Awards & Certifications.

---

### 2.4 Retro Minesweeper (`Minesweeper.jsx`)
- **File**: `web-app/src/components/apps/Minesweeper.jsx`
- **Default Size & Position**: 340px × 440px, position `(120px, 80px)`.
- **Window Title**: `Minesweeper`
- **Core Capabilities**:
  - **Authentic Win98 Emulation**: Classic Windows 98 `winmine.exe` aesthetics with beveled grid, digital LED counters, and interactive smiley face button (`🙂`, `😮`, `😵`, `😎`).
  - **Two Difficulty Modes**: Beginner (9×9 grid, 10 mines) and Intermediate (16×16 grid, 40 mines).
  - **First-Click Safety Guarantee**: Mine positions are generated *after* the initial click, guaranteeing that the first clicked cell and its 8 adjacent neighbors never contain a mine.
  - **Flood-Fill Zero Propagation**: Uncovering an empty cell recursively reveals all contiguous blank cells.
  - **Flagging Mechanics**: Right-click toggles flag (`🚩`); flags cannot be placed on already uncovered cells.
  - **7-Segment LED Clamping**: Timer and flag counts are formatted via `formatDigits()`, clamped between -99 and 999 with 3-digit zero padding.
  - **Sound Integration**: Detonation triggers `playMinesweeperExplosion()`; winning triggers `playMinesweeperWin()`.

---

### 2.5 Display & Sound Properties (`DisplaySettings.jsx`)
- **File**: `web-app/src/components/apps/DisplaySettings.jsx`
- **Default Size & Position**: 440px × 460px, position `(180px, 100px)`.
- **Window Title**: `Display & Sound Properties`
- **Core Capabilities**:
  - **Color Palette Selection**: Radio selection between 5 distinct themes (`retro`, `amber`, `matrix`, `cyberpunk`, `modern`).
  - **Live Mini Preview Box**: Interactive preview area displaying theme colors, titlebar gradient, and sample button.
  - **CRT Monitor Controls**: Checkboxes to toggle CRT scanline overlay (`crtScanlines`) and phosphor flicker (`crtFlicker`).
  - **Audio Engine Toggle**: Checkbox to enable or mute procedural 8-bit sound effects (`retroSoundMuted`).
  - **Persistence**: Writes settings directly to `localStorage`.

---

### 2.6 PhoenixBIOS Setup Utility (`BiosSetup.jsx`)
- **File**: `web-app/src/components/apps/BiosSetup.jsx`
- **Default Size & Position**: 100vw × 100vh full-screen overlay (`zIndex: 20000`).
- **Activation Hook**: Triggered by pressing `Del` / `Delete` during the boot splash screen or via BIOS shortcuts.
- **Core Capabilities**:
  - **PhoenixBIOS Emulation**: Monospace blue background (`#0000AA`), top tab bar, double-border bevels, and bottom key navigation legend.
  - **5 Menu Tabs**: `Main`, `Advanced`, `Security`, `Boot`, and `Exit`.
  - **Dual Navigation**: Supports full keyboard control (`←`/`→` tabs, `↑`/`↓` items, `Enter` select, `Esc` exit, `F10` save & restart) as well as mouse clicks.
  - **Display Mode Switching**: Advanced tab enables toggling between `RETRO` and `MODERN` display modes.

---

## 3. Interactive UI Components & Desktop Widgets

### 3.1 Taskbar & System Tray (`Taskbar.jsx`)
- **File**: `web-app/src/components/os/Taskbar.jsx`
- **Position**: Fixed bottom edge (`position: absolute; bottom: 0; left: 0; width: 100%; height: 1.75rem`).
- **Sub-Components**:
  - **Start Button**: Beveled button with retro Windows logo and "Start" label. Shows depressed/sunken visual state when active.
  - **Window Task Tabs**: Dynamic buttons for every unclosed window displaying app-specific icons (`💻` Terminal, `📄` Resume, `💣` Minesweeper, `⚙️` Settings, `📁` Projects). Active focused window has sunken `.active` border styling. Clicking unminimizes and elevates focus; clicking an already focused window minimizes it.
  - **Audio Mute/Unmute Toggle Button**: System tray icon (`🔇` when muted, `🔊` when active) that toggles global audio and persists to `localStorage`.
  - **Settings Launcher Button**: Quick-launch shortcut (`⚙️`) to Display & Sound Properties.
  - **Live Digital Clock**: Real-time 12/24-hour digital clock updating every 1000ms with a hover tooltip displaying the full calendar date.

---

### 3.2 Start Menu (`StartMenu.jsx`)
- **File**: `web-app/src/components/os/StartMenu.jsx`
- **Design**: Windows 98 classic cascading menu with a vertical blue gradient sidebar reading "SALAD OS 98".
- **Menu Hierarchy**:
  - **Programs ▶**: Cascading submenu containing shortcuts to all 14 projects with custom domain icons. Styled with `max-height: calc(100vh - 4rem)` and vertical scrolling to prevent screen overflow.
  - **Games & Apps ▶**: Minesweeper (`winmine.exe`).
  - **Documents ▶**: Resume (`Resume.txt`).
  - **Settings ▶**: Display & Sound Properties.
  - **System ▶**: Command Prompt, LinkedIn profile link, and Reboot System (`window.location.reload()`).
  - **Shut Down...**: Replaces DOM body with authentic retro orange message: `IT IS NOW SAFE TO TURN OFF YOUR COMPUTER.`

---

### 3.3 Desktop Icons (`DesktopIcons.jsx`)
- **File**: `web-app/src/components/os/DesktopIcons.jsx`
- **Layout**: Left-aligned vertical icon grid.
- **Shortcuts**:
  1. `My Computer` (Launches Terminal)
  2. `Resume.txt` (Launches ResumeViewer)
  3. `Projects` (Launches Terminal `projects` list)
  4. `Minesweeper` (Launches Minesweeper)
  5. `Control Panel` (Launches DisplaySettings)
  6. `Recruiter Card` (Brings RecruiterWidget into view)
  7. `GitHub` (Outbound link to GitHub profile)
  8. `LinkedIn` (Outbound link to LinkedIn profile)
- **Interaction Logic**: Single click triggers dotted selection border; double-click (or single tap on mobile `onTouchEnd`) executes the shortcut and plays `playOpen()`.

---

### 3.4 Active Desktop Recruiter Widget (`RecruiterWidget.jsx`)
- **File**: `web-app/src/components/apps/RecruiterWidget.jsx`
- **Position**: Pinned to top-right of desktop (`top: 36px, right: 20px, width: 320px`).
- **Capabilities**:
  - Highlights candidate credentials: B.Tech CSE (AI/ML) @ VIT Chennai, CGPA 8.93 / 10.0, Location: New Delhi.
  - 4 Fast-Action Buttons: `📄 View Full CV`, `🖨️ Save PDF`, `🛡️ RiskShield`, `⚡ Kestrel KV`.
  - One-Click Email Copy: Interactive button that copies candidate email to clipboard, displaying `COPIED TO CLIPBOARD!` for 2.5 seconds.
  - Responsive Behavior: Automatically hidden on viewports `<= 768px` (`display: none`) to keep mobile screens uncluttered.

---

### 3.5 LinkedIn Dialog Widget (`LinkedInWidget.jsx`)
- **File**: `web-app/src/components/apps/LinkedInWidget.jsx`
- **Position**: Top-right desktop card (`top: 1.25rem, right: 1.25rem, width: 20rem`).
- **Capabilities**:
  - Displays LinkedIn CDN avatar, name, current headline, and location.
  - Outbound `[CONNECT]` button opening candidate's LinkedIn profile in a new browser tab.
  - Dismissible via standard top-right close control (`X`).

---

### 3.6 Tip Widget ("Did You Know?") (`TipWidget.jsx`)
- **File**: `web-app/src/components/common/TipWidget.jsx`
- **Position**: Floating yellow assistant balloon (`#FFFFE0`) in lower-right viewport.
- **Capabilities**:
  - Displays periodic technical tips and navigation trivia (e.g. keyboard shortcuts, project architecture highlights, theme commands).
  - Triggers initially after 10 seconds, then cycles through 10 unique tips every 50 seconds.
  - Accompanied by a synthesized Web Audio `playDing()` chime on appearance.
  - Dismissible via `[OK]` button or header close control.

---

## 4. Procedural Audio & Visual Effects Engine

### 4.1 Web Audio Synthesizer (`soundEngine.js`)
All sound effects are synthetically produced at runtime via browser oscillators and noise generators. **Zero external MP3, WAV, or OGG files are required.**

| Function | Node Graph / Synthesis Profile | Purpose & Trigger |
|---|---|---|
| `playClick()` | Triangle wave (1200 Hz -> 400 Hz exp ramp, 40 ms) -> Gain (0.12 -> 0.001) | Crisp mechanical mouse click on desktop icons, taskbar buttons, menus, and window controls. |
| `playOpen()` | Sine wave (440 Hz -> 880 Hz exp ramp, 150 ms) -> Gain (0.1 -> 0.001) | Rising retro chime on opening windows or launching applications. |
| `playMinimize()` | Sine wave (700 Hz -> 250 Hz exp ramp, 150 ms) -> Gain (0.08 -> 0.001) | Descending chirp on minimizing windows to the taskbar. |
| `playError()` | Sawtooth wave (150 Hz -> 130 Hz linear ramp, 280 ms) -> Gain (0.2 -> 0.001) | Low buzzy error tone on invalid commands or non-existent projects. |
| `playFloppySeek()` | Synthesized white noise buffer (150 ms) -> Biquad Bandpass Filter (1800 Hz, Q=3) -> Gain (0.15) | Mechanical 3.5-inch floppy disk drive head step chatter. Played when executing `open <project>`. |
| `playMinesweeperExplosion()` | Synthesized white noise buffer (600 ms) -> Biquad Lowpass Filter (600 Hz -> 80 Hz ramp, 550 ms) -> Gain (0.35) | Low resonant detonation rumble when stepping on a mine in Minesweeper. |
| `playMinesweeperWin()` | 4 staggered square wave oscillators [C5 (523 Hz), E5 (659 Hz), G5 (784 Hz), C6 (1046 Hz)], 100 ms spacing | Victorious 8-bit arpeggio fanfare when winning Minesweeper. |
| `playDing()` | Sine wave (880 Hz -> 440 Hz exp ramp, 500 ms) -> Gain (0.1 -> 0.001) | High-pitched notification chime on TipWidget popup. |

---

### 4.2 Visual Filters & Keyframe Animations

| Filter / Animation | CSS Selector | Implementation Technique | Visual Effect Description |
|---|---|---|---|
| **CRT Scanlines** | `.scanlines` | Repeating linear gradient & radial vignette | Simulates horizontal scanlines of a cathode ray tube and curved tube glass edge shading. |
| **Phosphor Flicker** | `.crt-flicker` | `@keyframes flicker` (0.15s infinite) | Modulates container opacity between 0.98 and 1.0, emulating electron gun decay. |
| **Stepped Window Open** | `.window-frame` | `@keyframes windowOpen` (0.3s steps(3, end)) | Stepped scaling (0.7 -> 0.85 -> 1.04 -> 1.0) mimicking 1990s low-frame-rate GUI rendering. |
| **Terminal Line Reveal** | `.output-line` | `@keyframes fadeInLine` (0.08s steps(1, end)) | Discrete text line reveal simulating serial baud-rate terminal transmission. |
| **BIOS Power Surge** | `.crt-flash` | `@keyframes flashOut` (0.4s ease-out) | Instant blinding white screen flash fading to black upon power-on. |
| **Tip Slide-In** | `.tip-widget` | `@keyframes slideIn` (0.5s ease-out) | Smooth vertical upward slide from screen bottom. |

---

## 5. Complete Asset Catalog

| Asset | Source / Location | Type / Size | Usage Context |
|---|---|---|---|
| `favicon.png` | `web-app/public/favicon.png` | Local PNG (239 KB) | Browser tab icon (`<link rel="icon" href="/favicon.png" />`). |
| LinkedIn Avatar | Remote HTTPS URL (`media.licdn.com`) | CDN Image | Profile portrait rendered in `LinkedInWidget.jsx`. |
| `VT323` Font | Google Fonts CDN | Web Font (WOFF2) | Primary retro monospace font for MS-DOS banner, terminal, splash screen, and desktop icon labels. |
| `Inter` Font | Google Fonts CDN | Web Font (WOFF2) | Clean modern sans-serif for modern theme UI elements and buttons. |
| `JetBrains Mono` Font | Google Fonts CDN | Web Font (WOFF2) | Monospace font for Markdown code blocks and JSON syntax. |
| Lucide SVG Glyphs | `lucide-react` package | Bundled Vector SVGs | Window control glyphs (`X`, `Minus`, `Square`, `Copy`), `Lightbulb`, `Linkedin`. |
| Retro Unicode Glyphs | UTF-8 String Constants | Inline Unicode | Emojis used across UI: 🪟, 📁, 🎮, 📄, ⚙️, 💻, ⏻, 🔇, 🔊, 💣, 🙂, 😮, 😵, 😎, 🚩, 📋, 🛡️, ⚡. |

---

## 6. Content Customization Points & Data Architecture

Content is completely decoupled from rendering logic across five data modules and 14 markdown files:

### 6.1 Data Modules (`web-app/src/data/`)
1. **`src/data/profile.js`**:
   - Contains candidate identity, verified contact links, academic records (VIT Chennai CGPA 8.93 / 10.0), categorized skills (programming, ML/data, databases, cloud), and verified honors.
   - Consumed by Terminal `about`, `education`, `skills`, `awards`.
2. **`src/data/experience.js`**:
   - Contains chronological employment history entries for Data Mavericks, Bidaal, and eMudhra with verified achievements and metrics.
   - Consumed by Terminal `experience`.
3. **`src/data/projects.js`**:
   - Imports all 14 project markdown writeups via Vite `?raw` loader.
   - Exports `projects` catalog array containing `name`, `link`, `category`, `highlights` badges, and `content`.
4. **`src/data/resumeData.js`**:
   - Canonical structured resume data matching the LaTeX resume sources in `Resumes/`.
   - Consumed by `ResumeViewer.jsx` to render the printable paper resume.
5. **`src/data/portfolio.js`**:
   - Central aggregator combining exports from `profile.js`, `experience.js`, `projects.js`, and `resumeData.js` into a unified `portfolioData` module.

---

### 6.2 Project Markdown Writeups (`web-app/src/content/projects/`)
Every project features a dedicated markdown case study loaded into `ProjectDetail.jsx`:

| # | File Name | Project Title | Size | Primary Domain & Focus |
|---|---|---|---|---|
| 1 | `riskshield.md` | RiskShield | 6,179 B | Real-time fraud detection with Go, Redpanda, Redis, and Cloudflare AI dual-path scoring. |
| 2 | `kestrel.md` | Kestrel | 5,772 B | Distributed Raft-consensus key-value store in Go with RESP2 TCP interface and AOF durability. |
| 3 | `confoundr.md` | Confoundr | 6,306 B | Causal validity linter for ML pipelines using FastAPI, Docker sandboxes, and Groq/LLaMA 3.1. |
| 4 | `pitwall.md` | Pitwall: F1 Race Prediction Platform | 6,426 B | PySpark Medallion lakehouse and 1D PatchTST-style Masked Autoencoder for 200 Hz telemetry. |
| 5 | `finflow.md` | FinFlow | 6,325 B | 10k+ TPS payment engine in Java 21 / Spring Boot 3.3 with Kafka KRaft and OpenTelemetry. |
| 6 | `hackerrank-orchestrate.md` | HackerRank Orchestrate: Notification Router | 5,676 B | Multimodal AI routing engine with LLaMA 3.1 8B, OCR/FFmpeg, and prompt injection defense. |
| 7 | `churn-hte-causal-ml.md` | Churn HTE: Causal ML | 1,397 B | Causal Forests and Doubly Robust Estimation for personalized customer intervention. |
| 8 | `codewhisper.md` | CodeWhisper | 903 B | CodeT5+ QLoRA fine-tuned developer code assistant with VS Code integration. |
| 9 | `microsegnet-optimizer.md` | MicroSegNet Optimizer | 3,289 B | Automated ML training pipeline with hyperparameter optimization in TensorFlow. |
| 10 | `attention-enhanced-rhn.md` | Attention-Enhanced RHN | 5,604 B | Recurrent Highway Networks enhanced with self-attention and auxiliary memory buffers. |
| 11 | `mustard-archives.md` | Mustard Archives | 1,657 B | Centralized consultancy platform and 100M+ session analytics lakehouse on AWS S3 and PySpark. |
| 12 | `aws-sentiment-analysis.md` | AWS Sentiment Analysis | 1,343 B | Cloud-native NLP pipeline using AWS Lambda, API Gateway, EC2, and Amazon Comprehend. |
| 13 | `artresgan.md` | ArtResGAN | 4,614 B | Hybrid U-Net + ResNet GAN for restoring degraded historical artwork from WikiArt. |
| 14 | `muse-gan.md` | MUSE-GAN | 6,187 B | Multi-view satellite imagery super-resolution GAN trained on WorldStrat satellite data. |
