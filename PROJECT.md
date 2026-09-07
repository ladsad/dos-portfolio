# Project: Portfolio Webpage — Retro OS & Documentation Suite

## Architecture
The repository hosts an interactive retro-computing personal portfolio emulating a **Windows 98** and **MS-DOS** operating environment ("Salad OS 98"), paired with a comprehensive modular technical documentation suite.

### Frontend Application Architecture (`web-app/`)
- **Core Framework**: React 19.2.0 + Vite 7.2.4 SPA.
- **Styling Pipeline**: Centralized monolithic stylesheet (`web-app/src/index.css`, 2,333 lines) supporting authentic 3D outset/inset bevels, CRT scanlines, screen flicker, and 5 phosphor/cyberpunk themes (`retro`, `amber`, `matrix`, `cyberpunk`, `modern`).
- **Window Management Subsystem**: `WindowManager.jsx` managing dynamic window spawning, mouse dragging (`react-draggable`), custom border resizing, active window focus with z-index cascade, and Alt+Tab task switching (`AltTabSwitcher.jsx`).
- **Interactive Command Prompt**: `Terminal.jsx` featuring command history, tab completion, clickable links, and a 5-tier fuzzy project resolver (`resolveProject`) supporting over 40 aliases.
- **Procedural Audio Engine**: `soundEngine.js` generating 8 distinct retro sound events in real-time via the native Web Audio API (zero audio media files).
- **LaTeX Resume Synchronization**: Unidirectional pipeline translating master LaTeX documents (`Resumes/Data_Resume.tex`, `Resumes/SDE_Resume.tex`) into structured JSON (`web-app/src/data/resumeData.js`), rendered in `ResumeViewer.jsx` with exact `#004f90` color matching and `@media print` print engine.

### Documentation Suite Architecture (`docs/`)
A modular technical documentation suite housed in `docs/` at repository root:
- `docs/INDEX.md`: Central documentation hub, technology stack matrix, and role-based reading guides.
- `docs/README.md`: Quick-reference landing page and developer commands.
- `docs/ARCHITECTURE.md`: High-level ASCII architecture map, tech stack breakdown, styling philosophy, component hierarchy, window lifecycle, and LaTeX resume sync pipeline.
- `docs/FEATURES.md`: Exhaustive catalog of all 8 applications, desktop shell widgets, procedural sound engine profiles, asset catalog, and 14 markdown project writeups.
- `docs/DEVELOPMENT.md`: Developer onboarding, local setup, native Node.js test runner (`npm test`), ESLint 9 status log, Vite build pipeline (`npm run build`), and GitHub Pages CI/CD.
- `docs/SCRIPTS.md`: Full npm scripts catalog, resume data synchronization utility (`scripts/sync-resume.mjs`), `make_handoff.js` CLI specification, and developer test utilities.

---

## Code Layout

```
Portfolio Webpage/
├── .agents/                                # Multi-agent coordination and audit logs
├── docs/                                   # Central Technical Documentation Suite
│   ├── INDEX.md                            # Master navigation index and reading guide
│   ├── README.md                           # Documentation entry point
│   ├── ARCHITECTURE.md                     # System architecture and design specification (R1)
│   ├── FEATURES.md                         # Feature & component reference catalog (R2)
│   ├── DEVELOPMENT.md                      # Developer onboarding, testing, build & deployment (R3)
│   └── SCRIPTS.md                          # Tooling & npm scripts reference (R4)
├── Resumes/                                # Master LaTeX Resume Sources
│   ├── Data_Resume.tex                     # Data Engineering & AI Resume (9,442 B)
│   └── SDE_Resume.tex                      # Software Engineering & Systems Resume (9,595 B)
├── make_handoff.js                         # Root auxiliary agent handoff utility stub
├── ORIGINAL_REQUEST.md                     # Upstream user requirements specification
├── PROJECT.md                              # This document
└── web-app/                                # React 19 + Vite Application & Git Root
    ├── .github/workflows/deploy.yml        # GitHub Pages CI/CD deployment workflow
    ├── public/favicon.png                  # Static browser favicon (239 KB)
    ├── scripts/sync-resume.mjs             # Resume context verification utility
    ├── src/
    │   ├── components/
    │   │   ├── apps/                       # Terminal, ProjectDetail, ResumeViewer, Minesweeper, etc.
    │   │   ├── common/                     # Layout, SplashScreen, TipWidget
    │   │   └── os/                         # WindowManager, Window, Taskbar, StartMenu, DesktopIcons
    │   ├── content/projects/               # 14 Markdown project case studies (*.md)
    │   ├── data/                           # profile.js, experience.js, projects.js, resumeData.js, portfolio.js
    │   ├── utils/soundEngine.js            # Procedural Web Audio synthesizer
    │   ├── App.jsx                         # Top-level state and boot controller
    │   ├── index.css                       # Monolithic retro stylesheet (2,333 lines)
    │   └── main.jsx                        # React root entry point
    ├── test/dataIntegrity.test.mjs         # Native Node.js test runner suite
    ├── eslint.config.js                    # ESLint 9 Flat Config
    ├── package.json                        # Node.js manifest & scripts
    └── vite.config.js                      # Bundler configuration (base: './')
```

---

## Feature Inventory

| # | Feature / Subsystem | Description | Milestone | Source | Status |
|---|---|---|---|---|---|
| 1 | Education & Profile Sync | VIT Chennai CGPA 8.93 / 10.0, coursework, modern skills taxonomy, verified certifications | Web-M1 | RESUME_CONTEXT.md | DONE |
| 2 | Work Experience Sync | Data Mavericks, Bidaal, eMudhra expanded achievements, metrics, tech stacks | Web-M1 | RESUME_CONTEXT.md | DONE |
| 3 | ResumeViewer Sync | Synchronize resumeData.js education, experience, technicalSkills, awards, and project catalog | Web-M1 | RESUME_CONTEXT.md | DONE |
| 4 | Project Catalog Expansion | 6 new markdown files (RiskShield, Kestrel, Confoundr, Pitwall, FinFlow, Orchestrate) in src/content/projects/ | Web-M2 | Repos & Specs | DONE |
| 5 | Terminal Routing & Aliases | Expanded open <project> matching, 40+ aliases, projects command, skills display in Terminal.jsx | Web-M3 | web-app Terminal.jsx | DONE |
| 6 | StartMenu & Overflow Protection | Adjusted CSS for 14-item Programs submenu and responsive window layouts | Web-M4 | web-app index.css | DONE |
| 7 | Application Build Integrity | Verified clean Vite production build with zero errors | Web-M4 | web-app build test | DONE |
| 8 | Central Docs Hub & Index | Navigation hub, documentation map, role-based reading guides (`docs/INDEX.md`, `docs/README.md`) | Doc-M1 | Scope & Surveys | DONE |
| 9 | Modular Architecture Guide | System design, React 19/Vite stack, index.css, 5 themes, WindowManager, soundEngine, LaTeX sync (`docs/ARCHITECTURE.md`) | Doc-M1 | R1 Spec | DONE |
| 10 | Feature & Component Reference | Apps catalog, interactive UI, widgets, audio profiles, asset catalog, 14 markdown writeups (`docs/FEATURES.md`) | Doc-M1 | R2 Spec | DONE |
| 11 | Development & Deployment Guide | Prerequisites, local setup, native Node test runner, ESLint status log, build pipeline, CI/CD (`docs/DEVELOPMENT.md`) | Doc-M1 | R3 Spec | DONE |
| 12 | Tooling & Scripts Reference | Full npm scripts catalog, sync-resume.mjs, make_handoff.js CLI spec, audio test harnesses (`docs/SCRIPTS.md`) | Doc-M1 | R4 Spec | DONE |
| 13 | Master PROJECT.md Sync | Synchronize PROJECT.md at root with current documentation suite and application layout | Doc-M1 | Architecture Spec | DONE |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| Web-M1 | Resume & Profile Sync | profile.js, experience.js, resumeData.js (header, edu, exp, skills, awards) | none | DONE |
| Web-M2 | Project Catalog & Content | 6 markdown files, projects.js imports & entries, resumeData.js projects array | Web-M1 | DONE |
| Web-M3 | Terminal Routing & Aliases | Terminal.jsx open command router, aliases, projects command, skills display | Web-M2 | DONE |
| Web-M4 | Web App Verification & Build | index.css submenu max-height, npm run build verification, dataIntegrity test | Web-M3 | DONE |
| Doc-M1 | Documentation Suite Authoring & Project Sync | Author `docs/INDEX.md`, `docs/README.md`, `docs/ARCHITECTURE.md`, `docs/FEATURES.md`, `docs/DEVELOPMENT.md`, `docs/SCRIPTS.md`, and update `PROJECT.md` | Web-M4, Survey Complete | DONE |
| Doc-M2 | Review, Adversarial Challenge & Forensic Audit | Independent review of documentation completeness, link integrity, build/tests, forensic audit | Doc-M1 | PLANNED |
