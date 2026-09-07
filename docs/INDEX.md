# Portfolio Web Application Documentation Index

Welcome to the comprehensive technical documentation suite for the **Retro DOS / Windows 98 Portfolio Web Application** ("Salad OS 98"). This documentation covers system architecture, feature inventories, development and deployment procedures, and tooling references.

---

## 1. Documentation Hub & Quick Links

| Guide | Description | Primary Audience |
|---|---|---|
| [**System Architecture & Design**](./ARCHITECTURE.md) | High-level system architecture, React 19 + Vite stack, single-file styling philosophy (`index.css`), 5 themes, component tree, window lifecycle, and LaTeX resume synchronization. | System Architects, Core Maintainers |
| [**Feature & Component Reference**](./FEATURES.md) | Comprehensive inventory of all 8 applications (Terminal, ProjectDetail, ResumeViewer, Minesweeper, etc.), desktop widgets, Web Audio procedural synthesizer, asset catalog, and 14 project case studies. | Product Designers, Frontend Engineers |
| [**Development, Build & Deployment**](./DEVELOPMENT.md) | Developer onboarding, local environment setup, testing framework (`npm test`), ESLint status log, Vite build pipeline (`npm run build`), and GitHub Pages CI/CD. | Developers, DevOps, QA Engineers |
| [**Tooling & Scripts Reference**](./SCRIPTS.md) | Complete npm script catalog, resume data synchronization (`sync-resume.mjs`), `make_handoff.js` CLI spec, and test harnesses. | Automation Engineers, Contributors |

---

## 2. Role-Based Reading Paths

Depending on your role and objective, recommended reading orders are outlined below:

### 🚀 For New Developers & Contributors
1. Start with [Development Guide: Prerequisites & Setup](./DEVELOPMENT.md#1-prerequisites--system-requirements) to clone and install dependencies.
2. Review [Development Guide: Testing Framework](./DEVELOPMENT.md#5-testing-framework--test-execution) to run `npm test`.
3. Read [Architecture: Component Hierarchy](./ARCHITECTURE.md#5-component-hierarchy--subsystem-map) to understand window and app layout.
4. Consult [Features: Content Customization Points](./FEATURES.md#6-content-customization-points--data-architecture) to learn where projects and credentials live.

### 🎨 For UI/UX & Frontend Engineers
1. Study [Architecture: Styling Architecture & Theme System](./ARCHITECTURE.md#4-styling-architecture--theme-system) for 3D bevels, CRT scanlines, and the 5 theme definitions.
2. Explore [Features: Interactive UI Components & Desktop Widgets](./FEATURES.md#3-interactive-ui-components--desktop-widgets) for taskbars, menus, and widgets.
3. Review [Features: Procedural Audio & Visual Effects](./FEATURES.md#4-procedural-audio--visual-effects-engine) for Web Audio synthesis formulas.

### 🛠️ For DevOps & CI/CD Engineers
1. Read [Development Guide: Production Build Pipeline](./DEVELOPMENT.md#7-production-build-pipeline) for Vite bundler mechanics and relative asset paths (`base: './'`).
2. Examine [Development Guide: Production Deployment Procedures](./DEVELOPMENT.md#8-production-deployment-procedures) for GitHub Actions and `gh-pages` workflows.
3. Check [Scripts Reference: Complete NPM Scripts Inventory](./SCRIPTS.md#2-complete-npm-scripts-inventory) for lifecycle commands.

### 📄 For Resume & Credential Maintainers
1. Review [Architecture: LaTeX Resume Synchronization Pipeline](./ARCHITECTURE.md#7-latex-resume-synchronization-pipeline) to trace data flow from `Resumes/*.tex` to `web-app/src/data/resumeData.js`.
2. Inspect [Features: Retro ATS Resume Viewer](./FEATURES.md#23-retro-ats-resume-viewer-resumeviewerjsx) and `@media print` rules.
3. Run `npm run sync:data` via [Scripts Reference: sync-resume.mjs](./SCRIPTS.md#3-resume-context-synchronization-sync-resumemjs) to verify credential parity.

---

## 3. Technology Stack Summary

```
+-------------------------------------------------------------------------------+
|  UI & Framework:       React 19.2.0, React DOM 19.2.0                         |
|  Bundler & Dev Server: Vite 7.2.4 (@vitejs/plugin-react 5.1.1)                |
|  Styling Architecture: Monolithic pure CSS (web-app/src/index.css, 2,333 lines)|
|  Window Dragging:      react-draggable 4.5.0                                  |
|  Markdown Engine:      react-markdown 10.1.0                                  |
|  Iconography:          lucide-react 0.554.0 + Retro Unicode Glyphs            |
|  Audio Synthesis:      Native Web Audio API (soundEngine.js, 8 sounds)        |
|  Testing Runner:       Native Node.js Test Runner (node:test, node:assert)    |
|  Linter:               ESLint 9.39.1 Flat Config (eslint.config.js)           |
|  Deployment:           GitHub Actions (peaceiris/actions-gh-pages@v4)         |
+-------------------------------------------------------------------------------+
```

---

## 4. Repository Structure Overview

```
Portfolio Webpage/
├── .agents/                    # Multi-agent coordination metadata & surveys
├── docs/                       # Technical documentation suite (You are here)
│   ├── INDEX.md                # Master documentation index & reading guide
│   ├── README.md               # Quick docs landing page
│   ├── ARCHITECTURE.md         # System design & architecture guide
│   ├── FEATURES.md             # Feature & component reference catalog
│   ├── DEVELOPMENT.md          # Setup, build, test, and deployment guide
│   └── SCRIPTS.md              # NPM scripts and tooling reference
├── Resumes/                    # Master LaTeX ATS resume documents
│   ├── Data_Resume.tex         # Data engineering & AI specialized resume
│   └── SDE_Resume.tex          # Systems & distributed engineering resume
├── make_handoff.js             # Root auxiliary agent handoff utility
├── PROJECT.md                  # Master project architecture & milestones
└── web-app/                    # Application and Git repository root
    ├── dist/                   # Compiled static production bundle
    ├── public/                 # Static assets (favicon.png)
    ├── scripts/                # Development scripts (sync-resume.mjs)
    ├── src/                    # Application source code
    │   ├── components/         # React components (apps, common, os)
    │   ├── content/projects/   # 14 project markdown case studies
    │   ├── data/               # Normalized state models (profile, projects, resume)
    │   └── utils/              # Procedural audio engine (soundEngine.js)
    ├── test/                   # Automated data integrity test suite
    ├── package.json            # Node.js project manifest & scripts
    └── vite.config.js          # Bundler configuration
```
