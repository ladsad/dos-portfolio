# Documentation Suite: Retro DOS & Windows 98 Portfolio

This directory contains the official, comprehensive technical documentation for the **Salad OS 98 / Retro DOS-Windows 98 Interactive Portfolio** web application.

For a full reading guide organized by engineering role, see the [Documentation Index](./INDEX.md).

---

## Modular Documentation Catalog

- **[System Architecture & Design (`ARCHITECTURE.md`)](./ARCHITECTURE.md)**  
  High-level system design, React 19 + Vite 7 stack breakdown, monolithic `index.css` styling philosophy, 5 themes, component hierarchy, window lifecycle and z-index cascade, and unidirectional LaTeX resume synchronization.

- **[Feature & Component Reference (`FEATURES.md`)](./FEATURES.md)**  
  Exhaustive catalog of all 8 applications (Terminal, ProjectDetail, ResumeViewer, Minesweeper, DisplaySettings, BiosSetup, RecruiterWidget, LinkedInWidget), desktop shell elements, Web Audio procedural synthesizer (8 sounds), asset catalog, and 14 markdown project writeups.

- **[Development, Build & Deployment Guide (`DEVELOPMENT.md`)](./DEVELOPMENT.md)**  
  Prerequisites, local setup instructions, development workflow (`npm run dev`), native Node.js test runner (`npm test`), ESLint 9 configuration and verified issue log, Vite build pipeline (`npm run build`), and GitHub Pages CI/CD automation.

- **[Tooling & Scripts Reference (`SCRIPTS.md`)](./SCRIPTS.md)**  
  Complete npm scripts catalog, resume data synchronization utility (`scripts/sync-resume.mjs`), `make_handoff.js` CLI specification, and developer audio test harnesses.

---

## Quick Developer Reference

```bash
# 1. Navigate to the application root (where package.json and git reside)
cd web-app

# 2. Install dependencies
npm install

# 3. Verify tests
npm test

# 4. Start local development server
npm run dev

# 5. Build for production
npm run build
```
