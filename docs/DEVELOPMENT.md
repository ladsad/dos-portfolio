# Development, Build & Deployment Guide

[Back to Documentation Index](./INDEX.md) | [Architecture Guide](./ARCHITECTURE.md) | [Feature Reference](./FEATURES.md) | [Scripts Reference](./SCRIPTS.md)

---

## 1. Prerequisites & System Requirements

Before setting up the project, ensure your workstation meets the following prerequisites:

- **Node.js**: Requires Node.js v18.0.0 or higher (**v20+ LTS recommended**; verified and confirmed on v20.x and v24.15.0).
- **npm**: npm v9.0.0 or higher (verified on npm v11.12.1).
- **Git**: Git 2.30+ for version control and remote deployment.
- **Operating System**: Cross-platform compatible (Windows, macOS, Linux).

---

## 2. Repository Layout & Working Directory Notice

> **CRITICAL DEVELOPER NOTICE**:
> The workspace contains a root directory containing documentation (`docs/`), agent coordination (`.agents/`), LaTeX resume sources (`Resumes/`), and auxiliary scripts (`make_handoff.js`).
> However, the active **Git repository root** and the **npm project root** reside in the `web-app/` subdirectory.
> **All npm commands (`npm run dev`, `npm test`, `npm run build`, `npm run lint`) must be executed inside `web-app/`**.

```
Portfolio Webpage/               <-- Workspace Root
├── .agents/                     <-- Multi-agent coordination metadata
├── docs/                        <-- Central technical documentation suite
├── Resumes/                     <-- Master LaTeX resume source documents
├── make_handoff.js              <-- Root-level auxiliary script stub
├── ORIGINAL_REQUEST.md          <-- Original project specification
├── PROJECT.md                   <-- Master project roadmap and feature inventory
└── web-app/                     <-- Git Root & Node.js Application Directory
    ├── .github/workflows/       <-- GitHub Actions deployment workflow
    ├── package.json             <-- Project manifest and dependencies
    ├── src/                     <-- React 19 application source code
    └── test/                    <-- Native Node.js test suite
```

---

## 3. Step-by-Step Developer Setup

### 3.1 Clone the Repository
```bash
git clone https://github.com/ladsad/dos-portfolio.git
cd dos-portfolio
```

> **Repository Structure Note**:
> In a fresh clone of `https://github.com/ladsad/dos-portfolio.git`, the application files (`package.json`, `src/`, `vite.config.js`, etc.) reside directly at the repository root (`cd dos-portfolio`).
> Within this local multi-agent workspace repository, the application files and Git root are located inside the `web-app/` subdirectory (`cd web-app`). All commands documented below should be run from that directory.

### 3.2 Install Dependencies
Install all required production and development dependencies:
```bash
npm install
```
For a clean, deterministic installation mirroring continuous integration:
```bash
npm ci
```

### 3.3 Verify Test Suite
Run the test runner to verify local environment integrity:
```bash
npm test
```

---

## 4. Local Development Workflow

Start the Vite development server with Hot Module Replacement (HMR):
```bash
npm run dev
```

- **Local Server URL**: `http://localhost:5173/`
- **Hot Module Replacement (HMR)**: Powered by `@vitejs/plugin-react` and `eslint-plugin-react-refresh`. Component edits update instantaneously without losing state.
- **Theme & Audio State Persistence**: Display preferences (`displayMode`), CRT filter toggles (`crtScanlines`, `crtFlicker`), and audio mute status (`retroSoundMuted`) are stored in browser `localStorage`. Clear browser local storage or use an incognito window if you wish to test the first-load boot splash sequence.

---

## 5. Testing Framework & Test Execution

### 5.1 Architecture & Runner
The project employs the native **Node.js Test Runner** (`node:test` and `node:assert/strict`). This eliminates third-party testing bloat (such as Jest or Vitest) and provides near-instantaneous test execution directly via Node.

- **Command**: `npm test`
- **Underlying Command**: `node --test test/*.test.mjs`
- **Test File**: `web-app/test/dataIntegrity.test.mjs`

### 5.2 Test Coverage & Verification Results
Executing `npm test` runs 3 comprehensive integrity tests:

```
✔ all project markdown files exist and are non-empty (3.7776ms)
✔ profile data exports valid header and academic metrics (0.4182ms)
✔ soundEngine module exports required audio functions (0.3911ms)
ℹ tests 3
ℹ suites 0
ℹ pass 3
ℹ fail 0
ℹ duration_ms 107.1876
```

1. **Project Markdown Verification**: Asserts that all 14 project markdown writeups exist in `src/content/projects/`, end with `.md`, and have a non-trivial file size (> 50 bytes).
2. **Profile Data Integrity**: Asserts that `src/data/profile.js` exports the verified student credentials, including name (`Shaurya Kumar`) and updated VIT Chennai CGPA (`8.93`).
3. **Sound Engine Integrity**: Asserts that `src/utils/soundEngine.js` exists and exports all essential audio functions (`playClick`, `playMinesweeperExplosion`, `toggleMuteState`).

---

## 6. Linting Configuration & Known Issue Log

### 6.1 Linter Configuration
Static code analysis is managed by **ESLint 9 Flat Config** in `web-app/eslint.config.js`:
- Uses `@eslint/js` recommended configuration.
- Enforces React 19 hook rules via `eslint-plugin-react-hooks`.
- Validates Vite Fast Refresh compatibility via `eslint-plugin-react-refresh`.
- Ignores production build artifacts (`globalIgnores(['dist'])`).

### 6.2 Lint Command
```bash
npm run lint
```

### 6.3 Forensic Lint Status Log
Running `npm run lint` identifies **5 errors and 1 warning** across 5 files:

| # | File Path | Line:Col | Severity | Rule Name | Cause / Problem Description | Recommended Remediation |
|---|---|---|---|---|---|---|
| 1 | `src/components/apps/BiosSetup.jsx` | 87:8 | Warning | `react-hooks/exhaustive-deps` | `useEffect` has missing dependencies: `'menuItems'`, `'onExit'`, and `'tabs.length'`. | Wrap `onExit` in `useCallback` or stabilize hook dependency array. |
| 2 | `src/components/apps/Minesweeper.jsx` | 64:34 | Error | `react-hooks/purity` | `Math.random` called during render in `populateMines`. | Move mine generation to an event handler or `useEffect` / `useCallback`. |
| 3 | `src/components/apps/Minesweeper.jsx` | 65:34 | Error | `react-hooks/purity` | `Math.random` called during render in `populateMines`. | Same as above. |
| 4 | `src/components/apps/ProjectDetail.jsx` | 85:43 | Error | `no-unused-vars` | `'node'` is defined but never used in `a: ({ node, ...props })`. | Rename `node` to `_node` (matches `^[A-Z_]`) or omit `node`. |
| 5 | `src/components/common/TipWidget.jsx` | 1:17 | Error | `no-unused-vars` | `'useEffect'` is imported from `'react'` but never used. | Remove unused `useEffect` import. |
| 6 | `src/components/os/Window.jsx` | 58:47 | Error | `react-hooks/immutability` | `handleResizeEnd` accessed before declaration inside `useCallback`. | Declare `handleResizeEnd` before use or define function reference cleanly. |

---

## 7. Production Build Pipeline

### 7.1 Build Command
```bash
npm run build
```

### 7.2 Bundler Mechanics (`vite.config.js`)
- Uses `@vitejs/plugin-react` for JSX compilation and Babel Fast Refresh transforms.
- Specifies `base: './'` so all compiled script, style, and image tags use relative URLs (`./assets/index-*.js`). This prevents 404 errors when hosted on GitHub Pages project subpaths (e.g. `https://<username>.github.io/<repo-name>/`).
- Inlines 14 markdown files into the bundle using Vite's `?raw` loader at compile time, eliminating asynchronous runtime markdown fetches.

### 7.3 Build Output Verification
The production build compiles in ~14-15 seconds with zero compilation or syntax errors, emitting static assets into `web-app/dist/`:
- `dist/index.html`: ~1.04 kB (HTML shell with relative asset links)
- `dist/assets/index-*.css`: ~36.06 kB (Bundled Windows 98 / DOS stylesheet)
- `dist/assets/index-*.js`: ~467.99 kB (Minified React 19, Lucide icons, ReactMarkdown, and 14 markdown writeups)

### 7.4 Production Preview
To inspect and test the compiled production artifacts locally prior to deployment:
```bash
npm run preview
```
Serves `dist/` on `http://localhost:4173/`.

---

## 8. Production Deployment Procedures

The portfolio supports dual-mode deployment targeting **GitHub Pages**.

### 8.1 Automated CI/CD (GitHub Actions)
The repository includes a production deployment workflow in `web-app/.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Build project
        run: npm run build

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dist
```

**Trigger**: Every push to the `main` branch or manual click on `Run workflow` in GitHub Actions.

---

### 8.2 Manual CLI Deployment (`npm run deploy`)
Developers can deploy directly from the command line using the `gh-pages` package:

```bash
npm run deploy
```

**Execution Sequence**:
1. npm automatically triggers the `predeploy` lifecycle script (`npm run build`).
2. Vite builds the project into `web-app/dist/`.
3. `gh-pages -d dist` commits the contents of `dist/` and force pushes to the remote `gh-pages` branch.

---

### 8.3 GitHub Repository Configuration
To activate GitHub Pages:
1. Navigate to **GitHub Repository -> Settings -> Pages**.
2. Under **Build and deployment -> Source**, select **Deploy from a branch**.
3. Under **Branch**, select `gh-pages` and directory `/ (root)`.
4. Click **Save**. The application will be published at `https://<username>.github.io/<repo-name>/`.
