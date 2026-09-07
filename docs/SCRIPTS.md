# Tooling & Scripts Reference

[Back to Documentation Index](./INDEX.md) | [Architecture Guide](./ARCHITECTURE.md) | [Feature Reference](./FEATURES.md) | [Development Guide](./DEVELOPMENT.md)

---

## 1. Executive Summary

This reference provides a complete technical inventory of all automation scripts, build commands, validation utilities, and auxiliary developer tools available within the portfolio repository.

---

## 2. Complete NPM Scripts Inventory

All npm scripts are configured in `web-app/package.json` and must be executed from within the `web-app/` directory:

| Script Name | Exact Command | Execution Context | Inputs & Preconditions | Outputs & Side Effects |
|---|---|---|---|---|
| `dev` | `vite` | Local development (`npm run dev`) | Source code in `src/`, `index.html`, `vite.config.js` | Starts local HMR dev server at `http://localhost:5173/`. |
| `build` | `vite build` | Production compile (`npm run build`) | All source files, markdown case studies in `src/content/projects/` | Bundles static production assets into `dist/`. |
| `lint` | `eslint .` | Static analysis (`npm run lint`) | JavaScript and JSX files in `web-app/` | Analyzes code against ESLint 9 rules; outputs formatted error/warning report. |
| `test` | `node --test test/*.test.mjs` | Automated testing (`npm test`) | Node.js v18+, test scripts in `test/`, source data | Executes native Node.js tests; reports TAP/spec results. |
| `sync:data` | `node scripts/sync-resume.mjs` | Context verification (`npm run sync:data`) | `Resume/RESUME_CONTEXT.md` (optional) | Validates sync status between workspace resume specs and application data. |
| `preview` | `vite preview` | Local production preview (`npm run preview`) | Pre-built `web-app/dist/` directory | Serves compiled production build at `http://localhost:4173/`. |
| `predeploy` | `npm run build` | Automatic lifecycle hook | Executed automatically before `npm run deploy` | Ensures fresh compilation of `dist/` before publishing. |
| `deploy` | `gh-pages -d dist` | Manual deployment (`npm run deploy`) | `dist/` directory, remote Git push credentials | Pushes static assets in `dist/` to remote `gh-pages` branch. |

---

## 3. Resume Context Synchronization: `sync-resume.mjs`

### 3.1 Overview
- **File Location**: `web-app/scripts/sync-resume.mjs`
- **Invocation**: `npm run sync:data` or `node scripts/sync-resume.mjs`
- **Runtime**: Node.js ES Module (`"type": "module"`)

### 3.2 Architectural Role & Logic
`sync-resume.mjs` acts as an integrity validator that compares the web application's local data snapshot (`src/data/`) against the upstream master specification in `Resume/RESUME_CONTEXT.md`:

```javascript
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webAppRoot = path.resolve(__dirname, '..');
const resumeContextPath = path.resolve(webAppRoot, '..', 'Resume', 'RESUME_CONTEXT.md');

console.log('--- Checking Resume & Project Context Sync ---');
if (fs.existsSync(resumeContextPath)) {
    console.log('Found source of truth: ' + resumeContextPath);
    const content = fs.readFileSync(resumeContextPath, 'utf-8');
    
    const hasLatestCGPA = content.includes('8.93');
    const hasRiskShield = content.includes('RiskShield');
    const hasKestrel = content.includes('Kestrel');
    const hasPitwall = content.includes('Pitwall') || content.includes('pitwall');

    console.log('- Latest CGPA (8.93) in sync: ' + (hasLatestCGPA ? 'YES' : 'NO'));
    console.log('- Flagship Projects in sync: ' + (hasRiskShield && hasKestrel && hasPitwall ? 'YES' : 'NO'));
} else {
    console.log('RESUME_CONTEXT.md not found at parent workspace path. Using local src/data/ snapshots.');
}

console.log('Data synchronization check complete.');
```

### 3.3 Graceful Fallback Behavior
- If `RESUME_CONTEXT.md` is present in the parent workspace, it validates that the updated CGPA (`8.93`) and flagship projects (`RiskShield`, `Kestrel`, `Pitwall`) are present.
- If `RESUME_CONTEXT.md` is not present, it logs a fallback notice:
  ```
  RESUME_CONTEXT.md not found at parent workspace path. Using local src/data/ snapshots.
  Data synchronization check complete.
  ```
  The script exits cleanly with exit code `0`, ensuring CI/CD workflows are never disrupted.

---

## 4. Root Orchestrator Utility: `make_handoff.js`

### 4.1 Overview & Repository Status
- **File Location**: `C:/Users/shaur/Desktop/Projects/Portfolio Webpage/make_handoff.js`
- **File Status**: Auxiliary root-level utility stub (0 bytes).

### 4.2 Multi-Agent Architectural Purpose
In multi-agent collaborative workflows (Teamwork protocol), agents submit findings and transfer context using a standardized **5-Component Handoff Protocol**:
1. **Observation**: Direct, verbatim code references, line numbers, and tool outputs.
2. **Logic Chain**: Deductive reasoning connecting observations to conclusions.
3. **Caveats**: Uninvestigated areas and edge conditions.
4. **Conclusion**: Final assessment and actionable status.
5. **Verification Method**: Concrete terminal commands (`npm test`, `npm run build`) to independently verify claims.

`make_handoff.js` is the orchestrator-level CLI utility designed to automate the scaffolding and validation of these reports.

### 4.3 Reference Specification & CLI Interface

```
NAME:
  make_handoff.js — Automated Agent Handoff Report Generator

USAGE:
  node make_handoff.js --agent=<agent_dir> [OPTIONS]

OPTIONS:
  -a, --agent <name>       Name of target agent directory in .agents/ (Required)
  -t, --type <type>        Handoff type: 'hard' (default), 'soft', or 'partial'
  -o, --output <path>      Custom destination path (default: .agents/<agent>/handoff.md)
  --validate               Validates that an existing handoff.md contains all 5 required sections
  -h, --help               Displays command-line help manual

BEHAVIOR:
  - Automatically captures the current Git branch and commit hash.
  - Automatically generates UTC ISO 8601 timestamp headers.
  - Generates template markdown with the mandatory 5 sections.
  - Prevents malformed handoffs from stalling downstream specialist workers.
```

---

## 5. Developer & Testing Utilities

### 5.1 Automated Data Integrity Test Suite (`dataIntegrity.test.mjs`)
- **File Location**: `web-app/test/dataIntegrity.test.mjs`
- **Invocation**: `npm test`
- **Architecture**:
  ```javascript
  import { test } from 'node:test';
  import assert from 'node:assert/strict';
  import fs from 'node:fs';
  import path from 'node:path';
  ```
- **Assertions**:
  - Validates that `src/content/projects/` contains 14 markdown files, all non-empty (> 50 bytes).
  - Validates `src/data/profile.js` exports CGPA 8.93 and author credentials.
  - Validates `src/utils/soundEngine.js` exports procedural audio functions (`playClick`, `playOpen`, `playMinimize`, `playError`, `playFloppySeek`, `playMinesweeperExplosion`, `playMinesweeperWin`, `toggleMuteState`, `getMuteState`, `setMuteState`).

### 5.2 Interactive Audio Synthesis Harness (`soundEngine.js`)
Developers can test synthesized audio effects directly in the browser developer console:
```javascript
// In browser DevTools on http://localhost:5173/
import('./src/utils/soundEngine.js').then(engine => {
  engine.playFloppySeek();              // Test 3.5" disk head stepping
  engine.playMinesweeperExplosion();    // Test detonation rumble
  engine.playMinesweeperWin();          // Test 4-note victory arpeggio
});
```
This enables rapid audio tweaking and testing without generating intermediate audio files.
