import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

test('all project markdown files exist and are non-empty', () => {
    const contentDir = path.resolve(projectRoot, 'src', 'content', 'projects');
    const files = fs.readdirSync(contentDir).filter(f => f.endsWith('.md'));
    assert.ok(files.length >= 14, 'Expected at least 14 project markdown files');

    for (const file of files) {
        const fullPath = path.join(contentDir, file);
        const stats = fs.statSync(fullPath);
        assert.ok(stats.size > 50, 'File ' + file + ' should have meaningful content');
    }
});

test('profile data exports valid header and academic metrics', async () => {
    const profilePath = path.resolve(projectRoot, 'src', 'data', 'profile.js');
    const content = fs.readFileSync(profilePath, 'utf-8');
    assert.ok(content.includes('8.93'), 'Profile should include updated CGPA of 8.93');
    assert.ok(content.includes('Shaurya Kumar'), 'Profile should include author name');
});

test('soundEngine module exports required audio functions', async () => {
    const soundEnginePath = path.resolve(projectRoot, 'src', 'utils', 'soundEngine.js');
    assert.ok(fs.existsSync(soundEnginePath), 'soundEngine.js must exist');
    const content = fs.readFileSync(soundEnginePath, 'utf-8');
    assert.ok(content.includes('playClick'), 'Must export playClick');
    assert.ok(content.includes('playMinesweeperExplosion'), 'Must export playMinesweeperExplosion');
    assert.ok(content.includes('toggleMuteState'), 'Must export toggleMuteState');
});
