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
    
    // Quick validation of key metrics
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
