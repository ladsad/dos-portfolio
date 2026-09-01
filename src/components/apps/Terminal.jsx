import React, { useState, useEffect, useRef } from 'react';
import { portfolioData } from '../../data/portfolio';
import { playClick, playFloppySeek, playError } from '../../utils/soundEngine';

const COMMANDS = [
    'about', 'education', 'experience', 'projects', 'open', 'skills', 
    'awards', 'neofetch', 'minesweeper', 'matrix', 'theme', 'clear', 'help'
];

const PROJECT_ALIASES = {
    // New projects
    'riskshield': 'RiskShield',
    'risk-shield': 'RiskShield',
    'risk shield': 'RiskShield',
    'risk_shield': 'RiskShield',
    'risk': 'RiskShield',
    'kestrel': 'Kestrel',
    'confoundr': 'Confoundr',
    'confounder': 'Confoundr',
    'pitwall': 'Pitwall: F1 Race Prediction Platform',
    'f1': 'Pitwall: F1 Race Prediction Platform',
    'f1-pyspark-analytics': 'Pitwall: F1 Race Prediction Platform',
    'f1 pyspark analytics': 'Pitwall: F1 Race Prediction Platform',
    'f1-pyspark': 'Pitwall: F1 Race Prediction Platform',
    'pitwall: f1 race prediction platform': 'Pitwall: F1 Race Prediction Platform',
    'pitwall f1': 'Pitwall: F1 Race Prediction Platform',
    'race prediction': 'Pitwall: F1 Race Prediction Platform',
    'finflow': 'FinFlow',
    'fin-flow': 'FinFlow',
    'fin flow': 'FinFlow',
    'fin_flow': 'FinFlow',
    'orchestrate': 'HackerRank Orchestrate: Message Notification Router',
    'hackerrank': 'HackerRank Orchestrate: Message Notification Router',
    'hackerrank-orchestrate': 'HackerRank Orchestrate: Message Notification Router',
    'hackerrank orchestrate': 'HackerRank Orchestrate: Message Notification Router',
    'hackerrank-orchestra': 'HackerRank Orchestrate: Message Notification Router',
    'hackerrank orchestrate: message notification router': 'HackerRank Orchestrate: Message Notification Router',
    'message notification router': 'HackerRank Orchestrate: Message Notification Router',

    // Existing projects
    'churn': 'Churn HTE: Causal ML',
    'churn-hte': 'Churn HTE: Causal ML',
    'churn hte': 'Churn HTE: Causal ML',
    'churn hte: causal ml': 'Churn HTE: Causal ML',
    'causal ml': 'Churn HTE: Causal ML',
    'causal-ml': 'Churn HTE: Causal ML',
    'causal': 'Churn HTE: Causal ML',
    'codewhisper': 'CodeWhisper',
    'code-whisper': 'CodeWhisper',
    'code whisper': 'CodeWhisper',
    'microsegnet': 'MicroSegNet Optimizer',
    'microsegnet-optimizer': 'MicroSegNet Optimizer',
    'microsegnet optimizer': 'MicroSegNet Optimizer',
    'microseg': 'MicroSegNet Optimizer',
    'rhn': 'Attention-Enhanced RHN',
    'attention-enhanced-rhn': 'Attention-Enhanced RHN',
    'attention enhanced rhn': 'Attention-Enhanced RHN',
    'attention': 'Attention-Enhanced RHN',
    'mustard': 'Mustard Archives',
    'mustard-archives': 'Mustard Archives',
    'mustard archives': 'Mustard Archives',
    'sentiment': 'AWS Sentiment Analysis',
    'aws-sentiment': 'AWS Sentiment Analysis',
    'aws-sentiment-analysis': 'AWS Sentiment Analysis',
    'aws sentiment analysis': 'AWS Sentiment Analysis',
    'aws': 'AWS Sentiment Analysis',
    'artresgan': 'ArtResGAN',
    'art-res-gan': 'ArtResGAN',
    'art res gan': 'ArtResGAN',
    'musegan': 'MUSE-GAN',
    'muse-gan': 'MUSE-GAN',
    'muse gan': 'MUSE-GAN',
    'muse': 'MUSE-GAN',
};

const resolveProject = (query) => {
    if (!query) return null;
    const cleanQuery = query.trim().toLowerCase();
    const strippedQuery = cleanQuery.replace(/[^a-z0-9]/g, '');

    // 1. Direct alias match
    if (PROJECT_ALIASES[cleanQuery]) {
        const targetName = PROJECT_ALIASES[cleanQuery];
        const proj = portfolioData.projects.find(p => p.name.toLowerCase() === targetName.toLowerCase());
        if (proj) return proj;
    }

    // 2. Stripped alias match (ignores spaces, hyphens, colons, underscores)
    for (const [aliasKey, targetName] of Object.entries(PROJECT_ALIASES)) {
        if (aliasKey.replace(/[^a-z0-9]/g, '') === strippedQuery) {
            const proj = portfolioData.projects.find(p => p.name.toLowerCase() === targetName.toLowerCase());
            if (proj) return proj;
        }
    }

    // 3. Exact project name match
    const exactMatch = portfolioData.projects.find(p => p.name.toLowerCase() === cleanQuery);
    if (exactMatch) return exactMatch;

    // 4. Stripped project name match
    const strippedMatch = portfolioData.projects.find(p => p.name.toLowerCase().replace(/[^a-z0-9]/g, '') === strippedQuery);
    if (strippedMatch) return strippedMatch;

    // 5. Substring / Partial match
    const partialMatch = portfolioData.projects.find(p => {
        const pClean = p.name.toLowerCase();
        return pClean.includes(cleanQuery) || cleanQuery.includes(pClean);
    });
    if (partialMatch) return partialMatch;

    return null;
};

const Terminal = ({ onOpenProject, onOpenApp, onSetTheme }) => {
    const [input, setInput] = useState('');
    const [history, setHistory] = useState([
        { type: 'output', content: 'Welcome to Shaurya Kumar\'s Retro Portfolio (Salad OS 98).' },
        { type: 'output', content: 'Type "help" for commands, "neofetch" for system info, or "minesweeper" to play.' },
        { type: 'output', content: ' ' }
    ]);
    const [cmdHistory, setCmdHistory] = useState([]);
    const [historyPointer, setHistoryPointer] = useState(-1);

    const bottomRef = useRef(null);
    const inputRef = useRef(null);

    useEffect(() => {
        if (bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [history]);

    const handleCommand = (cmd) => {
        const trimmedCmd = cmd.trim();
        if (trimmedCmd) {
            setCmdHistory(prev => [...prev, trimmedCmd]);
        }
        setHistoryPointer(-1);

        const args = trimmedCmd.split(' ');
        const command = args[0].toLowerCase();
        const newHistory = [...history, { type: 'input', content: cmd }];

        playClick();

        switch (command) {
            case 'help':
                newHistory.push({
                    type: 'output',
                    content: `Available commands:
  about         - Display contact info and links
  education     - Show education & coursework
  experience    - Show work experience & achievements
  projects      - List highlighted projects
  open <name>   - Open project window (e.g., "open kestrel", "open riskshield")
  skills        - List technical skills & toolchains
  awards        - Show certifications and honors
  neofetch      - Display retro ASCII hardware & stack info
  minesweeper   - Launch Windows 98 Minesweeper game
  matrix        - Toggle matrix code sequence
  theme <name>  - Set theme (retro, amber, matrix, cyberpunk, modern)
  clear / cls   - Clear terminal screen
  help          - Show this help menu`
                });
                break;

            case 'neofetch':
                newHistory.push({
                    type: 'output',
                    content: `
      .----------------.     shaurya@salad-os-98
     | .--------------. |    -------------------
     | |  /\\_/\\       | |    OS: Salad OS 98 (Build 2026.2)
     | | ( o.o )  AI  | |    Host: VIT Chennai / New Delhi
     | |  > ^ <       | |    Uptime: 199 epochs, 42 minutes
     | '----------------' |  Shell: DOS Command Interpreter
      '----------------'     Memory: 640 KB Base / 32 MB XMS
                             Architecture: Distributed ML & Full-Stack
                             Flagship: RiskShield, Kestrel, Pitwall, Confoundr
                             GitHub: https://github.com/ladsad
                             LinkedIn: https://linkedin.com/in/shaurya-kumar-22262b236`
                });
                break;

            case 'minesweeper':
            case 'game':
            case 'winmine':
                if (onOpenApp) {
                    onOpenApp('minesweeper');
                    newHistory.push({ type: 'output', content: 'Launching Minesweeper (winmine.exe)...' });
                }
                break;

            case 'theme': {
                const requestedTheme = args[1]?.toLowerCase();
                const validThemes = ['retro', 'amber', 'matrix', 'cyberpunk', 'modern'];
                if (!requestedTheme || !validThemes.includes(requestedTheme)) {
                    newHistory.push({
                        type: 'output',
                        content: `Usage: theme <name>\nValid themes: ${validThemes.join(', ')}`
                    });
                } else {
                    localStorage.setItem('displayMode', requestedTheme);
                    if (onSetTheme) onSetTheme(requestedTheme);
                    newHistory.push({
                        type: 'output',
                        content: `Display theme set to "${requestedTheme.toUpperCase()}".`
                    });
                }
                break;
            }

            case 'matrix':
                newHistory.push({
                    type: 'output',
                    content: `01000001 01001001 00100000 01010011 01111001 01110011 01110100 01100101 01101101 01110011
Wake up, Neo...
The Matrix has you.
Follow the white rabbit.
Knock, knock, Neo.`
                });
                break;

            case 'about':
                newHistory.push({
                    type: 'output',
                    content: `NAME:     ${portfolioData.header.name}
LOCATION: ${portfolioData.header.location}
EMAIL:    ${portfolioData.header.email}

LINKS:
  LinkedIn: ${portfolioData.header.linkedin}
  GitHub:   ${portfolioData.header.github}`
                });
                break;

            case 'education':
                portfolioData.education.forEach(edu => {
                    newHistory.push({
                        type: 'output',
                        content: `----------------------------------------
INSTITUTION: ${edu.institution}
DEGREE:      ${edu.degree}
PERIOD:      ${edu.period}

DETAILS:
${edu.details.map(d => `  * ${d}`).join('\n')}
----------------------------------------`
                    });
                });
                break;

            case 'experience':
                portfolioData.experience.forEach(exp => {
                    newHistory.push({
                        type: 'output',
                        content: `----------------------------------------
ROLE:    ${exp.role}
COMPANY: ${exp.company}
PERIOD:  ${exp.period}

HIGHLIGHTS:
${exp.highlights.map(h => `  * ${h}`).join('\n')}
----------------------------------------`
                    });
                });
                break;

            case 'projects': {
                const maxNameLength = Math.max(...portfolioData.projects.map(p => p.name.length));
                newHistory.push({
                    type: 'output',
                    content: `PROJECTS (Type "open <name>" to view details):
--------------------------------------------------------------------------------`
                });
                portfolioData.projects.forEach(proj => {
                    newHistory.push({
                        type: 'output',
                        content: `* ${proj.name.padEnd(maxNameLength + 2)} [${proj.category}]`
                    });
                });
                newHistory.push({ type: 'output', content: '--------------------------------------------------------------------------------' });
                break;
            }

            case 'open': {
                const projectName = args.slice(1).join(' ').trim();
                if (!projectName) {
                    newHistory.push({
                        type: 'output',
                        content: 'Usage: open <project name>'
                    });
                } else {
                    const matchedProject = resolveProject(projectName);
                    const targetName = matchedProject ? matchedProject.name : projectName;
                    playFloppySeek();
                    const success = onOpenProject(targetName);
                    if (success) {
                        newHistory.push({
                            type: 'output',
                            content: `Opening project "${targetName}"...`
                        });
                    } else {
                        playError();
                        newHistory.push({
                            type: 'output',
                            content: `Project "${projectName}" not found. Type "projects" for a complete list.`
                        });
                    }
                }
                break;
            }

            case 'skills':
                newHistory.push({
                    type: 'output',
                    content: `----------------------------------------
TECHNICAL SKILLS
----------------------------------------
PROGRAMMING:   ${portfolioData.skills.programming}
ML/DATA:       ${portfolioData.skills.ml_data}
DATABASES:     ${portfolioData.skills.databases}
CLOUD/INFRA:   ${portfolioData.skills.cloud_infra}
FULL STACK:    ${portfolioData.skills.full_stack}
AI / AGENTIC:  ${portfolioData.skills.ai_integration}
----------------------------------------`
                });
                break;

            case 'awards':
                newHistory.push({
                    type: 'output',
                    content: portfolioData.awards.map(a => `- ${a}`).join('\n')
                });
                break;

            case 'clear':
            case 'cls':
                setHistory([]);
                return;

            case '':
                break;

            default:
                playError();
                newHistory.push({
                    type: 'output',
                    content: `Command not found: "${command}". Type "help" for available commands.`
                });
        }

        setHistory(newHistory);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            handleCommand(input);
            setInput('');
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (cmdHistory.length === 0) return;
            const nextPointer = historyPointer === -1 ? cmdHistory.length - 1 : Math.max(0, historyPointer - 1);
            setHistoryPointer(nextPointer);
            setInput(cmdHistory[nextPointer] || '');
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (cmdHistory.length === 0 || historyPointer === -1) return;
            const nextPointer = historyPointer + 1;
            if (nextPointer >= cmdHistory.length) {
                setHistoryPointer(-1);
                setInput('');
            } else {
                setHistoryPointer(nextPointer);
                setInput(cmdHistory[nextPointer] || '');
            }
        } else if (e.key === 'Tab') {
            e.preventDefault();
            const current = input.trim().toLowerCase();
            if (!current) return;

            if (current.startsWith('open ')) {
                const subQuery = current.slice(5).trim();
                const projectMatches = portfolioData.projects
                    .map(p => p.name)
                    .filter(name => name.toLowerCase().includes(subQuery) || name.toLowerCase().startsWith(subQuery));
                if (projectMatches.length > 0) {
                    setInput(`open ${projectMatches[0]}`);
                }
            } else {
                const matches = COMMANDS.filter(cmd => cmd.startsWith(current));
                if (matches.length > 0) {
                    setInput(matches[0]);
                }
            }
        }
    };

    const formatContent = (content) => {
        if (!content) return '';
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        const parts = content.split(urlRegex);

        return parts.map((part, index) => {
            if (part.match(urlRegex)) {
                return (
                    <a
                        key={index}
                        href={part}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="terminal-link"
                        onClick={(e) => e.stopPropagation()} // Prevent input focus when clicking link
                    >
                        {part}
                    </a>
                );
            }
            return part;
        });
    };

    return (
        <div className="terminal-body" onClick={() => inputRef.current?.focus()}>
            {history.map((item, index) => (
                <div key={index} className={item.type === 'input' ? 'command-line' : 'output-line'}>
                    {item.type === 'input' && <span className="prompt">C:\Users\Shaurya&gt;</span>}
                    <span className={item.type === 'input' ? 'highlight' : ''}>
                        {item.type === 'output' ? formatContent(item.content) : item.content}
                    </span>
                </div>
            ))}
            <div className="command-line">
                <span className="prompt">C:\Users\Shaurya&gt;</span>
                <input
                    ref={inputRef}
                    type="text"
                    className="cmd-input"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    autoFocus
                />
            </div>
            <div ref={bottomRef} />
        </div>
    );
};

export default Terminal;
