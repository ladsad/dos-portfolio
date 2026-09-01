import React, { useState, useEffect, useRef } from 'react';
import { portfolioData } from '../../data/portfolio';

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

const Terminal = ({ onOpenProject }) => {
    const [input, setInput] = useState('');
    const [history, setHistory] = useState([
        { type: 'output', content: 'Welcome to Shaurya Kumar\'s Portfolio.' },
        { type: 'output', content: 'Type "help" to see available commands.' },
        { type: 'output', content: ' ' }
    ]);
    const bottomRef = useRef(null);
    const inputRef = useRef(null);

    useEffect(() => {
        if (bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [history]);

    const handleCommand = (cmd) => {
        const trimmedCmd = cmd.trim();
        const args = trimmedCmd.split(' ');
        const command = args[0].toLowerCase();
        const newHistory = [...history, { type: 'input', content: cmd }];

        switch (command) {
            case 'help':
                newHistory.push({
                    type: 'output',
                    content: `Available commands:
  about       - Display summary and contact info
  education   - Show education details
  experience  - Show work experience
  projects    - List highlighted projects
  open <name> - Open a project window (e.g., "open microsegnet")
  skills      - List technical skills
  awards      - Show awards and leadership
  clear       - Clear the terminal screen
  help        - Show this help message`
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
                    const success = onOpenProject(targetName);
                    if (success) {
                        newHistory.push({
                            type: 'output',
                            content: `Opening project "${targetName}"...`
                        });
                    } else {
                        newHistory.push({
                            type: 'output',
                            content: `Project "${projectName}" not found.`
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
                setHistory([]);
                return;

            case '':
                break;

            default:
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
