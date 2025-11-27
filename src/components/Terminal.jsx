import React, { useState, useEffect, useRef } from 'react';
import { portfolioData } from '../data/portfolio';

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

            case 'projects':
                newHistory.push({
                    type: 'output',
                    content: `PROJECTS (Type "open <name>" to view details):
----------------------------------------`
                });
                portfolioData.projects.forEach(proj => {
                    newHistory.push({
                        type: 'output',
                        content: `* ${proj.name.padEnd(25)} [${proj.category}]`
                    });
                });
                newHistory.push({ type: 'output', content: '----------------------------------------' });
                break;

            case 'open': {
                const projectName = args.slice(1).join(' ');
                if (!projectName) {
                    newHistory.push({
                        type: 'output',
                        content: 'Usage: open <project name>'
                    });
                } else {
                    const success = onOpenProject(projectName);
                    if (success) {
                        newHistory.push({
                            type: 'output',
                            content: `Opening project "${projectName}"...`
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
