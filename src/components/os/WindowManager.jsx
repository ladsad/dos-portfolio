import React, { useState, useEffect } from 'react';
import Window from './Window';
import Terminal from '../apps/Terminal';
import ProjectDetail from '../apps/ProjectDetail';
import ResumeViewer from '../apps/ResumeViewer';
import Minesweeper from '../apps/Minesweeper';
import DisplaySettings from '../apps/DisplaySettings';
import StartMenu from './StartMenu';
import Taskbar from './Taskbar';
import AltTabSwitcher from './AltTabSwitcher';
import { portfolioData } from '../../data/portfolio';
import TipWidget from '../common/TipWidget';
import LinkedInWidget from '../apps/LinkedInWidget';
import DesktopIcons from './DesktopIcons';
import RecruiterWidget from '../apps/RecruiterWidget';
import { playOpen, playMinimize } from '../../utils/soundEngine';

const tips = [
    "Press Shift+Tab to switch between windows quickly!",
    "Type 'help' in the terminal to see all available commands.",
    "You can drag and resize these windows just like in 1998.",
    "This website was built with React and Vite.",
    "Shaurya is a Full Stack Developer based in New Delhi.",
    "Try the 'open <project>' command to jump straight to a project.",
    "Double-click the 'Resume.txt' icon to view my CV.",
    "The 'about' command reveals my contact information.",
    "Maximize windows by double-clicking the title bar (just kidding, but maybe soon!).",
    "Don't forget to shut down your computer properly via the Start Menu."
];

// Sound Effect
const playDing = () => {
    try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return;

        const ctx = new AudioContext();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.type = 'sine';
        osc.frequency.setValueAtTime(880, ctx.currentTime); // A5
        osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.1); // Drop to A4

        gain.gain.setValueAtTime(0.1, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);

        osc.start();
        osc.stop(ctx.currentTime + 0.5);
    } catch (e) {
        console.error("Audio play failed", e);
    }
};

/**
 * WindowManager Component
 * Manages the state and rendering of all windows, taskbar, start menu, and alt-tab switcher.
 */
const WindowManager = ({ currentTheme = 'retro', onSetTheme }) => {
    // State initialization
    const [windows, setWindows] = useState(() => {
        const termWidth = 600;
        const termHeight = 400;
        let x = 0;
        let y = 0;
        if (typeof window !== 'undefined') {
            x = Math.max(0, (window.innerWidth - termWidth) / 2);
            y = Math.max(0, (window.innerHeight - termHeight) / 2);
        }
        return [{
            id: 'terminal',
            type: 'terminal',
            title: 'COMMAND PROMPT',
            zIndex: 1,
            minimized: false,
            initialPosition: { x, y },
            initialSize: { width: 600, height: 400 }
        }];
    });

    const [activeWindowId, setActiveWindowId] = useState('terminal');
    const [nextZIndex, setNextZIndex] = useState(2);
    const [isAltTabActive, setIsAltTabActive] = useState(false);
    const [altTabSelectedIndex, setAltTabSelectedIndex] = useState(0);
    const [isStartMenuOpen, setIsStartMenuOpen] = useState(false);
    const [isMobile, setIsMobile] = useState(typeof window !== 'undefined' ? window.innerWidth <= 768 : false);

    // Tip Widget State
    const [showTip, setShowTip] = useState(false);
    const [currentTip, setCurrentTip] = useState('');

    // LinkedIn Widget State
    const [showLinkedIn, setShowLinkedIn] = useState(true);

    // Recruiter Active Desktop Card State
    const [showRecruiterCard, setShowRecruiterCard] = useState(!isMobile);

    // Periodic Tip Interval
    useEffect(() => {
        const interval = setInterval(() => {
            const randomTip = tips[Math.floor(Math.random() * tips.length)];
            setCurrentTip(randomTip);
            setShowTip(true);
            playDing();
        }, 50000);

        const initialTimeout = setTimeout(() => {
            const randomTip = tips[Math.floor(Math.random() * tips.length)];
            setCurrentTip(randomTip);
            setShowTip(true);
            playDing();
        }, 10000);

        return () => {
            clearInterval(interval);
            clearTimeout(initialTimeout);
        };
    }, []);

    // Handle window resize for mobile detection
    useEffect(() => {
        const handleResize = () => {
            setIsMobile(window.innerWidth <= 768);
        };
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const bringToFront = React.useCallback((id) => {
        setActiveWindowId(id);
        setWindows(prev => prev.map(w =>
            w.id === id ? { ...w, zIndex: nextZIndex } : w
        ));
        setNextZIndex(prev => prev + 1);
    }, [nextZIndex]);

    // Open a project window
    const openProjectWindow = React.useCallback((projectName) => {
        const project = portfolioData.projects.find(p => p.name.toLowerCase() === projectName.toLowerCase() || p.name.toLowerCase().includes(projectName.toLowerCase()));

        if (!project) return false;

        playOpen();

        const existingWindow = windows.find(w => w.id === `project-${project.name}`);
        if (existingWindow) {
            bringToFront(existingWindow.id);
            if (existingWindow.minimized) {
                setWindows(prev => prev.map(w => w.id === existingWindow.id ? { ...w, minimized: false } : w));
            }
            return true;
        }

        const newWindow = {
            id: `project-${project.name}`,
            type: 'project',
            title: project.name,
            content: project,
            zIndex: nextZIndex,
            minimized: false,
            initialPosition: isMobile ? { x: 0, y: 0 } : { x: 50 + (windows.length * 20), y: 50 + (windows.length * 20) },
            initialSize: { width: '60vw', height: '60vh' }
        };

        setWindows(prev => [...prev, newWindow]);
        setNextZIndex(prev => prev + 1);
        setActiveWindowId(newWindow.id);
        return true;
    }, [windows, nextZIndex, isMobile, bringToFront]);

    // Open the resume window
    const openResumeWindow = React.useCallback(() => {
        playOpen();
        const existingWindow = windows.find(w => w.id === 'resume');
        if (existingWindow) {
            bringToFront('resume');
            if (existingWindow.minimized) {
                setWindows(prev => prev.map(w => w.id === 'resume' ? { ...w, minimized: false } : w));
            }
        } else {
            setWindows(prev => [...prev, {
                id: 'resume',
                type: 'resume',
                title: 'Resume.txt',
                zIndex: nextZIndex,
                minimized: false,
                initialPosition: isMobile ? { x: 0, y: 0 } : { x: 40, y: 40 },
                initialSize: { width: '50vw', height: '80vh' }
            }]);
            setNextZIndex(prev => prev + 1);
            setActiveWindowId('resume');
        }
    }, [windows, nextZIndex, isMobile, bringToFront]);

    // Open generic utility apps (Minesweeper, Display Settings)
    const openAppWindow = React.useCallback((appType) => {
        playOpen();
        if (appType === 'minesweeper') {
            const existingWindow = windows.find(w => w.id === 'minesweeper');
            if (existingWindow) {
                bringToFront('minesweeper');
                if (existingWindow.minimized) {
                    setWindows(prev => prev.map(w => w.id === 'minesweeper' ? { ...w, minimized: false } : w));
                }
            } else {
                setWindows(prev => [...prev, {
                    id: 'minesweeper',
                    type: 'minesweeper',
                    title: 'Minesweeper',
                    zIndex: nextZIndex,
                    minimized: false,
                    initialPosition: isMobile ? { x: 10, y: 10 } : { x: 120, y: 80 },
                    initialSize: { width: 340, height: 440 }
                }]);
                setNextZIndex(prev => prev + 1);
                setActiveWindowId('minesweeper');
            }
        } else if (appType === 'settings') {
            const existingWindow = windows.find(w => w.id === 'settings');
            if (existingWindow) {
                bringToFront('settings');
                if (existingWindow.minimized) {
                    setWindows(prev => prev.map(w => w.id === 'settings' ? { ...w, minimized: false } : w));
                }
            } else {
                setWindows(prev => [...prev, {
                    id: 'settings',
                    type: 'settings',
                    title: 'Display & Sound Properties',
                    zIndex: nextZIndex,
                    minimized: false,
                    initialPosition: isMobile ? { x: 10, y: 10 } : { x: 180, y: 100 },
                    initialSize: { width: 440, height: 460 }
                }]);
                setNextZIndex(prev => prev + 1);
                setActiveWindowId('settings');
            }
        }
    }, [windows, nextZIndex, isMobile, bringToFront]);

    // Close a window
    const closeWindow = React.useCallback((id) => {
        if (id === 'terminal') return;
        setWindows(prev => prev.filter(w => w.id !== id));
        if (activeWindowId === id) {
            setActiveWindowId(null);
        }
    }, [activeWindowId]);

    // Minimize a window
    const minimizeWindow = React.useCallback((id) => {
        playMinimize();
        setWindows(prev => prev.map(w => w.id === id ? { ...w, minimized: true } : w));
        if (activeWindowId === id) {
            setActiveWindowId(null);
        }
    }, [activeWindowId]);

    // Handle system actions from Start Menu
    const handleSystemAction = React.useCallback((action) => {
        setIsStartMenuOpen(false);
        if (action === 'terminal') {
            const terminalOpen = windows.some(w => w.id === 'terminal');
            if (!terminalOpen) {
                setWindows(prev => [...prev, {
                    id: 'terminal',
                    type: 'terminal',
                    title: 'COMMAND PROMPT',
                    zIndex: nextZIndex,
                    minimized: false,
                    initialPosition: isMobile ? { x: 0, y: 0 } : { x: 50, y: 50 },
                    initialSize: { width: 600, height: 400 }
                }]);
                setNextZIndex(prev => prev + 1);
                setActiveWindowId('terminal');
            } else {
                const terminal = windows.find(w => w.id === 'terminal');
                if (terminal.minimized) {
                    setWindows(prev => prev.map(w => w.id === 'terminal' ? { ...w, minimized: false } : w));
                }
                bringToFront('terminal');
            }
        } else if (action === 'reboot') {
            window.location.reload();
        } else if (action === 'shutdown') {
            document.body.innerHTML = '<div style="background:black;color:orange;height:100vh;display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:2rem;">IT IS NOW SAFE TO TURN OFF YOUR COMPUTER.</div>';
        } else if (action === 'linkedin') {
            setShowLinkedIn(true);
        }
    }, [windows, nextZIndex, isMobile, bringToFront]);

    // Handle Alt+Tab key events
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.shiftKey && e.key === 'Tab') {
                e.preventDefault();
                setIsAltTabActive(true);
                setAltTabSelectedIndex(prev => {
                    const nextIndex = (prev + 1) % windows.length;
                    return nextIndex;
                });
            }
        };

        const handleKeyUp = (e) => {
            if (e.key === 'Shift') {
                setIsAltTabActive(false);
                if (windows.length > 0) {
                    const selectedWindow = windows[altTabSelectedIndex];
                    if (selectedWindow) {
                        if (selectedWindow.minimized) {
                            setWindows(prev => prev.map(w => w.id === selectedWindow.id ? { ...w, minimized: false } : w));
                        }
                        bringToFront(selectedWindow.id);
                    }
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        window.addEventListener('keyup', handleKeyUp);

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
        };
    }, [windows, altTabSelectedIndex, bringToFront]);

    // Handle Taskbar window click
    const handleTaskbarWindowClick = React.useCallback((win) => {
        if (win.minimized) {
            setWindows(prev => prev.map(w => w.id === win.id ? { ...w, minimized: false } : w));
        }
        bringToFront(win.id);
    }, [bringToFront]);

    return (
        <>
            <DesktopIcons
                onOpenProject={openProjectWindow}
                onOpenResume={openResumeWindow}
                onOpenApp={openAppWindow}
                onToggleRecruiterWidget={() => setShowRecruiterCard(prev => !prev)}
            />

            {showRecruiterCard && (
                <RecruiterWidget
                    onClose={() => setShowRecruiterCard(false)}
                    onOpenResume={openResumeWindow}
                    onOpenProject={openProjectWindow}
                />
            )}

            {windows.map(win => (
                !win.minimized && (
                    <Window
                        key={win.id}
                        id={win.id}
                        title={win.title}
                        zIndex={win.zIndex}
                        isActive={activeWindowId === win.id}
                        onClose={closeWindow}
                        onMinimize={minimizeWindow}
                        onFocus={() => bringToFront(win.id)}
                        initialPosition={win.initialPosition}
                        initialSize={win.initialSize}
                        className={win.type === 'terminal' ? 'terminal-window' : ''}
                        isMobile={isMobile}
                    >
                        {win.type === 'terminal' ? (
                            <Terminal
                                onOpenProject={openProjectWindow}
                                onOpenApp={openAppWindow}
                                onSetTheme={onSetTheme}
                                onClear={() => { }}
                            />
                        ) : win.type === 'resume' ? (
                            <ResumeViewer />
                        ) : win.type === 'minesweeper' ? (
                            <Minesweeper />
                        ) : win.type === 'settings' ? (
                            <DisplaySettings
                                currentTheme={currentTheme}
                                onSave={(cfg) => {
                                    if (onSetTheme) onSetTheme(cfg.theme);
                                    closeWindow('settings');
                                }}
                            />
                        ) : (
                            <ProjectDetail project={win.content} />
                        )}
                    </Window >
                )
            ))}

            {isAltTabActive && (
                <AltTabSwitcher
                    windows={windows}
                    selectedIndex={altTabSelectedIndex}
                />
            )}

            <StartMenu
                isOpen={isStartMenuOpen}
                onClose={() => setIsStartMenuOpen(false)}
                onOpenProject={openProjectWindow}
                onOpenResume={openResumeWindow}
                onOpenApp={openAppWindow}
                onSystemAction={handleSystemAction}
                portfolioData={portfolioData}
            />

            <Taskbar
                windows={windows}
                activeWindowId={activeWindowId}
                isStartMenuOpen={isStartMenuOpen}
                onToggleStartMenu={() => setIsStartMenuOpen(!isStartMenuOpen)}
                onWindowClick={handleTaskbarWindowClick}
                onOpenSettings={() => openAppWindow('settings')}
            />

            {showTip && (
                <TipWidget
                    message={currentTip}
                    onClose={() => setShowTip(false)}
                />
            )}

            {showLinkedIn && (
                <LinkedInWidget
                    onClose={() => setShowLinkedIn(false)}
                />
            )}
        </>
    );
};

export default WindowManager;
