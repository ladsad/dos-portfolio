import React, { useState, useEffect } from 'react';
import { getMuteState, toggleMuteState, playClick } from '../../utils/soundEngine';

/**
 * Taskbar Component
 * Renders the Windows 98-style taskbar with Start button, window list, system tray audio, and clock.
 */
const Taskbar = ({ windows, activeWindowId, isStartMenuOpen, onToggleStartMenu, onWindowClick, onOpenSettings }) => {
    const [time, setTime] = useState(new Date());
    const [isMuted, setIsMuted] = useState(getMuteState);

    useEffect(() => {
        const timer = setInterval(() => setTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

    const handleToggleAudio = (e) => {
        e.stopPropagation();
        const nextMuted = toggleMuteState();
        setIsMuted(nextMuted);
        if (!nextMuted) playClick();
    };

    const handleStartClick = () => {
        playClick();
        onToggleStartMenu();
    };

    const handleItemClick = (win) => {
        playClick();
        onWindowClick(win);
    };

    return (
        <div className="taskbar">
            <div className={`start-button ${isStartMenuOpen ? 'active' : ''}`} onClick={handleStartClick}>
                <span className="start-icon">🪟</span> <strong>Start</strong>
            </div>
            <div className="window-list">
                {windows.map(win => (
                    <div
                        key={win.id}
                        className={`taskbar-item ${activeWindowId === win.id && !win.minimized ? 'active' : ''}`}
                        onClick={() => handleItemClick(win)}
                    >
                        {win.id === 'minesweeper' ? '💣 ' : win.id === 'settings' ? '⚙️ ' : win.type === 'resume' ? '📄 ' : win.type === 'terminal' ? '💻 ' : '📁 '}
                        {win.title}
                    </div>
                ))}
            </div>
            <div className="system-tray">
                <button
                    className="tray-icon-btn"
                    onClick={handleToggleAudio}
                    title={isMuted ? 'Sound Muted (Click to Unmute)' : 'Sound Enabled (Click to Mute)'}
                >
                    {isMuted ? '🔇' : '🔊'}
                </button>
                {onOpenSettings && (
                    <button
                        className="tray-icon-btn"
                        onClick={() => { playClick(); onOpenSettings(); }}
                        title="Display & Sound Properties"
                    >
                        ⚙️
                    </button>
                )}
                <div className="clock" title={time.toLocaleDateString(undefined, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}>
                    {time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
            </div>
        </div>
    );
};

export default Taskbar;

