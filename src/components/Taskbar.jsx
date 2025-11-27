import React, { useState, useEffect } from 'react';

/**
 * Taskbar Component
 * Renders the Windows 98-style taskbar with Start button, window list, and clock.
 * 
 * @param {Object} props
 * @param {Array} props.windows - List of open windows
 * @param {string} props.activeWindowId - ID of the currently active window
 * @param {boolean} props.isStartMenuOpen - Whether the start menu is open
 * @param {Function} props.onToggleStartMenu - Function to toggle start menu
 * @param {Function} props.onWindowClick - Function to handle window click in taskbar
 */
const Taskbar = ({ windows, activeWindowId, isStartMenuOpen, onToggleStartMenu, onWindowClick }) => {
    const [time, setTime] = useState(new Date());

    useEffect(() => {
        const timer = setInterval(() => setTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

    return (
        <div className="taskbar">
            <div className={`start-button ${isStartMenuOpen ? 'active' : ''}`} onClick={onToggleStartMenu}>
                <span className="start-icon">[S]</span> START
            </div>
            <div className="window-list">
                {windows.map(win => (
                    <div
                        key={win.id}
                        className={`taskbar-item ${activeWindowId === win.id && !win.minimized ? 'active' : ''}`}
                        onClick={() => onWindowClick(win)}
                    >
                        {win.title}
                    </div>
                ))}
            </div>
            <div className="clock">
                {time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
        </div>
    );
};

export default Taskbar;
