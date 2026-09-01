import React, { useState } from 'react';
import { playClick, playOpen } from '../../utils/soundEngine';

const ICONS = [
    { id: 'my-computer', label: 'My Computer', icon: '💻', action: 'system' },
    { id: 'resume', label: 'Resume.txt', icon: '📄', action: 'resume' },
    { id: 'projects', label: 'Projects', icon: '📁', action: 'project' },
    { id: 'minesweeper', label: 'Minesweeper', icon: '💣', action: 'minesweeper' },
    { id: 'settings', label: 'Control Panel', icon: '⚙️', action: 'settings' },
    { id: 'recruiter', label: 'Recruiter Card', icon: '📋', action: 'recruiter' },
    { id: 'github', label: 'GitHub', icon: '🐙', action: 'link', url: 'https://github.com/ladsad' },
    { id: 'linkedin', label: 'LinkedIn', icon: '💼', action: 'link', url: 'https://www.linkedin.com/in/shaurya-kumar-22262b236/' }
];

const DesktopIcons = ({ onOpenProject, onOpenResume, onOpenApp, onToggleRecruiterWidget }) => {
    const [selectedId, setSelectedId] = useState(null);

    const handleIconClick = (e, icon) => {
        e.stopPropagation();
        playClick();
        setSelectedId(icon.id);
    };

    const handleIconDoubleClick = (e, icon) => {
        e.stopPropagation();
        playOpen();

        switch (icon.action) {
            case 'resume':
                onOpenResume();
                break;
            case 'project':
                onOpenProject('RiskShield');
                break;
            case 'minesweeper':
                onOpenApp('minesweeper');
                break;
            case 'settings':
                onOpenApp('settings');
                break;
            case 'system':
                onOpenApp('terminal');
                break;
            case 'recruiter':
                if (onToggleRecruiterWidget) onToggleRecruiterWidget();
                break;
            case 'link':
                window.open(icon.url, '_blank', 'noopener,noreferrer');
                break;
            default:
                break;
        }
    };

    return (
        <div className="desktop-icons-container" onClick={() => setSelectedId(null)}>
            {ICONS.map(icon => (
                <div
                    key={icon.id}
                    className={`desktop-icon-item ${selectedId === icon.id ? 'selected' : ''}`}
                    onClick={(e) => handleIconClick(e, icon)}
                    onDoubleClick={(e) => handleIconDoubleClick(e, icon)}
                    onTouchEnd={(e) => {
                        handleIconDoubleClick(e, icon);
                    }}
                >
                    <div className="desktop-icon-graphic">{icon.icon}</div>
                    <span className="desktop-icon-label">{icon.label}</span>
                </div>
            ))}
        </div>
    );
};

export default DesktopIcons;