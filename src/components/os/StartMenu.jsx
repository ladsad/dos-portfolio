import React from 'react';
import { playClick } from '../../utils/soundEngine';

/**
 * StartMenu Component
 * Renders the Windows 98-style Start Menu.
 */
const StartMenu = ({ isOpen, onClose, onOpenProject, onOpenResume, onSystemAction, onOpenApp, portfolioData }) => {
    if (!isOpen) return null;

    const handleAction = (cb) => {
        playClick();
        cb();
        onClose();
    };

    return (
        <div className="start-menu">
            <div className="start-menu-sidebar">
                <div className="start-menu-text">SALAD OS 98</div>
            </div>
            <div className="start-menu-content">
                <div className="start-menu-item">
                    <span className="icon">📁</span> Programs
                    <div className="submenu">
                        {portfolioData.projects.map(p => (
                            <div key={p.name} className="submenu-item" onClick={(e) => {
                                e.stopPropagation();
                                handleAction(() => onOpenProject(p.name));
                            }}>
                                📁 {p.name}
                            </div>
                        ))}
                    </div>
                </div>
                <div className="start-menu-item">
                    <span className="icon">🎮</span> Games & Apps
                    <div className="submenu">
                        <div className="submenu-item" onClick={(e) => {
                            e.stopPropagation();
                            handleAction(() => onOpenApp('minesweeper'));
                        }}>
                            💣 Minesweeper
                        </div>
                    </div>
                </div>
                <div className="start-menu-item">
                    <span className="icon">📄</span> Documents
                    <div className="submenu">
                        <div className="submenu-item" onClick={(e) => {
                            e.stopPropagation();
                            handleAction(() => onOpenResume());
                        }}>
                            📄 Resume.txt
                        </div>
                    </div>
                </div>
                <div className="start-menu-item">
                    <span className="icon">⚙️</span> Settings
                    <div className="submenu">
                        <div className="submenu-item" onClick={(e) => {
                            e.stopPropagation();
                            handleAction(() => onOpenApp('settings'));
                        }}>
                            🖥️ Display & Sound Properties
                        </div>
                    </div>
                </div>
                <div className="start-menu-item">
                    <span className="icon">💻</span> System
                    <div className="submenu">
                        <div className="submenu-item" onClick={() => handleAction(() => onSystemAction('terminal'))}>💻 Command Prompt</div>
                        <div className="submenu-item" onClick={() => handleAction(() => onSystemAction('linkedin'))}>🔗 LinkedIn Profile</div>
                        <div className="submenu-item" onClick={() => handleAction(() => onSystemAction('reboot'))}>🔄 Reboot System</div>
                    </div>
                </div>
                <div className="start-menu-divider"></div>
                <div className="start-menu-item" onClick={() => handleAction(() => onSystemAction('shutdown'))}>
                    <span className="icon">⏻</span> Shut Down...
                </div>
            </div>
        </div>
    );
};

export default StartMenu;

