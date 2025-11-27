import React from 'react';

/**
 * StartMenu Component
 * Renders the Windows 98-style Start Menu.
 * 
 * @param {Object} props
 * @param {boolean} props.isOpen - Whether the start menu is open
 * @param {Function} props.onClose - Function to close the start menu
 * @param {Function} props.onOpenProject - Function to open a project window
 * @param {Function} props.onOpenResume - Function to open the resume window
 * @param {Function} props.onSystemAction - Function to handle system actions (terminal, reboot, shutdown)
 * @param {Object} props.portfolioData - The portfolio data containing projects
 */
const StartMenu = ({ isOpen, onClose, onOpenProject, onOpenResume, onSystemAction, portfolioData }) => {
    if (!isOpen) return null;

    return (
        <div className="start-menu">
            <div className="start-menu-sidebar">
                <div className="start-menu-text">SALAD OS 98</div>
            </div>
            <div className="start-menu-content">
                <div className="start-menu-item" onClick={onClose}>
                    <span className="icon">[P]</span> Programs
                    <div className="submenu">
                        {portfolioData.projects.map(p => (
                            <div key={p.name} className="submenu-item" onClick={(e) => {
                                e.stopPropagation();
                                onOpenProject(p.name);
                                onClose();
                            }}>
                                {p.name}
                            </div>
                        ))}
                    </div>
                </div>
                <div className="start-menu-item" onClick={() => {
                    onOpenResume();
                    onClose();
                }}>
                    <span className="icon">[D]</span> Documents
                    <div className="submenu">
                        <div className="submenu-item" onClick={(e) => {
                            e.stopPropagation();
                            onOpenResume();
                            onClose();
                        }}>Resume.txt</div>
                    </div>
                </div>
                <div className="start-menu-item">
                    <span className="icon">[S]</span> System
                    <div className="submenu">
                        <div className="submenu-item" onClick={() => onSystemAction('terminal')}>Command Prompt</div>
                        <div className="submenu-item" onClick={() => onSystemAction('reboot')}>Reboot System</div>
                    </div>
                </div>
                <div className="start-menu-divider"></div>
                <div className="start-menu-item" onClick={() => onSystemAction('shutdown')}>
                    <span className="icon">[X]</span> Shut Down...
                </div>
            </div>
        </div>
    );
};

export default StartMenu;
