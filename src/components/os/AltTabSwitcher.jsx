import React from 'react';

/**
 * AltTabSwitcher Component
 * Renders the Alt+Tab window switcher overlay.
 * 
 * @param {Object} props
 * @param {Array} props.windows - List of open windows
 * @param {number} props.selectedIndex - Index of the currently selected window
 */
const AltTabSwitcher = ({ windows, selectedIndex }) => {
    if (windows.length === 0) return null;

    return (
        <div className="alt-tab-overlay">
            <div className="alt-tab-box">
                <div className="alt-tab-title">Task Switcher</div>
                <div className="alt-tab-list">
                    {windows.map((win, index) => (
                        <div
                            key={win.id}
                            className={`alt-tab-item ${index === selectedIndex ? 'selected' : ''}`}
                        >
                            <div className="alt-tab-icon">
                                {win.type === 'terminal' ? '>_' : win.type === 'resume' ? '[DOC]' : '[PRJ]'}
                            </div>
                            <div className="alt-tab-name">{win.title}</div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default AltTabSwitcher;
