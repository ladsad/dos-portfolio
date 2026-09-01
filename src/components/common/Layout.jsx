import React from 'react';

const Layout = ({ children, theme = 'retro' }) => {
    const showScanlines = typeof window !== 'undefined' ? localStorage.getItem('crtScanlines') !== 'false' : true;
    const showFlicker = typeof window !== 'undefined' ? localStorage.getItem('crtFlicker') !== 'false' : true;

    return (
        <div className="dos-container" data-theme={theme}>
            {showScanlines && <div className="scanlines"></div>}
            {showFlicker && <div className="crt-flicker"></div>}
            {theme === 'retro' && (
                <div className="dos-header">
                    DOS-PORTFOLIO [Version 2.4.0] - (C) Copyright 2026 Shaurya Kumar
                </div>
            )}
            {children}
        </div>
    );
};

export default Layout;
