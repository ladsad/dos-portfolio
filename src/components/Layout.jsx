import React from 'react';

const Layout = ({ children }) => {
    return (
        <div className="dos-container">
            <div className="scanlines"></div>
            <div className="crt-flicker"></div>
            <div className="dos-header">
                DOS-PORTFOLIO [Version 1.0.0] - (C) Copyright 2025 Shaurya Kumar
            </div>
            {children}
        </div>
    );
};

export default Layout;
