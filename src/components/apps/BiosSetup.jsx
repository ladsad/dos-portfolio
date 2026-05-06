import React, { useState, useEffect } from 'react';

const BiosSetup = ({ onExit }) => {
    const [activeTab, setActiveTab] = useState(0);
    const [selectedItem, setSelectedItem] = useState(0);
    const [displayMode, setDisplayMode] = useState(
        () => localStorage.getItem('displayMode') || 'retro'
    );

    const tabs = ['Main', 'Advanced', 'Security', 'Boot', 'Exit'];

    const menuItems = [
        [ // Main
            { label: 'System Time', value: '10:08:20' },
            { label: 'System Date', value: '25/11/2025' },
            { label: 'CPU Type', value: 'SaladCore i9' },
            { label: 'CPU Speed', value: '999 GHz' },
            { label: 'Total Memory', value: '640 KB' },
        ],
        [ // Advanced
            { label: 'Hyper-Threading', value: '[Enabled]' },
            { label: 'Virtualization', value: '[Enabled]' },
            { label: 'USB Legacy Support', value: '[Auto]' },
            {
                label: 'Display Mode',
                value: `[${displayMode.toUpperCase()}]`,
                action: 'toggleDisplay',
                hint: 'Toggle between RETRO and MODERN display. Takes effect on next boot.'
            },
        ],
        [ // Security
            { label: 'Supervisor Password', value: 'Not Installed' },
            { label: 'User Password', value: 'Not Installed' },
            { label: 'Secure Boot', value: '[Disabled]' },
        ],
        [ // Boot
            { label: 'Boot Option #1', value: '[Salad OS]' },
            { label: 'Boot Option #2', value: '[Floppy]' },
            { label: 'Fast Boot', value: '[Enabled]' },
        ],
        [ // Exit
            { label: 'Save Changes and Exit', action: 'save' },
            { label: 'Discard Changes and Exit', action: 'discard' },
        ]
    ];

    const getHint = () => {
        const item = menuItems[activeTab][selectedItem];
        if (item?.hint) return item.hint;
        if (item?.action === 'save') return 'Save all changes and exit BIOS setup.';
        if (item?.action === 'discard') return 'Discard all changes and exit BIOS setup.';
        return 'Use arrow keys to navigate. Press Enter to select.';
    };

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'ArrowRight') {
                setActiveTab(prev => (prev + 1) % tabs.length);
                setSelectedItem(0);
            } else if (e.key === 'ArrowLeft') {
                setActiveTab(prev => (prev - 1 + tabs.length) % tabs.length);
                setSelectedItem(0);
            } else if (e.key === 'ArrowDown') {
                setSelectedItem(prev => (prev + 1) % menuItems[activeTab].length);
            } else if (e.key === 'ArrowUp') {
                setSelectedItem(prev => (prev - 1 + menuItems[activeTab].length) % menuItems[activeTab].length);
            } else if (e.key === 'Enter') {
                const item = menuItems[activeTab][selectedItem];
                if (item?.action === 'toggleDisplay') {
                    setDisplayMode(prev => prev === 'retro' ? 'modern' : 'retro');
                } else if (item?.action === 'save') {
                    localStorage.setItem('displayMode', displayMode);
                    onExit();
                } else if (item?.action === 'discard') {
                    onExit();
                }
            } else if (e.key === 'Escape') {
                onExit();
            } else if (e.key === 'F10') {
                localStorage.setItem('displayMode', displayMode);
                onExit();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [activeTab, selectedItem, displayMode]);

    return (
        <div style={{
            backgroundColor: '#0000AA',
            color: '#FFFFFF',
            height: '100vh',
            width: '100vw',
            fontFamily: "'Courier New', monospace",
            display: 'flex',
            flexDirection: 'column',
            padding: '20px',
            boxSizing: 'border-box',
            position: 'fixed',
            top: 0,
            left: 0,
            zIndex: 20000
        }}>
            <div style={{ textAlign: 'center', marginBottom: '20px', borderBottom: '2px solid white' }}>
                <h2 style={{ margin: 0 }}>PhoenixBIOS Setup Utility</h2>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-around', marginBottom: '20px', backgroundColor: '#00AAAA', color: 'black', padding: '5px' }}>
                {tabs.map((tab, index) => (
                    <div key={tab} style={{ fontWeight: activeTab === index ? 'bold' : 'normal', textDecoration: activeTab === index ? 'underline' : 'none', cursor: 'pointer' }}
                        onClick={() => { setActiveTab(index); setSelectedItem(0); }}>
                        {tab}
                    </div>
                ))}
            </div>

            <div style={{ display: 'flex', flex: 1 }}>
                <div style={{ flex: 2, border: '2px solid white', padding: '20px', marginRight: '20px' }}>
                    {menuItems[activeTab].map((item, index) => (
                        <div key={index}
                            style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                color: selectedItem === index ? 'white' : '#AAAAAA',
                                backgroundColor: selectedItem === index ? '#000000' : 'transparent',
                                padding: '2px 5px',
                                cursor: item.action ? 'pointer' : 'default',
                            }}
                            onClick={() => {
                                setSelectedItem(index);
                                if (item?.action === 'toggleDisplay') {
                                    setDisplayMode(prev => prev === 'retro' ? 'modern' : 'retro');
                                } else if (item?.action === 'save') {
                                    localStorage.setItem('displayMode', displayMode);
                                    onExit();
                                } else if (item?.action === 'discard') {
                                    onExit();
                                }
                            }}
                        >
                            <span>{item.label}</span>
                            <span style={{ color: item.action === 'toggleDisplay' ? '#00FFFF' : 'inherit' }}>
                                {item.action === 'toggleDisplay'
                                    ? `[${displayMode.toUpperCase()}]`
                                    : item.value}
                            </span>
                        </div>
                    ))}
                </div>
                <div style={{ flex: 1, border: '2px solid white', padding: '20px', color: '#AAAAAA' }}>
                    <p>Item Specific Help</p>
                    <p>-----------------</p>
                    <p style={{ color: '#FFFFFF', fontSize: '0.9rem' }}>{getHint()}</p>
                    <p>-----------------</p>
                    <p>&lt;Enter&gt; selects or toggles.</p>
                    <p>&lt;Up&gt; and &lt;Down&gt; arrows select item.</p>
                    <p>&lt;Left&gt; and &lt;Right&gt; arrows select menu.</p>
                    <p>&lt;Esc&gt; exits without saving.</p>
                    <p>&lt;F10&gt; saves and exits.</p>
                </div>
            </div>

            <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                <span>F1: Help</span>
                <span>Esc: Exit (no save)</span>
                <span>↑↓: Select Item</span>
                <span>←→: Select Menu</span>
                <span>Enter: Select/Toggle</span>
                <span>F10: Save and Exit</span>
            </div>
        </div>
    );
};

export default BiosSetup;
