import React, { useState } from 'react';
import { getMuteState, setMuteState, playClick } from '../../utils/soundEngine';

const DisplaySettings = ({ onSave, currentTheme = 'retro' }) => {
    const [theme, setTheme] = useState(() => localStorage.getItem('displayMode') || currentTheme);
    const [scanlines, setScanlines] = useState(() => localStorage.getItem('crtScanlines') !== 'false');
    const [flicker, setFlicker] = useState(() => localStorage.getItem('crtFlicker') !== 'false');
    const [isMuted, setIsMuted] = useState(getMuteState);

    const handleThemeChange = (newTheme) => {
        playClick();
        setTheme(newTheme);
    };

    const handleSave = () => {
        playClick();
        localStorage.setItem('displayMode', theme);
        localStorage.setItem('crtScanlines', String(scanlines));
        localStorage.setItem('crtFlicker', String(flicker));
        setMuteState(isMuted);

        if (onSave) {
            onSave({ theme, scanlines, flicker, isMuted });
        }
    };

    return (
        <div className="display-settings-dialog">
            <div className="display-preview-box" data-preview-theme={theme}>
                <div className="preview-screen">
                    <div className="preview-window">
                        <div className="preview-title">SALAD 98</div>
                        <div className="preview-body">Previewing {theme.toUpperCase()} style</div>
                    </div>
                </div>
            </div>

            <fieldset className="retro-fieldset">
                <legend>Color Palette & Theme</legend>
                <div className="radio-group">
                    {[
                        { id: 'retro', label: 'Classic Windows 98 (Teal)' },
                        { id: 'amber', label: 'Amber Phosphor CRT (VT220)' },
                        { id: 'matrix', label: 'Matrix Green (Terminal)' },
                        { id: 'cyberpunk', label: 'Cyberpunk Neon (Synth)' },
                        { id: 'modern', label: 'Modern Slate (Flat)' }
                    ].map(t => (
                        <label key={t.id} className="retro-radio-label">
                            <input
                                type="radio"
                                name="theme"
                                checked={theme === t.id}
                                onChange={() => handleThemeChange(t.id)}
                            />
                            {t.label}
                        </label>
                    ))}
                </div>
            </fieldset>

            <fieldset className="retro-fieldset">
                <legend>CRT Monitor Effects</legend>
                <label className="retro-checkbox-label">
                    <input
                        type="checkbox"
                        checked={scanlines}
                        onChange={(e) => { playClick(); setScanlines(e.target.checked); }}
                    />
                    Enable CRT Scanline Emulation
                </label>
                <label className="retro-checkbox-label">
                    <input
                        type="checkbox"
                        checked={flicker}
                        onChange={(e) => { playClick(); setFlicker(e.target.checked); }}
                    />
                    Enable Screen Phosphor Flicker
                </label>
            </fieldset>

            <fieldset className="retro-fieldset">
                <legend>Audio & Sound Effects</legend>
                <label className="retro-checkbox-label">
                    <input
                        type="checkbox"
                        checked={!isMuted}
                        onChange={(e) => { playClick(); setIsMuted(!e.target.checked); }}
                    />
                    Enable 8-bit System Sound Effects
                </label>
            </fieldset>

            <div className="display-settings-actions">
                <button className="retro-btn primary" onClick={handleSave}>
                    OK (Apply)
                </button>
            </div>
        </div>
    );
};

export default DisplaySettings;
