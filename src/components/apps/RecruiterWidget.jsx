import React, { useState } from 'react';
import { playClick } from '../../utils/soundEngine';

const RecruiterWidget = ({ onClose, onOpenResume, onOpenProject }) => {
    const [copiedEmail, setCopiedEmail] = useState(false);

    const handleCopyEmail = (e) => {
        e.stopPropagation();
        playClick();
        navigator.clipboard.writeText('emailofshauryak@gmail.com');
        setCopiedEmail(true);
        setTimeout(() => setCopiedEmail(false), 2500);
    };

    return (
        <div className="recruiter-active-card">
            <div className="recruiter-card-header">
                <span className="recruiter-card-title">📋 ACTIVE DESKTOP // RECRUITER FAST-PATH</span>
                <button
                    className="control-btn close-btn"
                    onClick={() => { playClick(); onClose(); }}
                    title="Hide Recruiter Card"
                >
                    ✕
                </button>
            </div>

            <div className="recruiter-card-body">
                <div className="recruiter-profile-summary">
                    <div className="candidate-name">Shaurya Kumar</div>
                    <div className="candidate-title">AI & Distributed Systems Engineer</div>
                    <div className="candidate-meta">🎓 VIT Chennai • CGPA 8.93 / 10.0 • New Delhi</div>
                </div>

                <div className="recruiter-actions-grid">
                    <button
                        className="retro-btn recruiter-btn"
                        onClick={() => { playClick(); onOpenResume(); }}
                    >
                        📄 View Full CV
                    </button>
                    <button
                        className="retro-btn recruiter-btn"
                        onClick={() => { playClick(); window.print(); }}
                    >
                        🖨️ Save PDF
                    </button>
                    <button
                        className="retro-btn recruiter-btn"
                        onClick={() => { playClick(); onOpenProject('RiskShield'); }}
                    >
                        🛡️ RiskShield
                    </button>
                    <button
                        className="retro-btn recruiter-btn"
                        onClick={() => { playClick(); onOpenProject('Kestrel'); }}
                    >
                        ⚡ Kestrel KV
                    </button>
                </div>

                <div className="recruiter-contact-row">
                    <button
                        className={`retro-btn-sm email-copy-btn ${copiedEmail ? 'copied' : ''}`}
                        onClick={handleCopyEmail}
                    >
                        {copiedEmail ? '✅ COPIED EMAIL!' : '✉️ Copy emailofshauryak@gmail.com'}
                    </button>
                    <a
                        href="https://www.linkedin.com/in/shaurya-kumar-22262b236/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="retro-btn-sm linkedin-link-btn"
                        onClick={() => playClick()}
                    >
                        💼 LinkedIn ↗
                    </a>
                </div>
            </div>
        </div>
    );
};

export default RecruiterWidget;