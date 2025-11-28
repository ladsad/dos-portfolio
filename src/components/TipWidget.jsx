import React, { useEffect } from 'react';
import { X, Lightbulb } from 'lucide-react';

const TipWidget = ({ message, onClose }) => {

    return (
        <div className="tip-widget">
            <div className="tip-header">
                <div className="tip-title">
                    <Lightbulb size={16} />
                    <span>DID YOU KNOW?</span>
                </div>
                <button className="tip-close-btn" onClick={onClose}>
                    <X size={14} />
                </button>
            </div>
            <div className="tip-content">
                <p>{message}</p>
            </div>
            <div className="tip-footer">
                <button className="tip-ok-btn" onClick={onClose}>OK</button>
            </div>
        </div>
    );
};

export default TipWidget;
