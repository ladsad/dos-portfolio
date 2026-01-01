import React from 'react';
import { X, Linkedin } from 'lucide-react';

/**
 * LinkedInWidget Component
 * A DOS-styled widget displaying LinkedIn profile information.
 * 
 * @param {Object} props
 * @param {Function} props.onClose - Function to close the widget
 */
const LinkedInWidget = ({ onClose }) => {
    const profileUrl = "https://www.linkedin.com/in/shaurya-kumar-22262b236/";

    return (
        <div className="linkedin-widget">
            <div className="linkedin-header">
                <div className="linkedin-title">
                    <Linkedin size={16} />
                    <span>LINKEDIN.EXE</span>
                </div>
                <button className="linkedin-close-btn" onClick={onClose}>
                    <X size={14} />
                </button>
            </div>
            <div className="linkedin-content">
                <img
                    src="https://media.licdn.com/dms/image/v2/D5603AQHcIlOPGBJt1Q/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1701709009418?e=2147483647&v=beta&t=hBUcWsVjlCRDJ4XRt6f3cCz3aLAfHQGPOhcK0vZG7Y8"
                    alt="Shaurya Kumar"
                    className="linkedin-avatar"
                />
                <div className="linkedin-info">
                    <div className="linkedin-name">Shaurya Kumar</div>
                    <div className="linkedin-headline">CS Student (AI/ML) @ VIT Chennai</div>
                    <div className="linkedin-location">New Delhi, India</div>
                </div>
            </div>
            <div className="linkedin-footer">
                <a
                    href={profileUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="linkedin-connect-btn"
                >
                    [CONNECT]
                </a>
            </div>
        </div>
    );
};

export default LinkedInWidget;
