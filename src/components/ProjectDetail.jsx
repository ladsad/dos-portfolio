import React from 'react';
import ReactMarkdown from 'react-markdown';

const ProjectDetail = ({ project }) => {
    // Simple ASCII art generator for title
    const getAsciiTitle = (text) => {
        // This is a placeholder. In a real app, we might use a library or a pre-generated map.
        // For now, let's just make it look "blocky" with spacing.
        return text.split('').join(' ').toUpperCase();
    };

    return (
        <div className="project-dashboard">
            <div className="dashboard-header">
                <pre className="ascii-title">
                    {`
  _____  _____  _____  _____  _____ 
 |  _  || __  ||     ||     ||   __|
 |   __||    -||  |  ||  |  ||   __|
 |__|   |__|__||_____||_____||_____|
`}
                    {/* We'll just use the project name for now, but the above is a static "PROJECT" banner */}
                    {getAsciiTitle(project.name)}
                </pre>
            </div>

            <div className="dashboard-grid">
                <div className="dashboard-sidebar">
                    <div className="panel system-info">
                        <div className="panel-header">SYSTEM INFO</div>
                        <div className="panel-content">
                            <div className="info-row">
                                <span className="label">STATUS:</span>
                                <span className="value active">ONLINE</span>
                            </div>
                            <div className="info-row">
                                <span className="label">CATEGORY:</span>
                                <span className="value">{project.category}</span>
                            </div>
                            <div className="info-row">
                                <span className="label">ACCESS:</span>
                                <a href={project.link} target="_blank" rel="noopener noreferrer" className="value link">
                                    [LINK]
                                </a>
                            </div>
                        </div>
                    </div>

                    <div className="panel modules-loaded">
                        <div className="panel-header">MODULES LOADED</div>
                        <div className="panel-content">
                            {project.highlights && project.highlights.map((highlight, idx) => (
                                <div key={idx} className="module-item">
                                    <span className="module-status">[OK]</span>
                                    <span className="module-name">{highlight.substring(0, 20)}...</span>
                                </div>
                            ))}
                            {!project.highlights && <div className="module-item">No modules detected.</div>}
                        </div>
                    </div>
                </div>

                <div className="dashboard-main">
                    <div className="panel content-viewer">
                        <div className="panel-header">CONTENT VIEWER v1.0</div>
                        <div className="panel-content markdown-body">
                            <ReactMarkdown
                                components={{
                                    a: ({ node, ...props }) => (
                                        <a {...props} target="_blank" rel="noopener noreferrer" />
                                    )
                                }}
                            >
                                {project.content || "No detailed content available."}
                            </ReactMarkdown>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ProjectDetail;
