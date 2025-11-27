import React from 'react';
import { portfolioData } from '../data/portfolio';

const ResumeViewer = () => {
    const { resumeData } = portfolioData;

    if (!resumeData) return <div>Resume data not found.</div>;

    const { header, education, experience, projects, technicalSkills, awards } = resumeData;

    return (
        <div className="resume-viewer">
            <div className="resume-paper">
                {/* Header */}
                <header className="resume-header">
                    <h1>{header.name}</h1>
                    <div className="contact-info">
                        <span> {header.location}</span>
                        <span> <a href={`mailto:${header.email}`}>{header.email}</a></span>
                        <span> <a href={header.linkedin} target="_blank" rel="noopener noreferrer">LinkedIn</a></span>
                        <span> <a href={header.github} target="_blank" rel="noopener noreferrer">GitHub</a></span>
                    </div>
                </header>

                <hr className="resume-divider" />

                {/* Education */}
                <section className="resume-section">
                    <h2>Education</h2>
                    {education.map((edu, index) => (
                        <div key={index} className="resume-item">
                            <div className="item-header">
                                <h3>{edu.institution}</h3>
                                <span className="date">{edu.dates}</span>
                            </div>
                            <div className="item-subheader">{edu.degree}</div>
                            <ul className="item-details">
                                {edu.details.map((detail, i) => (
                                    <li key={i}>{detail}</li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </section>

                {/* Experience */}
                <section className="resume-section">
                    <h2>Experience</h2>
                    {experience.map((exp, index) => (
                        <div key={index} className="resume-item">
                            <div className="item-header">
                                <h3>{exp.role}</h3>
                                <span className="date">{exp.dates}</span>
                            </div>
                            <div className="item-subheader">{exp.company}</div>
                            <ul className="item-details">
                                {exp.points.map((point, i) => (
                                    <li key={i}>{point}</li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </section>

                {/* Projects */}
                <section className="resume-section">
                    <h2>Projects</h2>
                    {projects.map((proj, index) => (
                        <div key={index} className="resume-item">
                            <div className="item-header">
                                <h3><a href={proj.link} target="_blank" rel="noopener noreferrer">{proj.name}</a></h3>
                                <span className="type">{proj.type}</span>
                            </div>
                            <ul className="item-details">
                                {proj.points.map((point, i) => (
                                    <li key={i}>{point}</li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </section>

                {/* Technical Skills */}
                <section className="resume-section">
                    <h2>Technical Skills</h2>
                    <div className="skills-grid">
                        <div className="skill-category"><strong>Programming:</strong> {technicalSkills.programming}</div>
                        <div className="skill-category"><strong>Backend:</strong> {technicalSkills.backend}</div>
                        <div className="skill-category"><strong>Databases:</strong> {technicalSkills.databases}</div>
                        <div className="skill-category"><strong>Tools & Cloud:</strong> {technicalSkills.toolsCloud}</div>
                        <div className="skill-category"><strong>Frontend:</strong> {technicalSkills.frontend}</div>
                        <div className="skill-category"><strong>ML/AI:</strong> {technicalSkills.mlAi}</div>
                    </div>
                </section>

                {/* Awards */}
                <section className="resume-section">
                    <h2>Awards & Leadership</h2>
                    <ul className="item-details">
                        {awards.map((award, index) => (
                            <li key={index}>{award}</li>
                        ))}
                    </ul>
                </section>
            </div>
        </div>
    );
};

export default ResumeViewer;
